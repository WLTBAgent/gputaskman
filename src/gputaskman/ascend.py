"""Huawei Ascend NPU backend (Linux): adapters and telemetry from npu-smi, per-process memory
from `npu-smi info -t proc-mem`.

The 310P-family NPUs don't use DRM: the host driver (devdrv) exposes /dev/davinci* character
devices, and telemetry lives behind the npu-smi management tool (PCI vendor 0x19E5). npu-smi
calls cost ~0.5s each, so live utilization comes from a persistent `npu-smi info watch` stream
(one line per chip per second) that a reader thread keeps running; absolute memory and
per-process memory are refreshed on a slow cadence. A dual-chip card (e.g. Atlas 300I Duo)
appears as two adapters, one per chip, each with its own DDR bank.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import dataclass

from .dxgi import Adapter
from .pdh import AdapterGpu, ProcGpu
from .telemetry import DeviceTelemetry

VENDOR_HUAWEI = 0x19E5
_MB = 1024 * 1024

_NPUSMI = None  # resolved lazily; see _npu_path


def _npu_path() -> str | None:
    """Path to npu-smi, or None when the tool (and likely the driver) is absent."""
    global _NPUSMI
    if _NPUSMI is None:
        _NPUSMI = shutil.which("npu-smi") or shutil.which("/usr/local/bin/npu-smi")
    return _NPUSMI


def _run(args: list[str], timeout: float = 10.0) -> str:
    """Return npu-smi output; callers must distinguish failure from an empty result."""
    path = _npu_path()
    if path is None:
        raise OSError("npu-smi not found")
    try:
        r = subprocess.run([path, *args], capture_output=True, text=True, errors="replace",
                           timeout=timeout, check=True)
    except subprocess.TimeoutExpired as error:
        raise OSError("npu-smi timed out") from error
    except subprocess.CalledProcessError as error:
        raise OSError(f"npu-smi exited with status {error.returncode}") from error
    return r.stdout


@dataclass
class AscendChip:
    npu: int  # npu-smi card ID (-i)
    chip: int  # chip ID within the card (-c)
    name: str
    pdev: str  # PCI bus id, "0000:05:00.0"
    mem_total: int = 0  # bytes
    mem_used: int | None = None
    product: str = ""


def parse_info(text: str) -> list[dict]:
    """Parse `npu-smi info` into one dict per chip.

    Each chip is two table rows sharing a cell layout (leading/trailing | stripped, split on |):
        head: | 3       310P3 | OK | NA   45   0 / 0 |        npu name | health | power temp hugepages
        tail: | 0       0     | 0000:05:00.0 | 0   1522 / 47302 |  chip dev | bus | aicore mem-usage
    """
    chips: list[dict] = []
    head: list[str] | None = None
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        m = re.match(r"^(\d+)\s+(\d+)$", cells[0])
        if m and re.fullmatch(r"[0-9a-fA-F]+:[0-9a-fA-F]+:[0-9a-fA-F]+\.[0-9a-fA-F]+", cells[1]):
            if head is None:
                continue
            hm = re.match(r"^(\d+)\s+(\S+)$", head[0])
            if not hm:
                head = None
                continue
            mt = re.search(r"(\d+)\s*/\s*(\d+)", cells[2])
            ht = head[2].split()
            chips.append({
                "npu": int(hm.group(1)), "name": hm.group(2),
                "health": head[1], "power": ht[0] if ht else None,
                "temp": int(ht[1]) if len(ht) > 1 and ht[1].lstrip("-").isdigit() else None,
                "chip": int(m.group(1)), "device": int(m.group(2)), "bus": cells[1],
                "aicore": int(cells[2].split()[0]) if cells[2].split() and cells[2].split()[0].isdigit() else None,
                "mem_used": int(mt.group(1)) if mt else None,
                "mem_total": int(mt.group(2)) if mt else None,
            })
            head = None
        elif re.match(r"^(\d+)\s+(\S+)$", cells[0]) and not cells[1].startswith("0000:"):
            head = cells
    return chips


_WATCH_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\S+)\s+(-?\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$")


def parse_watch_line(line: str) -> dict | None:
    """One `npu-smi info watch` data row: NpuID ChipId Pwr Temp AI-Core AI-Cpu CtrlCpu Mem% MemBW%."""
    m = _WATCH_RE.match(line)
    if not m:
        return None
    return {
        "npu": int(m.group(1)), "chip": int(m.group(2)), "temp": int(m.group(4)),
        "ai_core": int(m.group(5)), "ai_cpu": int(m.group(6)), "ctrl_cpu": int(m.group(7)),
        "mem_pct": int(m.group(8)), "mem_bw": int(m.group(9)),
    }


def parse_proc_mem(text: str) -> list[tuple[int, str, int]]:
    """`npu-smi info -t proc-mem` rows -> (pid, name, bytes)."""
    out = []
    for m in re.finditer(
            r"Process id:\s*(\d+)\s+Process name:\s*(\S+)\s+Process memory\((B|MB)\):\s*(\d+)", text):
        size = int(m.group(4)) * (_MB if m.group(3) == "MB" else 1)
        out.append((int(m.group(1)), m.group(2), size))
    if len(out) != text.count("Process id:") or (not out and
            not re.search(r"no (?:running )?process(?:es)? (?:found|running)", text, re.IGNORECASE)):
        raise ValueError("unrecognized npu-smi process output")
    return out


def _pci_key(pdev: str) -> int:
    return int(re.sub(r"[^0-9a-f]", "", pdev.lower()) or "0", 16)


def chip_adapter_key(pdev: str, chip: int) -> int:
    # Keep Ascend's PCI-plus-chip keys distinct from ordinary Linux PCI keys.
    return (1 << 63) | (_pci_key(pdev) << 8) | chip


def probe() -> list[AscendChip]:
    """Enumerate every Ascend chip npu-smi can see (two subprocess calls, ~1s)."""
    if _npu_path() is None:
        return []
    try:
        text = _run(["info"])
    except OSError:
        return []
    chips = []
    for c in parse_info(text):
        product = ""
        try:
            board = _run(["info", "-t", "board", "-i", str(c["npu"])])
        except OSError:
            board = ""
        pm = re.search(r"Product Name\s*:\s*(.+)", board)
        if pm:
            product = pm.group(1).strip()
        name = f"Ascend {c['name']} chip {c['chip']}"
        if product:
            name += f" ({product})"
        chip = AscendChip(npu=c["npu"], chip=c["chip"], name=name, pdev=c["bus"], product=product)
        if c["mem_total"] is not None:
            chip.mem_total = c["mem_total"] * _MB
            chip.mem_used = c["mem_used"] * _MB if c["mem_used"] is not None else None
        chips.append(chip)
    return chips


class AscendCounters:
    """Same interface as drm.DrmCounters: adapters + sample() -> per-chip per-process usage."""

    PROC_REFRESH_S = 10.0  # proc-mem is ~0.5s per chip; don't pay it every sample
    INFO_REFRESH_S = 10.0  # absolute memory refresh (watch only carries percentages)
    WATCH_STALE_S = 3.0
    WATCH_RETRY_S = 10.0

    def __init__(self, chips: list[AscendChip] | None = None) -> None:
        self.ok = False
        self.chips = chips if chips is not None else probe()
        self.ok = bool(self.chips)
        self.by_key = {chip_adapter_key(c.pdev, c.chip): c for c in self.chips}
        self._watch: dict[tuple[int, int], tuple[float, dict]] = {}
        self._lock = threading.Lock()
        self._last_proc = float("-inf")
        self._last_info = float("-inf")
        self._last_watch_start = float("-inf")
        self._procs: dict[int, dict[int, ProcGpu]] = {}
        self._thread: threading.Thread | None = None
        self._watch_process: subprocess.Popen | None = None
        self._info_error = ""
        self._proc_error = ""
        self._watch_error = "npu-smi watch starting"
        self.process_available = False

    def adapters(self) -> list[Adapter]:
        return [Adapter(luid=key, name=c.name, vendor_id=VENDOR_HUAWEI,
                        dedicated_vram=c.mem_total, shared_system=0, software=False)
                for key, c in self.by_key.items()]

    def start(self) -> None:
        """Start or retry the watch process without leaving an unowned child."""
        now = time.monotonic()
        if not self.ok or (self._thread is not None and self._thread.is_alive()):
            return
        if now - self._last_watch_start < self.WATCH_RETRY_S:
            return
        self._last_watch_start = now
        path = _npu_path()
        if path is None:
            self._watch_error = "npu-smi watch unavailable: npu-smi not found"
            return
        try:
            args = [path, "info", "watch"]
            if len(self.chips) == 1:
                args += ["-i", str(self.chips[0].npu), "-c", str(self.chips[0].chip)]
            self._watch_process = subprocess.Popen(args, stdout=subprocess.PIPE,
                                                   stderr=subprocess.DEVNULL, text=True, errors="replace")
        except OSError as error:
            self._watch_error = f"npu-smi watch unavailable: {error}"
            return
        self._watch_error = "npu-smi watch starting"
        self._thread = threading.Thread(target=self._watch_loop, args=(self._watch_process,), daemon=True)
        self._thread.start()

    def _watch_loop(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                row = parse_watch_line(line)
                if row:
                    with self._lock:
                        self._watch[(row["npu"], row["chip"])] = (time.monotonic(), row)
                        self._watch_error = ""
        finally:
            proc.stdout.close()
            proc.wait()
            with self._lock:
                self._watch.clear()
                self._watch_error = "npu-smi watch stopped"

    def _refresh(self, now: float) -> None:
        """Slow cadence: absolute memory from `info`, per-process memory from proc-mem."""
        self.start()
        if now - self._last_info >= self.INFO_REFRESH_S:
            self._last_info = now
            try:
                rows = parse_info(_run(["info"]))
                by_key = {chip_adapter_key(c["bus"], c["chip"]): c for c in rows}
                for key, chip in self.by_key.items():
                    c = by_key.get(key)
                    chip.mem_used = c["mem_used"] * _MB if c and c["mem_used"] is not None else None
                    if c and c["mem_total"] is not None:
                        chip.mem_total = c["mem_total"] * _MB
                self._info_error = "" if all(c.mem_used is not None for c in self.chips) else \
                    "npu-smi device memory unavailable: selected chip missing or output unrecognized"
            except OSError as error:
                for chip in self.chips:
                    chip.mem_used = None
                self._info_error = f"npu-smi device memory unavailable: {error}"
        if now - self._last_proc >= self.PROC_REFRESH_S:
            self._last_proc = now
            self._procs = {}
            try:
                for key, c in self.by_key.items():
                    per: dict[int, ProcGpu] = {}
                    for pid, name, size in parse_proc_mem(
                            _run(["info", "-t", "proc-mem", "-i", str(c.npu), "-c", str(c.chip)])):
                        g = per.setdefault(pid, ProcGpu(available={"dedicated"}))
                        g.dedicated += size
                    self._procs[key] = per
                self.process_available = True
                self._proc_error = ""
            except (OSError, ValueError) as error:
                self._procs = {}
                self.process_available = False
                self._proc_error = f"npu-smi process memory unavailable: {error}"

    def sample(self) -> tuple[dict[int, dict[int, ProcGpu]], dict[int, AdapterGpu]]:
        """({luid: {pid: ProcGpu}}, {luid: AdapterGpu}); per-chip process memory."""
        if not self.ok:
            return {}, {}
        self._refresh(time.monotonic())
        procs: dict[int, dict[int, ProcGpu]] = defaultdict(dict)
        adapters: dict[int, AdapterGpu] = defaultdict(AdapterGpu)
        for key, chip in self.by_key.items():
            procs[key] = self._procs.get(key, {})
            a = adapters[key]
            if chip.mem_used is not None:
                a.dedicated = float(chip.mem_used)
        return procs, adapters

    def watch_row(self, chip: AscendChip) -> dict | None:
        with self._lock:
            entry = self._watch.get((chip.npu, chip.chip))
            if entry is None or time.monotonic() - entry[0] > self.WATCH_STALE_S:
                return None
            return entry[1]

    @property
    def status(self) -> str:
        with self._lock:
            watch_error = self._watch_error
        if (not watch_error or (watch_error == "npu-smi watch starting" and
                                time.monotonic() - self._last_watch_start > self.WATCH_STALE_S)) \
                and any(self.watch_row(chip) is None for chip in self.chips):
            watch_error = "npu-smi watch data unavailable or stale"
        return ". ".join(filter(None, (self._info_error, self._proc_error, watch_error)))

    def read_device(self, chip: AscendChip, index: int) -> DeviceTelemetry:
        """NVIDIA-shaped telemetry for one chip; watch data is live, sysfs fills the PCIe link."""
        row = self.watch_row(chip)
        dev = DeviceTelemetry(
            index=index, name=chip.name, pci_bus=chip.pdev, driver_model="devdrv",
            source="npu-smi", mem_total=chip.mem_total or None, mem_used=chip.mem_used,
        )
        if row:
            dev.temp = row.get("temp")
            dev.util_gpu = float(row["ai_core"])
            dev.util_mem = float(row["mem_bw"])
        speed = _read_sysfs(f"/sys/bus/pci/devices/{chip.pdev}/current_link_speed")
        if speed:
            gen = {"2.5": 1, "5.0": 2, "8.0": 3, "16.0": 4, "32.0": 5}.get(speed.split()[0])
            dev.pcie_gen = gen
        dev.pcie_width = _int_sysfs(f"/sys/bus/pci/devices/{chip.pdev}/current_link_width")
        return dev

    def close(self) -> None:
        proc = self._watch_process
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass  # the child exited after poll()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if self._thread is not None:
            self._thread.join(timeout=5)


def _read_sysfs(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return None


def _int_sysfs(path: str) -> int | None:
    v = _read_sysfs(path)
    try:
        return int(v) if v else None
    except ValueError:
        return None


# Module-level TTL cache: app.py re-enumerates adapters every TUI cycle, and probe() costs ~1s.
_cache: list[AscendChip] | None = None
_cache_ts = 0.0
_CACHE_TTL = 120.0


def cached_chips() -> list[AscendChip]:
    global _cache, _cache_ts
    now = time.time()
    if _cache is None or now - _cache_ts > _CACHE_TTL:
        _cache = probe()
        _cache_ts = now
    return _cache


def cached_chips_adapters() -> list[Adapter]:
    """Adapters for the cached chip list (cheap: no npu-smi call once chips are cached)."""
    return AscendCounters(chips=cached_chips()).adapters()
