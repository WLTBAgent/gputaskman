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

import os
import re
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
        for candidate in ("npu-smi", "/usr/local/bin/npu-smi"):
            try:
                subprocess.run([candidate, "-h"], capture_output=True, timeout=5)
                _NPUSMI = candidate
                break
            except (OSError, subprocess.SubprocessError):
                continue
    return _NPUSMI


def _run(args: list[str], timeout: float = 10.0) -> str:
    """npu-smi output, or '' on any failure (missing tool, error exit, timeout)."""
    path = _npu_path()
    if path is None:
        return ""
    try:
        r = subprocess.run([path, *args], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout if r.returncode == 0 else ""


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
        if m and re.match(r"^\d+:\d+:\d+\.\d+$", cells[1]):
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
                "aicore": int(cells[2].split()[0]) if cells[2].split() else None,
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
    """`npu-smi info -t proc-mem` rows -> (pid, name, megabytes)."""
    out = []
    for m in re.finditer(r"Process id:(\d+)\s+Process name:(\S+)\s+Process memory\(MB\):(\d+)", text):
        out.append((int(m.group(1)), m.group(2), int(m.group(3))))
    return out


def _pci_key(pdev: str) -> int:
    return int(re.sub(r"[^0-9a-f]", "", pdev.lower()) or "0", 16)


def probe() -> list[AscendChip]:
    """Enumerate every Ascend chip npu-smi can see (two subprocess calls, ~1s)."""
    text = _run(["info"])
    if not text:
        return []
    chips = []
    for c in parse_info(text):
        product = ""
        board = _run(["info", "-t", "board", "-i", str(c["npu"])])
        pm = re.search(r"Product Name\s*:\s*(.+)", board)
        if pm:
            product = pm.group(1).strip()
        name = f"Ascend {c['name']} chip {c['chip']}"
        if product:
            name += f" ({product})"
        chip = AscendChip(npu=c["npu"], chip=c["chip"], name=name, pdev=c["bus"], product=product)
        if c["mem_total"] is not None:
            chip.mem_total = c["mem_total"] * _MB
            chip.mem_used = (c["mem_used"] or 0) * _MB
        chips.append(chip)
    return chips


class AscendCounters:
    """Same interface as drm.DrmCounters: adapters + sample() -> per-chip per-process usage."""

    PROC_REFRESH_S = 10.0  # proc-mem is ~0.5s per chip; don't pay it every sample
    INFO_REFRESH_S = 10.0  # absolute memory refresh (watch only carries percentages)

    def __init__(self, chips: list[AscendChip] | None = None) -> None:
        self.ok = False
        self.chips = chips if chips is not None else probe()
        self.ok = bool(self.chips)
        self.by_key = {(_pci_key(c.pdev) << 8) | c.chip: c for c in self.chips}
        self._watch: dict[tuple[int, int], dict] = {}  # (npu, chip) -> latest watch row
        self._watch_dead = False
        self._lock = threading.Lock()
        self._last_proc = 0.0
        self._last_info = 0.0
        self._procs: dict[int, ProcGpu] = {}
        self._thread: threading.Thread | None = None

    def adapters(self) -> list[Adapter]:
        return [Adapter(luid=key, name=c.name, vendor_id=VENDOR_HUAWEI,
                        dedicated_vram=c.mem_total, shared_system=0, software=False)
                for key, c in self.by_key.items()]

    chip_key = staticmethod(_pci_key)  # exposed so the Collector can match adapter keys

    def start(self) -> None:
        """Spawn the watch-stream reader (once). Cheap: one npu-smi process, a line per second."""
        if self._thread is not None or not self.ok:
            return
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def _watch_loop(self) -> None:
        path = _npu_path()
        if path is None:
            self._watch_dead = True
            return
        try:
            proc = subprocess.Popen([path, "info", "watch"], stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, text=True)
        except OSError:
            self._watch_dead = True
            return
        assert proc.stdout is not None
        for line in proc.stdout:
            row = parse_watch_line(line)
            if row:
                with self._lock:
                    self._watch[(row["npu"], row["chip"])] = row
        self._watch_dead = True  # stream ended (tool exit or broken pipe)
        try:
            proc.wait(timeout=5)
        except subprocess.SubprocessError:
            proc.kill()

    def _refresh(self, now: float) -> None:
        """Slow cadence: absolute memory from `info`, per-process memory from proc-mem."""
        self.start()
        if now - self._last_info >= self.INFO_REFRESH_S:
            self._last_info = now
            for c in parse_info(_run(["info"])):
                chip = self.by_key.get((_pci_key(c["bus"]) << 8) | c["chip"]) if c["bus"] else None
                if chip and c["mem_total"] is not None:
                    chip.mem_total = c["mem_total"] * _MB
                    chip.mem_used = (c["mem_used"] or 0) * _MB
        if now - self._last_proc >= self.PROC_REFRESH_S:
            self._last_proc = now
            procs: dict[int, ProcGpu] = {}
            for c in self.chips:
                for pid, name, mb in parse_proc_mem(
                        _run(["info", "-t", "proc-mem", "-i", str(c.npu), "-c", str(c.chip)])):
                    g = procs.setdefault(pid, ProcGpu())
                    g.dedicated += mb * _MB
            self._procs = procs

    def sample(self) -> tuple[dict[int, dict[int, ProcGpu]], dict[int, AdapterGpu]]:
        """({luid: {pid: ProcGpu}}, {luid: AdapterGpu}); per-chip process memory."""
        if not self.ok:
            return {}, {}
        self._refresh(time.time())
        procs: dict[int, dict[int, ProcGpu]] = defaultdict(dict)
        adapters: dict[int, AdapterGpu] = defaultdict(AdapterGpu)
        for key, chip in self.by_key.items():
            per = procs[key]
            for pid, g in getattr(self, "_procs", {}).items():
                per[pid] = ProcGpu(dedicated=g.dedicated, available={"dedicated"})
            a = adapters[key]
            if chip.mem_used is not None:
                a.dedicated = float(chip.mem_used)
        return procs, adapters

    def watch_row(self, chip: AscendChip) -> dict | None:
        with self._lock:
            return self._watch.get((chip.npu, chip.chip))

    def read_device(self, chip: AscendChip, index: int) -> DeviceTelemetry:
        """NVIDIA-shaped telemetry for one chip; watch data is live, sysfs fills the PCIe link."""
        row = self.watch_row(chip)
        if row is None and not self._watch_dead:
            self.start()
            time.sleep(0.2)  # give the just-started stream a moment (one-shot mode)
            row = self.watch_row(chip)
        if row is None:  # watch unavailable: one synchronous `info` (covers all chips)
            for c in parse_info(_run(["info"])):
                if c["npu"] == chip.npu and c["chip"] == chip.chip:
                    row = {"temp": c["temp"], "ai_core": c["aicore"] or 0, "ai_cpu": 0,
                           "ctrl_cpu": 0, "mem_pct": 0, "mem_bw": 0}
        dev = DeviceTelemetry(
            index=index, name=chip.name, pci_bus=chip.pdev, driver_model="devdrv",
            source="npu-smi", mem_total=chip.mem_total or None, mem_used=chip.mem_used,
        )
        if row:
            dev.temp = row.get("temp")
            dev.util_gpu = float(row.get("ai_core") or 0)
            dev.util_mem = float(row.get("mem_bw") or 0) or None
        speed = _read_sysfs(f"/sys/bus/pci/devices/{chip.pdev}/current_link_speed")
        if speed:
            gen = {"2.5": 1, "5.0": 2, "8.0": 3, "16.0": 4, "32.0": 5}.get(speed.split()[0])
            dev.pcie_gen = gen
        dev.pcie_width = _int_sysfs(f"/sys/bus/pci/devices/{chip.pdev}/current_link_width")
        return dev

    def close(self) -> None:
        pass  # the watch thread is a daemon; it dies with the process


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
