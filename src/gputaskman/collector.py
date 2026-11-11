"""Assembles one Snapshot from Windows counters, NVML or DRM, process inspection and LLM introspection."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from contextlib import ExitStack

import psutil

from .amdsmi import AmdSmi, _bdf
from .adl import Adl
from .dxgi import VENDOR_AMD, VENDOR_NVIDIA, VENDOR_HUAWEI, Adapter
from .handy import HandyEvidence, inspect_handy
from .llm import LlmView, ServerProber, api_key, build_view, gb, ollama_log_path
from . import nvml as nvml_mod
from .nvml import Nvml, NvDevice
from .pdh import ProcGpu
from .procinfo import AI_KINDS, ProcDetail, ProcInspector, classify
from .telemetry import DeviceTelemetry

if sys.platform == "win32":
    from .dxgi import enumerate_adapters
    from .pdh import GpuCounters

    BACKEND = "wddm"
elif sys.platform == "darwin":
    from .macos import MacCounters as GpuCounters
    from .macos import enumerate_adapters, macos_version

    BACKEND = "metal"
else:
    from .linux import LinuxCounters as GpuCounters
    from .linux import enumerate_adapters, is_wsl

    BACKEND = "wsl" if is_wsl() else "nvml"

from .wslhost import ContextTracker, HostStream, interop_enabled, match_adapter

MiB = 1 << 20


@dataclass
class ProcRow:
    pid: int
    name: str
    kind: str
    gpu: ProcGpu
    detail: ProcDetail
    llm: LlmView | None = None
    handy: HandyEvidence | None = None
    host: bool = False  # pid is from another PID namespace, never a local one (see origin)
    origin: str = ""  # with host: "windows" (a Windows process seen from WSL) | "outside" (outside this container)
    gpu_index: int | None = None
    ram: int | None = None  # system RAM (working set) from host.py; None = unknown, not zero
    cpu: float | None = None  # % of the whole machine from host.py; None until the second sample

    @property
    def key(self) -> str:
        """Unique row id: PIDs from other namespaces and local PIDs can collide."""
        pid = f"{self.origin[0]}{self.pid}" if self.host else str(self.pid)
        return f"{self.gpu_index}:{pid}" if self.gpu_index is not None else pid

    @property
    def pid_label(self) -> str:
        return f"{'W' if self.origin == 'windows' else 'H'}:{self.pid}" if self.host else str(self.pid)

    @property
    def runtime(self) -> str:
        return "+".join(self.detail.runtimes[:3])

    @property
    def inside(self) -> str:
        """One-line 'what's in there' summary for the process table."""
        if self.handy:
            if self.handy.last_model:
                return f"Handy last log ({self.handy.last_event}): {self.handy.last_model.rsplit('/', 1)[-1]}"
            if self.handy.selected_model:
                return f"Handy selected (not load proof): {self.handy.selected_model.rsplit('/', 1)[-1]}"
        if self.llm and self.llm.server and self.llm.server.kind == "vllm" and self.llm.server.models:
            source = "parent API" if self.llm.server_source == "parent" else "API"
            return f"reported ({source}): " + ", ".join(m["name"] for m in self.llm.server.models)
        if self.llm and self.llm.models:
            parts = []
            for i, g in enumerate(self.llm.models):
                if g.kind in ("vision projector", "LoRA adapter"):
                    parts.append(f"+{g.kind}")
                else:
                    parts.append(f"{self.llm.label(i)} {g.params_label} {g.quant}")
            return ", ".join(parts)
        if self.detail.models:
            return ", ".join((m.store_name or os.path.basename(m.path)) +
                             (f" ({m.kind_label})" if m.kind_hint else "") for m in self.detail.models[:2])
        if self.detail.candidates:
            return "installed (unproven): " + ", ".join(
                f"{m.store_name or os.path.basename(m.path)} {gb(m.size)}" +
                (f" ({m.kind_label})" if m.kind_hint else "") for m in self.detail.candidates[:2])
        return {
            "compositor": "desktop back buffers + composed frames",
            "wsl-vm": "all Linux GPU memory (every WSL distro + WSLg)",
            "external": "a process in another PID namespace (host or another container)",
            "kernel": "driver/kernel allocations",
            "browser": "page textures, video, canvas/WebGL",
            "video": "video frames (decode/encode)",
            "3D/game": "textures, meshes, render targets",
        }.get(self.kind, "")


@dataclass
class Snapshot:
    taken: float
    adapter: Adapter | None
    adapters: list[Adapter]
    nv: DeviceTelemetry | None  # legacy name; may contain telemetry from either provider
    driver: str
    cuda: str
    vram_total: int
    vram_used: int | None  # WDDM adapter-wide dedicated usage
    shared_used: int | None
    procs: list[ProcRow] = field(default_factory=list)
    backend: str = "wddm"  # wddm (Windows) | nvml (native Linux, NVIDIA) | drm (native Linux, AMD/Intel) | ascend (Huawei NPU) | wsl (device-wide only) | metal (macOS)
    process_status: str = ""  # amdgpu only: what the DRM process scan couldn't cover; empty when it saw everything
    host_status: str = ""  # WSL only: Windows host counters via interop -- ok | starting | off (...) | ...
    wsl_split: str = ""  # WSL only: why Linux rows have no (inferred) VRAM, when they don't
    # False on an APU (amdgpu iGPU): its GTT is ordinary GPU memory, not spill from a full VRAM.
    shared_is_spill: bool = True
    counter_status: str = ""  # Windows counter availability or adapter matching problem;
    # Intel DRM privilege explanation (needs root) when its load/video/power are N/A
    counter_adapter_matched: bool | None = None  # Windows PDH LUID matched the selected DXGI adapter
    process_memory_available: bool = True
    nv_lost: bool = False  # NVML faulted for this GPU after a disable/re-enable; restart restores it
    @property
    def host_data(self) -> bool:
        return self.host_status.startswith("ok")

    @property
    def attributed(self) -> int | None:
        if not self.process_memory_available or self.counter_adapter_matched is False or any(
                "dedicated" not in p.gpu.available for p in self.procs):
            return None
        return int(sum(p.gpu.dedicated for p in self.procs))


def _is_live(g: ProcGpu) -> bool:
    return (not g.attributed or g.dedicated >= MiB or g.gpu_pct >= 0.5 or g.shared >= 16 * MiB
            or g.paged_out >= 16 * MiB)


def select_adapter(adapters: list[Adapter], gpu_index: int) -> Adapter | None:
    choices = ordered_adapters(adapters)
    if gpu_index < 0:
        raise ValueError("--gpu must be a non-negative index")
    if gpu_index >= len(choices):
        if not choices and gpu_index == 0:
            return None
        raise ValueError(f"--gpu {gpu_index} is out of range; {len(choices)} GPU adapter(s) available")
    return choices[gpu_index]


def ordered_adapters(adapters: list[Adapter]) -> list[Adapter]:
    """List all hardware adapters; retain NVIDIA-first ordering for existing users."""
    return sorted((a for a in adapters if not a.software),
                  key=lambda a: (a.vendor_id != VENDOR_NVIDIA, -a.dedicated_vram, a.luid))


def select_nv_device(adapter: Adapter | None, devices: list[NvDevice], gpu_index: int,
                     adapters: list[Adapter] | None = None) -> NvDevice | None:
    if adapter:
        if adapter.vendor_id != VENDOR_NVIDIA:
            return None
        if sys.platform != "win32":
            from .linux import pci_key
            matches = [d for d in devices if d.pci_bus and pci_key(d.pci_bus) == adapter.luid]
            return matches[0] if len(matches) == 1 else None
        if adapter.pci_address is None:
            return None
        if adapters is not None and sum(a.pci_address == adapter.pci_address for a in adapters) != 1:
            return None
        matches = [d for d in devices if _bdf(d.pci_bus) == adapter.pci_address and d.driver_model != "MCDM"]
        if len(matches) == 1:
            return matches[0]
        return None
    matches = [d for d in devices if d.index == gpu_index]
    return matches[0] if len(matches) == 1 else None


def _llm_wanted(row, shared_is_spill: bool) -> bool:
    """Whether a process row gets an LLM breakdown: an AI-kind process with models (or kind LLM),
    not diffusion, and with measured GPU memory. Where spilled system memory is a counter of its own
    (WDDM, discrete DRM cards) both the dedicated and shared counters must be present; on an integrated
    GPU system memory is the memory and there is no spill counter, so dedicated alone suffices."""
    if row.detail.is_diffusion or row.kind not in AI_KINDS or not (row.detail.models or row.kind == "LLM"):
        return False
    need = {"dedicated", "shared"} if shared_is_spill else {"dedicated"}
    return need <= row.gpu.available


class Collector:
    DEEP_TOP_N = 25  # memory-map candidates: the top N VRAM users plus anything busy
    _mac_driver = ""  # "macOS 26.3.1" on the metal backend, read once at startup
    SCAN_BUDGET = 0.15  # seconds of memory-map scanning per sample; leftovers wait for the next sample
    drm_card = None  # the selected GPU's AMD or Intel card, when it is one (drm.DrmCard)
    ascend = None  # AscendCounters, when LinuxCounters owns one
    ascend_chip = None  # the selected GPU's Ascend chip, when it is one (ascend.AscendChip)

    @property
    def backend(self) -> str:
        if BACKEND == "nvml" and self.adapter and self.adapter.vendor_id == VENDOR_HUAWEI:
            return "ascend"  # Huawei NPU: npu-smi is the source, not NVML/DRM
        return "drm" if self.drm_card is not None else BACKEND

    def __init__(self, gpu_index: int = 0, scan_budget: float | None = SCAN_BUDGET, host: bool = True,
                 adapters: list[Adapter] | None = None) -> None:
        self.adapters = [a for a in (enumerate_adapters() if adapters is None else adapters) if not a.software]
        self.adapter = select_adapter(self.adapters, gpu_index)
        with ExitStack() as resources:
            self.counters = GpuCounters()
            resources.callback(self.counters.close)
            self.nvml = Nvml()
            resources.callback(self.nvml.close)
            self.amdsmi = AmdSmi(self.adapter, self.adapters)
            resources.callback(self.amdsmi.close)
            self.adl = Adl(self.adapter, self.adapters)
            resources.callback(self.adl.close)
            resources.pop_all()
        self._mac_driver = f"macOS {macos_version()}".strip() if BACKEND == "metal" else ""
        # Native Linux on an AMD or Intel GPU: device metrics from the driver's sysfs (Intel's are
        # partial and get their fdinfo-derived used-memory stamp), processes from DRM fdinfo.
        drm = getattr(self.counters, "drm", None)
        if drm is not None and self.adapter and self.adapter.vendor_id != VENDOR_NVIDIA:
            self.drm_card = next((c for c in drm.cards if c.key == self.adapter.luid
                                  and c.vendor_id == self.adapter.vendor_id), None)
        if drm is not None and self.drm_card is None:
            self.counters.drm = None  # an NVIDIA GPU is shown: don't walk every process's fds for nothing
        elif drm is not None:
            drm.start(self.drm_card)  # process baseline plus device counters for the selected GPU only
            if scan_budget is None:
                drm.scan_budget = None  # one-shot output sees every process
        # Huawei Ascend NPU (no DRM): one selected chip and its owned watch process.
        self.ascend_chip = None
        self.ascend = None
        if BACKEND == "nvml" and self.adapter and self.adapter.vendor_id == VENDOR_HUAWEI:
            from dataclasses import replace
            from .ascend import AscendCounters, cached_chips, chip_adapter_key

            chip = next((c for c in cached_chips()
                         if chip_adapter_key(c.pdev, c.chip) == self.adapter.luid), None)
            if chip is not None:
                self.ascend_chip = replace(chip)
                self.ascend = AscendCounters(chips=[self.ascend_chip])
                self.counters.ascend = self.ascend
        self.inspector = ProcInspector()
        self.prober = ServerProber()
        self.scan_budget = scan_budget  # None = scan every candidate now (one-shot output)
        self.gpu_index = gpu_index
        # WSL: Linux can't see WSL's own VRAM, but the Windows host can. Ask it when interop is enabled.
        self.host: HostStream | None = None
        self.host_status = ""
        if BACKEND == "wsl":
            if not host:
                self.host_status = "off (--no-host)"
            elif not interop_enabled():
                self.host_status = "off (WSL interop is disabled)"
            else:
                self.host = HostStream()
        self._host_luid: int | None = None  # host adapter for this GPU, matched once and kept
        self.tracker = ContextTracker(self_pid=os.getpid()) if self.host is not None else None
        self._own_start = time.time()  # our NVML context is created now, not when the process started
        self._host_restarted = time.time()

    def _parent_ports(self, pid: int, depth: int = 2, *, vllm: bool = False) -> list[int]:
        d = self.inspector.get(pid)
        if vllm:
            d.parents = []
            candidates = []
            try:
                p = psutil.Process(pid)
                if not d.create_time or p.create_time() != d.create_time or d.vllm_ambiguous:
                    return []
                child_start = d.create_time
                seen = {pid}
                for _ in range(depth):
                    p = p.parent()
                    if p is None:
                        break
                    parent = self.inspector.get(p.pid)
                    if p.pid in seen or not parent.create_time or parent.create_time != p.create_time():
                        return []
                    if parent.create_time > child_start:
                        return []
                    if not parent.vllm_role:
                        break
                    if parent.vllm_ambiguous:
                        return []
                    seen.add(p.pid)
                    d.parents.append(p.pid)
                    child_start = parent.create_time
                    ports = self.inspector.exclusive_ports(p.pid)
                    if parent.ports and set(parent.ports) != set(ports):
                        return []  # a shared listener cannot identify a server
                    if parent.vllm_role == "worker" and ports:
                        return []
                    if parent.vllm_role == "server":
                        candidates.extend(ports)
                return candidates if len(candidates) == 1 else []
            except (psutil.Error, OSError):
                return []
        if d.parents is None:
            d.parents = []
            try:
                p = psutil.Process(pid)
                for _ in range(depth):
                    p = p.parent()
                    if p is None:
                        break
                    d.parents.append(p.pid)
            except (psutil.Error, OSError):
                pass
        return [port for ppid in d.parents for port in self.inspector.exclusive_ports(ppid)]

    def sample(self) -> Snapshot:
        per_luid, adapters = self.counters.sample()
        luid = self.adapter.luid if self.adapter else (next(iter(per_luid), 0))
        procs = per_luid.get(luid, {})
        a = adapters.get(luid)
        if BACKEND == "metal":
            nv = self.counters.device  # IOKit's device stats in the shape the header renders (no NVML on macOS)
        elif self.drm_card is not None:
            from .drm import read_device

            nv = read_device(self.drm_card, self.gpu_index)
        elif self.ascend_chip is not None:
            nv = self.ascend.read_device(self.ascend_chip, self.gpu_index)
        elif self.adapter and self.adapter.vendor_id != VENDOR_NVIDIA:
            # NVML can't describe this GPU, and a sweep of every NVIDIA device costs ~60 ms per device.
            nv = (self.adl.sample() or self.amdsmi.sample()) if self.adapter.vendor_id == VENDOR_AMD else None
        else:
            nvs = self.nvml.sample() if self.nvml.ok else []
            nv = select_nv_device(self.adapter, nvs, self.gpu_index, self.adapters)
        total = (nv.mem_total if nv else 0) or (self.adapter.dedicated_vram if self.adapter else 0)

        snap = Snapshot(
            taken=time.time(), adapter=self.adapter, adapters=self.adapters, nv=nv,
            driver=self._mac_driver or (self.amdsmi.driver if nv and nv.source == "AMD SMI"
                                        else self.nvml.driver if nv and nv.source == "NVML"
                                        else f"kernel {os.uname().release}" if nv and nv.source == "AMD sysfs" else ""),
            cuda=self.nvml.cuda_version if nv and nv.source == "NVML" else "", vram_total=total,
            vram_used=int(a.dedicated) if a and a.dedicated is not None else None,
            shared_used=int(a.shared) if a and a.shared is not None else None, backend=self.backend,
            shared_is_spill=not (self.drm_card is not None and self.drm_card.integrated),
        )

        snap.host_status = self.host_status
        if BACKEND == "wddm" and self.adapter:
            snap.counter_adapter_matched = luid in per_luid or luid in adapters
            if not snap.counter_adapter_matched:
                snap.counter_status = ("Windows GPU counters could not be matched to the selected adapter; "
                                       "VRAM and processes are unavailable. The adapter may be temporarily "
                                       "unavailable, or GPU-PV VMs may not expose these counters.")
            elif a is None or a.dedicated is None or a.shared is None:
                snap.counter_status = ("Some Windows adapter memory counters are unavailable; "
                                       "GPU-PV VMs may not expose them.")
            if (nv is None and self.adapter.vendor_id == VENDOR_NVIDIA and self.adapter.pci_address
                    and any(_bdf(b) == self.adapter.pci_address for b in nvml_mod.LOST)):
                snap.nv_lost = True
                snap.counter_status = " ".join(filter(None, (
                    snap.counter_status, "NVIDIA telemetry stopped after this GPU was re-enabled; "
                                         "restart gputaskman to restore it.")))
            snap.counter_status = " ".join(filter(None, (snap.counter_status, self.adapter.identity_status)))
        if self.drm_card is not None:
            snap.process_status = self.counters.drm.status
            if self.drm_card.intel and nv is not None:
                from .drm import intel_counter_status

                snap.counter_status = intel_counter_status(self.drm_card, nv)
        if self.ascend is not None:
            snap.process_memory_available = self.ascend.process_available
            snap.counter_status = self.ascend.status
        elif self.backend == "ascend":
            snap.process_memory_available = False
            snap.counter_status = "Ascend chip unavailable: npu-smi could not match the selected adapter"
        live = [(pid, g) for pid, g in procs.items() if _is_live(g)]
        live.sort(key=lambda x: -(x[1].shown_vram or 0))
        for pid, g in live:
            d = self.inspector.get(pid)
            row = ProcRow(pid=pid, name=d.name, kind=classify(d, g.engines), gpu=g, detail=d)
            snap.procs.append(row)

        for pid, g in getattr(self.counters, "outside", {}).get(luid, {}).items():
            # A container sees the host's PIDs from NVML; unmatched ones are other containers or the host.
            snap.procs.append(ProcRow(pid, "(outside this container)", "external", g,
                                      ProcDetail(pid, "(outside this container)"), host=True, origin="outside"))
        if self.host is not None:  # current inferred sizes drive both scan priority and the LLM breakdown
            self._add_host_rows(snap)
        local_rows = [r for r in snap.procs if not r.host]
        deep_pids = {r.pid for r in local_rows[: self.DEEP_TOP_N]} | {
            r.pid for r in local_rows if r.gpu.gpu_pct >= 2}
        deadline = None if self.scan_budget is None else time.perf_counter() + self.scan_budget
        ollama_runners = []
        for row in local_rows:  # largest VRAM first, so the budget goes to what matters
            d, g = row.detail, row.gpu
            mem = (g.shown_vram or 0) + g.shared
            if (row.pid in deep_pids and (deadline is None or time.perf_counter() < deadline)
                    and self.inspector.needs_scan(d, mem)):
                self.inspector.scan(d, mem)
            row.kind = classify(d, g.engines)
            # Old Ollama: "ollama.exe runner ..."; newer Ollama: its own lib\ollama\llama-server.exe
            exe = d.exe.lower()
            if row.kind in AI_KINDS and ("\\ollama\\" in exe or "/ollama/" in exe or
                                         (d.name.lower().startswith("ollama") and "runner" in d.cmdline)):
                ollama_runners.append(row)
        for row in snap.procs:
            if row.host:
                continue
            if os.path.basename(row.detail.exe).lower() in ("handy.exe", "handy"):
                row.handy = inspect_handy(row.detail.create_time)
            if not _llm_wanted(row, snap.shared_is_spill):
                continue
            ports = self.inspector.exclusive_ports(row.pid)
            key = api_key(row.detail.cmdline)
            vllm = bool(row.detail.vllm_role)
            kind = "vllm" if vllm else None
            own_origin = ("process", row.pid, row.detail.create_time)
            if vllm and (row.detail.vllm_ambiguous or len(ports) > 1 or
                         set(ports) != set(row.detail.ports)):
                ports = []
                parent_ports = []
            else:
                parent_ports = self._parent_ports(row.pid, vllm=vllm)
            if vllm and ports and parent_ports:
                ports, parent_ports = [], []  # two possible server processes: do not pick one
            parent_origin = ("parent", tuple((pid, self.inspector.get(pid).create_time)
                                            for pid in row.detail.parents or []))
            own = self.prober.get(ports, key, kind=kind, origin=own_origin)
            host = self.prober.get(parent_ports, kind=kind, origin=parent_origin)
            log = ollama_log_path() if row in ollama_runners and len(ollama_runners) == 1 else None
            # On an APU, GTT is the GPU's ordinary memory (often most of it), not spill: count it as held memory.
            held = (row.gpu.shown_vram or 0) + (0 if snap.shared_is_spill else (row.gpu.shared or 0))
            row.llm = build_view(row.detail, held, row.gpu.shared if snap.shared_is_spill else 0,
                                 own or host, log, host=host if own else None,
                                 attributed=row.gpu.shown_vram is not None,
                                 server_source="process" if own else "parent" if host else None)
            locked = None if own else self.prober.locked_port(ports, key, kind=kind, origin=own_origin)
            if locked:
                why = ("rejected the key from the launch arguments" if key else
                       "wants an API key, and none was found in the launch arguments (keys from LLAMA_API_KEY "
                       "or --api-key-file are not read)")
                row.llm.notes.append(f"Local API on port {locked} {why}. Server details are skipped.")
            if not host and self.prober.locked_port(parent_ports, kind=kind, origin=parent_origin):
                row.llm.notes.append("Parent-process API locked (401). Parent probes never send the worker's "
                                     "API key; parent server details are skipped.")

        self.inspector.prune(set(procs))
        return snap

    def _add_host_rows(self, snap: Snapshot) -> None:
        """Windows processes on this GPU, plus the WSL VM itself (vmwp.exe: the exact total of Linux VRAM)."""
        if self.host.ended and time.time() - self._host_restarted > 30:
            self.host.close()  # PowerShell exited (e.g. killed, or the host was busy): try again
            self.host = HostStream()
            self._host_restarted = time.time()
        hs = self.host.latest()
        if hs is None:
            snap.host_status = "starting" if self.host.status in ("starting", "ok") else self.host.status
            return
        # Match the host adapter once: NVML is read now while the host sample can be a second old, so
        # re-matching every refresh fails whenever memory moves fast (a model loading) and the rows flicker.
        luid = self._host_luid if self._host_luid in hs.adapters else match_adapter(hs, snap.vram_used)
        if luid is None:
            # Memory can swing by GBs between the two readings while a workload loads: retry for a while.
            early = time.time() - self._host_restarted < 20
            snap.host_status = "starting" if early else "unmatched (no Windows adapter matches this GPU's usage)"
            return
        self._host_luid = luid
        age = time.time() - hs.taken
        snap.host_status = "ok" if age < 3 else f"ok (last Windows sample {age:.0f}s old)"
        a = hs.adapters.get(luid)
        if a is not None:
            snap.shared_used = int(a.shared) if a.shared is not None else None
        procs = hs.procs.get(luid, {})
        vm_workers = [pid for pid in procs if hs.names.get(pid, "").lower() == "vmwp"]
        host_rows: list[ProcRow] = []
        for pid, g in procs.items():
            if not _is_live(g):
                continue
            raw = hs.names.get(pid, "?")
            if raw.lower() == "vmwp":
                # One VM worker is WSL's; several could include Hyper-V VMs or Windows Sandbox.
                name = "WSL VM (vmwp.exe)" if len(vm_workers) == 1 else "VM worker (vmwp.exe)"
                kind = "wsl-vm"
            else:
                name = raw if raw in ("System", "Idle", "?") or raw.lower().endswith(".exe") else f"{raw}.exe"
                kind = "kernel" if pid == 4 else classify(ProcDetail(pid, name), g.engines)
            host_rows.append(ProcRow(pid, name, kind, g, ProcDetail(pid, name), host=True, origin="windows"))
        # Largest first; Linux rows sit right under the WSL VM row, whose total they share.
        host_rows.sort(key=lambda r: -r.gpu.dedicated)
        at = next((i + 1 for i, r in enumerate(host_rows) if r.kind == "wsl-vm"), 0)
        if len(vm_workers) == 1 and self.tracker is not None:
            self._attribute(snap, luid, hs.contexts.get(luid, {}).get(vm_workers[0], []))
        snap.procs.sort(key=lambda r: -(r.gpu.shown_vram or 0))
        snap.procs = host_rows[:at] + snap.procs + host_rows[at:]

    def _attribute(self, snap: Snapshot, luid: int, contexts: list[float]) -> None:
        linux = [r for r in snap.procs if not r.host and not r.gpu.attributed]
        # This monitor holds a (0-byte) context too, through NVML.
        holders = [(r.pid, r.detail.create_time) for r in linux] + [(os.getpid(), self._own_start)]
        guesses, reason = self.tracker.update(luid, contexts, holders)
        snap.wsl_split = reason
        for r in linux:
            g = guesses.get(r.pid)
            if g is not None:
                r.gpu.inferred, r.gpu.inferred_by = g.bytes, g.confidence

    def close(self) -> None:
        with ExitStack() as resources:
            if self.host is not None:
                resources.callback(self.host.close)
            resources.callback(self.adl.close)
            resources.callback(self.amdsmi.close)
            resources.callback(self.nvml.close)
            resources.callback(self.counters.close)
