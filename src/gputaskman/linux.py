"""Linux GPU backend: adapters and per-process usage from NVML (and DRM fdinfo for AMD), with an honest WSL mode.

Native Linux: NVML attributes VRAM (running compute/graphics processes) and SM/encoder/decoder load
(process utilization samples) to Linux PIDs, which plays the role the WDDM counters play on Windows.
There is no Linux equivalent of WDDM's shared-memory spill or commit counters, so those stay empty.

WSL: the GPU is paravirtualized through /dev/dxg and the Windows host charges every Linux process's
VRAM to the VM worker (vmwp.exe). NVML's process lists come back empty, and its utilization samples
carry *Windows host* PIDs, which must never be looked up in Linux's /proc. In WSL we therefore only
report device-wide numbers, plus the Linux processes that hold /dev/dxg open, marked unattributed.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from functools import lru_cache

from .dxgi import VENDOR_NVIDIA, Adapter
from .pdh import AdapterGpu, ProcGpu

try:
    import pynvml as N
except ImportError:  # pragma: no cover
    N = None


@lru_cache(maxsize=1)
def is_wsl() -> bool:
    if os.path.exists("/dev/dxg"):
        return True
    try:
        with open("/proc/sys/kernel/osrelease", encoding="ascii") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def pci_key(bus_id: str) -> int:
    """'00000000:01:00.0' -> stable integer adapter key (the Linux stand-in for a WDDM LUID)."""
    return int(re.sub(r"[^0-9a-f]", "", bus_id.lower()) or "0", 16)


def _bus_id(handle) -> str:
    pci = N.nvmlDeviceGetPciInfo(handle)
    bus = pci.busId
    return bus.decode() if isinstance(bus, bytes) else str(bus)


def _nvml_devices() -> list[tuple[int, object]]:
    """(adapter key, handle) per GPU NVML can open; a lost or failing GPU is skipped, not fatal."""
    out = []
    try:
        count = N.nvmlDeviceGetCount()
    except Exception:
        return out
    for i in range(count):
        try:
            h = N.nvmlDeviceGetHandleByIndex(i)
        except Exception:
            continue
        try:
            key = pci_key(_bus_id(h))
        except Exception:
            key = i + 1
        out.append((key, h))
    return out


def enumerate_adapters() -> list[Adapter]:
    """NVIDIA GPUs through NVML, plus (native Linux only) AMD and Intel GPUs through DRM,
    and Huawei Ascend NPUs through npu-smi."""
    out = _nvml_adapters()
    if not is_wsl():
        from . import drm

        seen = {a.luid for a in out}
        out += [a for a in drm.adapters(drm.cards()) if a.luid not in seen]
        # Ascend NPUs (no DRM): dual-chip cards yield one adapter per chip.
        from . import ascend

        seen |= {a.luid for a in out}
        out += [a for a in ascend.cached_chips_adapters() if a.luid not in seen]
    return out


def _nvml_adapters() -> list[Adapter]:
    if N is None:
        return []
    try:
        N.nvmlInit()
    except Exception:
        return []
    out = []
    try:
        for key, h in _nvml_devices():
            try:
                name = N.nvmlDeviceGetName(h)
                name = name.decode() if isinstance(name, bytes) else name
                total = N.nvmlDeviceGetMemoryInfo(h).total
            except Exception:
                continue
            out.append(Adapter(luid=key, name=name, vendor_id=VENDOR_NVIDIA, dedicated_vram=total,
                               shared_system=0, software=False))
    finally:
        try:
            N.nvmlShutdown()
        except Exception:
            pass
    return out


PROC_PID_INIT_INO = 0xEFFFFFFC  # the kernel's fixed inode number for the initial (host) PID namespace


@lru_cache(maxsize=1)
def in_child_pid_namespace() -> bool | None:
    """True inside a container with its own PID namespace, where NVML reports host PIDs. False in the host
    namespace (bare metal, or a container run with --pid=host), where NVML's PIDs are this /proc's PIDs.
    None when it can't be told."""
    try:
        link = os.readlink("/proc/self/ns/pid")  # "pid:[4026531836]"
        return int(link.split("[", 1)[1].rstrip("]")) != PROC_PID_INIT_INO
    except (OSError, IndexError, ValueError):
        return None


def dxg_pids(exclude: set[int] | None = None) -> set[int]:
    """Linux PIDs with /dev/dxg open: the processes that use the GPU under WSL. Other users' processes
    are only visible when running as root."""
    exclude = exclude or set()
    pids = set()
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit() or int(entry.name) in exclude:
            continue
        try:
            fds = os.scandir(f"/proc/{entry.name}/fd")
        except OSError:
            continue
        with fds:
            for fd in fds:
                try:
                    if os.readlink(fd.path) == "/dev/dxg":
                        pids.add(int(entry.name))
                        break
                except OSError:
                    continue
    return pids


def nvidia_holders() -> set[int]:
    """Local PIDs with an NVIDIA device file open (/dev/nvidia0, nvidiactl, nvidia-uvm, ...)."""
    pids = set()
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            fds = os.scandir(f"/proc/{entry.name}/fd")
        except OSError:
            continue
        with fds:
            for fd in fds:
                try:
                    if os.readlink(fd.path).startswith("/dev/nvidia"):
                        pids.add(int(entry.name))
                        break
                except OSError:
                    continue
    return pids


def has_cuda(pid: int) -> bool:
    """libcuda.so mapped: the process has (or is creating) a CUDA context. NVML-only tools (nvidia-smi,
    nvtop, this monitor) open the same device files but never load libcuda, so they don't count."""
    try:
        with open(f"/proc/{pid}/maps", encoding="utf-8", errors="replace") as f:
            return any("/libcuda.so" in line for line in f)
    except OSError:
        return False


class PidMatcher:
    """Containers: NVML reports *host* PIDs, which mean nothing in the container's /proc. Both PID
    namespaces number processes in creation order, so the container's CUDA processes are matched to
    NVML's unresolved PIDs by order. A match is sure ("consistent") when a new NVML PID and a new local
    CUDA process appear together (kept while both live) or the counts are equal; otherwise newest is
    paired with newest as an "order" guess, and leftover NVML PIDs belong to processes outside
    this container (other containers, the host's display server)."""

    def __init__(self) -> None:
        self.pairs: dict[int, int] = {}  # host pid -> local pid, from start events
        self.prev_host: set[int] | None = None
        self.prev_local: set[int] = set()

    def update(self, host: list[int], local: list[int]) -> tuple[dict[int, tuple[int, str]], list[int]]:
        """Returns ({host pid: (local pid, confidence)}, host pids left outside)."""
        first = self.prev_host is None
        hs, ls = set(host), set(local)
        self.pairs = {h: l for h, l in self.pairs.items() if h in hs and l in ls}
        if not first:
            new_h = [h for h in host if h not in self.prev_host and h not in self.pairs]
            new_l = [p for p in local if p not in self.prev_local and p not in self.pairs.values()]
            if len(new_h) == len(new_l) == 1:
                self.pairs[new_h[0]] = new_l[0]
        self.prev_host, self.prev_local = hs, ls
        out = {h: (l, "consistent") for h, l in self.pairs.items()}
        rest_h = sorted((h for h in host if h not in self.pairs), reverse=True)  # newest first
        rest_l = sorted((p for p in local if p not in self.pairs.values()), reverse=True)
        sure = len(rest_h) == len(rest_l)
        for h, l in zip(rest_h, rest_l):
            out[h] = (l, "consistent" if sure else "order")
        return out, rest_h[len(rest_l):]


def _value(v) -> int | None:
    # NVML reports NVML_VALUE_NOT_AVAILABLE (unsigned -1) or None when per-process memory is hidden.
    if v is None or v >= (1 << 63):
        return None
    return int(v)


class LinuxCounters:
    """Same interface as pdh.GpuCounters: sample() -> ({adapter key: {pid: ProcGpu}}, {adapter key: AdapterGpu})."""

    def __init__(self) -> None:
        self.wsl = is_wsl()
        self.ok = False
        self._devices: list[tuple[int, object]] = []
        self._last_ts: dict[int, int] = {}
        self._matcher = PidMatcher()
        # {adapter key: {host pid: ProcGpu}}: NVML processes in another PID namespace that couldn't be
        # matched to a local process. Their PIDs must never be looked up in this /proc.
        self.outside: dict[int, dict[int, ProcGpu]] = {}
        # AMD (amdgpu) and Intel (i915/xe) GPUs: per-process VRAM/GTT and engine time from DRM
        # fdinfo. Native Linux only;
        # under WSL the GPU is /dev/dxg and no DRM driver is involved.
        self.drm = None
        if not self.wsl:
            from .drm import DrmCounters

            drm = DrmCounters()  # cheap: only lists cards; the Collector starts it if an AMD GPU is selected
            self.drm = drm if drm.ok else None
        # A Collector attaches an Ascend provider only when an Ascend chip is selected.
        self.ascend = None
        if N is None:
            return
        try:
            N.nvmlInit()
        except Exception:
            return
        self.ok = True
        try:
            self._devices = _nvml_devices()
        except Exception:
            self._devices = []

    @staticmethod
    def _processes(fn, h) -> list:
        try:
            return list(fn(h))
        except Exception:
            return []

    def _utilization(self, key: int, h) -> dict[int, object]:
        """Latest utilization sample per PID since the previous call."""
        try:
            samples = N.nvmlDeviceGetProcessUtilization(h, self._last_ts.get(key, 0))
        except Exception:  # NVML_ERROR_NOT_FOUND when nothing ran since the last sample
            return {}
        latest: dict[int, object] = {}
        for s in samples:
            if s.pid not in latest or s.timeStamp > latest[s.pid].timeStamp:
                latest[s.pid] = s
        if latest:
            self._last_ts[key] = max(s.timeStamp for s in latest.values())
        return latest

    def sample(self) -> tuple[dict[int, dict[int, ProcGpu]], dict[int, AdapterGpu]]:
        procs: dict[int, dict[int, ProcGpu]] = defaultdict(dict)
        adapters: dict[int, AdapterGpu] = defaultdict(AdapterGpu)
        self._compute_pids: set[int] = set()
        for key, h in self._devices:
            try:
                adapters[key].dedicated = float(N.nvmlDeviceGetMemoryInfo(h).used)
            except Exception:
                pass
            if self.wsl:
                continue  # NVML per-process data is empty or keyed by Windows host PIDs here
            compute = self._processes(N.nvmlDeviceGetComputeRunningProcesses, h)
            graphics = self._processes(N.nvmlDeviceGetGraphicsRunningProcesses, h)
            compute_pids = {p.pid for p in compute}
            self._compute_pids |= compute_pids
            per = procs[key]
            for p in compute + graphics:
                g = per.setdefault(p.pid, ProcGpu())
                used = _value(getattr(p, "usedGpuMemory", None))
                if used is None:
                    g.attributed = False
                else:
                    # A process in both lists reports the same allocation twice; don't double count.
                    g.dedicated = max(g.dedicated, float(used))
            for pid, s in self._utilization(key, h).items():
                if pid not in per:
                    continue  # utilization for a process NVML no longer lists (or another PID namespace)
                g = per[pid]
                g.engines["compute" if pid in compute_pids else "3d"] += float(s.smUtil)
                g.engines["encode"] += float(s.encUtil)
                g.engines["decode"] += float(s.decUtil)
        if not self.wsl:
            self._resolve_namespaces(procs)
        if self.wsl and self._devices:
            users = dxg_pids(exclude={os.getpid()})
            # WSL does not say which GPU a process uses; with one GPU (the common case) that's unambiguous.
            for key, _ in self._devices:
                for pid in users:
                    procs[key][pid] = ProcGpu(attributed=False)
        if self.drm is not None:
            drm_procs, drm_adapters = self.drm.sample()
            procs.update(drm_procs)
            adapters.update(drm_adapters)
        if self.ascend is not None and self.ascend.ok:
            asc_procs, asc_adapters = self.ascend.sample()
            procs.update(asc_procs)
            adapters.update(asc_adapters)
        return procs, adapters

    def _resolve_namespaces(self, procs: dict[int, dict[int, ProcGpu]]) -> None:
        """Re-key NVML's PIDs to local ones. In the host PID namespace (bare metal, --pid=host) they already
        are local, including other users' processes whose /proc/<pid>/fd this monitor can't read. In a
        container's own namespace every NVML PID is a host PID, even one that happens to equal a local
        PID, so all of them are matched by creation order with PidMatcher."""
        self.outside = {}
        nv_pids = {pid for per in procs.values() for pid in per}
        child = in_child_pid_namespace()
        if child is None:  # unknown: PIDs that don't exist here can't be local
            unresolved = sorted(pid for pid in nv_pids if not os.path.exists(f"/proc/{pid}"))
        else:
            unresolved = sorted(nv_pids) if child else []
        if not unresolved:
            self._matcher.update([], [])
            return
        holders = nvidia_holders()
        me = os.getpid()
        local = sorted(p for p in holders - (nv_pids - set(unresolved)) if p != me and has_cuda(p))
        compute = [pid for pid in unresolved if pid in self._compute_pids]  # CUDA, so local libcuda users
        matched, _ = self._matcher.update(compute, local)
        unresolved = set(unresolved)
        for key, per in procs.items():
            moved: dict[int, ProcGpu] = {}  # kept apart: a local PID may equal a host PID not yet moved
            for pid in [p for p in per if p in unresolved]:
                g = per.pop(pid)
                if pid in matched:
                    lpid, confidence = matched[pid]
                    g.inferred_by = confidence
                    moved[lpid] = g
                else:
                    self.outside.setdefault(key, {})[pid] = g
            per.update(moved)

    def close(self) -> None:
        if self.drm is not None:
            self.drm.close()
        if self.ascend is not None:
            self.ascend.close()
        if self.ok:
            try:
                N.nvmlShutdown()
            except Exception:
                pass
            self.ok = False
