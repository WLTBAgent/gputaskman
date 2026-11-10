"""Plain-text / JSON snapshot output (no TUI) - handy for scripts, agents and bug reports."""

from __future__ import annotations

import dataclasses
import json
import os

from .collector import ProcRow, Snapshot
from .llm import gb, server_gauge_text
from .nvml import format_metric
from .pdh import ProcGpu
from .tcc import TccAdapter


WSL_HOST_NOTE = ("WSL + Windows host counters: 'WSL VM' is the exact total of Linux GPU memory (all distros). "
                 "Linux rows get their size by matching the VM's GPU contexts (* matched, ? guess). "
                 "W: rows are Windows processes.")


def wsl_note(s: Snapshot) -> str:
    if s.host_data:
        return WSL_HOST_NOTE
    status = f" Windows host counters: {s.host_status}." if s.host_status else ""
    return WSL_NOTE + status


WSL_EMPTY = ("No Linux process in this distro has the GPU open (other WSL distros and Docker Desktop "
             "are not visible from here).")
WSL_NOTE = ("WSL: VRAM and engine load are device-wide. The Windows host charges every Linux process's "
            "VRAM to the VM, so rows below are processes using the GPU, not their share of it.")


def has_wddm_counters(s: Snapshot, row: ProcRow) -> bool:
    return s.backend == "wddm" or row.origin == "windows"


def has_shared_counters(s: Snapshot, row: ProcRow) -> bool:
    """Per-process GPU memory in system RAM: WDDM's non-local usage, or a DRM client's GTT (amdgpu) or
    system0 (Intel) bytes on Linux."""
    return (has_wddm_counters(s, row) or (s.backend == "drm" and not row.host)) and "shared" in row.gpu.available


def has_engine_counters(s: Snapshot, row: ProcRow) -> bool:
    return (s.backend != "wsl" or row.origin == "windows") and "engine" in row.gpu.available


def vram_mark(inferred_by: object) -> str:
    """'*' / '?' for an inferred owner; anything else (including a malformed recorded value) is unmarked."""
    return {"order": "?", "consistent": "*"}.get(inferred_by, "") if isinstance(inferred_by, str) else ""


def vram_label(g: ProcGpu) -> str:
    """Measured size; '*' / '?' when the size is exact but which local process owns it is inferred."""
    return "n/a" if g.shown_vram is None else gb(g.shown_vram) + vram_mark(g.inferred_by)


INFERRED_NOTE = ("* matched: the size is exact (Windows' counter for a WSL VM GPU context, or NVML for a host PID "
                 "seen from a container); which local process owns it is inferred, and only one explanation fits "
                 "(or every one that fits gives the same size).")
ORDER_NOTE = ("? order guess: the size is exact but several owners fit, so newer processes are paired with newer "
              "GPU contexts / PIDs and outside ones are assumed oldest. Cached foreign-size hints only rank extra "
              "candidates; they never prove ownership. Becomes * with evidence, such as a process starting while "
              "gputaskman runs.")


DRM_NOTES = {
    "legacy": ("VRAM n/a: this driver's fdinfo has missing or unparseable drm-shared-* stats (older amdgpu kernels, "
               "Intel integrated GPUs, xe), so resident memory may include buffers imported from other processes "
               "and would be counted once per importer."),
    "shared_client": ("VRAM and GPU% n/a: this process holds a DRM client other processes hold too (fd passing, "
                      "fork). fdinfo counts the client once and doesn't say which of them uses it."),
    "shared_buffers": ("VRAM n/a: this process's DRM client shares buffers with other clients (dma-buf, e.g. "
                       "windows and video frames). fdinfo counts shared buffers by where they prefer to live, not "
                       "where they are, so its private part can't be measured."),
    "primary": ("VRAM n/a: this process opened the GPU through /dev/dri/card*, which root-owned services such as "
                "logind may also hold where a normal user can't see them; run as root to check."),
    "incomplete": "VRAM and GPU% n/a: some of this process's file descriptors or fdinfo couldn't be read.",
    "partial_scan": ("VRAM and GPU% n/a this refresh: the scan budget ran out before every process's fds were "
                     "listed, so another holder of this DRM client can't be ruled out."),
}
DRM_RESIDENT_NOTE = ("Resident: every buffer the process can reach, shared ones included. The same buffer can "
                     "appear under several processes, so these are not added up.")


def drm_note(g: ProcGpu) -> str:
    return DRM_NOTES.get(g.drm_reason, "")


def inferred_note(g) -> str:
    return ORDER_NOTE if g.inferred_by == "order" else INFERRED_NOTE


METAL_NOTE = ("macOS: unified memory, so GPU memory is part of system RAM. Rows show each process's graphics "
              "footprint (dirty bytes). Paged is footprint's swapped bytes for those same graphics categories, "
              "compressed memory, not a Windows commit estimate. Other users' processes (WindowServer, daemons) "
              "show GPU% only, since reading their memory needs root.")
INTEL_MEMORY_NOTE = ("Intel: used is a sum over inspectable processes (no device counter); "
                     "total is the PCI window size, an estimate. Free is not shown.")
WDDM_DEVICE_NOTE = "Windows does not report temperature, power, clocks or PCIe for this GPU."


def intel_memory_note(s: Snapshot) -> str:
    """Intel's device VRAM used is a sum over visible DRM clients and its total a PCI window size: neither is a
    device counter, so the output says so whenever it shows either."""
    return INTEL_MEMORY_NOTE if (s.backend == "drm" and s.nv and s.nv.source == "Intel sysfs"
                                 and (s.vram_used is not None or s.vram_total)) else ""


def vram_total_label(s: Snapshot) -> str:
    """A total of 0 means unknown (an Intel iGPU, or a card whose memory window can't be read), not 0 bytes."""
    return gb(s.vram_total) if s.vram_total else "N/A"


def to_dict(s: Snapshot) -> dict:
    nv = dataclasses.asdict(s.nv) if s.nv else None
    adapter = dataclasses.asdict(s.adapter) if s.adapter else None
    if isinstance(s.adapter, TccAdapter):
        adapter["pci_domain"] = s.adapter.pci_domain
    procs = []
    for p in s.procs:
        wddm = has_wddm_counters(s, p)
        engines = has_engine_counters(s, p)
        item = {
            "pid": p.pid, "host": p.origin if p.host else None, "name": p.name, "kind": p.kind, "runtimes": p.detail.runtimes,
            "started_at": round(p.detail.create_time, 3) or None,  # unknown (protected process, WSL host row) is null
            "vram_attributed": p.gpu.attributed, "vram_inferred_bytes": p.gpu.inferred,
            "vram_inferred_by": p.gpu.inferred_by or None,
            "vram_bytes": int(p.gpu.dedicated) if p.gpu.attributed and "dedicated" in p.gpu.available else None,
            "vram_resident_bytes": (int(p.gpu.resident_vram) if s.backend == "drm" and not p.host
                                    and p.gpu.resident_vram is not None else None),
            "gtt_resident_bytes": (int(p.gpu.resident_gtt) if s.backend == "drm" and not p.host
                                   and p.gpu.resident_gtt is not None else None),
            "vram_unattributed_reason": p.gpu.drm_reason or None if s.backend == "drm" and not p.host else None,
            "shared_bytes": int(p.gpu.shared) if has_shared_counters(s, p) else None,
            "committed_bytes": int(p.gpu.committed) if wddm and "committed" in p.gpu.available else None,
            "paged_out_bytes": int(p.gpu.paged_out) if wddm and p.gpu.paged_available else None,
            "graphics_swapped_bytes": (int(p.gpu.graphics_swapped) if s.backend == "metal" and p.gpu.attributed
                                       and p.gpu.graphics_swapped is not None else None),
            "gpu_pct": round(p.gpu.gpu_pct, 1) if engines else None,
            "engines": {k: round(v, 1) for k, v in p.gpu.engines.items() if v} if engines else None,
            "engine_unavailable_reason": p.gpu.engine_reason or None,
            "ports": p.detail.ports, "inside": p.inside,
            "model_files": [dataclasses.asdict(m) for m in p.detail.models],
            "installed_models": [dataclasses.asdict(m) for m in p.detail.candidates],
        }
        if p.handy:
            item["handy"] = dataclasses.asdict(p.handy)
        if p.llm:
            item["llm"] = {
                "models": [{"path": g.path, "name": g.name, "label": p.llm.label(i), "arch": g.arch, "kind": g.kind,
                            "kind_label": p.llm.kind_label(i),
                            "kind_hint": p.llm.files[i].kind_hint if i < len(p.llm.files) else "",
                            "decision_type": g.decision_type, "params": g.n_params,
                            "quant": g.quant, "layers": g.n_layers, "ctx_train": g.ctx_train,
                            "gpu_weight_bytes": g.gpu_weight_bytes(),
                            "sources": p.llm.files[i].sources if i < len(p.llm.files) else [],
                            "store_name": p.llm.files[i].store_name if i < len(p.llm.files) else None}
                           for i, g in enumerate(p.llm.models)],
                "segments": [dataclasses.asdict(x) for x in p.llm.segments],
                "kv_bytes_per_token": p.llm.kv_per_token, "n_ctx": p.llm.n_ctx, "flags": p.llm.flags,
                "server": dataclasses.asdict(p.llm.server) if p.llm.server else None,
                "server_source": p.llm.server_source,
                "log_source": p.llm.log_source, "notes": p.llm.notes,
            }
        procs.append(item)
    return {
        "taken": s.taken, "gpu": s.adapter.name if s.adapter else None, "driver": s.driver, "cuda": s.cuda,
        "adapter": adapter,
        "vram_total": s.vram_total or None, "vram_used": s.vram_used, "shared_used": s.shared_used,
        "attributed": s.attributed, "backend": s.backend, "shared_is_spill": s.shared_is_spill,
        "host_status": s.host_status or None,
        "process_status": s.process_status or None,
        "memory_note": intel_memory_note(s) or None,
        "counter_status": s.counter_status or None, "counter_adapter_matched": s.counter_adapter_matched,
        "wsl_split": s.wsl_split or None, "device": nv, "processes": procs,
    }


def to_text(s: Snapshot, limit: int = 30) -> str:
    out = []
    nv = s.nv
    name = s.adapter.name if s.adapter else "GPU"
    if s.backend == "metal":
        used = gb(s.vram_used) if s.vram_used is not None else "N/A"
        out.append(f"{name}  {s.driver}  Metal")
        out.append(f"GPU memory {used} allocated / {gb(s.vram_total)} unified")
        out.append(METAL_NOTE)
        if nv:
            out.append(f"GPU {format_metric(nv.util_gpu, unit='%')}")
    else:
        info = f"  driver {s.driver}" if s.driver else ""
        if s.cuda:
            info += f"  CUDA {s.cuda}"
        out.append(f"{name}{info}  {nv.driver_model if nv else ''}")
        used = gb(s.vram_used) if s.vram_used is not None else "N/A"
        shared = gb(s.shared_used) if s.shared_used is not None else "N/A"
        memory = f"VRAM {used} / {vram_total_label(s)}"
        if s.backend in ("wddm", "drm") or s.host_data:
            memory += f"   shared (in system RAM) {shared}"
        out.append(memory)
    if s.backend == "wsl":
        out.append(wsl_note(s))
    if intel_memory_note(s):
        out.append(intel_memory_note(s))
    if s.counter_status:
        out.append(s.counter_status)
    if nv and nv.source == "Windows WDDM":
        out.append(f"GPU {format_metric(nv.util_gpu, unit='%')}  ({WDDM_DEVICE_NOTE})")
    elif nv and s.backend != "metal":
        rx = nv.pcie_rx_kbs / 1024 if nv.pcie_rx_kbs is not None else None
        tx = nv.pcie_tx_kbs / 1024 if nv.pcie_tx_kbs is not None else None
        pstate = f"P{nv.pstate}" if nv.pstate is not None else "P N/A"
        is_intel = nv.source == "Intel sysfs"
        label = "AMD gfx" if nv.source.startswith("AMD ") else ("Intel GT" if is_intel else "GPU")
        clock = (f"gfx {format_metric(nv.gfx_clock)}" if nv.source.startswith("AMD ") else
                 f"GT {format_metric(nv.sm_clock)}" if is_intel else
                 f"SM {format_metric(nv.sm_clock)}")
        power_kind = f" ({nv.power_kind})" if nv.power_kind else ""
        temp_label = f" ({nv.temp_kind})" if nv.temp_kind else ""
        out.append(f"{label} {format_metric(nv.util_gpu, unit='%')}  mem-bandwidth {format_metric(nv.util_mem, unit='%')}  "
                   f"{format_metric(nv.temp, unit='C')}{temp_label}  {format_metric(nv.power_w, '.0f')}/"
                   f"{format_metric(nv.power_limit_w, '.0f')} W{power_kind}  {pstate}  "
                   f"{clock} MHz  "
                   f"video {format_metric(nv.video_clock)} MHz  "
                   f"PCIe rx {format_metric(rx, '.0f')} MB/s "
                   f"tx {format_metric(tx, '.0f')} MB/s")
    out.append("")
    shown = s.procs[:limit]
    w = max([7] + [len(p.pid_label) for p in shown])  # host PIDs like W:120880 are wider
    out.append(f"{'PID':>{w}}  {'PROCESS':<24} {'KIND':<10} {'RUNTIME':<24} {'VRAM':>9} {'SHARED':>8} {'PAGED':>8} {'GPU%':>5}  INSIDE")
    for p in shown:
        wddm = has_wddm_counters(s, p)
        shared = gb(p.gpu.shared) if has_shared_counters(s, p) and p.gpu.shared >= 1 << 20 else ""
        paged = gb(p.gpu.paged_out) if wddm and p.gpu.paged_out >= 1 << 20 else ""
        if (wddm or (s.backend == "drm" and not p.host)) and "shared" not in p.gpu.available:
            shared = "N/A"
        if wddm and not p.gpu.paged_available:
            paged = "N/A"
        if s.backend == "metal":
            paged = "N/A" if not p.gpu.attributed or p.gpu.graphics_swapped is None else gb(p.gpu.graphics_swapped)
        pid = p.pid_label
        pct = f"{p.gpu.gpu_pct:.0f}" if has_engine_counters(s, p) else "n/a"
        out.append(f"{pid:>{w}}  {p.name[:24]:<24} {p.kind:<10} {p.runtime[:24]:<24} {vram_label(p.gpu):>9} {shared:>8} "
                   f"{paged:>8} {pct:>5}  {p.inside}")
    if any(p.gpu.inferred_by == "consistent" for p in shown):
        out.append(INFERRED_NOTE)
    if any(p.gpu.inferred_by == "order" for p in shown):
        out.append(ORDER_NOTE)
    if s.backend == "drm":
        for note in dict.fromkeys(drm_note(p.gpu) for p in shown if not p.host):
            if note:
                out.append(note)
    if s.process_status:
        out.append(f"DRM process scan: {s.process_status}.")
    if s.backend == "wsl" and s.wsl_split and any(not p.host and not p.gpu.attributed and p.gpu.inferred is None
                                                  for p in s.procs):
        out.append(f"Linux VRAM split: {s.wsl_split}")
    if s.backend == "wsl" and not any(not p.host for p in s.procs):
        out.append(f"{'':>{w}}  {WSL_EMPTY}")
    for p in s.procs:
        if p.handy:
            out.append("")
            if p.handy.selected_model:
                out.append(f"== {p.name} ({p.pid}) Handy selected (settings, not load proof): {p.handy.selected_model}")
            if p.handy.last_event:
                out.append(f"   last Handy event: {p.handy.last_event} {p.handy.last_model or ''} "
                           f"at {p.handy.last_event_local} (runtime-reported history, not current GPU residency)")
        if not p.llm:
            path_args = [m for m in p.detail.models if "path-arg" in m.sources]
            if path_args:
                out.append("")
                out.append(f"== {p.name} ({p.pid_label})  VRAM {vram_label(p.gpu)}")
                for m in path_args:
                    out.append(f"   file: {m.store_name or m.path}  ({gb(m.size)})  evidence: {m.evidence}")
            continue
        out.append("")
        out.append(f"== {p.name} ({p.pid_label})  VRAM {vram_label(p.gpu)}")
        for i, g in enumerate(p.llm.models):
            kind = p.llm.kind_label(i)
            ev = p.llm.files[i].evidence if i < len(p.llm.files) else ""
            out.append(f"   {kind}: {p.llm.label(i)}  [{g.arch}, {g.params_label} params, {g.quant}, {g.n_layers} layers, "
                       f"ctx_train {g.ctx_train}]  {os.path.basename(g.path)}  evidence: {ev}")
        for f in p.llm.other_files:
            kind = f", {f.kind_label}" if f.kind_label else ""
            out.append(f"   file: {f.store_name or f.path}  ({gb(f.size)}{kind})  evidence: {f.evidence}")
        for seg in p.llm.segments:
            out.append(f"   {gb(seg.bytes):>9}  {seg.label}{'' if seg.exact else ''}")
        if p.llm.server:
            sv = p.llm.server
            out.append(f"   server: {sv.kind} {sv.url} ctx={sv.n_ctx} slots={sv.slots_busy}/{sv.slots_total}"
                       + (f" gen={sv.gen_tok_s:.1f} tok/s" if sv.gen_tok_s is not None else "")
                       + ("" if (sv.slots_busy or sv.gen_tok_s is not None) else " (idle)"))
            if p.llm.server_source == "parent":
                out.append("   server source: parent process (API report, not worker residency)")
            if sv.kind in ("vllm", "foundry-local"):
                for m in sv.models:
                    out.append(f"   API-reported model ID: {m['name']}")
            if gauges := server_gauge_text(sv):
                out.append(f"   {gauges}")
        if p.llm.log_source:
            out.append(f"   exact sizes from log: {p.llm.log_source}")
        for n in p.llm.notes:
            out.append(f"   - {n}")
    return "\n".join(out)


def to_json_dict(s: Snapshot, system=None, system_error: str | None = None) -> dict:
    """The --json object: to_dict plus the optional OS-reported "system" section (or why it is missing)."""
    from . import host as hostmod

    d = to_dict(s)
    if system is not None:
        d["system"] = hostmod.to_dict(system)
    elif system_error:
        d["system"] = {"error": system_error}
    return d


def dump(s: Snapshot, as_json: bool, system=None, system_error: str | None = None) -> str:
    """system: an optional host.HostSample (OS-reported system RAM/CPU); system_error: why it is missing."""
    from . import host as hostmod

    if as_json:
        return json.dumps(to_json_dict(s, system, system_error), indent=2, default=str)
    text = to_text(s)
    if system is not None:
        return text + "\n\n" + hostmod.to_text(system)
    return text + (f"\n\nSystem RAM/CPU unavailable: {system_error}" if system_error else "")
