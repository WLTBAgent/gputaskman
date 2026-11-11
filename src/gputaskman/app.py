"""The htop-style Textual UI."""

from __future__ import annotations

import os
from collections import deque
from contextlib import ExitStack

from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Markdown, Select, Static
from textual.worker import get_current_worker

from .collector import BACKEND, Collector, ProcRow, Snapshot, enumerate_adapters, ordered_adapters
from .host import HostSample, HostSampler
from .llm import gb, redact_cmdline
from .nvml import NvDevice, format_metric
from .recovery import LiveCollectors
from .report import (DRM_RESIDENT_NOTE, METAL_NOTE, WSL_EMPTY, drm_note, has_engine_counters, has_spill_counters, has_wddm_counters,
                     inferred_note, intel_memory_note, vram_label, vram_total_label, wsl_note)

KIND_STYLE = {
    "LLM": "bold magenta", "decision": "bold orchid", "decision?": "orchid",
    "embedding": "plum2", "speech": "light_sky_blue1", "diffusion": "orchid",
    "AI": "medium_purple1", "compute": "orange3", "3D/game": "green3",
    "video": "yellow", "browser": "dodger_blue2", "app": "cyan", "compositor": "grey50",
    "kernel": "grey50", "system": "grey50", "wsl-vm": "bold spring_green2", "external": "grey50",
}
SEG_STYLE = {"weights": "magenta", "kv": "dark_orange", "compute": "yellow", "runtime": "grey58", "unknown": "grey42"}
SPARK = "▁▂▃▄▅▆▇█"

EXPLAIN = {
    "wsl-vm": "The WSL virtual machine as Windows sees it. Windows charges all GPU memory used by Linux processes "
              "in every WSL distro (and WSLg) to vmwp.exe, so this is the exact total. The split across Linux "
              "processes is not exposed.",
    "compositor": "Desktop Window Manager: its own back buffers and the composed image for each monitor. "
                  "Windows' 'Dedicated Usage' counter shows dwm as huge (it counts every app surface it can see); "
                  "this tool prefers Local Usage for attribution. The adapter counter is the total; "
                  "per-process samples can overlap or differ in timing.",
    "kernel": "Driver/kernel-owned allocations: paging buffers, display scan-out, firmware, scheduler state.",
    "system": "Session/system process holding display and cursor surfaces.",
    "browser": "Chromium GPU process: page tiles, images, video frames, canvas, WebGL/WebGPU buffers. "
               "Deeper: edge://gpu, or DevTools > Performance with the GPU track.",
    "3D/game": "Textures, meshes, render targets and descriptor heaps. Windows only sees opaque allocations; "
               "per-resource names need a graphics debugger (PIX on Windows, NVIDIA Nsight Graphics).",
    "video": "Decoded/encoded video frames and codec surfaces (NVIDIA NVDEC/ENC)",
    "compute": "Compute allocations are opaque to the OS. Runtime-specific profilers can provide more detail (CUDA: cudaMalloc with Nsight Systems).",
    "diffusion": "Model components are identified from launch arguments; GPU allocations cannot be split by component.",
    "AI": "ML runtime allocations (tensors, workspaces). PyTorch: torch.cuda.memory_summary(); "
          "ONNX Runtime/DirectML pools are opaque from outside.",
    "app": "UI surfaces: swap chains, window back-buffers, cached bitmaps/glyphs.",
}

HELP_MD = """\
# What can be known about GPU memory?

Press **m** (or click **Slim/Full** in the footer) to switch between full diagnostics
and slim monitoring. Slim hides hardware identifiers and secondary telemetry details;
live readings, charts, processes, inspection, and provenance/availability warnings stay visible.
This display-only choice applies to all GPUs and resets to full when the app restarts.

| Layer | Source | Works for |
|---|---|---|
| **Device** telemetry | NVML, AMD ADL, or optional AMD SMI; amdgpu or Intel (i915) sysfs on Linux | availability varies by GPU and driver |
| **Per-process** VRAM, spill to RAM, per-engine load | Windows WDDM counters (dxgkrnl); DRM fdinfo for amdgpu or Intel (i915) on Linux | workloads visible to WDDM or DRM |
| **Which runtime** (CUDA, HIP, Vulkan, D3D12, ggml, ONNX, PyTorch...) | DLLs mapped in the process | everything |
| **Which model file** | memory-mapped files + command line | llama.cpp, stable-diffusion.cpp, LM Studio, Ollama |
| **Model name & kind** (LLM, decision, embedding, speech) | GGUF header, Ollama/LM Studio/Hugging Face stores, local server APIs | GGUF family |
| **Weights / KV cache / compute** split | GGUF header math, llama.cpp/Ollama logs, local server APIs | GGUF/llama.cpp family |
| Per-texture / per-buffer names | needs a debugger (PIX, Nsight) attached to the app | games, 3D apps |

**Evidence** shows how a model was identified: `api` (the runtime says it is loaded), `mapped`
(Windows shows the file mapped into the process right now), `cmdline` (a launch argument only).
Two sources, including `api` or `mapped`, is shown as *confirmed*.

**Local workloads only:** inference, decision models, vision, speech, training, graphics and video.
Cloud-hosted models are outside this monitor's scope. DLLs identify runtimes, not models or tasks.
`decision?` / *likely decision model (name)* is a filename or model-folder hint when a specific
GGUF kind is missing. It does not confirm the kind, GPU residency, or execution. Installed
candidates remain unproven; engine activity is measured per process, not per model.

**Linux / WSL:** on native Linux, per-process VRAM and SM/encoder/decoder load come from NVML
(no spill or commit counters exist there, so those columns stay blank). **AMD GPUs on Linux**
(`amdgpu`) use sysfs for the device and DRM fdinfo (`/proc/<pid>/fdinfo`) per process: VRAM,
GTT (GPU memory in system RAM, shown as SPILL), and gfx/compute/video engine time (`n/a` for a new client's
first sample and for ROCm/HIP processes: their compute runs on `/dev/kfd` queues the kernel doesn't time per
process). VRAM and SPILL are charged to a process only when its DRM clients are provably private: one holder,
opened through a render node, and no shared buffers. Otherwise they show `n/a` and the bytes stay unattributed;
the inspector still shows the resident bytes it can reach (shared buffers included, so not added up). **Intel GPUs
on Linux** (`i915`) take the same path: sysfs for the device (actual GT clock, throttle reasons), fdinfo for engine load and, on a discrete Arc, local-memory
VRAM. On an integrated GPU a provably private client's system memory is its GPU memory in the VRAM column (SPILL stays
`N/A`: system memory is its memory there, not spill); shared clients show `n/a`, with the resident bytes still in the inspector.
`xe` cards are detected but their per-process fdinfo isn't
decoded yet (rows show `n/a`). With no device-wide memory counter, an Intel card's VRAM used is a sum over the processes
it can inspect (local memory on a discrete card, system memory on an integrated GPU; shared buffers can be counted twice; kernel-owned memory and uninspectable processes are missing), its total is
the size of the card's local-memory PCI window (an estimate; unknown on an integrated GPU), and free memory is not shown. Only processes you can inspect are
listed, and the header says how many it couldn't; run with `sudo` to see other users'. Under **WSL**, Windows
charges all Linux GPU memory to the VM (`vmwp.exe`), so VRAM and load are device-wide only; rows
are Linux processes with `/dev/dxg` open and show `n/a` rather than a made-up share. When WSL
interop is enabled, the Windows host's counters are read through `powershell.exe`: **WSL VM**
(`vmwp.exe`) is then the exact total of Linux GPU memory, and Windows apps appear as `W:<pid>`
rows. `--no-host` turns this off. Linux rows get their VRAM by matching the VM's GPU contexts
(newest first) to Linux processes: `*` when candidates agree on the size, `?` for a heuristic owner.
Cached foreign sizes only help rank guesses; they never prove ownership across runs.

**Why does nvidia-smi say N/A?** GeForce cards on Windows run in *WDDM* mode: Windows, not the
NVIDIA driver, owns video memory, so NVML can't attribute it to processes. The Windows counters can.

**dwm.exe looks huge elsewhere?** Windows' per-process "Dedicated Usage" counts every surface a
process can *see*, so dwm can show nearly the whole card. VRAM here is what's charged to each
process ("Local Usage" when available). The adapter counter is the total, not the sum of rows.

**SPILL** = GPU-accessible system RAM. This can include pinned host buffers or intentional
allocations, not just overflow. Large usage under VRAM pressure can indicate fallback;
the counter alone does not prove it.

**PAGED** = max(Total Committed - Local Usage - Non Local Usage, 0). This is an estimate of
non-resident committed memory, not a measured eviction count or proof of pagefile activity.

**Inside** labels are inferred from runtimes and files. Log sizes describe allocated buffers,
not necessarily current residency. A game's textures cannot be enumerated here.

**GPU%** = busiest engine (like Task Manager). On GeForce/WDDM, CUDA kernels are usually reported
on the **3D** engine. AMD device GPU charts show graphics-engine activity, which is not
necessarily equivalent to NVML GPU utilization. Vulkan does not expose another process's HIP internals.

**System panel, RAM and CPU% columns** are reported by the OS (`NtQuerySystemInformation` on Windows,
psutil elsewhere), not by the GPU. RAM is a process's working set, shared pages included, so rows can add up
to more than the machine uses; CPU% is its share of the whole machine, like Task Manager. This is system memory:
it is never GPU bytes and never added to a GPU total. `n/a` means the OS wouldn't say (or it is a Windows-side
row seen from WSL); `…` means one more sample is needed. Inside WSL they describe the Linux VM.
`--no-system` hides them.

Keys: `↑/↓` select · `s` sort · `c` chart · `a` show all · `p` pause · `?` this help · `q` quit

All eight trend charts are visible by default. They stack in narrow terminals and
share rows when there is room. Click the **Charts** dropdown to show all charts or
one metric; `c` cycles the same choices. A blank sample means no reading; zero is
drawn at the baseline. Percentage charts use 0–100%, temperature uses 0–120°C,
and the power and SM clock scales are shown below the charts. In short terminals,
scroll the dashboard or inspector without losing the process list.
"""


def spark(values, width: int = 30, top: float = 100.0) -> str:
    vals = list(values)[-width:]
    return "".join(
        SPARK[min(max(int(v / top * (len(SPARK) - 1) + 0.5), 0), len(SPARK) - 1)]
        if v is not None else " "
        for v in vals
    )


def stacked_bar(parts: list[tuple[float, str]], total: float, width: int) -> Text:
    t = Text()
    used = 0
    for value, style in parts:
        n = int(round(value / total * width)) if total else 0
        n = min(n, width - used)
        if n > 0:
            t.append("█" * n, style=style)
            used += n
    t.append("░" * (width - used), style="grey27")
    return t


class HelpScreen(ModalScreen):
    BINDINGS = [Binding("escape,q,question_mark", "app.pop_screen", "Close")]
    DEFAULT_CSS = """
    HelpScreen { align: center middle; }
    HelpScreen > VerticalScroll { width: 140; max-width: 95%; height: 85%; border: thick $accent; background: $surface; padding: 1 2; }
    """

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Markdown(HELP_MD)


METAL_CHARTS = ("GPU", "VRAM")


class GpuTaskManager(App):
    TITLE = "gputaskman"
    CSS = """
    Screen { layout: vertical; }
    #dashboard { height: auto; max-height: 50%; }
    #system { height: auto; border: round $success; padding: 0 1; }
    #device { height: auto; border: round $accent; padding: 0 1; }
    #history { height: auto; border: round $secondary; padding: 0 1; }
    #chart-controls { height: 1; }
    #chart-label { width: 8; text-style: bold; }
    #chart-mode { width: 24; }
    #chart-mode SelectOverlay { border: solid $accent; padding: 0 1; }
    #chart-hint { width: 1fr; color: $text-muted; }
    #charts { height: auto; }
    #chart-note { height: auto; color: $text-muted; }
    #procs { height: 1fr; min-height: 4; }
    #inspector { height: auto; max-height: 30%; overflow-y: auto; border: round $secondary; padding: 0 1; }
    """
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("s", "cycle_sort", "Sort"),
        Binding("c", "cycle_chart", "Chart"),
        Binding("a", "toggle_all", "All/Top"),
        Binding("m", "toggle_monitoring", "Slim/Full"),
        Binding("p", "pause", "Pause"),
        Binding("question_mark", "help", "Explain"),
    ]
    SORTS = [("VRAM", lambda r: -(r.gpu.shown_vram or 0)), ("GPU%", lambda r: -r.gpu.gpu_pct),
             ("spill", lambda r: -r.gpu.shared), ("paged", lambda r: -r.gpu.paged_out),
             ("name", lambda r: r.name.lower())]
    SYSTEM_SORTS = [("RAM", lambda r: (r.ram is None, -(r.ram or 0))),  # unknown last; a measured 0 is not unknown
                    ("CPU%", lambda r: (r.cpu is None, -(r.cpu or 0)))]
    CHART_COLORS = {
        "GPU": "#4ade80",
        "SM clock": "#fb923c",
        "VRAM": "#60a5fa",
        "Mem activity": "#a78bfa",
        "Temperature": "#ef4444",
        "Power": "#facc15",
        "Encoder": "#e2e8f0",
        "Decoder": "#22d3ee",
    }
    CHARTS = tuple(CHART_COLORS)

    def __init__(self, gpu_index: int | None = None, interval: float = 1.0, host: bool = True,
                 system: bool = True) -> None:
        super().__init__()
        self.system = system
        self.sorts = self.SORTS + self.SYSTEM_SORTS if system else self.SORTS
        self.host_sample: HostSample | None = None
        self.host_error: str | None = None
        self._sampler: HostSampler | None = None
        self.hist_cpu: deque = deque(maxlen=60)
        self.hist_ram: deque = deque(maxlen=60)
        self.gpu_index = gpu_index
        self.interval = interval
        self.host = host
        self.snap: Snapshot | None = None
        self.snapshots: list[Snapshot] = []
        self._histories: dict[int, dict[str, deque]] = {}
        self._row_snapshots: dict[str, Snapshot] = {}
        self.selected_key: str | None = None  # ProcRow.key: Linux and Windows host PIDs can collide
        self.sort_idx = 0
        self.show_all = False
        self.paused = False
        self.monitoring = False
        self.hist_util: deque = deque(maxlen=60)
        self.hist_clock: deque = deque(maxlen=60)
        self.hist_bw: deque = deque(maxlen=60)
        self.hist_vram: deque = deque(maxlen=60)
        self.hist_temp: deque = deque(maxlen=60)
        self.hist_power: deque = deque(maxlen=60)
        self.hist_enc: deque = deque(maxlen=60)
        self.hist_dec: deque = deque(maxlen=60)
        self.chart_idx = -1
        self._auto_selected = False

    @property
    def selected_pid(self) -> int | None:
        """PID of the selected local process; None for Windows (W:) and outside (H:) rows."""
        key = self.selected_key
        pid = key.rsplit(":", 1)[-1] if key else ""
        return int(pid) if pid.isdigit() else None

    @selected_pid.setter
    def selected_pid(self, pid: int | None) -> None:
        self.selected_key = None if pid is None else str(pid)

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="dashboard"):
            yield Static("Sampling GPU counters...", id="device")
            if self.system:
                yield Static("Sampling RAM and CPU...", id="system")
            with Vertical(id="history"):
                with Horizontal(id="chart-controls"):
                    yield Static("Charts", id="chart-label")
                    yield Select(
                        [("All charts", -1), *((self.CHARTS[index], index) for index in self.available_charts)],
                        value=-1, allow_blank=False, compact=True, id="chart-mode",
                    )
                    yield Static("Click to choose · c cycles", id="chart-hint")
                yield Static(id="charts")
                yield Static(id="chart-note")
        table = DataTable(id="procs", cursor_type="row", zebra_stripes=True)
        yield table
        yield Static("", id="inspector")
        yield Footer()

    def on_mount(self) -> None:
        t = self.query_one("#procs", DataTable)
        cols = ["PID", "Process", "Kind", "Runtime", "VRAM", "Spill", "Paged", "GPU%", "3D", "Copy", "Video"]
        if self.system:
            cols += ["RAM", "CPU%"]
        for col in (*cols, "What's inside"):
            t.add_column(col, key=col)
        t.focus()
        self.poll()

    @work(thread=True, exclusive=True)
    def poll(self) -> None:
        worker = get_current_worker()
        self._sample_host()  # the first reading only primes the CPU rates
        if BACKEND == "wddm":
            live = LiveCollectors(self.gpu_index, collector_factory=Collector, discover=enumerate_adapters)
            try:
                worker.cancelled_event.wait(0.5)
                while not worker.is_cancelled:
                    if not self.paused:
                        snapshots = live.sample()
                        host = self._sample_host()
                        if not worker.is_cancelled:
                            self.call_from_thread(self.apply_all, snapshots, host)
                    worker.cancelled_event.wait(self.interval)
            finally:
                live.close()
            return
        indices = ([self.gpu_index] if self.gpu_index is not None else
                   list(range(len(ordered_adapters(enumerate_adapters())))) or [0])
        with ExitStack() as stack:
            collectors = []
            for index in indices:
                collector = Collector(index, scan_budget=Collector.SCAN_BUDGET / len(indices),
                                      host=self.host and not (BACKEND == "wsl" and len(indices) > 1))
                if BACKEND == "wsl" and len(indices) > 1 and self.host:
                    collector.host_status = "off (multi-GPU WSL host matching is not validated; use --gpu N)"
                stack.callback(collector.close)
                collectors.append(collector)
            worker.cancelled_event.wait(0.5)
            while not worker.is_cancelled:
                if not self.paused:
                    snapshots = [collector.sample() for collector in collectors]
                    host = self._sample_host()
                    if not worker.is_cancelled:
                        self.call_from_thread(self.apply_all, snapshots, host)
                worker.cancelled_event.wait(self.interval)

    def _sample_host(self) -> HostSample | None:
        """The RAM/CPU panel is optional: a failed OS query shows its reason and never stops GPU sampling."""
        if not self.system:
            return None
        try:
            if self._sampler is None:
                self._sampler = HostSampler()
            host = self._sampler.sample()
        except Exception as error:
            self.host_error = f"{type(error).__name__}: {error}"
            return None
        self.host_error = None
        return host

    # ------------------------------------------------------------------ rendering

    def apply(self, snap: Snapshot) -> None:
        self.apply_all([snap])

    def apply_all(self, snapshots: list[Snapshot], host: HostSample | None = None) -> None:
        if not self.is_running:
            return  # A worker may finish sampling after shutdown has removed the widgets.
        self.host_sample = host  # None after a failed sample: a stale reading is not shown as current
        self.hist_cpu.append(host.cpu_pct if host else None)
        self.hist_ram.append(100 * host.ram_used / host.ram_total if host and host.ram_total else None)
        for snap in snapshots:
            for row in snap.procs:  # Windows (W:) / outside (H:) rows have no local pid: RAM/CPU stay unknown
                hp = host.proc(row.pid, row.detail.create_time) if host and not row.host else None
                row.ram, row.cpu = (hp.rss, hp.cpu_pct) if hp else (None, None)
        multi = len(snapshots) > 1
        if self.snapshots and multi != (len(self.snapshots) > 1) and self.selected_key is not None:
            if multi:
                self.selected_key = f"0:{self.selected_key}"
            else:
                self.selected_key = self.selected_key[2:] if self.selected_key.startswith("0:") else None
        self.snapshots = snapshots
        table = self.query_one("#procs", DataTable)
        if multi and "GPU" not in table.columns:
            table.add_column("GPU", key="GPU")
        elif not multi and "GPU" in table.columns:
            table.remove_column("GPU")
        self._row_snapshots = {}
        for index, snap in enumerate(snapshots):
            for row in snap.procs:
                row.gpu_index = index if multi else None
                self._row_snapshots[row.key] = snap
            self._use_history(index)
            self._record_sample(snap)
        self._use_history(0)
        self.snap = snapshots[0]
        if not self._auto_selected:
            llm = next((r for r in sorted(
                (r for s in snapshots for r in s.procs), key=self.SORTS[0][1]) if r.llm), None)
            if llm:
                self.selected_key = llm.key
            self._auto_selected = True
        self.render_dashboard()
        self.render_table(self.snap)
        self.render_inspector()

    def _use_history(self, index: int) -> None:
        names = ("util", "clock", "bw", "vram", "temp", "power", "enc", "dec")
        if index not in self._histories:
            self._histories[index] = {name: deque(maxlen=60) for name in names}
        for name, history in self._histories[index].items():
            setattr(self, f"hist_{name}", history)

    def _record_sample(self, snap: Snapshot) -> None:
        self.snap = snap
        nv = snap.nv
        amd = bool(snap.adapter and snap.adapter.vendor_id == 0x1002)
        if nv:
            self.hist_util.append(nv.util_gpu)
            self.hist_clock.append(nv.gfx_clock if amd else nv.sm_clock)
            self.hist_bw.append(nv.util_mem)
            self.hist_temp.append(nv.temp)
            self.hist_power.append(nv.power_w)
            self.hist_enc.append(nv.enc)
            self.hist_dec.append(nv.dec)
        else:
            for history in (self.hist_util, self.hist_clock, self.hist_bw, self.hist_temp,
                            self.hist_power, self.hist_enc, self.hist_dec):
                history.append(None)
        self.hist_vram.append(100 * snap.vram_used / snap.vram_total
                              if snap.vram_total and snap.vram_used is not None else None)

    def render_dashboard(self) -> None:
        amd = bool(self.snapshots[0].adapter and self.snapshots[0].adapter.vendor_id == 0x1002)
        clock_label = "SM/GFX clock" if len(self.snapshots) > 1 else "GFX clock" if amd else "SM clock"
        if getattr(self, "_clock_label", None) != clock_label:
            self._clock_label = clock_label
            selector = self.query_one("#chart-mode", Select)
            selected = selector.value
            selector.set_options([("All charts", -1), *(
                (clock_label if self.CHARTS[index] == "SM clock" else self.CHARTS[index], index)
                for index in self.available_charts)])
            selector.value = selected
        self._amd_chart_labels = amd
        devices = []
        for index, snap in enumerate(self.snapshots):
            if len(self.snapshots) > 1:
                devices.append(Text(f"GPU {index}", style="bold"))
            devices.append(self.render_device(snap))
        if self.system:
            system = self.query_one("#system", Static)
            system.border_title = "System (WSL VM, OS-reported)" if BACKEND == "wsl" else "System (OS-reported)"
            if self.host_sample:
                system.update(self.render_system(self.host_sample))
            elif self.host_error:
                system.update(Text(f"RAM/CPU unavailable: {self.host_error}", style="yellow"))
        device = self.query_one("#device", Static)
        device.border_title = "Slim monitoring" if self.monitoring else "Full diagnostics"
        device.update(
            Group(*devices) if len(self.snapshots) > 1 else devices[0])
        self.render_charts()

    def render_system(self, h: HostSample) -> Group:
        """System RAM and CPU as the OS reports them: system memory, never GPU bytes and never in a GPU total."""
        min_bar = 8
        room = self.size.width - 12  # what a line has left once the label, panel border, padding and scrollbar are counted

        def line(text: str = "") -> Text:
            return Text(text, no_wrap=True, overflow="ellipsis")  # never wraps onto a second row

        ram_pct = 100 * h.ram_used / h.ram_total if h.ram_total else 0
        ram_tail = line()
        ram_tail.append(f" {gb(h.ram_used)} / {gb(h.ram_total)}", style="bold")
        ram_tail.append(f"  {ram_pct:.0f}%", style="grey62")
        optional = [(f"  free {gb(h.ram_available)}", "grey62")]
        if h.swap_total:
            optional.append((f"  {'pagefile' if os.name == 'nt' else 'swap'} {gb(h.swap_used)}/{gb(h.swap_total)}",
                             "yellow" if h.swap_used > h.swap_total / 4 else "grey62"))
        for extra, style in optional:  # detail is dropped, not wrapped, in a narrow window
            if ram_tail.cell_len + len(extra) + min_bar <= room:
                ram_tail.append(extra, style=style)
        cpu_tail = line()
        cpu_tail.append(f" {h.cpu_pct:4.0f}%", style="bold")
        threads = f"  {h.logical} threads" + (f"/{h.physical} cores" if h.physical else "")
        if not self.monitoring and cpu_tail.cell_len + len(threads) + min_bar <= room:
            cpu_tail.append(threads, style="grey62")
        # Both bars share one width, sized so the longer line still fits on a single row.
        width = max(min_bar, min(60, room - max(ram_tail.cell_len, cpu_tail.cell_len)))
        ram = line("RAM  ")
        ram.append_text(stacked_bar([(h.ram_used, "dodger_blue2")], h.ram_total, width))
        ram.append_text(ram_tail)
        cpu = line("CPU  ")
        cpu.append_text(stacked_bar([(h.cpu_pct, "green3")], 100, width))
        cpu.append_text(cpu_tail)
        if self.monitoring:
            return Group(ram, cpu)
        cores = line("     ")
        for v in h.per_core:
            cores.append(SPARK[min(int(v / 100 * (len(SPARK) - 1) + 0.5), len(SPARK) - 1)],
                         style="red" if v >= 90 else "yellow" if v >= 60 else "green3")
        cores.append("  per-thread", style="grey50")
        history = max(8, min(30, (room - 18) // 2))  # each sparkline's width: two of them plus their labels share a row
        hist = line("     ")
        hist.append(spark(self.hist_cpu, history), style="green3")
        hist.append(" CPU 60s   ", style="grey50")
        hist.append(spark(self.hist_ram, history), style="dodger_blue2")
        hist.append(" RAM 60s", style="grey50")
        top_ram = line()
        top_ram.append("     top RAM: ", style="grey62")
        top_ram.append("  ".join(f"{p.name} {gb(p.rss)}" for p in h.top("rss", 5)))
        top_cpu = line()
        top_cpu.append("     top CPU: ", style="grey62")
        rated = h.top("cpu_pct", 5)
        busy = [p for p in rated if p.cpu_pct >= 0.5]
        # "…" until some process has a CPU reading: no reading yet is not the same as idle
        top_cpu.append("  ".join(f"{p.name} {p.cpu_pct:.0f}%" for p in busy) if busy else "idle" if rated else "…")
        return Group(ram, cpu, cores, hist, top_ram, top_cpu)

    def render_device(self, s: Snapshot) -> Group:
        nv = s.nv
        name = s.adapter.name if s.adapter else (nv.name if nv else "GPU")
        if self.monitoring:
            head = Text(name, style="bold")
        elif s.backend == "metal":
            head = Text.assemble((name, "bold"), f"   {s.driver}  ", ("Metal", "bold yellow"), "  unified memory")
        else:
            info = (f"   driver {s.driver}" if s.driver else "") + (f"  CUDA {s.cuda}" if s.cuda else "")
            head = Text.assemble((name, "bold"), info + "  ",
                                 (nv.driver_model if nv else "", "bold yellow"), f"  {nv.pci_bus if nv else ''}")
        if self.paused:
            head.append("   [PAUSED]", style="bold red")

        # Fit category proportions to the adapter total; row samples need not sum exactly.
        by_kind: dict[str, float] = {}
        for r in s.procs:
            if r.gpu.attributed:  # unknown shares stay in "unattributed" instead of showing as 0
                by_kind[r.kind] = by_kind.get(r.kind, 0) + r.gpu.dedicated
        attributed = sum(by_kind.values())
        scale = min(1.0, s.vram_used / attributed) if attributed and s.vram_used is not None else 1.0
        order = sorted(by_kind.items(), key=lambda kv: -kv[1])
        parts = [(v * scale, KIND_STYLE.get(k, "white")) for k, v in order] if s.vram_used is not None else []
        other = max(s.vram_used - attributed * scale, 0) if s.vram_used is not None else 0
        parts.append((other, "grey35"))
        width = max(min(self.size.width - 40, 90), 20)
        vram = Text("GPU mem " if s.backend == "metal" else "VRAM ")
        vram.append_text(stacked_bar(parts, s.vram_total, width))
        used_label = gb(s.vram_used) if s.vram_used is not None else "N/A"
        vram.append(f" {used_label} / {vram_total_label(s)}", style="bold")
        legend = Text("     ")
        for k, v in order[:7]:
            legend.append("■ ", style=KIND_STYLE.get(k, "white"))
            legend.append(f"{k} {gb(v)}  ")
        if other > 32 << 20:
            legend.append("■ ", style="grey35")
            legend.append(f"unattributed {gb(other)}  ")
        if s.vram_used is not None and s.vram_total and not intel_memory_note(s):
            legend.append(f"free {gb(max(s.vram_total - s.vram_used, 0))}", style="grey62")
        if s.shared_used is not None and s.shared_used > 64 << 20:
            legend.append(f"   shared(RAM) {gb(s.shared_used)}", style="yellow")
        if scale < 1.0:
            legend.append("   [bar normalized; labels are raw]", style="grey62")
        if s.backend == "wsl":
            legend.append("\n     " + wsl_note(s), style="grey62")
        if s.backend == "metal":
            legend.append("\n     " + METAL_NOTE, style="grey62")
        if intel_memory_note(s):
            legend.append("\n     " + intel_memory_note(s), style="grey62")
        if s.counter_status:
            legend.append("\n     " + s.counter_status, style="grey62")
        if s.process_status:
            legend.append(f"\n     DRM process scan: {s.process_status}.", style="grey62")

        lines = [head, vram, legend]
        if nv and s.backend == "metal":
            lines.append(Text(f"GPU {format_metric(nv.util_gpu, unit='%')}",
                              style=f"bold {self.CHART_COLORS['GPU']}"))
        elif nv:
            rx = nv.pcie_rx_kbs / 1024 if nv.pcie_rx_kbs is not None else None
            tx = nv.pcie_tx_kbs / 1024 if nv.pcie_tx_kbs is not None else None
            pstate = f"P{nv.pstate}" if nv.pstate is not None else "P N/A"
            is_amd = nv.source.startswith("AMD ")
            is_intel = nv.source == "Intel sysfs"
            m = Text()
            label = "AMD gfx" if is_amd else ("Intel GT" if is_intel else "GPU")
            m.append(f"{label} {format_metric(nv.util_gpu, unit='%')}", style=f"bold {self.CHART_COLORS['GPU']}")
            m.append(f"  Mem activity {format_metric(nv.util_mem, unit='%')}", style=self.CHART_COLORS["Mem activity"])
            m.append(f"  PCIe↓{format_metric(rx, '.0f'):>5} ↑{format_metric(tx, '.0f'):>4} MB/s")
            if nv.enc or nv.dec:
                m.append(f"  enc {format_metric(nv.enc, unit='%')} dec {format_metric(nv.dec, unit='%')}")
            if nv.video is not None:
                m.append(f"  video {format_metric(nv.video, unit='%')}")
            lines.append(m)
            clock_label = (f"gfx {format_metric(nv.gfx_clock)} MHz" if is_amd else
                           f"GT {format_metric(nv.sm_clock)}/{format_metric(nv.sm_clock_max)} MHz" if is_intel else
                           f"SM {format_metric(nv.sm_clock)}/{format_metric(nv.sm_clock_max)} MHz")
            power_kind = f" {nv.power_kind}" if nv.power_kind else ""
            temp_kind = f" ({nv.temp_kind})" if nv.temp_kind else ""
            if self.monitoring:
                clock = nv.gfx_clock if is_amd else nv.sm_clock
                label = "gfx" if is_amd else ("GT" if is_intel else "SM")
                st = Text(f"{format_metric(nv.temp, unit='°C')}{temp_kind}  "
                          f"{format_metric(nv.power_w, '.0f')} W{power_kind}  "
                          f"{label} {format_metric(clock)} MHz", style="grey70")
            else:
                st = Text(f"{format_metric(nv.temp, unit='°C')}{temp_kind}  {format_metric(nv.power_w, '.0f')}/"
                          f"{format_metric(nv.power_limit_w, '.0f')} W{power_kind}  {pstate}  "
                          f"{clock_label}  "
                          f"mem {format_metric(nv.mem_clock)} MHz  video {format_metric(nv.video_clock)} MHz  "
                          f"PCIe gen{format_metric(nv.pcie_gen)} x{format_metric(nv.pcie_width)}  "
                          + (f"CC {nv.compute_cap or 'N/A'}" if nv.source == "NVML" else nv.source), style="grey70")
            if nv.throttle:
                st.append("  throttle: " + ",".join(nv.throttle), style="yellow")
            lines.append(st)
        return Group(*lines)

    def chart_series(self, index: int, nv: NvDevice | None) -> tuple[deque, float, str, str]:
        chart = self.CHARTS[index]
        color = self.CHART_COLORS[chart]
        if chart == "GPU":
            return self.hist_util, 100, "%", color
        if chart == "SM clock":
            values = [value for value in self.hist_clock if value is not None]
            scale = max((nv.sm_clock_max if nv else None) or 0, max(values, default=0), 1)
            return self.hist_clock, scale, " MHz", color
        if chart == "VRAM":
            return self.hist_vram, 100, "%", color
        if chart == "Mem activity":
            return self.hist_bw, 100, "%", color
        if chart == "Temperature":
            return self.hist_temp, 120, "°C", color
        if chart == "Power":
            values = [value for value in self.hist_power if value is not None]
            scale = max((nv.power_limit_w if nv else None) or 0, max(values, default=0), 1)
            return self.hist_power, scale, " W", color
        if chart == "Encoder":
            return self.hist_enc, 100, "%", color
        return self.hist_dec, 100, "%", color

    def render_charts(self) -> None:
        if len(self.snapshots) > 1:
            charts = []
            notes = []
            for index, snap in enumerate(self.snapshots):
                self.snap = snap
                self._use_history(index)
                self._amd_chart_labels = bool(snap.adapter and snap.adapter.vendor_id == 0x1002)
                grid, note = self._chart_content()
                charts.extend([Text(f"GPU {index}", style="bold"), grid])
                notes.append(f"GPU {index}: {note}")
            self.query_one("#charts", Static).update(Group(*charts))
            self.query_one("#chart-note", Static).update("\n".join(notes))
            self.snap = self.snapshots[0]
            self._use_history(0)
            self._amd_chart_labels = bool(self.snap.adapter and self.snap.adapter.vendor_id == 0x1002)
            return
        if self.snap is not None:
            grid, note = self._chart_content()
            self.query_one("#charts", Static).update(grid)
            self.query_one("#chart-note", Static).update(note)

    def _chart_content(self) -> tuple[Table, str]:
        # Reserve the border, padding, and scrollbar even before child resize events arrive.
        width = max(1, self.size.width - 6)
        indices = list(self.available_charts) if self.chart_idx == -1 else [self.chart_idx]
        columns = min(len(indices), 3, max(1, (width + 2) // 46))
        cell_width = max(1, (width - 2 * (columns - 1)) // columns)
        grid = Table.grid(padding=(0, 2), pad_edge=False, expand=True)
        for _ in range(columns):
            grid.add_column(width=cell_width, no_wrap=True, overflow="crop")
        cells = []
        for index in indices:
            values, scale, unit, style = self.chart_series(index, self.snap.nv)
            current = values[-1] if values else None
            name = self.CHARTS[index]
            if getattr(self, "_amd_chart_labels", False) and name == "SM clock":
                name = "GFX clock"
            cell = Text(f"{name:<12} ", style=style)
            cell.append(f"{format_metric(current, '.0f', unit):>8} ", style="bold")
            cell.append(spark(values, max(1, cell_width - 22), scale))
            cells.append(cell)
        for start in range(0, len(cells), columns):
            grid.add_row(*cells[start:start + columns])
        _, power_scale, _, _ = self.chart_series(self.CHARTS.index("Power"), self.snap.nv)
        _, clock_scale, _, _ = self.chart_series(self.CHARTS.index("SM clock"), self.snap.nv)
        note = (
            "Recent samples · blank = no reading" if self.snap.backend == "metal" else
            f"Recent samples · blank = no reading · power 0–{power_scale:.0f} W · "
            f"{'GFX' if getattr(self, '_amd_chart_labels', False) else 'SM'} clock 0–{clock_scale:.0f} MHz"
        )
        return grid, note

    def on_resize(self) -> None:
        if self.is_mounted and self.snap is not None:
            self.call_after_refresh(self.render_dashboard)

    def visible_rows(self, s: Snapshot) -> list[ProcRow]:
        rows = s.procs
        if not self.show_all:
            rows = [r for r in rows if not r.gpu.attributed or r.gpu.dedicated >= 32 << 20 or r.gpu.gpu_pct >= 1 or r.gpu.shared >= 64 << 20
                    or r.gpu.paged_out >= 128 << 20 or r.llm]
        rows = sorted(rows, key=self.sorts[self.sort_idx][1])
        vm = next((i for i, r in enumerate(rows) if r.kind == "wsl-vm"), None)
        if vm is not None and self.sorts[self.sort_idx][0] == "VRAM":
            # Linux rows share the WSL VM row's total: keep them under it, ordered by their inferred sizes.
            linux = [r for r in rows if not r.gpu.attributed and not r.host]
            rest = [r for r in rows if r not in linux]
            vm = rest.index(rows[vm])
            rows = rest[:vm + 1] + linux + rest[vm + 1:]
        return rows

    def render_table(self, s: Snapshot) -> None:
        t = self.query_one("#procs", DataTable)
        rows = [r for snap in self.snapshots for r in self.visible_rows(snap)] if len(self.snapshots) > 1 else self.visible_rows(s)
        if len(self.snapshots) > 1:
            rows = [r for r in rows if not (
                self._row_snapshots[r.key].backend == "wsl" and not r.host and r.gpu_index != 0)]
        if len(self.snapshots) > 1 and not any(snap.backend == "wsl" for snap in self.snapshots):
            rows.sort(key=self.sorts[self.sort_idx][1])
        keep = self.selected_key
        t.clear()

        def pct(v: float | None) -> Text:
            if v is None:
                return Text("n/a", justify="right", style="grey62")
            return Text(f"{v:.0f}" if v >= 0.5 else "", justify="right", style="bold" if v >= 20 else "")

        for r in rows:
            s = self._row_snapshots.get(r.key, s)
            e = r.gpu.engines
            wddm = has_wddm_counters(s, r)
            engines = has_engine_counters(s, r)
            split = engines and s.backend != "metal"  # Metal reports one GPU time, no per-engine split
            spill = Text(gb(r.gpu.shared) if has_spill_counters(s, r) and r.gpu.shared >= 1 << 20 else "", justify="right",
                         style="bold red" if r.gpu.shared >= 256 << 20 and r.kind in ("LLM", "AI") and s.shared_is_spill
                         else "yellow")
            paged = Text(gb(r.gpu.paged_out) if wddm and r.gpu.paged_out >= 1 << 20 else "", justify="right", style="grey62")
            if (wddm or (s.backend == "drm" and not r.host)) and "shared" not in r.gpu.available:
                spill = Text("n/a", justify="right", style="grey62")
            if wddm and not r.gpu.paged_available:
                paged = Text("n/a", justify="right", style="grey62")
            system_cells = []
            if self.system:
                known = r.ram is not None  # "…": readable, CPU% needs one more sample; "n/a": the OS wouldn't say
                system_cells = [Text(gb(r.ram) if known else "n/a", justify="right", style="" if known else "grey62"),
                                Text("…", justify="right", style="grey62") if known and r.cpu is None else pct(r.cpu)]
            t.add_row(
                Text(r.pid_label, justify="right", style="grey62" if r.host else ""),
                Text(r.name),
                Text(r.kind, style=KIND_STYLE.get(r.kind, "")),
                Text(" ".join(r.detail.runtimes[:3]), style="grey70"),
                Text(vram_label(r.gpu), justify="right", style="bold" if r.gpu.attributed else "grey62"), spill, paged,
                pct(r.gpu.gpu_pct if engines else None),
                pct(e.get("3d", 0) + e.get("compute", 0) if split else None),
                pct(e.get("copy", 0) if split else None),
                pct(e.get("decode", 0) + e.get("encode", 0) + e.get("video", 0) if split else None),
                *system_cells,
                Text(r.inside, style="magenta" if r.llm else "grey62", overflow="ellipsis", no_wrap=True),
                *([Text("?" if s.backend == "wsl" and not r.host else str(r.gpu_index))]
                  if len(self.snapshots) > 1 else []),
                key=r.key,
            )
        if keep is not None:
            try:
                t.move_cursor(row=t.get_row_index(keep), animate=False, scroll=False)
            except Exception:
                pass
        t.border_subtitle = f"sort: {self.sorts[self.sort_idx][0]}  ·  {'all' if self.show_all else 'top'} processes"
        if s.backend == "wsl" and not any(not r.host for r in s.procs):
            t.border_subtitle = WSL_EMPTY

    def render_inspector(self) -> None:
        # A queued RowHighlighted can arrive while the screen is being torn down, after the
        # inspector is gone. Nothing is left to render then, so ignore it instead of raising.
        panel = next(iter(self.query("#inspector")), None)
        if panel is None:
            return
        s = self._row_snapshots.get(self.selected_key, self.snap)
        r = next((x for x in s.procs if x.key == self.selected_key), None) if s else None
        if r is None:
            panel.update(Text("Select a process (↑/↓) to see what's inside its GPU memory.  Press ? to learn what's knowable.", style="grey62"))
            return
        where = {"windows": "Windows PID ", "outside": "host PID "}.get(r.origin, "") if r.host else ""
        panel.border_title = f"{r.name} ({where}{r.pid}) - {r.kind}"
        if r.gpu_index is not None:
            panel.border_title += " - GPU unknown" if s.backend == "wsl" and not r.host else f" - GPU {r.gpu_index}"
        out: list = []
        e = r.gpu.engines
        wddm = has_wddm_counters(s, r)
        engines = " ".join(f"{k} {v:.0f}%" for k, v in e.items() if v >= 0.5) or "idle"
        hdr = Text.assemble(("VRAM ", "bold"), (vram_label(r.gpu), "bold"), "   spill(RAM) ",
                            (gb(r.gpu.shared) if has_spill_counters(s, r) else "n/a",
                             "bold red" if r.gpu.shared > 256 << 20 and s.shared_is_spill else ""),
                            "   paged(est) ", gb(r.gpu.paged_out) if wddm and r.gpu.paged_available else "n/a",
                            "   engines ", engines if has_engine_counters(s, r) else "n/a")
        if r.detail.runtimes:
            hdr.append("   runtime ", style="grey62")
            hdr.append(" ".join(r.detail.runtimes), style="cyan")
        out.append(hdr)
        if not r.host and r.gpu.attributed and r.gpu.inferred_by:
            out.append(Text(inferred_note(r.gpu), style="grey62"))
        if not r.host and not r.gpu.attributed:
            if r.gpu.inferred is not None:
                out.append(Text(inferred_note(r.gpu), style="grey62"))
            elif s.wsl_split:
                out.append(Text(f"No per-process VRAM: {s.wsl_split}.", style="grey62"))
            elif s.backend == "drm":
                out.append(Text(drm_note(r.gpu), style="grey62"))
            elif s.backend == "metal":
                out.append(Text("GPU memory n/a: footprint can't read other users' processes without root "
                                "(or hasn't measured this one yet). GPU% comes from Metal's per-process GPU time.",
                                style="grey62"))
        if s.backend == "drm" and not r.host and (r.gpu.resident_vram is not None or r.gpu.resident_gtt is not None):
            def size(v): return gb(v) if v is not None else "n/a"
            out.append(Text(f"Resident VRAM {size(r.gpu.resident_vram)}, GTT {size(r.gpu.resident_gtt)}. "
                            + DRM_RESIDENT_NOTE, style="grey62"))
        if s.backend == "drm" and r.gpu.kfd:
            out.append(Text("Engines n/a: this process uses ROCm/HIP (/dev/kfd), whose compute queues the kernel doesn't "
                            "time per process. The device's AMD gfx load includes its work.", style="grey62"))
        if wddm and r.gpu.paged_out >= 64 << 20:
            out.append(Text(f"{gb(r.gpu.paged_out)} of {gb(r.gpu.committed)} committed is not accounted for by local/non-local "
                            "usage. Estimated non-residency, not proof of eviction or disk paging.",
                            style="yellow"))

        if r.handy:
            if r.handy.selected_model:
                out.append(Text(f"Handy selected (settings; not load proof): {r.handy.selected_model}", style="grey70"))
            if r.handy.last_event:
                out.append(Text(f"Handy last event: {r.handy.last_event} {r.handy.last_model or ''} "
                                f"at {r.handy.last_event_local} (runtime-reported history, not current GPU residency)",
                                style="green"))

        v = r.llm
        if v:
            if r.gpu_index is not None:
                out.append(Text("Model files, launch flags and runtime buffer logs can span GPUs. "
                                "The breakdown below is not a measured per-device model placement.", style="grey70"))
            width = max(min(self.size.width - 8, 100), 20)
            total = max(r.gpu.shown_vram or 0, sum(x.bytes for x in v.segments), 1)
            out.append(stacked_bar([(x.bytes, SEG_STYLE[x.kind]) for x in v.segments], total, width))
            legend = Text()
            for x in v.segments:
                legend.append("■ ", style=SEG_STYLE[x.kind])
                legend.append(f"{x.label} {gb(x.bytes)}", style="bold" if x.exact else "")
                legend.append("  ")
            if v.log_source:
                legend.append(f"(allocated sizes from {os.path.basename(v.log_source)})", style="green")
            out.append(legend)
            if v.models:
                mt = Table(box=None, padding=(0, 1), show_edge=False, header_style="bold grey70")
                for col in ("", "model", "arch", "params", "quant", "layers", "train ctx", "on GPU (est)", "evidence", "file"):
                    mt.add_column(col)
                for i, g in enumerate(v.models):
                    f = v.files[i] if i < len(v.files) else None
                    mt.add_row(Text(v.kind_label(i), style="magenta"), v.label(i), g.arch, g.params_label, g.quant,
                               str(g.n_layers or ""), f"{g.ctx_train:,}" if g.ctx_train else "",
                               gb(g.gpu_weight_bytes()), Text(f.evidence if f else "", style="green"),
                               Text(os.path.basename(g.path), style="grey50", overflow="ellipsis"))
                out.append(mt)
            for f in v.other_files:
                kind = f", {f.kind_label}" if f.kind_label else ""
                out.append(Text(f"file: {f.store_name or f.path}  ({gb(f.size)}, {f.evidence}{kind})", style="grey62"))
            if v.server:
                sv = v.server
                st = Text.assemble(("server ", "bold"), (f"{sv.kind} {sv.url}", "cyan"))
                if v.server_source == "parent":
                    st.append(" (parent process; API report, not worker residency)")
                if sv.n_ctx:
                    st.append(f"  ctx {sv.n_ctx:,}")
                if sv.slots_total is not None:
                    st.append(f"  slots busy {sv.slots_busy}/{sv.slots_total}")
                if sv.gen_tok_s:
                    st.append(f"  gen {sv.gen_tok_s:.1f} tok/s", style="bold green")
                if sv.prompt_tok_s:
                    st.append(f"  prompt {sv.prompt_tok_s:.0f} tok/s")
                for m in sv.models[:4]:
                    if sv.kind == "lmstudio":
                        st.append(f"\n  loaded: {m['name']} {m.get('type') or ''} {m.get('quant') or ''} ctx {m.get('ctx') or '?'}")
                    elif sv.kind == "vllm":
                        st.append(f"\n  API-reported model ID: {m['name']}")
                out.append(st)
            for n in v.notes:
                out.append(Text("• " + n, style="bold red" if n.startswith("!") else "grey70"))
        else:
            out.append(Text(EXPLAIN.get(r.kind, ""), style="grey70"))
            for m in r.detail.models:
                kind = f", {m.kind_label}" if m.kind_label else ""
                out.append(Text(f"file: {m.store_name or m.path}  ({gb(m.size)}, {m.evidence}{kind})", style="magenta"))
            if r.detail.candidates:
                out.append(Text("No mapped model file was observed. Some runtimes read then close model files. "
                                "Installed candidates (not proven loaded):", style="grey70"))
                for m in r.detail.candidates:
                    kind = f", {m.kind_label}" if m.kind_label else ""
                    out.append(Text(f"  {m.store_name or m.path}  ({gb(m.size)}{kind})", style="grey62"))
        if r.detail.ports:
            out.append(Text("listening: " + ", ".join(map(str, r.detail.ports[:8])), style="grey50"))
        if r.detail.cmdline:
            out.append(Text(" ".join(redact_cmdline(r.detail.cmdline))[:400], style="grey42", overflow="ellipsis"))
        elif r.detail.access_denied:
            out.append(Text("(protected process - run elevated for runtime/file details)", style="grey42"))
        panel.update(Group(*out))

    # ------------------------------------------------------------------ events/actions

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if self.is_running and event.row_key is not None and event.row_key.value is not None:
            self.selected_key = str(event.row_key.value)
            self.render_inspector()

    def action_cycle_sort(self) -> None:
        self.sort_idx = (self.sort_idx + 1) % len(self.sorts)
        if self.snap:
            self.render_table(self.snap)

    @property
    def available_charts(self) -> list[int]:
        """Chart indices this backend has data for: IOKit (macOS) has no clocks, power, temperature or codecs."""
        if BACKEND == "metal":
            return [i for i, name in enumerate(self.CHARTS) if name in METAL_CHARTS]
        return list(range(len(self.CHARTS)))

    def action_cycle_chart(self) -> None:
        order = [-1, *self.available_charts]
        nxt = order[(order.index(self.chart_idx) + 1) % len(order)] if self.chart_idx in order else -1
        self.query_one("#chart-mode", Select).value = nxt

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "chart-mode" and isinstance(event.value, int):
            self.chart_idx = event.value
            self.render_charts()

    def action_toggle_all(self) -> None:
        self.show_all = not self.show_all
        if self.snap:
            self.render_table(self.snap)

    def action_pause(self) -> None:
        self.paused = not self.paused
        if self.snap:
            self.render_dashboard()

    def action_toggle_monitoring(self) -> None:
        self.monitoring = not self.monitoring
        if self.snap:
            self.render_dashboard()

    def action_help(self) -> None:
        self.push_screen(HelpScreen())
