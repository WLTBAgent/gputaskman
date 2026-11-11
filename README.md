# gputaskman

**An `htop` for what's inside your GPU, not just how full it is.**

Windows + NVIDIA/AMD, with Linux and WSL support (NVIDIA everywhere, AMD and Intel on native Linux; see
[Linux and WSL](#linux-and-wsl)) and macOS on Apple Silicon (see [macOS](#macos-apple-silicon)). A terminal UI
that connects per-process GPU memory and engine activity with loaded runtimes, model files, and
LLM buffer breakdowns.

Task Manager says "15 GB used." This tries to answer: *by what? Model weights? KV cache?
Your browser? The desktop compositor? And why is so much GPU memory in system RAM?*

**Scope: local GPU workloads, not cloud services.** Decision models, vision, speech, LLM inference,
training/fine-tuning (including Unsloth), graphics, and video all belong here. Visibility varies:
process memory and engine activity are measured; identifying individual models and their work
requires supporting evidence. A PyTorch/CUDA DLL alone cannot identify a model or prove training,
and a mapped file does not prove that model is resident on the GPU or being exercised.
CPU-only, NPU-only, and cloud workloads are out of scope.

![Live GPU overview with an LLM's weights, KV cache, and compute buffer breakdown](docs/screenshots/overview.png)

> Early, experimental software. Device and process counters are measurements; classifications
> and GGUF-derived breakdowns are estimates. This does not dump VRAM or enumerate a game's textures.

## Install from PyPI

[gputaskman is on PyPI](https://pypi.org/project/gputaskman/). No repository clone is needed:

```powershell
uv tool install gputaskman
gputaskman
```

Or, with [pipx](https://pipx.pypa.io/):

```powershell
pipx install gputaskman
gputaskman
```

Works on Windows, Linux/WSL, and macOS with Python 3.11+ and the supported GPU/driver
configuration described below. Both installers keep the app in an isolated environment.
The shorter `gtm` command is also available. Upgrade with `uv tool upgrade gputaskman`
or `pipx upgrade gputaskman`. For Windows without Python, use the portable download below.

## Portable Windows download

The [Releases page](https://github.com/shanselman/gputaskman/releases) is where tagged builds
publish two portable ZIPs: `gputaskman-VERSION-windows-x64.zip` for Intel/AMD Windows PCs, and
`gputaskman-VERSION-windows-arm64.zip` for native Windows ARM64. These are separate native builds,
not a universal executable.

Extract the **entire ZIP**, keeping `_internal` beside `gputaskman.exe`. Open Windows Terminal
or PowerShell in the extracted folder:

```powershell
.\gputaskman.exe
.\gputaskman.exe --once
.\gputaskman.exe --json
```

No Python, uv, CUDA Toolkit, model download, or installation is needed. A supported GPU with a
current Windows WDDM driver is required. Builds are currently **unsigned**; code signing
and an installer are not part of this initial distribution. Each ZIP includes licenses and a
`.zip.sha256` companion file is published for integrity checks.

## Clone and run

Use **64-bit Windows 10/11**, an **NVIDIA or AMD GPU with a current WDDM driver**, and a terminal
(Windows Terminal is recommended). Live workload checks have included an RTX 4080 SUPER and an
RTX Spark N1X on Windows ARM64. CUDA workloads are supported on ARM64 with a compatible NVIDIA
driver and ARM64 CUDA runtime.

AMD live check: Radeon 860M Graphics on Windows 11 Enterprise build 26300, AMD driver 32.0.22050.1003.
During desktop/browser activity, WDDM reported per-process VRAM and AMD ADL reported GPU utilization,
clocks, and ASIC power; temperature, memory activity, and PCIe readings were unavailable. Vulkan/ROCm
workloads were not tested on this machine.

Install [Git for Windows](https://git-scm.com/downloads/win) and
[uv](https://docs.astral.sh/uv/getting-started/installation/) if needed. With WinGet:

```powershell
winget install --id Git.Git --exact
winget install --id astral-sh.uv --exact
```

Open a new PowerShell window so those tools are on `PATH`, then:

```powershell
git clone https://github.com/shanselman/gputaskman.git
cd gputaskman
uv run --locked gputaskman
```

**That's it.** uv obtains the pinned Python version if needed, creates `.venv`, and installs
the locked dependencies. The first run needs internet access. There is no CUDA Toolkit,
Visual Studio, administrator session, API key, or model download required to monitor the GPU.
NVML ships with the NVIDIA driver; AMD ADL is supplied by supported AMD Windows drivers.

Already have Python 3.11+ and prefer pip?

```powershell
git clone https://github.com/shanselman/gputaskman.git
cd gputaskman
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\gputaskman.exe
```

No PowerShell activation script or execution-policy change is needed. The pip route resolves
compatible dependencies; the uv route uses the checked-in lockfile.

If an alternate package index makes `--locked` report a stale lockfile on an unchanged checkout,
`uv run --frozen gputaskman` uses the checked-in lockfile without re-resolving it. `--frozen`
skips the consistency check, so do not use it after changing dependencies.

## Linux and WSL

Tagged releases include portable Linux builds: `gputaskman-VERSION-linux-x64.tar.gz` and
`gputaskman-VERSION-linux-arm64.tar.gz`. They need glibc 2.35 or newer (Ubuntu 22.04, Debian 12,
or newer) and an NVIDIA driver (or, on native Linux, an AMD GPU on the `amdgpu` driver, an Intel GPU on `i915`,
or a Huawei Ascend 310P with `npu-smi`). NVIDIA works natively, in containers, and in WSL:

```bash
tar -xzf gputaskman-VERSION-linux-x64.tar.gz
cd gputaskman-VERSION-linux-x64
./gputaskman
```

Or clone and run from source, in a Linux or WSL shell:

```bash
git clone https://github.com/shanselman/gputaskman.git
cd gputaskman
uv run --locked gputaskman
```

What can be measured depends on where it runs, and the UI says which mode it is in.

On native Linux, Ascend chips are listed individually. Their device memory and per-process memory
come from `npu-smi`; AI-core and memory-bandwidth activity come from its watch stream, not
per-process engine counters. Memory refreshes every 10 seconds, and a failed refresh or stale watch
is reported as unavailable. The contributor reported testing an Atlas 300I Duo with a vLLM
workload; driver version and these subsequent fixes have not been independently validated on hardware.
Ascend container and PID-namespace behavior is not validated.

| | Native Linux | WSL, interop on (default) | WSL, interop off or `--no-host` |
|---|---|---|---|
| Device totals, load, temperature, power, clocks | NVML | NVML | NVML |
| Per-process VRAM | NVML, measured (in a container: owner matched, `*` or `?`) | Matched from the Windows host, marked `*` or `?` | `n/a` |
| Per-process GPU load | NVML (SM, encoder, decoder) | `n/a` for Linux processes | `n/a` |
| Spill / Paged | Not available (blank) | Windows rows only | Not available |
| WSL VM total, Windows apps | — | Windows host counters | — |
| Runtimes, model files, LLM breakdown | `/proc/<pid>/maps`, command line | Same | Same |

**Native Linux** reads per-process memory from NVML's running compute and graphics process lists
and per-process SM, encoder, and decoder load from NVML utilization samples. Linux has no
equivalent of WDDM's shared-memory or commit counters, so Spill and Paged stay blank rather than
showing zero. Only processes you can inspect are listed; other users' processes need `sudo`.

JSON uses `null` for unavailable counters, not a measured zero: spill/commit/paged values on
native Linux, and per-process engine load for WSL Linux rows. Native NVML engine readings remain
available even if NVML hides that process's VRAM. Windows host rows retain their measured counters.

**Containers** (Docker, Kubernetes, rented GPU boxes) need one more step: NVML reports *host*
PIDs, which mean nothing in the container's `/proc`, and the container cannot see the host PID of
its own processes. gputaskman pairs NVML's unresolved PIDs with the container's processes that
have `libcuda.so` mapped (NVML-only tools such as `nvidia-smi` open the same device files but
never load libcuda). Both PID namespaces number processes in creation order, so the pairing is by
order: `*` when the counts match or a process starting while gputaskman runs pins it, `?` when
NVML lists more compute processes than the container has (another container on the same GPU).
Per-process GPU load follows the match. Anything left over is shown as `H:<pid>`,
*outside this container* (for example, the host's display server), and is never looked up locally.

Live check, on a Vast.ai Docker container (Ubuntu 22.04, RTX 3060 Ti, driver 580.82.09): two
PyTorch jobs already running at startup, a third started while watching, and one killed, were all
matched to the right container PIDs with their NVML sizes (1.12, 2.52 and 1.59 GB) and live GPU%.
An Ollama `qwen3:0.6b` runner was detected and named from Ollama's store with `api+mapped+cmdline`
evidence, its weights / KV-cache / compute split shown, and the TUI rendered.

Live check, bare-metal Linux (CachyOS, kernel 7.3.3, same PID namespace, no matching needed) on a
laptop with an RTX 4050 Laptop GPU (driver 620.70.06) and an AMD Phoenix iGPU: every per-process size
matched `nvidia-smi` (a llama-server with Qwen2.5-1.5B Q4_K_M at `-ngl 99 -c 4096` showed 1.19 GB
against 1218 MiB, and `gputaskman diff` put the device change at +1.20 GB, against +1224 MiB from
`nvidia-smi`). The model was identified with `api+mapped+cmdline` evidence, and with
`-lv 4 --log-file` the exact CUDA0 buffer sizes (935 MB weights, 112 MB KV cache) matched the
GGUF-based estimates. During generation the TUI showed the server at 96% GPU% against 95% from
`nvidia-smi pmon`.

**AMD GPUs on native Linux** (the `amdgpu` driver) need no ROCm, AMD SMI, or extra package: the
kernel already exposes what's needed.

| | Source |
|---|---|
| Device VRAM, GTT, load, temperature, power, clocks, fan, PCIe link | `/sys/class/drm/card*/device` and its `hwmon` |
| Per-process VRAM and GTT | `drm-resident-vram` / `drm-resident-gtt` (older kernels: `drm-memory-*`) and `drm-shared-*` in `/proc/<pid>/fdinfo/<fd>` |
| Per-process engine load (gfx, compute, video) | `drm-engine-*` busy time, sampled twice (`n/a` until a new client has two samples); `n/a` for ROCm/HIP processes (see below) |

GTT is GPU-accessible system RAM, so it fills the SPILL column (and `shared (in system RAM)` for
the device), like WDDM's non-local usage. On an APU (an integrated GPU, detected by amdgpu not
reporting a VRAM vendor, a heuristic validated on one APU) GTT is the GPU's normal memory and "VRAM"
is only a RAM carve-out, so GTT is still shown but never flagged as spill, and an LLM's memory is its
VRAM plus GTT. The SDMA engine (`dma`) fills the COPY column. Paged stays blank: there is no commit
counter. VRAM and SPILL are charged to a process only when nothing else can hold its bytes: each of
its DRM clients has exactly one visible holder, was opened through a render node (`/dev/dri/renderD*`),
and reports zero bytes in every `drm-shared-*` region. Then its resident VRAM and GTT are its own and add
up without double counting. Otherwise the process shows VRAM and SPILL as `n/a` (`null` in JSON, with
`vram_unattributed_reason`) and its bytes stay in *unattributed*:

- **Shared client** (fd passing, fork; 1Password and its crash handler on the test desktop): one set of
  counters, and nothing says which process uses it. GPU% is `n/a` too.
- **Shared buffers** (a window a compositor imports, video frames): `drm-shared-*` counts a buffer by
  where it *prefers* to live, resident or not, while `drm-resident-*` counts where buffers *are*, so
  resident minus shared is not private memory and is never computed.
- **Primary node** (`/dev/dri/card*`, typically a compositor): root-owned logind keeps a copy of that
  fd, which a normal user can't see, so a single holder can't be proven.
- **Legacy**: older kernels whose amdgpu fdinfo lacks `drm-shared-vram`/`drm-shared-gtt`, and Intel integrated GPUs
  and `xe` (below).
- **Incomplete**: some of the process's fds or fdinfo could not be read (GPU% is `n/a` too).
- **Partial scan**: the TUI's 50 ms listing budget ran out this refresh, so a process it skipped might
  hold the same client (GPU% is `n/a` too). Expected to be rare on a desktop (~9 ms for ~8,000 fds,
  not yet re-measured with this rule); one-shot output always scans fully.

The inspector and JSON (`vram_resident_bytes`, `gtt_resident_bytes`) still show every buffer the
process can reach, shared ones included; the same buffer can appear under several processes, so these
are a reference and are never summed. Unparseable or missing values are unknown (`null`), not 0. amdgpu has no device-wide
video engine counter, so the device's encode/decode load stays N/A; per-process video load is
measured. On VCN 4 and later (RDNA3 and newer, including the RX 9070) decode and encode share one
ring, so it is reported as `video` rather than split. Only processes you can inspect are listed; the header and JSON (`process_status`) say how many could not be, and `sudo` includes other users'. `--gpu 1`
selects the next card, such as an APU's integrated GPU next to a discrete one. A ROCm/HIP process
(tagged `HIP`) has its VRAM in fdinfo like any other. Its GPU% is not: HIP submits compute through user-mode queues on `/dev/kfd`, which the kernel
does not time per process, so a process holding `/dev/kfd` shows engines as `n/a` (`null` in JSON)
rather than a measured-looking 0. The device's `AMD gfx` load does include that work.
This path is native-Linux only; under WSL the GPU is `/dev/dxg`, with no DRM driver involved.

Live check, bare-metal Arch-based Linux (kernel 7.2.15, `amdgpu`), Radeon RX 9070-series (Navi 48)
plus a Radeon 610M iGPU: both cards were found, named, and read; VAAPI HEVC encode and decode with `ffmpeg`
each showed as a `video` row at ~95-100% video load (on the RX 9070 both were charged to amdgpu's
unified `enc` ring, hence `video`); desktop processes' VRAM summed
to less than the card's used VRAM, with the compositor's imported windows listed as shared.
llama.cpp's ROCm server image (`ghcr.io/ggml-org/llama.cpp:server-rocm`, Qwen3-4B-Instruct-2507
Q4_K_M, `-ngl 99 -c 8192`, next to a running game) showed as an `LLM` row with `llama.cpp+ggml+HIP`
and 3.95 GB of VRAM, matching the 2.49 GB of weights plus 1.21 GB of f16 KV cache plus compute
buffers, and the 4.11 GB KFD reports for the process in `/sys/class/kfd/kfd/proc/<pid>/vram_*`.
While it generated at 103 tok/s the device read 99% gfx and 339 W, but fdinfo had no engine lines
for it, which is why ROCm rows show engines as `n/a`. PyTorch-ROCm has not been run on this path.
That check predates the stricter attribution above (no subtraction of shared bytes; render-node,
single-holder and complete-scan requirements; per-client timestamps; PID-reuse checks), which is
covered by unit tests only and **not yet validated on AMD hardware**.

**Intel GPUs on native Linux** (the `i915` driver) use the same DRM fdinfo path, with these differences:

| | Source |
|---|---|
| Device temperature, power, GT clock, throttle, load and video | `hwmon`, `gt/gt0`, the i915 performance counters and RAPL's uncore zone. Power is the rate of the cumulative energy counter between two refreshes, so it is `n/a` on the first: the hwmon counter on a discrete card, the uncore RAPL zone on an integrated GPU (root-only). Without a GPU temperature sensor, an integrated GPU falls back to the CPU package sensor (`coretemp`), labelled as shared with the CPU. The GT clock is the actual frequency (`rps_act_freq_mhz`, the requested one as fallback); throttle reasons come from `throttle_reason_*`. GPU % is the busiest engine's busy share from the performance counters and video is the video engine's share (unified decode+encode, so `enc`/`dec` stay unset); both need root (`perf_event_paranoid=2` here blocks normal users), otherwise `N/A` with a reason |
| Device VRAM total | The size of the local-memory PCI window (`resource2`), an **estimate**: kernel-owned memory is missing from the sysfs view, resizable BAR rounds to a power of two, and a partial window can be smaller than the card, so the total can differ from the card's memory. A window under 1 GiB (an integrated GPU, or a discrete card without resizable BAR) shows `N/A` (`null` in JSON). Free memory is never shown for Intel |
| Device VRAM used | The sum of its clients' resident memory: i915 has no device-wide counter, so buffers shared between processes can be counted twice, and kernel-owned memory and processes gputaskman can't inspect are missing. Local memory on a discrete card, system memory on an integrated GPU. The UI, `--once`, `--json` (`memory_note`) and `gputaskman diff` all say so. `n/a` when any client's memory is unknown |
| Integrated or discrete | An iGPU has no `local0`, so a card with a local-memory window of 1 GiB or more is discrete, and so is any card whose fdinfo reports local memory, even when its window is smaller |
| Per-process VRAM and SPILL | `drm-resident-local0` and `drm-resident-system0`, plus their `stolen-` regions (the firmware-reserved part of each), under the same private-client rules as AMD. On an integrated GPU there is no local memory: a provably private client's `system0` bytes are its GPU memory in the VRAM column, and SPILL stays `N/A` (system memory is its memory, not spill). On a discrete card `system0` is memory the GPU's buffers currently sit in, including buffers placed there on purpose, so SPILL there (like AMD's GTT) is where they are, not proof they were pushed out of VRAM. The shared gate needs only the regions the device has, so a discrete card with a missing `drm-shared-local0` still shows `n/a` (`legacy`), while an integrated client with shared buffers shows `n/a` (`shared_buffers`). A region value that can't be parsed makes that figure `n/a`, not a smaller number |
| Per-process engine load (3d, compute, copy, video) | `drm-engine-render`, `-compute`, `-copy`, `-video`, `-video-enhance` busy time, sampled twice. `n/a` where the driver reports none (before Gen 8). A class with several identical engines (`drm-engine-capacity-*`, such as video on an Arc) is not divided by its capacity, so one busy engine reads 100% |

Device memory bandwidth and the PCIe link are `n/a`: i915 has no device-wide bandwidth counter, and its
sysfs link fields can report the power-on default instead of the live link. Device GPU load and video load
come from the i915 performance counters and need root; power comes from RAPL's uncore zone on an integrated
GPU and needs root too.
The load readings sample every exposed `*-busy` engine instance, including compute and additional
video/copy engines, and use the busiest instance rather than averaging across idle engines. Discrete
PMUs are matched by PCI address (`i915_<PCI BDF>`); the legacy `i915` PMU is used only for an integrated
card. If multiple cards resolve to the same PMU, their load stays `N/A` with an attribution reason.
Each collector opens device counters only for its selected GPU and owns those handles and baselines;
closing one does not reset another collector's readings.
`xe` cards are detected, but its fdinfo uses other keys (`drm-cycles-*`, `drm-resident-vram0`/`-gtt`), so its
per-process rows show `n/a` until they are decoded on hardware.

Live check, reported by the contributor and not repeated by the maintainer: Linux 6.12, an Arc A770 (discrete, 16 GiB
BAR) and a UHD 730 (integrated) on `i915`. Both cards were found, named, and selectable with `--gpu`; the Arc showed
per-process VRAM and engine load, and the TUI rendered both. The fdinfo formats of both drivers, taken from the kernel
sources, are covered by unit tests. **Not validated on hardware:** `xe`, Arc cards without resizable BAR, and
multi-GPU systems other than that pair.

Live check on an integrated GPU (ASUS Vivobook 14 X1404VA, Intel Core 3 100U Xe-LP, `i915`, kernel 7.3.1, Mesa 26.2.5):
the card is found and named, the header shows the actual GT clock (`GT 300/1250 MHz`) with no throttle reasons active,
and a headless `vkmark` run shows as a 3D/game Vulkan row at 82% GPU. Per-process memory stays `n/a`
(`shared_buffers`): the compositor imports every window, so no client is provably private. The `gtt_resident_bytes`
reference and the engine percentages are still measured.

Live check of the root-only readings on the same machine (`intel_gpu_top` from intel-gpu-tools as reference,
`perf_event_paranoid=2`): under `vkmark` load the header's GPU % reads 48-95% against `intel_gpu_top` Render
51-92% across scenes, and power reads 0.4 W idle against its 0.4 W (under load 9.0 W against 6.5-6.8 W in a
nearby window; scenes vary, and uncore is approximate while the CPU is active). A VA-API `ffmpeg` decode loop
drives the video engine to 99.9% on both (`intel_gpu_top` Video 99.4-99.9%). Without root the three stay `N/A`
with a reason, while the GT clock, throttle and per-process rows keep working.
Per-PCI PMU selection and compute/secondary-engine coverage are regression-tested with synthetic
counters, **not yet validated on hardware**; the live check above covers only the Vivobook's exercised engines.

**WSL** is different, because Windows owns the GPU. Linux cannot see how much VRAM its own
processes use: `nvidia-smi` lists no processes, NVML's per-process samples carry *Windows* PIDs
(never looked up in Linux's `/proc`), and Windows charges all Linux GPU memory to the VM worker
process, `vmwp.exe`. When WSL interop is enabled (`/proc/sys/fs/binfmt_misc/WSLInterop` reads
`enabled`), gputaskman starts one long-lived `powershell.exe` that reads the Windows GPU counters
through the same PDH array API as the Windows backend and streams them back once a second:

- **WSL VM (`vmwp.exe`)** is the exact total of Linux GPU memory, across every distro, WSLg, and
  Docker Desktop.
- **Windows apps** appear as `W:<pid>` rows with their VRAM, spill, paged, and engine load.
  Windows and Linux PIDs are separate namespaces, so the prefix keeps them apart.
- **Linux rows** get a size by matching `vmwp.exe`'s GPU contexts to Linux processes (below).

The first host sample takes about 2.5 s; the TUI shows `starting` until then, and `--once` waits
for it. The host adapter is paired with the NVML device by memory usage, because WSL's adapter
LUIDs differ from the host's. An ambiguous pairing shows `unmatched` instead of guessing. With other
Hyper-V VMs or Windows Sandbox running, the row is labeled `VM worker` because it cannot be tied
to WSL alone. `--no-host` turns the host stream off; without interop you get device-wide numbers
and a list of the Linux processes that hold `/dev/dxg`, with VRAM shown as `n/a`.

### How WSL per-process VRAM is matched

`vmwp.exe` exposes one GPU-memory instance per guest GPU context, all with the same name: sizes
without owners. Measured on WSL2 with driver 610.88, every Linux process that opens `/dev/dxg`
gets exactly one context (an NVML-only process too, at 0 bytes), and Windows lists them **newest
first**, keeping that order as contexts come and go. Contexts from outside this distro (WSLg, other
distros, Docker, services you cannot inspect) are interleaved by their own age.

gputaskman follows the contexts from sample to sample and labels them with evidence:

| Marker | Meaning | Established by |
|---|---|---|
| `*` matched | Windows' exact size; the owner is inferred, and only one explanation fits (or every one that fits gives the same size) | A new Linux GPU process and a new context appearing together; elimination; this monitor's own context being the only zero-byte candidate at startup; candidate contexts that are exactly the same size |
| `?` guess | Heuristic owner for processes that were already running | Newer process ↔ newer context, outside contexts assumed oldest; cached foreign sizes can rank extra candidates but never establish ownership |
| `n/a` | No context could be assigned | For example, a process that has not created its GPU context yet |

A guess is recomputed every refresh and never counts as evidence for itself. An exited process's
context is held aside until it disappears, so it is not handed to a newcomer, and a context that
grows by gigabytes between refreshes (weights uploading) keeps an established owner. Multiple
zero-byte contexts at startup do not pin the monitor to an arbitrary one that might later grow.
Foreign sizes observed while only the monitor is running are cached per WSL boot in
`~/.cache/gputaskman/`, but remain hints: a process may exit while the monitor is off and a local
process may reuse its size. Observed foreign-context exits and idle samples clear obsolete hints.

The matched size also feeds VRAM sorting, initial model selection, the bounded model-file scan
priority, and the LLM weights / KV-cache breakdown. Inferred Linux rows remain below the VM total
and are never added to it again. The inspector and `--json` (`vram_inferred_by`) say which kind
of match each row has.

In one live check, a 2 GB and a 5 GB PyTorch allocation were matched to the right PIDs at
2.119 GiB and 4.914 GiB: the allocator's own figure plus about 0.26 GiB of CUDA context.
That is one machine and driver, not a guarantee. A process that started long before it first used
the GPU can break the start-order guess, and processes started in the same refresh are ordered by
guess unless their sizes match.

### NixOS

With flakes enabled, run the published flake or a checkout:

```bash
nix run github:shanselman/gputaskman
nix run github:shanselman/gputaskman -- --list-gpus
nix run path:. -- --once
nix run path:. -- --json
```

Build both `gputaskman` and its `gtm` alias with `nix build path:.` (executables are in
`result/bin/`). The flake provides packages for `x86_64-linux` and `aarch64-linux`; `flake.lock`
pins nixpkgs. Dependencies come from nixpkgs, independently of `uv.lock`. No pip installation,
virtual environment, CUDA Toolkit, or ROCm installation is needed. GPU drivers remain supplied by
the host, never bundled with the package. NVIDIA's nixpkgs Python binding finds NVML through
`/run/opengl-driver/lib`; AMD and Intel use kernel sysfs and DRM fdinfo interfaces directly. On WSL2 the
host driver lives in `/usr/lib/wsl/lib` instead, and the wrappers add that directory to
`LD_LIBRARY_PATH` (it is ignored when absent, so NixOS is unaffected).

For source development and tests:

```bash
nix develop path:.
python -m gputaskman --once
python -m unittest discover -s tests -q
```

`nix flake check path:.` builds the package and runs the same hardware-independent tests.
For a flake-based NixOS configuration, add this repository as a `gputaskman` input, then include
`inputs.gputaskman.packages.${pkgs.stdenv.hostPlatform.system}.default` in
`environment.systemPackages`. Enable your GPU's normal NixOS driver configuration first.
The flake was contributed and checked on x86_64 NixOS 26.05; CI does not build it, and ARM64 has
not been validated.

## macOS (Apple Silicon)

Run from a clone (`uv run gputaskman`); there is no portable macOS download yet. Apple GPUs share
one pool of memory with the CPU, so there is no VRAM to split and no spill/commit counters. What
macOS exposes without root:

| Column | Source | Who |
|---|---|---|
| GPU memory (header) | IOAccelerator `PerformanceStatistics`: `Alloc system memory` (all GPU allocations) out of physical RAM | device |
| GPU% (header) | `Device Utilization %` | device |
| GPU% | Each Metal client (`AGXDeviceUserClient`) names its creator PID and keeps a cumulative `accumulatedGPUTime`; the rate between samples, per client (a closed client never hides the others' work) | every process |
| VRAM | `footprint`: the process's memory macOS charges to graphics (the `(graphics)` categories: Metal buffers, textures) | your own processes |
| Runtime, model files | Mapped files from `proc_pidinfo` region walks (psutil has no memory maps on macOS): `libggml*.dylib`, `libllama*.dylib`, `libmlx*.dylib`, `libtorch_cpu.dylib`, the Metal driver bundle | your own processes |

Other users' processes (WindowServer, system daemons) show GPU% with memory `n/a`: `footprint` needs
root for them. Idle ones are left out of the table, except WindowServer. Run with `sudo` to measure
every process's memory. `footprint` takes ~70 ms per process, so it runs in a background thread:
busy processes and new ones every pass (~2 s), idle ones every 30 s. A new process shows `n/a` for a
second or two until its first measurement; a single failed measurement keeps the last size, and three
in a row back off for a minute. Measurements are tied to the process's start time, so a reused PID
never shows the previous process's numbers.

Metal reports one GPU time per process, with no 3D/compute/copy/video split, so those columns show
`n/a`; a Metal process above 10% GPU is classified `3D/game` unless something more specific applies
(AI runtime, browser, compositor). Spill, paged, clocks, power, and temperature are not available, so
the chart picker offers only GPU and GPU memory. Handy's log is read from
`~/Library/Logs/com.pais.handy/handy.log`.

Live check on an M4 Mac mini (macOS 26.3.1): a PyTorch MPS job with a 2 GiB buffer and a matmul loop
showed `Python  AI  PyTorch+Metal  3.06 GB  99%`, with device utilization at 99% and allocated GPU
memory rising by the same amount.

Scans skip other apps' sandbox containers (`~/Library/Containers`, `~/Library/Group Containers`) and
system files without opening them: opening a file in a protected container blocks while macOS waits
for a privacy consent prompt.

## What you get

- **Device view:** VRAM usage, GPU activity, memory activity, PCIe traffic, temperature, power,
  video-engine clock, and responsive history charts with an all-charts or single-metric view.
- **Process view:** local GPU memory, GPU-accessible system RAM, estimated non-resident commit,
  engine activity, and detected runtimes such as CUDA, HIP, Vulkan, Direct3D, ggml and ONNX Runtime.
- **System RAM and CPU:** a panel plus RAM and CPU% columns, as the OS reports them: machine memory
  and load, never GPU memory (see [System RAM and CPU](#system-ram-and-cpu)).
- **Model inspector:** which model each process holds (LLM, decision, embedding, speech, or diffusion), how
  it was identified, architecture, quantization, layer count, weights, estimated KV cost, and
  buffer sizes from compatible load logs.
- **Live server hints:** context size, busy slots, throughput when available, and friendly Ollama
  or LM Studio model names.
- **Scriptable output:** one-shot text and JSON, using the same collector as the TUI.

### Screenshots

These are captures of real local workloads through Textual's screenshot renderer, not mock data.
Full filesystem paths and command lines were omitted in the capture copy. Measurements vary
with the workload; the examples are not benchmarks.

**The compositor, without treating all shared surfaces as its private VRAM**

![Compositor selected, showing local memory and the explanation of shared-surface accounting](docs/screenshots/compositor.png)

**GPU-accessible system RAM, sorted by the Spill column**

![Processes sorted by system-memory usage, alongside the selected process's memory explanation](docs/screenshots/system-memory.png)

**An explanation of what is measured, inferred, and unavailable**

![Built-in help explaining GPU counters and the limits of memory introspection](docs/screenshots/help.png)

## Usage

```powershell
uv run --locked gputaskman              # live TUI
uv run --locked gtm                     # shorter alias
uv run --locked gputaskman -i 2         # sample every two seconds
uv run --locked gputaskman --once       # one text snapshot
uv run --locked gputaskman --json       # one JSON snapshot; implies --once
uv run --locked gputaskman diff before.json after.json   # compare two --json snapshots
uv run --locked gputaskman record -o session.jsonl --duration 300   # append one --json snapshot per second for 5 minutes
uv run --locked gputaskman --no-host    # WSL: don't read Windows host counters through interop
uv run --locked gputaskman --no-system  # don't sample system RAM/CPU: no panel, columns, or JSON section
uv run --locked gputaskman --help
```

| Key | Action |
|---|---|
| Up / Down | Select a process and inspect it |
| `s` | Cycle sort: VRAM, GPU%, spill, paged, name (then RAM, CPU% unless `--no-system`) |
| `c` | Cycle all charts or a single metric (also available in the **Charts** dropdown) |
| `a` | Switch between top and all collected processes |
| `m` | Toggle full diagnostics / slim monitoring |
| `p` | Pause / resume collection |
| `?` | Explain the counters and their limits |
| Escape | Close help |
| `q` | Quit |

A wide terminal (roughly 140 columns or more) makes the full table easier to read.
The TUI starts in **Full diagnostics** mode. Press `m` (or click **Slim/Full** in the
footer) for **Slim monitoring**: each GPU header hides driver/CUDA versions, PCI bus
address, driver model, PCIe generation/width, compute capability, power/clock limits,
performance state, and secondary memory/video clocks. GPU identity, memory bars and
provenance/availability warnings, utilization, PCIe throughput, temperature, current power
and SM/GFX clock, throttling alerts, charts, processes, and process inspection remain.
The header border names the active mode; press `m` again to restore the full details.
This is a session-only display toggle, applies to every GPU, and does not change collection
or one-shot text/JSON output.
All eight history charts are shown by default: GPU, SM clock (GFX clock on AMD), VRAM, memory activity,
temperature, power, encoder, and decoder. They stack in narrow terminals and share rows as
the window widens. Click the **Charts** dropdown to focus on one metric or return to **All charts**.
A blank sample means no reading; zero is drawn at the baseline. Percentages use a 0–100%
scale, temperature uses 0–120°C, and the power and clock scales are displayed below the charts.
The dashboard and inspector scroll independently in short terminals so the process list stays usable.
Model/runtime discovery is throttled, so newly loaded models may take about 15 seconds to appear.

On multi-GPU systems, the TUI stacks a header and trend charts for each GPU and adds a **GPU**
column to the combined process list. A process using two GPUs has a separate row for each;
selecting a row inspects its memory on that GPU. Single-GPU layouts are unchanged.
Memory totals and histories stay separate per device; model files and runtime buffer logs
can span devices and do not prove per-device model placement. Discovery's 150 ms scan budget
is divided across GPUs, though each GPU has its own counter and telemetry queries.
Live check on a mixed-vendor Linux laptop (RTX 4050 Laptop GPU through NVML, AMD Phoenix iGPU
through DRM; CachyOS, kernel 7.3.3): both GPUs got their own header and charts, and the compositor
had a separate row on each GPU (measured on the NVIDIA one, `n/a` on the iGPU, where it opened
`/dev/dri/card*` and root-owned holders can't be ruled out without root). Systems with two
discrete GPUs, and multi-GPU on Windows, are **not yet validated**.
On Windows, the live TUI checks DXGI adapter topology every five seconds in the existing
sampling worker (or on the next sample with a longer interval). Removed adapters retain their
session GPU number with unavailable readings; surviving adapters keep their collectors and
histories, and newly discovered adapters are appended. Re-enabling an adapter rebuilds its
providers. LUIDs identify adapters; if a LUID changes, only a unique PCI location **and**
matching vendor/description can reconnect the old slot. Without that evidence, the old slot
remains unavailable rather than transferring its history or process ownership to another GPU.
`--gpu N` pins the adapter selected at startup, not whichever adapter later occupies index N;
it does not start monitoring other newly discovered devices. Session GPU numbers therefore
need not match a fresh `--list-gpus` after a topology change.
NVML handles are reacquired at the same low-frequency cadence; missing counters, lost telemetry
or sampling errors trigger provider reconstruction at the next successful discovery. Persistent
failures back off to at most one reconstruction per minute; a reappearing or changed adapter retries immediately.
Discovery errors retain the last known topology with a warning, not invented zero usage.
Pausing suspends discovery as well as sampling; resume performs an overdue check.
This recovery is covered by mocks, **not validated with hardware disable/re-enable** on the
reporter's hybrid AMD/NVIDIA laptop. On a three-GPU desktop (RTX 5060 Ti, RTX 3060, AMD iGPU;
Windows 11 build 26200, NVIDIA driver 610.88), disabling and re-enabling the RTX 3060 recovered
its slot, history, VRAM and processes, but NVML reads for that GPU then faulted inside `nvml.dll`
and `nvmlShutdown()` hung on exit. A fresh process read it normally. gputaskman therefore stops
reading NVML for a GPU whose reads fault, shows "restart gputaskman to restore it", and skips
`nvmlShutdown()` once an NVIDIA adapter has gone away (process exit releases the session). That
mitigation is covered by mocks and not yet validated on that hardware. It is not a claim of
physical hot-swap support.
Other backends and one-shot output retain startup-only discovery.
Under multi-GPU WSL, host attribution is disabled with a reason rather than risking matching
two devices by memory usage; Linux processes are listed once with GPU `?`, because `/dev/dxg`
does not identify their GPU. `--gpu N` retains the existing single-device host matching.

`--list-gpus` lists selectable non-software DXGI adapters on Windows (including root-enumerated
display adapters), or hardware adapters on other platforms, with their vendor and adapter key. `--gpu N` selects
one of these adapters: NVIDIA adapters are listed first, followed by other vendors, with each
group ordered by dedicated memory. An out-of-range
index is rejected rather than silently selecting another device. Compute-only devices such as an
NPU are not graphics adapters. AMD and NVIDIA hardware coverage is described below; configurations
with multiple GPUs and mixed vendors need further hardware validation. `--once` and `--json`
still default to GPU 0; use `--gpu N` to collect another device without changing the snapshot format.
Device telemetry is matched to WDDM adapters by PCI location, not display name or enumeration order.
If a unique match cannot be established, telemetry is unavailable; buffer logs can still span devices,
so model-buffer attribution across GPUs is not yet reliable. This conservative matching also applies
to NVIDIA and may suppress telemetry when Windows does not expose a usable PCI address.

On Windows, each exact DXGI LUID is also resolved through `DisplayConfigGetDeviceInfo` to its
PnP device interface, including inactive displays, then read through SetupAPI. A `ROOT\DISPLAY`
instance uses its PnP friendly name (device description as fallback) and is labelled `ROOT display
adapter` in `--list-gpus`, rather than inheriting the physical GPU's DXGI name or vendor label.
It stays selectable at the same index, is not paired with PCI telemetry, and still needs matching
LUID counters to show usage. A missing PnP name gives an explicit root-display label; a failed
identity lookup retains the DXGI description with a reason in the list, text, TUI and JSON
`counter_status`. JSON `adapter.device_instance_id` records the resolved instance (empty if unknown);
`adapter.identity_status` records lookup limitations. Physical adapter names and ordering are unchanged.
The StarDesk inactive virtual-display case is covered by mocked API regressions, **not yet validated
with StarDesk installed**. Read-only live checks on Windows 11 build 28000, NVIDIA RTX Spark N1X,
driver 32.0.16.2114 confirmed its physical PCI identity and unchanged display name; no display
configuration or drivers were changed.

AMD adapters can be selected and use the existing Windows GPU process counters; HIP runtime DLLs
are recognized as runtime evidence. On supported drivers, the AMD Display Library supplied by the
Windows driver can provide graphics activity, clocks, edge temperature, board or ASIC power, fan
and PCIe lane count. An optional AMD SMI Python installation is a fallback and may provide graphics
activity, edge temperature and socket power. Both require a PCI match to the selected WDDM adapter.
The SMI library and its native dependencies must be available to the Python environment; it is not
bundled in the portable build. The RX 9060 XT HIP workload and ADL telemetry have been observed on
this machine; AMD SMI telemetry and Vulkan workloads remain unvalidated.
A loaded DLL does not prove GPU execution.

Unavailable utilization, temperature, power, clock and PCIe readings are displayed as `N/A` in
text and the TUI and as `null` in JSON. For example, a missing NVML power limit means that the
driver did not expose that reading; it does not by itself indicate a hardware fault.
Missing per-process memory or engine samples likewise show `N/A`/`null`; the paged estimate is
unavailable when any of its input counters is missing.

On GPU-paravirtualized VMs, including Windows Sandbox, Windows may expose GPU counter LUIDs that
do not match the selected DXGI adapter. When that happens, the counters cannot safely be assigned
to the adapter: the process list and usage are unavailable, not zero. The text and TUI explain
the mismatch; JSON reports it in `counter_status` and sets `counter_adapter_matched` to `false`
(`null` when the check does not apply).
GPU-PV guests may also omit adapter memory counters. Counter values that are actually reported as
zero remain zero; a zero alone does not establish that a counter is unavailable.

Live check: Windows Sandbox on Windows 11 build 26100, hosted by an RTX 4080 SUPER with NVIDIA
driver 630.20. `D3DKMTEnumAdapters2/3` exposed the PDH LUID `0x4BF1` as a software adapter with a
different adapter/render GUID from DXGI's `0x4BB7`; it was not a proven alias for the NVIDIA GPU.
The portable build's `--once` explanation and `--json` unmatched status were verified in the guest.
Other Hyper-V GPU-PV configurations have not been validated.

### System RAM and CPU

A **System** panel under the GPU header, and **RAM** and **CPU%** columns in the process table, show what
each process costs the *machine*, as the OS reports it. The numbers come from psutil; on Windows the process
list comes from `NtQuerySystemInformation`, because psutil's per-process calls take about 6 s per pass for
~1,000 processes there. `s` sorts by them, unknown values last. Full diagnostics also shows a per-thread strip, 60-sample CPU and
RAM histories, and the top processes by RAM and CPU; slim monitoring keeps two lines (the RAM and CPU bars).
`--no-system` removes the panel, the columns and the sampling.

These are system-memory figures, never GPU bytes. They are not added to any GPU total or to the attributed VRAM:

- **RAM** is the process's working set (Linux: resident set), which includes pages shared with other processes,
  so it will not match Task Manager's default *Memory (active private working set)* column. `PRIVATE` in
  `--once` (`private` in `--json`) is the Windows commit charged to the process; it is `n/a` (`null`) elsewhere.
- **CPU%** is a share of the whole machine, like Task Manager: one busy thread on a 32-thread machine is about
  3%. It is a rate between two samples, so a new process (or one whose PID was reused) shows `…` for one
  refresh. Below 0.5% it is blank.
- **`n/a`** means the OS would not say (a protected process), the PID now belongs to a different process than the
  one the GPU row describes (compared by creation time when both are known), or the row is a Windows process
  listed from WSL, whose PID is not a Linux PID.
- Inside WSL the panel describes the Linux VM, and its border says so.
- When the numbers cannot be read, the panel, `--once` and `--json` say why, and the GPU view is unaffected.
  That covers 32-bit Python on Windows and a Windows build whose process-list layout does not match: the layout
  is checked against psutil for gputaskman's own process at startup instead of being trusted.

`--json` adds a `system` section: `cpu`, `ram` (with swap or pagefile), the top 10 processes machine-wide by
`rss` and by `cpu_pct`, `denied` (processes the OS would not describe) and `cost_ms`. Unknown values are `null`;
if sampling failed the section is `{"error": "<reason>"}`. `--once` and `--json` take their CPU readings inside
the second they already wait for GPU engine rates. `gputaskman diff` compares GPU memory only and ignores this
section.

Live checks: Windows 11 on the RTX 4080 SUPER machine with 995 processes (the process list matched psutil for the
same process; the TUI and `--json` ran), and Linux under WSL2 on the same machine (totals matched `/proc/meminfo`,
gputaskman's own resident size matched `VmRSS` within 1%, and a single busy thread read 3.12% of the machine
against the expected 100/32 = 3.1%). **Not validated on hardware:** macOS, Windows on ARM64, and bare-metal Linux.

### Comparing snapshots

To see which memory bucket actually grew when you load a model or change the context length, save
a JSON snapshot before and after and compare them:

```powershell
uv run --locked gputaskman --json > before.json
# load the model, or restart the server with a different -c, then send it one request
uv run --locked gputaskman --json > after.json
uv run --locked gputaskman diff before.json after.json          # add --json for machine-readable output
```

The device and process tables compare **measured** counters only. Runtime-reported segments
(llama.cpp buffer logs) and estimates (KV cache from GGUF metadata and `-c`) are listed separately
under each process and are never added to measured totals. A model file that appears in a process
is listed as evidence that the process opened or named it, not as GPU growth. The `system` section
(machine RAM and CPU) is not compared.

Processes are paired by PID, then by name and listening ports, then by a unique name, because a
model reload often restarts the server under a new PID. A process missing from one snapshot shows
`not listed`, not zero: it was not running or used too little GPU to be listed. Processes that
moved by less than 16 MB are hidden from the text output but included in `--json`.

A PID is not an identity, because both Windows and Linux reuse them. Each process in `--json` carries
`started_at` (its creation time in Unix seconds, taken from data the monitor already reads). The same
PID with start times more than 2 seconds apart is a different process, so no rule pairs it: the old
one shows as `gone`, the new one as `new`, and the device totals still compare normally. A missing or
`null` `started_at` (a protected process, a WSL host row, or a snapshot from an older version) never
counts as a conflict, so those rows are paired as described above. Linux derives start times from the
boot time in whole seconds, hence the 2-second allowance; a larger clock step can make one live
process look reused, which shows as `gone` plus `new` and never as a wrong change.

Two cards of one model share a name, so `diff` also compares the adapter each snapshot recorded.
It refuses a pair whose vendor ID, PCI address or (off Windows) adapter key conflicts. A Windows
adapter key changes at boot and when an adapter is re-enabled, so there it only produces a warning.
`adapter_check` in `--json` says what was compared: `pci`, `key` or `name-only`. `name-only` means
the snapshots (older ones, for example) do not both say which adapter they came from, so they were
matched by GPU name and backend alone, and the text output warns. These identifiers tell GPUs apart
within one machine; they do not prove two snapshots came from the same computer.

Take the "after" snapshot once the model has answered one request: CUDA and Vulkan can allocate
lazily, and allocators keep freed memory cached, so shrinkage may appear late. On an RTX 4080 SUPER
with Docker Model Runner's llama-server (Vulkan) and gemma-3-270m, changing `-c 4096` to `-c 32768`
estimated +504 MB of KV cache, but the server's measured dedicated memory grew by only +83 MB.
Gemma 3 uses sliding-window attention, so the estimate is an upper bound. The comparison makes
that gap visible instead of hiding it.

### Recording a session

To keep what a long run did to the GPU (a training job, an overnight server, a game session),
`record` appends one `--json` snapshot per interval to a file:

```powershell
uv run --locked gputaskman record -o session.jsonl                    # until Ctrl+C
uv run --locked gputaskman record --gpu 0 --interval 5 --duration 300 -o session.jsonl
```

Each line is one complete `--json` snapshot: the same fields, the same `null` for anything that could
not be measured, the same measured, inferred, and estimated labels. `taken` is the sample time in Unix
seconds. The file is only ever appended to, so recording again continues it (a last line cut short by
a crash is closed first). Every line is flushed to the operating system as it is written, so Ctrl+C
or killing the process keeps every sample taken so far; a power loss can still lose the last few
seconds, and a reader should skip a final line that does not parse. A full disk stops the recording
with an error and leaves the lines already written.

Recording samples the way the live TUI does, with no extra probes. Scanning is budgeted per sample, so
model details can appear a sample or two after a process starts; on WSL the first sample comes a few
seconds late while the Windows host counters warm up. On Windows, a GPU that is disabled and re-enabled
stays in the recording: the samples in between carry `null` measurements and the reason, then resume
when the adapter returns. The first version records one GPU (`--gpu`, default 0).

A snapshot of 52 processes is about 27 KB, so one second between samples is roughly 100 MB an hour; use
`--interval 5` or more for long runs. The lines carry the same paths, command-line flags, model names, and
local ports as `--json` (see [Privacy](#privacy-and-troubleshooting)), so review a recording before you share it.

Any line of the file is a normal snapshot, so two of them can go to `diff`:

```powershell
Get-Content session.jsonl -TotalCount 1 | Set-Content first.json
Get-Content session.jsonl -Tail 1 | Set-Content last.json
uv run --locked gputaskman diff first.json last.json
```

On Linux, WSL, and macOS use `head -n 1 session.jsonl > first.json` and `tail -n 1 session.jsonl > last.json`.
Replaying or summarizing a whole recording is not built yet.

## How can it know what's inside?

It combines evidence rather than reading arbitrary GPU memory. Windows' graphics kernel supplies
per-process memory and engine counters, while NVML or AMD ADL/SMI supplies device metrics. Process DLLs suggest
the runtime, and mapped files or command-line arguments identify candidate models. GGUF metadata
and tensor offsets let it estimate weights and KV-cache costs; compatible runtime logs and local
server APIs add allocated buffer sizes and live state. That can explain a llama.cpp process well,
but a runtime DLL or filename alone does not prove which allocation contains which object.

| Question | Evidence | Confidence / limitation |
|---|---|---|
| How full and busy is the card? | WDDM adapter counters and NVML or AMD ADL/SMI | Measured; driver support varies by metric |
| Which process uses GPU memory? | WDDM process counters; NVML on Linux | Measured accounting, not a resource ownership graph |
| Which WSL process uses GPU memory? | Windows host counters for `vmwp.exe`, matched to Linux processes | Exact VM total; per-process owner inferred (`*`) or guessed (`?`) |
| Which engines are active? | WDDM engine counters | CUDA can appear on the **3D** engine |
| Which runtime is loaded? | Mapped DLL names | Detection, not proof that every library is doing GPU work |
| Which model file is present? | Mapped files, CLI flags, GGUF magic/header | A file mapping does not prove GPU placement |
| Weights / KV / compute sizes? | Load logs, GGUF math, server state | Log-reported allocations or estimates, not a VRAM scan |
| Which texture, tensor, or kernel is this? | Not available here | Use PIX, Nsight, or runtime-specific instrumentation |

### Why not just `nvidia-smi`?

On GeForce under **WDDM**, Windows manages GPU memory and NVML may report per-process memory as
`N/A`. Windows GPU counters can still attribute memory and activity across Direct3D, Vulkan,
OpenGL, and CUDA. NVML remains useful for device-level readings. Under WSL, `nvidia-smi` lists
no processes at all; see [Linux and WSL](#linux-and-wsl).

### DWM, Spill, and Paged: read these carefully

**DWM is not everything.** Windows' per-process `Dedicated Usage` includes resources shared
between processes. The same surface may appear against its application, `csrss.exe`, and
`dwm.exe`. Adding those rows double-counts memory. This tool prefers `Local Usage` and
`Non Local Usage`; it falls back to `Dedicated Usage` / `Shared Usage` when the preferred
counters cannot be added. The adapter total remains the reference. See Microsoft's
[GPU accounting explanation](https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/).

In one live sample, DWM's older counter read about 15.3 GiB while `Local Usage` was about
1.4 GiB. The latter reduced the inflated process sum substantially. This is an observed
improvement, not a guarantee that every driver's process counters add up exactly.
When the row sum exceeds the adapter total, the category bar is normalized to fit;
legend values remain raw, and the UI labels the normalization.

**Spill means GPU-accessible system RAM, not proven overflow.** It can include intentionally
allocated or pinned host buffers. Large usage alongside a nearly full GPU can indicate
fallback and slower inference, but the counter alone cannot establish why it was allocated.

**Paged is an estimate:** `max(Total Committed - Local Usage - Non Local Usage, 0)`.
It is not an eviction trace, a pagefile measurement, or proof that Windows just moved
those bytes out of VRAM. The value is also less reliable when fallback counters are in use.

Memory labels currently use `GB` / `MB` for binary units (GiB / MiB). The `Mem activity`
reading is NVML's memory-activity percentage, **not measured bandwidth in GB/s**.

## LLM and model support

| Runtime | Current support |
|---|---|
| llama.cpp / llama-server | GGUF discovery, CLI settings, buffer logs, `/props` + `/v1/models` state; exercised live |
| Ollama | Bundled runners, mapped GGUF blobs named from Ollama manifests (`qwen3:14b`), parent `/api/ps`, compatible `server.log`; exercised live |
| LM Studio | Mapped files named from its models folder, `/api/v1/models` (falls back to `/api/v0/models`); tested with fixture responses, not yet end to end |
| vLLM | Recognized launch/worker signatures; `/v1/models` identifiers from an own listener or a verified parent, labelled API-reported, never worker residency or memory placement. Fixture-tested, including a real lightweight child process and HTTP listener; live Linux single-GPU and TP hardware **not validated** |
| stable-diffusion.cpp (`--diffusion-model`) | Diffusion, text-encoder and VAE files from launch arguments; no LLM memory breakdown |
| Hugging Face cache | `models--org--repo` paths are named `org/repo` |
| Other AI runtimes | DLL detection, plus model files installed with the app (marked *not proven loaded*) |
| Handy | Reads selected model from settings and bounded load/start/preview/unload events from `handy.log`; selection is not load evidence and log events are historical, not proof of current residency |
| WinML / local helper processes | GPU-backed execution is in scope; generic process counters and runtime detection only. No WinML-specific model/session or client-to-helper attribution yet |
| Windows Studio Effects | Only any observable GPU portion is in scope. NPU inference is out of scope; do not interpret GPU camera/render activity as proof of model execution |

WinML can select different execution backends; a local helper/service is still **local**, not
cloud-hosted. The GPU counters belong to whichever process holds/uses the GPU resources.
Parent-PID lookup currently helps find Ollama/LM Studio and local vLLM APIs; it is not a general cross-process
ownership graph for shared Windows services.

For vLLM, `vllm serve`, `python -m vllm.entrypoints.openai.api_server`, `VLLM::EngineCore`,
`VLLM::Worker` and `VLLM::Worker_TP<n>` are detection hints, not loading evidence. An inherited launch command
also works without process titles. Worker discovery checks at most two live ancestors, their
creation times and their signatures. Only one exclusively owned listening port on one recognized
server is accepted; shared/multiple listeners, unrelated or inaccessible ancestry and explicit
distributed/data-parallel options give no association. There is no port scanning or matching by
model name. Processes outside the visible PID namespace are not linked to local servers.
Source-grounded fixtures follow upstream vLLM commit
[`499a3d1`](https://github.com/vllm-project/vllm/tree/499a3d1e1e338e3e866c0719cfe1f429e2b0b762):
the default single-API launcher keeps its command line, starts EngineCore, and EngineCore starts
TP workers. `VLLM::APIServer_<index>` belongs to the separate multi-API path and is not supported.

The table, inspector and text label these identifiers as API-reported, with **parent process**
when applicable. JSON has `llm.server_source` (`"process"`, `"parent"`, or `null`) alongside
`llm.server.models[].name`. A reported ID is not a model file or evidence that a worker holds
that model; it adds no weights, KV-cache, context-size or GPU-placement estimate. Measured memory
and attributed totals are unchanged.

GPU-backed WinML and Studio Effects paths have **not been validated on a Copilot+ PC**. A future
hardware check should compare idle versus active workload samples, record the actual GPU adapter
and resource-owning PID (including helpers), and verify the execution provider using runtime
diagnostics. Match any reported model/session to that PID before claiming attribution. If execution
is CPU/NPU-only, record it as out of scope rather than a missing GPU model. No NPU collector is planned.

### How models are identified

Every model row shows its **evidence**, strongest first:

| Evidence | Meaning |
|---|---|
| `api` | The runtime's local API reports the model as loaded |
| `mapped` | Windows shows the file memory-mapped into the process right now |
| `cmdline` | The file was a launch argument (`-m model.gguf`) |
| `installed` | Found in the app's install or data folder; nothing proves it is loaded |
| Handy settings / log | Settings show selection; timestamped log events show what Handy reported, not current residency |

Two sources including `api` or `mapped` show as **confirmed**. Runtime logs can report model events,
but are displayed as timestamped history and do not prove current residency. Model-store folders
only supply a friendly name. The authoritative model *kind* comes from GGUF header keys: `decision.type` marks a
decision model (for example [Laya](https://huggingface.co/ggml-org/Laya-GGUF)), `pooling_type`
an embedding model or reranker, `stt.*` a speech model, and a causal model with layers and a
trained context an LLM. llama-server's `/v1/models` `output_modalities` (`text`, `decisions`)
confirms it. Non-causal encoders such as Laya allocate no KV cache, so none is estimated.

When a file lacks specific kind metadata, known local decision-model names (`openjev`, `laya`,
`julia-1`, `lev`, `kev`) supply a **likely decision model (name)** hint. Matching uses delimited
names in the filename, immediate model folder, or resolved store model name, not arbitrary path
substrings, publishers, or Ollama tags. The process category is `decision?`; specific GGUF kinds
take precedence. Hints do not change loading evidence, causal/KV calculations, or installed-candidate
status. JSON preserves the header-derived `kind` separately from `kind_hint` and `kind_label`.

For example, [ggml-org/OpenJev-GGUF](https://huggingface.co/ggml-org/OpenJev-GGUF) includes
`qwen35.decision.type=openjev`; the inspected community
[openjev/openjev-GGUF](https://huggingface.co/openjev/openjev-GGUF) Q4_K_M header lacks that key and
uses the generic internal name `Snapshot_Cfg`, so its filename or store name supplies only a hint.
Neither a type hint nor an API/file match proves which model produced the process's GPU activity.

Some runtimes (ONNX Runtime, whisper.cpp) read a model into memory and close the file. Windows then
has no record of which file is loaded, so the app's installed models are listed as candidates only.
For Handy, gputaskman reads the configured `selected_model` and only model-load, streaming-start,
preview, and unload events from its local log. Settings prove selection; a fresh log event proves
Handy reported that event after the current process started. Neither proves continuous GPU residency.
Log parsing is bounded to the last 512 KiB, cached until the file changes, and ignores transcription
content. Handy's selected Hugging Face model can live outside its Roaming `models` folder. Handy
logs in UTC; JSON keeps that UTC timestamp and the TUI/text output show local time.

**Other speech and vision apps.** Handy is the first *app adapter*: a read-only recipe for where an
app keeps its settings and which log lines mark a model load or unload. Most whisper-style apps
(whisper.cpp, faster-whisper, sherpa-onnx, ONNX Runtime) read a model and close the file, so the OS
keeps no record of it. Adapters give exact names when an app logs them, but each app needs its own
recipe, and a log-format change can break it. The runtime-agnostic answer is to see the file being
opened during the load: Windows kernel file-I/O ETW events record which process opened which model
file. Combined with that process's VRAM jump, this identifies the model without app-specific code.
It needs elevation and must already be running when the load happens, so it is planned as an
opt-in mode. Until then, unknown apps show their runtime DLLs, VRAM, engine activity, and installed
candidates. Reading model names out of another process's memory is deliberately not done.

Identification is designed to stay cheap. Measured on one Windows machine:

| Probe | Cost | Used |
|---|---|---|
| Process list, command line, listening ports | under 1 ms | Once per process / every 5 s |
| Windows adapter discovery (DXGI + exact-LUID PnP identity + PCI location, no telemetry sample) | ~0.86 ms median, 2.60 ms max in 30 warm reads, 26.1 ms first/cold call (Windows 11 build 28000, RTX Spark N1X, driver 32.0.16.2114); same machine without the PnP probe: ~0.36 ms median | Once at startup; TUI rediscovery every 5 s. PnP uses fixed buffers, one interface lookup per adapter and at most two name-property reads, no path/device enumeration or retries. Synchronous OS calls are not a hard latency bound. Other drivers, virtual displays and multi-GPU laptops unmeasured; provider reconstruction has additional driver-dependent cost |
| Windows NVML handle reacquisition (no telemetry sample) | ~8.7 ms median, 9.3 ms max in 30 reads (Windows 11 build 26340, RTX 4080 SUPER, NVIDIA 630.20) | Every 5 s per active collector; retries initialization after NVML session-loss errors. Other drivers unmeasured |
| Memory-map scan (DLLs + mapped files) | 1-50 ms per process | New processes, a 64 MiB / 10% memory change, or every 2 min; at most 150 ms per refresh |
| GGUF header, model-store manifests | ms, cached | Once per file / every 60 s |
| Local API probes | ms, local only | llama-server `/v1/models` only when its model path changes |
| vLLM `/v1/models` | 1.32 ms median, 21.36 ms max in 30 loopback fixture GETs; cached lookup ~0.0004 ms (Windows 11 build 28000, ARM64; no vLLM runtime/GPU workload) | One GET, 0.5 s timeout and 4 MiB response cap; success cached 3 s, misses/401 cached 30 s. No redirects; actual Linux/vLLM latency unmeasured |
| `psutil.open_files()` | 0.6-2.5 s per process | Never |
| AMD and Intel on Linux: DRM fdinfo scan (`readlink` every fd, read fdinfo of `/dev/dri` ones) | ~9 ms for ~8,000 fds (AMD, Linux desktop; Intel not measured) | Every refresh, only when an amdgpu or Intel card is shown; listing fds is bounded to 50 ms in the TUI and resumes next refresh, known GPU fds are still read |
| Ascend on Linux: `npu-smi` | Contributor reports ~0.5 s per command on an Atlas 300I Duo; not independently measured | Discovery is cached for 120 s. Each selected chip runs one `info` and one `proc-mem` command every 10 s and owns a watch process; multiple selected chips add their synchronous command costs |
| Intel on Linux: performance counters + RAPL uncore (root only) | ~0.04 ms for the counters, ~0.11 ms for RAPL per refresh on the Vivobook (30 reads each) | Every refresh, only when an Intel card is shown and the counters opened; first refresh arms the baselines |
| macOS: `ioreg` device stats + Metal clients | ~10 ms | Every refresh |
| macOS: memory-map scan (`proc_pidinfo` region walk) | ~4 ms per process | Same policy as the memory-map scan above |
| macOS: `footprint` GPU memory | ~70 ms per process, background thread | Busy and new processes every ~2 s, idle ones every 30 s (54 runs in 40 s with 27 GPU processes, M4 Mac mini) |
| System RAM/CPU: process list + psutil totals | Windows (`NtQuerySystemInformation`): ~28 ms median for 995 processes (Windows 11, RTX 4080 SUPER machine, 32 threads); the contributor measured ~6 ms for ~480 processes (Windows 11, i9-13900K). Linux under WSL2 (psutil): ~3.4 ms median for 44 processes, ~9.5 ms for the first sample of 123; the per-pid reuse check (`is_running()`) adds ~0.3 ms for 44 processes, about 7 µs each | Every refresh; `--no-system` turns it off. psutil's per-process `memory_info`/`cpu_times` took ~6 s per pass for 995 processes on Windows, so it is only the non-Windows path. macOS unmeasured |
| `record`: build and write one `--json` line | ~0.4 ms to build the object plus ~0.1 ms to serialize it, median of 30 for a 52-process, 27 KB line (Windows 11, RTX 4080 SUPER machine). A recorded sample is otherwise a normal refresh: 283 ms wall time for the first, ~90 ms from the fifth | Once per `--interval`, only while recording; no extra probes. Other platforms unmeasured |

With two llama-servers running, a steady refresh costs roughly 15-30 ms of CPU (about 340 ms before
process details were cached).

For log-reported buffer sizes, start your own llama-server with logging enabled:

```powershell
llama-server.exe -m .\model.gguf -ngl 99 -c 8192 --metrics --log-file server.log -lv 4
```

Use a real model you already have. This is an example workload, **not a prerequisite for
running gputaskman**. Verbosity 4 was needed to capture the buffer-size lines in the tested
llama.cpp build. Prefer an absolute log path if the server and monitor use different
working directories.

The collector probes existing local listening ports with GET requests to `/props`, `/slots`,
`/metrics`, `/v1/models`, `/api/ps`, `/api/v1/models`, and `/api/v0/models`. Endpoints may be
absent, disabled, or authenticated; other evidence is still useful. It does not launch `ollama`,
`lms`, or an inference request. Throughput may update only after a request completes.

If llama-server was started with `--api-key`, the probe sends that key as a bearer token to the same
server's own port, and never follows a redirect with it. The key is never stored, printed, or written
to JSON, and the inspector shows it as `***`. A key in `--api-key-file` or `LLAMA_API_KEY` is not used:
gputaskman does not open a path taken from another process's command line, and an environment variable
can't be seen from outside the process. In that case the probe stops at the first 401, so the server logs
one rejected request every 30 seconds, and the inspector says why server details are missing.

vLLM uses only `/v1/models`, with the same request bounds and cache intervals. Parent-discovered
ports are **always unauthenticated**, even when parent and worker launch arguments match:
the worker's key is never sent there. A 401 is shown as **Parent-process API locked**.
Probe caches separate own/parent origins, process creation times, runtime and credentials, so
an authenticated own-port result cannot silently satisfy a parent request.

Ollama's local `server.log` is considered when one Ollama runner is detected. Log formats,
rotation, and stale load blocks can affect attribution, so compare the reported model and
process when interpreting logs. Log sizes describe allocations at load time, not necessarily
which pages remain resident now. An inferred remainder is runtime/unknown overhead, not an
exact measurement of the CUDA context.

Without logs, estimates assume llama.cpp-style placement. Architecture, partial offload,
KV quantization, sliding-window attention, and multiple models in one process can all change
the result. Separate runner processes get separate rows; per-model KV accounting within
one process is not yet supported.

### Spot checks against llama.cpp

The following comparisons used llama.cpp b12061, CUDA 13.4, and an RTX 4080 SUPER.
They validate these examples, not every model architecture or backend.

| Allocation | GGUF estimate | Runtime log |
|---|---|---|
| Qwen3-0.6B Q8_0 GPU weights | 604.15 MiB | 604.15 MiB |
| Qwen3-0.6B KV, 8,192 context, f16 | 896 MiB | 896 MiB |
| Gemma-3-270M GPU weights | 271.81 MiB | 271.86 MiB |
| Gemma-3-270M KV, 4,096 context | 72 MiB upper bound | 49.5 MiB with sliding-window attention |

## Privacy and troubleshooting

The monitor reads local process metadata, compatible model headers, load logs, and local server
responses. It does not dump model weights or send telemetry. **The normal inspector and JSON
output can contain paths, command-line flags, model names, and local ports. Review and redact
captures before posting them publicly.** The screenshot helper omits command lines and full
file paths, but model and process names still require review.

| Symptom | What to check |
|---|---|
| `uv` or `git` is not recognized | Install the prerequisites, then open a new terminal |
| Empty or incomplete GPU readings | Check the GPU driver and WDDM 2.0+ support in `dxdiag`; use a normal local desktop session |
| "supports Windows and Linux/WSL only" | macOS is not supported |
| WSL: `Windows host counters: off (WSL interop is disabled)` | Enable `[interop]` in `/etc/wsl.conf`, then `wsl --shutdown`; or accept device-wide numbers |
| WSL: `starting` for a few seconds | The first Windows counter sample takes about 2.5 s |
| WSL: `unmatched` | No Windows adapter's usage matched the NVML device; multi-GPU WSL is not validated |
| WSL: Linux rows show `?` | Processes were running before gputaskman started; they become `*` with evidence (see the matching table) |
| Linux: a GPU process is missing | Processes owned by other users need `sudo`; Docker and other WSL distros are not visible from this one |
| Linux container: `H:<pid>` rows | GPU processes outside this container's PID namespace (host or other containers); NVML sees them, the container cannot name them |
| Missing DLLs or model details | Protected processes may deny inspection; basic counters can still work without elevation |
| No exact LLM breakdown | Enable compatible buffer logging, or use the GGUF estimate; API/log access is optional |
| The screen looks cramped | Widen the terminal or use `--once` / `--json` |

## Development

```powershell
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
uv build
```

CI runs hardware-independent checks on Windows, Ubuntu and macOS, Python 3.11 and 3.13. Live GPU behavior
still needs real hardware; the CI does not pretend to validate that. The WSL matching tests replay
context sequences measured on a real WSL2 machine.

Contributors and coding agents: [AGENTS.md](AGENTS.md) lists the evidence rules every change must
keep, a module map, verification steps, and the release process.

### Building portable releases

Use Python 3.13 of the **target architecture**. PyInstaller does not cross-compile: an x64
Python process makes an x64 executable even when running under ARM64 emulation.

```powershell
uv sync --locked --group release --python 3.13
uv run --locked --group release python tools\build_portable.py --arch x64
# On a native ARM64 interpreter, use --arch arm64 instead.
```

The same command on Linux (`tools/build_portable.py`) produces a `.tar.gz`, which keeps the
executable bit. The builder refuses to package NVIDIA driver libraries (`libnvidia-ml.so`,
`libcuda.so`) on either platform; they come from the installed driver.

The builder verifies the interpreter and executable architectures, includes Textual resources
and dependency licenses, and writes the complete portable ZIP and SHA-256 file to `dist`.
It does not bundle Windows or GPU driver DLLs, inference servers, or models. Keep uv for development;
the executable ZIP is for running the monitor without a Python environment.

The **Portable release** GitHub Actions workflow builds on `windows-2022` (x64), `windows-11-arm`
(ARM64), `ubuntu-22.04` (Linux x64), and `ubuntu-22.04-arm` (Linux ARM64). Linux builds use the
oldest supported Ubuntu because the frozen binary needs the build machine's glibc or newer. A manual run produces downloadable workflow artifacts without publishing
a release. A pushed `vVERSION` tag must match `project.version` in `pyproject.toml`; once all builds
pass, the workflow publishes all four archives and checksums together with generated release notes.
The workflow also builds a Python wheel and source distribution. On tags, after all builds
succeed, a separate job publishes them to PyPI using Trusted Publishing (GitHub OIDC), with no
stored API token. Manual runs produce artifacts only and publish neither a GitHub release nor
a PyPI package.

Before the first PyPI publication, configure a pending publisher for project `gputaskman` on
PyPI: owner `shanselman`, repository `gputaskman`, workflow `release.yml`, environment `pypi`.
The workflow uses that exact GitHub environment; any environment approval rules must be satisfied
before publishing. The first successful tagged publication creates the PyPI project. Once
published, install with `uv tool install gputaskman` or `pipx install gputaskman` (Python 3.11+).

Build jobs have read-only permissions; only the tag-triggered GitHub release job can write releases,
and only the PyPI publishing job has `id-token: write`.
CI checks the extracted executable's `--help` outside the checkout with Python removed from `PATH`;
interactive rendering and real GPU snapshots require local hardware testing.

To capture native SVG screenshots of your own running workloads:

```powershell
uv run --locked python tools\capture_screenshots.py --output .\captures
```

The checked-in gallery uses PNG renderings of these captures. No models, logs, or executable
downloads are bundled in this repository.

| Module | Responsibility |
|---|---|
| `pdh.py` / `dxgi.py` | Windows counters and adapter enumeration |
| `linux.py` | Linux backend: NVML adapters and per-process usage, container PID matching, WSL detection, `/dev/dxg` holders |
| `drm.py` | AMD and Intel on native Linux: amdgpu and i915 cards and device readings from sysfs, per-process VRAM/GTT/engine time from DRM fdinfo |
| `wslhost.py` | WSL: interop check, Windows host counter stream, adapter pairing, context matching |
| `macos.py` | macOS backend: IOKit device stats and per-process GPU time, `footprint` GPU memory poller, libproc memory maps |
| `nvml.py` | NVIDIA device readings |
| `adl.py`, `amdsmi.py` | Optional AMD device readings |
| `procinfo.py` | Runtime, mapped-file, and process discovery |
| `stores.py` | Friendly names from Ollama, LM Studio, and Hugging Face model stores |
| `gguf.py` | Model metadata, kind, and size estimates |
| `llm.py` | Buffer logs, local server probing, and breakdowns |
| `host.py` | System RAM and CPU as the OS reports them: psutil totals, per-process RAM and CPU%, and the Windows process-list reader |
| `collector.py` | Shared snapshot collection |
| `app.py` / `report.py` / `cli.py` | TUI, text/JSON, and command-line entry points |
| `diff.py` | Before/after comparison of two `--json` snapshots |
| `record.py` | `gputaskman record`: appends one `--json` snapshot per interval to a JSON Lines file |

## Next

- Model identity for non-GGUF runtimes (ONNX Runtime, PyTorch, TensorRT) beyond installed candidates.
- Explicit provenance and availability for every counter and estimate.
- Reliable multi-GPU matching and per-device model placement.
- Intel `xe` per-process memory and load (`drm-cycles-*`), the Intel GT clock in the device panel, and Intel
  validation beyond the Arc A770 / UHD 730 pair.
- macOS: a portable build, Ollama/llama.cpp/MLX validation, and compressed GPU memory (footprint's
  `swapped`) as the paged column.
- WSL: per-process GPU load for Linux rows, and Ollama's systemd journal as a log source.
- Live throughput deltas, LM Studio validation, and per-model context accounting.
- Replay and summaries of a `gputaskman record` file; today any one line works with `diff`.
- Opt-in ETW file-open tracing to name models loaded by read-and-close runtimes (any whisper or ONNX
  app), correlated with the process's VRAM jump at load time.
- Data-driven app adapters (settings path plus load/unload log patterns) once a second app needs one.
- Optional ETW / runtime instrumentation for allocation lifetime and residency detail.

## License

[MIT](LICENSE).
