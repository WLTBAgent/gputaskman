"""Per-process introspection: which GPU runtimes are loaded, which model files are mapped."""

from __future__ import annotations

import fnmatch
import os
import re
import sys
import time
from dataclasses import dataclass, field

import psutil

from .gguf import is_gguf, read_gguf
from .stores import store_name

# (tag, library patterns: Windows DLLs, Linux shared objects with the version suffix removed, and macOS
# dylibs, see _lib_name). Order = display priority (most specific first).
RUNTIMES: list[tuple[str, tuple[str, ...]]] = [
    ("llama.cpp", ("llama.dll", "libllama*.dll", "llama-*.dll", "llm_engine*", "libllama*.so", "libllama*.dylib")),
    ("whisper.cpp", ("whisper*.dll", "libwhisper*.so", "libwhisper*.dylib")),
    ("ggml", ("ggml*.dll", "libggml*.so", "libggml*.dylib")),
    ("PyTorch", ("torch_cuda*.dll", "torch_cpu.dll", "libtorch_cuda*.so", "libtorch_cpu.so", "libtorch_cpu.dylib")),
    ("MLX", ("libmlx*.dylib",)),
    ("TensorRT", ("nvinfer*.dll", "libnvinfer*.so")),
    ("ONNX RT", ("onnxruntime*.dll", "libonnxruntime*.so", "libonnxruntime*.dylib")),
    ("DirectML", ("directml.dll",)),
    ("HIP", ("amdhip64.dll", "amdhip64_*.dll", "libamdhip64.so")),
    ("cuDNN", ("cudnn*.dll", "libcudnn*.so")),
    ("CUDA", ("nvcuda.dll", "nvcuda64.dll", "cudart64_*.dll", "cublas64_*.dll",
              "libcuda.so", "libcudart.so", "libcublas.so")),
    ("Metal", ("agxmetal*",)),  # the Apple GPU's Metal driver bundle, mapped by every Metal process
    ("Vulkan", ("vulkan-1.dll", "libvulkan.so", "libmoltenvk.dylib")),
    ("D3D12", ("d3d12core.dll", "libd3d12core.so")),
    ("OpenGL", ("opengl32.dll", "libgl.so", "libglx*.so", "libegl*.so")),
    ("D3D11", ("d3d11.dll",)),
    ("NVENC", ("nvencodeapi64.dll", "libnvidia-encode.so")),
    ("NVDEC", ("nvcuvid.dll", "nvcuvid64.dll", "libnvcuvid.so")),
]
_SO = re.compile(r"^(.+\.so)(?:\.[0-9][0-9a-z.]*)?$")
AI_RUNTIMES = {"llama.cpp", "whisper.cpp", "ggml", "PyTorch", "MLX", "TensorRT", "ONNX RT", "DirectML", "cuDNN"}
_RUNTIME_RX = [(tag, re.compile("|".join(fnmatch.translate(p) for p in pats))) for tag, pats in RUNTIMES]
MODEL_EXTS = {".gguf", ".ggml", ".safetensors", ".onnx", ".pt", ".pth", ".bin", ".engine", ".plan", ".tflite", ".ckpt"}
AMBIGUOUS_EXTS = {".bin", ".pt", ".pth", ".ckpt"}  # also used for caches/tiles, so require size or a model store
_NOT_MODELS = {".dll", ".exe", ".sys", ".mui", ".ttf", ".ttc", ".otf", ".dat", ".db", ".nls", ".pak", ".mun"}
BROWSERS = {"msedge.exe", "chrome.exe", "firefox.exe", "msedgewebview2.exe", "brave.exe", "opera.exe", "arc.exe", "vivaldi.exe",
            "chrome", "chromium", "chromium-browser", "firefox", "firefox-bin", "brave", "msedge", "opera", "vivaldi-bin",
            # macOS: browsers draw through a GPU helper process
            "google chrome helper (gpu)", "microsoft edge helper (gpu)", "brave browser helper (gpu)",
            "arc helper (gpu)", "chromium helper (gpu)", "com.apple.webkit.gpu", "safari", "firefox gpu helper"}
# Linux display servers / compositors: they own the desktop back buffers, like dwm.exe on Windows.
COMPOSITORS = {"dwm.exe", "xorg", "xwayland", "gnome-shell", "kwin_wayland", "kwin_x11", "sway", "hyprland",
               "mutter", "weston", "wayfire", "labwc", "niri", "cosmic-comp", "picom", "compiz", "windowserver"}
# Paths in /proc/<pid>/maps that are not regular files: never stat or open them.
_SPECIAL_PREFIXES = ("[", "/dev/", "/proc/", "/sys/", "/memfd:", "/SYSV", "anon_inode:")
# macOS: other apps' sandbox containers are privacy-protected, and open() on them blocks waiting for a consent
# prompt (seen with mediaanalysisd's caches over SSH). System files are never models: skip both unopened.
_MAC_SKIP_PREFIXES = ("/System/", "/usr/lib/", "/usr/share/", "/private/var/db/", "/Library/Apple/")
_MAC_SKIP_PARTS = ("/Library/Containers/", "/Library/Group Containers/")


def _skip_path(path: str) -> bool:
    if path.startswith(_SPECIAL_PREFIXES) or path.endswith(" (deleted)"):
        return True
    return sys.platform == "darwin" and (path.startswith(_MAC_SKIP_PREFIXES) or any(p in path for p in _MAC_SKIP_PARTS))
LLM_NAMES = ("llama-", "llama.", "ollama", "lm studio", "lmstudio", "llmster", "koboldcpp", "jan", "gpt4all", "llamafile")
# GGUF file kind -> process kind, in display priority order
MODEL_KINDS = {"LLM": "LLM", "decision model": "decision", "embedding model": "embedding",
               "reranker": "embedding", "speech model": "speech"}
AI_KINDS = {"LLM", "decision", "decision?", "embedding", "speech", "diffusion", "AI"}
_DECISION_NAME = re.compile(r"(?<![a-z0-9])(?:openjev|laya|julia-1|lev|kev)(?![a-z0-9])", re.I)
_VLLM_WORKER = re.compile(r"VLLM::(?:EngineCore|Worker(?:_TP[0-9]+)?)")
MiB = 1 << 20
INSTALLED_TTL = 300.0
_SKIP_DIRS = {"ebwebview", "cache", "code cache", "gpucache", "node_modules", "logs", "crashpad", "locales", "__pycache__"}
_installed: dict[str, tuple[float, list[MappedModel]]] = {}


@dataclass
class MappedModel:
    path: str
    size: int
    resident: int  # bytes of the file currently in this process's RAM working set
    gguf: bool
    sources: list[str] = field(default_factory=lambda: ["mapped"])  # mapped | cmdline | api | installed
    store_name: str | None = None  # e.g. "qwen3:14b" from an Ollama manifest
    kind: str = ""  # GGUF kind: LLM, decision model, embedding model, ...
    kind_hint: str = ""  # name-based suggestion only; never changes loading evidence or GGUF math

    @property
    def kind_label(self) -> str:
        return f"likely {self.kind_hint} (name)" if self.kind_hint else self.kind

    def add_source(self, source: str) -> None:
        if source not in self.sources:
            self.sources.append(source)

    @property
    def evidence(self) -> str:
        s = set(self.sources)
        how = "+".join(x for x in ("api", "mapped", "cmdline") if x in s)
        if len(s) >= 2 and s & {"api", "mapped"}:
            return f"confirmed ({how})"
        if not how and "installed" in s:
            return "installed, not proven loaded"
        return {"mapped": "in use (mapped)", "cmdline": "launch arg", "api": "reported (api)"}.get(how, how)


def _is_diffusion_cmdline(cmd: list[str]) -> bool:
    return any(arg == "--diffusion-model" or arg.startswith("--diffusion-model=") for arg in cmd)


@dataclass
class ProcDetail:
    pid: int
    name: str = "?"
    exe: str = ""
    cmdline: list[str] = field(default_factory=list)
    create_time: float = 0.0
    runtimes: list[str] = field(default_factory=list)
    models: list[MappedModel] = field(default_factory=list)
    candidates: list[MappedModel] = field(default_factory=list)  # installed with the app, not proven loaded
    ports: list[int] = field(default_factory=list)
    parents: list[int] | None = None  # parent PIDs, looked up once (each lookup enumerates all processes)
    access_denied: bool = False
    maps_time: float = 0.0  # when memory maps were last scanned
    scan_mem: float = 0.0  # process GPU memory (dedicated + shared) at that scan

    @property
    def is_ai(self) -> bool:
        return bool(self.models) or any(r in AI_RUNTIMES for r in self.runtimes)

    @property
    def is_diffusion(self) -> bool:
        return _is_diffusion_cmdline(self.cmdline)

    @property
    def vllm_role(self) -> str:
        """A process signature, not evidence that a model is resident in this process."""
        if any(_VLLM_WORKER.fullmatch(s) for s in (self.name, *self.cmdline[:1])):
            return "worker"
        cmd = self.cmdline
        names = [arg.replace("\\", "/").rsplit("/", 1)[-1] for arg in cmd[:2]]
        if len(cmd) >= 2 and names[0] in ("vllm", "vllm.exe") and cmd[1] == "serve":
            return "server"
        if names and re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?(?:\.exe)?", names[0]):
            if len(cmd) >= 3 and names[1] in ("vllm", "vllm.exe") and cmd[2] == "serve":
                return "server"
            args = cmd[1:]
            while args and args[0] in ("-u", "-B", "-E", "-s", "-S", "-I", "-O", "-OO"):
                args = args[1:]
            if args[:2] == ["-m", "vllm.entrypoints.openai.api_server"]:
                return "server"
        return ""

    @property
    def vllm_ambiguous(self) -> bool:
        # Distributed/data-parallel layouts need an ownership graph, not just two ancestors.
        # Explicit single-node values are allowed. A --config path belongs to another process and is
        # never opened, so any layout it could hide stays ambiguous. Aliases: vllm/engine/arg_utils.py.
        single = {"--data-parallel-size": "1", "-dp": "1", "--api-server-count": "1", "--nnodes": "1",
                  "-n": "1", "--distributed-executor-backend": "mp"}
        watched = ("--config", "--headless", "--node-rank", *(f for f in single if f.startswith("--")))
        cmd = self.cmdline
        for i, arg in enumerate(cmd):
            flag, separator, value = arg.partition("=")
            if flag.startswith("--"):
                flag = flag.replace("_", "-")  # vLLM's parser accepts --data_parallel_size
                if len(flag) > 2 and any(name != flag and name.startswith(flag) for name in watched):
                    return True  # argparse accepts unambiguous prefixes such as --api-server
            if (flag in ("--config", "--headless", "--node-rank")
                    or (flag.startswith("--data-parallel") and flag != "--data-parallel-size")
                    or (flag.startswith("-dp") and flag != "-dp")
                    or (flag.startswith("-n") and flag != "-n")
                    or flag.startswith("-r")):
                return True
            if flag in single:
                if not separator:
                    value = cmd[i + 1] if i + 1 < len(cmd) else ""
                if value != single[flag]:
                    return True
        return False


def _lib_name(base: str) -> str | None:
    """Lower-cased library file name with any Linux version suffix removed ('libcudart.so.12.4' ->
    'libcudart.so'), or None when this is not a library."""
    ext = os.path.splitext(base)[1]
    if ext in (".dll", ".node", ".dylib") or base.startswith("agxmetal"):
        return base
    m = _SO.match(base)
    return m[1] if m else None


def _match_runtimes(dll_names: set[str]) -> list[str]:
    return [tag for tag, rx in _RUNTIME_RX if any(rx.match(d) for d in dll_names)]


def file_key(path: str) -> tuple[int, int] | str:
    """Identity of the file itself, so one model reached by two spellings (extended-length prefix, link,
    case) is one model. Falls back to the normalized path when the file can't be read or has no file id."""
    try:
        st = os.stat(path)
    except (OSError, ValueError):
        return os.path.normcase(os.path.abspath(path))
    return (st.st_dev, st.st_ino) if st.st_ino else os.path.normcase(os.path.abspath(path))


def _models_from_cmdline(cmd: list[str]) -> list[str]:
    paths = []
    flags = {"-m", "--model", "--mmproj", "-md", "--model-draft", "--lora"}
    if _is_diffusion_cmdline(cmd):
        flags.update(("--diffusion-model", "--vae", "--llm"))
    for i, arg in enumerate(cmd):
        val = None
        key, sep, value = arg.partition("=")
        if key in flags:
            val = value if sep else (cmd[i + 1] if i + 1 < len(cmd) else None)
        if val and os.path.isfile(val):
            paths.append(os.path.abspath(val))
    return paths


def _is_model_file(ext: str, size: int, gg: bool, store: str | None) -> bool:
    if gg:
        return True
    if ext not in MODEL_EXTS:
        return False
    if ext in AMBIGUOUS_EXTS:
        return size >= 64 * MiB or store is not None
    return size >= 4 * MiB


def make_model(path: str, size: int, resident: int, gg: bool, source: str, store: str | None = None) -> MappedModel:
    g = read_gguf(path) if gg else None
    kind = g.kind if g is not None and not g.error else ""
    store = store if store is not None else store_name(path)
    hint = _name_kind_hint(path, store) if kind in ("", "model", "LLM") else ""
    return MappedModel(path, size, resident, gg, [source], store, kind, hint)


def _name_kind_hint(path: str, store: str | None) -> str:
    names = [os.path.basename(path), os.path.basename(os.path.dirname(path))]
    # Match the model name, not a publisher, user directory, or Ollama tag.
    names += [name.rsplit("/", 1)[-1].split(":", 1)[0] for name in (store or "").split(", ")]
    return "decision model" if any(_DECISION_NAME.search(name) for name in names) else ""


def _walk(root: str, depth: int, out: list[tuple[str, int]], budget: list[int]) -> None:
    try:
        it = os.scandir(root)
    except OSError:
        return
    with it:
        for e in it:
            budget[0] -= 1
            if budget[0] < 0:
                return
            try:
                if e.is_dir(follow_symlinks=False):
                    if depth > 0 and e.name.lower() not in _SKIP_DIRS:
                        _walk(e.path, depth - 1, out, budget)
                    continue
                ext = os.path.splitext(e.name)[1].lower()
                if ext in MODEL_EXTS:
                    size = e.stat().st_size
                    if _is_model_file(ext, size, False, None):
                        out.append((e.path, size))
            except OSError:
                continue


_LINUX_SYSTEM_DIRS = ("/usr/bin/", "/usr/sbin/", "/bin/", "/sbin/", "/usr/lib", "/lib", "/usr/libexec/", "/snap/")


def _is_system_path(exe: str) -> bool:
    if sys.platform == "win32":
        return exe.startswith(os.path.normcase(os.environ.get("SystemRoot", r"C:\Windows")))
    # /usr/bin/python3 is shared by every Python app: walking /usr/bin finds nothing about the process.
    return exe.startswith(_LINUX_SYSTEM_DIRS) or (sys.platform == "darwin" and exe.startswith("/System/"))


def _data_dirs() -> list[str]:
    if sys.platform == "win32":
        return [d for d in (os.environ.get("APPDATA"), os.environ.get("LOCALAPPDATA")) if d]
    home = os.path.expanduser("~")
    if sys.platform == "darwin":
        return [os.path.join(home, "Library", "Application Support")]
    return [os.environ.get("XDG_DATA_HOME") or os.path.join(home, ".local", "share"),
            os.environ.get("XDG_CONFIG_HOME") or os.path.join(home, ".config")]


def installed_models(exe: str, now: float | None = None) -> list[MappedModel]:
    """Model files in an app's install and data folders. Fallback for runtimes that read a model into RAM and
    close the file (ONNX Runtime, whisper.cpp), which leaves no OS evidence of which model is loaded.
    Bounded walk (depth 4, 3000 entries) cached per exe for 5 minutes."""
    if not exe:
        return []
    now = now or time.time()
    key = os.path.normcase(exe)
    hit = _installed.get(key)
    if hit and now - hit[0] < INSTALLED_TTL:
        return hit[1]
    stem = os.path.splitext(os.path.basename(key))[0].lower()  # normcase only lowercases on Windows
    roots = []
    if not _is_system_path(key):
        roots.append(os.path.dirname(exe))
    if len(stem) >= 4:
        for base in _data_dirs():
            try:
                roots += [e.path for e in os.scandir(base) if stem in e.name.lower() and e.is_dir()]
            except OSError:
                pass
    files: list[tuple[str, int]] = []
    budget = [3000]
    for root in {os.path.normcase(r): r for r in roots}.values():
        _walk(root, 4, files, budget)
    by_dir: dict[str, list[tuple[str, int]]] = {}
    for path, size in files:
        by_dir.setdefault(os.path.dirname(path), []).append((path, size))
    found = []
    for folder, group in by_dir.items():
        if len(group) == 1:  # one file: name it by the file
            path, size = group[0]
            found.append(make_model(path, size, 0, path.lower().endswith(".gguf") and is_gguf(path), "installed"))
        else:  # several files (encoder + decoder, shards): name it by the folder
            found.append(make_model(folder, sum(s for _, s in group), 0, False, "installed", store_name(group[0][0])))
    found = sorted(found, key=lambda m: -m.size)[:6]
    _installed[key] = (now, found)
    return found


def _create_time(pid: int) -> float:
    try:
        return psutil.Process(pid).create_time()
    except (psutil.Error, OSError):
        return 0.0


class ProcInspector:
    """Caches per-process details. Memory-map scans are the only costly step (1-50 ms each), so they run
    for new processes, when a process's GPU memory changes noticeably, and on a slow backstop."""

    SCAN_MIN_GAP = 5.0  # never rescan one process more often than this
    SCAN_BACKSTOP = 120.0  # rescan even without a memory change
    SCAN_DELTA = 64 * MiB  # or 10% of the process's GPU memory, whichever is larger
    PORTS_INTERVAL = 5.0

    def __init__(self) -> None:
        self._cache: dict[int, ProcDetail] = {}
        self._ports: dict[int, list[int]] = {}
        self._ports_time = 0.0

    def _refresh_ports(self) -> None:
        if time.time() - self._ports_time < self.PORTS_INTERVAL:
            return
        ports: dict[int, set[int]] = {}
        try:
            for c in psutil.net_connections(kind="tcp"):
                if c.status == psutil.CONN_LISTEN and c.pid and c.laddr:
                    ports.setdefault(c.pid, set()).add(c.laddr.port)
        except (psutil.AccessDenied, OSError):
            pass
        self._ports = {pid: sorted(p) for pid, p in ports.items()}
        self._ports_time = time.time()

    def get(self, pid: int) -> ProcDetail:
        """Cheap details: name, exe, command line, listening ports. Built once per process: psutil.Process()
        enumerates every process for protected PIDs (System, csrss, dwm), ~40 ms each. Accessible processes
        are re-checked for PID reuse each call (~0.1 ms); protected ones leave through prune()."""
        self._refresh_ports()
        d = self._cache.get(pid)
        if d is not None and d.create_time and _create_time(pid) != d.create_time:
            d = None  # PID reused (e.g. while the TUI was paused)
        if d is None:
            d = self._cache[pid] = self._new(pid)
        d.ports = self._ports.get(pid, [])
        return d

    def ports(self, pid: int) -> list[int]:
        self._refresh_ports()
        return self._ports.get(pid, [])

    def exclusive_ports(self, pid: int) -> list[int]:
        """Only probe ports with one known listener; localhost may route shared ports to another PID."""
        self._refresh_ports()
        others = {port for other, ports in self._ports.items() if other != pid for port in ports}
        return [port for port in self._ports.get(pid, []) if port not in others]

    @staticmethod
    def _new(pid: int) -> ProcDetail:
        d = ProcDetail(pid=pid)
        if pid == 0:
            d.name = "System Idle"
            return d
        try:
            p = psutil.Process(pid)
        except (psutil.Error, OSError):
            d.access_denied = True
            return d
        try:
            d.name = p.name()
        except (psutil.Error, OSError):
            pass
        try:
            d.exe = p.exe()
            d.cmdline = p.cmdline()
            d.create_time = p.create_time()
        except (psutil.Error, OSError):
            d.access_denied = True
        return d

    def needs_scan(self, d: ProcDetail, mem: float, now: float | None = None) -> bool:
        if d.access_denied or d.pid == 0:
            return False
        if not d.maps_time:
            return True
        age = (now or time.time()) - d.maps_time
        if age < self.SCAN_MIN_GAP:
            return False
        return age > self.SCAN_BACKSTOP or abs(mem - d.scan_mem) > max(self.SCAN_DELTA, 0.1 * d.scan_mem)

    def scan(self, d: ProcDetail, mem: float = 0.0) -> None:
        """Memory maps -> loaded DLLs (runtimes) and mapped model files. Never uses open_files(): ~1-2 s/process."""
        d.maps_time, d.scan_mem = time.time(), mem
        try:
            maps = _memory_maps(d.pid)
        except (psutil.Error, OSError):
            d.access_denied = True
            return
        dlls: set[str] = set()
        models: dict[str, MappedModel] = {}
        for m in maps:
            path = m.path
            base = os.path.basename(path).lower()
            lib = _lib_name(base)
            if lib and base.startswith("agxmetal"):
                dlls.add(lib)  # the Metal driver lives under /System: record it before skipping system paths
                continue
            if _skip_path(path):
                continue
            ext = os.path.splitext(base)[1]
            if lib:
                dlls.add(lib)
                continue
            if ext in _NOT_MODELS:
                continue
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            gg = is_gguf(path) if (ext in (".gguf", "") or size > 64 * MiB) else False
            if not (gg or ext in MODEL_EXTS):
                continue
            store = store_name(path)
            if _is_model_file(ext, size, gg, store):
                models[file_key(path)] = make_model(path, size, m.rss, gg, "mapped", store)
        for path in _models_from_cmdline(d.cmdline):
            key = file_key(path)
            if key in models:
                models[key].add_source("cmdline")
            else:
                models[key] = make_model(path, os.path.getsize(path), 0, is_gguf(path), "cmdline")
        d.runtimes = _match_runtimes(dlls)
        d.models = sorted(models.values(), key=lambda m: -m.size)
        ai = any(r in AI_RUNTIMES for r in d.runtimes)
        d.candidates = installed_models(d.exe) if ai and not d.models else []

    def prune(self, live_pids: set[int]) -> None:
        for pid in list(self._cache):
            if pid not in live_pids:
                del self._cache[pid]


def _memory_maps(pid: int) -> list:
    if sys.platform == "darwin":  # psutil has no memory_maps() on macOS
        from .macos import memory_maps
        return memory_maps(pid)
    return psutil.Process(pid).memory_maps(grouped=True)


def classify(d: ProcDetail, engines: dict[str, float]) -> str:
    n = d.name.lower()
    if d.pid == 4 and sys.platform == "win32":  # "System"; PID 4 is an ordinary process on Linux
        return "kernel"
    if n in COMPOSITORS:
        return "compositor"
    if n in ("csrss.exe", "winlogon.exe", "lsass.exe", "svchost.exe"):
        return "system"
    if d.is_diffusion:
        return "diffusion"
    kinds = {MODEL_KINDS[m.kind] for m in d.models if m.kind in MODEL_KINDS and not m.kind_hint}
    for k in ("LLM", "decision", "embedding", "speech"):
        if k in kinds:
            return k
    if any(m.kind_hint == "decision model" for m in d.models):
        return "decision?"
    if d.vllm_role or "llama.cpp" in d.runtimes or n.startswith(LLM_NAMES):
        return "LLM"
    video = engines.get("decode", 0) + engines.get("encode", 0) + engines.get("video", 0) > 1
    if d.is_ai and not video:
        return "AI"
    if n in BROWSERS:
        return "browser"
    if video or "NVENC" in d.runtimes:
        return "video"
    if "CUDA" in d.runtimes or "HIP" in d.runtimes:
        return "compute"
    if ("D3D12" in d.runtimes or "Vulkan" in d.runtimes) and engines.get("3d", 0) > 10:
        return "3D/game"
    if "Metal" in d.runtimes and engines.get("gpu", 0) > 10 and n not in COMPOSITORS:
        return "3D/game"  # macOS: Metal reports one GPU time, no 3D/compute split
    return "app"
