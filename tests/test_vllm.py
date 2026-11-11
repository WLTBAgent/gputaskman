from __future__ import annotations

import contextlib
import http.server
import json
import os
import subprocess
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import psutil
from textual.widgets import Static

from gputaskman.app import GpuTaskManager
from gputaskman.collector import Collector
from gputaskman.llm import ServerProber, probe_server, vllm_models
from gputaskman.pdh import AdapterGpu, ProcGpu
from gputaskman.procinfo import ProcDetail, ProcInspector, classify
from gputaskman.report import to_dict, to_text

MiB = 1 << 20
LAUNCH = ["vllm", "serve", "org/model"]
# Title/layout fixtures follow vllm-project/vllm at 499a3d1e1e338e3e866c0719cfe1f429e2b0b762:
# utils/system_utils.py:set_process_title, v1/executor/multiproc_executor.py:setup_proc_title_and_log_prefix,
# entrypoints/cli/serve.py:cmd, and entrypoints/launchers/api_server/entry.py:run_server.


@contextlib.contextmanager
def listener(*, locked=False, redirect=False):
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            seen.append((self.path, self.headers.get("Authorization")))
            body = json.dumps({"data": [{"id": "org/model", "root": "/not/a/model/file",
                                         "max_model_len": 8192, "size": 999999}]}).encode()
            status = 401 if locked and not self.headers.get("Authorization") else 302 if redirect else 200
            self.send_response(status)
            self.send_header("Content-Length", str(len(body) if status == 200 else 0))
            if redirect:
                self.send_header("Location", "/elsewhere")
            self.end_headers()
            if status == 200:
                self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield server.server_address[1], seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def collector(details, ports):
    c = Collector.__new__(Collector)
    c.adapter, c.adapters = None, []
    c.gpu_index, c.scan_budget, c.host, c.host_status = 0, None, None, ""
    c.nvml = Mock(ok=False)
    c.counters = Mock(outside={})
    c.counters.sample.return_value = (
        {0: {10: ProcGpu(dedicated=512 * MiB, shared=0, engines={"compute": 10})}},
        {0: AdapterGpu(dedicated=512 * MiB, shared=0)})
    c.inspector = ProcInspector()
    c.inspector._cache = details
    c.inspector._ports = ports
    c.inspector._ports_time = float("inf")
    c.inspector.needs_scan = Mock(return_value=False)
    c.inspector.prune = Mock()
    c.prober = ServerProber()
    return c


def chain(details, parents):
    nodes = {pid: Mock(pid=pid) for pid in details}
    for pid, node in nodes.items():
        node.create_time.return_value = details[pid].create_time
        node.parent.return_value = nodes.get(parents.get(pid))
    return nodes


class VllmTests(unittest.TestCase):
    def test_vllm_topology_flags_are_value_aware(self):
        # vllm/engine/arg_utils.py aliases (-dp, -n, -r, -tp, -pp); vllm/utils/argparse_utils.py
        # FlexibleArgumentParser maps underscores to dashes and keeps argparse prefix matching.
        allowed = [
            [],
            ["--data-parallel-size", "1"], ["--data-parallel-size=1"], ["-dp", "1"], ["-dp=1"],
            ["--api-server-count", "1"], ["--api-server-count=1"],
            ["--distributed-executor-backend", "mp"], ["--distributed-executor-backend=mp"],
            ["--nnodes", "1"], ["--nnodes=1"], ["-n", "1"], ["-n=1"],
            ["--data_parallel_size", "1"], ["--api_server_count=1"], ["--distributed_executor_backend", "mp"],
            ["--tensor-parallel-size", "2"], ["-tp", "2"],
            ["--pipeline-parallel-size", "1"], ["--pipeline-parallel-size=1"], ["-pp", "1"], ["-pp=1"],
            ["--pipeline_parallel_size=1"],
            ["--data-parallel-size=1", "--api-server-count=1", "--distributed-executor-backend=mp"],
        ]
        rejected = [
            ["--data-parallel-size", "2"], ["--data-parallel-size=2"], ["-dp", "2"], ["-dp=2"],
            ["-dp2"], ["-dpn", "0"], ["-dpl", "2"],
            ["--data-parallel-rank", "0"], ["--data-parallel-size-local", "1"],
            ["--data-parallel-external-lb"],
            ["--api-server-count", "2"], ["--api-server-count=0"],
            ["--distributed-executor-backend", "ray"],
            ["--distributed-executor-backend=external_launcher"],
            ["--nnodes", "2"], ["--nnodes=2"], ["-n", "2"], ["-n2"],
            ["--node-rank", "0"], ["--node-rank=0"], ["-r", "0"], ["-r0"],
            ["--headless"], ["--config", "launch.yaml"], ["--config=launch.yaml"],
            ["--data_parallel_size", "2"], ["--api_server_count=2"], ["--distributed_executor_backend", "ray"],
            ["--node_rank", "0"], ["--data_parallel_rank=0"],
            ["--api-server", "2"], ["--nnode", "2"], ["--distributed-executor", "ray"], ["--head"],
            ["--conf", "launch.yaml"], ["--node-r", "0"],
            ["--pipeline-parallel-size", "2"], ["--pipeline-parallel-size=2"],
            ["-pp", "2"], ["-pp=2"], ["-pp2"],
            ["--pipeline-parallel", "1"], ["--pipeline-parallel=1"],
            ["--data-parallel-size"], ["-dp"], ["--api-server-count"],
            ["--distributed-executor-backend"], ["--nnodes"], ["--pipeline-parallel-size"],
            ["--data-parallel-size", "--api-server-count", "1"],
            ["--data-parallel-size=1", "--config", "launch.yaml"],
            ["-dp", "2", "--api-server-count", "1"],
        ]
        for args in allowed:
            with self.subTest(allowed=args):
                self.assertFalse(ProcDetail(10, "python", cmdline=LAUNCH + args).vllm_ambiguous)
        for args in rejected:
            with self.subTest(rejected=args):
                self.assertTrue(ProcDetail(10, "python", cmdline=LAUNCH + args).vllm_ambiguous)

    def test_signatures_are_hints_even_without_mapped_models_or_titles(self):
        cases = [
            ("vllm", LAUNCH, "server"),
            ("python3", ["/env/bin/python3", "/env/bin/vllm", "serve", "org/model"], "server"),
            ("python", ["python", "-u", "-m", "vllm.entrypoints.openai.api_server", "--model", "org/model"], "server"),
            ("VLLM::EngineCore", [], "worker"),
            ("VLLM::Worker", [], "worker"),
            ("VLLM::Worker_TP12", [], "worker"),
            ("python", ["VLLM::Worker_TP0"], "worker"),
            ("python", LAUNCH, "server"),  # inherited command line when setproctitle is absent
            ("vllm", ["vllm", "bench", "serve"], ""),
            ("python", ["python", "script.py", "-m", "vllm.entrypoints.openai.api_server"], ""),
            ("echo", ["echo", "-m", "vllm.entrypoints.openai.api_server"], ""),
            ("VLLM::Worker_TP0-extra", [], ""),
            ("VLLM::EngineCore_DP0", [], ""),
            ("VLLM::Worker_DP0_TP0", [], ""),
            ("VLLM::Worker_PP0_TP0", [], ""),
            ("VLLM::APIServer_0", [], ""),
            ("vllm-helper", [], ""),
        ]
        for name, cmd, role in cases:
            with self.subTest(name=name, cmd=cmd):
                d = ProcDetail(10, name, cmdline=cmd)
                self.assertEqual(d.vllm_role, role)
                self.assertEqual(classify(d, {}), "LLM" if role else "app")

    def test_models_are_full_identifiers_only(self):
        for bad in (None, [], {}, {"data": None}, {"data": {}}, {"data": [None]},
                    {"data": [{"id": 5}]}, {"data": [{"id": ""}]}, {"data": [{"id": "  "}]}):
            with self.subTest(response=bad):
                self.assertIsNone(vllm_models(bad))
        self.assertEqual(vllm_models({"data": []}), [])
        model = vllm_models({"data": [{"id": "org/model", "size": 100, "max_model_len": 200}]})[0]
        self.assertEqual(model["name"], "org/model")
        self.assertTrue(all(value is None for key, value in model.items() if key != "name"))

    def test_one_endpoint_cache_origin_and_credential_isolation(self):
        with listener(locked=True) as (port, seen):
            p = ServerProber()
            own = ("process", 1, 100)
            parent = ("parent", ((1, 100),))
            self.assertIsNotNone(p.get([port], "private-key", kind="vllm", origin=own))
            self.assertIsNotNone(p.get([port], "private-key", kind="vllm", origin=own))
            self.assertEqual(len(seen), 1)
            self.assertIsNone(p.get([port], kind="vllm", origin=parent))
            self.assertIsNone(p.get([port], kind="vllm", origin=parent))
            self.assertEqual(p.locked_port([port], kind="vllm", origin=parent), port)
            self.assertIsNone(p.locked_port([port], "private-key", kind="vllm", origin=own))
            self.assertEqual(len(seen), 2)
            self.assertIsNone(seen[-1][1])
            self.assertIsNotNone(p.get([port], "different-key", kind="vllm", origin=own))
            self.assertEqual(len(seen), 3)  # changed keys do not use the previous cache
            self.assertIsNotNone(p.get([port], "private-key", kind="vllm", origin=("process", 1, 101)))
            self.assertEqual(len(seen), 4)  # reused PID/port does not use old identity
            self.assertEqual({path for path, _ in seen}, {"/v1/models"})
            self.assertNotIn("private-key", repr(vars(p)))
            with patch("gputaskman.llm.time.time", return_value=time.time() + 31):
                self.assertIsNone(p.get([port], kind="vllm", origin=parent))
            self.assertEqual(len(seen), 5)
            self.assertEqual(p.locked_port([port], kind="vllm", origin=parent), port)

    def test_unauthenticated_parent_redirect_is_not_followed(self):
        with listener(redirect=True) as (port, seen):
            self.assertIsNone(probe_server(port, kind="vllm"))
            self.assertEqual(seen, [("/v1/models", None)])

    def sample(self, details, ports, parents, *, nodes=None, prober=None):
        c = collector(details, ports)
        if prober is not None:
            c.prober = prober
        nodes = nodes or chain(details, parents)
        with patch("gputaskman.collector.BACKEND", "nvml"), \
                patch("gputaskman.collector.psutil.Process", side_effect=lambda pid: nodes[pid]), \
                patch("gputaskman.procinfo._create_time", side_effect=lambda pid: details[pid].create_time):
            return c, c.sample()

    def test_collector_two_hops_no_files_no_invented_memory(self):
        with listener() as (port, seen):
            details = {10: ProcDetail(10, "VLLM::Worker_TP0", create_time=3),
                       20: ProcDetail(20, "VLLM::EngineCore", create_time=2),
                       30: ProcDetail(30, "python", cmdline=LAUNCH, create_time=1)}
            _, snap = self.sample(details, {30: [port]}, {10: 20, 20: 30})
        row = snap.procs[0]
        self.assertEqual(row.kind, "LLM")
        self.assertEqual(row.llm.server_source, "parent")
        self.assertEqual(row.llm.server.models[0]["name"], "org/model")
        self.assertIn("reported (parent API): org/model", row.inside)
        self.assertEqual(seen, [("/v1/models", None)])
        self.assertEqual(row.gpu.dedicated, 512 * MiB)
        self.assertEqual(snap.attributed, 512 * MiB)
        self.assertEqual([(s.kind, s.bytes) for s in row.llm.segments], [("unknown", 512 * MiB)])
        self.assertEqual(row.llm.models, [])
        self.assertEqual(row.llm.files, [])
        self.assertIsNone(row.llm.kv_per_token)
        self.assertIsNone(row.llm.n_ctx)
        data = to_dict(snap)["processes"][0]["llm"]
        self.assertEqual(data["server_source"], "parent")
        self.assertEqual(data["server"]["models"][0]["name"], "org/model")
        text = to_text(snap)
        self.assertIn("parent process", text)
        self.assertIn("API-reported model ID: org/model", text)
        self.assertIn("do not prove this process's model residency", text)

    def test_upstream_default_single_gpu_and_tp_layouts(self):
        # Default CLI calls run_server in the launcher process (no set_process_title there).
        # APIServer_<index> titles belong to APIServerProcessManager's multi-API worker path.
        with listener() as (port, seen):
            for name in ("VLLM::EngineCore", "VLLM::Worker", "VLLM::Worker_TP0", "VLLM::Worker_TP1"):
                with self.subTest(name=name):
                    details = {10: ProcDetail(10, name, create_time=3),
                               20: ProcDetail(20, "VLLM::EngineCore", create_time=2),
                               30: ProcDetail(30, "python", cmdline=LAUNCH, create_time=1)}
                    parents = {10: 30} if name == "VLLM::EngineCore" else {10: 20, 20: 30}
                    _, snap = self.sample(details, {30: [port]}, parents)
                    self.assertEqual(snap.procs[0].llm.server_source, "parent")
                    self.assertEqual(snap.procs[0].llm.server.models[0]["name"], "org/model")
            self.assertEqual(seen, [("/v1/models", None)] * 4)

    def test_unclear_ancestry_or_listeners_never_probe(self):
        for reason in ("two servers", "shared port", "two ports", "unrelated parent",
                       "newer parent", "reused parent", "reused worker", "unknown start", "cycle",
                       "distributed parent", "distributed worker", "protected parent", "no parent",
                       "own and parent", "worker listener"):
            with self.subTest(reason=reason):
                details = {10: ProcDetail(10, "VLLM::Worker_TP0", create_time=3),
                           20: ProcDetail(20, "python", cmdline=list(LAUNCH), create_time=2),
                           30: ProcDetail(30, "python", cmdline=list(LAUNCH), create_time=1)}
                ports, parents = {20: [8000]}, {10: 20}
                if reason == "two servers":
                    ports[30], parents[20] = [8001], 30
                elif reason == "shared port":
                    ports[30] = [8000]
                elif reason == "two ports":
                    ports[20] = [8000, 8001]
                elif reason == "unrelated parent":
                    details[20].cmdline = ["python", "unrelated.py"]
                elif reason == "newer parent":
                    details[20].create_time = 4
                elif reason == "unknown start":
                    details[20].create_time = 0
                elif reason == "cycle":
                    parents[20] = 10
                elif reason.startswith("distributed"):
                    details[20 if reason.endswith("parent") else 10].cmdline += ["--data-parallel-size", "2"]
                elif reason == "no parent":
                    parents = {}
                elif reason == "own and parent":
                    ports[10] = [8001]
                elif reason == "worker listener":
                    details[20].name, details[20].cmdline = "VLLM::EngineCore", []
                nodes = chain(details, parents)
                if reason in ("reused parent", "reused worker"):
                    nodes[20 if reason.endswith("parent") else 10].create_time.return_value = 99
                if reason == "protected parent":
                    nodes[10].parent.side_effect = psutil.AccessDenied(10)
                with patch("gputaskman.llm.probe_server") as probe:
                    _, snap = self.sample(details, ports, parents, nodes=nodes)
                probe.assert_not_called()
                self.assertIsNone(snap.procs[0].llm.server)

    def test_parent_401_does_not_receive_worker_key_or_use_authenticated_cache(self):
        with listener(locked=True) as (port, seen):
            details = {10: ProcDetail(10, "VLLM::EngineCore",
                                      cmdline=LAUNCH + ["--api-key", "private-worker-key"], create_time=2),
                       20: ProcDetail(20, "python", cmdline=LAUNCH + ["--api-key", "private-worker-key"],
                                      create_time=1)}
            prober = ServerProber()
            self.assertIsNotNone(prober.get([port], "private-worker-key", kind="vllm", origin=("process", 20, 1)))
            c, snap = self.sample(details, {20: [port]}, {10: 20}, prober=prober)
        self.assertIsNone(snap.procs[0].llm.server)
        self.assertIn("Parent-process API locked (401)", " ".join(snap.procs[0].llm.notes))
        self.assertEqual([path for path, _ in seen], ["/v1/models", "/v1/models"])
        self.assertIsNotNone(seen[0][1])
        self.assertIsNone(seen[1][1])
        self.assertNotIn("private-worker-key", json.dumps(to_dict(snap)))
        self.assertNotIn("private-worker-key", to_text(snap))
        self.assertNotIn("private-worker-key", repr(vars(c.prober)))

    def test_own_listener_keeps_own_provenance(self):
        with listener() as (port, seen):
            details = {10: ProcDetail(10, "python", cmdline=LAUNCH, create_time=1)}
            _, snap = self.sample(details, {10: [port]}, {})
        self.assertEqual(snap.procs[0].llm.server_source, "process")
        self.assertIn("reported (API): org/model", snap.procs[0].inside)
        self.assertEqual(seen, [("/v1/models", None)])

    def test_api_unavailable_does_not_change_measured_memory(self):
        for response in (None, {"error": "disabled"}, {"data": [None]}):
            with self.subTest(response=response):
                details = {10: ProcDetail(10, "VLLM::EngineCore", create_time=2),
                           20: ProcDetail(20, "python", cmdline=LAUNCH, create_time=1)}
                with patch("gputaskman.llm._get", return_value=response):
                    _, snap = self.sample(details, {20: [8000]}, {10: 20})
                self.assertIsNone(snap.procs[0].llm.server)
                self.assertEqual(snap.attributed, 512 * MiB)

    def test_real_child_ancestry_and_http_through_collector(self):
        """Real cheap processes/listener, not a vLLM runtime or a synthetic ancestry lookup."""
        with listener() as (port, seen):
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
            try:
                c = collector({}, {})
                c.inspector = ProcInspector()
                original_new = c.inspector._new

                def details(pid):
                    d = original_new(pid)
                    if pid == os.getpid():
                        d.cmdline = LAUNCH
                    elif pid == child.pid:
                        d.cmdline = LAUNCH  # no setproctitle; inherited launch signature
                    return d

                c.inspector._new = details
                c.inspector.needs_scan = Mock(return_value=False)
                c.counters.sample.return_value = (
                    {0: {child.pid: ProcGpu(dedicated=512 * MiB, shared=0)}},
                    {0: AdapterGpu(dedicated=512 * MiB, shared=0)})
                connections = [SimpleNamespace(status=psutil.CONN_LISTEN, pid=os.getpid(),
                                               laddr=SimpleNamespace(port=port))]
                with patch("gputaskman.collector.BACKEND", "nvml"), \
                        patch("gputaskman.procinfo.psutil.net_connections", return_value=connections):
                    snap = c.sample()
                    again = c.sample()
                self.assertEqual(snap.procs[0].llm.server_source, "parent")
                self.assertEqual(again.procs[0].llm.server.models[0]["name"], "org/model")
                self.assertEqual(seen, [("/v1/models", None)])
            finally:
                child.terminate()
                child.wait(timeout=5)


class VllmUiTests(unittest.IsolatedAsyncioTestCase):
    async def test_parent_report_in_table_inspector_and_json(self):
        with listener() as (port, _):
            details = {10: ProcDetail(10, "VLLM::EngineCore", create_time=2),
                       20: ProcDetail(20, "python", cmdline=LAUNCH, create_time=1)}
            _, snap = VllmTests().sample(details, {20: [port]}, {10: 20})
        with patch.object(GpuTaskManager, "poll"), patch("gputaskman.app.BACKEND", "nvml"):
            app = GpuTaskManager()
            app.selected_key = "10"
            async with app.run_test(size=(180, 50)) as pilot:
                app.apply(snap)
                await pilot.pause()
                text = "\n".join(getattr(line, "plain", "") for line in
                                 app.query_one("#inspector", Static).content.renderables)
                self.assertIn("parent process; API report, not worker residency", text)
                self.assertIn("API-reported model ID: org/model", text)
