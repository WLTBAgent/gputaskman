from __future__ import annotations

import subprocess
import sys
import time
import unittest
from unittest.mock import Mock, patch

from gputaskman import ascend, linux
from gputaskman.collector import Collector
from gputaskman.telemetry import DeviceTelemetry


MB = 1024 * 1024
INFO = (
    "| 0 310P3 | OK | NA 45 0 / 0 |\n"
    "| 0 0 | 0000:0a:00.0 | 0 12 / 100 |\n"
    "| 0 310P3 | OK | NA 45 0 / 0 |\n"
    "| 1 1 | 0000:0b:00.0 | 0 24 / 100 |\n"
)


def proc(pid: int, mb: int) -> str:
    return f"Process id:{pid} Process name:worker Process memory(MB):{mb}"


class AscendTests(unittest.TestCase):
    def setUp(self):
        self.chips = [
            ascend.AscendChip(0, 0, "chip 0", "0000:0a:00.0", 100 * MB),
            ascend.AscendChip(0, 1, "chip 1", "0000:0b:00.0", 100 * MB),
        ]
        self.counters = ascend.AscendCounters(self.chips)

    def test_hexadecimal_pci_and_unavailable_core(self):
        rows = ascend.parse_info(INFO.replace("0 12 / 100", "NA 12 / 100"))
        self.assertEqual([r["bus"] for r in rows], ["0000:0a:00.0", "0000:0b:00.0"])
        self.assertIsNone(rows[0]["aicore"])

    def test_chip_keys_do_not_collide_with_other_pci_devices(self):
        self.assertNotEqual(ascend.chip_adapter_key("0000:0a:00.0", 0),
                            linux.pci_key("000a:00:00.0"))

    def test_per_chip_attribution_never_combines_or_copies_other_chip(self):
        def run(args):
            if args == ["info"]:
                return INFO
            return proc(12, 10) if args[-1] == "0" else proc(12, 20) + "\n" + proc(13, 3)

        with patch("gputaskman.ascend._run", side_effect=run), patch.object(self.counters, "start"):
            per, adapters = self.counters.sample()
        key0, key1 = self.counters.by_key
        self.assertEqual({pid: gpu.dedicated for pid, gpu in per[key0].items()}, {12: 10 * MB})
        self.assertEqual({pid: gpu.dedicated for pid, gpu in per[key1].items()},
                         {12: 20 * MB, 13: 3 * MB})
        self.assertEqual(adapters[key0].dedicated, 12 * MB)
        self.assertTrue(self.counters.process_available)

    def test_failed_refresh_clears_previous_values_and_recovers(self):
        def run(args):
            return INFO if args == ["info"] else proc(12, 10)

        with patch("gputaskman.ascend._run", side_effect=run), patch.object(self.counters, "start"):
            self.counters._refresh(100)
        with patch("gputaskman.ascend._run", side_effect=OSError("timed out")), \
                patch.object(self.counters, "start"):
            self.counters._refresh(111)
        self.assertEqual(self.counters._procs, {})
        self.assertFalse(self.counters.process_available)
        self.assertTrue(all(c.mem_used is None for c in self.chips))
        self.assertIn("device memory unavailable", self.counters.status)
        self.assertIn("process memory unavailable", self.counters.status)
        with patch("gputaskman.ascend._run", side_effect=run), patch.object(self.counters, "start"):
            self.counters._refresh(122)
        self.assertTrue(self.counters.process_available)
        self.assertNotIn("memory unavailable", self.counters.status)

    def test_bad_process_record_is_not_treated_as_no_processes(self):
        with self.assertRaises(ValueError):
            ascend.parse_proc_mem("Process id:7 Process name:worker Process memory(MB):NA")
        with self.assertRaises(ValueError):
            ascend.parse_proc_mem("")
        self.assertEqual(ascend.parse_proc_mem("No running processes found"), [])

    def test_driver_byte_and_megabyte_process_values(self):
        data = "Process id:38329 Process name:hlt_host_devmm_ Process memory(B):100683776\n" + proc(7, 3)
        self.assertEqual(ascend.parse_proc_mem(data),
                         [(38329, "hlt_host_devmm_", 100683776), (7, "worker", 3 * MB)])

    def test_watch_zero_and_stale_sample(self):
        chip = self.chips[0]
        self.counters._watch[(chip.npu, chip.chip)] = (
            10.0, {"temp": 40, "ai_core": 0, "mem_bw": 0})
        self.counters._watch_error = ""
        with patch("gputaskman.ascend.time.monotonic", return_value=11):
            dev = self.counters.read_device(chip, 0)
            self.assertEqual((dev.util_gpu, dev.util_mem), (0.0, 0.0))
        with patch("gputaskman.ascend.time.monotonic", return_value=14):
            dev = self.counters.read_device(chip, 0)
            self.assertIsNone(dev.util_gpu)
            self.assertIsNone(dev.util_mem)
            self.assertIn("stale", self.counters.status)

    def test_close_reaps_watch_child(self):
        self.counters = ascend.AscendCounters([self.chips[1]])
        popen = subprocess.Popen
        commands = []

        def start_child(*args, **kwargs):
            commands.append(args[0])
            return popen([sys.executable, "-u", "-c",
                          "import time; print('0 1 0 40 1 0 0 0 0', flush=True); time.sleep(60)"],
                         **kwargs)

        with patch("gputaskman.ascend._npu_path", return_value="npu-smi"), \
                patch("gputaskman.ascend.subprocess.Popen", side_effect=start_child):
            self.counters.start()
            child = self.counters._watch_process
            try:
                deadline = time.monotonic() + 5
                while self.counters.watch_row(self.chips[1]) is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertIsNotNone(self.counters.watch_row(self.chips[1]))
            finally:
                self.counters.close()
        self.assertIsNotNone(child.poll())
        self.assertFalse(self.counters._thread.is_alive())
        self.assertEqual(commands, [["npu-smi", "info", "watch", "-i", "0", "-c", "1"]])

    def test_linux_counters_do_not_probe_ascend_without_selection(self):
        with patch("gputaskman.linux.is_wsl", return_value=False), \
                patch("gputaskman.drm.DrmCounters") as drm, \
                patch("gputaskman.linux.N", None), patch("gputaskman.ascend.probe") as probe:
            drm.return_value.ok = False
            counters = linux.LinuxCounters()
            counters.sample()
            counters.close()
        self.assertIsNone(counters.ascend)
        probe.assert_not_called()

    def test_collector_attaches_only_selected_chip(self):
        adapter = self.counters.adapters()[1]
        counters = Mock(drm=None)
        with patch("gputaskman.collector.BACKEND", "nvml"), \
                patch("gputaskman.collector.GpuCounters", return_value=counters), \
                patch("gputaskman.collector.Nvml"), patch("gputaskman.collector.AmdSmi"), \
                patch("gputaskman.collector.Adl"), \
                patch("gputaskman.ascend.cached_chips", return_value=self.chips):
            collector = Collector(adapters=[adapter])
        self.assertEqual(len(collector.ascend.chips), 1)
        self.assertEqual(collector.ascend.chips[0].chip, 1)
        self.assertIsNot(collector.ascend.chips[0], self.chips[1])
        self.assertIs(counters.ascend, collector.ascend)

    def test_collector_dispatches_and_propagates_unavailable_processes(self):
        c = Collector.__new__(Collector)
        chip = self.chips[0]
        c.adapter = self.counters.adapters()[0]
        c.adapters = [c.adapter]
        c.ascend_chip = chip
        c.ascend = Mock(process_available=False, status="process memory unavailable")
        c.ascend.read_device.return_value = DeviceTelemetry(index=0, name=chip.name, source="npu-smi")
        c.counters = Mock(outside={})
        c.counters.sample.return_value = ({}, {})
        c.nvml = Mock(ok=False)
        c.gpu_index, c.scan_budget, c.host, c.host_status = 0, None, None, ""
        c.inspector, c.prober = Mock(), Mock()
        with patch("gputaskman.collector.BACKEND", "nvml"):
            snap = c.sample()
        self.assertEqual(snap.backend, "ascend")
        self.assertEqual(snap.nv.source, "npu-smi")
        self.assertIsNone(snap.attributed)
        self.assertIn("process memory unavailable", snap.counter_status)
        c.ascend.read_device.assert_called_once_with(chip, 0)


if __name__ == "__main__":
    unittest.main()
