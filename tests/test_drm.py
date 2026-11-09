from __future__ import annotations

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from gputaskman import drm

GiB = 1 << 30
MiB = 1 << 20
KiB = 1 << 10

FDINFO = """pos:\t0
flags:\t02100002
mnt_id:\t28
ino:\t415
drm-driver:\tamdgpu
drm-client-id:\t26
drm-pdev:\t0000:03:00.0
pasid:\t25
drm-total-cpu:\t0
drm-shared-cpu:\t0
drm-resident-cpu:\t0
drm-purgeable-cpu:\t0
drm-total-gtt:\t2 MiB
drm-shared-gtt:\t0
drm-resident-gtt:\t2060 KiB
drm-purgeable-gtt:\t0
drm-total-vram:\t12 KiB
drm-shared-vram:\t0
drm-resident-vram:\t0
drm-purgeable-vram:\t0
drm-memory-vram:\t0 KiB
drm-memory-gtt: \t2060 KiB
drm-memory-cpu: \t0 KiB
amd-evicted-vram:\t12 KiB
amd-requested-vram:\t12 KiB
amd-requested-gtt:\t2048 KiB
drm-engine-gfx:\t5308400172 ns
drm-engine-compute:\t0 ns
drm-engine-enc:\t1200 ns
drm-engine-capacity-gfx:\t1
"""

# The Intel samples are laid out the way the kernels print them (i915_drm_client.c; the xe_drm_client.c documentation
# sample; sizes in the units drm_print_memory_stats uses). They are built from that code, not captured from hardware.
I915_FDINFO = """pos:\t0
flags:\t02100002
mnt_id:\t28
ino:\t415
drm-driver:\ti915
drm-pdev:\t0000:03:00.0
drm-client-id:\t42
drm-total-system0:\t8 MiB
drm-shared-system0:\t0
drm-active-system0:\t0
drm-resident-system0:\t6 MiB
drm-purgeable-system0:\t0
drm-total-local0:\t2048 MiB
drm-shared-local0:\t0
drm-active-local0:\t16 MiB
drm-resident-local0:\t2048 MiB
drm-purgeable-local0:\t0
drm-total-stolen-local0:\t0
drm-shared-stolen-local0:\t0
drm-active-stolen-local0:\t0
drm-resident-stolen-local0:\t64 KiB
drm-purgeable-stolen-local0:\t0
drm-engine-render:\t5308400172 ns
drm-engine-copy:\t1200 ns
drm-engine-video:\t0 ns
drm-engine-capacity-video:\t2
drm-engine-video-enhance:\t0 ns
drm-engine-capacity-video-enhance:\t2
drm-engine-compute:\t750000000 ns
drm-engine-capacity-compute:\t4
"""

# An integrated GPU has no local0 region, only system0 and its stolen carve-out.
I915_IGPU_FDINFO = """pos:\t0
flags:\t02100002
mnt_id:\t28
ino:\t415
drm-driver:\ti915
drm-pdev:\t0000:00:02.0
drm-client-id:\t7
drm-total-system0:\t64 MiB
drm-shared-system0:\t0
drm-active-system0:\t0
drm-resident-system0:\t60 MiB
drm-purgeable-system0:\t0
drm-total-stolen-system0:\t0
drm-shared-stolen-system0:\t0
drm-active-stolen-system0:\t0
drm-resident-stolen-system0:\t0
drm-purgeable-stolen-system0:\t0
drm-engine-render:\t1200000000 ns
drm-engine-copy:\t0 ns
drm-engine-video:\t300000000 ns
drm-engine-video-enhance:\t0 ns
"""

# The Vivobook's vkmark client (Xe-LP iGPU, i915) under load: resident system memory, part of it
# shared with the compositor, so the bytes are measurable but not attributable to the process.
I915_IGPU_SHARED_FDINFO = """pos:\t0
flags:\t02100002
mnt_id:\t41
ino:\t427
drm-driver:\ti915
drm-client-id:\t54
drm-pdev:\t0000:00:02.0
drm-total-system0:\t37476 KiB
drm-shared-system0:\t8576 KiB
drm-active-system0:\t0
drm-resident-system0:\t37476 KiB
drm-purgeable-system0:\t156 KiB
drm-total-stolen-system0:\t0
drm-shared-stolen-system0:\t0
drm-active-stolen-system0:\t0
drm-resident-stolen-system0:\t0
drm-purgeable-stolen-system0:\t0
drm-engine-render:\t8242337792 ns
drm-engine-copy:\t0 ns
drm-engine-video:\t0 ns
drm-engine-video-enhance:\t0 ns
"""

# i915 before Gen8 prints the memory stats but no drm-engine lines at all.
I915_GEN7_FDINFO = """drm-driver:\ti915
drm-pdev:\t0000:00:02.0
drm-client-id:\t5
drm-total-system0:\t4 MiB
drm-shared-system0:\t0
drm-active-system0:\t0
drm-resident-system0:\t4 MiB
drm-purgeable-system0:\t0
"""

# xe: memory regions are named system/gtt/vram0/stolen and the load is drm-cycles-*, not drm-engine-<class> ns.
XE_FDINFO = """pos:\t0
flags:\t0100002
mnt_id:\t26
ino:\t685
drm-driver:\txe
drm-client-id:\t3
drm-pdev:\t0000:03:00.0
drm-total-system:\t0
drm-shared-system:\t0
drm-active-system:\t0
drm-resident-system:\t0
drm-purgeable-system:\t0
drm-total-gtt:\t192 KiB
drm-shared-gtt:\t0
drm-active-gtt:\t0
drm-resident-gtt:\t192 KiB
drm-total-vram0:\t23992 KiB
drm-shared-vram0:\t16 MiB
drm-active-vram0:\t0
drm-resident-vram0:\t23992 KiB
drm-total-stolen:\t0
drm-shared-stolen:\t0
drm-active-stolen:\t0
drm-resident-stolen:\t0
drm-cycles-rcs:\t28257900
drm-total-cycles-rcs:\t7655183225
drm-cycles-bcs:\t0
drm-total-cycles-bcs:\t7655183225
drm-cycles-vcs:\t0
drm-total-cycles-vcs:\t7655183225
drm-engine-capacity-vcs:\t2
drm-cycles-vecs:\t0
drm-total-cycles-vecs:\t7655183225
drm-engine-capacity-vecs:\t2
drm-cycles-ccs:\t0
drm-total-cycles-ccs:\t7655183225
drm-engine-capacity-ccs:\t4
"""


def _make_card(root: Path, name, pdev, vendor, device, vram, driver="amdgpu", gtt=0, apu=False) -> Path:
    dev = root / "drm" / name / "device"
    dev.mkdir(parents=True)
    (dev / "vendor").write_text(f"0x{vendor:04x}\n")
    (dev / "device").write_text(f"0x{device:04x}\n")
    (dev / "mem_info_vram_total").write_text(str(vram))
    (dev / "mem_info_gtt_total").write_text(str(gtt))
    if not apu:
        (dev / "mem_info_vram_vendor").write_text("samsung\n")  # only discrete cards have VRAM chips
    driver_dir = root / "drivers" / driver
    driver_dir.mkdir(parents=True, exist_ok=True)
    os.symlink(driver_dir, dev / "driver")
    (dev / "uevent").write_text(f"DRIVER={driver}\nPCI_SLOT_NAME={pdev}\n")
    return dev


def _make_intel_card(root: Path, name, pdev, device, driver="i915") -> Path:
    """An Intel card's sysfs: there are no mem_info_* files, the local-memory BAR is the resource2 file."""
    dev = root / "drm" / name / "device"
    dev.mkdir(parents=True)
    (dev / "vendor").write_text("0x8086\n")
    (dev / "device").write_text(f"0x{device:04x}\n")
    driver_dir = root / "drivers" / driver
    driver_dir.mkdir(parents=True, exist_ok=True)
    os.symlink(driver_dir, dev / "driver")
    (dev / "uevent").write_text(f"DRIVER={driver}\nPCI_SLOT_NAME={pdev}\n")
    return dev


def _card(dev, pdev, name="AMD GPU", vram=16 * GiB, vendor=0x1002, device=0x7550) -> drm.DrmCard:
    return drm.DrmCard(
        key=drm.pci_key(pdev), pdev=pdev, sysfs=str(dev), driver="amdgpu",
        vendor_id=vendor, device_id=device, name=name, vram_total=vram,
        integrated=vram <= 4 * GiB,
    )


def _write_fd(proc: Path, pid, fd, text, target="/dev/dri/renderD128") -> None:
    """One process fd: the fdinfo contents plus the /proc/<pid>/fd symlink DrmCounters filters on."""
    fdinfo = proc / str(pid) / "fdinfo"
    fdinfo.mkdir(parents=True, exist_ok=True)
    (fdinfo / str(fd)).write_text(text)
    fds = proc / str(pid) / "fd"
    fds.mkdir(parents=True, exist_ok=True)
    link = fds / str(fd)
    if not link.is_symlink():
        os.symlink(target, link)


def _fdinfo(client, pdev, gfx=0, vram=0, gtt=0, shared_vram=0, shared_gtt=0, shared_cpu=0) -> str:
    return (
        "pos:\t0\n"
        "drm-driver:\tamdgpu\n"
        f"drm-client-id:\t{client}\n"
        f"drm-pdev:\t{pdev}\n"
        f"drm-shared-vram:\t{shared_vram} KiB\n"
        f"drm-resident-vram:\t{vram} KiB\n"
        f"drm-shared-gtt:\t{shared_gtt} KiB\n"
        f"drm-resident-gtt:\t{gtt} KiB\n"
        f"drm-shared-cpu:\t{shared_cpu} KiB\n"
        "drm-resident-cpu:\t0\n"
        f"drm-engine-gfx:\t{gfx} ns\n"
        f"drm-engine-capacity-gfx:\t1\n"
    )


def _i915_fdinfo(client, pdev, render=0, video=0, local_mib=0, system_mib=0, shared_local_mib=0, engines=True) -> str:
    """A discrete i915 client's fdinfo: memory by region (sizes in MiB) and, unless engines is False, the engine
    classes the way i915 prints them."""
    text = (
        "pos:\t0\n"
        "drm-driver:\ti915\n"
        f"drm-client-id:\t{client}\n"
        f"drm-pdev:\t{pdev}\n"
        f"drm-shared-system0:\t0\n"
        f"drm-resident-system0:\t{system_mib} MiB\n"
        f"drm-shared-local0:\t{shared_local_mib} MiB\n"
        f"drm-resident-local0:\t{local_mib} MiB\n"
    )
    if engines:
        text += (f"drm-engine-render:\t{render} ns\n"
                 "drm-engine-copy:\t0 ns\n"
                 f"drm-engine-video:\t{video} ns\n")
    return text


class Clock:
    """A monotonic_ns stand-in the test advances by hand: every fdinfo read in one refresh sees the same time."""

    def __init__(self) -> None:
        self.t = 0

    def __call__(self) -> int:
        return self.t

    def tick(self, seconds: float = 1.0) -> None:
        self.t += int(seconds * 1_000_000_000)


class ParseTests(unittest.TestCase):
    def test_parse_size(self):
        self.assertEqual(drm.parse_size("2060 KiB"), 2060 * 1024)
        self.assertEqual(drm.parse_size("2 MiB"), 2 * 1024**2)
        self.assertEqual(drm.parse_size("1 GiB"), 1024**3)
        self.assertEqual(drm.parse_size("12 KiB"), 12 * 1024)
        self.assertEqual(drm.parse_size("0"), 0)
        self.assertEqual(drm.parse_size("512"), 512)
        self.assertEqual(drm.parse_size("  4 KiB  "), 4096)
        self.assertIsNone(drm.parse_size("bogus"))  # unknown, not 0
        self.assertIsNone(drm.parse_size("12 bananas"))
        self.assertIsNone(drm.parse_size(None))

    def test_parse_fdinfo_example(self):
        info = drm.parse_fdinfo(FDINFO)
        self.assertIsNotNone(info)
        self.assertEqual(info["driver"], "amdgpu")
        self.assertEqual(info["client_id"], 26)
        self.assertEqual(info["pdev"], "0000:03:00.0")
        self.assertEqual(info["vram"], 0)
        self.assertEqual(info["gtt"], 2060 * 1024)
        self.assertEqual(info["cpu"], 0)
        self.assertEqual(info["engines"], {"gfx": 5308400172, "compute": 0, "enc": 1200})

    def test_parse_fdinfo_non_drm(self):
        self.assertIsNone(drm.parse_fdinfo("pos:\t0\nflags:\t02\nmnt_id:\t28\n"))

    def test_parse_fdinfo_resident_fallback(self):
        info = drm.parse_fdinfo(
            "drm-driver:\tamdgpu\n"
            "drm-resident-vram:\t8 KiB\n"
            "drm-resident-gtt:\t3 MiB\n"
            "drm-resident-cpu:\t0\n"
        )
        self.assertIsNotNone(info)
        self.assertEqual(info["vram"], 8 * 1024)
        self.assertEqual(info["gtt"], 3 * 1024**2)
        self.assertEqual(info["cpu"], 0)
        self.assertEqual(info["engines"], {})
        self.assertIsNone(info["client_id"])
        self.assertIsNone(info["shared"])  # no drm-shared-* keys: unknown, not 0

    def test_parse_fdinfo_prefers_resident_keys(self):
        info = drm.parse_fdinfo("drm-driver:\tamdgpu\ndrm-resident-vram:\t8 KiB\ndrm-memory-vram:\t9 KiB\n"
                                "drm-shared-vram:\t0\ndrm-shared-gtt:\t4 KiB\ndrm-shared-cpu:\t0\n")
        self.assertEqual(info["vram"], 8 * 1024)
        self.assertEqual(info["shared"], 4 * 1024)

    def test_parse_fdinfo_shared_needs_all_memory_regions(self):
        info = drm.parse_fdinfo("drm-driver:\tamdgpu\ndrm-memory-vram:\t300 MiB\ndrm-shared-vram:\t0\n")
        self.assertIsNone(info["shared"])  # no drm-shared-gtt: whether GTT buffers are shared is unknown
        info = drm.parse_fdinfo("drm-driver:\tamdgpu\ndrm-shared-vram:\t0\ndrm-shared-gtt:\t0\n")
        self.assertIsNone(info["shared"])  # no drm-shared-cpu: a shared CPU-preferred buffer may be resident in GTT

    def test_parse_fdinfo_shared_cpu_counts_toward_private_gate(self):
        info = drm.parse_fdinfo("drm-driver:\tamdgpu\ndrm-shared-vram:\t0\ndrm-shared-gtt:\t0\n"
                                "drm-shared-cpu:\t64 KiB\n")
        self.assertEqual(info["shared"], 64 * 1024)
        info = drm.parse_fdinfo("drm-driver:\tamdgpu\ndrm-shared-vram:\t0\ndrm-shared-gtt:\t0\n"
                                "drm-shared-cpu:\tlots\n")
        self.assertIsNone(info["shared"])

    def test_parse_fdinfo_unparseable_is_unknown(self):
        info = drm.parse_fdinfo("drm-driver:\tamdgpu\ndrm-resident-vram:\tlots\ndrm-shared-vram:\t0\n"
                                "drm-shared-gtt:\t?\ndrm-engine-gfx:\ttorn\n")
        self.assertIsNone(info["vram"])
        self.assertIsNone(info["gtt"])  # missing entirely
        self.assertIsNone(info["shared"])
        self.assertIsNone(info["engines"])


PCI_IDS = """# fake
1002  Advanced Micro Devices, Inc. [AMD/ATI]
\t7550  Navi 48 [Radeon RX 9070/9070 XT/9070 GRE]
\t\t7550 1234  Subsystem line that must be skipped
\t13c0  Granite Ridge [Radeon 610M]
1003  Somebody else
\t9999  Not this vendor
"""


class CardsTests(unittest.TestCase):
    def test_cards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # An APU with a large BIOS carve-out is still integrated: the flag doesn't depend on size.
            _make_card(root, "card0", "0000:05:00.0", 0x1002, 0x13C0, 8 * GiB, gtt=3 * GiB, apu=True)
            _make_card(root, "card1", "0000:03:00.0", 0x1002, 0x7550, 16 * GiB, gtt=32 * GiB)
            _make_card(root, "card2", "0000:0a:00.0", 0x10DE, 0x9999, 24 * GiB, driver="nvidia")
            for junk in ("card1-DP-1", "renderD128"):
                (root / "drm" / junk).mkdir(parents=True)
            pci_ids = root / "pci.ids"
            pci_ids.write_text(PCI_IDS)

            found = drm.cards(str(root / "drm"), str(pci_ids))

            self.assertEqual([c.pdev for c in found], ["0000:03:00.0", "0000:05:00.0"])
            self.assertEqual([c.name for c in found], [
                "AMD Radeon RX 9070/9070 XT/9070 GRE",
                "AMD Radeon 610M",
            ])
            self.assertFalse(found[0].integrated)
            self.assertTrue(found[1].integrated)
            self.assertEqual(found[0].vram_total, 16 * GiB)
            self.assertEqual(found[0].key, drm.pci_key("0000:03:00.0"))
            self.assertEqual(found[0].vendor_id, 0x1002)
            self.assertEqual(found[0].device_id, 0x7550)

            adapters = drm.adapters(found)
            self.assertEqual([a.luid for a in adapters], [c.key for c in found])
            self.assertEqual(adapters[0].dedicated_vram, 16 * GiB)
            self.assertEqual(adapters[0].shared_system, 32 * GiB)
            self.assertEqual(adapters[1].shared_system, 3 * GiB)
            self.assertFalse(any(a.software for a in adapters))


class ReadDeviceTests(unittest.TestCase):
    def test_read_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            dev = Path(tmp) / "card0" / "device"
            dev.mkdir(parents=True)
            (dev / "mem_info_vram_total").write_text(str(16 * GiB))
            (dev / "mem_info_vram_used").write_text(str(3 * GiB))
            (dev / "gpu_busy_percent").write_text("42\n")
            (dev / "mem_busy_percent").write_text("7\n")
            (dev / "current_link_speed").write_text("32.0 GT/s PCIe\n")
            (dev / "current_link_width").write_text("16\n")
            (dev / "pp_dpm_sclk").write_text("0: 500Mhz\n1: 2100Mhz\n2: 2570Mhz *\nS: 44Mhz *\n")
            hwmon = dev / "hwmon" / "hwmon0"
            hwmon.mkdir(parents=True)
            (hwmon / "temp1_label").write_text("edge\n")
            (hwmon / "temp1_input").write_text("57000\n")
            (hwmon / "temp2_label").write_text("junction\n")
            (hwmon / "temp2_input").write_text("65000\n")
            (hwmon / "power1_average").write_text("47000000\n")
            (hwmon / "power1_label").write_text("PPT\n")
            (hwmon / "power1_cap").write_text("340000000\n")
            (hwmon / "freq1_label").write_text("sclk\n")
            (hwmon / "freq1_input").write_text("27000000\n")
            (hwmon / "freq2_label").write_text("mclk\n")
            (hwmon / "freq2_input").write_text("1258000000\n")
            (hwmon / "fan1_input").write_text("1000\n")
            (hwmon / "fan1_max").write_text("2000\n")

            card = _card(dev, "0000:03:00.0", name="AMD Radeon RX 9070")
            d = drm.read_device(card, 0)

            self.assertEqual(d.index, 0)
            self.assertEqual(d.name, "AMD Radeon RX 9070")
            self.assertEqual(d.pci_bus, "0000:03:00.0")
            self.assertEqual(d.driver_model, "amdgpu")
            self.assertEqual(d.mem_total, 16 * GiB)
            self.assertEqual(d.mem_used, 3 * GiB)
            self.assertEqual(d.util_gpu, 42)
            self.assertEqual(d.util_mem, 7)
            self.assertEqual(d.temp, 57)
            self.assertEqual(d.power_w, 47.0)
            self.assertEqual(d.power_limit_w, 340.0)
            self.assertEqual(d.fan, 50)
            self.assertEqual(d.gfx_clock, 27)
            self.assertEqual(d.source, "AMD sysfs")
            self.assertEqual(d.power_kind, "PPT")
            self.assertEqual(d.mem_clock, 1258)
            self.assertEqual(d.sm_clock_max, 2570)
            self.assertEqual(d.pcie_gen, 5)
            self.assertEqual(d.pcie_width, 16)
            self.assertIsNone(d.enc)
            self.assertIsNone(d.dec)

    def test_cpu_package_temp_prefers_package_id_0(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nvme = root / "hwmon0"
            nvme.mkdir()
            (nvme / "name").write_text("nvme\n")
            cpu = root / "hwmon1"
            cpu.mkdir()
            (cpu / "name").write_text("coretemp\n")
            (cpu / "temp1_label").write_text("Package id 0\n")
            (cpu / "temp1_input").write_text("55000\n")
            (cpu / "temp2_label").write_text("Core 0\n")
            (cpu / "temp2_input").write_text("99000\n")  # a core is not the package
            self.assertEqual(drm._cpu_package_temp(str(root)), 55)

    def test_cpu_package_temp_missing_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cpu = root / "hwmon0"
            cpu.mkdir()
            (cpu / "name").write_text("coretemp\n")
            (cpu / "temp1_label").write_text("Core 0\n")  # no package sensor: no guessing from cores
            (cpu / "temp1_input").write_text("99000\n")
            self.assertIsNone(drm._cpu_package_temp(str(root)))
            self.assertIsNone(drm._cpu_package_temp(str(root / "no-such-dir")))

    def test_cpu_package_temp_skips_torn_readings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cpu = root / "hwmon0"
            cpu.mkdir()
            (cpu / "name").write_text("coretemp\n")
            (cpu / "temp1_label").write_text("Package id 0\n")
            (cpu / "temp1_input").write_text("lots\n")  # torn: try the other sensors
            (cpu / "temp2_label").write_text("Package id 1\n")
            (cpu / "temp2_input").write_text("48000\n")
            self.assertEqual(drm._cpu_package_temp(str(root)), 48)

    def test_read_device_intel_falls_back_to_cpu_package_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            card_sysfs = Path(tmp) / "card1"
            dev = card_sysfs / "device"
            dev.mkdir(parents=True)
            card = drm.DrmCard(key=32, pdev="0000:00:02.0", sysfs=str(dev), driver="i915",
                               vendor_id=0x8086, device_id=0xA7AD, name="Intel Intel Graphics",
                               vram_total=0, integrated=True, intel=True)
            with patch.object(drm, "_cpu_package_temp", return_value=55):
                d = drm.read_device(card, 0)
            self.assertEqual((d.temp, d.temp_kind), (55, "CPU package"))

    def test_read_device_intel_clock_and_throttle(self):
        with tempfile.TemporaryDirectory() as tmp:
            card_sysfs = Path(tmp) / "card1"
            dev = card_sysfs / "device"
            dev.mkdir(parents=True)
            gt = card_sysfs / "gt" / "gt0"
            gt.mkdir(parents=True)
            (gt / "rps_act_freq_mhz").write_text("300\n")
            (gt / "rps_cur_freq_mhz").write_text("350\n")
            (gt / "rps_RP0_freq_mhz").write_text("1250\n")
            (gt / "throttle_reason_pl1").write_text("1\n")
            (gt / "throttle_reason_pl2").write_text("0\n")
            (gt / "throttle_reason_thermal").write_text("0\n")

            card = drm.DrmCard(key=32, pdev="0000:00:02.0", sysfs=str(dev), driver="i915",
                               vendor_id=0x8086, device_id=0xA7AD, name="Intel Intel Graphics",
                               vram_total=0, integrated=True, intel=True)
            d = drm.read_device(card, 0)

            self.assertEqual(d.sm_clock, 300)  # the actual frequency, not the requested one
            self.assertEqual(d.gfx_clock, 350)
            self.assertEqual(d.sm_clock_max, 1250)
            self.assertEqual(d.throttle, ["PL1"])
            self.assertEqual(d.source, "Intel sysfs")

    def test_read_device_intel_clock_falls_back_to_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            card_sysfs = Path(tmp) / "card1"
            dev = card_sysfs / "device"
            dev.mkdir(parents=True)
            gt = card_sysfs / "gt" / "gt0"
            gt.mkdir(parents=True)
            (gt / "rps_cur_freq_mhz").write_text("300\n")
            (gt / "rps_RP0_freq_mhz").write_text("1250\n")

            card = drm.DrmCard(key=32, pdev="0000:00:02.0", sysfs=str(dev), driver="i915",
                               vendor_id=0x8086, device_id=0xA7AD, name="Intel Intel Graphics",
                               vram_total=0, integrated=True, intel=True)
            d = drm.read_device(card, 0)

            self.assertEqual(d.sm_clock, 300)
            # No throttle node is readable here: unknown, not "no throttling".
            self.assertIsNone(d.throttle)

    def test_read_device_intel_throttle_all_idle_is_empty_not_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            card_sysfs = Path(tmp) / "card1"
            dev = card_sysfs / "device"
            dev.mkdir(parents=True)
            gt = card_sysfs / "gt" / "gt0"
            gt.mkdir(parents=True)
            (gt / "throttle_reason_pl1").write_text("0\n")
            (gt / "throttle_reason_thermal").write_text("0\n")

            card = drm.DrmCard(key=32, pdev="0000:00:02.0", sysfs=str(dev), driver="i915",
                               vendor_id=0x8086, device_id=0xA7AD, name="Intel Intel Graphics",
                               vram_total=0, integrated=True, intel=True)
            self.assertEqual(drm.read_device(card, 0).throttle, [])

    def test_intel_throttle_skips_unreadable_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            gt = Path(tmp)
            (gt / "throttle_reason_pl2").write_text("1\n")
            (gt / "throttle_reason_thermal").write_text("lots\n")  # unparseable: unknown, not active
            self.assertEqual(drm._intel_throttle(str(gt)), ["PL2"])

    def test_read_device_intel_gpu_sensor_wins_over_cpu_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            card_sysfs = Path(tmp) / "card1"
            dev = card_sysfs / "device"
            dev.mkdir(parents=True)
            hwmon = dev / "hwmon" / "hwmon0"
            hwmon.mkdir(parents=True)
            (hwmon / "temp1_label").write_text("edge\n")
            (hwmon / "temp1_input").write_text("61000\n")
            card = drm.DrmCard(key=32, pdev="0000:00:02.0", sysfs=str(dev), driver="i915",
                               vendor_id=0x8086, device_id=0xA7AD, name="Intel Intel Graphics",
                               vram_total=0, integrated=True, intel=True)
            with patch.object(drm, "_cpu_package_temp", return_value=55):
                d = drm.read_device(card, 0)
            self.assertEqual((d.temp, d.temp_kind), (61, ""))

    def test_read_device_intel_reports_pmu_and_rapl(self):
        with tempfile.TemporaryDirectory() as tmp:
            card_sysfs = Path(tmp) / "card1"
            dev = card_sysfs / "device"
            dev.mkdir(parents=True)
            card = drm.DrmCard(key=32, pdev="0000:00:02.0", sysfs=str(dev), driver="i915",
                               vendor_id=0x8086, device_id=0xA7AD, name="Intel Intel Graphics",
                               vram_total=0, integrated=True, intel=True)
            with patch.object(drm, "_intel_pmu_rates", return_value=(42.0, 7.0)), \
                    patch.object(drm, "_intel_rapl_w", return_value=5.0):
                d = drm.read_device(card, 0)
            self.assertEqual((d.util_gpu, d.video), (42.0, 7.0))
            self.assertEqual((d.power_w, d.power_kind), (5.0, "RAPL uncore"))

    def test_read_device_discrete_intel_without_sensor_stays_unknown(self):
        # The CPU package shares a die with an integrated GPU only; on a discrete card it is
        # not the GPU's temperature, so there is no fallback.
        with tempfile.TemporaryDirectory() as tmp:
            card_sysfs = Path(tmp) / "card1"
            dev = card_sysfs / "device"
            dev.mkdir(parents=True)
            card = drm.DrmCard(key=12288, pdev="0000:03:00.0", sysfs=str(dev), driver="i915",
                               vendor_id=0x8086, device_id=0x56A0, name="Intel Arc A770",
                               vram_total=16 * GiB, integrated=False, intel=True)
            with patch.object(drm, "_cpu_package_temp", return_value=55):
                d = drm.read_device(card, 0)
            self.assertEqual((d.temp, d.temp_kind), (None, ""))

    def test_read_device_discrete_intel_skips_rapl(self):
        with tempfile.TemporaryDirectory() as tmp:
            card_sysfs = Path(tmp) / "card1"
            dev = card_sysfs / "device"
            dev.mkdir(parents=True)
            card = drm.DrmCard(key=12288, pdev="0000:03:00.0", sysfs=str(dev), driver="i915",
                               vendor_id=0x8086, device_id=0x56A0, name="Intel Arc A770",
                               vram_total=16 * GiB, integrated=False, intel=True)
            with patch.object(drm, "_intel_pmu_rates", return_value=(42.0, None)), \
                    patch.object(drm, "_intel_rapl_w") as rapl:
                d = drm.read_device(card, 0)
            rapl.assert_not_called()  # uncore is not a discrete card's power
            self.assertEqual(d.util_gpu, 42.0)
            self.assertIsNone(d.power_w)


class DrmCountersTests(unittest.TestCase):
    def test_drm_counters(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pdev = "0000:03:00.0"
            dev = tmp / "drm" / "card1" / "device"
            dev.mkdir(parents=True)
            (dev / "mem_info_vram_total").write_text(str(16 * GiB))
            (dev / "mem_info_vram_used").write_text(str(1 * GiB))
            (dev / "mem_info_gtt_total").write_text(str(8 * GiB))
            (dev / "mem_info_gtt_used").write_text(str(64 * 1024))
            card = _card(dev, pdev)

            proc = tmp / "proc"
            _write_fd(proc, 100, 10, _fdinfo(26, pdev, gfx=1_000_000_000, vram=3, gtt=2))
            _write_fd(proc, 300, 12, _fdinfo(27, "0000:ff:00.0", gfx=9_999_999_999, vram=4))
            # DRM-looking fdinfo behind a non-DRM fd: the /proc/<pid>/fd target rules it out.
            _write_fd(proc, 400, 13, _fdinfo(28, pdev, gfx=9_999_999_999, vram=4), target="socket:[123]")

            clock = Clock()
            counters = drm.DrmCounters(found=[card], proc=str(proc), clock=clock)
            counters.start()
            clock.tick()

            self.assertTrue(counters.ok)
            procs, adapters = counters.sample()
            self.assertEqual(list(procs), [card.key])
            self.assertIn(100, procs[card.key])
            self.assertNotIn(300, procs[card.key])  # unknown pdev
            self.assertNotIn(400, procs[card.key])  # non-DRM fd
            g = procs[card.key][100]
            self.assertEqual(g.dedicated, 3 * 1024)  # private client, no shared buffers: resident is its own
            self.assertEqual(g.resident_vram, 3 * 1024)
            self.assertEqual(g.shared, 2048)
            self.assertEqual(g.drm_reason, "")
            self.assertFalse(g.paged_available)  # no commit counter: unavailable, not 0
            self.assertEqual(g.committed, 0)
            self.assertEqual(g.engines.get("3d", 0.0), 0.0)  # idle since the baseline
            self.assertEqual(adapters[card.key].dedicated, 1 * GiB)
            self.assertEqual(adapters[card.key].shared, 64 * 1024)

            (dev / "mem_info_vram_used").write_text(str(2 * GiB))
            _write_fd(proc, 100, 10, _fdinfo(26, pdev, gfx=1_500_000_000, vram=1, gtt=2))
            clock.tick()
            procs, adapters = counters.sample()
            self.assertEqual(procs[card.key][100].engines["3d"], 50.0)
            self.assertIn("engine", procs[card.key][100].available)
            self.assertEqual(adapters[card.key].dedicated, 2 * GiB)
            counters.close()

    def test_constructor_does_not_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            dev = Path(tmp) / "drm" / "card1" / "device"
            dev.mkdir(parents=True)
            with patch.object(drm.DrmCounters, "sample") as sample:
                drm.DrmCounters(found=[_card(dev, "0000:03:00.0")], proc=str(Path(tmp) / "proc"))
            sample.assert_not_called()  # an NVIDIA GPU may be selected: no fd walk until start()

    def test_rocm_process_engine_load_is_unknown(self):
        # A HIP process (llama.cpp-ROCm) holds /dev/kfd next to its render node: its VRAM is in fdinfo, but
        # its compute runs on KFD user queues with no per-process busy time, so engines are unavailable.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pdev = "0000:03:00.0"
            dev = tmp / "drm" / "card1" / "device"
            dev.mkdir(parents=True)
            card = _card(dev, pdev)
            proc = tmp / "proc"
            _write_fd(proc, 100, 3, "pos:\t0\nflags:\t02\n", target="/dev/kfd")
            _write_fd(proc, 100, 7, _fdinfo(500, pdev, vram=3891436))
            _write_fd(proc, 200, 7, _fdinfo(501, pdev, gfx=5, vram=4))  # a graphics client, for contrast
            clock = Clock()
            counters = drm.DrmCounters(found=[card], proc=str(proc), clock=clock)
            counters.start()
            clock.tick()
            procs, _ = counters.sample()
            self.assertEqual(procs[card.key][100].dedicated, 3891436 * 1024)
            self.assertTrue(procs[card.key][100].kfd)
            self.assertNotIn("engine", procs[card.key][100].available)
            self.assertIn("engine", procs[card.key][200].available)


class DrmEdgeCaseTests(unittest.TestCase):
    def _counters(self, tmp, fds, unified_video=False, sysfs=True):
        pdev = "0000:03:00.0"
        dev = tmp / "drm" / "card1" / "device"
        dev.mkdir(parents=True)
        if sysfs:
            (dev / "mem_info_vram_used").write_text(str(GiB))
            (dev / "mem_info_gtt_used").write_text("0")
        card = _card(dev, pdev)
        card.unified_video = unified_video
        proc = tmp / "proc"
        for fd_spec in fds:
            pid, fd, text, *target = fd_spec
            _write_fd(proc, pid, fd, text.replace("PDEV", pdev), *target)
        self.clock = Clock()
        counters = drm.DrmCounters(found=[card], proc=str(proc), clock=self.clock)
        counters.start()
        self.clock.tick()
        return card, proc, counters

    def test_old_kernel_without_shared_keys_is_not_summed(self):
        old = "drm-driver:\tamdgpu\ndrm-client-id:\t5\ndrm-pdev:\tPDEV\ndrm-memory-vram:\t4096 KiB\n"
        with tempfile.TemporaryDirectory() as tmp:
            card, _, counters = self._counters(Path(tmp), [(100, 5, old)])
            g = counters.sample()[0][card.key][100]
            self.assertFalse(g.attributed)  # resident may include imports: would double count
            self.assertEqual(g.drm_reason, drm.REASON_LEGACY)
            self.assertEqual(g.dedicated, 0)
            self.assertNotIn("dedicated", g.available)
            self.assertNotIn("shared", g.available)
            self.assertEqual(g.resident_vram, 4096 * 1024)  # still shown as a reference

    def test_any_shared_buffer_leaves_the_process_unattributed(self):
        # drm-shared-* counts by preferred placement and drm-resident-* by current placement, so resident minus
        # shared is not this process's private memory. Any sharing at all: n/a, never a subtraction.
        for extra in ({"shared_vram": 2}, {"shared_gtt": 4}, {"shared_cpu": 6}):
            with self.subTest(extra), tempfile.TemporaryDirectory() as tmp:
                card, _, counters = self._counters(Path(tmp), [(100, 5, _fdinfo(7, "PDEV", vram=3, gtt=10, **extra))])
                g = counters.sample()[0][card.key][100]
                self.assertEqual(g.drm_reason, drm.REASON_SHARED_BUFFERS)
                self.assertIsNone(g.shown_vram)
                self.assertNotIn("shared", g.available)
                self.assertEqual(g.resident_vram, 3 * 1024)
                self.assertEqual(g.resident_gtt, 10 * 1024)
                self.assertIn("engine", g.available)  # sharing buffers doesn't share the client's engine time

    def test_primary_node_is_not_charged(self):
        # logind (root) holds a copy of a compositor's /dev/dri/card* fd that a normal user can't see.
        with tempfile.TemporaryDirectory() as tmp:
            card, _, counters = self._counters(Path(tmp), [(100, 5, _fdinfo(7, "PDEV", vram=3), "/dev/dri/card1")])
            g = counters.sample()[0][card.key][100]
            self.assertEqual(g.drm_reason, drm.REASON_PRIMARY)
            self.assertIsNone(g.shown_vram)

    def test_unparseable_memory_is_unknown(self):
        text = _fdinfo(7, "PDEV", vram=3).replace("drm-resident-vram:\t3 KiB", "drm-resident-vram:\tlots")
        with tempfile.TemporaryDirectory() as tmp:
            card, _, counters = self._counters(Path(tmp), [(100, 5, text)])
            g = counters.sample()[0][card.key][100]
            self.assertIsNone(g.resident_vram)
            self.assertIsNone(g.shown_vram)

    def test_torn_engine_counter_is_unknown(self):
        text = _fdinfo(7, "PDEV").replace("drm-engine-gfx:\t0 ns", "drm-engine-gfx:\t")
        with tempfile.TemporaryDirectory() as tmp:
            card, _, counters = self._counters(Path(tmp), [(100, 5, text)])
            self.assertNotIn("engine", counters.sample()[0][card.key][100].available)

    def test_engine_counter_stepping_back_is_not_counted_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            card, proc, counters = self._counters(Path(tmp), [(100, 5, _fdinfo(7, "PDEV", gfx=1_000_000_000))])
            path = proc / "100" / "fdinfo" / "5"
            path.write_text(_fdinfo(7, card.pdev, gfx=900_000_000))  # the DRM spec allows a brief step back
            self.assertEqual(counters.sample()[0][card.key][100].engines.get("3d", 0.0), 0.0)
            self.clock.tick()
            path.write_text(_fdinfo(7, card.pdev, gfx=1_500_000_000))
            self.assertEqual(counters.sample()[0][card.key][100].engines["3d"], 50.0)  # from the high-water mark

    def test_unified_vcn_reports_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            card, proc, counters = self._counters(Path(tmp), [(100, 5, _fdinfo(9, "PDEV").replace(
                "drm-engine-gfx:\t0 ns", "drm-engine-enc:\t0 ns"))], unified_video=True)
            (proc / "100" / "fdinfo" / "5").write_text(_fdinfo(9, card.pdev).replace(
                "drm-engine-gfx:\t0 ns", "drm-engine-enc:\t1000000000 ns"))
            engines = counters.sample()[0][card.key][100].engines
            self.assertEqual(engines["video"], 100.0)
            self.assertNotIn("encode", engines)

    def test_missing_sysfs_memory_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            card, _, counters = self._counters(Path(tmp), [], sysfs=False)
            a = counters.sample()[1][card.key]
            self.assertIsNone(a.dedicated)
            self.assertIsNone(a.shared)
            self.assertIsNone(drm.read_device(card, 0).mem_used)

    def test_power_average_of_zero_falls_back_to_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            dev = Path(tmp) / "device"
            hwmon = dev / "hwmon" / "hwmon0"
            hwmon.mkdir(parents=True)
            (hwmon / "power1_average").write_text("0\n")
            (hwmon / "power1_input").write_text("42000000\n")
            self.assertEqual(drm.read_device(_card(dev, "0000:03:00.0"), 0).power_w, 42.0)
            (hwmon / "power1_average").unlink()
            (hwmon / "power1_input").write_text("0\n")
            self.assertEqual(drm.read_device(_card(dev, "0000:03:00.0"), 0).power_w, 0.0)  # measured 0 stays 0

    def test_vcn_major_from_ip_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            uvd = Path(tmp) / "ip_discovery" / "die" / "0" / "UVD" / "0"
            uvd.mkdir(parents=True)
            (uvd / "major").write_text("5\n")
            self.assertEqual(drm._vcn_major(tmp), 5)
            self.assertEqual(drm._vcn_major(str(Path(tmp) / "none")), 0)


class DrmScanTests(unittest.TestCase):
    PDEV = "0000:03:00.0"

    def _setup(self, tmp, timer=None):
        dev = tmp / "drm" / "card1" / "device"
        dev.mkdir(parents=True)
        card = _card(dev, self.PDEV)
        proc = tmp / "proc"
        proc.mkdir()
        self.clock = Clock()
        kw = {"clock": self.clock}
        if timer is not None:
            kw["timer"] = timer
        return card, proc, kw

    def test_new_client_engine_load_is_unknown_until_two_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            card, proc, kw = self._setup(Path(tmp))
            _write_fd(proc, 100, 5, _fdinfo(1, self.PDEV, gfx=0, vram=4096))
            counters = drm.DrmCounters(found=[card], proc=str(proc), **kw)
            counters.start()
            self.clock.tick()
            _write_fd(proc, 200, 5, _fdinfo(2, self.PDEV, gfx=500_000_000, vram=4096))
            procs = counters.sample()[0][card.key]
            self.assertIn("engine", procs[100].available)  # in the baseline: a real, measured idle
            self.assertNotIn("engine", procs[200].available)  # first seen now: unknown, not 0
            self.assertEqual(procs[200].dedicated, 4096 * 1024)  # memory doesn't need two samples
            self.assertFalse(procs[200].kfd)
            self.clock.tick()
            _write_fd(proc, 200, 5, _fdinfo(2, self.PDEV, gfx=1_000_000_000, vram=4096))
            self.assertEqual(counters.sample()[0][card.key][200].engines["3d"], 50.0)

    def test_engine_first_reported_counts_from_zero(self):
        # amdgpu omits an engine until it has run, so its whole counter is new work.
        with tempfile.TemporaryDirectory() as tmp:
            card, proc, kw = self._setup(Path(tmp))
            _write_fd(proc, 100, 5, _fdinfo(1, self.PDEV).replace("drm-engine-gfx:\t0 ns\n", ""))
            counters = drm.DrmCounters(found=[card], proc=str(proc), **kw)
            counters.start()
            self.clock.tick()
            _write_fd(proc, 100, 5, _fdinfo(1, self.PDEV, gfx=250_000_000))
            self.assertEqual(counters.sample()[0][card.key][100].engines["3d"], 25.0)

    def test_client_held_by_several_processes_is_charged_to_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            card, proc, kw = self._setup(Path(tmp))
            _write_fd(proc, 100, 5, _fdinfo(1, self.PDEV, gfx=0, vram=4096))
            _write_fd(proc, 200, 9, _fdinfo(1, self.PDEV, gfx=0, vram=4096))
            _write_fd(proc, 300, 5, _fdinfo(2, self.PDEV, gfx=0, vram=8192))
            counters = drm.DrmCounters(found=[card], proc=str(proc), **kw)
            counters.start()
            self.clock.tick()
            procs = counters.sample()[0][card.key]
            for pid in (100, 200):
                self.assertEqual(procs[pid].drm_reason, drm.REASON_SHARED_CLIENT)
                self.assertIsNone(procs[pid].shown_vram)
                self.assertNotIn("engine", procs[pid].available)
            self.assertEqual(procs[300].shown_vram, 8192 * 1024)
            # The other rows still add up: the shared client's bytes are left out, not the whole total.
            self.assertEqual(sum(g.dedicated for g in procs.values()), 8192 * 1024)

    def test_private_and_co_owned_clients_leave_the_whole_process_unattributed(self):
        # pid 100 alone holds client 1 and shares client 2 with pid 200: charging client 1 to it would depend on
        # which other processes happened to be visible, so neither process gets bytes.
        with tempfile.TemporaryDirectory() as tmp:
            card, proc, kw = self._setup(Path(tmp))
            _write_fd(proc, 100, 5, _fdinfo(1, self.PDEV, vram=1024))
            _write_fd(proc, 100, 6, _fdinfo(2, self.PDEV, vram=2048))
            _write_fd(proc, 200, 5, _fdinfo(2, self.PDEV, vram=2048))
            counters = drm.DrmCounters(found=[card], proc=str(proc), **kw)
            counters.start()
            procs = counters.sample()[0][card.key]
            for pid in (100, 200):
                self.assertEqual(procs[pid].drm_reason, drm.REASON_SHARED_CLIENT)
                self.assertEqual(procs[pid].dedicated, 0)
                self.assertNotIn("dedicated", procs[pid].available)
            self.assertEqual(procs[100].resident_vram, 3072 * 1024)

    def test_uninspectable_processes_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            card, proc, kw = self._setup(Path(tmp))
            _write_fd(proc, 100, 5, _fdinfo(1, self.PDEV, vram=4096))
            (proc / "200" / "fd").mkdir(parents=True)
            real = os.scandir

            def scandir(path):
                if str(path).endswith(os.path.join("200", "fd")):
                    raise PermissionError(path)
                return real(path)

            with patch.object(drm.os, "scandir", scandir):
                counters = drm.DrmCounters(found=[card], proc=str(proc), **kw)
                self.assertIn(100, counters.sample()[0][card.key])
            self.assertIn("1 processes not inspectable", counters.status)
            counters.sample()
            self.assertEqual(counters.status, "")  # readable now

    def test_unreadable_fd_after_listing_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            card, proc, kw = self._setup(Path(tmp))
            _write_fd(proc, 100, 5, _fdinfo(1, self.PDEV, vram=4096))
            _write_fd(proc, 100, 6, _fdinfo(2, self.PDEV, vram=4096))
            real = os.readlink

            def readlink(path):
                if str(path).endswith(os.path.join("100", "fd", "6")):
                    raise PermissionError(path)
                return real(path)

            counters = drm.DrmCounters(found=[card], proc=str(proc), **kw)
            with patch.object(drm.os, "readlink", readlink):
                g = counters.sample()[0][card.key][100]
            self.assertEqual(g.drm_reason, drm.REASON_INCOMPLETE)  # client 2 might be shared: don't charge 1
            self.assertIsNone(g.shown_vram)

    def test_unreadable_fdinfo_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            card, proc, kw = self._setup(Path(tmp))
            _write_fd(proc, 100, 5, _fdinfo(1, self.PDEV, vram=4096))
            _write_fd(proc, 100, 6, _fdinfo(2, self.PDEV, vram=4096))
            real = open

            def guarded(path, *a, **k):
                if str(path).endswith(os.path.join("fdinfo", "6")):
                    raise PermissionError(path)
                return real(path, *a, **k)

            counters = drm.DrmCounters(found=[card], proc=str(proc), **kw)
            with patch("builtins.open", guarded):
                g = counters.sample()[0][card.key][100]
            self.assertEqual(g.drm_reason, drm.REASON_INCOMPLETE)

    def test_reused_pid_drops_cached_fds(self):
        with tempfile.TemporaryDirectory() as tmp:
            card, proc, kw = self._setup(Path(tmp))
            _write_fd(proc, 100, 5, _fdinfo(1, self.PDEV, vram=4096))
            (proc / "100" / "stat").write_text("100 (old) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 1000 0\n")
            counters = drm.DrmCounters(found=[card], proc=str(proc), **kw)
            counters.sample()
            self.assertIn(100, counters._known)
            (proc / "100" / "stat").write_text("100 (new) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 2000 0\n")
            counters.scan_budget = 0.0  # nothing is rescanned: only the cache could keep pid 100
            self.assertNotIn(100, counters.sample()[0][card.key])
            self.assertNotIn(100, counters._known)

    def test_scan_budget_keeps_rows_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            ticks = [0.0]

            def timer():
                ticks[0] += 1.0
                return ticks[0]

            card, proc, kw = self._setup(Path(tmp), timer)
            for pid in (100, 200, 300, 400):
                _write_fd(proc, pid, 5, _fdinfo(pid, self.PDEV, gfx=0, vram=4096))
            counters = drm.DrmCounters(found=[card], proc=str(proc), **kw)
            counters.start()  # the baseline is never limited
            self.assertEqual(sorted(counters._known), [100, 200, 300, 400])
            counters.scan_budget = 2.5  # one process per refresh with this timer
            _write_fd(proc, 500, 5, _fdinfo(500, self.PDEV, gfx=0, vram=4096))
            rows = counters.sample()[0][card.key]
            self.assertTrue({100, 200, 300, 400} <= set(rows))  # no row flickers while the scan is partial
            self.assertIn(500, rows)  # processes not seen before are scanned first
            self.assertIn("scan budget reached", counters.status)
            counters.scan_budget = None
            counters.sample()
            self.assertEqual(counters.status, "")

    def test_budget_is_checked_inside_one_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            ticks = [0.0]

            def timer():
                ticks[0] += 1.0
                return ticks[0]

            card, proc, kw = self._setup(Path(tmp), timer)
            _write_fd(proc, 50, 5, _fdinfo(1, self.PDEV, vram=4096))
            for fd in range(200):  # a process with many fds
                _write_fd(proc, 100, fd, "pos:\t0\n", target=f"socket:[{fd}]")
            counters = drm.DrmCounters(found=[card], proc=str(proc), **kw)
            counters.scan_budget = 3.5  # pid 50 and the first 127 fds of pid 100
            counters.sample()
            self.assertIn(50, counters._known)  # the first process of a refresh is listed in full
            self.assertNotIn(100, counters._known)  # interrupted partway: deferred, not cached as fd-less
            self.assertIn("scan budget reached", counters.status)

    def test_one_huge_process_does_not_stall_the_rotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            ticks = [0.0]

            def timer():
                ticks[0] += 1.0
                return ticks[0]

            card, proc, kw = self._setup(Path(tmp), timer)
            for fd in range(200):
                _write_fd(proc, 100, fd, "pos:\t0\n", target=f"socket:[{fd}]")
            _write_fd(proc, 200, 5, _fdinfo(2, self.PDEV, vram=4096))
            counters = drm.DrmCounters(found=[card], proc=str(proc), **kw)
            counters.scan_budget = 1.5
            for _ in range(3):
                counters.sample()
            self.assertEqual(sorted(counters._known), [100, 200])

    def test_partial_scan_charges_no_one(self):
        # A process the budget skipped may hold the same client, so single ownership isn't proven this refresh.
        with tempfile.TemporaryDirectory() as tmp:
            ticks = [0.0]

            def timer():
                ticks[0] += 1.0
                return ticks[0]

            card, proc, kw = self._setup(Path(tmp), timer)
            for pid in (100, 200):
                _write_fd(proc, pid, 5, _fdinfo(pid, self.PDEV, vram=4096))
            counters = drm.DrmCounters(found=[card], proc=str(proc), **kw)
            counters.start()
            counters.scan_budget = 1.5
            procs = counters.sample()[0][card.key]
            self.assertTrue(counters.partial)
            for pid in (100, 200):
                self.assertEqual(procs[pid].drm_reason, drm.REASON_PARTIAL_SCAN)
                self.assertIsNone(procs[pid].shown_vram)
                self.assertNotIn("engine", procs[pid].available)
                self.assertEqual(procs[pid].resident_vram, 4096 * 1024)

    def test_rotation_resumes_at_a_known_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            ticks = [0.0]

            def timer():
                ticks[0] += 1.0
                return ticks[0]

            card, proc, kw = self._setup(Path(tmp), timer)
            for pid in (100, 200, 300):
                _write_fd(proc, pid, 5, _fdinfo(pid, self.PDEV, vram=4096))
            counters = drm.DrmCounters(found=[card], proc=str(proc), **kw)
            counters.start()
            for pid in (900, 901):  # new processes are scanned first and run out the budget
                _write_fd(proc, pid, 5, _fdinfo(pid, self.PDEV, vram=4096))
            counters._cursor = 200
            counters.scan_budget = 1.5
            counters.sample()
            self.assertEqual(counters._cursor, 200)  # not 901: known processes 200 and 300 aren't skipped

    def test_incomplete_process_has_no_partial_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            card, proc, kw = self._setup(Path(tmp))
            _write_fd(proc, 100, 5, _fdinfo(1, self.PDEV, gfx=0, vram=4096))
            _write_fd(proc, 100, 6, _fdinfo(2, self.PDEV, gfx=0, vram=4096))
            real = open

            def guarded(path, *a, **k):
                if str(path).endswith(os.path.join("fdinfo", "6")):
                    raise PermissionError(path)
                return real(path, *a, **k)

            counters = drm.DrmCounters(found=[card], proc=str(proc), **kw)
            with patch("builtins.open", guarded):
                counters.start()
                self.clock.tick()
                g = counters.sample()[0][card.key][100]
            self.assertIsNone(g.resident_vram)
            self.assertNotIn("engine", g.available)
            self.assertIn("1 processes whose fds or fdinfo", counters.status)

class DrmReportTests(unittest.TestCase):
    def test_json_and_text_keep_unknown_apart_from_zero(self):
        from gputaskman.collector import ProcRow, Snapshot
        from gputaskman.pdh import ProcGpu
        from gputaskman.procinfo import ProcDetail
        from gputaskman.report import DRM_NOTES, to_dict, to_text

        private = ProcGpu(dedicated=GiB, shared=0, available={"dedicated", "shared", "engine"},
                          resident_vram=float(GiB), resident_gtt=0.0)
        compositor = ProcGpu(attributed=False, available={"engine"}, resident_vram=float(2 * GiB),
                             resident_gtt=None, drm_reason=drm.REASON_SHARED_BUFFERS)
        rows = [ProcRow(100, "llama-server", "LLM", private, ProcDetail(100, "llama-server")),
                ProcRow(200, "kwin_wayland", "compositor", compositor, ProcDetail(200, "kwin_wayland"))]
        snap = Snapshot(taken=1, adapter=None, adapters=[], nv=None, driver="", cuda="", vram_total=16 * GiB,
                        vram_used=4 * GiB, shared_used=0, backend="drm", procs=rows, shared_is_spill=True)
        data = to_dict(snap)
        self.assertTrue(data["shared_is_spill"])
        llama, kwin = data["processes"]
        self.assertEqual((llama["vram_bytes"], llama["shared_bytes"], llama["vram_resident_bytes"]), (GiB, 0, GiB))
        self.assertIsNone(llama["vram_unattributed_reason"])
        self.assertEqual((kwin["vram_bytes"], kwin["shared_bytes"]), (None, None))
        self.assertEqual((kwin["vram_resident_bytes"], kwin["gtt_resident_bytes"]), (2 * GiB, None))
        self.assertEqual(kwin["vram_unattributed_reason"], "shared_buffers")
        text = to_text(snap)
        line = next(x for x in text.splitlines() if x.lstrip().startswith("200"))
        self.assertEqual(line.split()[3:5], ["n/a", "N/A"])  # VRAM and SPILL unknown, not blank or 0
        self.assertIn(DRM_NOTES["shared_buffers"], text)
        self.assertNotIn(DRM_NOTES["legacy"], text)


class EngineAndSelectionTests(unittest.TestCase):
    def test_engine_names(self):
        self.assertEqual(drm._engine_key("gfx"), "3d")
        self.assertEqual(drm._engine_key("compute"), "compute")
        self.assertEqual(drm._engine_key("dma"), "copy")
        self.assertEqual(drm._engine_key("enc_1"), "encode")
        self.assertEqual(drm._engine_key("jpeg"), "decode")
        self.assertEqual(drm._engine_key("vpe"), "other")
        self.assertEqual(drm._engine_key("enc", unified_video=True), "video")
        self.assertEqual(drm._engine_key("gfx", unified_video=True), "3d")
        self.assertEqual(drm._engine_key("rcs0", intel=True), "3d")
        self.assertEqual(drm._engine_key("ccs0", intel=True), "compute")
        self.assertEqual(drm._engine_key("bcs0", intel=True), "copy")
        self.assertEqual(drm._engine_key("vcs0", intel=True), "video")
        self.assertEqual(drm._engine_key("vecs0", intel=True), "video")  # video-enhance is post-processing, not encode
        self.assertEqual(drm._engine_key("rcs0", intel=False), "other")  # Intel-only prefix on AMD

    def test_amd_selectable_next_to_nvidia_on_linux(self):
        from gputaskman import collector
        from gputaskman.dxgi import Adapter

        nvidia = Adapter(1, "NVIDIA GPU", 0x10DE, 8 * GiB, 0, False)
        amd = Adapter(2, "AMD Radeon 610M", 0x1002, 2 * GiB, 0, False)
        with patch.object(collector, "BACKEND", "nvml"):
            self.assertIs(collector.select_adapter([amd, nvidia], 0), nvidia)
            self.assertIs(collector.select_adapter([amd, nvidia], 1), amd)

    def test_intel_selectable_next_to_nvidia_on_linux(self):
        from gputaskman import collector
        from gputaskman.dxgi import Adapter

        nvidia = Adapter(1, "NVIDIA GPU", 0x10DE, 8 * GiB, 0, False)
        intel = Adapter(2, "Intel Arc A770", 0x8086, 16 * GiB, 0, False)
        with patch.object(collector, "BACKEND", "nvml"):
            self.assertIs(collector.select_adapter([intel, nvidia], 0), nvidia)
            self.assertIs(collector.select_adapter([intel, nvidia], 1), intel)


class IntelCardTests(unittest.TestCase):
    """Intel (i915/xe) cards through the same DRM fdinfo backend, from a synthetic sysfs tree."""

    INTEL = drm.DrmCard(key=12288, pdev="0000:03:00.0", sysfs="/x/card2/device", driver="i915",
                        vendor_id=0x8086, device_id=0x56A0, name="Intel Arc A770",
                        vram_total=16 * GiB, integrated=False, intel=True)
    IGPU = drm.DrmCard(key=32, pdev="0000:00:02.0", sysfs="/x/card1/device", driver="i915",
                       vendor_id=0x8086, device_id=0xA782, name="Intel UHD Graphics",
                       vram_total=0, integrated=True, intel=True)
    AMD = drm.DrmCard(key=0, pdev="0000:43:00.0", sysfs="/x/amd/device", driver="amdgpu",
                      vendor_id=0x1002, device_id=0x7480, name="AMD GPU", vram_total=8 * GiB,
                      integrated=False)

    def make(self, cards):
        return drm.DrmCounters(found=cards)

    def test_intel_engine_names(self):
        self.assertEqual(drm._engine_key("rcs0", intel=True), "3d")
        self.assertEqual(drm._engine_key("ccs0", intel=True), "compute")
        self.assertEqual(drm._engine_key("bcs0", intel=True), "copy")
        self.assertEqual(drm._engine_key("vcs0", intel=True), "video")
        self.assertEqual(drm._engine_key("vecs0", intel=True), "video")
        self.assertEqual(drm._engine_key("rcs0", intel=False), "other")

    def test_intel_engine_classes_as_i915_prints_them(self):
        # i915 names whole classes (i915_drm_client.c uabi_class_names), not command streamers.
        for name, expected in (("render", "3d"), ("copy", "copy"), ("video", "video"),
                               ("video-enhance", "video"), ("compute", "compute")):
            with self.subTest(name=name):
                self.assertEqual(drm._engine_key(name, intel=True), expected)
        self.assertEqual(drm._engine_key("something-new", intel=True), "other")
        self.assertEqual(drm._engine_key("render", intel=False), "other")  # not an AMD engine name

    def test_intel_card_is_recognized_and_amd_unchanged(self):
        counters = self.make([self.INTEL])
        self.assertTrue(counters.ok)
        self.assertEqual([c.pdev for c in counters.cards], ["0000:03:00.0"])
        self.assertTrue(counters.cards[0].intel)

    def test_intel_fdinfo_fields_parse(self):
        text = ("drm-driver:\ti915\n"
                "drm-client-id:\t99\n"
                "drm-pdev:\t0000:03:00.0\n"
                "drm-engine-rcs0:\t1200000000 ns\n"
                "drm-engine-bcs0:\t400000000 ns\n"
                "drm-engine-ccs0:\t800000000 ns\n"
                "drm-engine-vcs0:\t200000000 ns\n"
                "drm-engine-vecs0:\t100000000 ns\n"
                "drm-resident-local0:\t1073741824\n"
                "drm-resident-stolen-local0:\t1048576\n"
                "drm-resident-system0:\t268435456\n")
        fields = drm.parse_fdinfo(text)
        self.assertEqual(fields["driver"], "i915")
        self.assertEqual(fields["pdev"], "0000:03:00.0")
        self.assertEqual(fields["client_id"], 99)
        self.assertEqual(fields["engines"], {"rcs0": 1_200_000_000, "bcs0": 400_000_000,
                                             "ccs0": 800_000_000, "vcs0": 200_000_000,
                                             "vecs0": 100_000_000})
        self.assertEqual(fields["vram"], GiB + 1_048_576)  # local0 + stolen-local0
        self.assertEqual(fields["gtt"], 268_435_456)  # system0
        self.assertIsNone(fields["shared"])  # shared gate needs both Intel shared regions

    def test_intel_missing_regions_stay_unknown_not_zero(self):
        text = ("drm-driver:\txe\n"
                "drm-client-id:\t7\n"
                "drm-engine-rcs0:\t1000 ns\n")
        fields = drm.parse_fdinfo(text)
        self.assertIsNone(fields["vram"])  # an iGPU has no local0: unknown, not 0
        self.assertIsNone(fields["gtt"])

    def test_intel_adapters_trust_the_client_stamp(self):
        self.INTEL.vram_used = 15 * GiB
        counters = self.make([self.INTEL])
        out = counters._adapters()
        self.assertEqual(out[self.INTEL.key].dedicated, float(15 * GiB))
        self.assertIsNone(out[self.INTEL.key].shared)  # no device-wide shared counter on Intel
        self.IGPU.vram_used = None
        counters = self.make([self.IGPU])
        out = counters._adapters()
        self.assertIsNone(out[self.IGPU.key].dedicated)
        self.assertIsNone(out[self.IGPU.key].shared)


class IntelFdinfoTests(unittest.TestCase):
    """fdinfo laid out the way the Intel kernels print it (the fixtures at the top of this file)."""

    def test_i915_discrete_client(self):
        info = drm.parse_fdinfo(I915_FDINFO)
        self.assertEqual((info["driver"], info["client_id"], info["pdev"]), ("i915", 42, "0000:03:00.0"))
        self.assertEqual(info["vram"], 2048 * MiB + 64 * 1024)  # local0 plus stolen-local0 (the fixture's 64 KiB is synthetic)
        self.assertEqual(info["gtt"], 6 * MiB)  # system0
        self.assertEqual(info["shared"], 0)  # both shared regions are reported, and empty
        # Only the drm-engine-<class> counters are engines; the capacity lines are not.
        self.assertEqual(info["engines"], {"render": 5308400172, "copy": 1200, "video": 0,
                                           "video-enhance": 0, "compute": 750_000_000})

    def test_i915_integrated_client(self):
        info = drm.parse_fdinfo(I915_IGPU_FDINFO)
        self.assertIsNone(info["vram"])  # no local0 on an integrated GPU: unknown, not 0
        self.assertEqual(info["gtt"], 60 * MiB)  # system0 plus an empty stolen-system0
        self.assertEqual(info["shared"], 0)  # no local0 to require: the gate needs only system0, empty here
        self.assertEqual(set(info["engines"]), {"render", "copy", "video", "video-enhance"})

    def test_i915_integrated_client_with_shared_buffers(self):
        # The Vivobook's vkmark client under load: resident system memory, some of it shared
        # (the compositor imports the window), so the bytes stay unattributed but measurable.
        info = drm.parse_fdinfo(I915_IGPU_SHARED_FDINFO)
        self.assertIsNone(info["vram"])
        self.assertEqual(info["gtt"], 37476 * KiB)
        self.assertEqual(info["shared"], 8576 * KiB)

    def test_i915_discrete_client_missing_shared_local0_stays_legacy(self):
        # A discrete card prints local0 regions, so a missing drm-shared-local0 is a kernel gap,
        # not an integrated GPU: the gate stays closed.
        text = "\n".join(line for line in I915_FDINFO.splitlines() if "drm-shared-local0" not in line)
        self.assertIn("drm-resident-local0", text)
        self.assertIsNone(drm.parse_fdinfo(text)["shared"])

    def test_i915_without_engine_lines_has_unknown_load(self):
        info = drm.parse_fdinfo(I915_GEN7_FDINFO)
        self.assertIsNone(info["engines"])  # unknown, not an empty (zero) load
        self.assertEqual(info["gtt"], 4 * MiB)  # the memory is still read

    def test_i915_unparseable_region_makes_that_figure_unknown(self):
        # Treating a region that won't parse as 0 would show a smaller number than the process holds.
        info = drm.parse_fdinfo(I915_FDINFO.replace("drm-resident-local0:\t2048 MiB", "drm-resident-local0:\tlots"))
        self.assertIsNone(info["vram"])  # stolen-local0 still parses, but the local0 half is unknown
        self.assertEqual(info["gtt"], 6 * MiB)  # the other figure is unaffected
        info = drm.parse_fdinfo(I915_FDINFO.replace("drm-resident-system0:\t6 MiB", "drm-resident-system0:\t?"))
        self.assertIsNone(info["gtt"])
        self.assertEqual(info["vram"], 2048 * MiB + 64 * 1024)

    def test_i915_region_it_does_not_print_adds_nothing(self):
        # i915 prints a line for every region the device has, so a missing stolen-local0 line is a region it lacks.
        text = "\n".join(line for line in I915_FDINFO.splitlines() if "stolen-local0" not in line)
        self.assertEqual(drm.parse_fdinfo(text)["vram"], 2048 * MiB)

    def test_xe_client_reports_neither_memory_nor_load(self):
        info = drm.parse_fdinfo(XE_FDINFO)
        self.assertEqual((info["driver"], info["client_id"], info["pdev"]), ("xe", 3, "0000:03:00.0"))
        # xe calls its regions system/gtt/vram0/stolen and reports drm-cycles-*; none of that is decoded yet.
        for field in ("vram", "gtt", "shared", "engines"):
            with self.subTest(field=field):
                self.assertIsNone(info[field])


class IntelBarTests(unittest.TestCase):
    """cards() takes an Intel card's total from the size of its local-memory BAR file. The sizes are faked: a real
    BAR is GiBs of address space, not data."""

    def _cards(self, bars: dict[str, int], driver="i915"):
        real_getsize = os.path.getsize

        def getsize(path):
            name = os.path.basename(path)
            if name not in ("resource2_wc", "resource2"):
                return real_getsize(path)
            if name not in bars:
                raise FileNotFoundError(path)
            return bars[name]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_intel_card(root, "card0", "0000:03:00.0", 0x56A0, driver)
            with patch.object(os.path, "getsize", getsize):
                return drm.cards(str(root / "drm"), str(root / "pci.ids"))

    def test_resizable_bar_marks_a_discrete_card_and_sizes_it(self):
        (card,) = self._cards({"resource2_wc": 16 * GiB})
        self.assertTrue(card.intel)
        self.assertEqual((card.vram_total, card.integrated), (16 * GiB, False))

    def test_resource2_alone_is_enough(self):
        (card,) = self._cards({"resource2": 8 * GiB})  # no write-combining alias to read
        self.assertEqual((card.vram_total, card.integrated), (8 * GiB, False))

    def test_empty_resource2_wc_falls_back_to_resource2(self):
        (card,) = self._cards({"resource2_wc": 0, "resource2": 8 * GiB})
        self.assertEqual(card.vram_total, 8 * GiB)

    def test_small_or_missing_window_is_an_unknown_total_not_a_size(self):
        (card,) = self._cards({"resource2_wc": 256 * MiB})  # an iGPU's system-memory window
        self.assertEqual((card.vram_total, card.integrated), (0, True))
        (card,) = self._cards({})  # no BAR file at all
        self.assertEqual((card.vram_total, card.integrated), (0, True))

    def test_xe_card_is_sized_the_same_way(self):
        (card,) = self._cards({"resource2_wc": 16 * GiB}, driver="xe")
        self.assertEqual((card.driver, card.intel, card.vram_total), ("xe", True, 16 * GiB))


class IntelScanTests(unittest.TestCase):
    """DrmCounters.sample() over Intel clients in a fake /proc. Each test builds its own card, because a sample
    stamps the card's used memory."""

    PDEV = "0000:03:00.0"

    def _setup(self, tmp, driver="i915", vram_total=16 * GiB, integrated=False):
        proc = Path(tmp) / "proc"
        proc.mkdir()
        card = drm.DrmCard(key=drm.pci_key(self.PDEV), pdev=self.PDEV, sysfs=str(Path(tmp) / "card1" / "device"),
                           driver=driver, vendor_id=0x8086, device_id=0x56A0, name="Intel Arc A770",
                           vram_total=vram_total, integrated=integrated, intel=True)
        self.clock = Clock()
        return card, proc, drm.DrmCounters(found=[card], proc=str(proc), clock=self.clock)

    def test_local_memory_in_fdinfo_makes_a_card_with_a_small_window_discrete(self):
        # An Arc without resizable BAR has a window under 1 GiB, so cards() calls it integrated; only a card with
        # local memory prints local0, and that is what shows it is discrete.
        with tempfile.TemporaryDirectory() as tmp:
            card, proc, counters = self._setup(tmp, vram_total=0, integrated=True)
            counters.sample()
            self.assertTrue(card.integrated)  # no client yet: nothing shows otherwise
            _write_fd(proc, 100, 5, _i915_fdinfo(1, self.PDEV, local_mib=64))
            counters.sample()
            self.assertFalse(card.integrated)
            self.assertEqual(card.vram_total, 0)  # and its total is still unknown, not invented

    def test_integrated_gpu_stays_integrated_when_its_clients_print_no_local_memory(self):
        pdev = "0000:00:02.0"
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp) / "proc"
            proc.mkdir()
            card = drm.DrmCard(key=drm.pci_key(pdev), pdev=pdev, sysfs=str(Path(tmp) / "card0" / "device"),
                               driver="i915", vendor_id=0x8086, device_id=0xA782, name="Intel UHD Graphics",
                               vram_total=0, integrated=True, intel=True)
            counters = drm.DrmCounters(found=[card], proc=str(proc), clock=Clock())
            _write_fd(proc, 100, 5, I915_IGPU_FDINFO)
            counters.sample()
            self.assertTrue(card.integrated)

    def test_render_and_video_load_are_measured_between_two_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            card, proc, counters = self._setup(tmp)
            _write_fd(proc, 100, 5, _i915_fdinfo(1, self.PDEV, local_mib=256, system_mib=8))
            counters.start()
            self.clock.tick()
            _write_fd(proc, 100, 5, _i915_fdinfo(1, self.PDEV, render=500_000_000, video=250_000_000,
                                                 local_mib=256, system_mib=8))
            g = counters.sample()[0][card.key][100]
            self.assertIn("engine", g.available)
            self.assertEqual(dict(g.engines), {"3d": 50.0, "copy": 0.0, "video": 25.0})  # none fell into "other"
            self.assertEqual(g.gpu_pct, 50.0)
            self.assertEqual((g.drm_reason, g.dedicated, g.shared), ("", 256 * MiB, 8 * MiB))

    def test_xe_client_is_unknown_not_idle_and_not_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            card, proc, counters = self._setup(tmp, driver="xe")
            _write_fd(proc, 100, 5, XE_FDINFO)
            counters.start()
            self.clock.tick()
            procs, adapters = counters.sample()
            g = procs[card.key][100]
            self.assertEqual(g.drm_reason, drm.REASON_LEGACY)
            self.assertNotIn("engine", g.available)  # drm-cycles-* is not decoded: unknown, not 0%
            self.assertNotIn("dedicated", g.available)
            self.assertIsNone(g.resident_vram)
            self.assertIsNone(adapters[card.key].dedicated)  # and so is the card's used memory

    def test_unknown_engine_load_does_not_hide_measured_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            card, proc, counters = self._setup(tmp)
            _write_fd(proc, 100, 5, _i915_fdinfo(1, self.PDEV, local_mib=64, engines=False))
            counters.start()
            self.clock.tick()
            g = counters.sample()[0][card.key][100]
            self.assertNotIn("engine", g.available)  # no drm-engine lines: the load is unknown
            self.assertIn("dedicated", g.available)  # and that has no bearing on the memory
            self.assertEqual(g.dedicated, 64 * MiB)

    def test_integrated_gpu_charges_private_system_memory(self):
        # An integrated GPU has no local memory: a private client's system0 bytes are its GPU memory.
        pdev = "0000:00:02.0"
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp) / "proc"
            proc.mkdir()
            card = drm.DrmCard(key=drm.pci_key(pdev), pdev=pdev, sysfs=str(Path(tmp) / "card0" / "device"),
                               driver="i915", vendor_id=0x8086, device_id=0xA782, name="Intel UHD Graphics",
                               vram_total=0, integrated=True, intel=True)
            clock = Clock()
            counters = drm.DrmCounters(found=[card], proc=str(proc), clock=clock)
            _write_fd(proc, 100, 5, I915_IGPU_FDINFO)
            counters.start()
            clock.tick()
            _write_fd(proc, 100, 5, I915_IGPU_FDINFO.replace("render:\t1200000000", "render:\t1700000000"))
            procs, adapters = counters.sample()
            g = procs[card.key][100]
            self.assertEqual((g.drm_reason, g.dedicated), ("", 60 * MiB))
            self.assertNotIn("shared", g.available)  # system0 is memory here, not spill
            self.assertIn("engine", g.available)
            self.assertEqual(g.engines["3d"], 50.0)
            self.assertEqual(adapters[card.key].dedicated, 60 * MiB)  # the stamped device total

    def test_integrated_gpu_with_shared_buffers_stays_unattributed(self):
        # The shared case from the Vivobook: measurable and referenced, but charged to nobody.
        pdev = "0000:00:02.0"
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp) / "proc"
            proc.mkdir()
            card = drm.DrmCard(key=drm.pci_key(pdev), pdev=pdev, sysfs=str(Path(tmp) / "card0" / "device"),
                               driver="i915", vendor_id=0x8086, device_id=0xA782, name="Intel UHD Graphics",
                               vram_total=0, integrated=True, intel=True)
            counters = drm.DrmCounters(found=[card], proc=str(proc), clock=Clock())
            _write_fd(proc, 100, 5, I915_IGPU_SHARED_FDINFO)
            procs, adapters = counters.sample()
            g = procs[card.key][100]
            self.assertEqual(g.drm_reason, drm.REASON_SHARED_BUFFERS)
            self.assertNotIn("dedicated", g.available)
            self.assertEqual(g.resident_gtt, 37476 * KiB)  # still shown as a reference
            # The stamped device total sums resident system memory (shared buffers can double-count it).
            self.assertEqual(adapters[card.key].dedicated, 37476 * KiB)

    def test_mixed_discrete_and_integrated_cards_charge_each_by_its_rule(self):
        disc_pdev = "0000:03:00.0"
        igpu_pdev = "0000:00:02.0"
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp) / "proc"
            proc.mkdir()
            disc = drm.DrmCard(key=drm.pci_key(disc_pdev), pdev=disc_pdev, sysfs=str(Path(tmp) / "d" / "device"),
                               driver="i915", vendor_id=0x8086, device_id=0x56A0, name="Intel Arc A770",
                               vram_total=16 * GiB, integrated=False, intel=True)
            igpu = drm.DrmCard(key=drm.pci_key(igpu_pdev), pdev=igpu_pdev, sysfs=str(Path(tmp) / "i" / "device"),
                               driver="i915", vendor_id=0x8086, device_id=0xA782, name="Intel UHD Graphics",
                               vram_total=0, integrated=True, intel=True)
            counters = drm.DrmCounters(found=[disc, igpu], proc=str(proc), clock=Clock())
            _write_fd(proc, 100, 5, _i915_fdinfo(1, disc_pdev, local_mib=64, system_mib=8))
            _write_fd(proc, 200, 6, I915_IGPU_FDINFO)
            procs, adapters = counters.sample()
            g_disc = procs[disc.key][100]
            self.assertEqual((g_disc.dedicated, g_disc.shared), (64 * MiB, 8 * MiB))
            g_igpu = procs[igpu.key][200]
            self.assertEqual(g_igpu.dedicated, 60 * MiB)
            self.assertNotIn("shared", g_igpu.available)
            self.assertEqual(adapters[disc.key].dedicated, 64 * MiB)
            self.assertEqual(adapters[igpu.key].dedicated, 60 * MiB)

    def test_start_arms_device_wide_baselines(self):
        # --once sleeps a second after Collector init, so the primed baselines make its
        # single sample a rate over a real interval instead of a first-sample N/A.
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp) / "proc"
            proc.mkdir()
            card = drm.DrmCard(key=32, pdev="0000:00:02.0", sysfs=str(Path(tmp) / "card0" / "device"),
                               driver="i915", vendor_id=0x8086, device_id=0xA7AD, name="Intel Intel Graphics",
                               vram_total=0, integrated=True, intel=True)
            counters = drm.DrmCounters(found=[card], proc=str(proc), clock=Clock())
            with patch.object(drm, "_intel_pmu_rates") as pmu, \
                    patch.object(drm, "_intel_rapl_w") as rapl:
                counters.start()
            pmu.assert_called_once_with(card.sysfs)
            rapl.assert_called_once_with()

    def test_device_used_sums_each_client_once_and_is_unknown_when_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            card, proc, counters = self._setup(tmp)
            self.assertIsNone(counters.sample()[1][card.key].dedicated)  # no client visible: unknown, not 0
            _write_fd(proc, 100, 5, _i915_fdinfo(1, self.PDEV, local_mib=100))
            _write_fd(proc, 200, 5, _i915_fdinfo(1, self.PDEV, local_mib=100))  # the same client, held by a second process
            _write_fd(proc, 300, 5, _i915_fdinfo(2, self.PDEV, local_mib=50))
            adapters = counters.sample()[1]
            self.assertEqual(adapters[card.key].dedicated, 150 * MiB)
            self.assertIsNone(adapters[card.key].shared)  # Intel has no device-wide system-memory counter
            _write_fd(proc, 400, 5, f"drm-driver:\ti915\ndrm-client-id:\t3\ndrm-pdev:\t{self.PDEV}\n"
                                    "drm-resident-local0:\tlots\n")
            self.assertIsNone(counters.sample()[1][card.key].dedicated)  # one client unreadable: the sum is unknown


class IntelPmuTests(unittest.TestCase):
    """Whole-GPU load from the i915 performance counters, against a fake PMU sysfs tree."""

    def setUp(self):
        drm._INTEL_PMU_STATE.clear()

    def _pmu(self, root, events=("rcs0-busy", "bcs0-busy", "vcs0-busy"), cpumask="0\n"):
        (root / "type").write_text("28\n")
        (root / "cpumask").write_text(cpumask)
        events_dir = root / "events"
        events_dir.mkdir(exist_ok=True)
        for i, event in enumerate(events):
            (events_dir / event).write_text(f"config=0x{i * 0x1000:x}\n")

    def test_spec_parses_type_cpu_and_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._pmu(root)
            pmu_type, cpu, configs = drm._intel_pmu_spec(str(root))
            self.assertEqual((pmu_type, cpu), (28, 0))
            self.assertEqual(configs, {"render": 0, "copy": 0x1000, "video": 0x2000})

    def test_spec_missing_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(drm._intel_pmu_spec(str(Path(tmp) / "no-such-dir")))
            root = Path(tmp)
            (root / "type").write_text("28\n")
            (root / "cpumask").write_text("not-a-cpu\n")
            self.assertIsNone(drm._intel_pmu_spec(str(root)))

    def test_rates_need_two_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._pmu(root)
            with patch.object(drm, "_intel_pmu_spec", return_value=(28, 0, {"render": 0, "copy": 0x1000})), \
                    patch.object(drm, "_perf_open", side_effect=[(10, None), (11, None)]) as opened, \
                    patch.object(drm, "_read_counter", side_effect=[0, 0, 500_000_000, 250_000_000]):
                self.assertEqual(drm._intel_pmu_rates("/x/card1/device", now=0), (None, None))
                self.assertEqual(opened.call_count, 2)
                self.assertEqual(drm._intel_pmu_rates("/x/card1/device", now=1_000_000_000), (50.0, None))
                # "copy" is not video; with no video engine the video figure stays unknown
                self.assertEqual(drm._INTEL_PMU_STATE["/x/card1/device"]["fail"], None)

    def test_video_is_the_video_engines_busy_share(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(drm, "_intel_pmu_spec", return_value=(28, 0, {"video": 0, "video-enhance": 1})), \
                    patch.object(drm, "_perf_open", side_effect=[(10, None), (11, None)]), \
                    patch.object(drm, "_read_counter", side_effect=[0, 0, 800_000_000, 200_000_000]):
                drm._intel_pmu_rates("/x/card1/device", now=0)
                self.assertEqual(drm._intel_pmu_rates("/x/card1/device", now=1_000_000_000), (80.0, 80.0))

    @unittest.skipIf(sys.platform == "win32", "powercap sysfs names contain a colon")
    def test_stepped_back_counter_re_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(drm, "_intel_pmu_spec", return_value=(28, 0, {"render": 0})), \
                    patch.object(drm, "_perf_open", return_value=(10, None)), \
                    patch.object(drm, "_read_counter", side_effect=[1_000_000_000, 900_000_000, 1_500_000_000]) as reads:
                drm._intel_pmu_rates("/x/card1/device", now=0)
                self.assertEqual(drm._intel_pmu_rates("/x/card1/device", now=1_000_000_000), (None, None))
                self.assertEqual(drm._intel_pmu_rates("/x/card1/device", now=2_000_000_000), (60.0, None))
                self.assertEqual(reads.call_count, 3)

    def test_denied_open_reports_denied(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(drm, "_intel_pmu_spec", return_value=(28, 0, {"render": 0})), \
                    patch.object(drm, "_perf_open", return_value=(None, "denied")):
                self.assertEqual(drm._intel_pmu_rates("/x/card1/device"), (None, None))
                self.assertEqual(drm._INTEL_PMU_STATE["/x/card1/device"]["fail"], "denied")

    def test_missing_spec_reports_missing(self):
        with patch.object(drm, "_intel_pmu_spec", return_value=None):
            self.assertEqual(drm._intel_pmu_rates("/x/card1/device"), (None, None))
            self.assertEqual(drm._INTEL_PMU_STATE["/x/card1/device"]["fail"], "missing")

    def test_unsupported_arch_reports_missing_not_denied(self):
        with patch.object(drm.platform, "machine", return_value="riscv"), \
                patch.object(drm.ctypes, "get_errno", return_value=13):
            self.assertEqual(drm._perf_open(28, 0, 0), (None, "missing"))

    def test_busy_cannot_exceed_wall_time(self):
        with patch.object(drm, "_intel_pmu_spec", return_value=(28, 0, {"render": 0})), \
                patch.object(drm, "_perf_open", return_value=(10, None)), \
                patch.object(drm, "_read_counter", side_effect=[0, 3_000_000_000]):
            drm._intel_pmu_rates("/x/card1/device", now=0)
            util, _video = drm._intel_pmu_rates("/x/card1/device", now=1_000_000_000)
            self.assertEqual(util, 100.0)

    def test_two_intel_gpus_share_no_pmu(self):
        # One i915 PMU for the whole driver: with two Intel cards neither gets device load,
        # instead of both showing the same numbers.
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp) / "proc"
            proc.mkdir()
            a = drm.DrmCard(key=1, pdev="0000:00:02.0", sysfs=str(Path(tmp) / "a"), driver="i915",
                            vendor_id=0x8086, device_id=0xA782, name="Intel UHD Graphics",
                            vram_total=0, integrated=True, intel=True)
            b = drm.DrmCard(key=2, pdev="0000:03:00.0", sysfs=str(Path(tmp) / "b"), driver="i915",
                            vendor_id=0x8086, device_id=0x56A0, name="Intel Arc A770",
                            vram_total=16 * GiB, integrated=False, intel=True)
            counters = drm.DrmCounters(found=[a, b], proc=str(proc), clock=Clock())
            counters.start()
            self.assertEqual(drm._INTEL_PMU_STATE[str(Path(tmp) / "a")]["fail"], "shared")
            self.assertEqual(drm._INTEL_PMU_STATE[str(Path(tmp) / "b")]["fail"], "shared")


class IntelRaplTests(unittest.TestCase):
    """GPU power from RAPL's uncore zone, against a fake powercap tree."""

    def setUp(self):
        drm._INTEL_RAPL_PREV.clear()
        drm._INTEL_RAPL_FAIL = None

    def _zone(self, root, energy="1000000\n", max_range="100000000\n", name="uncore"):
        package = root / "intel-rapl:0"
        package.mkdir(parents=True, exist_ok=True)
        sibling = package / "intel-rapl:0:0"
        sibling.mkdir(exist_ok=True)
        (sibling / "name").write_text("core\n")  # a named non-uncore zone the search must skip
        zone = package / "intel-rapl:0:1"
        zone.mkdir(exist_ok=True)
        (zone / "name").write_text(name + "\n")
        energy_path = zone / "energy_uj"
        energy_path.write_text(energy)
        if max_range is not None:
            (zone / "max_energy_range_uj").write_text(max_range)
        return str(energy_path)

    @unittest.skipIf(sys.platform == "win32", "powercap sysfs names contain a colon")
    def test_zone_discovery_prefers_uncore(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            energy = self._zone(root)
            self.assertEqual(drm._intel_rapl_zone(str(root))[0], energy)

    @unittest.skipIf(sys.platform == "win32", "powercap sysfs names contain a colon")
    def test_zone_missing_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(drm._intel_rapl_zone(str(Path(tmp) / "no-such-dir")))
            root = Path(tmp)
            package = root / "intel-rapl:0"
            package.mkdir()
            self.assertIsNone(drm._intel_rapl_zone(str(root)))  # no uncore zone: unknown, not 0 W

    @unittest.skipIf(sys.platform == "win32", "powercap sysfs names contain a colon")
    def test_power_needs_two_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            energy = self._zone(Path(tmp))
            with patch.object(drm, "_intel_rapl_zone", return_value=(energy, 100_000_000)):
                self.assertIsNone(drm._intel_rapl_w(now=0.0))
                Path(energy).write_text("6000000\n")  # 5 J in 1 s
                self.assertAlmostEqual(drm._intel_rapl_w(now=1.0), 5.0)

    def test_stepped_back_counter_re_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            energy = self._zone(Path(tmp))
            with patch.object(drm, "_intel_rapl_zone", return_value=(energy, 100_000_000)):
                self.assertIsNone(drm._intel_rapl_w(now=0.0))
                Path(energy).write_text("1000\n")  # reset, not a negative power draw
                self.assertIsNone(drm._intel_rapl_w(now=1.0))
                Path(energy).write_text("6001000\n")
                self.assertAlmostEqual(drm._intel_rapl_w(now=2.0), 6.0)

    def test_missing_zone_reports_missing(self):
        with patch.object(drm, "_intel_rapl_zone", return_value=None):
            self.assertIsNone(drm._intel_rapl_w(now=0.0))
            self.assertEqual(drm._INTEL_RAPL_FAIL, "missing")

    def test_unreadable_zone_reports_denied(self):
        if os.geteuid() == 0:
            self.skipTest("root bypasses file permissions")
        with tempfile.TemporaryDirectory() as tmp:
            energy = self._zone(Path(tmp))
            os.chmod(energy, 0)
            with patch.object(drm, "_intel_rapl_zone", return_value=(energy, 10_000_000)):
                self.assertIsNone(drm._intel_rapl_w(now=0.0))
                self.assertEqual(drm._INTEL_RAPL_FAIL, "denied")

    @unittest.skipIf(sys.platform == "win32", "powercap sysfs names contain a colon")
    def test_unparseable_zone_reports_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            energy = self._zone(Path(tmp), energy="lots\n")
            with patch.object(drm, "_intel_rapl_zone", return_value=(energy, 10_000_000)):
                self.assertIsNone(drm._intel_rapl_w(now=0.0))
                self.assertEqual(drm._INTEL_RAPL_FAIL, "missing")


class IntelCounterStatusTests(unittest.TestCase):
    """The N/A explanation names what is missing and why; silence when all present."""

    def _card(self, pdev="0000:00:02.0", integrated=True):
        return drm.DrmCard(key=32, pdev=pdev, sysfs="/x/card1/device", driver="i915",
                           vendor_id=0x8086, device_id=0xA7AD, name="Intel Intel Graphics",
                           vram_total=0, integrated=integrated, intel=True)

    def setUp(self):
        drm._INTEL_PMU_STATE.clear()
        drm._INTEL_RAPL_PREV.clear()
        drm._INTEL_RAPL_FAIL = None

    def test_not_intel_is_silent(self):
        card = drm.DrmCard(key=0, pdev="0000:03:00.0", sysfs="/x/amd/device", driver="amdgpu",
                           vendor_id=0x1002, device_id=0x7480, name="AMD GPU", vram_total=8 * GiB,
                           integrated=False)
        self.assertEqual(drm.intel_counter_status(card, None), "")

    def test_denied_names_privileges(self):
        drm._INTEL_PMU_STATE["/x/card1/device"] = {"fail": "denied"}
        drm._INTEL_RAPL_FAIL = "denied"
        with patch.object(drm, "_read", return_value="2"):
            status = drm.intel_counter_status(self._card(), None)
        self.assertIn("perf_event_paranoid=2", status)
        self.assertIn("root", status)
        self.assertIn("unknown, not zero", status)

    def test_missing_names_the_gap(self):
        drm._INTEL_PMU_STATE["/x/card1/device"] = {"fail": "missing"}
        drm._INTEL_RAPL_FAIL = "missing"
        status = drm.intel_counter_status(self._card(), None)
        self.assertIn("performance counters", status)
        self.assertIn("uncore", status)

    def test_hwmon_power_needs_no_rapal_reason(self):
        # A discrete card's hwmon sensor already answered: RAPL was never needed, so its
        # failure must not be reported. (start() primes RAPL only for integrated cards.)
        drm._INTEL_RAPL_FAIL = "denied"
        nv = SimpleNamespace(power_kind="Board")
        self.assertEqual(drm.intel_counter_status(self._card(), nv), "")

    def test_all_present_is_silent(self):
        self.assertEqual(drm.intel_counter_status(self._card(), None), "")

    def test_discrete_card_omits_rapal(self):
        drm._INTEL_PMU_STATE["/x/card1/device"] = {"fail": "denied"}
        drm._INTEL_RAPL_FAIL = "denied"
        with patch.object(drm, "_read", return_value="2"):
            status = drm.intel_counter_status(self._card(integrated=False), None)
        self.assertIn("load", status)
        self.assertNotIn("RAPL", status)

    def test_shared_pmu_is_refused(self):
        drm._INTEL_PMU_STATE["/x/card1/device"] = {"fail": "shared"}
        status = drm.intel_counter_status(self._card(), None)
        self.assertIn("multiple Intel GPUs", status)


class IntelNoteTests(unittest.TestCase):
    """Intel's device VRAM used is a sum over the clients gputaskman can inspect and its total is a PCI window size,
    and the output says so."""

    def _snapshot(self, source, vram_used=4 * GiB, vram_total=16 * GiB):
        from gputaskman.collector import Snapshot
        from gputaskman.telemetry import DeviceTelemetry

        return Snapshot(taken=1, adapter=None, adapters=[], nv=DeviceTelemetry(index=0, source=source), driver="",
                        cuda="", vram_total=vram_total, vram_used=vram_used, shared_used=None, backend="drm",
                        procs=[])

    def _panel(self, snapshot):
        from gputaskman.app import GpuTaskManager

        return "\n".join(item.plain for item in GpuTaskManager().render_device(snapshot).renderables)

    def test_text_carries_the_note_for_intel_only(self):
        from gputaskman.report import INTEL_MEMORY_NOTE, to_text

        self.assertIn(INTEL_MEMORY_NOTE, to_text(self._snapshot("Intel sysfs")))
        self.assertNotIn(INTEL_MEMORY_NOTE, to_text(self._snapshot("AMD sysfs")))  # amdgpu has a device counter
        # Nothing derived from the client sum or the window is shown (N/A used, unknown total): no caveat to make.
        self.assertNotIn(INTEL_MEMORY_NOTE, to_text(self._snapshot("Intel sysfs", vram_used=None, vram_total=0)))

    def test_device_panel_carries_the_note_for_intel_only(self):
        from gputaskman.report import INTEL_MEMORY_NOTE

        self.assertIn(INTEL_MEMORY_NOTE, self._panel(self._snapshot("Intel sysfs")))
        self.assertNotIn(INTEL_MEMORY_NOTE, self._panel(self._snapshot("AMD sysfs")))
        self.assertNotIn(INTEL_MEMORY_NOTE, self._panel(self._snapshot("Intel sysfs", vram_used=None, vram_total=0)))

    def test_unknown_total_reads_n_a_and_free_is_left_out(self):
        from gputaskman.report import to_text

        snapshot = self._snapshot("Intel sysfs", vram_used=3 * GiB, vram_total=0)
        self.assertIn("VRAM 3.00 GB / N/A", to_text(snapshot))
        panel = self._panel(snapshot)
        self.assertIn("3.00 GB / N/A", panel)
        self.assertNotIn("/ 0 MB", panel)
        self.assertNotRegex(panel, r"free \d")  # a free figure needs a total (the note's own "free memory" wording is not one)

    def test_window_total_is_not_turned_into_a_free_figure(self):
        panel = self._panel(self._snapshot("Intel sysfs"))  # 4 GB used of a 16 GB window
        self.assertIn("4.00 GB / 16.00 GB", panel)
        self.assertNotRegex(panel, r"free \d")
        self.assertIn("free 12.00 GB", self._panel(self._snapshot("AMD sysfs")))  # AMD's total is a device counter

    def test_json_says_unknown_and_carries_the_note(self):
        from gputaskman.report import INTEL_MEMORY_NOTE, to_dict

        unknown = to_dict(self._snapshot("Intel sysfs", vram_used=3 * GiB, vram_total=0))
        self.assertIsNone(unknown["vram_total"])  # unknown, not 0 bytes
        self.assertEqual(unknown["vram_used"], 3 * GiB)
        self.assertEqual(unknown["memory_note"], INTEL_MEMORY_NOTE)
        window = to_dict(self._snapshot("Intel sysfs"))
        self.assertEqual((window["vram_total"], window["memory_note"]), (16 * GiB, INTEL_MEMORY_NOTE))
        amd = to_dict(self._snapshot("AMD sysfs"))
        self.assertEqual((amd["vram_total"], amd["memory_note"]), (16 * GiB, None))


if __name__ == "__main__":
    unittest.main()
