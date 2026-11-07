"""Enumerate GPU adapters via DXGI so performance-counter LUIDs map to real GPU names."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class _DESC1(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", wintypes.UINT),
        ("DeviceId", wintypes.UINT),
        ("SubSysId", wintypes.UINT),
        ("Revision", wintypes.UINT),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _LUID),
        ("Flags", wintypes.UINT),
    ]


class _GUID(ctypes.Structure):
    _fields_ = [("d1", wintypes.DWORD), ("d2", wintypes.WORD), ("d3", wintypes.WORD), ("d4", ctypes.c_ubyte * 8)]


_IID_IDXGIFactory1 = _GUID(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
VENDOR_NVIDIA = 0x10DE
VENDOR_AMD = 0x1002
VENDOR_HUAWEI = 0x19E5  # Ascend NPUs (telemetry via npu-smi, not DRM)
VENDORS = {0x10DE: "NVIDIA", 0x1002: "AMD", 0x8086: "Intel", 0x19E5: "Ascend",
           0x1414: "Microsoft", 0x5143: "Qualcomm"}


@dataclass
class Adapter:
    luid: int
    name: str
    vendor_id: int
    dedicated_vram: int
    shared_system: int
    software: bool
    pci_address: tuple[int, int, int] | None = None
    device_instance_id: str = ""
    identity_status: str = ""

    @property
    def root_display(self) -> bool:
        return self.device_instance_id.upper().startswith("ROOT\\DISPLAY\\")

    @property
    def vendor(self) -> str:
        if self.root_display:
            return "ROOT display adapter"
        return VENDORS.get(self.vendor_id, hex(self.vendor_id))


def _vcall(obj: ctypes.c_void_p, index: int, restype, *argtypes):
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtbl[index])


class _OpenAdapter(ctypes.Structure):
    _fields_ = [("luid", _LUID), ("handle", wintypes.UINT)]


class _AdapterAddress(ctypes.Structure):
    _fields_ = [("bus", wintypes.UINT), ("device", wintypes.UINT), ("function", wintypes.UINT)]


class _QueryAdapter(ctypes.Structure):
    _fields_ = [("handle", wintypes.UINT), ("kind", wintypes.UINT),
                ("data", ctypes.c_void_p), ("size", wintypes.UINT)]


class _DeviceInfoHeader(ctypes.Structure):
    _fields_ = [("kind", wintypes.UINT), ("size", wintypes.UINT),
                ("luid", _LUID), ("id", wintypes.UINT)]


class _AdapterName(ctypes.Structure):
    _fields_ = [("header", _DeviceInfoHeader), ("path", ctypes.c_wchar * 128)]


class _DeviceInterface(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("guid", _GUID), ("flags", wintypes.DWORD),
                ("reserved", ctypes.c_size_t)]


class _DeviceInfo(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("guid", _GUID), ("instance", wintypes.DWORD),
                ("reserved", ctypes.c_size_t)]


def adapter_identity(luid: int) -> tuple[str, str]:
    """Resolve this exact LUID, including inactive displays, without enumerating other devices.

    Fixed buffers and one call per property bound the work; OS calls themselves have no timeout.
    A failed lookup leaves the DXGI description intact, with a reason supplied by the caller.
    """
    user = ctypes.WinDLL("user32")
    get_name = user.DisplayConfigGetDeviceInfo
    get_name.argtypes = [ctypes.POINTER(_DeviceInfoHeader)]
    get_name.restype = wintypes.LONG
    name = _AdapterName()
    name.header = _DeviceInfoHeader(
        4, ctypes.sizeof(name), _LUID(luid & 0xffffffff, ctypes.c_int32(luid >> 32).value), 0)
    status = get_name(ctypes.byref(name.header))
    if status or not name.path:
        raise OSError(f"DisplayConfig adapter path unavailable (error {status})")

    setup = ctypes.WinDLL("setupapi", use_last_error=True)
    setup.SetupDiCreateDeviceInfoList.argtypes = [ctypes.POINTER(_GUID), wintypes.HWND]
    setup.SetupDiCreateDeviceInfoList.restype = wintypes.HANDLE
    setup.SetupDiOpenDeviceInterfaceW.argtypes = [
        wintypes.HANDLE, wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(_DeviceInterface)]
    setup.SetupDiOpenDeviceInterfaceW.restype = wintypes.BOOL
    setup.SetupDiGetDeviceInterfaceDetailW.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(_DeviceInterface), ctypes.c_void_p, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(_DeviceInfo)]
    setup.SetupDiGetDeviceInterfaceDetailW.restype = wintypes.BOOL
    setup.SetupDiGetDeviceInstanceIdW.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(_DeviceInfo), wintypes.LPWSTR, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD)]
    setup.SetupDiGetDeviceInstanceIdW.restype = wintypes.BOOL
    setup.SetupDiGetDeviceRegistryPropertyW.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(_DeviceInfo), wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD)]
    setup.SetupDiGetDeviceRegistryPropertyW.restype = wintypes.BOOL
    setup.SetupDiDestroyDeviceInfoList.argtypes = [wintypes.HANDLE]
    setup.SetupDiDestroyDeviceInfoList.restype = wintypes.BOOL
    devices = setup.SetupDiCreateDeviceInfoList(None, None)
    if devices == ctypes.c_void_p(-1).value:
        raise OSError(f"PnP device list unavailable (error {ctypes.get_last_error()})")
    try:
        interface = _DeviceInterface()
        interface.size = ctypes.sizeof(interface)
        if not setup.SetupDiOpenDeviceInterfaceW(devices, name.path, 0, ctypes.byref(interface)):
            raise OSError(f"PnP adapter interface unavailable (error {ctypes.get_last_error()})")
        detail = ctypes.create_string_buffer(4096)
        # SP_DEVICE_INTERFACE_DETAIL_DATA_W has a packed WCHAR path but pointer-size alignment on x64.
        ctypes.cast(detail, ctypes.POINTER(wintypes.DWORD))[0] = 8 if ctypes.sizeof(ctypes.c_void_p) == 8 else 6
        info = _DeviceInfo()
        info.size = ctypes.sizeof(info)
        if not setup.SetupDiGetDeviceInterfaceDetailW(
                devices, ctypes.byref(interface), detail, ctypes.sizeof(detail), None, ctypes.byref(info)):
            raise OSError(f"PnP adapter details unavailable (error {ctypes.get_last_error()})")
        instance = ctypes.create_unicode_buffer(512)
        if not setup.SetupDiGetDeviceInstanceIdW(devices, ctypes.byref(info), instance, len(instance), None):
            raise OSError(f"PnP adapter instance unavailable (error {ctypes.get_last_error()})")
        if not instance.value:
            raise OSError("PnP adapter instance is empty")
        for prop in (12, 0):  # SPDRP_FRIENDLYNAME, then SPDRP_DEVICEDESC
            text = ctypes.create_unicode_buffer(1024)
            kind, size = wintypes.DWORD(), wintypes.DWORD()
            if (setup.SetupDiGetDeviceRegistryPropertyW(
                    devices, ctypes.byref(info), prop, ctypes.byref(kind), text,
                    ctypes.sizeof(text), ctypes.byref(size))
                    and kind.value == 1 and 2 <= size.value <= ctypes.sizeof(text)
                    and text.value.strip()):
                return instance.value, text.value.strip()
        # Even when the name is missing, a ROOT display must not inherit a physical GPU's name.
        return instance.value, ""
    finally:
        setup.SetupDiDestroyDeviceInfoList(devices)


def _identify_adapter(adapter: Adapter) -> None:
    try:
        instance, name = adapter_identity(adapter.luid)
    except (OSError, AttributeError) as error:
        adapter.identity_status = f"Windows PnP identity unavailable ({error}); name and vendor are DXGI-reported."
        return
    adapter.device_instance_id = instance
    if adapter.root_display:
        adapter.name = name or f"ROOT display adapter ({instance})"
    # Keep physical DXGI names and capacity-based ordering unchanged.
    if not name:
        adapter.identity_status = "Windows PnP adapter name unavailable; device instance is known."


def adapter_pci_address(luid: int) -> tuple[int, int, int] | None:
    """Obtain the WDDM adapter's bus/device/function; the PCI domain is not exposed."""
    try:
        gdi = ctypes.WinDLL("gdi32")
        opened = _OpenAdapter(_LUID(luid & 0xffffffff, ctypes.c_int32(luid >> 32).value), 0)
        if gdi.D3DKMTOpenAdapterFromLuid(ctypes.byref(opened)) != 0:
            return None
        try:
            address = _AdapterAddress()
            query = _QueryAdapter(opened.handle, 6, ctypes.addressof(address), ctypes.sizeof(address))
            if gdi.D3DKMTQueryAdapterInfo(ctypes.byref(query)) != 0:
                return None
            return address.bus, address.device, address.function
        finally:
            gdi.D3DKMTCloseAdapter(ctypes.byref(wintypes.UINT(opened.handle)))
    except (OSError, AttributeError):
        return None


def enumerate_adapters(*, strict: bool = False) -> list[Adapter]:
    try:
        dxgi = ctypes.WinDLL("dxgi")
    except OSError:
        if strict:
            raise
        return []
    factory = ctypes.c_void_p()
    if dxgi.CreateDXGIFactory1(ctypes.byref(_IID_IDXGIFactory1), ctypes.byref(factory)) != 0:
        if strict:
            raise OSError("DXGI adapter discovery failed")
        return []
    out: list[Adapter] = []
    try:
        enum_adapters1 = _vcall(factory, 12, ctypes.c_long, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))
        i = 0
        while True:
            adapter = ctypes.c_void_p()
            status = enum_adapters1(factory, i, ctypes.byref(adapter))
            if status != 0:
                if strict and status & 0xffffffff != 0x887A0002:  # DXGI_ERROR_NOT_FOUND
                    raise OSError("DXGI adapter enumeration failed")
                break
            try:
                desc = _DESC1()
                status = _vcall(adapter, 10, ctypes.c_long, ctypes.POINTER(_DESC1))(adapter, ctypes.byref(desc))
                if status == 0:
                    luid = ((desc.AdapterLuid.HighPart & 0xffffffff) << 32) | desc.AdapterLuid.LowPart
                    found = Adapter(
                        luid=luid,
                        name=desc.Description,
                        vendor_id=desc.VendorId,
                        dedicated_vram=desc.DedicatedVideoMemory,
                        shared_system=desc.SharedSystemMemory,
                        software=bool(desc.Flags & 2),
                    )
                    _identify_adapter(found)
                    if not found.root_display:
                        found.pci_address = adapter_pci_address(luid)
                    out.append(found)
                elif strict:
                    raise OSError("DXGI adapter description unavailable")
            finally:
                _vcall(adapter, 2, wintypes.ULONG)(adapter)  # Release
            i += 1
    finally:
        _vcall(factory, 2, wintypes.ULONG)(factory)
    return out
