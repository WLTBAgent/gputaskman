"""Optional device-wide telemetry shared by vendor providers.

Percentages and clocks are provider-specific; unsupported metrics are None.
"""
from dataclasses import dataclass, field


@dataclass
class DeviceTelemetry:
    index: int
    name: str = "?"
    pci_bus: str = ""
    driver_model: str = ""
    mem_total: int | None = None
    mem_used: int | None = None
    mem_reserved: int | None = None
    util_gpu: float | None = None
    util_mem: float | None = None
    enc: float | None = None
    dec: float | None = None
    video: float | None = None  # video-engine busy % (unified decode+encode on Intel; enc/dec stay unset)
    temp: float | None = None
    temp_kind: str = ""  # where a fallback temperature comes from ("CPU package"), else its own sensor
    power_w: float | None = None
    power_kind: str = ""
    power_limit_w: float | None = None
    fan: int | None = None
    pstate: int | None = None
    sm_clock: int | None = None
    sm_clock_max: int | None = None
    gfx_clock: int | None = None
    mem_clock: int | None = None
    video_clock: int | None = None
    pcie_rx_kbs: int | None = None
    pcie_tx_kbs: int | None = None
    pcie_gen: int | None = None
    pcie_width: int | None = None
    throttle: list[str] | None = field(default_factory=list)  # None = unreadable (unknown), [] = readable and idle
    compute_cap: str = ""
    source: str = "NVML"
