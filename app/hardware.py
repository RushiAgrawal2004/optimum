from dataclasses import dataclass
import pynvml, psutil
from config import VRAM_SAFETY_MB

@dataclass
class Hardware:
    name: str
    total_mb: float
    free_mb: float
    ram_total_mb: float
    ram_free_mb: float

def probe() -> Hardware:
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
    name = pynvml.nvmlDeviceGetName(h)
    if isinstance(name, bytes):
        name = name.decode()
    ram = psutil.virtual_memory()
    return Hardware(
        name=name,
        total_mb=mem.total / 1024**2,
        free_mb=mem.free / 1024**2,
        ram_total_mb=ram.total / 1024**2,
        ram_free_mb=ram.available / 1024**2,
    )

def free_vram_mb() -> float:
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    return pynvml.nvmlDeviceGetMemoryInfo(h).free / 1024**2

def usable_vram_mb(hw: Hardware) -> float:
    return hw.free_mb - VRAM_SAFETY_MB

if __name__ == "__main__":
    hw = probe()
    print(hw)
    print(f"usable: {usable_vram_mb(hw):.0f} MB")
