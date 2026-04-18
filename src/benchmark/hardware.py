"""
Hardware profiling — VRAM, GPU utilization, temperature.
Works with both NVIDIA (pynvml) and AMD (ROCm SMI via subprocess).
"""

import subprocess
import torch
from typing import Dict, Optional
from src.utils import get_logger, get_gpu_memory_usage, DEVICE_INFO

logger = get_logger(__name__)


def get_gpu_stats() -> Dict:
    """Unified GPU stats for NVIDIA and AMD."""
    mem = get_gpu_memory_usage()
    stats = {
        "backend": DEVICE_INFO.backend,
        "device_name": DEVICE_INFO.name,
        "total_vram_gb": DEVICE_INFO.total_memory_gb,
        **mem,
    }

    if DEVICE_INFO.backend == "nvidia":
        stats.update(_get_nvidia_stats())
    elif DEVICE_INFO.backend == "amd":
        stats.update(_get_rocm_stats())

    return stats


def _get_nvidia_stats() -> Dict:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        return {"gpu_utilization_pct": util.gpu, "temperature_c": temp}
    except Exception:
        return {"gpu_utilization_pct": None, "temperature_c": None}


def _get_rocm_stats() -> Dict:
    """Query AMD GPU stats via rocm-smi CLI."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showuse", "--showtemp", "--csv"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            # Parse CSV output (format varies by ROCm version)
            return {"rocm_smi_raw": lines[1] if len(lines) > 1 else ""}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return {"gpu_utilization_pct": None, "temperature_c": None}
