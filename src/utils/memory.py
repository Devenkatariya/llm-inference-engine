"""GPU and system memory profiling utilities."""

import torch
import psutil
from typing import Dict, Optional
from .logger import get_logger

logger = get_logger(__name__)


def get_gpu_memory_usage() -> Dict[str, float]:
    """Returns allocated, reserved, and free VRAM in GB."""
    if not torch.cuda.is_available():
        return {"allocated_gb": 0, "reserved_gb": 0, "free_gb": 0, "total_gb": 0}

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    free = total - reserved

    return {
        "allocated_gb": round(allocated, 3),
        "reserved_gb": round(reserved, 3),
        "free_gb": round(free, 3),
        "total_gb": round(total, 3),
    }


def get_system_memory_usage() -> Dict[str, float]:
    """Returns system RAM stats in GB."""
    vm = psutil.virtual_memory()
    return {
        "total_gb": round(vm.total / (1024 ** 3), 2),
        "used_gb": round(vm.used / (1024 ** 3), 2),
        "available_gb": round(vm.available / (1024 ** 3), 2),
        "percent": vm.percent,
    }


def clear_gpu_cache():
    """Aggressively free GPU memory between benchmark runs."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.debug("GPU cache cleared.")
