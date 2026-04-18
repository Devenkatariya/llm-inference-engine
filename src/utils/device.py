"""
Device detection utility.
Handles CUDA (NVIDIA), ROCm (AMD), and CPU backends transparently.
"""

import torch
from dataclasses import dataclass
from typing import Optional
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class DeviceInfo:
    device: str
    device_type: str          # "cuda" | "rocm" | "cpu"
    backend: str              # "nvidia" | "amd" | "cpu"
    name: Optional[str]
    total_memory_gb: Optional[float]
    compute_capability: Optional[str]
    driver_version: Optional[str]


def detect_device() -> DeviceInfo:
    """
    Auto-detect the best available hardware backend.
    AMD ROCm exposes itself via the same torch.cuda API — detection
    is done via device name inspection.
    """
    if not torch.cuda.is_available():
        logger.warning("No GPU detected — falling back to CPU.")
        return DeviceInfo(
            device="cpu", device_type="cpu", backend="cpu",
            name="CPU", total_memory_gb=None,
            compute_capability=None, driver_version=None
        )

    device_name = torch.cuda.get_device_name(0).lower()
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    # AMD GPUs expose via ROCm under the CUDA API
    is_amd = any(k in device_name for k in ["amd", "radeon", "mi2", "mi3", "rx 6", "rx 7"])
    backend = "amd" if is_amd else "nvidia"
    device_type = "rocm" if is_amd else "cuda"

    compute_cap = None
    if not is_amd:
        props = torch.cuda.get_device_properties(0)
        compute_cap = f"{props.major}.{props.minor}"

    logger.info(f"Detected backend: {backend.upper()} | Device: {device_name} | VRAM: {total_mem:.1f} GB")

    return DeviceInfo(
        device="cuda",
        device_type=device_type,
        backend=backend,
        name=torch.cuda.get_device_name(0),
        total_memory_gb=round(total_mem, 2),
        compute_capability=compute_cap,
        driver_version=None,
    )


def get_optimal_dtype(device_info: DeviceInfo, requested: str = "fp16") -> torch.dtype:
    """
    Return the best torch dtype given hardware and user request.
    BF16 requires Ampere+ on NVIDIA; AMD MI series supports BF16 natively.
    """
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    if requested in ("int8", "int4"):
        return torch.float16

    if requested == "bf16":
        if device_info.backend == "nvidia" and device_info.compute_capability:
            major = int(device_info.compute_capability.split(".")[0])
            if major < 8:
                logger.warning("BF16 not supported on this NVIDIA GPU (needs Ampere+). Falling back to FP16.")
                return torch.float16
        return torch.bfloat16

    return dtype_map.get(requested, torch.float16)


DEVICE_INFO = detect_device()
DEVICE = DEVICE_INFO.device
