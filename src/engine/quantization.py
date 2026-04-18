"""
Quantization helpers — wraps bitsandbytes and provides
utility functions for model size estimation and precision comparison.
"""

import torch
from typing import Dict
from transformers import AutoModelForCausalLM
from src.utils import get_logger

logger = get_logger(__name__)


def estimate_model_size(model: AutoModelForCausalLM) -> Dict[str, float]:
    """Estimate model size in memory for different precisions."""
    param_count = sum(p.numel() for p in model.parameters())
    return {
        "params_billions": round(param_count / 1e9, 3),
        "fp32_gb": round(param_count * 4 / (1024 ** 3), 2),
        "fp16_gb": round(param_count * 2 / (1024 ** 3), 2),
        "int8_gb": round(param_count * 1 / (1024 ** 3), 2),
        "int4_gb": round(param_count * 0.5 / (1024 ** 3), 2),
    }


def get_model_dtype_info(model: AutoModelForCausalLM) -> Dict[str, any]:
    """Inspect the actual dtypes used in a loaded model."""
    dtype_counts: Dict[str, int] = {}
    for p in model.parameters():
        dt = str(p.dtype)
        dtype_counts[dt] = dtype_counts.get(dt, 0) + p.numel()

    total = sum(dtype_counts.values())
    return {
        "dtype_distribution": {k: f"{v/total*100:.1f}%" for k, v in dtype_counts.items()},
        "primary_dtype": max(dtype_counts, key=dtype_counts.get),
        "total_params": total,
    }
