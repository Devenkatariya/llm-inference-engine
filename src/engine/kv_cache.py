"""
KV Cache management.
Phi-3 uses HuggingFace's built-in dynamic cache — this module
provides monitoring utilities and cache configuration helpers.
"""

import torch
from typing import Optional, Dict
from src.utils import get_logger

logger = get_logger(__name__)


def get_cache_config(max_seq_length: int = 2048, cache_type: str = "dynamic") -> Dict:
    """
    Return generation kwargs to configure KV caching strategy.

    cache_type:
        "dynamic"  — HF DynamicCache (grows as needed, default)
        "static"   — pre-allocated fixed size (lower allocation overhead)
    """
    config = {"use_cache": True}

    if cache_type == "static":
        # Static cache requires fixed max length — trades flexibility for speed
        logger.info(f"Using static KV cache with max_seq_length={max_seq_length}")
        config["max_length"] = max_seq_length
    else:
        logger.info("Using dynamic KV cache (HF default).")

    return config


def estimate_kv_cache_size_gb(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    max_seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,  # fp16
) -> float:
    """
    Estimate KV cache VRAM footprint in GB.

    Formula: 2 (K+V) * layers * heads * head_dim * seq_len * batch * dtype_bytes
    """
    bytes_total = 2 * num_layers * num_heads * head_dim * max_seq_len * batch_size * dtype_bytes
    return round(bytes_total / (1024 ** 3), 4)


# Phi-3 Mini 3.8B architecture constants
PHI3_MINI_KV_CONFIG = {
    "num_layers": 32,
    "num_heads": 32,
    "head_dim": 96,   # hidden_size(3072) / num_heads(32)
}
