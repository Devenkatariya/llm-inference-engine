"""
Batching strategies — static, dynamic, and simple continuous batching.
"""

import time
import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class InferenceRequest:
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    request_id: Optional[str] = None
    arrival_time: float = field(default_factory=time.time)


@dataclass
class BatchConfig:
    strategy: str = "dynamic"      # static | dynamic
    max_batch_size: int = 8
    max_wait_ms: float = 50.0       # Max time to wait before flushing partial batch
    pad_to_multiple: int = 8        # Pad sequence lengths to multiple of N (tensor core alignment)


def form_static_batch(requests: List[InferenceRequest], batch_size: int) -> List[List[InferenceRequest]]:
    """Split requests into fixed-size batches."""
    return [requests[i:i + batch_size] for i in range(0, len(requests), batch_size)]


def form_dynamic_batch(
    requests: List[InferenceRequest],
    max_batch_size: int,
    max_tokens_per_batch: int = 4096,
) -> List[List[InferenceRequest]]:
    """
    Group requests into batches respecting a token budget.
    Shorter prompts are grouped together to minimize padding waste.
    """
    # Sort by estimated length for efficient packing
    sorted_reqs = sorted(requests, key=lambda r: len(r.prompt))
    batches = []
    current_batch = []
    current_tokens = 0

    for req in sorted_reqs:
        estimated_tokens = len(req.prompt.split()) * 1.3  # rough estimate
        if (len(current_batch) >= max_batch_size or
                current_tokens + estimated_tokens > max_tokens_per_batch):
            if current_batch:
                batches.append(current_batch)
            current_batch = [req]
            current_tokens = estimated_tokens
        else:
            current_batch.append(req)
            current_tokens += estimated_tokens

    if current_batch:
        batches.append(current_batch)

    logger.debug(f"Dynamic batching: {len(requests)} requests → {len(batches)} batches")
    return batches
