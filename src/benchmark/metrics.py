"""
Benchmark metrics collection and percentile computation.
"""

import time
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class LatencyRecord:
    latency_ms: float
    tokens_generated: int
    prompt_tokens: int
    batch_size: int
    precision: str
    timestamp: float = field(default_factory=time.time)


def compute_percentiles(latencies: List[float]) -> Dict[str, float]:
    arr = np.array(latencies)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p75_ms": round(float(np.percentile(arr, 75)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "mean_ms": round(float(np.mean(arr)), 2),
        "std_ms": round(float(np.std(arr)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
    }


def compute_throughput(records: List[LatencyRecord]) -> Dict[str, float]:
    total_tokens = sum(r.tokens_generated for r in records)
    total_time_s = sum(r.latency_ms for r in records) / 1000
    return {
        "total_tokens_generated": total_tokens,
        "total_time_s": round(total_time_s, 3),
        "avg_tokens_per_second": round(total_tokens / total_time_s, 2) if total_time_s > 0 else 0,
        "avg_tokens_per_request": round(total_tokens / len(records), 1) if records else 0,
    }


def summarize_benchmark(records: List[LatencyRecord]) -> Dict[str, Any]:
    latencies = [r.latency_ms for r in records]
    return {
        "n_runs": len(records),
        "precision": records[0].precision if records else "unknown",
        "batch_size": records[0].batch_size if records else 1,
        "latency": compute_percentiles(latencies),
        "throughput": compute_throughput(records),
    }
