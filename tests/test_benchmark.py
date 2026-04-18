"""Tests for benchmark metrics computation."""

import pytest
from src.benchmark.metrics import compute_percentiles, compute_throughput, LatencyRecord


def test_compute_percentiles():
    latencies = [100.0, 200.0, 300.0, 400.0, 500.0, 1000.0]
    p = compute_percentiles(latencies)
    assert p["p50_ms"] > 0
    assert p["p99_ms"] >= p["p95_ms"] >= p["p50_ms"]
    assert p["min_ms"] <= p["mean_ms"] <= p["max_ms"]


def test_summarize_benchmark():
    from src.benchmark.metrics import summarize_benchmark
    records = [
        LatencyRecord(latency_ms=900.0, tokens_generated=256, prompt_tokens=50, batch_size=1, precision="fp16")
        for _ in range(10)
    ]
    summary = summarize_benchmark(records)
    assert summary["n_runs"] == 10
    assert summary["precision"] == "fp16"
    assert "latency" in summary
    assert "throughput" in summary
    assert summary["throughput"]["avg_tokens_per_second"] > 0


def test_device_detection_cpu():
    """On CI without GPU, should gracefully fall back to CPU."""
    import torch
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(torch.cuda, "is_available", lambda: False)
        from importlib import reload
        import src.utils.device as dev_mod
        reload(dev_mod)
        info = dev_mod.detect_device()
        assert info.device == "cpu"
        assert info.backend == "cpu"
