"""
Benchmark orchestration — runs warmup + eval rounds,
collects LatencyRecords, and feeds into report generation.
"""

import gc
import torch
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.engine.inference import generate
from src.utils import get_logger, clear_gpu_cache
from .metrics import LatencyRecord, summarize_benchmark

logger = get_logger(__name__)

SAMPLE_PROMPTS = [
    "Explain the concept of attention mechanism in transformers in detail.",
    "Write a Python function to implement binary search with docstring.",
    "What are the key differences between ROCm and CUDA for GPU computing?",
    "Describe the mathematical foundations of backpropagation in neural networks.",
    "How does quantization affect model accuracy and inference speed?",
]


def run_benchmark(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    precision: str,
    batch_size: int = 1,
    max_new_tokens: int = 256,
    warmup_runs: int = 3,
    eval_runs: int = 20,
    prompts: Optional[List[str]] = None,
) -> dict:
    """
    Run a full benchmark: warmup → eval → summarize.

    Args:
        precision:     Label for this run (fp16, int8, etc.)
        batch_size:    For reporting (actual batching handled externally)
        warmup_runs:   Discarded runs to warm up GPU caches
        eval_runs:     Measured runs
        prompts:       Custom prompts (cycles through SAMPLE_PROMPTS if None)
    """
    prompts = prompts or SAMPLE_PROMPTS
    records: List[LatencyRecord] = []

    logger.info(f"Starting benchmark | precision={precision} | batch={batch_size} "
                f"| warmup={warmup_runs} | eval={eval_runs}")

    # Warmup
    for i in range(warmup_runs):
        prompt = prompts[i % len(prompts)]
        generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        logger.debug(f"Warmup {i+1}/{warmup_runs} done.")

    clear_gpu_cache()

    # Evaluation
    for i in range(eval_runs):
        prompt = prompts[i % len(prompts)]
        result = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        records.append(LatencyRecord(
            latency_ms=result["latency_ms"],
            tokens_generated=result["tokens_generated"],
            prompt_tokens=result["prompt_tokens"],
            batch_size=batch_size,
            precision=precision,
        ))
        if (i + 1) % 5 == 0:
            logger.info(f"  Eval {i+1}/{eval_runs} | "
                        f"latency={result['latency_ms']:.0f}ms | "
                        f"tps={result['tokens_per_second']:.1f}")

    summary = summarize_benchmark(records)
    logger.info(f"Benchmark complete | p50={summary['latency']['p50_ms']}ms | "
                f"avg_tps={summary['throughput']['avg_tokens_per_second']}")
    return summary
