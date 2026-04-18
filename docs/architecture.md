# Architecture Overview

## System Design

The inference engine is structured in three layers:

**1. Engine Layer** (`src/engine/`)
Handles all model concerns — loading, quantization, generation, KV cache, and batching. Stateless functions that accept model + tokenizer as arguments, making them easy to test and swap.

**2. API Layer** (`src/api/`)
FastAPI application with async request handling. The model is loaded once at startup via the lifespan context manager and shared across requests. Long-running benchmark jobs are dispatched as background tasks.

**3. Benchmark Layer** (`src/benchmark/`)
Orchestrates warmup + evaluation rounds, collects `LatencyRecord` objects, computes percentile statistics, and generates matplotlib visualizations. Integrates with MLflow for experiment tracking.

## Hardware Abstraction

AMD ROCm and NVIDIA CUDA both expose the same `torch.cuda.*` API — the engine detects the backend by inspecting the GPU device name string. This means all PyTorch code runs unchanged on both backends; only Docker images and driver-level setup differ.

## Quantization Pipeline

```
FP32 weights
    │
    ▼ BitsAndBytesConfig(load_in_8bit=True)
INT8 weights  ──── ~50% VRAM reduction, ~15% throughput gain
    │
    ▼ BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
NF4 weights   ──── ~75% VRAM reduction, ~50% throughput gain
```

## KV Cache

Phi-3 uses HuggingFace's `DynamicCache` by default — the cache grows as sequence length increases. For production serving with fixed-length contracts, switching to `StaticCache` eliminates allocation overhead at the cost of flexibility.
