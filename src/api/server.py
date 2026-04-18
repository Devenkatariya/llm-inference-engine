"""
FastAPI inference server.
Endpoints: POST /generate, POST /benchmark, GET /health, GET /metrics
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.engine import load_model_and_tokenizer, generate
from src.benchmark import run_benchmark, get_gpu_stats
from src.utils import get_logger, DEVICE_INFO, get_gpu_memory_usage
from .schemas import GenerateRequest, GenerateResponse, BenchmarkRequest, HealthResponse
from .middleware import log_requests, CORS_CONFIG

logger = get_logger(__name__)

# Global state
_model = None
_tokenizer = None
_benchmark_results = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global _model, _tokenizer
    logger.info("Loading Phi-3 Mini on startup...")
    _model, _tokenizer = load_model_and_tokenizer(
        precision="fp16"
    )
    logger.info("Model ready. Server is live.")
    yield
    logger.info("Shutting down. Releasing model.")
    del _model, _tokenizer


app = FastAPI(
    title="LLM Inference Engine",
    description="Production-grade Phi-3 Mini inference with hardware benchmarking",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, **CORS_CONFIG)
app.middleware("http")(log_requests)


@app.get("/health", response_model=HealthResponse)
async def health():
    mem = get_gpu_memory_usage()
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        device=DEVICE_INFO.device,
        backend=DEVICE_INFO.backend,
        vram_free_gb=mem.get("free_gb"),
        vram_total_gb=mem.get("total_gb"),
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(request: GenerateRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: generate(
            _model, _tokenizer, request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            do_sample=request.do_sample,
            repetition_penalty=request.repetition_penalty,
        )
    )
    return GenerateResponse(**result)


@app.post("/benchmark")
async def benchmark_endpoint(request: BenchmarkRequest, background_tasks: BackgroundTasks):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    job_id = f"bench_{request.precision}_{request.eval_runs}"

    def _run():
        result = run_benchmark(
            _model, _tokenizer,
            precision=request.precision,
            warmup_runs=request.warmup_runs,
            eval_runs=request.eval_runs,
            max_new_tokens=request.max_new_tokens,
        )
        _benchmark_results[job_id] = result
        logger.info(f"Benchmark {job_id} complete.")

    background_tasks.add_task(_run)
    return {"job_id": job_id, "status": "started"}


@app.get("/benchmark/{job_id}")
async def get_benchmark_result(job_id: str):
    if job_id not in _benchmark_results:
        return JSONResponse({"status": "pending"}, status_code=202)
    return _benchmark_results[job_id]


@app.get("/metrics")
async def metrics():
    """Basic Prometheus-style metrics (extend with prometheus_fastapi_instrumentator for full support)."""
    mem = get_gpu_memory_usage()
    gpu = get_gpu_stats()
    return {
        "gpu_vram_allocated_gb": mem.get("allocated_gb"),
        "gpu_vram_free_gb": mem.get("free_gb"),
        "gpu_backend": DEVICE_INFO.backend,
        "gpu_name": DEVICE_INFO.name,
    }
