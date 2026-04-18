"""Pydantic request/response schemas for the FastAPI server."""

from pydantic import BaseModel, Field
from typing import Optional


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input prompt text")
    max_new_tokens: int = Field(256, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=0)
    do_sample: bool = True
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    stream: bool = False

    model_config = {"json_schema_extra": {
        "example": {
            "prompt": "Explain how KV cache works in transformer inference.",
            "max_new_tokens": 256,
            "temperature": 0.7,
        }
    }}


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    tokens_generated: int
    latency_ms: float
    tokens_per_second: float


class BenchmarkRequest(BaseModel):
    precision: str = Field("fp16", description="fp16 | int8 | int4")
    warmup_runs: int = 3
    eval_runs: int = 20
    max_new_tokens: int = 256


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    backend: str
    vram_free_gb: Optional[float]
    vram_total_gb: Optional[float]
