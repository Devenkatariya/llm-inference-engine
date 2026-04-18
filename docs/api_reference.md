# API Reference

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

## POST /generate

Generate text from a prompt.

**Request body:**
```json
{
  "prompt": "string (required)",
  "max_new_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "do_sample": true,
  "repetition_penalty": 1.1,
  "stream": false
}
```

**Response:**
```json
{
  "text": "generated output",
  "prompt_tokens": 42,
  "tokens_generated": 187,
  "latency_ms": 923.4,
  "tokens_per_second": 202.6
}
```

## POST /benchmark

Start an async benchmark job.

**Request body:**
```json
{
  "precision": "fp16",
  "warmup_runs": 3,
  "eval_runs": 20,
  "max_new_tokens": 256
}
```

**Response:** `{ "job_id": "bench_fp16_20", "status": "started" }`

## GET /benchmark/{job_id}

Poll benchmark job result. Returns `{"status": "pending"}` (202) until done.

## GET /health

Returns server status, device info, VRAM stats.

## GET /metrics

Returns Prometheus-compatible GPU metrics.
