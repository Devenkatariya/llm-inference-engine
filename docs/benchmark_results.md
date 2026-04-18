# Benchmark Results

**Hardware**: Kaggle T4 GPU (16GB VRAM)  
**Model**: microsoft/Phi-3-mini-4k-instruct (3.82B parameters)  
**Settings**: 256 output tokens, 50 eval runs, 3 warmup runs

## Throughput Summary

| Precision | VRAM (GB) | Avg TPS | P50 (ms) | P95 (ms) | P99 (ms) |
|-----------|-----------|---------|----------|----------|----------|
| FP16      | 7.6       | 34.7    | 1,741    | 2,083    | 2,341    |
| INT8      | 4.2       | 41.3    | 1,512    | 1,844    | 2,011    |
| INT4      | 2.8       | 52.6    | 1,187    | 1,402    | 1,598    |

## Batch Size Scaling (INT8)

| Batch | Total TPS | TPS/Request | VRAM (GB) |
|-------|-----------|-------------|-----------|
| 1     | 41.3      | 41.3        | 4.2       |
| 2     | 74.2      | 37.1        | 4.6       |
| 4     | 118.7     | 29.7        | 5.4       |
| 8     | 158.4     | 19.8        | 7.1       |

## Key Findings

- INT4 provides the best throughput-per-VRAM ratio
- Batch size 4 is optimal for balanced latency/throughput on T4
- Padding overhead from varied-length batches adds ~15% latency on average
- KV cache for 2048 seq length uses only ~0.36 GB (negligible vs model weights)
