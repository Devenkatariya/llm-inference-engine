#!/bin/bash
set -e
echo "Running benchmark suite..."
python -c "
from src.engine import load_model_and_tokenizer
from src.benchmark import run_benchmark, plot_throughput_comparison, plot_latency_percentiles, save_results_json

model, tokenizer = load_model_and_tokenizer(precision='fp16')
results = []
for precision in ['fp16', 'int8', 'int4']:
    print(f'Benchmarking {precision}...')
    r = run_benchmark(model, tokenizer, precision=precision, warmup_runs=3, eval_runs=20)
    r['precision'] = precision
    results.append(r)

save_results_json({'results': results})
plot_throughput_comparison(results, save_path='results/throughput.png')
plot_latency_percentiles(results, save_path='results/latency.png')
print('Done! Check results/ folder.')
"
