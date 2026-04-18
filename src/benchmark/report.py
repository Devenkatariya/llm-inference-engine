"""
Benchmark report generation — JSON export, markdown tables, and matplotlib plots.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from src.utils import get_logger

logger = get_logger(__name__)


def save_results_json(results: Dict[str, Any], output_dir: str = "results") -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"benchmark_{ts}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved: {path}")
    return path


def plot_throughput_comparison(results: List[Dict], save_path: Optional[str] = None):
    """Bar chart comparing tokens/sec across precisions."""
    precisions = [r["precision"] for r in results]
    tps = [r["throughput"]["avg_tokens_per_second"] for r in results]

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#937860"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(precisions, tps, color=colors[:len(precisions)], width=0.5, edgecolor="white")

    for bar, val in zip(bars, tps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xlabel("Precision", fontsize=12)
    ax.set_ylabel("Tokens / second", fontsize=12)
    ax.set_title("Inference Throughput by Quantization Precision\n(Phi-3 Mini 3.8B — Kaggle T4)", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved: {save_path}")
    plt.show()


def plot_latency_percentiles(results: List[Dict], save_path: Optional[str] = None):
    """Grouped bar chart for P50/P95/P99 latency."""
    precisions = [r["precision"] for r in results]
    p50 = [r["latency"]["p50_ms"] for r in results]
    p95 = [r["latency"]["p95_ms"] for r in results]
    p99 = [r["latency"]["p99_ms"] for r in results]

    x = range(len(precisions))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width for i in x], p50, width, label="P50", color="#4C72B0")
    ax.bar(x, p95, width, label="P95", color="#55A868")
    ax.bar([i + width for i in x], p99, width, label="P99", color="#C44E52")

    ax.set_xticks(list(x))
    ax.set_xticklabels(precisions)
    ax.set_xlabel("Precision", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Latency Percentiles by Precision\n(Phi-3 Mini 3.8B — 256 output tokens)", fontsize=13)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def results_to_dataframe(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "Precision": r["precision"],
            "Batch Size": r["batch_size"],
            "P50 (ms)": r["latency"]["p50_ms"],
            "P95 (ms)": r["latency"]["p95_ms"],
            "P99 (ms)": r["latency"]["p99_ms"],
            "Avg TPS": r["throughput"]["avg_tokens_per_second"],
            "N Runs": r["n_runs"],
        })
    return pd.DataFrame(rows)
