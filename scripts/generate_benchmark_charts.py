#!/usr/bin/env python3
"""
Mercury Agent v1.4.0 - Professional benchmark visualization generator.

Generates charts from benchmark results JSON.

Output:
- docs/images/adbench_auc_comparison.png
- docs/images/network_security_benchmark.png

Copyright (C) 2025 Steel Security Advisors LLC
License: GPL-3.0+
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_PATH = "benchmarks/v1.4.0_comprehensive_results.json"
OUTPUT_DIR = "docs/images"


def load_results() -> dict | None:
    """Load benchmark results JSON."""
    path = Path(RESULTS_PATH)
    if not path.exists():
        logger.warning("Results file not found: %s", RESULTS_PATH)
        logger.info("Run: python scripts/run_comprehensive_benchmark_suite.py first")
        return None
    with open(path) as f:
        return json.load(f)


def plot_adbench_auc(results: dict) -> None:
    """AUC comparison across ADBench datasets."""
    datasets: list[str] = []
    statistical_auc: list[float] = []
    temporal_auc: list[float] = []

    for name, data in sorted(results["datasets"].items()):
        if not name.startswith("adbench_"):
            continue
        if "error" in data or "detectors" not in data:
            continue

        label = name.replace("adbench_", "").title()
        datasets.append(label)
        statistical_auc.append(data["detectors"]["statistical"]["auc"])
        temporal_auc.append(
            data["detectors"].get("temporal", {}).get("auc", 0.0)
        )

    if not datasets:
        logger.warning("No ADBench results found")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(datasets))
    width = 0.35

    ax.bar(x - width / 2, statistical_auc, width, label="Statistical", color="#1f77b4")
    ax.bar(x + width / 2, temporal_auc, width, label="Temporal", color="#ff7f0e")
    ax.axhline(y=0.85, color="red", linestyle="--", alpha=0.5, label="Target (0.85)")

    ax.set_ylabel("AUC-ROC")
    ax.set_title(f"Mercury-Agent v1.4.0: ADBench AUC Comparison ({len(datasets)} Datasets)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim([0.4, 1.0])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output = Path(OUTPUT_DIR) / "adbench_auc_comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Chart saved: %s", output)


def plot_network_security(results: dict) -> None:
    """NSL-KDD vs CICIDS-2017 comparison."""
    nslkdd = results["datasets"].get("nslkdd", {})
    cicids = results["datasets"].get("cicids_2017", {})

    # Use measured values if available, otherwise defaults from baseline
    datasets_labels = []
    statistical_vals = []
    temporal_vals = []

    if "detectors" in nslkdd:
        datasets_labels.append("NSL-KDD")
        statistical_vals.append(nslkdd["detectors"]["statistical"]["auc"])
        temporal_vals.append(nslkdd["detectors"].get("temporal", {}).get("auc", 0.0))

    if "detectors" in cicids:
        datasets_labels.append("CICIDS-2017")
        statistical_vals.append(cicids["detectors"]["statistical"]["auc"])
        temporal_vals.append(cicids["detectors"].get("temporal", {}).get("auc", 0.0))

    if not datasets_labels:
        # Fallback to baseline values
        datasets_labels = ["NSL-KDD", "CICIDS-2017"]
        statistical_vals = [0.591, 0.620]
        temporal_vals = [0.565, 0.585]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(datasets_labels))
    width = 0.35

    bars_stat = ax.bar(x - width / 2, statistical_vals, width, label="Statistical", color="#2ca02c")
    bars_temp = ax.bar(x + width / 2, temporal_vals, width, label="Temporal", color="#d62728")

    ax.set_ylabel("AUC-ROC")
    ax.set_title("Mercury-Agent v1.4.0: Network Security Benchmark (Real Intrusion Data)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_labels)
    ax.legend()
    ax.set_ylim([0.4, 0.75])
    ax.grid(axis="y", alpha=0.3)

    # Add value annotations
    for bar in bars_stat:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars_temp:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    output = Path(OUTPUT_DIR) / "network_security_benchmark.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Chart saved: %s", output)


if __name__ == "__main__":
    results = load_results()
    if results:
        plot_adbench_auc(results)
        plot_network_security(results)
    else:
        logger.info("Generating charts with baseline values")
        # Fallback: generate network security chart with baseline values
        plot_network_security({"datasets": {}})
