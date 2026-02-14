#!/usr/bin/env python3
"""
Mercury Agent — Generate Documentation Images

Reads ONLY from benchmarks/honest_benchmark_results.json.
No hardcoded values. Every number comes from measured data.

Usage:
    python scripts/generate_docs_images.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_FILE = Path(__file__).parent.parent / "benchmarks" / "honest_benchmark_results.json"
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "images"


def load_results() -> dict:
    """Load honest benchmark results JSON."""
    if not RESULTS_FILE.exists():
        print(f"ERROR: {RESULTS_FILE} not found. Run honest_benchmark.py first.", file=sys.stderr)
        sys.exit(1)
    with open(RESULTS_FILE) as f:
        return json.load(f)


def generate_auc_bar_chart(data: dict) -> Path:
    """Bar chart of per-dataset ROC-AUC, sorted descending."""
    results = [r for r in data["results"] if "roc_auc" in r and "error" not in r]
    results.sort(key=lambda r: r["roc_auc"], reverse=True)

    names = [r["dataset"] for r in results]
    aucs = [r["roc_auc"] for r in results]

    fig, ax = plt.subplots(figsize=(14, max(6, len(names) * 0.35)))
    colors = ["#2ecc71" if a >= 0.7 else "#f39c12" if a >= 0.5 else "#e74c3c" for a in aucs]
    bars = ax.barh(range(len(names)), aucs, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("ROC-AUC")
    ax.set_title(
        f"Mercury Agent — Per-Dataset ROC-AUC\n"
        f"Mean: {data['summary']['mean_auc']:.3f}  "
        f"Median: {data['summary']['median_auc']:.3f}  "
        f"({data['summary']['n_datasets_succeeded']}/{data['summary']['n_datasets_attempted']} datasets)"
    )
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=data["summary"]["mean_auc"], color="#3498db", linestyle="--", alpha=0.7, label="Mean")
    ax.legend()
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2, axis="x")

    for bar, val in zip(bars, aucs):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=6,
        )

    plt.tight_layout()
    out = OUTPUT_DIR / "benchmark_auc_per_dataset.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Generated: {out}")
    return out


def generate_summary_dashboard(data: dict) -> Path:
    """2x2 dashboard: AUC distribution, F1 distribution, component AUC, timing."""
    results = [r for r in data["results"] if "roc_auc" in r and "error" not in r]
    summary = data["summary"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"Mercury Agent — Benchmark Summary  (commit {data.get('git_commit', '?')})",
        fontsize=14, fontweight="bold",
    )

    # Panel 1: AUC histogram
    ax1 = axes[0, 0]
    aucs = [r["roc_auc"] for r in results]
    ax1.hist(aucs, bins=20, color="#3498db", edgecolor="black", alpha=0.8)
    ax1.axvline(x=summary["mean_auc"], color="#e74c3c", linestyle="--", label=f"Mean={summary['mean_auc']:.3f}")
    ax1.axvline(x=summary["median_auc"], color="#2ecc71", linestyle="--", label=f"Median={summary['median_auc']:.3f}")
    ax1.set_xlabel("ROC-AUC")
    ax1.set_ylabel("Count")
    ax1.set_title("AUC Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Oracle F1 histogram
    ax2 = axes[0, 1]
    f1s = [r["oracle_f1"] for r in results]
    ax2.hist(f1s, bins=20, color="#9b59b6", edgecolor="black", alpha=0.8)
    ax2.axvline(x=summary["mean_oracle_f1"], color="#e74c3c", linestyle="--", label=f"Mean={summary['mean_oracle_f1']:.3f}")
    ax2.set_xlabel("Oracle F1")
    ax2.set_ylabel("Count")
    ax2.set_title("Oracle F1 Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Component AUC comparison
    ax3 = axes[1, 0]
    components = ["resonance_auc", "kinematic_auc", "info_geometry_auc"]
    labels = ["Resonance", "Kinematic", "InfoGeometry"]
    comp_means = []
    for comp in components:
        vals = [r[comp] for r in results if comp in r]
        comp_means.append(float(np.mean(vals)) if vals else 0.0)
    bars = ax3.bar(labels, comp_means, color=["#e74c3c", "#3498db", "#2ecc71"], edgecolor="black")
    ax3.set_ylabel("Mean AUC")
    ax3.set_title("Per-Component Mean AUC")
    ax3.set_ylim(0, 1.05)
    for bar, val in zip(bars, comp_means):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.3f}", ha="center")
    ax3.grid(True, alpha=0.3, axis="y")

    # Panel 4: Fit + Score timing
    ax4 = axes[1, 1]
    fit_times = [r.get("fit_time", 0) for r in results]
    score_times = [r.get("score_time", 0) for r in results]
    ax4.scatter(fit_times, score_times, alpha=0.6, color="#e67e22", edgecolors="black", linewidth=0.5)
    ax4.set_xlabel("Fit Time (s)")
    ax4.set_ylabel("Score Time (s)")
    ax4.set_title("Timing: Fit vs Score")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "benchmark_summary_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Generated: {out}")
    return out


def main() -> None:
    """Generate all documentation images from measured data."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_results()
    print(f"Loaded {RESULTS_FILE} ({data['summary']['n_datasets_succeeded']} datasets)")
    generate_auc_bar_chart(data)
    generate_summary_dashboard(data)
    print("Done.")


if __name__ == "__main__":
    main()
