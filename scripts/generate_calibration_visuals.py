"""
Mercury Agent - Calibration Validation Visualizations
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Generates three visualizations from calibration_validation_results.json:
1. Calibration Improvement Bar Chart (MD-011)
2. Conformal Coverage Plot (MD-005)
3. Adaptive Weight Distribution (MD-003)

Usage:
    python scripts/generate_calibration_visuals.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_PATH = Path(__file__).parent.parent / "benchmarks" / "calibration_validation_results.json"
IMAGES_DIR = Path(__file__).parent.parent / "docs" / "images"


def load_results() -> dict:
    """Load calibration validation results."""
    with open(RESULTS_PATH) as f:
        return json.load(f)


def generate_calibration_improvement(data: dict) -> None:
    """Generate calibration improvement bar chart (MD-011)."""
    results = [
        r for r in data["results"]
        if r.get("error") is None and "calibration" in r
        and "error" not in r["calibration"]
    ]

    # Sort by delta descending
    results.sort(key=lambda r: r["calibration"]["delta_f1"], reverse=True)

    names = [r["name"] for r in results]
    cal_f1 = [r["calibration"]["calibrated_f1"] for r in results]
    uncal_f1 = [r["calibration"]["uncalibrated_f1"] for r in results]
    deltas = [r["calibration"]["delta_f1"] for r in results]

    fig, ax = plt.subplots(figsize=(12, max(8, len(names) * 0.3)))

    y_pos = np.arange(len(names))
    bar_height = 0.35

    # Uncalibrated (gray)
    ax.barh(y_pos + bar_height / 2, uncal_f1, bar_height,
            color="#CCCCCC", label="Uncalibrated (0.5)", edgecolor="white")
    # Calibrated (blue)
    ax.barh(y_pos - bar_height / 2, cal_f1, bar_height,
            color="#4C72B0", label="Calibrated", edgecolor="white")

    # Delta markers
    for i, delta in enumerate(deltas):
        color = "#2CA02C" if delta > 0 else "#D62728" if delta < 0 else "#999999"
        marker = ">" if delta > 0 else "<" if delta < 0 else "o"
        ax.plot(max(cal_f1[i], uncal_f1[i]) + 0.02, i, marker=marker,
                color=color, markersize=6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("F1 Score")
    ax.set_title("Calibration Improvement: Calibrated vs Default Threshold (MD-011)")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)

    # Add summary annotation
    n_improved = sum(1 for d in deltas if d > 0)
    mean_delta = np.mean(deltas)
    ax.annotate(
        f"Improved: {n_improved}/{len(deltas)} | Mean \u0394F1: {mean_delta:+.3f}",
        xy=(0.98, 0.02), xycoords="axes fraction",
        ha="right", fontsize=9, bbox={"boxstyle": "round", "fc": "wheat", "alpha": 0.8},
    )

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "calibration_improvement.png", dpi=150)
    plt.close()
    print(f"  Saved: {IMAGES_DIR / 'calibration_improvement.png'}")


def generate_conformal_coverage(data: dict) -> None:
    """Generate conformal coverage scatter plot (MD-005)."""
    results = [
        r for r in data["results"]
        if r.get("error") is None and "conformal" in r
        and "coverage_results" in r.get("conformal", {})
    ]

    if not results:
        print("  SKIPPED: No conformal coverage data available")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    targets = []
    empiricals = []
    names = []

    for r in results:
        for cov in r["conformal"]["coverage_results"]:
            if "error" in cov:
                continue
            targets.append(cov["target_coverage"])
            empiricals.append(cov["empirical_coverage"])
            names.append(r["name"])

    targets_arr = np.array(targets)
    empiricals_arr = np.array(empiricals)

    # Color by target level
    colors = {0.90: "#4C72B0", 0.95: "#DD8452", 0.99: "#55A868"}
    for tgt in [0.90, 0.95, 0.99]:
        mask = targets_arr == tgt
        ax.scatter(
            targets_arr[mask], empiricals_arr[mask],
            c=colors[tgt], alpha=0.6, s=40,
            label=f"Target={tgt:.0%}",
            edgecolors="white", linewidth=0.5,
        )

    # Diagonal reference line
    ax.plot([0.5, 1.0], [0.5, 1.0], "k--", linewidth=1, alpha=0.5,
            label="Perfect calibration")

    ax.set_xlabel("Target Coverage")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("Conformal Coverage: Empirical vs Target (MD-005)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0.85, 1.02)
    ax.set_ylim(0.4, 1.05)

    # Count meets
    meets_90 = sum(1 for t, e in zip(targets, empiricals) if t == 0.90 and e >= t)
    meets_95 = sum(1 for t, e in zip(targets, empiricals) if t == 0.95 and e >= t)
    meets_99 = sum(1 for t, e in zip(targets, empiricals) if t == 0.99 and e >= t)
    total_per_level = sum(1 for t in targets if t == 0.90)

    ax.annotate(
        f"Meets guarantee: 90%={meets_90}/{total_per_level}, "
        f"95%={meets_95}/{total_per_level}, 99%={meets_99}/{total_per_level}\n"
        f"Below diagonal = overconfident",
        xy=(0.02, 0.02), xycoords="axes fraction",
        fontsize=8, bbox={"boxstyle": "round", "fc": "wheat", "alpha": 0.8},
    )

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "conformal_coverage.png", dpi=150)
    plt.close()
    print(f"  Saved: {IMAGES_DIR / 'conformal_coverage.png'}")


def generate_weight_distribution(data: dict) -> None:
    """Generate adaptive weight distribution box plot (MD-003)."""
    results = [
        r for r in data["results"]
        if r.get("error") is None and "fusion" in r
        and r["fusion"].get("adaptive_weights") is not None
    ]

    if not results:
        print("  SKIPPED: No adaptive weight data available")
        return

    weights = np.array([r["fusion"]["adaptive_weights"] for r in results])

    fig, ax = plt.subplots(figsize=(8, 5))

    component_names = ["Resonance", "Kinematic", "InfoGeometry"]
    default_weights = [0.40, 0.30, 0.30]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    positions = np.arange(len(component_names))

    # Box plots
    bp = ax.boxplot(
        [weights[:, i] for i in range(3)],
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=True,
        flierprops={"markersize": 4, "alpha": 0.5},
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Reference lines at defaults
    for i, (default, color) in enumerate(zip(default_weights, colors)):
        ax.hlines(default, i - 0.3, i + 0.3, colors="red",
                  linestyles="dashed", linewidth=1.5,
                  label="Default" if i == 0 else None)

    ax.set_xticks(positions)
    ax.set_xticklabels(component_names)
    ax.set_ylabel("Weight")
    ax.set_title("Adaptive Ensemble Weight Distribution Across Datasets (MD-003)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right")

    # Summary stats
    for i, name in enumerate(component_names):
        mean = np.mean(weights[:, i])
        std = np.std(weights[:, i])
        ax.annotate(
            f"\u03bc={mean:.2f}\n\u03c3={std:.2f}",
            xy=(i, 1.0), ha="center", fontsize=8,
            bbox={"boxstyle": "round", "fc": "white", "alpha": 0.8},
        )

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "adaptive_weight_distribution.png", dpi=150)
    plt.close()
    print(f"  Saved: {IMAGES_DIR / 'adaptive_weight_distribution.png'}")


def main() -> None:
    """Generate all calibration visualizations."""
    print("Generating calibration validation visuals ...")

    if not RESULTS_PATH.exists():
        print(f"ERROR: {RESULTS_PATH} not found. Run calibration_validation.py first.")
        sys.exit(1)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    data = load_results()

    generate_calibration_improvement(data)
    generate_conformal_coverage(data)
    generate_weight_distribution(data)

    print("Done.")


if __name__ == "__main__":
    main()
