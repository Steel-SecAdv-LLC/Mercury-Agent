# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Calibration Validation Visualizations.

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
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_PATH = Path(__file__).parent.parent / "benchmarks" / "calibration_validation_results.json"
IMAGES_DIR = Path(__file__).parent.parent / "docs" / "images"


def load_results() -> dict[str, Any]:
    """Load calibration validation results."""
    with open(RESULTS_PATH) as f:
        # json.load returns Any; the validation results file is a JSON object.
        return cast("dict[str, Any]", json.load(f))


def generate_calibration_improvement(data: dict[str, Any]) -> None:
    """Generate calibration improvement bar chart (MD-011)."""
    results = [
        r
        for r in data["results"]
        if r.get("error") is None and "calibration" in r and "error" not in r["calibration"]
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
    ax.barh(
        y_pos + bar_height / 2,
        uncal_f1,
        bar_height,
        color="#CCCCCC",
        label="Uncalibrated (0.5)",
        edgecolor="white",
    )
    # Calibrated (blue)
    ax.barh(
        y_pos - bar_height / 2,
        cal_f1,
        bar_height,
        color="#4C72B0",
        label="Calibrated",
        edgecolor="white",
    )

    # Delta markers
    for i, delta in enumerate(deltas):
        color = "#2CA02C" if delta > 0 else "#D62728" if delta < 0 else "#999999"
        marker = ">" if delta > 0 else "<" if delta < 0 else "o"
        ax.plot(max(cal_f1[i], uncal_f1[i]) + 0.02, i, marker=marker, color=color, markersize=6)

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
        xy=(0.98, 0.02),
        xycoords="axes fraction",
        ha="right",
        fontsize=9,
        bbox={"boxstyle": "round", "fc": "wheat", "alpha": 0.8},
    )

    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "calibration_improvement.png", dpi=150)
    plt.close()
    print(f"  Saved: {IMAGES_DIR / 'calibration_improvement.png'}")


def generate_weight_distribution(data: dict[str, Any]) -> None:
    """Generate adaptive weight distribution box plot with CV overlay (MD-003)."""
    results = [
        r
        for r in data["results"]
        if r.get("error") is None
        and "fusion" in r
        and r["fusion"].get("adaptive_weights") is not None
    ]

    if not results:
        print("  SKIPPED: No adaptive weight data available")
        return

    weights = np.array([r["fusion"]["adaptive_weights"] for r in results])

    # Collect CV-optimal weights where available
    cv_optimal_weights: list[list[float]] = []
    cv_default_validated = 0
    cv_adaptive_validated = 0
    cv_total = 0
    for r in results:
        fcv = r.get("fusion_cv", {})
        if fcv.get("mean_optimal_weights") is not None:
            cv_optimal_weights.append(fcv["mean_optimal_weights"])
            cv_total += 1
            if fcv.get("delta_optimal_vs_default", 1.0) < 0.02:
                cv_default_validated += 1
            if fcv.get("delta_optimal_vs_adaptive", 1.0) < 0.02:
                cv_adaptive_validated += 1

    fig, ax = plt.subplots(figsize=(8, 5))

    component_names = ["Resonance", "Kinematic", "InfoGeometry"]
    default_weights = [0.40, 0.30, 0.30]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    positions = np.arange(len(component_names))

    # Box plots of adaptive weights
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

    # Reference lines at defaults (red dashed)
    for i, default in enumerate(default_weights):
        ax.hlines(
            default,
            i - 0.3,
            i + 0.3,
            colors="red",
            linestyles="dashed",
            linewidth=1.5,
            # "_nolegend_" hides the entry exactly like None, but satisfies hlines' str type.
            label="Default (0.4/0.3/0.3)" if i == 0 else "_nolegend_",
        )

    # CV-optimal weight overlay (diamond markers)
    if cv_optimal_weights:
        cv_arr = np.array(cv_optimal_weights)
        cv_means = np.mean(cv_arr, axis=0)
        for i, mean_w in enumerate(cv_means):
            ax.plot(
                i,
                mean_w,
                marker="D",
                color="#8B0000",
                markersize=10,
                zorder=5,
                markeredgecolor="white",
                markeredgewidth=1.0,
                label="CV-Optimal (L-BFGS-B)" if i == 0 else None,
            )

    # Add invisible artist for legend entry for box plots
    ax.plot(
        [], [], "s", color="#4C72B0", alpha=0.6, markersize=8, label="Adaptive (AUC-proportional)"
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(component_names)
    ax.set_ylabel("Weight")
    ax.set_title("Ensemble Weight Distribution: Adaptive vs CV-Optimal (MD-003)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=8)

    # Summary stats
    for i, name in enumerate(component_names):
        mean = np.mean(weights[:, i])
        std = np.std(weights[:, i])
        ax.annotate(
            f"\u03bc={mean:.2f}\n\u03c3={std:.2f}",
            xy=(i, 1.0),
            ha="center",
            fontsize=8,
            bbox={"boxstyle": "round", "fc": "white", "alpha": 0.8},
        )

    # CV validation annotation
    if cv_total > 0:
        ax.annotate(
            f"CV validation (<0.02 gap):\n"
            f"  Default: {cv_default_validated}/{cv_total} "
            f"({cv_default_validated / cv_total * 100:.1f}%)\n"
            f"  Adaptive: {cv_adaptive_validated}/{cv_total} "
            f"({cv_adaptive_validated / cv_total * 100:.1f}%)",
            xy=(0.02, 0.02),
            xycoords="axes fraction",
            fontsize=8,
            bbox={"boxstyle": "round", "fc": "wheat", "alpha": 0.8},
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
    # The standalone conformal_coverage.png was retired in PR #139: its
    # empirical-vs-target scatter is a panel of calibration_improvement.png
    # (docs/BENCHMARKS.md, MD-005).
    generate_weight_distribution(data)

    print("Done.")


if __name__ == "__main__":
    main()
