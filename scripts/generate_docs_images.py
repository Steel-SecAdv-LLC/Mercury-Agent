#!/usr/bin/env python3
"""Generate docs/images/ dashboard PNGs from measured benchmark data.

Reads benchmarks/honest_benchmark_results.json as the sole data source.
Every number on every chart comes from that JSON.  If a metric was not
measured, the chart section shows "Not measured".

Produces:
  docs/images/neuro_symbolic_benchmark_report.png
  docs/images/anomaly_detection_panel.png
  docs/images/mercury_performance_dashboard.png
  docs/images/benchmark_summary_live_data.png

Run:
  python scripts/generate_docs_images.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_PATH = Path(__file__).parent.parent / "benchmarks" / "honest_benchmark_results.json"
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "images"

# ---------------------------------------------------------------------------
# Dark theme
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#1A1A2E",
    "card": "#16213E",
    "primary": "#00D4AA",
    "secondary": "#4ECDC4",
    "accent": "#FFB347",
    "text": "#E0E0E0",
    "muted": "#888888",
    "good": "#00D4AA",
    "warn": "#FFB347",
    "bad": "#FF6B6B",
}


def _apply_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": COLORS["card"],
            "axes.edgecolor": COLORS["muted"],
            "axes.labelcolor": COLORS["text"],
            "text.color": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "font.size": 10,
            "savefig.dpi": 200,
            "savefig.facecolor": COLORS["bg"],
        }
    )


def _load_data() -> dict:
    if not RESULTS_PATH.exists():
        print(
            f"ERROR: {RESULTS_PATH} not found.\n"
            "Run 'python benchmarks/honest_benchmark.py' first to generate measured results."
        )
        sys.exit(1)
    with open(RESULTS_PATH) as f:
        return json.load(f)


def _stamp(ax: plt.Axes) -> None:
    """Add source file and timestamp label."""
    ax.annotate(
        f"Source: honest_benchmark_results.json | {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        xy=(0.99, 0.01),
        xycoords="figure fraction",
        fontsize=6,
        color=COLORS["muted"],
        ha="right",
        va="bottom",
    )


# ---------------------------------------------------------------------------
# Chart 1: Neuro-Symbolic Benchmark Report
# ---------------------------------------------------------------------------
def generate_neuro_symbolic_report(data: dict) -> None:
    summary = data.get("summary", {})
    comp = data.get("component_summary", {})
    per_ds = [r for r in data.get("per_dataset", []) if r.get("error") is None]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Mercury Agent - Ensemble Benchmark Report", fontsize=16, fontweight="bold")

    # Top-left: per-component AUC comparison
    ax = axes[0, 0]
    comp_names = []
    comp_means = []
    comp_stds = []
    for name in ["resonance", "kinematic", "info_geometry"]:
        if name in comp:
            comp_names.append(name.replace("_", "\n"))
            comp_means.append(comp[name]["mean_auc"])
            comp_stds.append(comp[name]["std_auc"])
    if comp.get("resonance") is not None:
        comp_names.append("ensemble")
        comp_means.append(summary.get("mean_auc", 0))
        comp_stds.append(summary.get("std_auc", 0))

    if comp_names:
        colors = [COLORS["secondary"], COLORS["accent"], COLORS["bad"], COLORS["primary"]]
        bars = ax.bar(
            comp_names, comp_means, yerr=comp_stds, color=colors[: len(comp_names)], capsize=5
        )
        for bar, val in zip(bars, comp_means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                fontsize=9,
            )
        ax.set_ylabel("Mean AUC")
        ax.set_title("Component vs Ensemble AUC")
        ax.set_ylim(0, 1.05)
    else:
        ax.text(0.5, 0.5, "Not measured", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Component vs Ensemble AUC")

    # Top-right: AUC distribution histogram
    ax = axes[0, 1]
    aucs = [r["ensemble_auc"] for r in per_ds if not np.isnan(r.get("ensemble_auc", float("nan")))]
    if aucs:
        ax.hist(aucs, bins=20, color=COLORS["primary"], alpha=0.8, edgecolor=COLORS["bg"])
        ax.axvline(
            np.median(aucs),
            color=COLORS["accent"],
            linestyle="--",
            label=f"median={np.median(aucs):.3f}",
        )
        ax.set_xlabel("AUC")
        ax.set_ylabel("Count")
        ax.set_title("AUC Distribution Across Datasets")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Not measured", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("AUC Distribution")

    # Bottom-left: Top/Bottom datasets by AUC
    ax = axes[1, 0]
    sorted_ds = sorted(per_ds, key=lambda r: r.get("ensemble_auc", 0), reverse=True)
    if len(sorted_ds) >= 6:
        show = sorted_ds[:5] + sorted_ds[-5:]
        names = [r["name"][:20] for r in show]
        vals = [r["ensemble_auc"] for r in show]
        colors_bar = [
            COLORS["good"] if v >= 0.7 else COLORS["warn"] if v >= 0.5 else COLORS["bad"]
            for v in vals
        ]
        ax.barh(names[::-1], vals[::-1], color=colors_bar[::-1])
        ax.set_xlabel("AUC")
        ax.set_title("Top 5 / Bottom 5 Datasets")
        ax.set_xlim(0, 1.05)
    else:
        ax.text(0.5, 0.5, "Not enough datasets", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Top/Bottom Datasets")

    # Bottom-right: summary stats text
    ax = axes[1, 1]
    ax.axis("off")
    meta = data.get("metadata", {})
    lines = [
        f"Detector: {meta.get('detector', 'N/A')}",
        f"Weights: R={meta.get('ensemble_weights', {}).get('resonance', '?')}, "
        f"K={meta.get('ensemble_weights', {}).get('kinematic', '?')}, "
        f"IG={meta.get('ensemble_weights', {}).get('info_geometry', '?')}",
        "",
        f"Datasets tested: {summary.get('total_datasets', 0)}",
        f"Successful: {summary.get('successful', 0)}",
        f"Failed: {summary.get('failed', 0)}",
        "",
        f"Mean AUC: {summary.get('mean_auc', 'N/A')}",
        f"Median AUC: {summary.get('median_auc', 'N/A')}",
        f"Mean Oracle F1: {summary.get('mean_oracle_f1', 'N/A')}",
        "",
        f"Git: {meta.get('git_commit', 'N/A')[:12]}",
        f"Generated: {meta.get('timestamp', 'N/A')[:19]}",
    ]
    ax.text(
        0.1,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        family="monospace",
    )
    ax.set_title("Summary")

    _stamp(axes[0, 0])
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(OUTPUT_DIR / "neuro_symbolic_benchmark_report.png")
    plt.close(fig)
    print("  Saved neuro_symbolic_benchmark_report.png")


# ---------------------------------------------------------------------------
# Chart 2: Anomaly Detection Panel
# ---------------------------------------------------------------------------
def generate_anomaly_detection_panel(data: dict) -> None:
    per_ds = [r for r in data.get("per_dataset", []) if r.get("error") is None]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Mercury Agent - Anomaly Detection Panel", fontsize=16, fontweight="bold")

    # Top-left: AUC by category
    ax = axes[0, 0]
    cats: dict[str, list[float]] = {}
    for r in per_ds:
        cat = r.get("category", "other")
        auc = r.get("ensemble_auc", float("nan"))
        if not np.isnan(auc):
            cats.setdefault(cat, []).append(auc)
    if cats:
        cat_names = sorted(cats.keys())
        cat_means = [np.mean(cats[c]) for c in cat_names]
        ax.barh(cat_names, cat_means, color=COLORS["secondary"])
        for i, v in enumerate(cat_means):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
        ax.set_xlabel("Mean AUC")
        ax.set_title("AUC by Dataset Category")
        ax.set_xlim(0, 1.05)
    else:
        ax.text(0.5, 0.5, "Not measured", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("AUC by Category")

    # Top-right: Oracle F1 distribution
    ax = axes[0, 1]
    f1s = [r["oracle_f1"] for r in per_ds if r.get("oracle_f1", 0) > 0]
    if f1s:
        ax.hist(f1s, bins=20, color=COLORS["accent"], alpha=0.8, edgecolor=COLORS["bg"])
        ax.axvline(
            np.median(f1s),
            color=COLORS["primary"],
            linestyle="--",
            label=f"median={np.median(f1s):.3f}",
        )
        ax.set_xlabel("Oracle F1")
        ax.set_ylabel("Count")
        ax.set_title("Oracle F1 Distribution (upper bound)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Not measured", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Oracle F1 Distribution")

    # Bottom-left: per-component scatter (resonance vs info_geo)
    ax = axes[1, 0]
    res_aucs = [r.get("resonance_auc", float("nan")) for r in per_ds]
    ig_aucs = [r.get("info_geometry_auc", float("nan")) for r in per_ds]
    valid = [(r, ig) for r, ig in zip(res_aucs, ig_aucs) if not np.isnan(r) and not np.isnan(ig)]
    if valid:
        rx, igx = zip(*valid)
        ax.scatter(rx, igx, c=COLORS["primary"], alpha=0.6, s=20)
        ax.plot([0, 1], [0, 1], "--", color=COLORS["muted"], alpha=0.5)
        ax.set_xlabel("Resonance AUC")
        ax.set_ylabel("InfoGeometry AUC")
        ax.set_title("Component Correlation")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
    else:
        ax.text(0.5, 0.5, "Not measured", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Component Correlation")

    # Bottom-right: timing
    ax = axes[1, 1]
    fit_times = [r["fit_ms"] for r in per_ds if "fit_ms" in r]
    score_times = [r["score_ms"] for r in per_ds if "score_ms" in r]
    if fit_times:
        ax.boxplot(
            [fit_times, score_times],
            labels=["fit()", "detect()"],
            patch_artist=True,
            boxprops={"facecolor": COLORS["card"], "edgecolor": COLORS["primary"]},
            medianprops={"color": COLORS["accent"]},
            whiskerprops={"color": COLORS["text"]},
            capprops={"color": COLORS["text"]},
        )
        ax.set_ylabel("Time (ms)")
        ax.set_title("Timing Distribution")
    else:
        ax.text(0.5, 0.5, "Not measured", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Timing")

    _stamp(axes[0, 0])
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(OUTPUT_DIR / "anomaly_detection_panel.png")
    plt.close(fig)
    print("  Saved anomaly_detection_panel.png")


# ---------------------------------------------------------------------------
# Chart 3: Performance Dashboard
# ---------------------------------------------------------------------------
def generate_performance_dashboard(data: dict) -> None:
    per_ds = [r for r in data.get("per_dataset", []) if r.get("error") is None]
    summary = data.get("summary", {})

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Mercury Agent - Performance Dashboard", fontsize=16, fontweight="bold")

    # Top-left: AUC vs anomaly ratio
    ax = axes[0, 0]
    anom_ratios = [r.get("test_anomaly_ratio", 0) for r in per_ds]
    aucs = [r.get("ensemble_auc", float("nan")) for r in per_ds]
    valid = [(ar, a) for ar, a in zip(anom_ratios, aucs) if not np.isnan(a)]
    if valid:
        arx, ax_vals = zip(*valid)
        ax.scatter(arx, ax_vals, c=COLORS["primary"], alpha=0.6, s=20)
        ax.set_xlabel("Anomaly Ratio")
        ax.set_ylabel("Ensemble AUC")
        ax.set_title("AUC vs Anomaly Ratio")
        ax.set_ylim(0, 1.05)
    else:
        ax.text(0.5, 0.5, "Not measured", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("AUC vs Anomaly Ratio")

    # Top-right: AUC vs n_features
    ax = axes[0, 1]
    nf = [r.get("n_features", 0) for r in per_ds]
    valid2 = [(n, a) for n, a in zip(nf, aucs) if not np.isnan(a)]
    if valid2:
        nx, ax2 = zip(*valid2)
        ax.scatter(nx, ax2, c=COLORS["accent"], alpha=0.6, s=20)
        ax.set_xlabel("Number of Features")
        ax.set_ylabel("Ensemble AUC")
        ax.set_title("AUC vs Dimensionality")
        ax.set_ylim(0, 1.05)
    else:
        ax.text(0.5, 0.5, "Not measured", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("AUC vs Dimensionality")

    # Bottom-left: Precision vs Recall scatter
    ax = axes[1, 0]
    precs = [r.get("oracle_precision", 0) for r in per_ds]
    recs = [r.get("oracle_recall", 0) for r in per_ds]
    valid3 = [(p, r) for p, r in zip(precs, recs) if p > 0 or r > 0]
    if valid3:
        px, rx = zip(*valid3)
        ax.scatter(px, rx, c=COLORS["secondary"], alpha=0.6, s=20)
        ax.set_xlabel("Oracle Precision")
        ax.set_ylabel("Oracle Recall")
        ax.set_title("Precision-Recall Trade-off (oracle threshold)")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
    else:
        ax.text(0.5, 0.5, "Not measured", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Precision vs Recall")

    # Bottom-right: key metrics
    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        f"Mean AUC:        {summary.get('mean_auc', 'N/A')}",
        f"Median AUC:      {summary.get('median_auc', 'N/A')}",
        f"Mean Oracle F1:  {summary.get('mean_oracle_f1', 'N/A')}",
        f"Median Oracle F1:{summary.get('median_oracle_f1', 'N/A')}",
        "",
        "NOTE: Oracle F1 is an upper bound.",
        "Threshold was selected per-dataset",
        "on the test set (101 thresholds).",
    ]
    ax.text(
        0.1,
        0.9,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
    )
    ax.set_title("Key Metrics")

    _stamp(axes[0, 0])
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(OUTPUT_DIR / "mercury_performance_dashboard.png")
    plt.close(fig)
    print("  Saved mercury_performance_dashboard.png")


# ---------------------------------------------------------------------------
# Chart 4: Benchmark Summary (Live Data)
# ---------------------------------------------------------------------------
def generate_benchmark_summary(data: dict) -> None:
    per_ds = [r for r in data.get("per_dataset", []) if r.get("error") is None]
    summary = data.get("summary", {})
    meta = data.get("metadata", {})

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Mercury Agent - Benchmark Summary", fontsize=16, fontweight="bold")

    # Left: sorted AUC bar chart (all datasets)
    ax = axes[0]
    sorted_ds = sorted(per_ds, key=lambda r: r.get("ensemble_auc", 0), reverse=True)
    if sorted_ds:
        names = [r["name"][:18] for r in sorted_ds]
        vals = [r["ensemble_auc"] for r in sorted_ds]
        colors_list = [
            COLORS["good"] if v >= 0.7 else COLORS["warn"] if v >= 0.5 else COLORS["bad"]
            for v in vals
        ]
        ax.barh(names[::-1], vals[::-1], color=colors_list[::-1], height=0.7)
        ax.set_xlabel("Ensemble AUC")
        ax.set_title(f"Per-Dataset AUC ({len(sorted_ds)} datasets)")
        ax.set_xlim(0, 1.05)
        ax.tick_params(axis="y", labelsize=6)
    else:
        ax.text(0.5, 0.5, "Not measured", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Per-Dataset AUC")

    # Right: metadata and summary
    ax = axes[1]
    ax.axis("off")
    lines = [
        "BENCHMARK METADATA",
        f"  Detector: {meta.get('detector', 'N/A')}",
        f"  Git commit: {meta.get('git_commit', 'N/A')[:12]}",
        f"  Python: {meta.get('python_version', 'N/A')}",
        f"  Timestamp: {meta.get('timestamp', 'N/A')[:19]}",
        f"  Max samples/dataset: {meta.get('max_samples_per_dataset', 'N/A')}",
        "",
        "AGGREGATE RESULTS",
        f"  Datasets: {summary.get('successful', 0)} / {summary.get('total_datasets', 0)}",
        f"  Mean AUC: {summary.get('mean_auc', 'N/A')}",
        f"  Median AUC: {summary.get('median_auc', 'N/A')}",
        f"  Std AUC: {summary.get('std_auc', 'N/A')}",
        f"  Mean Oracle F1: {summary.get('mean_oracle_f1', 'N/A')}",
        "",
        "ENSEMBLE WEIGHTS",
        f"  Resonance:    {meta.get('ensemble_weights', {}).get('resonance', '?')} (40%)",
        f"  Kinematic:    {meta.get('ensemble_weights', {}).get('kinematic', '?')} (30%)",
        f"  InfoGeometry: {meta.get('ensemble_weights', {}).get('info_geometry', '?')} (30%)",
    ]
    ax.text(
        0.05,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        family="monospace",
    )

    _stamp(axes[0])
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(OUTPUT_DIR / "benchmark_summary_live_data.png")
    plt.close(fig)
    print("  Saved benchmark_summary_live_data.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    _apply_theme()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = _load_data()

    print("Generating dashboard images from measured data ...")
    generate_neuro_symbolic_report(data)
    generate_anomaly_detection_panel(data)
    generate_performance_dashboard(data)
    generate_benchmark_summary(data)
    print("Done.")


if __name__ == "__main__":
    main()
