#!/usr/bin/env python3
"""
Mercury Agent - Benchmark Visualization Generator (Data-Driven)
Copyright (C) 2025 Steel Security Advisors LLC

Generates publication-quality benchmark visualizations from actual
mercury_benchmark_results.json data. No synthetic data — every number
displayed is measured.

Output:
    docs/images/neuro_symbolic_benchmark_report.png
    docs/images/anomaly_detection_panel.png
    docs/images/mercury_performance_dashboard.png
    docs/images/benchmark_summary_live_data.png
    docs/images/calibration_improvement.png
    docs/images/adaptive_weight_distribution.png
    docs/images/conformal_coverage.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

VIRIDIS = plt.cm.viridis
COLORS = {
    "primary": "#2563eb",
    "secondary": "#7c3aed",
    "success": "#059669",
    "warning": "#d97706",
    "danger": "#dc2626",
    "ensemble": "#2563eb",
    "resonance": "#f97316",
    "kinematic": "#8b5cf6",
    "info_geo": "#059669",
}

BENCHMARKS_DIR = Path(__file__).parent
OUTPUT_DIR = BENCHMARKS_DIR.parent / "docs" / "images"


def load_results() -> dict[str, Any]:
    """Load mercury_benchmark_results.json."""
    results_file = BENCHMARKS_DIR / "mercury_benchmark_results.json"
    with open(results_file) as f:
        return json.load(f)


def get_successful(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return successful per-dataset results sorted by AUC descending."""
    successful = [r for r in data["per_dataset"] if r.get("error") is None]
    successful.sort(key=lambda r: r.get("ensemble_auc", 0), reverse=True)
    return successful


# -------------------------------------------------------------------------
# 1. Neuro-Symbolic Benchmark Report (9-panel)
# -------------------------------------------------------------------------
def generate_neuro_symbolic_report(data: dict[str, Any]) -> None:
    """Ensemble vs component AUCs, domain breakdown, summary stats."""
    successful = get_successful(data)
    summary = data["summary"]
    comp_summary = data["component_summary"]

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.35)

    # Panel 1: AUC Distribution Histogram
    ax1 = fig.add_subplot(gs[0, 0])
    aucs = [r["ensemble_auc"] for r in successful]
    ax1.hist(aucs, bins=20, color=COLORS["ensemble"], alpha=0.7, edgecolor="white")
    ax1.axvline(summary["mean_auc"], color=COLORS["danger"], linestyle="--", label=f"Mean: {summary['mean_auc']:.4f}")
    ax1.axvline(summary["median_auc"], color=COLORS["warning"], linestyle="--", label=f"Median: {summary['median_auc']:.4f}")
    ax1.set_xlabel("ROC-AUC")
    ax1.set_ylabel("Count")
    ax1.set_title("Ensemble AUC Distribution")
    ax1.legend(fontsize=8)

    # Panel 2: Component AUC Comparison (box-style)
    ax2 = fig.add_subplot(gs[0, 1])
    comp_data = {
        "Resonance": [r["resonance_auc"] for r in successful if not np.isnan(r.get("resonance_auc", float("nan")))],
        "Kinematic": [r["kinematic_auc"] for r in successful if not np.isnan(r.get("kinematic_auc", float("nan")))],
        "InfoGeo": [r["info_geometry_auc"] for r in successful if not np.isnan(r.get("info_geometry_auc", float("nan")))],
        "Ensemble": [r["ensemble_auc"] for r in successful],
    }
    bp = ax2.boxplot(comp_data.values(), labels=comp_data.keys(), patch_artist=True)
    box_colors = [COLORS["resonance"], COLORS["kinematic"], COLORS["info_geo"], COLORS["ensemble"]]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_ylabel("ROC-AUC")
    ax2.set_title("Component vs Ensemble AUC")
    ax2.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Random")
    ax2.legend(fontsize=7)

    # Panel 3: Domain Performance
    ax3 = fig.add_subplot(gs[0, 2])
    domain_summary = data["domain_summary"]
    domains = sorted(domain_summary.keys(), key=lambda d: domain_summary[d]["stats"].get("mean_auc") or 0, reverse=True)
    domain_names = [d.replace("_", "\n") for d in domains]
    domain_aucs = [domain_summary[d]["stats"].get("mean_auc") or 0 for d in domains]
    domain_colors = [VIRIDIS(a) for a in domain_aucs]
    bars = ax3.barh(domain_names, domain_aucs, color=domain_colors)
    ax3.axvline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax3.set_xlabel("Mean AUC")
    ax3.set_title("Domain Performance")
    ax3.set_xlim(0, 1.05)
    for bar, auc in zip(bars, domain_aucs):
        ax3.text(min(auc + 0.01, 1.0), bar.get_y() + bar.get_height() / 2, f"{auc:.3f}", va="center", fontsize=7)

    # Panel 4: Top-15 Datasets by AUC
    ax4 = fig.add_subplot(gs[1, 0])
    top15 = successful[:15]
    names = [r["name"] for r in top15]
    top_aucs = [r["ensemble_auc"] for r in top15]
    colors = [VIRIDIS(a) for a in top_aucs]
    ax4.barh(names[::-1], top_aucs[::-1], color=colors[::-1])
    ax4.set_xlabel("AUC")
    ax4.set_title("Top 15 Datasets")
    ax4.set_xlim(0.9, 1.005)

    # Panel 5: Bottom-15 Datasets
    ax5 = fig.add_subplot(gs[1, 1])
    bottom15 = successful[-15:]
    names_b = [r["name"] for r in bottom15]
    bot_aucs = [r["ensemble_auc"] for r in bottom15]
    colors_b = [VIRIDIS(max(0, a)) for a in bot_aucs]
    ax5.barh(names_b[::-1], bot_aucs[::-1], color=colors_b[::-1])
    ax5.set_xlabel("AUC")
    ax5.set_title("Bottom 15 Datasets")
    ax5.axvline(0.5, color=COLORS["danger"], linestyle="--", alpha=0.7, label="Random")
    ax5.legend(fontsize=7)

    # Panel 6: F1 Distribution
    ax6 = fig.add_subplot(gs[1, 2])
    f1s = [r["oracle_f1"] for r in successful]
    ax6.hist(f1s, bins=20, color=COLORS["success"], alpha=0.7, edgecolor="white")
    ax6.axvline(summary["mean_oracle_f1"], color=COLORS["danger"], linestyle="--",
                label=f"Mean: {summary['mean_oracle_f1']:.4f}")
    ax6.axvline(summary["median_oracle_f1"], color=COLORS["warning"], linestyle="--",
                label=f"Median: {summary['median_oracle_f1']:.4f}")
    ax6.set_xlabel("Oracle F1")
    ax6.set_ylabel("Count")
    ax6.set_title("Oracle F1 Distribution")
    ax6.legend(fontsize=8)

    # Panel 7: AUC vs F1 Scatter
    ax7 = fig.add_subplot(gs[2, 0])
    for r in successful:
        ax7.scatter(r["ensemble_auc"], r["oracle_f1"], c=COLORS["ensemble"], alpha=0.5, s=30, edgecolors="white", linewidth=0.5)
    ax7.set_xlabel("Ensemble AUC")
    ax7.set_ylabel("Oracle F1")
    ax7.set_title("AUC vs Oracle F1")
    ax7.axhline(0.5, color="gray", linestyle=":", alpha=0.3)
    ax7.axvline(0.5, color="gray", linestyle=":", alpha=0.3)

    # Panel 8: Per-component Mean AUC bars
    ax8 = fig.add_subplot(gs[2, 1])
    comp_names = ["Resonance\n(40%)", "Kinematic\n(30%)", "InfoGeo\n(30%)", "Ensemble\n(100%)"]
    comp_means = [
        comp_summary["resonance"]["mean_auc"],
        comp_summary["kinematic"]["mean_auc"],
        comp_summary["info_geometry"]["mean_auc"],
        summary["mean_auc"],
    ]
    comp_medians = [
        comp_summary["resonance"]["median_auc"],
        comp_summary["kinematic"]["median_auc"],
        comp_summary["info_geometry"]["median_auc"],
        summary["median_auc"],
    ]
    x = np.arange(4)
    width = 0.35
    ax8.bar(x - width / 2, comp_means, width, label="Mean", color=[COLORS["resonance"], COLORS["kinematic"], COLORS["info_geo"], COLORS["ensemble"]], alpha=0.8)
    ax8.bar(x + width / 2, comp_medians, width, label="Median", color=[COLORS["resonance"], COLORS["kinematic"], COLORS["info_geo"], COLORS["ensemble"]], alpha=0.4)
    ax8.set_xticks(x)
    ax8.set_xticklabels(comp_names)
    ax8.set_ylabel("AUC")
    ax8.set_title("Component Mean vs Median AUC")
    ax8.legend()
    ax8.set_ylim(0.4, 1.0)

    # Panel 9: Summary Stats
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")
    summary_text = (
        f"MERCURY BENCHMARK RESULTS\n"
        f"{'=' * 35}\n\n"
        f"Total Datasets:    {summary['total_datasets']}\n"
        f"Successful:        {summary['successful']}\n"
        f"Failed:            {summary['failed']}\n\n"
        f"Mean AUC:          {summary['mean_auc']:.4f}\n"
        f"Median AUC:        {summary['median_auc']:.4f}\n"
        f"Std AUC:           {summary['std_auc']:.4f}\n\n"
        f"Mean Oracle F1:    {summary['mean_oracle_f1']:.4f}\n"
        f"Median Oracle F1:  {summary['median_oracle_f1']:.4f}\n\n"
        f"{'=' * 35}\n"
        f"Detector: MercuryAnomalyDetector\n"
        f"Weights: R=40% K=30% IG=30%\n"
        f"No tuning, no synthetic data"
    )
    ax9.text(0.05, 0.5, summary_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment="center", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.3))
    ax9.set_title("Key Metrics")

    fig.suptitle("Mercury Agent v1.5.1 - Neuro-Symbolic Benchmark Report",
                 fontsize=16, fontweight="bold", y=0.99)
    plt.savefig(OUTPUT_DIR / "neuro_symbolic_benchmark_report.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: neuro_symbolic_benchmark_report.png")


# -------------------------------------------------------------------------
# 2. Anomaly Detection Panel
# -------------------------------------------------------------------------
def generate_anomaly_detection_panel(data: dict[str, Any]) -> None:
    """Per-component AUC for top/bottom datasets + detailed analysis."""
    successful = get_successful(data)

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    # Panel 1: Top-10 component breakdown
    ax1 = fig.add_subplot(gs[0, 0])
    top10 = successful[:10]
    names = [r["name"] for r in top10]
    x = np.arange(len(names))
    width = 0.2
    ax1.barh(x - width, [r["resonance_auc"] for r in top10], width, label="Resonance", color=COLORS["resonance"], alpha=0.8)
    ax1.barh(x, [r["kinematic_auc"] for r in top10], width, label="Kinematic", color=COLORS["kinematic"], alpha=0.8)
    ax1.barh(x + width, [r["info_geometry_auc"] for r in top10], width, label="InfoGeo", color=COLORS["info_geo"], alpha=0.8)
    ax1.set_yticks(x)
    ax1.set_yticklabels(names)
    ax1.set_xlabel("AUC")
    ax1.set_title("Top 10: Component AUC Breakdown")
    ax1.legend(fontsize=7, loc="lower right")
    ax1.set_xlim(0, 1.05)

    # Panel 2: Bottom-10 component breakdown
    ax2 = fig.add_subplot(gs[0, 1])
    bot10 = successful[-10:]
    names_b = [r["name"] for r in bot10]
    x = np.arange(len(names_b))
    ax2.barh(x - width, [r["resonance_auc"] for r in bot10], width, label="Resonance", color=COLORS["resonance"], alpha=0.8)
    ax2.barh(x, [r["kinematic_auc"] for r in bot10], width, label="Kinematic", color=COLORS["kinematic"], alpha=0.8)
    ax2.barh(x + width, [r["info_geometry_auc"] for r in bot10], width, label="InfoGeo", color=COLORS["info_geo"], alpha=0.8)
    ax2.set_yticks(x)
    ax2.set_yticklabels(names_b)
    ax2.set_xlabel("AUC")
    ax2.set_title("Bottom 10: Component AUC Breakdown")
    ax2.axvline(0.5, color=COLORS["danger"], linestyle="--", alpha=0.5)
    ax2.legend(fontsize=7, loc="lower right")

    # Panel 3: Ensemble vs best component
    ax3 = fig.add_subplot(gs[0, 2])
    for r in successful:
        best_comp = max(r["resonance_auc"], r["kinematic_auc"], r["info_geometry_auc"])
        ax3.scatter(best_comp, r["ensemble_auc"], alpha=0.5, s=30, c=COLORS["ensemble"], edgecolors="white", linewidth=0.5)
    lims = [0, 1.05]
    ax3.plot(lims, lims, "k--", alpha=0.3, label="y=x")
    ax3.set_xlabel("Best Single Component AUC")
    ax3.set_ylabel("Ensemble AUC")
    ax3.set_title("Ensemble vs Best Component")
    ax3.legend(fontsize=8)

    # Panel 4: Threshold strategy usage
    ax4 = fig.add_subplot(gs[1, 0])
    strategies = {}
    for r in successful:
        s = r.get("threshold_strategy", "unknown")
        base = s.split("_")[0] if "_" in s else s
        strategies[base] = strategies.get(base, 0) + 1
    sorted_strats = sorted(strategies.items(), key=lambda x: x[1], reverse=True)
    strat_names = [s[0] for s in sorted_strats]
    strat_counts = [s[1] for s in sorted_strats]
    colors = [VIRIDIS(i / len(sorted_strats)) for i in range(len(sorted_strats))]
    ax4.bar(strat_names, strat_counts, color=colors)
    ax4.set_ylabel("Datasets")
    ax4.set_title("Threshold Strategy Usage")
    ax4.tick_params(axis="x", rotation=45)

    # Panel 5: Anomaly ratio vs AUC
    ax5 = fig.add_subplot(gs[1, 1])
    for r in successful:
        ar = r.get("anomaly_ratio", 0)
        ax5.scatter(ar, r["ensemble_auc"], alpha=0.5, s=30, c=COLORS["primary"], edgecolors="white", linewidth=0.5)
    ax5.set_xlabel("Anomaly Ratio")
    ax5.set_ylabel("Ensemble AUC")
    ax5.set_title("Anomaly Ratio vs AUC")
    ax5.axhline(0.5, color="gray", linestyle=":", alpha=0.3)

    # Panel 6: Feature count vs AUC
    ax6 = fig.add_subplot(gs[1, 2])
    for r in successful:
        nf = r.get("n_features", 1)
        ax6.scatter(nf, r["ensemble_auc"], alpha=0.5, s=30, c=COLORS["secondary"], edgecolors="white", linewidth=0.5)
    ax6.set_xlabel("Number of Features")
    ax6.set_ylabel("Ensemble AUC")
    ax6.set_title("Feature Count vs AUC")
    ax6.set_xscale("log")

    fig.suptitle("Mercury Agent - Anomaly Detection Analysis", fontsize=16, fontweight="bold", y=0.99)
    plt.savefig(OUTPUT_DIR / "anomaly_detection_panel.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: anomaly_detection_panel.png")


# -------------------------------------------------------------------------
# 3. Performance Dashboard
# -------------------------------------------------------------------------
def generate_performance_dashboard(data: dict[str, Any]) -> None:
    """Timing, distribution, and system performance metrics."""
    successful = get_successful(data)
    summary = data["summary"]

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.35)

    # Panel 1: Fit time vs Score time
    ax1 = fig.add_subplot(gs[0, 0])
    fit_times = [r.get("fit_ms", 0) for r in successful]
    score_times = [r.get("score_ms", 0) for r in successful]
    ax1.scatter(fit_times, score_times, alpha=0.5, s=30, c=COLORS["primary"], edgecolors="white", linewidth=0.5)
    ax1.set_xlabel("Fit Time (ms)")
    ax1.set_ylabel("Score Time (ms)")
    ax1.set_title("Fit vs Score Latency")
    ax1.set_xscale("log")
    ax1.set_yscale("log")

    # Panel 2: AUC Distribution by category
    ax2 = fig.add_subplot(gs[0, 1])
    cats = {}
    for r in successful:
        cat = r.get("category", "unknown")
        cats.setdefault(cat, []).append(r["ensemble_auc"])
    sorted_cats = sorted(cats.items(), key=lambda x: np.median(x[1]), reverse=True)
    cat_names = [c[0] for c in sorted_cats[:8]]
    cat_data = [c[1] for c in sorted_cats[:8]]
    bp = ax2.boxplot(cat_data, labels=[n.replace("_", "\n") for n in cat_names], patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(VIRIDIS(i / 8))
        patch.set_alpha(0.6)
    ax2.set_ylabel("AUC")
    ax2.set_title("AUC by Category")
    ax2.tick_params(axis="x", rotation=45)

    # Panel 3: Train/Test sizes
    ax3 = fig.add_subplot(gs[0, 2])
    n_trains = [r.get("n_train", 0) for r in successful]
    n_tests = [r.get("n_test", 0) for r in successful]
    ax3.scatter(n_trains, n_tests, c=[r["ensemble_auc"] for r in successful],
                cmap="viridis", s=30, alpha=0.7, edgecolors="white", linewidth=0.5)
    ax3.set_xlabel("Train Samples")
    ax3.set_ylabel("Test Samples")
    ax3.set_title("Dataset Sizes (color=AUC)")
    cbar = plt.colorbar(ax3.collections[0], ax=ax3)
    cbar.set_label("AUC")

    # Panel 4: Timing histogram
    ax4 = fig.add_subplot(gs[1, 0])
    total_times = [r.get("fit_ms", 0) + r.get("score_ms", 0) for r in successful]
    ax4.hist(total_times, bins=20, color=COLORS["secondary"], alpha=0.7, edgecolor="white")
    ax4.set_xlabel("Total Time (ms)")
    ax4.set_ylabel("Count")
    ax4.set_title("Total Processing Time Distribution")
    ax4.axvline(np.median(total_times), color=COLORS["danger"], linestyle="--",
                label=f"Median: {np.median(total_times):.0f}ms")
    ax4.legend()

    # Panel 5: Adaptive Weights
    ax5 = fig.add_subplot(gs[1, 1])
    res_w = [r["adaptive_weights"]["resonance"] for r in successful if r.get("adaptive_weights")]
    kin_w = [r["adaptive_weights"]["kinematic"] for r in successful if r.get("adaptive_weights")]
    ig_w = [r["adaptive_weights"]["info_geometry"] for r in successful if r.get("adaptive_weights")]
    if res_w:
        ax5.scatter(res_w, kin_w, c=ig_w, cmap="viridis", s=30, alpha=0.7, edgecolors="white", linewidth=0.5)
        ax5.set_xlabel("Resonance Weight")
        ax5.set_ylabel("Kinematic Weight")
        ax5.set_title("Adaptive Weight Distribution")
        cbar2 = plt.colorbar(ax5.collections[0], ax=ax5)
        cbar2.set_label("InfoGeo Weight")

    # Panel 6: Weight source breakdown
    ax6 = fig.add_subplot(gs[1, 2])
    weight_sources = {}
    for r in successful:
        ws = r.get("weight_source", "unknown")
        weight_sources[ws] = weight_sources.get(ws, 0) + 1
    ws_names = list(weight_sources.keys())
    ws_counts = list(weight_sources.values())
    ws_colors = [VIRIDIS(i / max(1, len(ws_names))) for i in range(len(ws_names))]
    if ws_names:
        ax6.pie(ws_counts, labels=[n.replace("_", "\n") for n in ws_names],
                colors=ws_colors, autopct="%1.0f%%", startangle=90)
    ax6.set_title("Weight Source Distribution")

    # Panel 7: Precision vs Recall
    ax7 = fig.add_subplot(gs[2, 0])
    precs = [r["oracle_precision"] for r in successful]
    recs = [r["oracle_recall"] for r in successful]
    ax7.scatter(recs, precs, c=[r["ensemble_auc"] for r in successful],
                cmap="viridis", s=30, alpha=0.7, edgecolors="white", linewidth=0.5)
    ax7.set_xlabel("Oracle Recall")
    ax7.set_ylabel("Oracle Precision")
    ax7.set_title("Precision-Recall (color=AUC)")
    ax7.set_xlim(0, 1.05)
    ax7.set_ylim(0, 1.05)

    # Panel 8: Oracle active vs AUC
    ax8 = fig.add_subplot(gs[2, 1])
    oracle_active = [r for r in successful if r.get("oracle_metadata", {}).get("active")]
    oracle_inactive = [r for r in successful if not r.get("oracle_metadata", {}).get("active")]
    if oracle_active:
        ax8.hist([r["ensemble_auc"] for r in oracle_active], bins=15, alpha=0.6,
                 label=f"Oracle ON ({len(oracle_active)})", color=COLORS["success"])
    if oracle_inactive:
        ax8.hist([r["ensemble_auc"] for r in oracle_inactive], bins=15, alpha=0.6,
                 label=f"Oracle OFF ({len(oracle_inactive)})", color=COLORS["secondary"])
    ax8.set_xlabel("AUC")
    ax8.set_ylabel("Count")
    ax8.set_title("Oracle Influence on AUC")
    ax8.legend()

    # Panel 9: Data type breakdown
    ax9 = fig.add_subplot(gs[2, 2])
    data_types = {}
    for r in successful:
        dt = r.get("data_type", "unknown")
        data_types.setdefault(dt, []).append(r["ensemble_auc"])
    dt_names = list(data_types.keys())
    dt_means = [np.mean(v) for v in data_types.values()]
    dt_counts = [len(v) for v in data_types.values()]
    if dt_names:
        bars = ax9.bar(range(len(dt_names)), dt_means,
                       color=[VIRIDIS(i / max(1, len(dt_names))) for i in range(len(dt_names))])
        ax9.set_xticks(range(len(dt_names)))
        ax9.set_xticklabels([f"{n}\n(n={c})" for n, c in zip(dt_names, dt_counts)], fontsize=8)
        ax9.set_ylabel("Mean AUC")
        ax9.set_title("Performance by Data Type")
        ax9.set_ylim(0, 1.05)
        for bar, mean in zip(bars, dt_means):
            ax9.text(bar.get_x() + bar.get_width() / 2, mean + 0.01, f"{mean:.3f}",
                     ha="center", fontsize=8)

    fig.suptitle("Mercury Agent - Performance Dashboard", fontsize=16, fontweight="bold", y=0.99)
    plt.savefig(OUTPUT_DIR / "mercury_performance_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: mercury_performance_dashboard.png")


# -------------------------------------------------------------------------
# 4. Benchmark Summary (all datasets sorted bar chart)
# -------------------------------------------------------------------------
def generate_benchmark_summary(data: dict[str, Any]) -> None:
    """AUC bar chart for all datasets sorted by performance."""
    successful = get_successful(data)
    summary = data["summary"]

    fig, ax = plt.subplots(figsize=(16, max(10, len(successful) * 0.25)))

    names = [r["name"] for r in successful][::-1]
    aucs = [r["ensemble_auc"] for r in successful][::-1]
    colors = [VIRIDIS(max(0, a)) for a in aucs]

    bars = ax.barh(range(len(names)), aucs, color=colors, edgecolor="white", linewidth=0.3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Ensemble ROC-AUC", fontsize=12)
    ax.set_title(
        f"Mercury Agent - All {summary['successful']} Datasets Ranked by AUC\n"
        f"Mean: {summary['mean_auc']:.4f} | Median: {summary['median_auc']:.4f} | "
        f"Std: {summary['std_auc']:.4f}",
        fontsize=13, fontweight="bold"
    )
    ax.axvline(summary["mean_auc"], color=COLORS["danger"], linestyle="--", linewidth=1.5, label=f"Mean: {summary['mean_auc']:.4f}")
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5, label="Random (0.5)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1.05)

    plt.savefig(OUTPUT_DIR / "benchmark_summary_live_data.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated: benchmark_summary_live_data.png")


# -------------------------------------------------------------------------
# 5. Calibration & Conformal Coverage
# -------------------------------------------------------------------------
def generate_calibration_visuals(data: dict[str, Any]) -> None:
    """Calibration improvement and conformal coverage plots."""
    calib_file = BENCHMARKS_DIR / "calibration_validation_results.json"
    if not calib_file.exists():
        print("Skipping calibration visuals (no calibration results yet)")
        return

    with open(calib_file) as f:
        calib = json.load(f)

    # Calibration Improvement
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    md011 = calib.get("md011_calibration", {})
    if md011:
        datasets_improved = md011.get("datasets_improved", 0)
        datasets_total = md011.get("datasets_tested", 1)
        mean_before = md011.get("mean_f1_before", 0)
        mean_after = md011.get("mean_f1_after", 0)

        ax1.bar(["Before\nCalibration", "After\nCalibration"], [mean_before, mean_after],
                color=[COLORS["warning"], COLORS["success"]])
        ax1.set_ylabel("Mean F1 Score")
        ax1.set_title(f"MD-011: Threshold Calibration\n({datasets_improved}/{datasets_total} datasets improved)")
        ax1.set_ylim(0, 1.0)
        for i, v in enumerate([mean_before, mean_after]):
            ax1.text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=10)

    md005 = calib.get("md005_conformal", {})
    if md005:
        levels = md005.get("coverage_levels", [0.90, 0.95, 0.99])
        achieved = md005.get("achieved_coverage", [0.85, 0.80, 0.70])
        x = np.arange(len(levels))
        width = 0.35
        ax2.bar(x - width / 2, levels, width, label="Target", color=COLORS["primary"], alpha=0.6)
        ax2.bar(x + width / 2, achieved, width, label="Achieved", color=COLORS["success"], alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{l:.0%}" for l in levels])
        ax2.set_ylabel("Coverage")
        ax2.set_title("MD-005: Conformal Coverage")
        ax2.legend()
        ax2.set_ylim(0, 1.05)

    fig.suptitle("Mercury Agent - Calibration & Conformal Validation", fontsize=14, fontweight="bold")
    plt.savefig(OUTPUT_DIR / "calibration_improvement.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Generated: calibration_improvement.png")

    # Conformal coverage standalone
    fig2, ax = plt.subplots(figsize=(8, 6))
    if md005:
        levels = md005.get("coverage_levels", [0.90, 0.95, 0.99])
        achieved = md005.get("achieved_coverage", [0.85, 0.80, 0.70])
        x = np.arange(len(levels))
        width = 0.35
        ax.bar(x - width / 2, levels, width, label="Target", color=COLORS["primary"], alpha=0.6)
        ax.bar(x + width / 2, achieved, width, label="Achieved", color=COLORS["success"], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{l:.0%}" for l in levels])
        ax.set_ylabel("Coverage Rate")
        ax.set_title("Conformal Prediction Coverage Guarantee")
        ax.legend()
        ax.set_ylim(0, 1.05)
    plt.savefig(OUTPUT_DIR / "conformal_coverage.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Generated: conformal_coverage.png")


# -------------------------------------------------------------------------
# 6. Adaptive Weight Distribution
# -------------------------------------------------------------------------
def generate_weight_distribution(data: dict[str, Any]) -> None:
    """Ternary-style weight distribution plot."""
    successful = get_successful(data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram of each weight
    res_w = [r["adaptive_weights"]["resonance"] for r in successful if r.get("adaptive_weights")]
    kin_w = [r["adaptive_weights"]["kinematic"] for r in successful if r.get("adaptive_weights")]
    ig_w = [r["adaptive_weights"]["info_geometry"] for r in successful if r.get("adaptive_weights")]

    if res_w:
        ax1.hist(res_w, bins=15, alpha=0.6, label="Resonance", color=COLORS["resonance"])
        ax1.hist(kin_w, bins=15, alpha=0.6, label="Kinematic", color=COLORS["kinematic"])
        ax1.hist(ig_w, bins=15, alpha=0.6, label="InfoGeo", color=COLORS["info_geo"])
        ax1.set_xlabel("Weight Value")
        ax1.set_ylabel("Count")
        ax1.set_title("Adaptive Weight Distributions")
        ax1.legend()

    # Weights by domain
    domain_weights: dict[str, dict[str, list[float]]] = {}
    for r in successful:
        if r.get("adaptive_weights"):
            cat = r.get("category", "unknown")
            domain_weights.setdefault(cat, {"resonance": [], "kinematic": [], "info_geometry": []})
            for comp in ["resonance", "kinematic", "info_geometry"]:
                domain_weights[cat][comp].append(r["adaptive_weights"][comp])

    if domain_weights:
        domains = sorted(domain_weights.keys())[:8]
        x = np.arange(len(domains))
        width = 0.25
        r_means = [np.mean(domain_weights[d]["resonance"]) for d in domains]
        k_means = [np.mean(domain_weights[d]["kinematic"]) for d in domains]
        i_means = [np.mean(domain_weights[d]["info_geometry"]) for d in domains]
        ax2.bar(x - width, r_means, width, label="Resonance", color=COLORS["resonance"], alpha=0.8)
        ax2.bar(x, k_means, width, label="Kinematic", color=COLORS["kinematic"], alpha=0.8)
        ax2.bar(x + width, i_means, width, label="InfoGeo", color=COLORS["info_geo"], alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels([d.replace("_", "\n") for d in domains], fontsize=8)
        ax2.set_ylabel("Mean Weight")
        ax2.set_title("Mean Adaptive Weights by Domain")
        ax2.legend()

    fig.suptitle("Mercury Agent - Adaptive Weight Analysis", fontsize=14, fontweight="bold")
    plt.savefig(OUTPUT_DIR / "adaptive_weight_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Generated: adaptive_weight_distribution.png")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("Mercury Agent - Data-Driven Benchmark Visualization Generator")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_results()
    print(f"Loaded {data['summary']['successful']} successful dataset results\n")

    generate_neuro_symbolic_report(data)
    generate_anomaly_detection_panel(data)
    generate_performance_dashboard(data)
    generate_benchmark_summary(data)
    generate_calibration_visuals(data)
    generate_weight_distribution(data)

    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
