#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmark Visualization Generator (Data-Driven).

Generates publication-quality benchmark visualizations from actual
mercury_benchmark_results.json and calibration_validation_results.json.
No synthetic data — every number displayed is measured.

Output (dark theme):
    docs/images/neuro_symbolic_benchmark_report.png   (9-panel)
    docs/images/anomaly_detection_panel.png           (6-panel)
    docs/images/mercury_performance_dashboard.png     (9-panel)
    docs/images/benchmark_summary_live_data.png       (full bar chart)
    docs/images/calibration_improvement.png           (6-panel calibration + conformal)
    docs/images/adaptive_weight_distribution.png      (6-panel weight analysis)
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

# ---------------------------------------------------------------------------
# Dark theme configuration
# ---------------------------------------------------------------------------
plt.style.use("dark_background")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "figure.dpi": 150,
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "axes.grid": True,
        "grid.color": "#21262d",
        "grid.alpha": 0.6,
        "text.color": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "#0d1117",
    }
)

VIRIDIS = plt.cm.viridis
COLORS = {
    "primary": "#58a6ff",
    "secondary": "#bc8cff",
    "success": "#3fb950",
    "warning": "#d29922",
    "danger": "#f85149",
    "ensemble": "#58a6ff",
    "resonance": "#f0883e",
    "kinematic": "#bc8cff",
    "info_geo": "#3fb950",
    "accent1": "#79c0ff",
    "accent2": "#d2a8ff",
    "grid_line": "#30363d",
    "text_muted": "#8b949e",
}

BENCHMARKS_DIR = Path(__file__).parent
OUTPUT_DIR = BENCHMARKS_DIR.parent / "docs" / "images"


def load_results() -> dict[str, Any]:
    """Load mercury_benchmark_results.json."""
    results_file = BENCHMARKS_DIR / "mercury_benchmark_results.json"
    with open(results_file) as f:
        return json.load(f)


def load_calibration() -> dict[str, Any] | None:
    """Load calibration_validation_results.json if available."""
    calib_file = BENCHMARKS_DIR / "calibration_validation_results.json"
    if not calib_file.exists():
        return None
    with open(calib_file) as f:
        return json.load(f)


def get_successful(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return successful per-dataset results sorted by AUC descending."""
    successful = [r for r in data["per_dataset"] if r.get("error") is None]
    successful.sort(key=lambda r: r.get("ensemble_auc", 0), reverse=True)
    return successful


def _style_ax(ax, title: str = "") -> None:
    """Apply consistent dark styling to an axis."""
    if title:
        ax.set_title(title, color="#e6edf3", fontweight="bold", pad=8)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")


# -------------------------------------------------------------------------
# 1. Neuro-Symbolic Benchmark Report (9-panel)
# -------------------------------------------------------------------------
def generate_neuro_symbolic_report(data: dict[str, Any]) -> None:
    """Ensemble vs component AUCs, domain breakdown, summary stats."""
    successful = get_successful(data)
    summary = data["summary"]
    comp_summary = data["component_summary"]

    fig = plt.figure(figsize=(18, 14), facecolor="#0d1117")
    gs = GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.35)

    # Panel 1: AUC Distribution Histogram
    ax1 = fig.add_subplot(gs[0, 0])
    aucs = [r["ensemble_auc"] for r in successful]
    ax1.hist(
        aucs, bins=20, color=COLORS["ensemble"], alpha=0.75, edgecolor="#0d1117", linewidth=0.5
    )
    ax1.axvline(
        summary["mean_auc"],
        color=COLORS["danger"],
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {summary['mean_auc']:.4f}",
    )
    ax1.axvline(
        summary["median_auc"],
        color=COLORS["warning"],
        linestyle="--",
        linewidth=1.5,
        label=f"Median: {summary['median_auc']:.4f}",
    )
    ax1.set_xlabel("ROC-AUC")
    ax1.set_ylabel("Count")
    _style_ax(ax1, "Ensemble AUC Distribution")
    ax1.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d")

    # Panel 2: Component AUC Comparison (box-style)
    ax2 = fig.add_subplot(gs[0, 1])
    comp_data = {
        "Resonance": [
            r["resonance_auc"]
            for r in successful
            if not np.isnan(r.get("resonance_auc", float("nan")))
        ],
        "Kinematic": [
            r["kinematic_auc"]
            for r in successful
            if not np.isnan(r.get("kinematic_auc", float("nan")))
        ],
        "InfoGeo": [
            r["info_geometry_auc"]
            for r in successful
            if not np.isnan(r.get("info_geometry_auc", float("nan")))
        ],
        "Ensemble": [r["ensemble_auc"] for r in successful],
    }
    bp = ax2.boxplot(
        comp_data.values(),
        tick_labels=comp_data.keys(),
        patch_artist=True,
        boxprops=dict(linewidth=0.5),
        whiskerprops=dict(color="#8b949e"),
        capprops=dict(color="#8b949e"),
        medianprops=dict(color=COLORS["warning"], linewidth=2),
        flierprops=dict(markeredgecolor="#8b949e", markersize=4),
    )
    box_colors = [COLORS["resonance"], COLORS["kinematic"], COLORS["info_geo"], COLORS["ensemble"]]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_ylabel("ROC-AUC")
    _style_ax(ax2, "Component vs Ensemble AUC")
    ax2.axhline(0.5, color=COLORS["text_muted"], linestyle=":", alpha=0.5)

    # Panel 3: Domain Performance
    ax3 = fig.add_subplot(gs[0, 2])
    domain_summary = data["domain_summary"]
    domains = sorted(
        domain_summary.keys(),
        key=lambda d: domain_summary[d]["stats"].get("mean_auc") or 0,
        reverse=True,
    )
    domain_names = [d.replace("_", "\n") for d in domains]
    domain_aucs = [domain_summary[d]["stats"].get("mean_auc") or 0 for d in domains]
    domain_colors = [VIRIDIS(a) for a in domain_aucs]
    bars = ax3.barh(
        domain_names, domain_aucs, color=domain_colors, edgecolor="#0d1117", linewidth=0.3
    )
    ax3.axvline(0.5, color=COLORS["text_muted"], linestyle=":", alpha=0.5)
    ax3.set_xlabel("Mean AUC")
    _style_ax(ax3, "Domain Performance")
    ax3.set_xlim(0, 1.05)
    for bar, auc in zip(bars, domain_aucs):
        ax3.text(
            min(auc + 0.01, 1.0),
            bar.get_y() + bar.get_height() / 2,
            f"{auc:.3f}",
            va="center",
            fontsize=7,
            color="#c9d1d9",
        )

    # Panel 4: Top-15 Datasets by AUC
    ax4 = fig.add_subplot(gs[1, 0])
    top15 = successful[:15]
    names = [r["name"] for r in top15]
    top_aucs = [r["ensemble_auc"] for r in top15]
    colors = [VIRIDIS(a) for a in top_aucs]
    ax4.barh(names[::-1], top_aucs[::-1], color=colors[::-1], edgecolor="#0d1117", linewidth=0.3)
    ax4.set_xlabel("AUC")
    _style_ax(ax4, "Top 15 Datasets")
    ax4.set_xlim(0.9, 1.005)

    # Panel 5: Bottom-15 Datasets
    ax5 = fig.add_subplot(gs[1, 1])
    bottom15 = successful[-15:]
    names_b = [r["name"] for r in bottom15]
    bot_aucs = [r["ensemble_auc"] for r in bottom15]
    colors_b = [VIRIDIS(max(0, a)) for a in bot_aucs]
    ax5.barh(
        names_b[::-1], bot_aucs[::-1], color=colors_b[::-1], edgecolor="#0d1117", linewidth=0.3
    )
    ax5.set_xlabel("AUC")
    _style_ax(ax5, "Bottom 15 Datasets")
    ax5.axvline(0.5, color=COLORS["danger"], linestyle="--", alpha=0.7)

    # Panel 6: F1 Distribution
    ax6 = fig.add_subplot(gs[1, 2])
    f1s = [r["oracle_f1"] for r in successful]
    ax6.hist(f1s, bins=20, color=COLORS["success"], alpha=0.75, edgecolor="#0d1117", linewidth=0.5)
    ax6.axvline(
        summary["mean_oracle_f1"],
        color=COLORS["danger"],
        linestyle="--",
        label=f"Mean: {summary['mean_oracle_f1']:.4f}",
    )
    ax6.axvline(
        summary["median_oracle_f1"],
        color=COLORS["warning"],
        linestyle="--",
        label=f"Median: {summary['median_oracle_f1']:.4f}",
    )
    ax6.set_xlabel("Oracle F1")
    ax6.set_ylabel("Count")
    _style_ax(ax6, "Oracle F1 Distribution")
    ax6.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d")

    # Panel 7: AUC vs F1 Scatter
    ax7 = fig.add_subplot(gs[2, 0])
    scatter_aucs = [r["ensemble_auc"] for r in successful]
    scatter_f1s = [r["oracle_f1"] for r in successful]
    ax7.scatter(
        scatter_aucs,
        scatter_f1s,
        c=COLORS["ensemble"],
        alpha=0.6,
        s=30,
        edgecolors="#0d1117",
        linewidth=0.5,
    )
    ax7.set_xlabel("Ensemble AUC")
    ax7.set_ylabel("Oracle F1")
    _style_ax(ax7, "AUC vs Oracle F1")
    ax7.axhline(0.5, color=COLORS["grid_line"], linestyle=":", alpha=0.3)
    ax7.axvline(0.5, color=COLORS["grid_line"], linestyle=":", alpha=0.3)

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
    bar_colors = [COLORS["resonance"], COLORS["kinematic"], COLORS["info_geo"], COLORS["ensemble"]]
    ax8.bar(
        x - width / 2,
        comp_means,
        width,
        label="Mean",
        color=bar_colors,
        alpha=0.85,
        edgecolor="#0d1117",
        linewidth=0.5,
    )
    ax8.bar(
        x + width / 2,
        comp_medians,
        width,
        label="Median",
        color=bar_colors,
        alpha=0.40,
        edgecolor="#0d1117",
        linewidth=0.5,
    )
    ax8.set_xticks(x)
    ax8.set_xticklabels(comp_names)
    ax8.set_ylabel("AUC")
    _style_ax(ax8, "Component Mean vs Median AUC")
    ax8.legend(facecolor="#161b22", edgecolor="#30363d")
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
    ax9.text(
        0.05,
        0.5,
        summary_text,
        transform=ax9.transAxes,
        fontsize=10,
        verticalalignment="center",
        fontfamily="monospace",
        color="#e6edf3",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#21262d", edgecolor="#30363d", alpha=0.9),
    )
    _style_ax(ax9, "Key Metrics")

    fig.suptitle(
        "Mercury Agent v1.7.0 — Neuro-Symbolic Benchmark Report",
        fontsize=16,
        fontweight="bold",
        color="#e6edf3",
        y=0.99,
    )
    plt.savefig(OUTPUT_DIR / "neuro_symbolic_benchmark_report.png")
    plt.close()
    print("Generated: neuro_symbolic_benchmark_report.png")


# -------------------------------------------------------------------------
# 2. Anomaly Detection Panel (6-panel)
# -------------------------------------------------------------------------
def generate_anomaly_detection_panel(data: dict[str, Any]) -> None:
    """Per-component AUC breakdown, ensemble analysis, thresholds."""
    successful = get_successful(data)

    fig = plt.figure(figsize=(18, 10), facecolor="#0d1117")
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    # Panel 1: Top-10 component breakdown
    ax1 = fig.add_subplot(gs[0, 0])
    top10 = successful[:10]
    names = [r["name"] for r in top10]
    x = np.arange(len(names))
    width = 0.2
    ax1.barh(
        x - width,
        [r["resonance_auc"] for r in top10],
        width,
        label="Resonance",
        color=COLORS["resonance"],
        alpha=0.85,
    )
    ax1.barh(
        x,
        [r["kinematic_auc"] for r in top10],
        width,
        label="Kinematic",
        color=COLORS["kinematic"],
        alpha=0.85,
    )
    ax1.barh(
        x + width,
        [r["info_geometry_auc"] for r in top10],
        width,
        label="InfoGeo",
        color=COLORS["info_geo"],
        alpha=0.85,
    )
    ax1.set_yticks(x)
    ax1.set_yticklabels(names)
    ax1.set_xlabel("AUC")
    _style_ax(ax1, "Top 10: Component AUC Breakdown")
    ax1.legend(fontsize=7, loc="lower right", facecolor="#161b22", edgecolor="#30363d")
    ax1.set_xlim(0, 1.05)

    # Panel 2: Bottom-10 component breakdown
    ax2 = fig.add_subplot(gs[0, 1])
    bot10 = successful[-10:]
    names_b = [r["name"] for r in bot10]
    x = np.arange(len(names_b))
    ax2.barh(
        x - width,
        [r["resonance_auc"] for r in bot10],
        width,
        label="Resonance",
        color=COLORS["resonance"],
        alpha=0.85,
    )
    ax2.barh(
        x,
        [r["kinematic_auc"] for r in bot10],
        width,
        label="Kinematic",
        color=COLORS["kinematic"],
        alpha=0.85,
    )
    ax2.barh(
        x + width,
        [r["info_geometry_auc"] for r in bot10],
        width,
        label="InfoGeo",
        color=COLORS["info_geo"],
        alpha=0.85,
    )
    ax2.set_yticks(x)
    ax2.set_yticklabels(names_b)
    ax2.set_xlabel("AUC")
    _style_ax(ax2, "Bottom 10: Component AUC Breakdown")
    ax2.axvline(0.5, color=COLORS["danger"], linestyle="--", alpha=0.5)
    ax2.legend(fontsize=7, loc="lower right", facecolor="#161b22", edgecolor="#30363d")

    # Panel 3: Ensemble vs best component scatter
    ax3 = fig.add_subplot(gs[0, 2])
    for r in successful:
        best_comp = max(r["resonance_auc"], r["kinematic_auc"], r["info_geometry_auc"])
        ax3.scatter(
            best_comp,
            r["ensemble_auc"],
            alpha=0.6,
            s=30,
            c=COLORS["ensemble"],
            edgecolors="#0d1117",
            linewidth=0.5,
        )
    lims = [0, 1.05]
    ax3.plot(lims, lims, color=COLORS["text_muted"], linestyle="--", alpha=0.5, label="y=x")
    ax3.set_xlabel("Best Single Component AUC")
    ax3.set_ylabel("Ensemble AUC")
    _style_ax(ax3, "Ensemble vs Best Component")
    ax3.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d")

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
    strat_colors = [VIRIDIS(i / max(1, len(sorted_strats) - 1)) for i in range(len(sorted_strats))]
    bars = ax4.bar(
        strat_names, strat_counts, color=strat_colors, edgecolor="#0d1117", linewidth=0.5
    )
    ax4.set_ylabel("Datasets")
    _style_ax(ax4, "Threshold Strategy Usage")
    ax4.tick_params(axis="x", rotation=45)
    for bar, count in zip(bars, strat_counts):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            count + 0.3,
            str(count),
            ha="center",
            fontsize=8,
            color="#c9d1d9",
        )

    # Panel 5: Anomaly ratio vs AUC
    ax5 = fig.add_subplot(gs[1, 1])
    for r in successful:
        ar = r.get("anomaly_ratio", 0)
        ax5.scatter(
            ar,
            r["ensemble_auc"],
            alpha=0.6,
            s=30,
            c=COLORS["primary"],
            edgecolors="#0d1117",
            linewidth=0.5,
        )
    ax5.set_xlabel("Anomaly Ratio")
    ax5.set_ylabel("Ensemble AUC")
    _style_ax(ax5, "Anomaly Ratio vs AUC")
    ax5.axhline(0.5, color=COLORS["grid_line"], linestyle=":", alpha=0.3)

    # Panel 6: Feature count vs AUC
    ax6 = fig.add_subplot(gs[1, 2])
    for r in successful:
        nf = r.get("n_features", 1)
        ax6.scatter(
            nf,
            r["ensemble_auc"],
            alpha=0.6,
            s=30,
            c=COLORS["secondary"],
            edgecolors="#0d1117",
            linewidth=0.5,
        )
    ax6.set_xlabel("Number of Features")
    ax6.set_ylabel("Ensemble AUC")
    _style_ax(ax6, "Feature Count vs AUC")
    ax6.set_xscale("log")

    fig.suptitle(
        "Mercury Agent — Anomaly Detection Analysis",
        fontsize=16,
        fontweight="bold",
        color="#e6edf3",
        y=0.99,
    )
    plt.savefig(OUTPUT_DIR / "anomaly_detection_panel.png")
    plt.close()
    print("Generated: anomaly_detection_panel.png")


# -------------------------------------------------------------------------
# 3. Performance Dashboard (9-panel)
# -------------------------------------------------------------------------
def generate_performance_dashboard(data: dict[str, Any]) -> None:
    """Timing, distribution, and system performance metrics."""
    successful = get_successful(data)
    summary = data["summary"]

    fig = plt.figure(figsize=(18, 12), facecolor="#0d1117")
    gs = GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.35)

    # Panel 1: Fit time vs Score time
    ax1 = fig.add_subplot(gs[0, 0])
    fit_times = [r.get("fit_ms", 0) for r in successful]
    score_times = [r.get("score_ms", 0) for r in successful]
    ax1.scatter(
        fit_times,
        score_times,
        alpha=0.6,
        s=30,
        c=COLORS["primary"],
        edgecolors="#0d1117",
        linewidth=0.5,
    )
    ax1.set_xlabel("Fit Time (ms)")
    ax1.set_ylabel("Score Time (ms)")
    _style_ax(ax1, "Fit vs Score Latency")
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
    bp = ax2.boxplot(
        cat_data,
        tick_labels=[n.replace("_", "\n") for n in cat_names],
        patch_artist=True,
        whiskerprops=dict(color="#8b949e"),
        capprops=dict(color="#8b949e"),
        medianprops=dict(color=COLORS["warning"], linewidth=2),
        flierprops=dict(markeredgecolor="#8b949e", markersize=4),
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(VIRIDIS(i / max(1, len(cat_names) - 1)))
        patch.set_alpha(0.7)
    ax2.set_ylabel("AUC")
    _style_ax(ax2, "AUC by Category")
    ax2.tick_params(axis="x", rotation=45)

    # Panel 3: Train/Test sizes colored by AUC
    ax3 = fig.add_subplot(gs[0, 2])
    n_trains = [r.get("n_train", 0) for r in successful]
    n_tests = [r.get("n_test", 0) for r in successful]
    sc = ax3.scatter(
        n_trains,
        n_tests,
        c=[r["ensemble_auc"] for r in successful],
        cmap="viridis",
        s=30,
        alpha=0.7,
        edgecolors="#0d1117",
        linewidth=0.5,
    )
    ax3.set_xlabel("Train Samples")
    ax3.set_ylabel("Test Samples")
    _style_ax(ax3, "Dataset Sizes (color=AUC)")
    cbar = plt.colorbar(sc, ax=ax3)
    cbar.set_label("AUC", color="#c9d1d9")
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#8b949e")

    # Panel 4: Timing histogram
    ax4 = fig.add_subplot(gs[1, 0])
    total_times = [r.get("fit_ms", 0) + r.get("score_ms", 0) for r in successful]
    ax4.hist(
        total_times,
        bins=20,
        color=COLORS["secondary"],
        alpha=0.75,
        edgecolor="#0d1117",
        linewidth=0.5,
    )
    ax4.set_xlabel("Total Time (ms)")
    ax4.set_ylabel("Count")
    _style_ax(ax4, "Total Processing Time Distribution")
    ax4.axvline(
        np.median(total_times),
        color=COLORS["danger"],
        linestyle="--",
        label=f"Median: {np.median(total_times):.0f}ms",
    )
    ax4.legend(facecolor="#161b22", edgecolor="#30363d")

    # Panel 5: Weight source breakdown (pie)
    ax5 = fig.add_subplot(gs[1, 1])
    weight_sources = {}
    for r in successful:
        ws = r.get("weight_source", "unknown")
        weight_sources[ws] = weight_sources.get(ws, 0) + 1
    ws_names = list(weight_sources.keys())
    ws_counts = list(weight_sources.values())
    ws_colors = [VIRIDIS(i / max(1, len(ws_names) - 1)) for i in range(len(ws_names))]
    if ws_names:
        wedges, texts, autotexts = ax5.pie(
            ws_counts,
            labels=[n.replace("_", "\n") for n in ws_names],
            colors=ws_colors,
            autopct="%1.0f%%",
            startangle=90,
            textprops={"color": "#c9d1d9", "fontsize": 8},
        )
        for at in autotexts:
            at.set_color("#e6edf3")
            at.set_fontweight("bold")
    _style_ax(ax5, "Weight Source Distribution")

    # Panel 6: Oracle active vs AUC
    ax6 = fig.add_subplot(gs[1, 2])
    oracle_active = [r for r in successful if r.get("oracle_metadata", {}).get("active")]
    oracle_inactive = [r for r in successful if not r.get("oracle_metadata", {}).get("active")]
    if oracle_active:
        ax6.hist(
            [r["ensemble_auc"] for r in oracle_active],
            bins=15,
            alpha=0.65,
            label=f"Oracle ON ({len(oracle_active)})",
            color=COLORS["success"],
        )
    if oracle_inactive:
        ax6.hist(
            [r["ensemble_auc"] for r in oracle_inactive],
            bins=15,
            alpha=0.65,
            label=f"Oracle OFF ({len(oracle_inactive)})",
            color=COLORS["secondary"],
        )
    ax6.set_xlabel("AUC")
    ax6.set_ylabel("Count")
    _style_ax(ax6, "Oracle Influence on AUC")
    ax6.legend(facecolor="#161b22", edgecolor="#30363d")

    # Panel 7: Precision vs Recall
    ax7 = fig.add_subplot(gs[2, 0])
    precs = [r["oracle_precision"] for r in successful]
    recs = [r["oracle_recall"] for r in successful]
    sc2 = ax7.scatter(
        recs,
        precs,
        c=[r["ensemble_auc"] for r in successful],
        cmap="viridis",
        s=30,
        alpha=0.7,
        edgecolors="#0d1117",
        linewidth=0.5,
    )
    ax7.set_xlabel("Oracle Recall")
    ax7.set_ylabel("Oracle Precision")
    _style_ax(ax7, "Precision-Recall (color=AUC)")
    ax7.set_xlim(0, 1.05)
    ax7.set_ylim(0, 1.05)
    cbar2 = plt.colorbar(sc2, ax=ax7)
    cbar2.set_label("AUC", color="#c9d1d9")
    cbar2.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(plt.getp(cbar2.ax.axes, "yticklabels"), color="#8b949e")

    # Panel 8: Data type breakdown
    ax8 = fig.add_subplot(gs[2, 1])
    data_types = {}
    for r in successful:
        dt = r.get("data_type", "unknown")
        data_types.setdefault(dt, []).append(r["ensemble_auc"])
    dt_names = list(data_types.keys())
    dt_means = [np.mean(v) for v in data_types.values()]
    dt_counts = [len(v) for v in data_types.values()]
    if dt_names:
        dt_colors = [VIRIDIS(i / max(1, len(dt_names) - 1)) for i in range(len(dt_names))]
        bars = ax8.bar(
            range(len(dt_names)), dt_means, color=dt_colors, edgecolor="#0d1117", linewidth=0.5
        )
        ax8.set_xticks(range(len(dt_names)))
        ax8.set_xticklabels([f"{n}\n(n={c})" for n, c in zip(dt_names, dt_counts)], fontsize=8)
        ax8.set_ylabel("Mean AUC")
        _style_ax(ax8, "Performance by Data Type")
        ax8.set_ylim(0, 1.05)
        for bar, mean in zip(bars, dt_means):
            ax8.text(
                bar.get_x() + bar.get_width() / 2,
                mean + 0.02,
                f"{mean:.3f}",
                ha="center",
                fontsize=8,
                color="#c9d1d9",
            )

    # Panel 9: Summary stats
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")
    perf_text = (
        f"PERFORMANCE SUMMARY\n"
        f"{'─' * 35}\n\n"
        f"Fit Time (median):   {np.median(fit_times):.0f} ms\n"
        f"Score Time (median): {np.median(score_times):.0f} ms\n"
        f"Total (median):      {np.median(total_times):.0f} ms\n\n"
        f"Oracle Active:       {len(oracle_active)}/{len(successful)}\n"
        f"Categories:          {len(cats)}\n"
        f"Data Types:          {len(data_types)}\n\n"
        f"Mean AUC:            {summary['mean_auc']:.4f}\n"
        f"Mean Oracle F1:      {summary['mean_oracle_f1']:.4f}\n"
        f"{'─' * 35}\n"
        f"All timings measured on benchmark run"
    )
    ax9.text(
        0.05,
        0.5,
        perf_text,
        transform=ax9.transAxes,
        fontsize=10,
        verticalalignment="center",
        fontfamily="monospace",
        color="#e6edf3",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#21262d", edgecolor="#30363d", alpha=0.9),
    )
    _style_ax(ax9, "Runtime Statistics")

    fig.suptitle(
        "Mercury Agent — Performance Dashboard",
        fontsize=16,
        fontweight="bold",
        color="#e6edf3",
        y=0.99,
    )
    plt.savefig(OUTPUT_DIR / "mercury_performance_dashboard.png")
    plt.close()
    print("Generated: mercury_performance_dashboard.png")


# -------------------------------------------------------------------------
# 4. Benchmark Summary (all datasets sorted bar chart)
# -------------------------------------------------------------------------
def generate_benchmark_summary(data: dict[str, Any]) -> None:
    """AUC bar chart for all datasets sorted by performance."""
    successful = get_successful(data)
    summary = data["summary"]

    fig, ax = plt.subplots(figsize=(16, max(10, len(successful) * 0.25)), facecolor="#0d1117")

    names = [r["name"] for r in successful][::-1]
    aucs = [r["ensemble_auc"] for r in successful][::-1]
    colors = [VIRIDIS(max(0, a)) for a in aucs]

    ax.barh(range(len(names)), aucs, color=colors, edgecolor="#0d1117", linewidth=0.3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Ensemble ROC-AUC", fontsize=12)
    _style_ax(ax)
    ax.set_title(
        f"Mercury Agent — All {summary['successful']} Datasets Ranked by AUC\n"
        f"Mean: {summary['mean_auc']:.4f} | Median: {summary['median_auc']:.4f} | "
        f"Std: {summary['std_auc']:.4f}",
        fontsize=13,
        fontweight="bold",
        color="#e6edf3",
    )
    ax.axvline(
        summary["mean_auc"],
        color=COLORS["danger"],
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {summary['mean_auc']:.4f}",
    )
    ax.axvline(0.5, color=COLORS["text_muted"], linestyle=":", alpha=0.5, label="Random (0.5)")
    ax.legend(loc="lower right", facecolor="#161b22", edgecolor="#30363d")
    ax.set_xlim(0, 1.05)

    plt.savefig(OUTPUT_DIR / "benchmark_summary_live_data.png")
    plt.close()
    print("Generated: benchmark_summary_live_data.png")


# -------------------------------------------------------------------------
# 5. Calibration & Conformal Coverage (6-panel — consolidated)
# -------------------------------------------------------------------------
def generate_calibration_visuals(data: dict[str, Any]) -> None:
    """Calibration improvement, conformal coverage, and weight CV — all in one rich image."""
    calib = load_calibration()
    if calib is None:
        print("Skipping calibration visuals (no calibration results)")
        return

    summary = calib.get("summary", {})
    md011 = summary.get("md_011", {})
    md005_score = summary.get("md_005_score_based", {})
    md005_acc = summary.get("md_005_accuracy_based", {})
    md003 = summary.get("md_003", {})
    md003_cv = summary.get("md_003_cv", {})
    results = calib.get("results", [])

    fig = plt.figure(figsize=(18, 10), facecolor="#0d1117")
    gs = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    # Panel 1: MD-011 Calibration Before vs After
    ax1 = fig.add_subplot(gs[0, 0])
    if md011:
        mean_before = md011.get("mean_uncalibrated_f1", 0)
        mean_after = md011.get("mean_calibrated_f1", 0)
        improved = md011.get("calibration_improved", 0)
        total_tested = (
            improved + md011.get("calibration_same", 0) + md011.get("calibration_degraded", 0)
        )

        bars = ax1.bar(
            ["Uncalibrated", "Calibrated"],
            [mean_before, mean_after],
            color=[COLORS["warning"], COLORS["success"]],
            edgecolor="#0d1117",
            linewidth=0.5,
            width=0.5,
        )
        for bar, val in zip(bars, [mean_before, mean_after]):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.015,
                f"{val:.4f}",
                ha="center",
                fontsize=11,
                fontweight="bold",
                color="#e6edf3",
            )
        delta = mean_after - mean_before
        ax1.text(
            0.5,
            0.92,
            f"+{delta:.4f} improvement",
            transform=ax1.transAxes,
            ha="center",
            fontsize=10,
            color=COLORS["success"],
        )
        ax1.set_ylabel("Mean F1 Score")
        _style_ax(
            ax1, f"MD-011: Threshold Calibration\n({improved}/{total_tested} datasets improved)"
        )
        ax1.set_ylim(0, max(mean_before, mean_after) * 1.3)
    else:
        ax1.text(0.5, 0.5, "No MD-011 data", transform=ax1.transAxes, ha="center", color="#8b949e")
        _style_ax(ax1, "MD-011: Threshold Calibration")

    # Panel 2: Per-dataset calibration delta F1 (waterfall chart)
    ax2 = fig.add_subplot(gs[0, 1])
    if results:
        deltas = []
        ds_names = []
        for r in results:
            if r.get("error") is None and "md_011" in r:
                d = r["md_011"]
                delta = d.get("calibrated_f1", 0) - d.get("uncalibrated_f1", 0)
                deltas.append(delta)
                ds_names.append(r.get("dataset", "?")[:15])
        if deltas:
            # Sort by delta
            pairs = sorted(zip(deltas, ds_names), reverse=True)
            deltas_sorted = [p[0] for p in pairs]
            names_sorted = [p[1] for p in pairs]
            bar_colors = [COLORS["success"] if d >= 0 else COLORS["danger"] for d in deltas_sorted]
            ax2.barh(
                range(len(deltas_sorted)),
                deltas_sorted,
                color=bar_colors,
                edgecolor="#0d1117",
                linewidth=0.3,
            )
            ax2.set_yticks(range(len(names_sorted)))
            ax2.set_yticklabels(names_sorted, fontsize=6)
            ax2.axvline(0, color=COLORS["text_muted"], linewidth=0.8)
            ax2.set_xlabel("ΔF1 (calibrated − uncalibrated)")
            _style_ax(ax2, "Per-Dataset Calibration Impact")
        else:
            ax2.text(
                0.5,
                0.5,
                "No per-dataset data",
                transform=ax2.transAxes,
                ha="center",
                color="#8b949e",
            )
            _style_ax(ax2, "Per-Dataset Calibration Impact")
    else:
        ax2.text(0.5, 0.5, "No results data", transform=ax2.transAxes, ha="center", color="#8b949e")
        _style_ax(ax2, "Per-Dataset Calibration Impact")

    # Panel 3: MD-005 Conformal Coverage (score-based) — grouped bar
    ax3 = fig.add_subplot(gs[0, 2])
    if md005_score:
        levels = ["90%", "95%", "99%"]
        level_keys = ["0.9", "0.95", "0.99"]
        methods = ["split", "cross", "normal"]
        method_labels = ["Split", "Cross-Val", "Normal"]
        method_colors = [COLORS["primary"], COLORS["success"], COLORS["secondary"]]
        x = np.arange(len(levels))
        total_w = 0.7
        w = total_w / len(methods)
        for i, (method, mlabel, mcolor) in enumerate(zip(methods, method_labels, method_colors)):
            pcts = []
            for lk in level_keys:
                level_data = md005_score.get(lk, {})
                pcts.append(level_data.get(f"{method}_pct", 0))
            offset = (i - len(methods) / 2 + 0.5) * w
            bars = ax3.bar(
                x + offset,
                pcts,
                w,
                label=mlabel,
                color=mcolor,
                alpha=0.8,
                edgecolor="#0d1117",
                linewidth=0.5,
            )
        ax3.set_xticks(x)
        ax3.set_xticklabels(levels)
        ax3.set_xlabel("Coverage Level")
        ax3.set_ylabel("% Datasets Meeting Guarantee")
        _style_ax(ax3, "MD-005: Conformal Coverage (Score-Based)")
        ax3.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d")
        ax3.set_ylim(0, 100)
    else:
        ax3.text(
            0.5, 0.5, "No MD-005 score data", transform=ax3.transAxes, ha="center", color="#8b949e"
        )
        _style_ax(ax3, "MD-005: Conformal Coverage (Score-Based)")

    # Panel 4: MD-005 Accuracy-based coverage
    ax4 = fig.add_subplot(gs[1, 0])
    if md005_acc:
        levels_a = ["90%", "95%", "99%"]
        level_keys_a = ["0.9", "0.95", "0.99"]
        meets = []
        totals = []
        pcts = []
        for lk in level_keys_a:
            ld = md005_acc.get(lk, {})
            meets.append(ld.get("meets_guarantee", 0))
            totals.append(ld.get("total", 1))
            pcts.append(ld.get("pct", 0))

        x = np.arange(len(levels_a))
        bars = ax4.bar(
            x,
            pcts,
            color=[COLORS["primary"], COLORS["success"], COLORS["secondary"]],
            edgecolor="#0d1117",
            linewidth=0.5,
            width=0.5,
        )
        for bar, pct, m, t in zip(bars, pcts, meets, totals):
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                pct + 1.5,
                f"{m}/{t}\n({pct:.1f}%)",
                ha="center",
                fontsize=9,
                color="#c9d1d9",
            )
        ax4.set_xticks(x)
        ax4.set_xticklabels(levels_a)
        ax4.set_xlabel("Coverage Level")
        ax4.set_ylabel("% Datasets")
        _style_ax(ax4, "MD-005: Accuracy-Based Coverage")
        ax4.set_ylim(0, max(pcts) * 1.35 if pcts else 100)
    else:
        ax4.text(
            0.5,
            0.5,
            "No MD-005 accuracy data",
            transform=ax4.transAxes,
            ha="center",
            color="#8b949e",
        )
        _style_ax(ax4, "MD-005: Accuracy-Based Coverage")

    # Panel 5: MD-003 Weight Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    if md003 and md003.get("weight_distribution"):
        wd = md003["weight_distribution"]
        comp_names = ["Resonance", "Kinematic", "InfoGeo"]
        comp_keys = ["resonance", "kinematic", "infogeo"]
        comp_colors = [COLORS["resonance"], COLORS["kinematic"], COLORS["info_geo"]]
        means = [wd[k]["mean"] for k in comp_keys]
        stds = [wd[k]["std"] for k in comp_keys]

        x = np.arange(len(comp_names))
        bars = ax5.bar(
            x,
            means,
            yerr=stds,
            capsize=5,
            color=comp_colors,
            edgecolor="#0d1117",
            linewidth=0.5,
            width=0.5,
            alpha=0.85,
            error_kw={"ecolor": "#c9d1d9", "capthick": 1.5},
        )
        for bar, mean, std in zip(bars, means, stds):
            ax5.text(
                bar.get_x() + bar.get_width() / 2,
                mean + std + 0.02,
                f"{mean:.3f}±{std:.3f}",
                ha="center",
                fontsize=8,
                color="#c9d1d9",
            )
        ax5.set_xticks(x)
        ax5.set_xticklabels(comp_names)
        ax5.set_ylabel("Weight Value")
        _style_ax(
            ax5,
            f"MD-003: Adaptive Weight Distribution\n(n={md003.get('n_datasets_with_adaptive_weights', '?')})",
        )
        ax5.set_ylim(0, 1.0)
    else:
        ax5.text(0.5, 0.5, "No MD-003 data", transform=ax5.transAxes, ha="center", color="#8b949e")
        _style_ax(ax5, "MD-003: Adaptive Weight Distribution")

    # Panel 6: MD-003 CV summary
    ax6 = fig.add_subplot(gs[1, 2])
    if md003_cv:
        ax6.axis("off")
        cv_text = (
            f"MD-003 CROSS-VALIDATION RESULTS\n"
            f"{'─' * 40}\n\n"
            f"Strategy:              {md003_cv.get('strategy', 'N/A')}\n"
            f"Datasets Tested:       {md003_cv.get('n_datasets', 'N/A')}\n\n"
            f"Δ(optimal vs default): {md003_cv.get('mean_delta_optimal_vs_default', 0):+.4f}\n"
            f"Δ(optimal vs adaptive):{md003_cv.get('mean_delta_optimal_vs_adaptive', 0):+.4f}\n\n"
            f"Default Validated:     {md003_cv.get('default_validated_count', '?')}"
            f"/{md003_cv.get('n_datasets', '?')}"
            f" ({md003_cv.get('default_validated_pct', 0):.1f}%)\n"
            f"Adaptive Validated:    {md003_cv.get('adaptive_validated_count', '?')}"
            f"/{md003_cv.get('n_datasets', '?')}"
            f" ({md003_cv.get('adaptive_validated_pct', 0):.1f}%)\n\n"
            f"{'─' * 40}\n"
            f"Default validated = optimal ≤ default F1\n"
            f"Adaptive validated = optimal ≤ adaptive F1"
        )
        ax6.text(
            0.05,
            0.5,
            cv_text,
            transform=ax6.transAxes,
            fontsize=9,
            verticalalignment="center",
            fontfamily="monospace",
            color="#e6edf3",
            bbox=dict(
                boxstyle="round,pad=0.8", facecolor="#21262d", edgecolor="#30363d", alpha=0.9
            ),
        )
        _style_ax(ax6, "MD-003: Cross-Validation Summary")
    else:
        ax6.axis("off")
        ax6.text(0.5, 0.5, "No CV data", transform=ax6.transAxes, ha="center", color="#8b949e")
        _style_ax(ax6, "MD-003: Cross-Validation Summary")

    fig.suptitle(
        "Mercury Agent — Calibration & Conformal Validation",
        fontsize=16,
        fontweight="bold",
        color="#e6edf3",
        y=0.99,
    )
    plt.savefig(OUTPUT_DIR / "calibration_improvement.png")
    plt.close()
    print("Generated: calibration_improvement.png")


# -------------------------------------------------------------------------
# 6. Adaptive Weight Distribution (6-panel)
# -------------------------------------------------------------------------
def generate_weight_distribution(data: dict[str, Any]) -> None:
    """Detailed adaptive weight analysis across datasets and domains."""
    successful = get_successful(data)

    fig = plt.figure(figsize=(18, 10), facecolor="#0d1117")
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    # Collect weight data
    res_w = [r["adaptive_weights"]["resonance"] for r in successful if r.get("adaptive_weights")]
    kin_w = [r["adaptive_weights"]["kinematic"] for r in successful if r.get("adaptive_weights")]
    ig_w = [r["adaptive_weights"]["info_geometry"] for r in successful if r.get("adaptive_weights")]

    # Panel 1: Weight histograms (overlaid)
    ax1 = fig.add_subplot(gs[0, 0])
    if res_w:
        ax1.hist(
            res_w,
            bins=15,
            alpha=0.6,
            label=f"Resonance (μ={np.mean(res_w):.3f})",
            color=COLORS["resonance"],
        )
        ax1.hist(
            kin_w,
            bins=15,
            alpha=0.6,
            label=f"Kinematic (μ={np.mean(kin_w):.3f})",
            color=COLORS["kinematic"],
        )
        ax1.hist(
            ig_w,
            bins=15,
            alpha=0.6,
            label=f"InfoGeo (μ={np.mean(ig_w):.3f})",
            color=COLORS["info_geo"],
        )
        ax1.set_xlabel("Weight Value")
        ax1.set_ylabel("Count")
        ax1.legend(fontsize=7, facecolor="#161b22", edgecolor="#30363d")
    _style_ax(ax1, "Adaptive Weight Distributions")

    # Panel 2: Weights by domain (grouped bar)
    ax2 = fig.add_subplot(gs[0, 1])
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
        ax2.bar(x - width, r_means, width, label="Resonance", color=COLORS["resonance"], alpha=0.85)
        ax2.bar(x, k_means, width, label="Kinematic", color=COLORS["kinematic"], alpha=0.85)
        ax2.bar(x + width, i_means, width, label="InfoGeo", color=COLORS["info_geo"], alpha=0.85)
        ax2.set_xticks(x)
        ax2.set_xticklabels([d.replace("_", "\n") for d in domains], fontsize=8)
        ax2.set_ylabel("Mean Weight")
        ax2.legend(fontsize=7, facecolor="#161b22", edgecolor="#30363d")
    _style_ax(ax2, "Mean Adaptive Weights by Domain")

    # Panel 3: Resonance vs Kinematic scatter (colored by InfoGeo)
    ax3 = fig.add_subplot(gs[0, 2])
    if res_w:
        sc = ax3.scatter(
            res_w,
            kin_w,
            c=ig_w,
            cmap="viridis",
            s=30,
            alpha=0.7,
            edgecolors="#0d1117",
            linewidth=0.5,
        )
        ax3.set_xlabel("Resonance Weight")
        ax3.set_ylabel("Kinematic Weight")
        cbar = plt.colorbar(sc, ax=ax3)
        cbar.set_label("InfoGeo Weight", color="#c9d1d9")
        cbar.ax.yaxis.set_tick_params(color="#8b949e")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#8b949e")
    _style_ax(ax3, "Weight Space (3-Component)")

    # Panel 4: Weight vs AUC correlation
    ax4 = fig.add_subplot(gs[1, 0])
    if res_w:
        w_aucs = [r["ensemble_auc"] for r in successful if r.get("adaptive_weights")]
        ax4.scatter(res_w, w_aucs, alpha=0.5, s=25, c=COLORS["resonance"], label="Resonance")
        ax4.scatter(kin_w, w_aucs, alpha=0.5, s=25, c=COLORS["kinematic"], label="Kinematic")
        ax4.scatter(ig_w, w_aucs, alpha=0.5, s=25, c=COLORS["info_geo"], label="InfoGeo")
        ax4.set_xlabel("Component Weight")
        ax4.set_ylabel("Ensemble AUC")
        ax4.legend(fontsize=7, facecolor="#161b22", edgecolor="#30363d")
    _style_ax(ax4, "Weight vs AUC Correlation")

    # Panel 5: Stacked weight per dataset (top 20)
    ax5 = fig.add_subplot(gs[1, 1])
    weighted = [r for r in successful if r.get("adaptive_weights")][:20]
    if weighted:
        w_names = [r["name"][:15] for r in weighted]
        w_res = [r["adaptive_weights"]["resonance"] for r in weighted]
        w_kin = [r["adaptive_weights"]["kinematic"] for r in weighted]
        w_ig = [r["adaptive_weights"]["info_geometry"] for r in weighted]
        y_pos = np.arange(len(w_names))
        ax5.barh(y_pos, w_res, color=COLORS["resonance"], alpha=0.85, label="Resonance")
        ax5.barh(y_pos, w_kin, left=w_res, color=COLORS["kinematic"], alpha=0.85, label="Kinematic")
        left2 = [a + b for a, b in zip(w_res, w_kin)]
        ax5.barh(y_pos, w_ig, left=left2, color=COLORS["info_geo"], alpha=0.85, label="InfoGeo")
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(w_names, fontsize=7)
        ax5.set_xlabel("Weight (stacked)")
        ax5.legend(fontsize=7, loc="lower right", facecolor="#161b22", edgecolor="#30363d")
    _style_ax(ax5, "Top 20: Stacked Weights by Dataset")

    # Panel 6: Weight statistics summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    if res_w:
        weight_text = (
            f"ADAPTIVE WEIGHT STATISTICS\n"
            f"{'─' * 38}\n\n"
            f"{'Component':<14} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7}\n"
            f"{'─' * 38}\n"
            f"{'Resonance':<14} {np.mean(res_w):>7.3f} {np.std(res_w):>7.3f} "
            f"{np.min(res_w):>7.3f} {np.max(res_w):>7.3f}\n"
            f"{'Kinematic':<14} {np.mean(kin_w):>7.3f} {np.std(kin_w):>7.3f} "
            f"{np.min(kin_w):>7.3f} {np.max(kin_w):>7.3f}\n"
            f"{'InfoGeo':<14} {np.mean(ig_w):>7.3f} {np.std(ig_w):>7.3f} "
            f"{np.min(ig_w):>7.3f} {np.max(ig_w):>7.3f}\n\n"
            f"{'─' * 38}\n"
            f"Datasets with adaptive weights: {len(res_w)}\n"
            f"Default weights: R=0.40 K=0.30 IG=0.30\n"
            f"Strategy: statistical_adaptive_weights"
        )
        ax6.text(
            0.05,
            0.5,
            weight_text,
            transform=ax6.transAxes,
            fontsize=9,
            verticalalignment="center",
            fontfamily="monospace",
            color="#e6edf3",
            bbox=dict(
                boxstyle="round,pad=0.8", facecolor="#21262d", edgecolor="#30363d", alpha=0.9
            ),
        )
    _style_ax(ax6, "Weight Summary")

    fig.suptitle(
        "Mercury Agent — Adaptive Weight Analysis",
        fontsize=16,
        fontweight="bold",
        color="#e6edf3",
        y=0.99,
    )
    plt.savefig(OUTPUT_DIR / "adaptive_weight_distribution.png")
    plt.close()
    print("Generated: adaptive_weight_distribution.png")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("Mercury Agent — Data-Driven Benchmark Visualization Generator")
    print("Dark Theme | All data from measured benchmark results")
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
