#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate docs/images/ dashboard PNGs from measured benchmark data.

Reads benchmarks/mercury_benchmark_results.json as the sole data source.
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
RESULTS_PATH = Path(__file__).parent.parent / "benchmarks" / "mercury_benchmark_results.json"
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


def _fmt(value: object, decimals: int = 4) -> str:
    """Format a numeric value to at most *decimals* places, or 'N/A'."""
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value) if value is not None else "N/A"


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
            "font.size": 11,
            "savefig.dpi": 200,
            "savefig.facecolor": COLORS["bg"],
        }
    )


def _load_data() -> dict:
    if not RESULTS_PATH.exists():
        print(
            f"ERROR: {RESULTS_PATH} not found.\n"
            "Run 'python benchmarks/mercury_benchmark.py' first to generate measured results."
        )
        sys.exit(1)
    with open(RESULTS_PATH) as f:
        return json.load(f)


def _stamp(fig: plt.Figure) -> None:
    """Add source file and timestamp label to the figure."""
    fig.text(
        0.99,
        0.005,
        f"Source: mercury_benchmark_results.json | {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
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

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(
        "Mercury Agent - Ensemble Benchmark Report", fontsize=18, fontweight="bold", y=0.97
    )

    # Top-left: per-component AUC comparison
    ax = axes[0, 0]
    comp_labels = []
    comp_means = []
    comp_stds = []
    label_map = {"resonance": "Resonance", "kinematic": "Kinematic", "info_geometry": "InfoGeo"}
    for name in ["resonance", "kinematic", "info_geometry"]:
        if name in comp:
            comp_labels.append(label_map[name])
            comp_means.append(comp[name]["mean_auc"])
            comp_stds.append(comp[name]["std_auc"])
    if comp.get("resonance") is not None:
        comp_labels.append("Ensemble")
        comp_means.append(summary.get("mean_auc", 0))
        comp_stds.append(summary.get("std_auc", 0))

    if comp_labels:
        colors = [COLORS["secondary"], COLORS["accent"], COLORS["bad"], COLORS["primary"]]
        bars = ax.bar(
            comp_labels, comp_means, yerr=comp_stds, color=colors[: len(comp_labels)], capsize=5
        )
        for bar, val in zip(bars, comp_means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.4f}",
                ha="center",
                fontsize=10,
            )
        ax.set_ylabel("Mean AUC")
        ax.set_title("Component vs Ensemble AUC")
        ax.set_ylim(0, 1.10)
        ax.tick_params(axis="x", rotation=15)
    else:
        ax.text(0.5, 0.5, "Not measured", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Component vs Ensemble AUC")

    # Top-right: AUC distribution histogram
    ax = axes[0, 1]
    aucs = [r["ensemble_auc"] for r in per_ds if not np.isnan(r.get("ensemble_auc", float("nan")))]
    if aucs:
        ax.hist(aucs, bins=20, color=COLORS["primary"], alpha=0.8, edgecolor=COLORS["bg"])
        med = np.median(aucs)
        ax.axvline(
            med,
            color=COLORS["accent"],
            linestyle="--",
            linewidth=2,
            label=f"median = {med:.4f}",
        )
        ax.set_xlabel("AUC")
        ax.set_ylabel("Count")
        ax.set_title("AUC Distribution Across Datasets")
        ax.legend(fontsize=10, loc="upper left")
    else:
        ax.text(0.5, 0.5, "Not measured", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("AUC Distribution")

    # Bottom-left: Top/Bottom datasets by AUC
    ax = axes[1, 0]
    sorted_ds = sorted(per_ds, key=lambda r: r.get("ensemble_auc", 0), reverse=True)
    if len(sorted_ds) >= 6:
        show = sorted_ds[:5] + sorted_ds[-5:]
        names = [r["name"] for r in show]
        vals = [r["ensemble_auc"] for r in show]
        colors_bar = [
            COLORS["good"] if v >= 0.7 else COLORS["warn"] if v >= 0.5 else COLORS["bad"]
            for v in vals
        ]
        bars = ax.barh(names[::-1], vals[::-1], color=colors_bar[::-1], height=0.6)
        for bar, val in zip(bars, vals[::-1]):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=8,
            )
        ax.set_xlabel("AUC")
        ax.set_title("Top 5 / Bottom 5 Datasets")
        ax.set_xlim(0, 1.15)
    else:
        ax.text(0.5, 0.5, "Not enough datasets", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Top/Bottom Datasets")

    # Bottom-right: summary stats text
    ax = axes[1, 1]
    ax.axis("off")
    meta = data.get("metadata", {})
    detector_name = meta.get("detector", "N/A")
    lines = [
        f"Detector: {detector_name}",
        f"Weights: R={meta.get('ensemble_weights', {}).get('resonance', '?')}, "
        f"K={meta.get('ensemble_weights', {}).get('kinematic', '?')}, "
        f"IG={meta.get('ensemble_weights', {}).get('info_geometry', '?')}",
        "",
        f"Datasets tested:  {summary.get('total_datasets', 0)}",
        f"Successful:       {summary.get('successful', 0)}",
        f"Failed:           {summary.get('failed', 0)}",
        "",
        f"Mean AUC:         {_fmt(summary.get('mean_auc'))}",
        f"Median AUC:       {_fmt(summary.get('median_auc'))}",
        f"Mean Oracle F1:   {_fmt(summary.get('mean_oracle_f1'))}",
        "",
        f"Git: {meta.get('git_commit', 'N/A')[:12]}",
        f"Generated: {meta.get('timestamp', 'N/A')[:19]}",
    ]
    ax.text(
        0.05,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        family="monospace",
    )
    ax.set_title("Summary")

    _stamp(fig)
    plt.subplots_adjust(wspace=0.30, hspace=0.35, left=0.07, right=0.97, top=0.93, bottom=0.05)
    fig.savefig(OUTPUT_DIR / "neuro_symbolic_benchmark_report.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved neuro_symbolic_benchmark_report.png")


# ---------------------------------------------------------------------------
# Chart 2: Anomaly Detection Panel
# ---------------------------------------------------------------------------
def generate_anomaly_detection_panel(data: dict) -> None:
    per_ds = [r for r in data.get("per_dataset", []) if r.get("error") is None]

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle("Mercury Agent - Anomaly Detection Panel", fontsize=18, fontweight="bold", y=0.97)

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
            ax.text(v + 0.01, i, f"{v:.4f}", va="center", fontsize=9)
        ax.set_xlabel("Mean AUC")
        ax.set_title("AUC by Dataset Category")
        ax.set_xlim(0, 1.10)
    else:
        ax.text(0.5, 0.5, "Not measured", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("AUC by Category")

    # Top-right: Oracle F1 distribution
    ax = axes[0, 1]
    f1s = [r["oracle_f1"] for r in per_ds if r.get("oracle_f1", 0) > 0]
    if f1s:
        ax.hist(f1s, bins=20, color=COLORS["accent"], alpha=0.8, edgecolor=COLORS["bg"])
        med = np.median(f1s)
        ax.axvline(
            med,
            color=COLORS["primary"],
            linestyle="--",
            linewidth=2,
            label=f"median = {med:.4f}",
        )
        ax.set_xlabel("Oracle F1")
        ax.set_ylabel("Count")
        ax.set_title("Oracle F1 Distribution\n(upper bound, not operational)")
        ax.legend(fontsize=10, loc="upper left")
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
        ax.scatter(
            rx,
            igx,
            c=COLORS["primary"],
            alpha=0.6,
            s=40,
            edgecolors=COLORS["muted"],
            linewidths=0.5,
        )
        ax.plot([0, 1], [0, 1], "--", color=COLORS["muted"], alpha=0.5, label="y = x")
        ax.set_xlabel("Resonance AUC")
        ax.set_ylabel("InfoGeometry AUC")
        ax.set_title("Component Correlation\n(Resonance vs InfoGeometry per dataset)")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc="lower right")
    else:
        ax.text(0.5, 0.5, "Not measured", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Component Correlation")

    # Bottom-right: timing (log scale)
    ax = axes[1, 1]
    fit_times = [r["fit_ms"] for r in per_ds if "fit_ms" in r]
    score_times = [r["score_ms"] for r in per_ds if "score_ms" in r]
    if fit_times:
        ax.boxplot(
            [fit_times, score_times],
            tick_labels=["fit()", "detect()"],
            patch_artist=True,
            boxprops={"facecolor": COLORS["card"], "edgecolor": COLORS["primary"]},
            medianprops={"color": COLORS["accent"], "linewidth": 2},
            whiskerprops={"color": COLORS["text"]},
            capprops={"color": COLORS["text"]},
            flierprops={"marker": ".", "markersize": 4, "markerfacecolor": COLORS["muted"]},
        )
        ax.set_yscale("log")
        ax.set_ylabel("Time (ms, log scale)")
        ax.set_title("Timing Distribution")
    else:
        ax.text(0.5, 0.5, "Not measured", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Timing")

    _stamp(fig)
    plt.subplots_adjust(wspace=0.30, hspace=0.40, left=0.07, right=0.97, top=0.93, bottom=0.05)
    fig.savefig(OUTPUT_DIR / "anomaly_detection_panel.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved anomaly_detection_panel.png")


# ---------------------------------------------------------------------------
# Chart 3: Performance Dashboard
# ---------------------------------------------------------------------------
def generate_performance_dashboard(data: dict) -> None:
    per_ds = [r for r in data.get("per_dataset", []) if r.get("error") is None]
    summary = data.get("summary", {})
    domain_summary = data.get("domain_summary", {})

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle("Mercury Agent - Performance Dashboard", fontsize=18, fontweight="bold", y=0.97)

    # Top-left: Domain Performance — mean AUC and F1 per domain (from domain_summary)
    ax = axes[0, 0]
    if domain_summary:
        domains = sorted(domain_summary.keys())
        d_aucs = [domain_summary[d].get("stats", {}).get("mean_auc") or 0 for d in domains]
        d_f1s = [domain_summary[d].get("stats", {}).get("mean_f1") or 0 for d in domains]
        y_pos = np.arange(len(domains))
        bar_h = 0.35
        ax.barh(y_pos - bar_h / 2, d_aucs, bar_h, label="Mean AUC", color=COLORS["primary"])
        ax.barh(y_pos + bar_h / 2, d_f1s, bar_h, label="Mean F1", color=COLORS["accent"])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(domains, fontsize=8)
        ax.set_xlabel("Score")
        ax.set_title("Domain Performance (AUC & F1)")
        ax.set_xlim(0, 1.15)
        ax.legend(fontsize=9, loc="lower right")
        for i, (auc_v, f1_v) in enumerate(zip(d_aucs, d_f1s)):
            if auc_v > 0:
                ax.text(auc_v + 0.01, i - bar_h / 2, f"{auc_v:.3f}", va="center", fontsize=7)
            if f1_v > 0:
                ax.text(f1_v + 0.01, i + bar_h / 2, f"{f1_v:.3f}", va="center", fontsize=7)
    else:
        ax.text(0.5, 0.5, "Not measured", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Domain Performance")

    # Top-right: Component Heatmap — best component per domain
    ax = axes[0, 1]
    if domain_summary:
        domains = sorted(domain_summary.keys())
        components = ["resonance", "kinematic", "info_geometry"]
        heatmap_data = []
        for d in domains:
            row = []
            comp_aucs = domain_summary[d].get("stats", {}).get("component_mean_aucs", {})
            for c in components:
                row.append(comp_aucs.get(c, 0))
            heatmap_data.append(row)
        heatmap_arr = np.array(heatmap_data)
        im = ax.imshow(heatmap_arr, cmap="YlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(components)))
        ax.set_xticklabels(["Resonance", "Kinematic", "InfoGeo"], fontsize=9)
        ax.set_yticks(np.arange(len(domains)))
        ax.set_yticklabels(domains, fontsize=8)
        for i in range(len(domains)):
            for j in range(len(components)):
                val = heatmap_arr[i, j]
                if val > 0:
                    text_color = "black" if val > 0.6 else COLORS["text"]
                    ax.text(
                        j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color=text_color
                    )
        fig.colorbar(im, ax=ax, shrink=0.7, label="Mean AUC")
        ax.set_title("Component AUC by Domain")
    else:
        ax.text(0.5, 0.5, "Not measured", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Component Heatmap")

    # Bottom-left: Precision vs Recall scatter
    ax = axes[1, 0]
    precs = [r.get("oracle_precision", 0) for r in per_ds]
    recs = [r.get("oracle_recall", 0) for r in per_ds]
    valid3 = [(p, r) for p, r in zip(precs, recs) if p > 0 or r > 0]
    if valid3:
        px, rx = zip(*valid3)
        ax.scatter(
            px,
            rx,
            c=COLORS["secondary"],
            alpha=0.6,
            s=40,
            edgecolors=COLORS["muted"],
            linewidths=0.5,
        )
        ax.set_xlabel("Oracle Precision", fontsize=12)
        ax.set_ylabel("Oracle Recall", fontsize=12)
        ax.set_title("Precision-Recall Trade-off\n(oracle threshold)")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
    else:
        ax.text(0.5, 0.5, "Not measured", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Precision vs Recall")

    # Bottom-right: Oracle Status & Key Metrics
    ax = axes[1, 1]
    ax.axis("off")
    oracle_active = sum(1 for d in domain_summary.values() if d.get("oracle_active_count", 0) > 0)
    oracle_total = sum(d.get("oracle_active_count", 0) for d in domain_summary.values())
    lines = [
        f"Mean AUC:         {_fmt(summary.get('mean_auc'))}",
        f"Median AUC:       {_fmt(summary.get('median_auc'))}",
        f"Std AUC:          {_fmt(summary.get('std_auc'))}",
        f"Mean Oracle F1:   {_fmt(summary.get('mean_oracle_f1'))}",
        f"Median Oracle F1: {_fmt(summary.get('median_oracle_f1'))}",
        "",
        "ORACLE STATUS",
        f"  Domains w/ Oracle active: {oracle_active}",
        f"  Total datasets w/ Oracle: {oracle_total}",
        "",
        "NOTE: Oracle F1 is an upper bound.",
        "Threshold selected per-dataset",
        "on the test set (101 thresholds).",
    ]
    ax.text(
        0.05,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        family="monospace",
    )
    ax.set_title("Key Metrics & Oracle Status")

    _stamp(fig)
    plt.subplots_adjust(wspace=0.30, hspace=0.35, left=0.07, right=0.97, top=0.93, bottom=0.05)
    fig.savefig(OUTPUT_DIR / "mercury_performance_dashboard.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved mercury_performance_dashboard.png")


# ---------------------------------------------------------------------------
# Chart 4: Benchmark Summary (Live Data)
# ---------------------------------------------------------------------------
def generate_benchmark_summary(data: dict) -> None:
    per_ds = [r for r in data.get("per_dataset", []) if r.get("error") is None]
    summary = data.get("summary", {})
    meta = data.get("metadata", {})

    fig, axes = plt.subplots(1, 2, figsize=(20, 12), gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle("Mercury Agent - Benchmark Summary", fontsize=18, fontweight="bold", y=0.97)

    # Left: sorted AUC bar chart (all datasets)
    ax = axes[0]
    sorted_ds = sorted(per_ds, key=lambda r: r.get("ensemble_auc", 0), reverse=True)
    if sorted_ds:
        names = [r["name"] for r in sorted_ds]
        vals = [r["ensemble_auc"] for r in sorted_ds]
        colors_list = [
            COLORS["good"] if v >= 0.7 else COLORS["warn"] if v >= 0.5 else COLORS["bad"]
            for v in vals
        ]
        ax.barh(names[::-1], vals[::-1], color=colors_list[::-1], height=0.7)
        ax.set_xlabel("Ensemble AUC")
        ax.set_title(f"Per-Dataset AUC ({len(sorted_ds)} datasets)")
        ax.set_xlim(0, 1.05)
        ax.tick_params(axis="y", labelsize=7)
    else:
        ax.text(0.5, 0.5, "Not measured", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Per-Dataset AUC")

    # Right: metadata and summary
    ax = axes[1]
    ax.axis("off")
    lines = [
        "BENCHMARK METADATA",
        f"  Detector:   {meta.get('detector', 'N/A')}",
        f"  Git commit: {meta.get('git_commit', 'N/A')[:12]}",
        f"  Python:     {meta.get('python_version', 'N/A')}",
        f"  Timestamp:  {meta.get('timestamp', 'N/A')[:19]}",
        f"  Max samples: {meta.get('max_samples_per_dataset', 'N/A')}",
        "",
        "AGGREGATE RESULTS",
        f"  Datasets:       {summary.get('successful', 0)} / {summary.get('total_datasets', 0)}",
        f"  Mean AUC:       {_fmt(summary.get('mean_auc'))}",
        f"  Median AUC:     {_fmt(summary.get('median_auc'))}",
        f"  Std AUC:        {_fmt(summary.get('std_auc'))}",
        f"  Mean Oracle F1: {_fmt(summary.get('mean_oracle_f1'))}",
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
        fontsize=10,
        verticalalignment="top",
        family="monospace",
    )

    _stamp(fig)
    plt.subplots_adjust(wspace=0.15, left=0.10, right=0.97, top=0.93, bottom=0.05)
    fig.savefig(OUTPUT_DIR / "benchmark_summary_live_data.png", bbox_inches="tight")
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
