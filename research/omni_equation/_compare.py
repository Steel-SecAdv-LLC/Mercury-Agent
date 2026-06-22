# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure base-vs-current diff for the ADBench transductive harness.

Kept engine-free (stdlib only) so the per-dataset win/tie/loss comparison can be
imported and unit-tested without building the AMA native backend that importing
``MercuryAnomalyDetector`` requires. ``harness_adbench.py`` calls this when run
with ``--baseline``; ``tests/research/test_adbench_baseline_compare.py`` pins it.
"""

from __future__ import annotations

from typing import Any


def compare_to_baseline(
    current: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    tie_tol: float = 0.0,
) -> dict[str, Any]:
    """Per-dataset AUROC diff of a current run against a baseline run.

    Both inputs are the ``results`` arrays the harness writes (lists of per-set
    rows). Only datasets scored in *both* runs are compared; a row missing an
    ``auroc`` (a load/scoring failure) is skipped. Each delta is rounded to 4 dp
    and classified ``win`` (delta > ``tie_tol``), ``loss`` (delta < -``tie_tol``),
    else ``tie``.

    ``tie_tol`` defaults to 0.0 on purpose: a sub-noise regression is still a
    loss, not rounded into a tie. The two PR-302 losses (Waveform -0.0003,
    WPBC -0.0002) are negligible but real, and are counted as losses so the
    committed ledger matches the headline rather than flattering it.

    Returns ``{"summary": {...}, "per_set": [...]}`` with ``per_set`` sorted by
    delta descending.
    """
    base_by = {r["dataset"]: r for r in baseline if "auroc" in r}
    rows: list[dict[str, Any]] = []
    for cur in current:
        if "auroc" not in cur or cur["dataset"] not in base_by:
            continue
        base = base_by[cur["dataset"]]
        delta = round(float(cur["auroc"]) - float(base["auroc"]), 4)
        if delta > tie_tol:
            verdict = "win"
        elif delta < -tie_tol:
            verdict = "loss"
        else:
            verdict = "tie"
        rows.append(
            {
                "dataset": cur["dataset"],
                "baseline_auroc": base["auroc"],
                "auroc": cur["auroc"],
                "delta": delta,
                "verdict": verdict,
                "baseline_data_type": base.get("data_type"),
                "data_type": cur.get("data_type"),
            }
        )
    rows.sort(key=lambda r: -r["delta"])

    def _mean(key: str) -> float:
        return round(sum(r[key] for r in rows) / len(rows), 4) if rows else float("nan")

    mean_baseline = _mean("baseline_auroc")
    mean_current = _mean("auroc")
    return {
        "summary": {
            "n_scored": len(rows),
            "mean_baseline": mean_baseline,
            "mean_current": mean_current,
            "mean_delta": round(mean_current - mean_baseline, 4),
            "wins": sum(r["verdict"] == "win" for r in rows),
            "ties": sum(r["verdict"] == "tie" for r in rows),
            "losses": sum(r["verdict"] == "loss" for r in rows),
            "tie_tol": tie_tol,
        },
        "per_set": rows,
    }
