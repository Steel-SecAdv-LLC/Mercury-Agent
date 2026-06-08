#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare runtime equation profiles against ``baseline_original_v1``.

The harness measures anomaly quality, neuro-symbolic agreement, calibration,
latency/efficiency, stability, σ_Immutable readiness, and ethical-gate health
using the same R/H/O rows consumed by the equation optimizer. A dataset JSON can
be supplied; otherwise the deterministic optimizer fixture is used.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
for _path in (_REPO_ROOT, _SRC):
    if str(_path) not in sys.path:  # pragma: no cover - import bootstrap
        sys.path.insert(0, str(_path))

from omni_mercury_engine.core.calibration import compute_ece
from omni_mercury_engine.core.equation_profiles import (
    BASELINE_PROFILE_ID,
    QUIET_HORIZON_PROFILE_ID,
    score_runtime_equation_profile,
)
from tools.equation_optimizer import _build_default_dataset, _load_dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a runtime equation profile against baseline_original_v1."
    )
    parser.add_argument("--dataset", type=Path, default=None, help="Optional optimizer-row JSON.")
    parser.add_argument("--baseline", default=BASELINE_PROFILE_ID, help="Baseline profile id.")
    parser.add_argument(
        "--candidate", default=QUIET_HORIZON_PROFILE_ID, help="Candidate profile id."
    )
    parser.add_argument("--seed", type=int, default=17, help="Synthetic dataset seed.")
    parser.add_argument("--n", type=int, default=800, help="Synthetic dataset size.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Return 1 on AUC/F1, hard-gate, calibration, stability, or latency regression.",
    )
    return parser


def _scores_for_profile(
    rows: list[dict[str, Any]], profile_id: str
) -> tuple[np.ndarray[Any, Any], float]:
    raw = _baseline_oae(rows)
    r = np.array([float(row["r"]) for row in rows], dtype=np.float64)
    h = np.array([float(row["h"]) for row in rows], dtype=np.float64)
    o = np.array([float(row["o"]) for row in rows], dtype=np.float64)
    eta = np.array([float(row["eta"]) for row in rows], dtype=np.float64)

    start = time.perf_counter()
    scores, _ = score_runtime_equation_profile(raw, r, h, o, eta=eta, profile_id=profile_id)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return scores, elapsed_ms


def _baseline_oae(rows: list[dict[str, Any]]) -> np.ndarray[Any, Any]:
    phi = 1.618033988749895
    w_r = phi / (phi + 2.0)
    w_h = 1.0 / (phi + 2.0)
    w_o = 1.0 / (phi + 2.0)
    r = np.array([float(row["r"]) for row in rows], dtype=np.float64)
    h = np.array([float(row["h"]) for row in rows], dtype=np.float64)
    o = np.array([float(row["o"]) for row in rows], dtype=np.float64)
    eta = np.array([float(row["eta"]) for row in rows], dtype=np.float64)
    return np.clip((w_r * r + w_h * h + w_o * o) * np.power(eta, phi), 0.0, 1.0)


def _profile_metrics(
    rows: list[dict[str, Any]], scores: np.ndarray[Any, Any], elapsed_ms: float
) -> dict[str, Any]:
    labels_cont = np.array([float(row["label"]) for row in rows], dtype=np.float64)
    labels = (labels_cont >= 0.5).astype(int)
    f1, threshold = _compute_f1_max(labels, scores)

    stable = np.array(
        [
            float(row.get("alpha", 1.0)) < 0.999 and float(row.get("lyapunov_lambda", 0.0)) > 1e-6
            for row in rows
        ],
        dtype=bool,
    )
    sigma_ok = np.array([bool(row.get("sigma_ok", True)) for row in rows], dtype=bool)
    gate_ok = np.array([bool(row.get("gate_ok", True)) for row in rows], dtype=bool)
    component_mean = np.mean(
        np.vstack(
            [
                np.array([float(row["r"]) for row in rows], dtype=np.float64),
                np.array([float(row["h"]) for row in rows], dtype=np.float64),
                np.array([float(row["o"]) for row in rows], dtype=np.float64),
            ]
        ),
        axis=0,
    )

    domains: dict[str, dict[str, Any]] = {}
    for domain in sorted({str(row.get("domain", "general")) for row in rows}):
        idx = np.array([str(row.get("domain", "general")) == domain for row in rows], dtype=bool)
        d_labels = labels[idx]
        d_scores = scores[idx]
        d_f1, _ = _compute_f1_max(d_labels, d_scores)
        domains[domain] = {
            "n": int(np.sum(idx)),
            "auc": _compute_auroc(d_labels, d_scores),
            "f1": d_f1,
            "ece": compute_ece(d_labels, d_scores),
        }

    return {
        "auc": _compute_auroc(labels, scores),
        "f1": f1,
        "threshold": threshold,
        "ece": compute_ece(labels, scores),
        "latency_ms": elapsed_ms,
        "latency_ms_per_1k": elapsed_ms / max(len(rows), 1) * 1000.0,
        "stability": float(np.mean(stable)),
        "sigma_immutable_violations": int(np.sum(~sigma_ok)),
        "ethical_gate_violations": int(np.sum(~gate_ok)),
        "neuro_symbolic_satisfaction": float(1.0 - np.mean(np.abs(scores - component_mean))),
        "domains": domains,
    }


def _compute_auroc(y_true: np.ndarray[Any, Any], y_score: np.ndarray[Any, Any]) -> float:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    sorted_scores = y_score[order]
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = float(np.mean(np.arange(start + 1, end + 1)))
        start = end

    pos_ranks = float(np.sum(ranks[y_true == 1]))
    return float((pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _compute_f1_max(
    y_true: np.ndarray[Any, Any], y_score: np.ndarray[Any, Any], n_thresholds: int = 100
) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y_score.size == 0:
        return 0.0, 0.5
    thresholds = np.linspace(float(np.min(y_score)), float(np.max(y_score)), n_thresholds)
    best_f1 = 0.0
    best_threshold = 0.5
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall <= 0.0:
            continue
        f1 = 2.0 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return float(best_f1), best_threshold


def _delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "auc",
        "f1",
        "ece",
        "latency_ms",
        "latency_ms_per_1k",
        "stability",
        "neuro_symbolic_satisfaction",
        "sigma_immutable_violations",
        "ethical_gate_violations",
    )
    return {field: candidate[field] - baseline[field] for field in fields}


def _regressed(delta: dict[str, Any]) -> bool:
    return bool(
        delta["auc"] < -0.005
        or delta["f1"] < -0.005
        or delta["ece"] > 0.005
        or delta["latency_ms_per_1k"] > 0.25
        or delta["stability"] < 0.0
        or delta["sigma_immutable_violations"] > 0
        or delta["ethical_gate_violations"] > 0
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.dataset is None:
        rows = _build_default_dataset(seed=args.seed, n=args.n)
        dataset_source = f"synthetic_default:seed={args.seed}:n={args.n}"
    else:
        rows, dataset_source = _load_dataset(args.dataset, seed=args.seed)

    baseline_scores, baseline_latency = _scores_for_profile(rows, args.baseline)
    candidate_scores, candidate_latency = _scores_for_profile(rows, args.candidate)
    baseline = _profile_metrics(rows, baseline_scores, baseline_latency)
    candidate = _profile_metrics(rows, candidate_scores, candidate_latency)
    delta = _delta(candidate, baseline)
    regressed = _regressed(delta)

    report = {
        "dataset_source": dataset_source,
        "baseline_profile": args.baseline,
        "candidate_profile": args.candidate,
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "hard_gates_preserved": (
            delta["sigma_immutable_violations"] <= 0
            and delta["ethical_gate_violations"] <= 0
            and delta["stability"] >= 0.0
        ),
        "regressed": regressed,
    }

    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 1 if args.fail_on_regression and regressed else 0


if __name__ == "__main__":
    raise SystemExit(main())
