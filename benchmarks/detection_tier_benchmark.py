# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Benchmark harness for the streaming anomaly-detector tier.

This runs every temporal / state-space / probabilistic / generative /
neuromorphic detector in
:data:`omni_mercury_engine.detectors.detection_tier.STREAMING_TIER` (the 1-D
subset -- the multivariate ``rca`` / ``deeplog_sequence`` / ``frequent_pattern``
members are excluded) plus the three :class:`StreamingScoreEnsemble` combiners
(``stacking`` / ``bma`` / ``average``) across the synthetic scenarios defined in
:mod:`benchmarks.detection_tier_synthetic`.

Each scenario series is split 50/50 into train/test. Member detectors are fitted
on the *normal* points of the train split and scored on the test split; the
ensembles are trained on the full labelled train split and evaluated on the test
split. Metrics -- precision, recall, F1, ROC-AUC (computed with NumPy via the
Mann-Whitney U rank identity, so there is no scikit-learn dependency) and mean
detect latency -- are aggregated per detector and per ensemble method.

Run it directly to (re)produce :data:`RESULTS_PATH`::

    python -m benchmarks.detection_tier_benchmark

Per-detector failures are caught and recorded as ``{"error": ...}`` so a single
misbehaving detector never aborts the run. Everything is deterministic under a
fixed seed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks.detection_tier_synthetic import SCENARIOS, generate_scenario
from omni_mercury_engine.detectors.detection_tier import (
    StreamingScoreEnsemble,
    align_point_scores,
    build_tier_detectors,
)

if TYPE_CHECKING:
    from omni_mercury_engine.core.base import BaseDetector

__all__ = [
    "ENSEMBLE_METHODS",
    "MEMBER_DETECTORS",
    "RESULTS_PATH",
    "evaluate_detector",
    "evaluate_ensemble",
    "main",
    "run_benchmark",
]

#: 1-D-capable tier detectors benchmarked here (multivariate members excluded).
MEMBER_DETECTORS: tuple[str, ...] = (
    "spectral_residual",
    "bocpd",
    "spot_evt",
    "hawkes",
    "particle_filter",
    "imm",
    "gaussian_process",
    "echo_state",
    "spiking",
    "digital_twin",
    "survival",
    "energy_based",
    "deep_svdd",
)

#: Ensemble combiners evaluated per scenario.
ENSEMBLE_METHODS: tuple[str, ...] = ("stacking", "bma", "average")

#: Where :func:`main` writes the benchmark results.
RESULTS_PATH = Path(__file__).with_name("detection_tier_results.json")

#: Point-decision threshold applied to member detectors' per-point scores.
_POINT_THRESHOLD = 0.5

#: Number of timed ``detect`` calls averaged for the latency metric.
_LATENCY_REPEATS = 3


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return 1-based average ranks with tie handling (fractional ranking).

    Args:
        values: 1-D array to rank.

    Returns:
        Float array of the same shape; tied values share their mean rank.
    """
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    n = values.size
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney U rank identity (no scikit-learn).

    Args:
        scores: Continuous anomaly scores.
        labels: 0/1 ground-truth labels.

    Returns:
        Area under the ROC curve in ``[0, 1]``, or ``0.5`` when only one class is
        present (AUC is undefined there).
    """
    labels = np.asarray(labels, dtype=np.int64).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _average_ranks(scores)
    rank_sum_pos = float(ranks[labels == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _classification_metrics(preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Precision / recall / F1 from 0/1 predictions and labels.

    Args:
        preds: 0/1 predicted flags.
        labels: 0/1 ground-truth labels.

    Returns:
        Mapping with ``precision``, ``recall`` and ``f1`` keys.
    """
    preds = np.asarray(preds, dtype=np.int64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _round_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Round metric values to 6 decimals for stable JSON output."""
    return {key: round(float(value), 6) for key, value in metrics.items()}


def evaluate_detector(
    name: str,
    detector: BaseDetector,
    series: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Score one fitted detector on a series and compute its metrics.

    Args:
        name: Detector name (echoed into the result).
        detector: A *fitted* tier detector.
        series: Evaluation series (1-D); the detector's per-point scores are
            aligned to it via
            :func:`omni_mercury_engine.detectors.detection_tier.align_point_scores`.
        labels: 0/1 ground-truth labels aligned to ``series``.

    Returns:
        Mapping with ``detector``, ``precision``, ``recall``, ``f1``, ``roc_auc``
        and ``latency_ms`` (mean wall time of one ``detect`` call).
    """
    scores = align_point_scores(detector, series)
    preds = (scores > _POINT_THRESHOLD).astype(np.int64)
    metrics = _classification_metrics(preds, labels)
    metrics["roc_auc"] = _roc_auc(scores, labels)

    start = time.perf_counter()
    for _ in range(_LATENCY_REPEATS):
        detector.detect(series)
    metrics["latency_ms"] = (time.perf_counter() - start) / _LATENCY_REPEATS * 1000.0

    result = _round_metrics(metrics)
    result["detector"] = name
    return result


def evaluate_ensemble(
    members: list[str] | tuple[str, ...],
    series_train: np.ndarray,
    labels_train: np.ndarray,
    series_test: np.ndarray,
    labels_test: np.ndarray,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Train and evaluate each ensemble combiner on a train/test split.

    For each method in :data:`ENSEMBLE_METHODS` a fresh set of member detectors is
    built, the ensemble is fitted on the labelled train split, and its per-point
    probabilities score the test split. Precision/recall/F1 use the ensemble's own
    calibrated decision (:meth:`StreamingScoreEnsemble.predict`) while ROC-AUC uses
    the continuous :meth:`StreamingScoreEnsemble.score`. Errors are trapped per
    method.

    Args:
        members: Member detector names to combine.
        series_train: Train split series.
        labels_train: Train split labels.
        series_test: Test split series.
        labels_test: Test split labels.
        seed: Ensemble RNG seed.

    Returns:
        Mapping ``method -> metrics`` (or ``method -> {"error": ...}``).
    """
    results: dict[str, dict[str, Any]] = {}
    for method in ENSEMBLE_METHODS:
        try:
            detectors = build_tier_detectors(list(members))
            ensemble = StreamingScoreEnsemble(detectors, method=method, seed=seed)
            ensemble.fit(series_train, labels_train)
            scores = ensemble.score(series_test)
            preds = ensemble.predict(series_test)
            metrics = _classification_metrics(preds, labels_test)
            metrics["roc_auc"] = _roc_auc(scores, labels_test)

            start = time.perf_counter()
            for _ in range(_LATENCY_REPEATS):
                ensemble.score(series_test)
            metrics["latency_ms"] = (time.perf_counter() - start) / _LATENCY_REPEATS * 1000.0
            result = _round_metrics(metrics)
            result["threshold"] = round(float(ensemble.threshold), 6)
            results[method] = result
        except Exception as exc:
            results[method] = {"error": f"{type(exc).__name__}: {exc}"}
    return results


def _split_scenario(
    series: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a scenario 50/50 into ``(train_series, train_labels, test_*)``."""
    split = series.size // 2
    return series[:split], labels[:split], series[split:], labels[split:]


def _fit_member(name: str, train_series: np.ndarray, train_labels: np.ndarray) -> BaseDetector:
    """Build one detector and fit it on the normal points of the train split.

    NaN dropouts are neutralised before fitting so detectors that assume finite
    input do not choke on the ``missing_data`` scenario.
    """
    detector = build_tier_detectors([name])[name]
    normal = train_series[train_labels == 0]
    fit_input = normal if normal.size > 0 else train_series
    detector.fit(np.nan_to_num(fit_input.astype(np.float64)))
    return detector


def _aggregate(
    per_scenario: dict[str, dict[str, Any]], keys: list[str]
) -> dict[str, dict[str, Any]]:
    """Mean F1 / AUC / latency for each entry across scenarios (skipping errors).

    Args:
        per_scenario: ``scenario -> {entry_key -> metrics}`` mapping.
        keys: Entry keys to aggregate (detector or ensemble-method names).

    Returns:
        ``entry_key -> {mean_f1, mean_auc, mean_latency_ms, n_scenarios}`` (or an
        ``{"error": ...}`` note when every scenario failed for that entry).
    """
    aggregate: dict[str, dict[str, Any]] = {}
    for key in keys:
        f1s, aucs, lats = [], [], []
        for scenario in per_scenario.values():
            entry = scenario.get(key, {})
            if "f1" in entry:
                f1s.append(entry["f1"])
                aucs.append(entry["roc_auc"])
                lats.append(entry["latency_ms"])
        if not f1s:
            aggregate[key] = {"error": "no successful scenarios"}
            continue
        aggregate[key] = {
            "mean_f1": round(float(np.mean(f1s)), 6),
            "mean_auc": round(float(np.mean(aucs)), 6),
            "mean_latency_ms": round(float(np.mean(lats)), 6),
            "n_scenarios": len(f1s),
        }
    return aggregate


def run_benchmark(
    seed: int = 0,
    n: int = 1800,
    scenarios: list[str] | None = None,
    members: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Run the full detector x scenario benchmark and aggregate results.

    Args:
        seed: Master seed forwarded to every scenario generator and ensemble.
        n: Points per scenario series (kept modest to bound runtime).
        scenarios: Subset of :data:`SCENARIOS` to run (defaults to all). Useful
            for fast smoke tests.
        members: Subset of :data:`MEMBER_DETECTORS` to run (defaults to all).

    Returns:
        Nested results dict with ``config``, per-scenario ``scenarios`` tables
        (``detectors`` + ``ensembles``), ``aggregate`` means, and a ``skipped``
        list of ``(scenario, detector, reason)`` triples.
    """
    scenario_names = list(scenarios) if scenarios is not None else list(SCENARIOS)
    member_names = list(members) if members is not None else list(MEMBER_DETECTORS)

    scenario_tables: dict[str, Any] = {}
    detector_by_scenario: dict[str, dict[str, Any]] = {}
    ensemble_by_scenario: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, str]] = []

    for scenario_name in scenario_names:
        series, labels = generate_scenario(scenario_name, n=n, seed=seed)
        train_s, train_l, test_s, test_l = _split_scenario(series, labels)

        detector_results: dict[str, Any] = {}
        for name in member_names:
            try:
                detector = _fit_member(name, train_s, train_l)
                detector_results[name] = evaluate_detector(name, detector, test_s, test_l)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                detector_results[name] = {"error": reason}
                skipped.append({"scenario": scenario_name, "detector": name, "reason": reason})

        ensemble_results = evaluate_ensemble(member_names, train_s, train_l, test_s, test_l, seed)

        detector_by_scenario[scenario_name] = detector_results
        ensemble_by_scenario[scenario_name] = ensemble_results
        scenario_tables[scenario_name] = {
            "anomaly_rate": round(float(np.mean(labels)), 6),
            "n_train": int(train_s.size),
            "n_test": int(test_s.size),
            "detectors": detector_results,
            "ensembles": ensemble_results,
        }

    return {
        "config": {
            "seed": seed,
            "n": n,
            "scenarios": scenario_names,
            "members": member_names,
            "ensemble_methods": list(ENSEMBLE_METHODS),
            "point_threshold": _POINT_THRESHOLD,
        },
        "scenarios": scenario_tables,
        "aggregate": {
            "detectors": _aggregate(detector_by_scenario, member_names),
            "ensembles": _aggregate(ensemble_by_scenario, list(ENSEMBLE_METHODS)),
        },
        "skipped": skipped,
    }


def _format_summary(results: dict[str, Any]) -> str:
    """Render the aggregate results as a compact markdown table."""
    lines = [
        "| detector | mean_f1 | mean_auc | mean_latency_ms |",
        "| --- | ---: | ---: | ---: |",
    ]

    def _row(label: str, stats: dict[str, Any]) -> str:
        if "error" in stats:
            return f"| {label} | - | - | {stats['error']} |"
        return (
            f"| {label} | {stats['mean_f1']:.4f} | "
            f"{stats['mean_auc']:.4f} | {stats['mean_latency_ms']:.3f} |"
        )

    detectors = results["aggregate"]["detectors"]
    for name in results["config"]["members"]:
        lines.append(_row(name, detectors.get(name, {"error": "missing"})))

    ensembles = results["aggregate"]["ensembles"]
    for method in results["config"]["ensemble_methods"]:
        lines.append(_row(f"ensemble:{method}", ensembles.get(method, {"error": "missing"})))

    return "\n".join(lines)


def main() -> None:
    """Run the benchmark, persist JSON to :data:`RESULTS_PATH`, print a summary."""
    results = run_benchmark()
    RESULTS_PATH.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    print(f"# Streaming detector tier benchmark (seed={results['config']['seed']})")
    print(_format_summary(results))
    skipped = results["skipped"]
    if skipped:
        print(f"\nskipped ({len(skipped)}):")
        for item in skipped:
            print(f"  - {item['scenario']}/{item['detector']}: {item['reason']}")
    print(f"\nresults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
