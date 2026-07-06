# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Real-data benchmark for the streaming anomaly-detector tier (NAB).

This evaluates the tier on **real, human-labelled** anomaly data -- the Numenta
Anomaly Benchmark (NAB) real categories (``realKnownCause`` /
``realAWSCloudwatch`` / ``realTraffic``), pulled through the shared dataset
layer (:class:`omni_mercury_engine.datasets.timeseries.NABLoader`, the same
loader the main :mod:`benchmarks.mercury_benchmark` registers). Nothing scored
here is generated: NAB's synthetic ``artificial*`` categories are excluded by
the loader's default real-category selection.

It is a *library*, not a standalone results silo: :func:`run_realdata_benchmark`
returns a results dict that :mod:`benchmarks.mercury_benchmark` merges into the
one canonical ``mercury_benchmark_results.json`` under the ``detection_tier``
key (there is no separate committed results file).

Protocol -- NAB is an **unsupervised streaming** benchmark, so the honest,
non-leaking evaluation is:

* **Members + ``average`` ensemble (headline).** Each 1-D member (and the
  unsupervised score-mean ensemble) is fitted on an initial *normal* warm-up
  window, then scores the whole series; per-point ROC-AUC (Mann-Whitney rank
  identity, no scikit-learn) and an oracle best-F1 are computed over every
  labelled point. Every series with an anomaly is measurable this way.
* **Supervised ``stacking`` / ``bma`` (subset).** These combiners need labelled
  anomalies to fit, which NAB does not provide up front. They are evaluated only
  on the subset of series where a 50/50 temporal split leaves *both* classes in
  *both* folds (fit on the labelled train split, score the test split); series
  that cannot support it are recorded in ``skipped``, never silently dropped.

Long series are cropped to a contiguous window centred on their labelled
anomalies (real data, real anomaly, temporal crop only) so the O(n) detectors
stay fast. Per-detector failures are trapped and recorded so one misbehaving
detector never aborts the run.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from omni_mercury_engine.detectors.detection_tier import (
    StreamingScoreEnsemble,
    align_point_scores,
    build_tier_detectors,
)

__all__ = [
    "ENSEMBLE_METHODS",
    "MEMBER_DETECTORS",
    "evaluate_member",
    "load_nab_series",
    "main",
    "run_realdata_benchmark",
]

#: 1-D-capable tier members benchmarked here (multivariate ``rca`` /
#: ``deeplog_sequence`` / ``frequent_pattern`` and the torch-gated ``srcnn`` /
#: ``diffusion_ad`` are excluded so the lane is pure-NumPy and always importable).
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

#: Ensemble combiners. ``average`` is unsupervised (headline); ``stacking`` /
#: ``bma`` are supervised and evaluated on the split-measurable subset.
ENSEMBLE_METHODS: tuple[str, ...] = ("average", "bma", "stacking")

#: NAB real categories to pull (synthetic ``artificial*`` sets are excluded).
_NAB_REAL_CATEGORIES: tuple[str, ...] = (
    "realKnownCause",
    "realAWSCloudwatch",
    "realTraffic",
)

#: Fraction of a series used as the normal warm-up the members fit on.
_WARMUP_FRAC = 0.15
#: Floor on the warm-up length so short series still fit meaningfully.
_WARMUP_MIN = 200
#: Length cap; longer series are cropped anomaly-preservingly to bound runtime.
_MAX_LEN = 6000
#: Timed ``detect`` calls averaged for the latency/throughput metric.
_LATENCY_REPEATS = 2


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return 1-based average ranks with tie handling (fractional ranking)."""
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
    """ROC-AUC via the Mann-Whitney U rank identity (equals scikit-learn's AUC).

    Returns ``0.5`` when only one class is present (AUC is undefined there).
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
    """Precision / recall / F1 from 0/1 predictions and labels."""
    preds = np.asarray(preds, dtype=np.int64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _oracle_f1(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Best F1 over a percentile + linear threshold sweep (oracle threshold).

    Mirrors the multi-strategy sweep in :mod:`benchmarks.mercury_benchmark` so
    the tier's F1 is comparable to the main harness's ``oracle_f1``.
    """
    labels = np.asarray(labels, dtype=np.int64).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if int(labels.sum()) in (0, labels.size):
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": 0.5}
    candidates = list(np.linspace(0.0, 1.0, 51))
    candidates += [float(np.percentile(scores, p)) for p in (80, 85, 90, 93, 95, 97, 99)]
    best = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": 0.5}
    for thr in candidates:
        metrics = _classification_metrics((scores > thr).astype(np.int64), labels)
        if metrics["f1"] > best["f1"]:
            best = {**metrics, "threshold": float(thr)}
    return best


def _round_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Round metric values to 6 decimals for stable JSON output."""
    return {key: round(float(value), 6) for key, value in metrics.items()}


def _crop_to_anomaly(
    series: np.ndarray, labels: np.ndarray, max_len: int
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a long series to a ``max_len`` window that retains its anomalies."""
    n = int(series.size)
    if n <= max_len:
        return series, labels
    pos = np.flatnonzero(labels == 1)
    if pos.size == 0:
        lo = 0
    else:
        center = int((pos[0] + pos[-1]) // 2)
        lo = max(0, min(center - max_len // 2, n - max_len))
    hi = lo + max_len
    return series[lo:hi], labels[lo:hi]


def load_nab_series(
    categories: tuple[str, ...] = _NAB_REAL_CATEGORIES,
    *,
    max_len: int = _MAX_LEN,
    max_files: int | None = None,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Load real NAB 1-D series (name, values, labels) via the shared loader.

    Args:
        categories: NAB real categories to include.
        max_len: Length cap; longer series are anomaly-preservingly cropped.
        max_files: Optional cap on the number of series (for smoke runs).

    Returns:
        One ``(name, values, labels)`` triple per real NAB file with at least one
        labelled anomaly in the (cropped) window.
    """
    from omni_mercury_engine.datasets.base import DatasetConfig
    from omni_mercury_engine.datasets.timeseries import NABLoader

    loader = NABLoader(DatasetConfig(name="nab", preprocessing={"categories": list(categories)}))
    out: list[tuple[str, np.ndarray, np.ndarray]] = []
    for name, values, labels in loader.iter_series():
        series, lab = _crop_to_anomaly(
            np.asarray(values, dtype=np.float64), np.asarray(labels, dtype=np.int64), max_len
        )
        if int(lab.sum()) == 0:
            continue
        out.append((name, series, lab))
        if max_files is not None and len(out) >= max_files:
            break
    return out


def _warmup_len(n: int) -> int:
    """Warm-up length: ``_WARMUP_FRAC`` of the series, floored, kept below n/2."""
    return int(min(max(_WARMUP_MIN, int(_WARMUP_FRAC * n)), max(2, n // 2)))


def evaluate_member(name: str, series: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Fit one member on the normal warm-up, score the whole series, score it.

    Args:
        name: Member detector name.
        series: 1-D real series (temporal order preserved).
        labels: 0/1 per-point labels aligned to ``series``.

    Returns:
        Metrics dict (``detector``, ``roc_auc``, ``f1``, ``precision``,
        ``recall``, ``threshold``, ``latency_ms``, ``throughput_pps``) or
        ``{"detector", "error"}``.
    """
    try:
        detector = build_tier_detectors([name])[name]
        warmup = _warmup_len(series.size)
        detector.fit(np.nan_to_num(series[:warmup]))
        scores = align_point_scores(detector, series)
        metrics: dict[str, Any] = {"roc_auc": _roc_auc(scores, labels)}
        metrics.update(_oracle_f1(scores, labels))

        start = time.perf_counter()
        for _ in range(_LATENCY_REPEATS):
            detector.detect(series)
        metrics["latency_ms"] = (time.perf_counter() - start) / _LATENCY_REPEATS * 1000.0
        per_call_s = metrics["latency_ms"] / 1000.0
        metrics["throughput_pps"] = float(series.size / per_call_s) if per_call_s > 0 else 0.0

        result: dict[str, Any] = {**_round_metrics(metrics), "detector": name}
        return result
    except Exception as exc:  # noqa: BLE001 - one bad detector must not abort the run
        return {"detector": name, "error": f"{type(exc).__name__}: {exc}"}


def _evaluate_average_ensemble(
    members: list[str], series: np.ndarray, labels: np.ndarray, seed: int
) -> dict[str, Any]:
    """Unsupervised ``average`` ensemble: fit members on warm-up, score all."""
    try:
        detectors = build_tier_detectors(members)
        ensemble = StreamingScoreEnsemble(detectors, method="average", seed=seed)
        warmup = _warmup_len(series.size)
        ensemble.fit(series[:warmup])
        scores = ensemble.score(series)
        metrics: dict[str, Any] = {"roc_auc": _roc_auc(scores, labels)}
        metrics.update(_oracle_f1(scores, labels))
        return _round_metrics(metrics)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def _evaluate_supervised_ensembles(
    members: list[str], series: np.ndarray, labels: np.ndarray, seed: int
) -> dict[str, dict[str, Any]] | None:
    """Supervised ``stacking`` / ``bma`` / ``average`` on a 50/50 temporal split.

    Returns ``None`` when the split is not measurable (a fold is single-class),
    so the caller can record the series as skipped for the supervised table.
    """
    split = series.size // 2
    train_s, train_l = series[:split], labels[:split]
    test_s, test_l = series[split:], labels[split:]
    both = lambda y: 0 < int(y.sum()) < int(y.size)  # noqa: E731 - tiny local predicate
    if not (both(train_l) and both(test_l)):
        return None

    results: dict[str, dict[str, Any]] = {}
    for method in ENSEMBLE_METHODS:
        try:
            detectors = build_tier_detectors(members)
            ensemble = StreamingScoreEnsemble(detectors, method=method, seed=seed)
            ensemble.fit(train_s, train_l)
            scores = ensemble.score(test_s)
            metrics: dict[str, Any] = {"roc_auc": _roc_auc(scores, test_l)}
            metrics.update(_oracle_f1(scores, test_l))
            results[method] = _round_metrics(metrics)
        except Exception as exc:  # noqa: BLE001
            results[method] = {"error": f"{type(exc).__name__}: {exc}"}
    return results


def _aggregate(
    per_dataset: dict[str, dict[str, Any]], keys: list[str]
) -> dict[str, dict[str, Any]]:
    """Mean ROC-AUC / F1 / latency for each key across datasets (skipping errors)."""
    aggregate: dict[str, dict[str, Any]] = {}
    for key in keys:
        aucs, f1s, lats, tputs = [], [], [], []
        for entry in per_dataset.values():
            metrics = entry.get(key, {})
            if "roc_auc" in metrics and "error" not in metrics:
                aucs.append(metrics["roc_auc"])
                f1s.append(metrics["f1"])
                if "latency_ms" in metrics:
                    lats.append(metrics["latency_ms"])
                    tputs.append(metrics.get("throughput_pps", 0.0))
        if not aucs:
            aggregate[key] = {"error": "no successful datasets"}
            continue
        summary = {
            "mean_auc": round(float(np.mean(aucs)), 6),
            "median_auc": round(float(np.median(aucs)), 6),
            "mean_f1": round(float(np.mean(f1s)), 6),
            "n_datasets": len(aucs),
        }
        if lats:
            summary["mean_latency_ms"] = round(float(np.mean(lats)), 6)
            summary["mean_throughput_pps"] = round(float(np.mean(tputs)), 2)
        aggregate[key] = summary
    return aggregate


def run_realdata_benchmark(
    seed: int = 0,
    *,
    max_len: int = _MAX_LEN,
    max_files: int | None = None,
    members: list[str] | tuple[str, ...] | None = None,
    datasets: list[tuple[str, np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Evaluate the tier on real NAB series and aggregate the results.

    Args:
        seed: Master seed forwarded to every ensemble.
        max_len: Length cap forwarded to :func:`load_nab_series`.
        max_files: Optional cap on datasets (for smoke runs).
        members: Subset of :data:`MEMBER_DETECTORS` (defaults to all 1-D members).
        datasets: Pre-loaded ``(name, series, labels)`` triples; when given, the
            NAB loader is not called (used by network-free tests).

    Returns:
        A results dict with ``config``, per-dataset ``datasets`` tables, ``aggregate``
        means (``members`` + unsupervised ``average`` + supervised ``stacking`` /
        ``bma``), a ``summary``, ``measured_datasets``, and a ``skipped`` list.
    """
    member_names = list(members) if members is not None else list(MEMBER_DETECTORS)
    if datasets is None:
        datasets = load_nab_series(max_len=max_len, max_files=max_files)

    dataset_tables: dict[str, Any] = {}
    member_by_ds: dict[str, dict[str, Any]] = {}
    unsup_by_ds: dict[str, dict[str, Any]] = {}
    sup_by_ds: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, str]] = []
    measured: list[str] = []

    for name, series, labels in datasets:
        members_result = {m: evaluate_member(m, series, labels) for m in member_names}
        average_result = _evaluate_average_ensemble(member_names, series, labels, seed)
        supervised = _evaluate_supervised_ensembles(member_names, series, labels, seed)

        member_by_ds[name] = members_result
        unsup_by_ds[name] = {"average": average_result}
        if supervised is None:
            skipped.append(
                {
                    "dataset": name,
                    "reason": "temporal split single-class (supervised not measurable)",
                }
            )
        else:
            sup_by_ds[name] = supervised

        # Compact per-dataset row (the full per-member matrix stays in the
        # cross-dataset ``aggregate`` means; keeping it out of the committed file
        # keeps the headline artefact reviewable).
        scored_members = {m: r["roc_auc"] for m, r in members_result.items() if "roc_auc" in r}
        best_member = (
            max(scored_members, key=scored_members.__getitem__) if scored_members else None
        )
        sup_row: dict[str, Any] = {"measurable": supervised is not None}
        if supervised is not None:
            sup_aucs = {m: r["roc_auc"] for m, r in supervised.items() if "roc_auc" in r}
            if sup_aucs:
                best_sup = max(sup_aucs, key=sup_aucs.__getitem__)
                sup_row.update(best_method=best_sup, best_auc=round(sup_aucs[best_sup], 6))
        dataset_tables[name] = {
            "n": int(series.size),
            "anomaly_rate": round(float(np.mean(labels)), 6),
            "best_member": best_member,
            "best_member_auc": round(scored_members[best_member], 6) if best_member else None,
            "ensemble_average_auc": average_result.get("roc_auc"),
            "ensemble_average_f1": average_result.get("f1"),
            "supervised": sup_row,
        }
        measured.append(name)

    member_agg = _aggregate(member_by_ds, member_names)
    average_agg = _aggregate(unsup_by_ds, ["average"])
    supervised_agg = _aggregate(sup_by_ds, list(ENSEMBLE_METHODS))

    measurable_members = {k: v for k, v in member_agg.items() if "mean_auc" in v}
    best_member = (
        max(measurable_members.items(), key=lambda kv: kv[1]["mean_auc"])[0]
        if measurable_members
        else None
    )

    return {
        "config": {
            "source": "NAB (Numenta Anomaly Benchmark) real categories",
            "loader": "omni_mercury_engine.datasets.timeseries.NABLoader",
            "categories": list(_NAB_REAL_CATEGORIES),
            "license": "AGPL-3.0",
            "seed": seed,
            "max_len": max_len,
            "warmup_frac": _WARMUP_FRAC,
            "members": member_names,
            "ensemble_methods": list(ENSEMBLE_METHODS),
            "protocol": (
                "unsupervised streaming: members + average ensemble fit on the normal "
                "warm-up and score the whole series (per-point ROC-AUC + oracle F1 over "
                "all labels); supervised stacking/bma on the 50/50-split subset where both "
                "folds carry both classes"
            ),
            "n_datasets_measured": len(measured),
        },
        "datasets": dataset_tables,
        "aggregate": {
            "members": member_agg,
            "ensemble_average": average_agg.get("average", {"error": "no successful datasets"}),
            "ensembles_supervised": supervised_agg,
        },
        "summary": {
            "n_datasets": len(measured),
            "n_supervised_measurable": len(sup_by_ds),
            "best_member": best_member,
            "best_member_mean_auc": (
                measurable_members[best_member]["mean_auc"] if best_member else None
            ),
            "average_ensemble_mean_auc": average_agg.get("average", {}).get("mean_auc"),
            "bma_ensemble_mean_auc": supervised_agg.get("bma", {}).get("mean_auc"),
            "stacking_ensemble_mean_auc": supervised_agg.get("stacking", {}).get("mean_auc"),
        },
        "measured_datasets": measured,
        "skipped": skipped,
    }


def _format_summary(results: dict[str, Any]) -> str:
    """Render the aggregate results as a compact markdown table."""
    lines = [
        "| entry | mean_auc | median_auc | mean_f1 | n |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    def _row(label: str, stats: dict[str, Any]) -> str:
        if "mean_auc" not in stats:
            return f"| {label} | - | - | - | {stats.get('error', 'n/a')} |"
        return (
            f"| {label} | {stats['mean_auc']:.4f} | {stats.get('median_auc', float('nan')):.4f} | "
            f"{stats['mean_f1']:.4f} | {stats['n_datasets']} |"
        )

    members = results["aggregate"]["members"]
    for name in results["config"]["members"]:
        lines.append(_row(name, members.get(name, {"error": "missing"})))
    lines.append(_row("ensemble:average", results["aggregate"]["ensemble_average"]))
    for method in ("bma", "stacking"):
        lines.append(
            _row(
                f"ensemble:{method}",
                results["aggregate"]["ensembles_supervised"].get(method, {"error": "missing"}),
            )
        )
    return "\n".join(lines)


def main() -> None:
    """Run the real-data benchmark standalone and print a summary (no committed file).

    The canonical committed artefact is the ``detection_tier`` section of
    ``benchmarks/mercury_benchmark_results.json``, produced when
    :mod:`benchmarks.mercury_benchmark` merges this library's output. Running
    this module directly is for local inspection only.
    """
    results = run_realdata_benchmark()
    print(f"# Detector tier -- REAL data (NAB), {results['config']['n_datasets_measured']} series")
    print(_format_summary(results))
    if results["skipped"]:
        print(f"\nsupervised-skipped ({len(results['skipped'])}): single-class temporal split")


if __name__ == "__main__":
    main()
