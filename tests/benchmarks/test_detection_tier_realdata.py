# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Network-free tests for the real-data detector-tier benchmark library.

These exercise the scoring/aggregation plumbing of
:mod:`benchmarks.detection_tier_benchmark` (ROC-AUC, oracle F1, anomaly-preserving
crop, and the full :func:`run_realdata_benchmark` structure) on tiny in-memory
fixtures -- NO network, NO NAB download. The tiny arrays here are deterministic
*plumbing fixtures*, not a benchmark: the tier's real performance is measured on
the Numenta Anomaly Benchmark in ``mercury_benchmark.py``'s ``detection_tier``
section. The 1-D streaming accessor the benchmark depends on
(``NABLoader.iter_series``) is contract-checked without touching the network.
"""

from __future__ import annotations

import numpy as np

from benchmarks.detection_tier_benchmark import (
    ENSEMBLE_METHODS,
    _crop_to_anomaly,
    _oracle_f1,
    _roc_auc,
    evaluate_member,
    run_realdata_benchmark,
)


def _fixture_series(n: int = 320) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic plumbing fixture: a smooth base with two elevated blocks.

    An anomaly block sits in each half so the supervised 50/50-split path is
    measurable. Not random, not a benchmark -- just enough signal to drive the
    plumbing.
    """
    t = np.arange(n, dtype=np.float64)
    series = np.sin(t / 7.0)
    labels = np.zeros(n, dtype=np.int64)
    for lo, hi in ((int(n * 0.28), int(n * 0.34)), (int(n * 0.72), int(n * 0.78))):
        series[lo:hi] += 5.0
        labels[lo:hi] = 1
    return series, labels


def test_roc_auc_rank_identity() -> None:
    """Perfect separation -> 1.0; single-class -> 0.5 (AUC undefined)."""
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert abs(_roc_auc(scores, labels) - 1.0) < 1e-9
    assert abs(_roc_auc(scores[::-1], labels) - 0.0) < 1e-9
    assert _roc_auc(scores, np.zeros(4, dtype=np.int64)) == 0.5


def test_oracle_f1_separable() -> None:
    """A separable score set yields a perfect oracle F1 at some threshold."""
    scores = np.array([0.05, 0.1, 0.9, 0.95])
    labels = np.array([0, 0, 1, 1])
    result = _oracle_f1(scores, labels)
    assert result["f1"] == 1.0
    assert 0.0 <= result["threshold"] <= 1.0
    # Single-class labels are non-measurable -> zeroed, never raising.
    assert _oracle_f1(scores, np.zeros(4, dtype=np.int64))["f1"] == 0.0


def test_crop_preserves_anomaly_window() -> None:
    """A long series crops to max_len while retaining its labelled anomalies."""
    n = 10_000
    series = np.zeros(n, dtype=np.float64)
    labels = np.zeros(n, dtype=np.int64)
    labels[8_000:8_100] = 1
    cropped_series, cropped_labels = _crop_to_anomaly(series, labels, max_len=2_000)
    assert cropped_series.size == 2_000
    assert int(cropped_labels.sum()) == 100


def test_crop_is_noop_when_short() -> None:
    """A series at or under max_len is returned unchanged."""
    series = np.arange(100.0)
    labels = np.zeros(100, dtype=np.int64)
    labels[50] = 1
    cropped_series, cropped_labels = _crop_to_anomaly(series, labels, max_len=6_000)
    assert cropped_series.size == 100
    assert int(cropped_labels.sum()) == 1


def test_evaluate_member_returns_valid_metrics() -> None:
    """A member scores the fixture and returns bounded metrics (or an error dict)."""
    series, labels = _fixture_series()
    result = evaluate_member("spectral_residual", series, labels)
    assert result["detector"] == "spectral_residual"
    if "error" not in result:
        assert 0.0 <= result["roc_auc"] <= 1.0
        assert 0.0 <= result["f1"] <= 1.0
        assert result["latency_ms"] >= 0.0


def test_run_realdata_benchmark_structure_network_free() -> None:
    """The full run over an in-memory fixture yields the expected nested shape."""
    series, labels = _fixture_series()
    results = run_realdata_benchmark(
        datasets=[("fixture", series, labels)],
        members=["spectral_residual", "energy_based"],
    )

    assert {"config", "datasets", "aggregate", "summary", "measured_datasets", "skipped"} <= set(
        results
    )
    assert results["config"]["source"].startswith("NAB")
    assert results["measured_datasets"] == ["fixture"]

    dataset = results["datasets"]["fixture"]
    assert {"n", "anomaly_rate", "best_member", "ensemble_average_auc", "supervised"} <= set(
        dataset
    )
    # Average-ensemble AUC is a bounded float or a trapped None.
    avg_auc = dataset["ensemble_average_auc"]
    assert avg_auc is None or 0.0 <= avg_auc <= 1.0
    # The supervised row always reports whether the split was measurable.
    assert "measurable" in dataset["supervised"]

    # Aggregates and summary are present and internally consistent.
    assert set(ENSEMBLE_METHODS) == {"average", "bma", "stacking"}
    assert "members" in results["aggregate"]
    assert "ensemble_average" in results["aggregate"]
    assert results["summary"]["n_datasets"] == 1


def test_nab_loader_exposes_iter_series() -> None:
    """The streaming 1-D accessor the benchmark depends on exists (no network)."""
    from omni_mercury_engine.datasets.timeseries import NABLoader

    assert hasattr(NABLoader, "iter_series")
    assert callable(NABLoader.iter_series)
