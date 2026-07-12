# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the real PyOD comparison layer.

The layer under test actually runs PyOD detectors, so most tests skip cleanly
when the optional ``pyod`` package is absent (``pip install
mercury-agent[benchmark]``). The end-to-end tests run on a *real* ADBench
dataset fixture (``wine``, the smallest Classical set) served from the local
dataset cache; when the NPZ is not cached the fixture downloads it, and if
that requires network in an offline environment the test skips rather than
fabricating data.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from omni_mercury_engine.comparison import CombinationMethod, PyODAlgorithm, PyODComparison
from omni_mercury_engine.comparison.pyod_integration import (
    DEFAULT_BASELINES,
    build_pyod_detector,
    pyod_available,
    run_pyod_baselines,
)

requires_pyod = pytest.mark.skipif(
    not pyod_available(), reason="pyod not installed (pip install mercury-agent[benchmark])"
)


# ---------------------------------------------------------------------------
# Real ADBench fixture (wine: 129 samples x 13 features, ground-truth labels)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def adbench_wine() -> tuple[np.ndarray, np.ndarray]:
    """Real ADBench 'wine' dataset (cached locally; skip if unobtainable).

    Deterministic-by-default: if the NPZ is not already cached AND network tests
    are not explicitly enabled (``MERCURY_NETWORK_TESTS=1``), skip *immediately*
    rather than reaching for the network -- so the default per-PR run neither
    stalls on a download nor depends on external availability. The ``except`` is
    narrowed to the expected offline/uncached failure modes so a genuine loader
    regression (a parsing bug, an unexpected ``ValueError``) still fails loudly
    instead of masquerading as an "offline" skip.

    When ``MERCURY_NETWORK_TESTS=1`` is explicitly set the network CI lane is
    *supposed* to exercise the real dataset path, so a download/load failure
    there is a real regression (broken source / caching) and is re-raised to
    FAIL the test -- only the default offline run skips.
    """
    from omni_mercury_engine.datasets.adbench import ADBenchLoader
    from omni_mercury_engine.datasets.base import DatasetConfig
    from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError

    net_enabled = os.environ.get("MERCURY_NETWORK_TESTS") == "1"
    loader = ADBenchLoader(DatasetConfig(name="adbench", preprocessing={"dataset": "wine"}))
    cached = (loader.data_path / loader.npz_filename).exists()
    if not cached and not net_enabled:
        pytest.skip(
            "ADBench 'wine' NPZ not cached and MERCURY_NETWORK_TESTS != 1 "
            "(offline-deterministic default; set MERCURY_NETWORK_TESTS=1 to fetch)"
        )
    try:
        loader.download()  # no-op when cached
        X, y = loader._load_raw()
    except (DataSourceUnavailableError, FileNotFoundError, OSError) as exc:  # offline/uncached
        if net_enabled:
            # The network lane exists to catch exactly this -- a real failure of
            # the dataset path must not be swallowed as an "offline" skip.
            raise
        pytest.skip(f"ADBench wine unavailable (offline, not cached): {exc}")
    return X.astype(np.float64), (y > 0).astype(int)


@pytest.fixture(scope="module")
def wine_split(
    adbench_wine: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normal-only train half + shuffled labelled test rows (fixed seed 42)."""
    X, y = adbench_wine
    rng = np.random.RandomState(42)
    normal = X[y == 0]
    idx = rng.choice(len(normal), len(normal) // 2, replace=False)
    mask = np.ones(len(normal), dtype=bool)
    mask[idx] = False
    X_test = np.vstack([normal[mask], X[y == 1]])
    y_test = np.concatenate(
        [np.zeros(int(mask.sum()), dtype=int), np.ones(int(y.sum()), dtype=int)]
    )
    perm = np.random.RandomState(42).permutation(len(X_test))
    return normal[idx], X_test[perm], y_test[perm]


# ---------------------------------------------------------------------------
# Availability / construction
# ---------------------------------------------------------------------------


def test_default_baselines_are_the_standard_cheap_set() -> None:
    """The default set is the documented CPU-fair sextet, deep nets excluded."""
    values = {a.value for a in DEFAULT_BASELINES}
    assert values == {
        "isolation_forest",
        "ecod",
        "copod",
        "local_outlier_factor",
        "knn",
        "hbos",
    }
    assert PyODAlgorithm.AUTOENCODER not in DEFAULT_BASELINES


@requires_pyod
def test_build_pyod_detector_constructs_every_default() -> None:
    for algo in DEFAULT_BASELINES:
        detector = build_pyod_detector(algo, seed=42)
        assert hasattr(detector, "fit") and hasattr(detector, "decision_function")


def test_build_pyod_detector_rejects_non_algorithm() -> None:
    with pytest.raises((ValueError, ImportError)):
        build_pyod_detector("not_an_algorithm")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# run_pyod_baselines on the real fixture
# ---------------------------------------------------------------------------


@requires_pyod
def test_run_pyod_baselines_scores_every_default(
    wine_split: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X_train, X_test, _y_test = wine_split
    results = run_pyod_baselines(X_train, X_test, seed=42)
    assert set(results) == {a.value for a in DEFAULT_BASELINES}
    for name, run in results.items():
        assert "error" not in run, f"{name}: {run.get('error')}"
        assert run["scores"].shape == (len(X_test),)
        assert np.isfinite(run["scores"]).all(), f"{name} emitted non-finite scores"
        assert run["fit_seconds"] >= 0.0 and run["score_seconds"] >= 0.0


@requires_pyod
def test_run_pyod_baselines_is_deterministic_for_fixed_seed(
    wine_split: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X_train, X_test, _y_test = wine_split
    first = run_pyod_baselines(X_train, X_test, seed=42)
    second = run_pyod_baselines(X_train, X_test, seed=42)
    for name in first:
        np.testing.assert_array_equal(
            first[name]["scores"], second[name]["scores"], err_msg=f"{name} not deterministic"
        )


@requires_pyod
def test_run_pyod_baselines_beats_random_on_real_labels(
    wine_split: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """On the real wine fixture the standard baselines must show real skill.

    This is the non-vacuity check for the layer: if scores were mis-aligned,
    inverted, or shuffled, ROC-AUC would collapse toward 0.5.
    """
    from omni_mercury_engine.ml.mercury_ml import roc_auc_score

    X_train, X_test, y_test = wine_split
    results = run_pyod_baselines(X_train, X_test, seed=42)
    aucs = {n: roc_auc_score(y_test, r["scores"]) for n, r in results.items()}
    # wine is an easy benchmark set; every default baseline clears 0.6 and
    # the best clears 0.9 (measured: knn/lof > 0.97).
    assert max(aucs.values()) > 0.9, aucs
    assert all(a > 0.6 for a in aucs.values()), aucs


@requires_pyod
def test_run_pyod_baselines_records_failures_not_drops(
    wine_split: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """A baseline that cannot run is recorded with an error, never dropped."""
    X_train, X_test, _y_test = wine_split
    # LOF cannot fit on a single row -> per-algorithm error entry.
    results = run_pyod_baselines(X_train[:1], X_test, algorithms=[PyODAlgorithm.LOF], seed=42)
    assert set(results) == {"local_outlier_factor"}
    assert "error" in results["local_outlier_factor"]


def test_run_pyod_baselines_raises_actionable_import_error_when_missing() -> None:
    if pyod_available():
        pytest.skip("pyod installed; ImportError path not reachable")
    with pytest.raises(ImportError, match="mercury-agent\\[benchmark\\]"):
        run_pyod_baselines(np.zeros((4, 2)), np.zeros((2, 2)))


# ---------------------------------------------------------------------------
# PyODComparison.benchmark_against_pyod (the real end-to-end comparison)
# ---------------------------------------------------------------------------


@requires_pyod
def test_benchmark_against_pyod_end_to_end(
    wine_split: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X_train, X_test, y_test = wine_split
    # Mercury stand-in: any score vector aligned to X_test. Use the real tier
    # detector so this exercises the same integration the benchmark uses.
    from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

    detector = MercuryAnomalyDetector()
    detector.fit(X_train)
    # Match the benchmark's eval path exactly: competitive_benchmark._run_mercury_tier
    # plants this runtime-only domain marker, which routes detect() through the
    # same "adbench" blending. Without it this test would exercise a different path.
    detector._benchmark_domain = "adbench"  # type: ignore[attr-defined]
    mercury_scores = np.asarray(detector.detect(X_test)["scores"], dtype=np.float64)

    comparison = PyODComparison()
    results = comparison.benchmark_against_pyod(
        {"mercury_tier": mercury_scores}, X_train, X_test, y_test, seed=42
    )

    assert set(results) == {"mercury", "pyod", "comparison_summary"}
    assert 0.0 <= results["mercury"]["mercury_tier"]["roc_auc"] <= 1.0
    assert set(results["pyod"]) == {a.value for a in DEFAULT_BASELINES}
    for name, metrics in results["pyod"].items():
        assert "error" not in metrics, f"{name}: {metrics.get('error')}"
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert 0.0 <= metrics["average_precision"] <= 1.0

    summary = results["comparison_summary"]["mercury_tier"]
    assert set(summary["vs"]) == {a.value for a in DEFAULT_BASELINES}
    for entry in summary["vs"].values():
        assert entry["result"] in ("win", "loss", "tie")
        # delta sign must agree with the win/loss verdict
        if entry["result"] == "win":
            assert entry["auc_delta"] > 0
        elif entry["result"] == "loss":
            assert entry["auc_delta"] < 0


@requires_pyod
def test_benchmark_summary_reports_losses_as_losses(
    wine_split: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """A deliberately bad Mercury score vector must be reported as all losses."""
    X_train, X_test, y_test = wine_split
    rng = np.random.RandomState(0)
    # Anti-signal: the inverse of the labels plus noise loses to everything.
    bad_scores = (1 - y_test).astype(np.float64) + rng.uniform(0, 0.01, size=len(y_test))
    comparison = PyODComparison()
    results = comparison.benchmark_against_pyod(
        {"mercury_bad": bad_scores}, X_train, X_test, y_test, seed=42
    )
    verdicts = [e["result"] for e in results["comparison_summary"]["mercury_bad"]["vs"].values()]
    assert verdicts.count("loss") == len(verdicts), verdicts


# ---------------------------------------------------------------------------
# Score-combination utilities (pure numpy, no pyod needed)
# ---------------------------------------------------------------------------


def test_combine_predictions_average() -> None:
    comparison = PyODComparison()
    predictions = {
        "detector1": np.array([0.1, 0.2, 0.3, 0.4]),
        "detector2": np.array([0.2, 0.3, 0.4, 0.5]),
        "detector3": np.array([0.15, 0.25, 0.35, 0.45]),
    }
    combined = comparison.combine_predictions(predictions, CombinationMethod.AVERAGE)
    assert combined.shape == (4,)
    assert np.allclose(combined, [0.15, 0.25, 0.35, 0.45])


def test_combine_predictions_maximum() -> None:
    comparison = PyODComparison()
    predictions = {
        "detector1": np.array([0.1, 0.2, 0.3, 0.4]),
        "detector2": np.array([0.2, 0.3, 0.4, 0.5]),
    }
    combined = comparison.combine_predictions(predictions, CombinationMethod.MAXIMUM)
    assert combined.shape == (4,)
    assert np.allclose(combined, [0.2, 0.3, 0.4, 0.5])


def test_combine_predictions_aom() -> None:
    comparison = PyODComparison()
    predictions = {
        "detector1": np.array([0.1, 0.2, 0.3, 0.4]),
        "detector2": np.array([0.2, 0.3, 0.4, 0.5]),
        "detector3": np.array([0.15, 0.25, 0.35, 0.45]),
        "detector4": np.array([0.25, 0.35, 0.45, 0.55]),
    }
    combined = comparison.combine_predictions(predictions, CombinationMethod.AOM)
    assert combined.shape == (4,)


def test_algorithm_recommendation() -> None:
    comparison = PyODComparison()
    result = comparison.recommend_algorithm(
        {"num_samples": 1000, "num_features": 10, "has_clusters": False}
    )
    assert result["recommendations"]
    assert "algorithm" in result["recommendations"][0]


def test_algorithm_recommendation_large_dataset() -> None:
    comparison = PyODComparison()
    result = comparison.recommend_algorithm(
        {"num_samples": 200000, "num_features": 50}, {"max_time_seconds": 30}
    )
    assert result["recommendations"][0]["algorithm"] == PyODAlgorithm.COPOD
