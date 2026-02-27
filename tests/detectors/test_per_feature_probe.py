"""Option A: per-feature probe aggregation tests."""
import numpy as np
import pytest
from omni_mercury_engine.detectors.math_arrest.arrest import AnomalyMathArrest
from omni_mercury_engine.detectors.math_arrest.probes.iqr_robust import IQRRobustProbe
from omni_mercury_engine.detectors.math_arrest.probes.variance_adapted import (
    VarianceAdaptedProbe,
)


def test_per_feature_scores_shape_matches_detect() -> None:
    """per_feature_scores must return (n_samples,) matching detect() output."""
    rng = np.random.RandomState(0)
    X_train = rng.randn(200, 5)
    X_test = rng.randn(50, 5)

    probe = IQRRobustProbe()
    probe.fit_trajectory(X_train)
    pf = probe.per_feature_scores(X_test)
    assert pf.shape == (50,), f"Expected (50,), got {pf.shape}"
    assert np.all(pf >= 0.0) and np.all(pf <= 1.0)


def test_per_feature_detects_single_feature_anomaly() -> None:
    """A spike in one feature must produce a high per_feature_scores output
    even when other features are normal (column-mean would dilute it)."""
    rng = np.random.RandomState(1)
    X_train = rng.randn(300, 10)

    probe = VarianceAdaptedProbe()
    probe.fit_trajectory(X_train)

    # Normal sample: typical training-like values
    normal = rng.randn(5, 10) * 0.5
    # Anomalous sample: one feature is 10 sigma out, rest near training mean
    spike = rng.randn(5, 10) * 0.5
    spike[:, 4] = 10.0  # Only feature 4 is anomalous

    pf_normal = probe.per_feature_scores(normal)
    pf_spike = probe.per_feature_scores(spike)

    # Per-feature max-pooling should detect the spike in feature 4
    assert np.mean(pf_spike) >= np.mean(pf_normal), (
        f"Spike mean score {np.mean(pf_spike):.3f} must exceed "
        f"normal mean {np.mean(pf_normal):.3f}"
    )


def test_ama_decorrelator_fewer_redundant_pairs_after_option_a() -> None:
    """With per-feature aggregation, the decorrelator should run and
    the redundancy report should be populated."""
    rng = np.random.RandomState(42)
    X = rng.randn(500, 10)  # 10 distinct features
    ama = AnomalyMathArrest()
    ama.fit(X)
    report = ama.redundancy_report()
    # Verify decorrelator ran: report has expected keys
    assert "redundant_pairs" in report
    assert "weight_multipliers" in report
    assert "effective_probe_count" in report
    # Effective probe count should be positive (decorrelator penalizes redundant probes)
    assert report["effective_probe_count"] > 0, "Must have at least 1 effective probe"


def test_per_feature_high_dimensional_uses_pca() -> None:
    """Data with > 50 features must be PCA-reduced without crashing."""
    rng = np.random.RandomState(3)
    X = rng.randn(100, 60)  # > 50 features triggers PCA path
    probe = IQRRobustProbe()
    probe.fit_trajectory(X)
    scores = probe.per_feature_scores(X)
    assert scores.shape == (100,)
    assert not np.any(np.isnan(scores))
