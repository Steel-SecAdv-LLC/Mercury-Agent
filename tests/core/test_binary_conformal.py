"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests for the class-conditional binary conformal classifier
(``omni_mercury_engine.core.conformal_prediction.BinaryConformalClassifier``).

The headline test verifies the *distribution-free coverage guarantee*: on
exchangeable synthetic data the empirical fraction of prediction sets that
contain the true label meets the target -- the property the conformal method
promises, measured directly rather than via accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.core.conformal_prediction import (
    BinaryConformalClassifier,
    BinaryPredictionSet,
)


def _synthetic_probs(
    n: int, seed: int, anomaly_rate: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """Overlapping but informative calibrated probabilities and labels."""
    rng = np.random.RandomState(seed)
    y = (rng.random(n) < anomaly_rate).astype(int)
    probs = np.where(
        y == 1,
        np.clip(rng.normal(0.7, 0.2, n), 0.01, 0.99),
        np.clip(rng.normal(0.3, 0.2, n), 0.01, 0.99),
    )
    return probs, y


class TestCoverageGuarantee:
    """Empirical coverage must meet the target on exchangeable data."""

    @pytest.mark.parametrize("coverage", [0.8, 0.9, 0.95])
    def test_marginal_and_per_class_coverage(self, coverage: float) -> None:
        probs, y = _synthetic_probs(6000, seed=0)
        p_cal, y_cal = probs[:3000], y[:3000]
        p_te, y_te = probs[3000:], y[3000:]

        clf = BinaryConformalClassifier(coverage=coverage).fit(p_cal, y_cal)
        report = clf.coverage_report(p_te, y_te)

        # Distribution-free guarantee, with a small finite-sample tolerance.
        assert report["empirical_coverage"] >= coverage - 0.03
        for label in (0, 1):
            assert report["coverage_by_class"][label] >= coverage - 0.05
        # Sets must be informative, not the trivial always-{0,1}.
        assert report["average_set_size"] < 2.0

    def test_higher_coverage_gives_larger_sets(self) -> None:
        probs, y = _synthetic_probs(6000, seed=1)
        p_cal, y_cal = probs[:3000], y[:3000]
        p_te = probs[3000:]
        small = BinaryConformalClassifier(coverage=0.8).fit(p_cal, y_cal)
        large = BinaryConformalClassifier(coverage=0.99).fit(p_cal, y_cal)
        assert large.predict(p_te).set_size.mean() >= small.predict(p_te).set_size.mean()


class TestPredictionSets:
    """Prediction-set structure and helpers."""

    def test_set_sizes_in_range(self) -> None:
        probs, y = _synthetic_probs(2000, seed=2)
        clf = BinaryConformalClassifier(coverage=0.9).fit(probs, y)
        pred = clf.predict(probs)
        assert isinstance(pred, BinaryPredictionSet)
        assert set(np.unique(pred.set_size)).issubset({0, 1, 2})

    def test_confident_anomaly_is_singleton_anomaly(self) -> None:
        probs, y = _synthetic_probs(2000, seed=3)
        clf = BinaryConformalClassifier(coverage=0.9).fit(probs, y)
        pred = clf.predict(np.array([0.999]))
        assert pred.label_sets()[0] == [1]

    def test_label_sets_round_trip(self) -> None:
        probs, y = _synthetic_probs(1000, seed=4)
        clf = BinaryConformalClassifier(coverage=0.9).fit(probs, y)
        pred = clf.predict(probs[:5])
        sets = pred.label_sets()
        assert len(sets) == 5
        for s in sets:
            assert s == sorted(s)
            assert set(s).issubset({0, 1})


class TestValidationAndEdgeCases:
    """Constructor / input validation and degenerate calibration."""

    def test_invalid_coverage_rejected(self) -> None:
        with pytest.raises(ValueError, match="coverage"):
            BinaryConformalClassifier(coverage=1.5)

    def test_mismatched_lengths_rejected(self) -> None:
        clf = BinaryConformalClassifier()
        with pytest.raises(ValueError, match="same length"):
            clf.fit(np.array([0.1, 0.2, 0.3]), np.array([0, 1]))

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            BinaryConformalClassifier().predict(np.array([0.5]))

    def test_absent_class_included_conservatively(self) -> None:
        # No anomalies in calibration -> label 1 always admitted (threshold 1.0).
        probs = np.linspace(0.01, 0.99, 50)
        y = np.zeros(50, dtype=int)
        clf = BinaryConformalClassifier(coverage=0.9).fit(probs, y)
        pred = clf.predict(np.array([0.01, 0.5, 0.99]))
        assert bool(pred.contains_anomaly.all())

    def test_deterministic_thresholds(self) -> None:
        probs, y = _synthetic_probs(1500, seed=5)
        a = BinaryConformalClassifier(coverage=0.9).fit(probs, y)
        b = BinaryConformalClassifier(coverage=0.9).fit(probs, y)
        assert a.coverage_report(probs, y)["thresholds"] == b.coverage_report(probs, y)["thresholds"]
