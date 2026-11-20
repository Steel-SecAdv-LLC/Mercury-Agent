# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the class-conditional binary conformal classifier (``omni_mercury_engine.core.conformal_prediction.BinaryConformalClassifier``).

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


def _synthetic_probs(n: int, seed: int, anomaly_rate: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
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
        assert (
            a.coverage_report(probs, y)["thresholds"] == b.coverage_report(probs, y)["thresholds"]
        )


class TestStateRoundTrip:
    """export_state()/from_state() reproduce the full post-fit read surface."""

    def test_round_trip_reproduces_serving_surface(self) -> None:
        probs, y = _synthetic_probs(2000, seed=3)
        original = BinaryConformalClassifier(coverage=0.9).fit(probs[:1000], y[:1000])
        restored = BinaryConformalClassifier.from_state(original.export_state())

        assert restored.coverage == original.coverage
        assert restored.seed == original.seed
        assert restored.anomaly_score_threshold() == original.anomaly_score_threshold()

        grid = np.linspace(0.0, 1.0, 201)
        sets_a, sets_b = original.predict(grid), restored.predict(grid)
        np.testing.assert_array_equal(sets_a.contains_anomaly, sets_b.contains_anomaly)
        np.testing.assert_array_equal(sets_a.contains_normal, sets_b.contains_normal)
        assert restored.coverage_report(probs[1000:], y[1000:]) == original.coverage_report(
            probs[1000:], y[1000:]
        )

    def test_export_is_checkpoint_safe_primitives(self) -> None:
        """The mapping matches the ``conformal_state`` checkpoint layout."""
        probs, y = _synthetic_probs(800, seed=11)
        state = BinaryConformalClassifier(coverage=0.85).fit(probs, y).export_state()
        assert set(state) == {"coverage", "coverage_achieved", "seed", "thresholds"}
        assert isinstance(state["coverage"], float)
        assert isinstance(state["coverage_achieved"], float)
        assert isinstance(state["seed"], int)
        for label, threshold in state["thresholds"].items():
            assert type(label) is int and type(threshold) is float

    def test_export_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            BinaryConformalClassifier().export_state()

    def test_from_state_coerces_string_labels(self) -> None:
        """A JSON round-trip of the mapping (str-keyed thresholds) loads equally."""
        probs, y = _synthetic_probs(800, seed=13)
        state = BinaryConformalClassifier(coverage=0.9).fit(probs, y).export_state()
        json_like = dict(state, thresholds={str(k): v for k, v in state["thresholds"].items()})
        restored = BinaryConformalClassifier.from_state(json_like)
        assert restored._thresholds == {int(k): v for k, v in state["thresholds"].items()}
        assert (
            restored.anomaly_score_threshold()
            == BinaryConformalClassifier.from_state(state).anomaly_score_threshold()
        )


class TestHonestSmallSampleCoverage:
    """Fix 6: a clamped quantile must report the ACHIEVED bound, not the request.

    With ``n`` calibration rows per class, the split-conformal quantile rank is
    clamped to ``n`` whenever ``ceil((n + 1) * coverage) > n``; the guarantee
    actually achieved is then ``n / (n + 1)`` -- e.g. ~0.833 for ``n = 5`` --
    while the certificate used to claim the requested 0.9.
    """

    def test_n5_per_class_certificate_reports_achieved_bound(self) -> None:
        # 5 calibration rows per class: requested 0.9 is unattainable.
        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.8, 0.9, 0.7, 0.6, 0.85])
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        clf = BinaryConformalClassifier(coverage=0.9).fit(probs, labels)

        achieved = 5.0 / 6.0
        assert clf.coverage_achieved == pytest.approx(achieved)

        pred = clf.predict(np.array([0.5, 0.9]))
        assert pred.coverage_level == pytest.approx(achieved), (
            "certificate claims the requested coverage although only "
            f"{achieved:.3f} is guaranteed with 5 calibration rows per class"
        )
        report = clf.coverage_report(probs, labels)
        assert report["coverage_achieved"] == pytest.approx(achieved)

    def test_large_calibration_reports_the_requested_level(self) -> None:
        probs, y = _synthetic_probs(2000, seed=3)
        clf = BinaryConformalClassifier(coverage=0.9).fit(probs, y)
        assert clf.coverage_achieved == pytest.approx(0.9)
        assert clf.predict(probs[:5]).coverage_level == pytest.approx(0.9)

    def test_small_sample_fit_logs_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.8, 0.9, 0.7, 0.6, 0.85])
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        with caplog.at_level(
            logging.WARNING, logger="omni_mercury_engine.core.conformal_prediction"
        ):
            BinaryConformalClassifier(coverage=0.9).fit(probs, labels)
        assert any(
            "coverage" in rec.message.lower() for rec in caplog.records
        ), "an unattainable requested coverage must be visible in a WARNING"

    def test_export_state_round_trips_achieved_coverage(self) -> None:
        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.8, 0.9, 0.7, 0.6, 0.85])
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        clf = BinaryConformalClassifier(coverage=0.9).fit(probs, labels)
        rebuilt = BinaryConformalClassifier.from_state(clf.export_state())
        assert rebuilt.coverage_achieved == pytest.approx(clf.coverage_achieved)
        assert rebuilt.predict(np.array([0.5])).coverage_level == pytest.approx(
            clf.coverage_achieved
        )

    def test_decider_wording_uses_the_achieved_coverage(self) -> None:
        # End-to-end: the decision layer's "at NN% coverage" reason must carry
        # the achieved bound once the certificate does.
        from omni_mercury_engine.decision import DecisionAbstentionResponder

        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.8, 0.9, 0.7, 0.6, 0.85])
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        clf = BinaryConformalClassifier(coverage=0.9).fit(probs, labels)
        pred = clf.predict(np.array([0.95]))
        result = {
            "anomaly_prob": 0.95,
            "is_anomaly": True,
            "threshold_used": 0.5,
            "conformal": {
                "prediction_set": pred.label_sets()[0],
                "set_size": int(pred.set_size[0]),
                "coverage": float(pred.coverage_level),
            },
        }
        rec = DecisionAbstentionResponder().decide(result)
        assert rec.coverage == pytest.approx(5.0 / 6.0)
        assert any("83%" in reason for reason in rec.reasons), rec.reasons
