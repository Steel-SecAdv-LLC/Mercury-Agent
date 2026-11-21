# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``calibrate_detector`` reports held-out calibration quality, not a refit score.

It used to fit the calibrator on the calibration rows and report
``brier_after`` (and with it ``meets_target``) on those same rows. A flexible
calibrator scores well on the rows it was fitted to whether or not it
calibrates anything: isotonic regression on sixty noisy scores reported a
Brier score a fifth below what the same calibrator measured on fresh data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.core import calibration as cal


class _ScoreColumnDetector:
    """A detector whose probability is the first feature column."""

    def predict_proba(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.asarray(X[:, 0], dtype=float)


def _noisy(n: int, seed: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Scores only weakly related to the label: room for a calibrator to overfit."""
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0.0, 1.0, size=n)
    labels = (rng.uniform(0.0, 1.0, size=n) < 0.35 + 0.3 * scores).astype(int)
    return scores, labels


def _brier(labels: np.ndarray[Any, Any], probabilities: np.ndarray[Any, Any]) -> float:
    return float(np.mean((probabilities - labels) ** 2))


SEEDS = tuple(range(12))


def _measure(seed: int, method: str = "isotonic") -> tuple[float, float, float]:
    """(reported brier_after, in-sample brier, brier on 20,000 fresh rows)."""
    scores, labels = _noisy(60, seed)
    calibrator, result = cal.calibrate_detector(
        _ScoreColumnDetector(), scores.reshape(-1, 1), labels, method=method
    )
    assert result is not None
    in_sample = _brier(labels, calibrator.calibrate(scores))
    fresh_scores, fresh_labels = _noisy(20_000, 10_000 + seed)
    fresh = _brier(fresh_labels, calibrator.calibrate(fresh_scores))
    return float(result.brier_after), in_sample, fresh


class TestReportedMetricsAreHeldOut:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_the_reported_brier_is_not_the_in_sample_one(self, seed: int) -> None:
        """Isotonic fits its own rows better than it can be said to calibrate."""
        reported, in_sample, _fresh = _measure(seed)

        assert reported > in_sample + 0.01

    def test_the_reported_brier_is_close_to_a_fresh_holdout(self) -> None:
        """Averaged over datasets, the report matches fresh data; in-sample does not."""
        measured = np.array([_measure(seed) for seed in SEEDS])
        reported, in_sample, fresh = measured.mean(axis=0)

        assert abs(reported - fresh) < 0.015
        assert fresh - in_sample > 0.03

    def test_the_result_says_how_it_was_evaluated(self) -> None:
        scores, labels = _noisy(60, 3)

        calibrator, result = cal.calibrate_detector(
            _ScoreColumnDetector(), scores.reshape(-1, 1), labels, method="isotonic"
        )

        assert result is not None
        assert result.evaluation == "cross_fitted"
        assert result.n_folds == 5
        # The optimistic number is still available, under its own name.
        assert result.brier_after_in_sample == pytest.approx(
            _brier(labels, calibrator.calibrate(scores))
        )
        payload = result.to_dict()
        assert payload["evaluation"] == "cross_fitted"
        assert payload["brier_after_in_sample"] == pytest.approx(result.brier_after_in_sample)
        assert payload["meets_target"] == (result.brier_after < 0.05)

    def test_the_returned_calibrator_is_fitted_on_every_row(self) -> None:
        """Cross-fitting is for the report; the calibrator keeps all the data."""
        scores, labels = _noisy(60, 5)

        calibrator, _result = cal.calibrate_detector(
            _ScoreColumnDetector(), scores.reshape(-1, 1), labels, method="isotonic"
        )

        reference = cal.IsotonicCalibration().fit(scores, labels)
        grid = np.linspace(0.0, 1.0, 101)
        np.testing.assert_allclose(calibrator.calibrate(grid), reference.calibrate(grid))

    def test_the_report_is_reproducible(self) -> None:
        scores, labels = _noisy(60, 7)
        detector = _ScoreColumnDetector()

        first = cal.calibrate_detector(detector, scores.reshape(-1, 1), labels, method="platt")[1]
        second = cal.calibrate_detector(detector, scores.reshape(-1, 1), labels, method="platt")[1]

        assert first is not None and second is not None
        assert first.brier_after == second.brier_after

    @pytest.mark.parametrize("method", ["platt", "isotonic", "strict_isotonic", "temperature"])
    def test_every_method_reports_cross_fitted(self, method: str) -> None:
        scores, labels = _noisy(120, 11)

        _calibrator, result = cal.calibrate_detector(
            _ScoreColumnDetector(), scores.reshape(-1, 1), labels, method=method
        )

        assert result is not None
        assert result.method == method
        assert result.evaluation == "cross_fitted"
        assert 0.0 <= result.brier_after <= 1.0

    def test_auto_reports_cross_fitted(self) -> None:
        scores, labels = _noisy(200, 13)

        calibrator, result = cal.calibrate_detector(
            _ScoreColumnDetector(), scores.reshape(-1, 1), labels, method="auto"
        )

        assert result is not None
        assert result.method == calibrator.best_method
        assert result.evaluation == "cross_fitted"


class TestWhenNothingCanBeHeldOut:
    def test_one_positive_row_is_reported_as_in_sample(self) -> None:
        """A class with a single member cannot be both fitted on and held out."""
        scores = np.linspace(0.05, 0.95, 30)
        labels = np.zeros(30, dtype=int)
        labels[-1] = 1

        _calibrator, result = cal.calibrate_detector(
            _ScoreColumnDetector(), scores.reshape(-1, 1), labels, method="isotonic"
        )

        assert result is not None
        assert result.evaluation == "in_sample"
        assert result.n_folds is None
        assert result.brier_after == result.brier_after_in_sample

    def test_evaluate_calibration_does_not_claim_a_protocol(self) -> None:
        """Arrays handed to ``evaluate_calibration`` are evaluated as supplied."""
        scores, labels = _noisy(80, 17)

        result = cal.evaluate_calibration(labels, scores, scores, method="identity")

        assert result.evaluation == "as_supplied"
        assert "brier_after_in_sample" not in result.to_dict()
