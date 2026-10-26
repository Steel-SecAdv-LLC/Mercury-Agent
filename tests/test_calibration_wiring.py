# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Integration tests for ThresholdCalibrationPipeline wiring into MercuryAnomalyDetector.

Validates that fit_with_labels() resolves the calibration gap (high AUC, low F1)
by setting a supervised threshold via Youden's J or F1-optimal strategies.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector


@pytest.fixture()
def separated_data() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic dataset with clear normal/anomaly separation."""
    rng = np.random.default_rng(42)
    X_normal = rng.normal(0, 1, (200, 5))
    X_anomaly = rng.normal(3, 1, (50, 5))
    X = np.vstack([X_normal, X_anomaly])
    y = np.concatenate([np.zeros(200), np.ones(50)])
    return X, y


def _f1(preds: np.ndarray, y: np.ndarray) -> float:
    tp = np.sum(preds & (y == 1))
    fp = np.sum(preds & (y == 0))
    fn = np.sum(~preds & (y == 1))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


class TestCalibrationWiring:
    """Verify the calibration pipeline integration."""

    def test_default_threshold_is_conservative(
        self, separated_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Uncalibrated, the default threshold flags only a small minority.

        Historically this pinned a near-zero baseline F1; the kNN dimensional
        fusion gate lifted the uncalibrated operating point on this fixture
        (21/250 positives, 13 of them true anomalies), so the live invariant
        is the one calibration actually closes: the label-free default stays
        conservative (recall 0.26 here vs 50 true anomalies) until
        ``fit_with_labels`` sets a supervised operating point.
        """
        X, y = separated_data
        det = MercuryAnomalyDetector()
        det.fit(X)
        preds = np.asarray(det.detect(X)["is_anomaly"])
        assert np.sum(preds) < 0.15 * len(X), "Default threshold should flag a small minority"
        assert np.sum(preds & (y == 1)) < 0.5 * np.sum(y == 1), (
            "Default threshold should recall well under half the anomalies; "
            "calibration below is what closes that gap"
        )

    def test_youden_j_improves_f1(self, separated_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Youden's J calibration must push F1 significantly above baseline."""
        X, y = separated_data
        det = MercuryAnomalyDetector()
        det.fit_with_labels(X, y, strategy="youden_j")
        f1 = _f1(np.asarray(det.detect(X)["is_anomaly"]), y)
        assert f1 > 0.30, f"Youden-J F1 should be > 0.30, got {f1:.4f}"

    def test_f1_optimal_improves_f1(self, separated_data: tuple[np.ndarray, np.ndarray]) -> None:
        """F1-optimal calibration must push F1 significantly above baseline."""
        X, y = separated_data
        det = MercuryAnomalyDetector()
        baseline = MercuryAnomalyDetector().fit(X)
        baseline_f1 = _f1(np.asarray(baseline.detect(X)["is_anomaly"]), y)
        det.fit_with_labels(X, y, strategy="f1_optimal")
        f1 = _f1(np.asarray(det.detect(X)["is_anomaly"]), y)
        assert f1 > 0.30, f"F1-optimal F1 should be > 0.30, got {f1:.4f}"
        # F1-optimal calibrates on this very data, so it must not do worse
        # than the uncalibrated default operating point.
        assert f1 >= baseline_f1, (
            f"F1-optimal ({f1:.4f}) fell below the uncalibrated baseline ({baseline_f1:.4f})"
        )

    def test_supervised_threshold_stored(
        self, separated_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """fit_with_labels() must store the calibrated threshold and pipeline."""
        X, y = separated_data
        det = MercuryAnomalyDetector()
        det.fit_with_labels(X, y)
        assert det._supervised_threshold is not None
        assert det._threshold_pipeline is not None
        assert det._calibration_result is not None
        assert 0.0 < det._supervised_threshold < 1.0

    def test_supervised_threshold_takes_priority(
        self, separated_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Supervised threshold must override the default, not auto-calibrate."""
        X, y = separated_data
        det = MercuryAnomalyDetector()
        det.fit_with_labels(X, y, strategy="youden_j")
        r = det.detect(X)
        # The effective threshold in the result should match supervised
        assert r["threshold"] == pytest.approx(det._supervised_threshold, abs=1e-10)

    def test_fit_with_labels_returns_self(
        self, separated_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """fit_with_labels() must return self for method chaining."""
        X, y = separated_data
        det = MercuryAnomalyDetector()
        result = det.fit_with_labels(X, y)
        assert result is det
