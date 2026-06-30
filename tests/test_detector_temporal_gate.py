# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the Ljung-Box temporal-structure gate + per-component calibration (issue #6). The Kinematic component assumes temporal ordering; gate it behind a real white-noise-null significance test (not just a magnitude heuristic), and calibrate each component to [0,1] via isotonic on a labeled holdout."""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.core.config import DataCharacteristics
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector


def _ar1(n: int, phi: float, d: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cols = []
    for j in range(d):
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t - 1] + rng.normal(0, 1)
        cols.append(x)
    return np.column_stack(cols)


class TestLjungBox:
    def test_white_noise_not_significant(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 500)
        _, p = MercuryAnomalyDetector._ljung_box(x, lags=10)
        assert p > 0.05  # fails to reject white-noise null

    def test_autocorrelated_is_significant(self) -> None:
        x = _ar1(500, phi=0.7, d=1, seed=1)[:, 0]
        q, p = MercuryAnomalyDetector._ljung_box(x, lags=10)
        assert p < 0.05  # rejects white noise
        assert q > 0.0


class TestTemporalGate:
    def test_genuine_ar_is_temporal(self) -> None:
        det = MercuryAnomalyDetector()
        dtype = det._detect_data_characteristics(_ar1(400, phi=0.7, d=4, seed=2))
        assert dtype == DataCharacteristics.TEMPORAL

    def test_high_adjacency_white_noise_is_not_temporal(self) -> None:
        # Rows all share a fixed base vector (high adjacency -> old magnitude
        # heuristic would call this TEMPORAL) but each column is white noise
        # across rows (Ljung-Box fails to reject) -> the real test says TABULAR.
        rng = np.random.default_rng(3)
        base = rng.normal(0, 5, 8)
        X = base[None, :] + rng.normal(0, 0.1, (400, 8))
        det = MercuryAnomalyDetector()
        assert det._detect_data_characteristics(X) != DataCharacteristics.TEMPORAL
        # With the gate disabled, the legacy magnitude heuristic still fires.
        det_legacy = MercuryAnomalyDetector(config={"kinematic_temporal_gate": False})
        assert det_legacy._detect_data_characteristics(X) == DataCharacteristics.TEMPORAL

    def test_shuffled_ar_is_not_temporal(self) -> None:
        X = _ar1(400, phi=0.7, d=4, seed=4)
        rng = np.random.default_rng(5)
        X_shuffled = X[rng.permutation(len(X))]
        det = MercuryAnomalyDetector()
        assert det._detect_data_characteristics(X_shuffled) != DataCharacteristics.TEMPORAL


class TestComponentCalibration:
    def _labeled_data(self, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        normal = rng.normal(0, 1, (200, 5))
        anom = rng.normal(0, 1, (40, 5)) + 6.0
        X = np.vstack([normal, anom])
        y = np.array([0] * 200 + [1] * 40)
        return X, y

    def test_default_off_no_calibrators(self) -> None:
        X, y = self._labeled_data()
        det = MercuryAnomalyDetector()
        det.fit(X, calibration_labels=y)
        assert det._component_calibrators is None  # default-off

    def test_enabled_fits_calibrators_and_detects(self) -> None:
        X, y = self._labeled_data()
        det = MercuryAnomalyDetector(config={"component_calibration": True})
        det.fit(X, calibration_labels=y)
        assert det._component_calibrators is not None
        assert set(det._component_calibrators) == {"resonance", "kinematic", "info_geometry"}
        result = det.detect(X)
        scores = np.asarray(result["scores"])
        assert np.all((scores >= 0.0) & (scores <= 1.0))

    def test_calibration_preserves_component_auroc(self) -> None:
        X, y = self._labeled_data(seed=1)
        det = MercuryAnomalyDetector(config={"component_calibration": True})
        det.fit(X, calibration_labels=y)
        # StrictIsotonic is AUROC-preserving: the calibrated kinematic component
        # keeps the same separation as the raw one.
        raw = det._compute_info_geometry_score(X)
        cal = det._apply_component_calibration("info_geometry", raw)
        auc_raw = MercuryAnomalyDetector._component_separation(raw, y)
        auc_cal = MercuryAnomalyDetector._component_separation(cal, y)
        assert abs(auc_raw - auc_cal) < 0.02
