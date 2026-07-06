# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the SPOT/DSPOT Peaks-Over-Threshold (EVT) detector.

These verify the :class:`~omni_mercury_engine.core.base.BaseDetector` contract,
that the EVT threshold flags genuine extremes, that the risk budget ``q`` bounds
the empirical false-positive rate, and that DSPOT mode (``depth > 0``) copes
with a drifting baseline that would defeat a static threshold.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.spot_evt import SPOTDetector


class TestContract:
    def test_unfitted_then_fitted(self) -> None:
        det = SPOTDetector()
        assert det.is_fitted() is False
        det.fit(np.random.default_rng(0).normal(size=500))
        assert det.is_fitted() is True

    def test_detect_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="must be fit"):
            SPOTDetector().detect(np.random.default_rng(0).normal(size=100))

    def test_features_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="must be fit"):
            SPOTDetector().extract_features(np.random.default_rng(0).normal(size=100))

    def test_scores_shape_and_range(self) -> None:
        rng = np.random.default_rng(1)
        det = SPOTDetector().fit(rng.normal(size=500))
        scores = np.asarray(det.detect(rng.normal(size=500))["scores"])
        assert scores.shape == (500,)
        assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0

    def test_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            SPOTDetector(q=0.0)
        with pytest.raises(ValueError):
            SPOTDetector(init_level=1.5)
        with pytest.raises(ValueError):
            SPOTDetector(depth=-1)

    def test_too_few_calibration_samples(self) -> None:
        with pytest.raises(ValueError, match="at least 10"):
            SPOTDetector().fit(np.arange(5.0))


class TestSignal:
    def test_extreme_values_flagged(self) -> None:
        rng = np.random.default_rng(2)
        train = rng.normal(0.0, 1.0, 2000)
        test = rng.normal(0.0, 1.0, 500)
        test[250] = 12.0  # unambiguous extreme
        det = SPOTDetector(q=1e-3, init_level=0.98).fit(train)
        out = det.detect(test)
        scores = np.asarray(out["scores"])
        assert scores[250] > 0.5
        assert bool(np.asarray(out["is_anomaly"])[250])

    def test_risk_budget_bounds_false_positives(self) -> None:
        rng = np.random.default_rng(3)
        train = rng.normal(0.0, 1.0, 4000)
        test = rng.normal(0.0, 1.0, 4000)  # pure normal regime
        det = SPOTDetector(q=1e-3, init_level=0.98).fit(train)
        scores = np.asarray(det.detect(test)["scores"])
        # EVT control: empirical exceedance stays small (well under 2%).
        assert (scores > 0.5).mean() < 0.02

    def test_dspot_handles_drift(self) -> None:
        rng = np.random.default_rng(4)
        n = 2000
        drift = np.linspace(0.0, 20.0, n)  # strong upward trend
        train = drift + rng.normal(0.0, 1.0, n)
        test = np.linspace(20.0, 40.0, 500) + rng.normal(0.0, 1.0, 500)
        test[250] += 12.0  # spike on top of the trend
        det = SPOTDetector(q=1e-3, init_level=0.98, depth=50).fit(train)
        scores = np.asarray(det.detect(test)["scores"])
        # The spike is caught, and the ramp itself is not one giant alert.
        assert scores[250] > 0.5
        assert (scores > 0.5).mean() < 0.1
