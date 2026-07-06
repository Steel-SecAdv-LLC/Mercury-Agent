# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the survival/hazard (Kaplan-Meier + Cox) detector."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.survival import SurvivalHazardDetector


def _series_with_spike(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, 500)
    x[250] += 10.0
    return x.astype(np.float64)


class TestContract:
    def test_unfitted_then_fitted(self) -> None:
        det = SurvivalHazardDetector()
        assert det.is_fitted() is False
        det.fit(np.random.default_rng(0).normal(size=300))
        assert det.is_fitted() is True

    def test_scores_shape_and_range(self) -> None:
        x = _series_with_spike()
        scores = np.asarray(SurvivalHazardDetector().fit(x).detect(x)["scores"])
        assert scores.shape == (len(x),)
        assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0

    def test_extract_features_shape(self) -> None:
        x = _series_with_spike()
        feats = SurvivalHazardDetector().fit(x).extract_features(x)
        assert feats.shape == (len(x), 1)
        assert np.isfinite(feats).all()

    def test_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            SurvivalHazardDetector(covariate_window=0)
        with pytest.raises(ValueError):
            SurvivalHazardDetector(calibration_quantile=1.0)

    def test_deterministic(self) -> None:
        x = _series_with_spike(7)
        a = SurvivalHazardDetector().fit(x).detect(x)["scores"]
        b = SurvivalHazardDetector().fit(x).detect(x)["scores"]
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


class TestSignal:
    def test_spike_flagged(self) -> None:
        train = np.random.default_rng(1).normal(size=500)
        x = _series_with_spike(2)
        scores = np.asarray(SurvivalHazardDetector().fit(train).detect(x)["scores"])
        assert scores[250] > 0.5
        assert scores[250] > np.median(scores) + 0.3

    def test_low_false_positive_rate(self) -> None:
        train = np.random.default_rng(3).normal(size=800)
        test = np.random.default_rng(4).normal(size=800)
        scores = np.asarray(
            SurvivalHazardDetector(calibration_quantile=0.98).fit(train).detect(test)["scores"]
        )
        assert (scores > 0.5).mean() < 0.08
