# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the DeepSVDDDetector regime-novelty detector."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.deep_svdd import DeepSVDDDetector


def _series_with_segment(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, 800)
    x[400:500] = rng.normal(3.0, 2.5, size=100)  # sustained regime shift
    return x.astype(np.float64)


class TestContract:
    def test_unfitted_then_fitted(self) -> None:
        det = DeepSVDDDetector()
        assert det.is_fitted() is False
        det.fit(np.random.default_rng(0).normal(size=400))
        assert det.is_fitted() is True

    def test_scores_shape_and_range(self) -> None:
        x = _series_with_segment()
        scores = np.asarray(DeepSVDDDetector().fit(x).detect(x)["scores"])
        assert scores.shape == (len(x),)
        assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0

    def test_extract_features_shape(self) -> None:
        x = _series_with_segment()
        feats = DeepSVDDDetector().fit(x).extract_features(x)
        assert feats.shape == (len(x), 1)
        assert np.isfinite(feats).all()

    def test_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            DeepSVDDDetector(n_features=1)
        with pytest.raises(ValueError):
            DeepSVDDDetector(bandwidth=0.0)

    def test_deterministic(self) -> None:
        x = _series_with_segment(7)
        a = DeepSVDDDetector().fit(x).detect(x)["scores"]
        b = DeepSVDDDetector().fit(x).detect(x)["scores"]
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


class TestSignal:
    def test_segment_flagged(self) -> None:
        train = np.random.default_rng(1).normal(size=800)
        x = _series_with_segment(2)
        scores = np.asarray(DeepSVDDDetector().fit(train).detect(x)["scores"])
        seg = scores[400:500]
        assert float(seg.max()) > 0.5
        assert (seg > 0.5).mean() > 0.3

    def test_low_false_positive_rate(self) -> None:
        train = np.random.default_rng(3).normal(size=800)
        test = np.random.default_rng(4).normal(size=800)
        scores = np.asarray(
            DeepSVDDDetector(calibration_quantile=0.98).fit(train).detect(test)["scores"]
        )
        assert (scores > 0.5).mean() < 0.1
