# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the Spectral-Residual saliency detector.

These verify the :class:`~omni_mercury_engine.core.base.BaseDetector` contract
(``fit``/``is_fitted``/``extract_features``/``detect`` shapes and ranges) and
that the SR saliency signal actually localises an injected spike against a
noisy background while keeping the normal-regime false-positive rate near the
``calibration_quantile`` budget.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.spectral_residual import SpectralResidualDetector


def _series_with_spike(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, 600)
    labels = np.zeros(600, dtype=int)
    x[300:305] += 8.0
    labels[300:305] = 1
    return x.astype(np.float64), labels


class TestContract:
    def test_unfitted_then_fitted(self) -> None:
        det = SpectralResidualDetector()
        assert det.is_fitted() is False
        det.fit(np.random.default_rng(0).normal(size=400))
        assert det.is_fitted() is True

    def test_detect_scores_shape_and_range(self) -> None:
        x, _ = _series_with_spike()
        out = SpectralResidualDetector().fit(x).detect(x)
        scores = np.asarray(out["scores"])
        assert scores.shape == (len(x),)
        assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0

    def test_extract_features_shape(self) -> None:
        x, _ = _series_with_spike()
        feats = SpectralResidualDetector().fit(x).extract_features(x)
        assert feats.shape == (len(x), 1)
        assert np.isfinite(feats).all()

    def test_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            SpectralResidualDetector(window_amp=0)
        with pytest.raises(ValueError):
            SpectralResidualDetector(calibration_quantile=1.0)

    def test_constant_input_no_crash(self) -> None:
        # A degenerate constant series must not divide by zero or emit NaNs.
        const = np.full(200, 3.0)
        out = SpectralResidualDetector().fit(const).detect(const)
        assert np.isfinite(np.asarray(out["scores"])).all()

    def test_detect_before_fit_uses_self_scale(self) -> None:
        x, _ = _series_with_spike()
        out = SpectralResidualDetector().detect(x)
        scores = np.asarray(out["scores"])
        assert scores.shape == (len(x),)
        assert np.isfinite(scores).all()


class TestSignal:
    def test_spike_scores_above_background(self) -> None:
        x, labels = _series_with_spike()
        det = SpectralResidualDetector().fit(np.random.default_rng(1).normal(size=600))
        scores = np.asarray(det.detect(x)["scores"])
        assert scores[300:305].max() > 0.5
        # The spike neighbourhood must rank far above the median background.
        assert scores[300:305].max() > 5.0 * (np.median(scores) + 1e-6)

    def test_low_false_positive_rate_on_normal(self) -> None:
        train = np.random.default_rng(2).normal(size=800)
        test = np.random.default_rng(3).normal(size=800)
        det = SpectralResidualDetector(calibration_quantile=0.98).fit(train)
        scores = np.asarray(det.detect(test)["scores"])
        # ~2% budget; allow generous headroom for finite-sample variation.
        assert (scores > 0.5).mean() < 0.08
