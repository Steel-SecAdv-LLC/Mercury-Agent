# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the Hawkes-process event-rate / burst detector.

These verify the :class:`~omni_mercury_engine.core.base.BaseDetector` contract
and that the Hawkes intensity residual flags a burst in a Poisson count stream
while keeping the stationary-regime false-positive rate near the
``calibration_quantile`` budget. Counts are clamped non-negative on ingest.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.hawkes import HawkesBurstDetector


def _counts_with_burst(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    counts = rng.poisson(2.0, 600).astype(np.float64)
    labels = np.zeros(600, dtype=int)
    counts[300:306] += rng.poisson(25.0, 6)  # dense burst
    labels[300:306] = 1
    return counts, labels


class TestContract:
    def test_unfitted_then_fitted(self) -> None:
        det = HawkesBurstDetector()
        assert det.is_fitted() is False
        det.fit(np.random.default_rng(0).poisson(2.0, 400).astype(float))
        assert det.is_fitted() is True

    def test_scores_shape_and_range(self) -> None:
        counts, _ = _counts_with_burst()
        out = HawkesBurstDetector().fit(counts).detect(counts)
        scores = np.asarray(out["scores"])
        assert scores.shape == (len(counts),)
        assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0

    def test_extract_features_shape(self) -> None:
        counts, _ = _counts_with_burst()
        feats = HawkesBurstDetector().fit(counts).extract_features(counts)
        assert feats.shape == (len(counts), 1)
        assert np.isfinite(feats).all()

    def test_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            HawkesBurstDetector(beta=0.0)
        with pytest.raises(ValueError):
            HawkesBurstDetector(alpha=-1.0)
        with pytest.raises(ValueError):
            HawkesBurstDetector(calibration_quantile=0.0)

    def test_negative_inputs_clamped(self) -> None:
        # Counts are non-negative; negative inputs must not crash or emit NaNs.
        x = np.array([-5.0, 1.0, 2.0, -3.0, 4.0] * 20)
        scores = np.asarray(HawkesBurstDetector().fit(x).detect(x)["scores"])
        assert np.isfinite(scores).all()


class TestSignal:
    def test_burst_flagged(self) -> None:
        train = np.random.default_rng(1).poisson(2.0, 600).astype(float)
        counts, labels = _counts_with_burst(2)
        det = HawkesBurstDetector().fit(train)
        scores = np.asarray(det.detect(counts)["scores"])
        assert scores[300:306].max() > 0.5
        assert scores[300:306].max() > np.median(scores[:290]) + 0.3

    def test_low_false_positive_rate(self) -> None:
        train = np.random.default_rng(3).poisson(3.0, 800).astype(float)
        test = np.random.default_rng(4).poisson(3.0, 800).astype(float)
        det = HawkesBurstDetector(calibration_quantile=0.98).fit(train)
        scores = np.asarray(det.detect(test)["scores"])
        assert (scores > 0.5).mean() < 0.08
