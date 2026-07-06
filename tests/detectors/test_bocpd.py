# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the Bayesian Online Change-Point Detection (BOCPD) detector.

These verify the :class:`~omni_mercury_engine.core.base.BaseDetector` contract
and that the run-length posterior spikes at a genuine mean shift while staying
low on a stationary stream. Scores are already probabilities, so they must lie
in ``[0, 1]`` without further squashing.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.bocpd import BOCPDDetector


def _mean_shift(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 1.0, 300)
    b = rng.normal(6.0, 1.0, 300)  # abrupt level shift at t=300
    return np.concatenate([a, b]).astype(np.float64)


class TestContract:
    def test_unfitted_then_fitted(self) -> None:
        det = BOCPDDetector()
        assert det.is_fitted() is False
        det.fit(np.random.default_rng(0).normal(size=200))
        assert det.is_fitted() is True

    def test_scores_are_probabilities(self) -> None:
        x = _mean_shift()
        scores = np.asarray(BOCPDDetector().fit(x[:300]).detect(x)["scores"])
        assert scores.shape == (len(x),)
        assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0

    def test_extract_features_shape(self) -> None:
        x = _mean_shift()
        feats = BOCPDDetector().fit(x[:300]).extract_features(x)
        assert feats.shape == (len(x), 1)
        assert np.isfinite(feats).all()

    def test_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            BOCPDDetector(hazard_lambda=1.0)
        with pytest.raises(ValueError):
            BOCPDDetector(change_grace=0)
        with pytest.raises(ValueError):
            BOCPDDetector(change_grace=10, max_run_length=5)

    def test_constant_input_no_crash(self) -> None:
        const = np.full(150, 2.0)
        scores = np.asarray(BOCPDDetector().fit(const).detect(const)["scores"])
        assert np.isfinite(scores).all()


class TestSignal:
    def test_changepoint_detected_at_shift(self) -> None:
        x = _mean_shift(1)
        det = BOCPDDetector(hazard_lambda=200, change_grace=5).fit(x[:300])
        scores = np.asarray(det.detect(x)["scores"])
        # The change-point probability must peak in the window just after the
        # shift, and clearly exceed the stationary-regime background.
        post_shift = scores[300:315].max()
        pre_shift = np.median(scores[50:290])
        assert post_shift > 0.5
        assert post_shift > 5.0 * (pre_shift + 1e-6)

    def test_stationary_stream_stays_low(self) -> None:
        rng = np.random.default_rng(4)
        x = rng.normal(0.0, 1.0, 600)
        det = BOCPDDetector(hazard_lambda=250).fit(rng.normal(0.0, 1.0, 400))
        scores = np.asarray(det.detect(x)["scores"])
        # Ignore the unavoidable warm-up transient at the very start.
        assert np.median(scores[20:]) < 0.2
