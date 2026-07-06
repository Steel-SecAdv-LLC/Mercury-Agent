# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the SR-CNN saliency-discriminator detector."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.detectors.srcnn import SRCNNDetector


def _series(seed: int = 0, n: int = 400) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=n)
    return x


class TestContract:
    def test_unfitted_then_fitted(self) -> None:
        det = SRCNNDetector(epochs=3)
        assert det.is_fitted() is False
        det.fit(_series())
        assert det.is_fitted() is True

    def test_scores_shape_and_range(self) -> None:
        x = _series()
        det = SRCNNDetector(epochs=3).fit(x)
        result = det.detect(x)
        scores = np.asarray(result["scores"])
        assert scores.shape == (x.shape[0],)
        assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0
        assert 0.0 <= float(result["anomaly_score"]) <= 1.0

    def test_extract_features_shape(self) -> None:
        x = _series()
        feats = SRCNNDetector(epochs=3).fit(x).extract_features(x)
        assert feats.shape == (x.shape[0], 1)
        assert np.isfinite(feats).all()

    def test_even_window_forced_odd(self) -> None:
        assert SRCNNDetector(window=32).window == 33

    def test_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            SRCNNDetector(window=2)
        with pytest.raises(ValueError):
            SRCNNDetector(inject_ratio=1.5)
        with pytest.raises(ValueError):
            SRCNNDetector(lr=0.0)


class TestSignal:
    def test_injected_spike_scored_high(self) -> None:
        rng = np.random.default_rng(5)
        train = rng.normal(0.0, 1.0, size=500)
        test = rng.normal(0.0, 1.0, size=500)
        test[250] += 10.0  # a clear point anomaly
        det = SRCNNDetector(epochs=60).fit(train)
        scores = np.asarray(det.detect(test)["scores"])
        # the spike neighbourhood should be among the most anomalous points
        assert scores[248:253].max() >= np.quantile(scores, 0.9)
