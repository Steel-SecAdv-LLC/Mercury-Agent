# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the DDPM reconstruction anomaly detector."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.detectors.diffusion_ad import (
    DiffusionReconstructionDetector,
)


def _series(seed: int = 0, n: int = 400) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # a smooth sinusoid + noise: a coherent normal manifold for the denoiser
    t = np.linspace(0, 8 * np.pi, n)
    return np.sin(t) + rng.normal(0.0, 0.1, size=n)


class TestContract:
    def test_unfitted_then_fitted(self) -> None:
        det = DiffusionReconstructionDetector(epochs=3)
        assert det.is_fitted() is False
        det.fit(_series())
        assert det.is_fitted() is True

    def test_scores_shape_and_range(self) -> None:
        x = _series()
        det = DiffusionReconstructionDetector(epochs=3).fit(x)
        result = det.detect(x)
        scores = np.asarray(result["scores"])
        assert scores.shape == (x.shape[0],)
        assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0

    def test_extract_features_shape(self) -> None:
        x = _series()
        feats = DiffusionReconstructionDetector(epochs=3).fit(x).extract_features(x)
        assert feats.shape == (x.shape[0], 1)
        assert np.isfinite(feats).all()

    def test_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            DiffusionReconstructionDetector(window=2)
        with pytest.raises(ValueError):
            DiffusionReconstructionDetector(n_steps=1)
        with pytest.raises(ValueError):
            DiffusionReconstructionDetector(eval_steps=(999,))
        with pytest.raises(ValueError):
            DiffusionReconstructionDetector(calibration_quantile=1.0)


class TestSignal:
    def test_out_of_manifold_scored_high(self) -> None:
        train = _series(1)
        test = _series(2)
        test[200:205] += 6.0  # a burst well off the learned sinusoidal manifold
        det = DiffusionReconstructionDetector(epochs=40).fit(train)
        scores = np.asarray(det.detect(test)["scores"])
        assert scores[198:207].max() >= np.quantile(scores, 0.9)

    def test_deterministic(self) -> None:
        x = _series(3)
        a = DiffusionReconstructionDetector(epochs=5, seed=7).fit(x).detect(x)["scores"]
        b = DiffusionReconstructionDetector(epochs=5, seed=7).fit(x).detect(x)["scores"]
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)
