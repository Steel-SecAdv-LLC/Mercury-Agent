# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the particle-filter state-space residual detector.

These verify the :class:`~omni_mercury_engine.core.base.BaseDetector` contract,
determinism (a pinned seed yields identical scores across runs), that a level
shift / spike drives the normalised innovation up, and that the stationary
false-positive rate stays near the ``calibration_quantile`` budget.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.particle_filter import ParticleFilterDetector


def _series_with_spike(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, 500)
    labels = np.zeros(500, dtype=int)
    x[250] += 10.0  # sharp spike the filter has not absorbed
    labels[250] = 1
    return x.astype(np.float64), labels


class TestContract:
    def test_unfitted_then_fitted(self) -> None:
        det = ParticleFilterDetector(n_particles=64)
        assert det.is_fitted() is False
        det.fit(np.random.default_rng(0).normal(size=300))
        assert det.is_fitted() is True

    def test_scores_shape_and_range(self) -> None:
        x, _ = _series_with_spike()
        out = ParticleFilterDetector(n_particles=64).fit(x).detect(x)
        scores = np.asarray(out["scores"])
        assert scores.shape == (len(x),)
        assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0

    def test_extract_features_shape(self) -> None:
        x, _ = _series_with_spike()
        feats = ParticleFilterDetector(n_particles=64).fit(x).extract_features(x)
        assert feats.shape == (len(x), 1)
        assert np.isfinite(feats).all()

    def test_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            ParticleFilterDetector(n_particles=4)
        with pytest.raises(ValueError):
            ParticleFilterDetector(calibration_quantile=1.0)

    def test_deterministic_with_seed(self) -> None:
        x, _ = _series_with_spike(7)
        a = ParticleFilterDetector(n_particles=128, seed=42).fit(x).detect(x)["scores"]
        b = ParticleFilterDetector(n_particles=128, seed=42).fit(x).detect(x)["scores"]
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


class TestSignal:
    def test_spike_flagged(self) -> None:
        train = np.random.default_rng(1).normal(size=500)
        x, labels = _series_with_spike(2)
        det = ParticleFilterDetector(n_particles=200, seed=0).fit(train)
        scores = np.asarray(det.detect(x)["scores"])
        assert scores[250] > 0.5
        assert scores[250] > np.median(scores) + 0.3

    def test_low_false_positive_rate(self) -> None:
        train = np.random.default_rng(3).normal(size=800)
        test = np.random.default_rng(4).normal(size=800)
        det = ParticleFilterDetector(n_particles=200, calibration_quantile=0.98).fit(train)
        scores = np.asarray(det.detect(test)["scores"])
        assert (scores > 0.5).mean() < 0.08
