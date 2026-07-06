# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the shared detector calibration helpers.

These lock in the two root-cause fixes the helpers exist to provide — empty-safe
squash-scale estimation and a genuinely-finite score finaliser — while proving
the ordinary (non-empty, all-finite) path is unchanged versus the historical
inline implementation the detectors used to duplicate.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors._calibration import LN2, finite_scores, squash_scale


def _legacy_squash_scale(raw: np.ndarray, calibration_quantile: float) -> float:
    """The exact inline body the detectors historically duplicated (for parity)."""
    q = float(np.quantile(raw, calibration_quantile))
    if q < 1e-9:
        q = float(np.mean(raw)) + 1e-9
    return max(q / LN2, 1e-9)


class TestSquashScale:
    def test_matches_legacy_on_ordinary_finite_input(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(20):
            raw = np.abs(rng.normal(size=rng.integers(5, 500)))
            q = float(rng.uniform(0.5, 0.999))
            assert squash_scale(raw, q) == pytest.approx(_legacy_squash_scale(raw, q))

    def test_empty_returns_safe_default_not_indexerror(self) -> None:
        # Historical inline copy raised IndexError here.
        assert squash_scale(np.array([]), 0.98) == 1.0

    def test_all_nonfinite_returns_safe_default(self) -> None:
        assert squash_scale(np.array([np.inf, -np.inf, np.nan]), 0.98) == 1.0

    def test_ignores_nonfinite_entries(self) -> None:
        clean = np.abs(np.random.default_rng(1).normal(size=100))
        poisoned = np.concatenate([clean, [np.inf, np.nan, -np.inf]])
        # Poisoning the calibration signal must not move the scale off the
        # finite quantile (it certainly must not make it inf/NaN).
        assert squash_scale(poisoned, 0.98) == pytest.approx(squash_scale(clean, 0.98))

    def test_degenerate_constant_uses_mean_fallback_and_is_positive(self) -> None:
        # Near-zero quantile -> mean fallback; result strictly positive & finite.
        scale = squash_scale(np.zeros(50), 0.98)
        assert np.isfinite(scale) and scale >= 1e-9

    def test_result_always_positive_finite(self) -> None:
        rng = np.random.default_rng(2)
        for _ in range(50):
            raw = rng.normal(size=rng.integers(1, 200)) ** 2
            scale = squash_scale(raw, float(rng.uniform(0.5, 0.99)))
            assert np.isfinite(scale) and scale >= 1e-9


class TestFiniteScores:
    def test_nan_maps_to_lo(self) -> None:
        out = finite_scores(np.array([np.nan, 0.3, np.nan]))
        assert out[0] == 0.0 and out[2] == 0.0
        assert out[1] == pytest.approx(0.3)

    def test_posinf_maps_to_hi_neginf_to_lo(self) -> None:
        out = finite_scores(np.array([np.inf, -np.inf]))
        assert out[0] == 1.0 and out[1] == 0.0

    def test_clips_out_of_range_finite(self) -> None:
        out = finite_scores(np.array([-5.0, 2.0, 0.5]))
        assert out[0] == 0.0 and out[1] == 1.0 and out[2] == pytest.approx(0.5)

    def test_always_finite_and_in_range(self) -> None:
        rng = np.random.default_rng(3)
        for _ in range(50):
            vals = rng.normal(size=rng.integers(1, 64)) * 1e6
            # Sprinkle in non-finite values.
            if vals.size >= 3:
                vals[0] = np.nan
                vals[1] = np.inf
                vals[2] = -np.inf
            out = finite_scores(vals)
            assert np.all(np.isfinite(out))
            assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_preserves_shape(self) -> None:
        vals = np.array([[0.1, np.nan], [np.inf, 0.9]])
        out = finite_scores(vals)
        assert out.shape == (2, 2)
        assert np.all(np.isfinite(out))

    def test_custom_bounds(self) -> None:
        out = finite_scores(np.array([np.nan, np.inf, 5.0]), lo=-1.0, hi=1.0)
        assert out[0] == -1.0 and out[1] == 1.0 and out[2] == 1.0

    def test_empty_input(self) -> None:
        out = finite_scores(np.array([]))
        assert out.size == 0
