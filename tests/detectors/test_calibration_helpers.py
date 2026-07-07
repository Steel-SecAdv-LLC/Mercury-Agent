# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the shared detector calibration helpers.

These lock in the two root-cause fixes the helpers exist to provide — empty-safe
squash-scale estimation and a genuinely-finite score finaliser — while proving
the ordinary (non-empty, all-finite) path is unchanged versus the historical
inline implementation the detectors used to duplicate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.detectors._calibration import (
    FINITE_CAP,
    LN2,
    bound_finite,
    bound_finite_config,
    finite_features,
    finite_scores,
    squash_scale,
)
from omni_mercury_engine.detectors.detection_config import active_max_magnitude


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


class TestBoundFinite:
    def test_leaves_realistic_finite_input_unchanged(self) -> None:
        rng = np.random.default_rng(4)
        # Realistic magnitudes: sensor values, ns timestamps (~1.7e18), financials.
        for vals in (rng.normal(size=50), rng.normal(size=50) * 1e18, np.array([1.7e18, -3.2e12])):
            arr = np.asarray(vals, dtype=np.float64)
            assert np.array_equal(bound_finite(arr.copy()), arr)

    def test_inf_bounded_to_cap_not_overflow_sentinel(self) -> None:
        out = bound_finite(np.array([np.inf, -np.inf]))
        assert out[0] == FINITE_CAP and out[1] == -FINITE_CAP
        # The whole point: the bounded value squares without overflowing.
        assert np.isfinite(out[0] ** 2)

    def test_nan_to_zero(self) -> None:
        out = bound_finite(np.array([np.nan, 1.0]))
        assert out[0] == 0.0 and out[1] == 1.0

    def test_absurd_finite_magnitude_clipped(self) -> None:
        out = bound_finite(np.array([1e300, -1e300]))
        assert out[0] == FINITE_CAP and out[1] == -FINITE_CAP

    def test_square_of_bounded_stays_finite(self) -> None:
        # Regression: nan_to_num's 1.8e308 sentinel overflowed on squaring; the
        # FINITE_CAP bound must not.
        assert np.isfinite((bound_finite(np.array([np.inf])) ** 2).sum())

    def test_preserves_shape(self) -> None:
        arr = np.array([[np.inf, 1.0], [np.nan, -np.inf]])
        out = bound_finite(arr)
        assert out.shape == (2, 2) and np.all(np.isfinite(out))


class TestBoundFiniteConfigurableCap:
    """The input cap honours ``OMNI_DETECTOR_MAX_MAGNITUDE``.

    Regression for the hard-coded ``FINITE_CAP``: the documented
    ``OMNI_DETECTOR_MAX_MAGNITUDE`` / ``DetectionConfig.max_magnitude`` knob had no
    effect on the tier's main input guard. ``bound_finite`` now resolves the cap
    from :func:`active_max_magnitude` (env, default ``FINITE_CAP``) with an explicit
    per-call override.
    """

    def test_default_cap_is_finite_cap(self) -> None:
        assert active_max_magnitude() == FINITE_CAP

    def test_env_lowers_the_input_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OMNI_DETECTOR_MAX_MAGNITUDE", "100")
        out = bound_finite(np.array([50.0, 500.0, -9999.0, np.inf, -np.inf]))
        assert out.tolist() == [50.0, 100.0, -100.0, 100.0, -100.0]

    def test_finite_in_range_unchanged_under_env_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OMNI_DETECTOR_MAX_MAGNITUDE", "1e6")
        arr = np.array([1.0, -2.0, 1e5])
        assert np.array_equal(bound_finite(arr.copy()), arr)

    @pytest.mark.parametrize("bad", ["0", "-5", "nan", "inf", "abc", ""])
    def test_invalid_env_falls_back_to_default(
        self, bad: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Mirrors DetectionConfig.max_magnitude validation: a value the dataclass
        # would reject must not become an active cap via the hot-path read.
        monkeypatch.setenv("OMNI_DETECTOR_MAX_MAGNITUDE", bad)
        assert active_max_magnitude() == FINITE_CAP

    def test_explicit_override_beats_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A caller holding a resolved DetectionConfig (config file / per-detector
        # dict) can pass max_magnitude directly; it wins over the env read.
        monkeypatch.setenv("OMNI_DETECTOR_MAX_MAGNITUDE", "10")
        out = bound_finite(np.array([1e6, 1e12]), max_magnitude=1e9)
        assert out.tolist() == [1e6, 1e9]


class _FakeDetector:
    """Minimal detector-shaped object for bound_finite_config."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.name = "fake"
        self._config = config or {}


class TestBoundFiniteConfig:
    """``bound_finite_config`` threads a detector's per-detector config onto input."""

    def test_per_detector_config_applies_on_input(self) -> None:
        det = _FakeDetector({"max_magnitude": 100.0})
        out = bound_finite_config(det, np.array([50.0, 5000.0, np.inf]))
        assert out.tolist() == [50.0, 100.0, 100.0]
        # Resolution is cached on the object for the hot path.
        assert getattr(det, "_detection_config", None) is not None

    def test_no_override_matches_default_cap(self) -> None:
        det = _FakeDetector()
        out = bound_finite_config(det, np.array([np.inf]))
        assert out.tolist() == [FINITE_CAP]

    def test_reuses_preresolved_config(self) -> None:
        from omni_mercury_engine.detectors.detection_config import DetectionConfig

        det = _FakeDetector()
        det._detection_config = DetectionConfig(max_magnitude=100.0)  # type: ignore[attr-defined]
        out = bound_finite_config(det, np.array([5000.0]))
        assert out.tolist() == [100.0]


class TestFiniteFeatures:
    def test_returns_float32(self) -> None:
        out = finite_features(np.array([0.1, 0.2]))
        assert out.dtype == np.float32

    def test_maps_float32_overflow_to_finite_bound(self) -> None:
        # A float64 value beyond float32 range would cast to inf; must be bounded.
        f32_max = float(np.finfo(np.float32).max)
        out = finite_features(np.array([1e300, -1e300, np.inf, np.nan]))
        assert np.all(np.isfinite(out))
        assert out[0] == pytest.approx(f32_max) and out[1] == pytest.approx(-f32_max)
        assert out[3] == 0.0

    def test_preserves_ordinary_values(self) -> None:
        out = finite_features(np.array([1.5, -2.5, 0.0]))
        assert out == pytest.approx(np.array([1.5, -2.5, 0.0], dtype=np.float32))
