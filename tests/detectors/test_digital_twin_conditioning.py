# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Numerical-conditioning tests for the digital-twin scale-relative ridge.

Exercises the scale-relative Tikhonov ridge (``ridge = ridge_factor *
trace(gram) / d``, floored by the absolute ``ridge``) on near-singular and
large-/tiny-magnitude Gram matrices, asserting the solve stays stable and
bounded and that the regularisation tracks the data scale.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.digital_twin import DigitalTwinResidualDetector


def _finite_bounded(det: DigitalTwinResidualDetector, data: np.ndarray) -> None:
    assert det._coef is not None
    assert np.all(np.isfinite(det._coef)), "twin coefficients must be finite"
    assert np.isfinite(det._intercept)
    out = det.detect(data)
    scores = np.asarray(out["scores"])
    assert np.all(np.isfinite(scores))
    assert np.all((scores >= 0.0) & (scores <= 1.0))


class TestRidgeParam:
    def test_ridge_factor_defaults_from_config(self) -> None:
        det = DigitalTwinResidualDetector()
        assert det.ridge_factor == pytest.approx(1e-6)

    def test_ridge_factor_explicit(self) -> None:
        det = DigitalTwinResidualDetector(ridge_factor=1e-2)
        assert det.ridge_factor == 1e-2

    def test_negative_ridge_factor_rejected(self) -> None:
        with pytest.raises(ValueError, match="ridge_factor must be >= 0"):
            DigitalTwinResidualDetector(ridge_factor=-1.0)

    def test_ridge_factor_from_config_dict(self) -> None:
        det = DigitalTwinResidualDetector(config={"ridge_factor": 3e-4})
        assert det.ridge_factor == 3e-4


class TestNearSingular:
    def test_constant_series_gram_near_zero(self) -> None:
        # A constant training series makes the AR Gram matrix (near-)singular:
        # trace(gram)/d -> 0, so the absolute ridge floor keeps the solve stable.
        det = DigitalTwinResidualDetector(order=4).fit(np.full(300, 5.0))
        probe = np.concatenate([np.full(60, 5.0), [40.0], np.full(59, 5.0)])
        _finite_bounded(det, probe)

    def test_repeated_lag_pattern(self) -> None:
        # A short-period square wave makes lag columns nearly collinear.
        base = np.tile([1.0, 1.0, -1.0, -1.0], 100).astype(float)
        det = DigitalTwinResidualDetector(order=6).fit(base)
        _finite_bounded(det, base)

    def test_near_singular_solution_is_bounded(self) -> None:
        rng = np.random.default_rng(0)
        # Two lags almost identical -> collinear design.
        x = rng.normal(size=400)
        series = x + 1e-9 * np.arange(400)
        det = DigitalTwinResidualDetector(order=3).fit(series)
        assert float(np.linalg.norm(det._coef)) < 1e3, "coefficients must stay bounded"
        _finite_bounded(det, series)


class TestMagnitudeScaling:
    def test_large_magnitude_gram(self) -> None:
        rng = np.random.default_rng(1)
        big = rng.normal(size=500) * 1e6 + 1e7
        det = DigitalTwinResidualDetector(order=4).fit(big)
        # A fixed absolute ridge (1e-6) would be numerically negligible here; the
        # scale-relative ridge keeps coefficients well-conditioned and bounded.
        assert float(np.linalg.norm(det._coef)) < 10.0
        _finite_bounded(det, big)

    def test_tiny_magnitude_gram(self) -> None:
        rng = np.random.default_rng(2)
        tiny = rng.normal(size=500) * 1e-9
        det = DigitalTwinResidualDetector(order=4).fit(tiny)
        _finite_bounded(det, tiny)

    def test_coefficients_bounded_across_scales(self) -> None:
        # The scale-relative ridge keeps the solve well-conditioned across many
        # orders of magnitude: coefficient norm stays bounded whether the signal
        # is tiny or huge (a fixed absolute ridge blows up / vanishes at extremes).
        rng = np.random.default_rng(3)
        x = rng.normal(size=600)
        signal = np.convolve(x, [1.0, 0.5, 0.25], mode="same")
        norms = []
        for scale in (1e-6, 1.0, 1e3, 1e9):
            det = DigitalTwinResidualDetector(order=3).fit(signal * scale)
            assert np.all(np.isfinite(det._coef))
            norms.append(float(np.linalg.norm(det._coef)))
        # All coefficient norms stay in a tight, bounded band regardless of scale.
        assert max(norms) < 10.0
        assert max(norms) - min(norms) < 1.0


hyp = pytest.importorskip("hypothesis")
from hypothesis import (
    given,
    settings,
    strategies as st,
)
from hypothesis.extra import numpy as hnp


class TestConditioningProperty:
    @settings(max_examples=60, deadline=None)
    @given(
        hnp.arrays(
            dtype=np.float64,
            shape=st.integers(20, 200),
            elements=st.floats(allow_nan=True, allow_infinity=True, width=64),
        )
    )
    def test_any_series_yields_finite_bounded_solution(self, series: np.ndarray) -> None:
        det = DigitalTwinResidualDetector(order=4).fit(series)
        assert det._coef is not None and np.all(np.isfinite(det._coef))
        scores = np.asarray(det.detect(series)["scores"])
        assert np.all(np.isfinite(scores))
        assert np.all((scores >= 0.0) & (scores <= 1.0))
