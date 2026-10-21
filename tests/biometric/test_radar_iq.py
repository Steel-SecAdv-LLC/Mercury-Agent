# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Radar I/Q ellipse correction recovers displacement raw arctangent distorts."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.biometric.radar_iq import (
    IQEllipse,
    demodulate_iq,
    fit_iq_ellipse,
)

WAVELENGTH_M = 299_792_458.0 / 24e9


def _imbalanced_iq(
    displacement_m: np.ndarray,
    *,
    dc: tuple[float, float] = (0.3, -0.5),
    gains: tuple[float, float] = (0.8, 1.3),
    phase_error_rad: float = 0.35,
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """I/Q of a CW radar with DC offsets, gain and phase imbalance."""
    phi = 4.0 * np.pi * displacement_m / WAVELENGTH_M + 0.7
    rng = np.random.default_rng(seed)
    i = gains[0] * np.cos(phi) + dc[0] + noise * rng.normal(size=phi.size)
    q = gains[1] * np.sin(phi + phase_error_rad) + dc[1] + noise * rng.normal(size=phi.size)
    return i, q


def _breathing(seconds: float = 60.0, fs: float = 100.0) -> np.ndarray:
    t = np.arange(0.0, seconds, 1.0 / fs)
    return 0.004 * np.sin(2 * np.pi * 0.25 * t) + 0.0003 * np.sin(2 * np.pi * 1.2 * t)


def _rms_error_m(recovered: np.ndarray, truth: np.ndarray) -> float:
    """RMS error up to the demodulator's arbitrary sign and offset."""
    errors = []
    for sign in (1.0, -1.0):
        err = sign * recovered - truth
        errors.append(float(np.sqrt(np.mean((err - err.mean()) ** 2))))
    return min(errors)


class TestEllipseFit:
    def test_recovers_centre_axes_and_residual(self) -> None:
        i, q = _imbalanced_iq(_breathing(), noise=0.002)
        ellipse = fit_iq_ellipse(i, q)
        assert ellipse.center == pytest.approx((0.3, -0.5), abs=0.01)
        # The phase error shears the ellipse, so the semi-axes are not the
        # channel gains; they follow from the parametric curve's covariance
        # (semi-axis = sqrt(2 eigenvalue) for x = a cos t, y = b sin(t + e)).
        t = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
        curve = np.column_stack([0.8 * np.cos(t), 1.3 * np.sin(t + 0.35)])
        expected = np.sqrt(2.0 * np.linalg.eigvalsh(np.cov(curve.T, bias=True)))
        assert sorted(ellipse.axes) == pytest.approx(sorted(expected), abs=0.02)
        assert ellipse.residual < 0.02
        assert ellipse.arc_rad > np.pi

    def test_short_arc_is_reported(self) -> None:
        # 0.5 mm of motion sweeps ~1 rad at 24 GHz: a sliver of the ellipse.
        i, q = _imbalanced_iq(0.125 * _breathing())
        assert fit_iq_ellipse(i, q).arc_rad < np.pi

    def test_degenerate_inputs_are_rejected(self) -> None:
        t = np.linspace(0.0, 1.0, 100)
        with pytest.raises(ValueError, match="constant"):
            fit_iq_ellipse(np.ones(100), np.ones(100))
        with pytest.raises(ValueError, match=r"ellipse|collinear|degenerate"):
            fit_iq_ellipse(t, 2.0 * t + 1.0)
        with pytest.raises(ValueError, match="at least 6"):
            fit_iq_ellipse(t[:4], t[:4])
        with pytest.raises(ValueError, match="non-finite"):
            fit_iq_ellipse(np.r_[t, np.nan], np.r_[t, 1.0])
        with pytest.raises(TypeError, match="separately"):
            fit_iq_ellipse(t.astype(complex), t)

    def test_mismatched_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="differ in length"):
            fit_iq_ellipse(np.zeros(10), np.zeros(9))


class TestDemodulation:
    def test_correction_beats_raw_arctangent(self) -> None:
        truth = _breathing()
        i, q = _imbalanced_iq(truth, noise=0.002)
        corrected = demodulate_iq(i, q, wavelength_m=WAVELENGTH_M)
        raw = np.unwrap(np.arctan2(q, i)) * WAVELENGTH_M / (4.0 * np.pi)
        corrected_err = _rms_error_m(corrected, truth)
        raw_err = _rms_error_m(raw, truth)
        # The raw error (~0.5 mm) swamps cardiac motion (~0.3 mm); corrected
        # recovery is tens of micrometres.
        assert corrected_err < 10e-6
        assert corrected_err < raw_err / 10.0

    def test_a_prefit_ellipse_is_reused(self) -> None:
        truth = _breathing()
        i, q = _imbalanced_iq(truth)
        ellipse = fit_iq_ellipse(i, q)
        a = demodulate_iq(i, q, wavelength_m=WAVELENGTH_M, ellipse=ellipse)
        b = demodulate_iq(i, q, wavelength_m=WAVELENGTH_M)
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_starts_at_zero(self) -> None:
        i, q = _imbalanced_iq(_breathing())
        assert demodulate_iq(i, q, wavelength_m=WAVELENGTH_M)[0] == 0.0

    @pytest.mark.parametrize("wavelength", [0.0, -1.0, float("nan")])
    def test_bad_wavelength_is_rejected(self, wavelength: float) -> None:
        i, q = _imbalanced_iq(_breathing())
        with pytest.raises(ValueError, match="wavelength_m"):
            demodulate_iq(i, q, wavelength_m=wavelength)

    def test_to_circle_maps_the_fit_onto_the_unit_circle(self) -> None:
        i, q = _imbalanced_iq(_breathing())
        circle = fit_iq_ellipse(i, q).to_circle(i, q)
        assert np.hypot(circle[:, 0], circle[:, 1]) == pytest.approx(1.0, abs=1e-6)

    def test_ellipse_is_a_frozen_record(self) -> None:
        i, q = _imbalanced_iq(_breathing())
        ellipse = fit_iq_ellipse(i, q)
        assert isinstance(ellipse, IQEllipse)
        with pytest.raises(AttributeError):
            ellipse.residual = 0.0  # type: ignore[misc]
