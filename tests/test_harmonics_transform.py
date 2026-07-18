# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for :mod:`omni_mercury_engine.harmonics.transform`.

Pins the NumPy-2.0 compatibility fix: ``np.math.factorial`` was removed in
NumPy 2.0, so every ortho-normalized associated-Legendre evaluation (and with
it every ``SphericalHarmonicTransform``/``FastSHTransform`` forward pass)
raised ``AttributeError`` on the pinned ``numpy>=2.4`` floor.  These tests
exercise exactly those paths so the crash class cannot silently return.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from omni_mercury_engine.harmonics.transform import (
    AssociatedLegendre,
    FastSHTransform,
    SphericalHarmonicTransform,
)


class TestAssociatedLegendreOrthoNormalization:
    """The ortho branch computes factorial norms — the NumPy-2.0 crash site."""

    def test_ortho_compute_runs_on_numpy_2(self) -> None:
        """Ortho-normalized compute() must not touch the removed np.math."""
        legendre = AssociatedLegendre(l_max=8, normalization="ortho")
        cos_theta = np.linspace(-0.9, 0.9, 5)

        values = legendre.compute(4, 2, cos_theta)

        assert values.shape == cos_theta.shape
        assert np.all(np.isfinite(values))

    def test_ortho_norm_matches_analytic_factorial_ratio(self) -> None:
        """The normalization constant equals the textbook factorial ratio."""
        degree, m = 5, 3
        raw = AssociatedLegendre(l_max=8, normalization="none")
        ortho = AssociatedLegendre(l_max=8, normalization="ortho")
        cos_theta = np.array([0.3, -0.5])

        expected_norm = math.sqrt(
            (2 * degree + 1)
            / (4 * math.pi)
            * math.factorial(degree - m)
            / math.factorial(degree + m)
        )

        np.testing.assert_allclose(
            ortho.compute(degree, m, cos_theta),
            raw.compute(degree, m, cos_theta) * expected_norm,
            rtol=1e-12,
        )

    @pytest.mark.parametrize("l_max", [16, 32])
    def test_high_l_max_all_orders_finite(self, l_max: int) -> None:
        """High-degree ortho evaluation stays finite for every (l, m)."""
        legendre = AssociatedLegendre(l_max=l_max, normalization="ortho")
        cos_theta = np.linspace(-0.99, 0.99, 7)

        values = legendre.compute(l_max, l_max // 2, cos_theta)

        assert np.all(np.isfinite(values))


class TestTransformRoundTrip:
    """Forward/inverse SH transforms drive the ortho Legendre path end-to-end."""

    def test_forward_produces_finite_coefficients(self) -> None:
        """SphericalHarmonicTransform.forward crashed pre-fix; now runs."""
        transform = SphericalHarmonicTransform(l_max=4)
        theta = np.linspace(0.1, np.pi - 0.1, 40)
        phi = np.linspace(0.0, 2 * np.pi, 40, endpoint=False)
        f = 1.0 + 0.5 * np.cos(theta)

        coefficients = transform.forward(f, theta, phi)

        assert coefficients.l_max == 4
        assert coefficients.coefficients.shape == (transform.n_coefficients,)
        assert np.all(np.isfinite(coefficients.coefficients))

    def test_forward_inverse_reconstructs_smooth_function(self) -> None:
        """A band-limited function survives a forward/inverse round trip."""
        transform = SphericalHarmonicTransform(l_max=6)
        theta = np.repeat(np.linspace(0.15, np.pi - 0.15, 24), 48)
        phi = np.tile(np.linspace(0.0, 2 * np.pi, 48, endpoint=False), 24)
        f = 1.0 + 0.4 * np.cos(theta)

        coefficients = transform.forward(f, theta, phi)
        reconstruction = transform.inverse(coefficients, theta, phi)

        assert np.all(np.isfinite(reconstruction))
        # The built-in quadrature is approximate: assert the reconstruction
        # correlates strongly with the input rather than matching pointwise
        # (measured 0.995 on this grid).
        correlation = np.corrcoef(f, reconstruction)[0, 1]
        assert correlation > 0.95

    def test_reusing_transform_on_new_grid_matches_fresh_instance(self) -> None:
        """The Legendre cache must invalidate when the sampling grid changes.

        Pre-fix, ``AssociatedLegendre`` cached P_l^m arrays keyed only by
        ``(degree, m)``: reusing a transform on a different-length grid
        crashed with a broadcast ``ValueError``, and a same-length but
        different grid silently returned the previous grid's values.
        """
        transform = SphericalHarmonicTransform(l_max=6)
        theta_a = np.repeat(np.linspace(0.15, np.pi - 0.15, 24), 48)
        phi_a = np.tile(np.linspace(0.0, 2 * np.pi, 48, endpoint=False), 24)
        transform.forward(1.0 + 0.4 * np.cos(theta_a), theta_a, phi_a)

        theta_b = np.repeat(np.linspace(0.2, np.pi - 0.2, 20), 40)
        phi_b = np.tile(np.linspace(0.0, 2 * np.pi, 40, endpoint=False), 20)
        f_b = 1.0 + 0.4 * np.cos(theta_b)

        reused = transform.forward(f_b, theta_b, phi_b)
        fresh = SphericalHarmonicTransform(l_max=6).forward(f_b, theta_b, phi_b)

        np.testing.assert_allclose(reused.coefficients, fresh.coefficients)

    def test_fast_transform_precompute_and_forward(self) -> None:
        """FastSHTransform precomputes the ortho Legendre table at init."""
        fast = FastSHTransform(l_max=8)
        f = np.ones((fast._n_theta, fast._n_phi))

        coefficients = fast.forward(f)

        assert np.all(np.isfinite(coefficients.coefficients))
        grid = fast.inverse(coefficients)
        assert grid.shape == (fast._n_theta, fast._n_phi)
        assert np.all(np.isfinite(grid))
