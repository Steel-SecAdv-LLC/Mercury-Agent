# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
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
    SHBasis,
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


def _complex_harmonic(fast: FastSHTransform, degree: int, order: int) -> np.ndarray:
    """``Y_l^m`` on the transform's own grid, from the independent basis class."""
    basis = SHBasis(fast._l_max, real_basis=False)
    values = basis.compute(degree, order, fast._theta_grid.ravel(), fast._phi_grid.ravel())
    return np.asarray(values).reshape(fast._theta_grid.shape)


class TestFastTransformIsARoundTrip:
    """``FastSHTransform.inverse(forward(f))`` is ``f`` for a band-limited ``f``.

    The forward pass had no ``2 pi / n_phi`` on the phi sum and a rectangle
    rule in theta, and the inverse wrote the negative orders into column
    ``|m|`` (they belong in ``n_phi - |m|``) before taking the real part:
    a round trip of Y_1^0 came back at 0.14 of its amplitude.
    """

    #: (degree, order) pairs, negative and odd/even orders among them.
    MODES: tuple[tuple[int, int], ...] = (
        (0, 0),
        (1, 0),
        (1, 1),
        (1, -1),
        (2, -1),
        (3, 2),
        (5, -4),
        (8, 8),
        (8, -7),
    )

    @pytest.mark.parametrize(("degree", "order"), MODES)
    def test_a_pure_harmonic_is_one_unit_coefficient(self, degree: int, order: int) -> None:
        """Forward of ``Y_l^m`` is one at (l, m) and zero everywhere else."""
        fast = FastSHTransform(l_max=8)

        coefficients = fast.forward(_complex_harmonic(fast, degree, order))

        expected = np.zeros((fast._l_max + 1) ** 2, dtype=np.complex128)
        expected[degree * (degree + 1) + order] = 1.0
        np.testing.assert_allclose(coefficients.coefficients, expected, atol=1e-10)
        assert coefficients.real_basis is False

    @pytest.mark.parametrize(("degree", "order"), MODES)
    def test_a_pure_harmonic_round_trips(self, degree: int, order: int) -> None:
        """The complex field of one harmonic comes back, negative orders included."""
        fast = FastSHTransform(l_max=8)
        field_in = _complex_harmonic(fast, degree, order)

        field_out = fast.inverse(fast.forward(field_in), real=False)

        relative = np.linalg.norm(field_out - field_in) / np.linalg.norm(field_in)
        assert relative < 1e-6

    def test_a_real_band_limited_field_round_trips(self) -> None:
        """A real mixture (so every order has its negative partner) comes back."""
        fast = FastSHTransform(l_max=8)
        rng = np.random.default_rng(20240611)
        field_in = np.zeros(fast._theta_grid.shape)
        for degree in range(fast._l_max + 1):
            for order in range(-degree, degree + 1):
                weight = complex(rng.normal(), rng.normal())
                field_in += (weight * _complex_harmonic(fast, degree, order)).real

        coefficients = fast.forward(field_in)
        field_out = fast.inverse(coefficients)

        assert field_out.dtype == np.float64
        relative = np.linalg.norm(field_out - field_in) / np.linalg.norm(field_in)
        assert relative < 1e-6
        # Negative orders carry real content here; a transform that dropped
        # them could not have reproduced the field.
        negative = [
            abs(coefficients.get_coefficient(degree, -order))
            for degree in range(1, fast._l_max + 1)
            for order in range(1, degree + 1)
        ]
        assert max(negative) > 0.1

    def test_a_constant_has_the_analytic_coefficient(self) -> None:
        """``f = 1`` is ``sqrt(4 pi) Y_0^0``: the scale is absolute, not relative."""
        fast = FastSHTransform(l_max=8)

        coefficients = fast.forward(np.ones((fast._n_theta, fast._n_phi)))

        assert coefficients.get_coefficient(0, 0) == pytest.approx(math.sqrt(4 * math.pi))
        np.testing.assert_allclose(coefficients.coefficients[1:], 0.0, atol=1e-10)

    def test_the_weights_integrate_the_sphere(self) -> None:
        """The grid weights sum to the area of the unit sphere."""
        fast = FastSHTransform(l_max=8)

        assert float(np.sum(fast._weights)) == pytest.approx(4 * math.pi, rel=1e-12)

    def test_the_real_part_of_a_complex_field_is_said_to_be_partial(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``real=True`` on non-real coefficients warns instead of dropping silently."""
        fast = FastSHTransform(l_max=4)
        coefficients = fast.forward(_complex_harmonic(fast, 2, 1))

        with caplog.at_level("WARNING", logger="omni_mercury_engine.harmonics.transform"):
            fast.inverse(coefficients)

        assert any("imaginary part" in record.message for record in caplog.records)


#: Where the high-degree values below were evaluated, as ``cos(theta)``.
REFERENCE_POINTS: tuple[float, float, float] = (-0.5, 0.11, 0.8)

#: The orthonormal function at those points, from the recurrence carried at 80
#: significant digits with mpmath (no step of it leaves that range).
REFERENCE_2500_1000: tuple[float, float, float] = (
    0.29087820003191149,
    0.12477315947303664,
    -0.41331989303550859,
)
REFERENCE_3000_1500: tuple[float, float, float] = (
    0.091399397238941737,
    -0.32233807077575756,
    -0.13046405459567576,
)


class TestHighDegreeIsStable:
    """The orthonormal functions stay finite and orthonormal at high degree.

    Was: ``OverflowError`` at l = 150 (``(l+m)!`` no longer fits a float from
    l + m = 171) and NaN before that (``(2m-1)!!`` passes 1e308 at m = 151),
    from a class that offers "numerical stability for high degrees".
    """

    @pytest.mark.parametrize(
        ("degree", "order"),
        [(150, 0), (150, 75), (150, 150), (300, 150), (300, 300), (1000, 500), (1000, 1000)],
    )
    def test_a_high_degree_function_has_unit_norm(self, degree: int, order: int) -> None:
        nodes, weights = np.polynomial.legendre.leggauss(1200)
        values = AssociatedLegendre(l_max=degree, normalization="ortho").compute(
            degree, order, nodes
        )

        assert np.isfinite(values).all()
        assert 2.0 * np.pi * float(np.sum(weights * values * values)) == pytest.approx(
            1.0, abs=1e-9
        )

    @pytest.mark.parametrize(("degree", "order"), [(2500, 1000), (3000, 1500), (3000, 5)])
    def test_a_degree_past_the_reach_of_the_seed_has_unit_norm(
        self, degree: int, order: int
    ) -> None:
        """Was 0.82 at (2500, 1000) and 6e252 at (3000, 1500): finite, and wrong.

        ``sin^m`` left the range of a float on the way to a value of order
        one, and the recurrence climbed back from the digits it had lost.
        """
        nodes, weights = np.polynomial.legendre.leggauss(degree + 40)
        values = AssociatedLegendre(l_max=degree, normalization="ortho").compute(
            degree, order, nodes
        )

        assert 2.0 * np.pi * float(np.sum(weights * values * values)) == pytest.approx(
            1.0, abs=1e-9
        )

    @pytest.mark.parametrize(
        ("degree", "order", "expected"),
        [
            (2500, 1000, REFERENCE_2500_1000),
            (3000, 1500, REFERENCE_3000_1500),
        ],
    )
    def test_the_values_are_the_ones_eighty_digits_give(
        self, degree: int, order: int, expected: tuple[float, float, float]
    ) -> None:
        """The same recurrence carried at 80 significant digits (mpmath), where nothing underflows."""
        values = AssociatedLegendre(l_max=degree, normalization="ortho").compute(
            degree, order, np.array(REFERENCE_POINTS)
        )

        np.testing.assert_allclose(values, expected, rtol=0.0, atol=1e-11)

    def test_two_high_degrees_of_one_order_are_orthogonal(self) -> None:
        nodes, weights = np.polynomial.legendre.leggauss(1200)
        legendre = AssociatedLegendre(l_max=300, normalization="ortho")

        overlap = float(
            np.sum(weights * legendre.compute(300, 20, nodes) * legendre.compute(298, 20, nodes))
        )

        assert abs(2.0 * np.pi * overlap) < 1e-9

    def test_the_stable_recurrence_gives_what_the_factorial_formula_gave(self) -> None:
        """Where the old formula could be evaluated, the values have not moved."""
        x = np.linspace(-0.999, 0.999, 41)
        for degree in range(0, 31):
            for order in range(-degree, degree + 1):
                raw = AssociatedLegendre(l_max=degree, normalization="none").compute(
                    degree, order, x
                )
                factor = math.sqrt(
                    (2 * degree + 1)
                    / (4 * math.pi)
                    * math.factorial(degree - abs(order))
                    / math.factorial(degree + abs(order))
                )
                ortho = AssociatedLegendre(l_max=degree, normalization="ortho").compute(
                    degree, order, x
                )
                np.testing.assert_allclose(ortho, raw * factor, rtol=1e-11, atol=1e-13)
