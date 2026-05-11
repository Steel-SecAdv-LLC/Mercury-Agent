"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.
"""

from __future__ import annotations

"""
Pin the cross-SciPy-version equivalence of the inline ``sph_harm`` shim used
inside ``ThreeRMechanism.analyze_with_spherical_harmonics``.

SciPy ≥ 1.14 removed the legacy ``scipy.special.sph_harm`` and replaced it with
``sph_harm_y``.  The two functions have the same positional shape but disagree
on which angle is azimuthal and which is polar:

    legacy   sph_harm   (m, n, θ_az,  φ_pol)   math convention
    modern   sph_harm_y (n, m, θ_pol, φ_az)    physics convention

The shim must produce the same Y_l^m on either SciPy.  Without a regression
pin, an "obvious" tidy-up of one branch can silently desync the math across
SciPy releases.

This test reconstructs both wrappers and asserts they agree against a known
reference value at a fixed (m, n, polar, azimuthal) point.
"""

import numpy as np
import pytest

scipy_special = pytest.importorskip("scipy.special")


def _modern_wrapper(m: int, n: int, phi_az: float, theta_pol: float) -> complex:
    """Replicates the modern ``sph_harm_y`` branch of the shim."""
    if not hasattr(scipy_special, "sph_harm_y"):
        pytest.skip("scipy.special.sph_harm_y unavailable on this SciPy version")
    return complex(scipy_special.sph_harm_y(n, m, theta_pol, phi_az))


def _legacy_wrapper(m: int, n: int, phi_az: float, theta_pol: float) -> complex:
    """Replicates the legacy ``sph_harm`` branch of the shim."""
    if not hasattr(scipy_special, "sph_harm"):
        pytest.skip("scipy.special.sph_harm unavailable on this SciPy version")
    return complex(scipy_special.sph_harm(m, n, phi_az, theta_pol))


@pytest.mark.parametrize(
    "m,n,phi_az,theta_pol",
    [
        (0, 0, 0.0, 0.0),
        (0, 1, 1.3, 0.7),
        (1, 2, 1.3, 0.7),
        (-1, 2, 0.5, 1.1),
        (2, 3, 2.1, 2.0),
    ],
)
def test_sph_harm_modern_branch_returns_finite_complex(
    m: int, n: int, phi_az: float, theta_pol: float
) -> None:
    """Modern branch produces a finite complex Y_l^m at the chosen point."""
    y = _modern_wrapper(m, n, phi_az, theta_pol)
    assert isinstance(y, complex)
    assert np.isfinite(y.real) and np.isfinite(y.imag)


@pytest.mark.parametrize(
    "m,n,phi_az,theta_pol",
    [
        (0, 0, 0.0, 0.0),
        (0, 1, 1.3, 0.7),
        (1, 2, 1.3, 0.7),
        (-1, 2, 0.5, 1.1),
        (2, 3, 2.1, 2.0),
    ],
)
def test_sph_harm_branches_agree_when_both_available(
    m: int, n: int, phi_az: float, theta_pol: float
) -> None:
    """
    When both SciPy spellings are present (only on the SciPy 1.13–1.14
    transitional band), the two branches MUST return the same value.  This
    is the most direct guard against an argument-order regression.
    """
    if not (hasattr(scipy_special, "sph_harm") and hasattr(scipy_special, "sph_harm_y")):
        pytest.skip("Need both legacy sph_harm and modern sph_harm_y for cross-check")
    modern = _modern_wrapper(m, n, phi_az, theta_pol)
    legacy = _legacy_wrapper(m, n, phi_az, theta_pol)
    assert abs(modern - legacy) < 1e-10, (
        f"sph_harm shim branches disagree at (m={m}, n={n}, "
        f"phi_az={phi_az}, theta_pol={theta_pol}): "
        f"modern={modern}, legacy={legacy}"
    )


def test_sph_harm_y_00_normalization() -> None:
    """
    Y_0^0 ≡ 1/(2√π) — independent of angles.  Pins the modern branch's
    polar/azimuthal arg order: if theta and phi are swapped, Y_0^0 is
    still 1/(2√π) (degenerate case), but Y_1^0 at a non-zero polar angle
    would not match the closed-form expression below.
    """
    # Y_0^0 = 1 / (2 * sqrt(pi))
    expected_00 = 1.0 / (2.0 * np.sqrt(np.pi))
    assert abs(_modern_wrapper(0, 0, 1.3, 0.7).real - expected_00) < 1e-10

    # Y_1^0(θ) = (1/2) * sqrt(3/π) * cos(θ).  This DOES depend on the
    # polar angle, so a swapped (θ, φ) would fail the assertion.
    theta_pol = 0.7
    phi_az = 1.3
    expected_10 = 0.5 * np.sqrt(3.0 / np.pi) * np.cos(theta_pol)
    actual = _modern_wrapper(0, 1, phi_az, theta_pol)
    assert abs(actual.real - expected_10) < 1e-10
    assert abs(actual.imag) < 1e-10
