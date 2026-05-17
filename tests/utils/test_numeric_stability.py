"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from omni_mercury_engine.utils import numeric_stability
from omni_mercury_engine.utils.numeric_stability import robust_sqrt, robust_sqrt_vec

# ---------------------------------------------------------------------------
# Deleted-API contract (Vedic helpers removed by spec)
# ---------------------------------------------------------------------------


def test_vedic_helpers_removed() -> None:
    """The renamed module must NOT expose the deleted Vedic helpers."""
    for sym in (
        "vedic_reciprocal",
        "vedic_multiply",
        "set_vedic_optimization",
        "is_vedic_enabled",
        "_ENABLE_VEDIC_OPT",
    ):
        assert not hasattr(
            numeric_stability, sym
        ), f"Deleted symbol {sym!r} re-appeared in numeric_stability"


# ---------------------------------------------------------------------------
# Scalar: robust_sqrt
# ---------------------------------------------------------------------------


def test_robust_sqrt_known_perfect_squares() -> None:
    """Integer perfect squares converge exactly within default tolerance."""
    for n in (0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 100.0, 10_000.0):
        assert math.isclose(robust_sqrt(n), math.sqrt(n), rel_tol=0, abs_tol=1e-10)


def test_robust_sqrt_irrational_values_match_math_sqrt() -> None:
    """Irrational results agree with ``math.sqrt`` to within tol.

    Stays inside the documented convergence regime (5-10 iterations for
    double precision); callers with extreme inputs must bump ``max_iter``.
    """
    for n in (2.0, 3.0, 5.0, 7.0, 0.5, 0.1, 100.0, 1000.0):
        assert math.isclose(robust_sqrt(n), math.sqrt(n), rel_tol=0, abs_tol=1e-9)


def test_robust_sqrt_extreme_magnitudes_require_more_iterations() -> None:
    """For magnitudes far from 1, callers must opt into more iterations."""
    # Bumping max_iter is the documented path for extreme inputs.
    assert math.isclose(robust_sqrt(1e-6, max_iter=40), 1e-3, abs_tol=1e-9)
    assert math.isclose(robust_sqrt(1e8, max_iter=40), 1e4, abs_tol=1e-3)


def test_robust_sqrt_subunit_initial_guess() -> None:
    """Values strictly between 0 and 1 take the ``guess = x`` branch and still converge."""
    for n in (0.25, 0.49, 0.81, 0.99):
        assert math.isclose(robust_sqrt(n), math.sqrt(n), rel_tol=0, abs_tol=1e-10)


def test_robust_sqrt_rejects_negative() -> None:
    """Negative input must raise rather than return NaN."""
    with pytest.raises(ValueError):
        robust_sqrt(-1.0)
    with pytest.raises(ValueError):
        robust_sqrt(-1e-9)


def test_robust_sqrt_zero_is_handled_explicitly() -> None:
    """``robust_sqrt(0)`` must short-circuit to 0 (avoiding division by zero)."""
    assert robust_sqrt(0.0) == 0.0


def test_robust_sqrt_early_termination_via_tol() -> None:
    """A loose tolerance must terminate early without losing correctness."""
    result = robust_sqrt(2.0, tol=1e-3, max_iter=20)
    assert math.isclose(result, math.sqrt(2.0), abs_tol=1e-3)


def test_robust_sqrt_caps_at_max_iter() -> None:
    """When tolerance is unachievably tight, ``max_iter`` bounds the loop."""
    # 1 iteration from guess=8.0 on x=16: (8+16/8)/2 = 5.0
    # 2 iterations: (5+16/5)/2 = 4.1; tol=0 forces full max_iter.
    result = robust_sqrt(16.0, max_iter=2, tol=0.0)
    assert result == pytest.approx(4.1, abs=1e-9)


# ---------------------------------------------------------------------------
# Vector: robust_sqrt_vec
# ---------------------------------------------------------------------------


def test_robust_sqrt_vec_matches_numpy_sqrt() -> None:
    """Vector form must agree with ``np.sqrt`` on a mixed-magnitude input.

    Inputs stay inside the documented convergence regime (5-10 iterations
    for double precision); extreme magnitudes require a higher
    ``max_iter`` and are exercised in a separate test.
    """
    arr = np.array([0.0, 1.0, 2.0, 4.0, 9.0, 100.0, 0.25, 1000.0], dtype=np.float64)
    out = robust_sqrt_vec(arr)
    np.testing.assert_allclose(out, np.sqrt(arr), atol=1e-9)


def test_robust_sqrt_vec_all_zero_short_circuits() -> None:
    """An all-zero array must return zeros without iterating."""
    arr = np.zeros(5, dtype=np.float64)
    out = robust_sqrt_vec(arr)
    np.testing.assert_array_equal(out, arr)


def test_robust_sqrt_vec_preserves_zero_entries() -> None:
    """Mixed zero / nonzero arrays preserve zeros and compute sqrt elsewhere."""
    arr = np.array([0.0, 4.0, 0.0, 9.0], dtype=np.float64)
    out = robust_sqrt_vec(arr)
    np.testing.assert_allclose(out, np.array([0.0, 2.0, 0.0, 3.0]), atol=1e-10)


def test_robust_sqrt_vec_rejects_any_negative() -> None:
    """Any negative element must raise (not silently return NaN)."""
    arr = np.array([1.0, 2.0, -3.0, 4.0], dtype=np.float64)
    with pytest.raises(ValueError):
        robust_sqrt_vec(arr)


def test_robust_sqrt_vec_empty_array() -> None:
    """Empty input returns an empty result without iterating."""
    out = robust_sqrt_vec(np.array([], dtype=np.float64))
    assert out.shape == (0,)
