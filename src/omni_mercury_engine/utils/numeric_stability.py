"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Numerically-stable square-root utilities.

Iterative Babylonian / Heron square-root implementations for use as a
hardware-independent fallback when :func:`numpy.sqrt` is unavailable
(embedded targets, restricted execution sandboxes) or when an explicit
iteration count is required for reproducibility audits.

The iteration is :math:`x_{n+1} = (x_n + S / x_n) / 2`, which converges
quadratically — five to ten iterations are typically sufficient for
double precision.
"""

import numpy as np
import numpy.typing as npt


def robust_sqrt(x: float, max_iter: int = 10, tol: float = 1e-10) -> float:
    """
    Compute the square root of a scalar via Babylonian iteration.

    Args:
        x: Non-negative value.
        max_iter: Maximum number of iterations (5-10 typically sufficient
            for double precision).
        tol: Absolute tolerance for early termination.

    Returns:
        Square root of ``x``.

    Raises:
        ValueError: If ``x`` is negative.

    Examples:
        >>> robust_sqrt(16.0)
        4.0
        >>> abs(robust_sqrt(2.0) - 2.0 ** 0.5) < 1e-10
        True
    """
    if x < 0:
        raise ValueError(f"Cannot compute sqrt of negative number: {x}")
    if x == 0:
        return 0.0

    guess = x / 2.0 if x >= 1 else x
    for _ in range(max_iter):
        next_guess = (guess + x / guess) / 2.0
        if abs(next_guess - guess) < tol:
            return next_guess
        guess = next_guess
    return guess


def robust_sqrt_vec(
    arr: npt.NDArray[np.float64], max_iter: int = 10, tol: float = 1e-10
) -> npt.NDArray[np.float64]:
    """
    Vectorised Babylonian square root over a NumPy array.

    Args:
        arr: Array of non-negative values.
        max_iter: Maximum number of iterations per element.
        tol: Absolute tolerance for early termination (applied element-wise
            via :func:`numpy.all`).

    Returns:
        Element-wise square root of ``arr``.

    Raises:
        ValueError: If any element is negative.
    """
    if np.any(arr < 0):
        raise ValueError("Cannot compute sqrt of negative numbers")

    result = np.zeros_like(arr, dtype=np.float64)
    nonzero_mask = arr > 0
    if not np.any(nonzero_mask):
        return result

    arr_nonzero = arr[nonzero_mask]
    guess = np.where(arr_nonzero >= 1, arr_nonzero / 2.0, arr_nonzero)

    for _ in range(max_iter):
        next_guess = (guess + arr_nonzero / guess) / 2.0
        if np.all(np.abs(next_guess - guess) < tol):
            guess = next_guess
            break
        guess = next_guess

    result[nonzero_mask] = guess
    return result
