# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mercury-native matrix profile: numpy STOMP with MASS sliding dot products.

The matrix profile of a series records, for every length-``m`` subsequence,
the distance to its nearest non-trivial neighbour, so discords (anomalies)
are the profile's largest values and motifs its smallest. This module
computes the self-join profile in pure numpy, matching the conventions of
``stumpy.stump`` so the two are interchangeable backends:

- z-normalized Euclidean distance by default, plain Euclidean with
  ``normalize=False``;
- a trivial-match exclusion zone of ``ceil(m / 4)`` around the diagonal;
- nearest-neighbour, left (``j < i``) and right (``j > i``) indices, with
  ``-1`` where no neighbour exists;
- the constant-subsequence convention: two flat windows are distance 0,
  a flat window against a varying one is ``sqrt(m)``.

The rows are produced by the STOMP recurrence (one O(n) update per row)
seeded and periodically refreshed by exact FFT sliding dot products, which
arrests the floating-point drift the bare recurrence accumulates. The
series is centred before any arithmetic: both the z-normalized and the
plain Euclidean distances are invariant to a global shift, and centring
removes the catastrophic cancellation a large offset would otherwise cause
in the rolling moments.

Reference:
    Yeh et al., "Matrix Profile I: All Pairs Similarity Joins for Time
    Series" (ICDM 2016); Zhu et al., "Matrix Profile II: Exploiting a
    Novel Algorithm and GPUs" (ICDM 2016; STOMP).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["MatrixProfileResult", "matrix_profile", "sliding_dot_product"]

_MIN_WINDOW = 3
_EXCLUSION_DENOM = 4
_REFRESH_INTERVAL = 512
_CONSTANT_STD_RTOL = 1e-10


@dataclass(frozen=True)
class MatrixProfileResult:
    """Self-join matrix profile of one series.

    Attributes:
        profile: Distance from each subsequence to its nearest non-trivial
            neighbour, shape ``[n - m + 1]``; ``inf`` where none exists.
        index: Index of that neighbour; ``-1`` where none exists.
        left_index: Nearest neighbour at a strictly smaller index, ``-1``
            where none exists.
        right_index: Nearest neighbour at a strictly larger index, ``-1``
            where none exists.
        window_size: The subsequence length ``m`` the profile was built with.
    """

    profile: np.ndarray[Any, Any]
    index: np.ndarray[Any, Any]
    left_index: np.ndarray[Any, Any]
    right_index: np.ndarray[Any, Any]
    window_size: int


def sliding_dot_product(
    query: np.ndarray[Any, Any], series: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Dot product of ``query`` against every window of ``series`` via FFT.

    Args:
        query: 1-D query of length ``m``.
        series: 1-D series of length ``n >= m``.

    Returns:
        Array of shape ``[n - m + 1]`` where entry ``j`` is
        ``query @ series[j : j + m]``, computed in O(n log n).
    """
    q = np.ascontiguousarray(query, dtype=np.float64)
    t = np.ascontiguousarray(series, dtype=np.float64)
    m, n = q.shape[0], t.shape[0]
    size = 1 << int(n + m - 1).bit_length()
    conv = np.fft.irfft(np.fft.rfft(q[::-1], size) * np.fft.rfft(t, size), size)
    return np.asarray(conv[m - 1 : n], dtype=np.float64)


def _rolling_moments(
    centred: np.ndarray[Any, Any], m: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Rolling mean, standard deviation and sum of squares over windows of ``m``.

    The mean and deviation are two-pass reductions over a strided window
    view: the one-pass ``sum(x^2)/m - mean^2`` form loses the cancellation
    fight on flat windows (a constant run reads as sigma ~ 1e-7 instead of
    exactly 0, which breaks the constant-window convention). The O(n * m)
    cost never dominates the O(n^2) STOMP sweep, since ``m <= n``. The sum
    of squares, used only by the non-normalized distance, keeps the cumsum
    form, which the brute-force equivalence tests hold at ~1e-13.
    """
    windows = np.lib.stride_tricks.sliding_window_view(centred, m)
    mean = windows.mean(axis=1)
    std = windows.std(axis=1)
    csum_sq = np.empty(centred.shape[0] + 1, dtype=np.float64)
    csum_sq[0] = 0.0
    np.cumsum(np.square(centred), out=csum_sq[1:])
    return mean, std, csum_sq[m:] - csum_sq[:-m]


def matrix_profile(
    series: np.ndarray[Any, Any],
    window_size: int,
    *,
    normalize: bool = True,
) -> MatrixProfileResult:
    """Compute the self-join matrix profile of a 1-D series.

    Args:
        series: 1-D series of finite values, longer than ``window_size``.
        window_size: Subsequence length ``m``; at least 3.
        normalize: Use z-normalized Euclidean distance (the matrix profile
            standard) when True, plain Euclidean when False.

    Returns:
        The profile with nearest, left and right neighbour indices, matching
        ``stumpy.stump`` conventions (see the module docstring).

    Raises:
        ValueError: If the series is not 1-D, holds non-finite values, is
            not longer than the window, or the window is shorter than 3.
    """
    t = np.asarray(series, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError(f"matrix_profile expects a 1-D series; got shape {tuple(t.shape)}.")
    if window_size < _MIN_WINDOW:
        raise ValueError(f"window_size must be at least {_MIN_WINDOW}; got {window_size}.")
    n, m = t.shape[0], int(window_size)
    if n <= m:
        raise ValueError(
            f"matrix_profile needs a series longer than window_size={m}; got {n} points."
        )
    if not np.all(np.isfinite(t)):
        raise ValueError("matrix_profile expects finite values; the series holds NaN or infinity.")

    # Both distance modes are invariant to a global shift; centring keeps the
    # rolling moments and the STOMP recurrence well conditioned.
    t = t - t.mean()
    length = n - m + 1
    zone = int(np.ceil(m / _EXCLUSION_DENOM))
    mean, std, sum_sq = _rolling_moments(t, m)
    scale = max(1.0, float(np.max(np.abs(t))))
    constant = std < _CONSTANT_STD_RTOL * scale
    sqrt_m = np.sqrt(float(m))

    profile = np.full(length, np.inf, dtype=np.float64)
    index = np.full(length, -1, dtype=np.int64)
    left_index = np.full(length, -1, dtype=np.int64)
    right_index = np.full(length, -1, dtype=np.int64)

    first_row = sliding_dot_product(t[:m], t)
    qt = first_row.copy()
    row = np.empty(length, dtype=np.float64)
    for i in range(length):
        if i > 0:
            if i % _REFRESH_INTERVAL == 0:
                qt = sliding_dot_product(t[i : i + m], t)
            else:
                # STOMP: QT(i, j) = QT(i-1, j-1) - T[j-1] T[i-1] + T[j+m-1] T[i+m-1]
                qt[1:] = qt[:-1] - t[: length - 1] * t[i - 1] + t[m : m + length - 1] * t[i + m - 1]
                qt[0] = first_row[i]
        if normalize:
            if constant[i]:
                np.copyto(row, sqrt_m)
                row[constant] = 0.0
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    rho = (qt - m * mean[i] * mean) / (m * std[i] * std)
                np.clip(rho, -1.0, 1.0, out=rho)
                np.sqrt(2.0 * m * (1.0 - rho), out=row)
                row[constant] = sqrt_m
        else:
            np.sqrt(np.maximum(sum_sq[i] + sum_sq - 2.0 * qt, 0.0), out=row)
        row[max(0, i - zone) : i + zone + 1] = np.inf

        j = int(np.argmin(row))
        if np.isfinite(row[j]):
            profile[i] = row[j]
            index[i] = j
        if i > 0:
            j_left = int(np.argmin(row[:i]))
            if np.isfinite(row[j_left]):
                left_index[i] = j_left
        if i < length - 1:
            j_right = i + 1 + int(np.argmin(row[i + 1 :]))
            if np.isfinite(row[j_right]):
                right_index[i] = j_right

    return MatrixProfileResult(
        profile=profile,
        index=index,
        left_index=left_index,
        right_index=right_index,
        window_size=m,
    )
