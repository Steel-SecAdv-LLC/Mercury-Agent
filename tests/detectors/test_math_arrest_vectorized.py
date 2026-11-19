# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The vectorized probe hot paths compute the released scalar mathematics.

Three probes carried pure-Python loops on the forward path; each is held here
to a scalar reference that is a verbatim transcription of the released loop:

* ``LyapunovChaosProbe._compute_divergence`` was an O(n^2) double loop —
  2.59 s of a 2.69 s 21-probe ``detect`` at 4 096 samples — and is required to
  be bit-identical to it (same first-index tie-breaks, same EPSILON guards).
* ``FractalSelfSimilarityProbe._compute_similarities`` called ``np.corrcoef``
  per window; the sliding-window form computes the same statistic to float
  round-off.
* ``_ewma`` ran the IIR recursion in Python inside ``minimize_scalar``'s
  objective; the ``lfilter`` form is required to be bit-identical.

The latency floor at the bottom is what fails on the looped code: the scalar
Lyapunov scan takes seconds where the vectorized scan takes milliseconds.
"""

from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
import pytest

from omni_mercury_engine.detectors.math_arrest.base_probe import EPSILON
from omni_mercury_engine.detectors.math_arrest.probes.exponential import _ewma
from omni_mercury_engine.detectors.math_arrest.probes.fractal_similarity import (
    FractalSelfSimilarityProbe,
)
from omni_mercury_engine.detectors.math_arrest.probes.lyapunov_chaos import (
    LyapunovChaosProbe,
)


def _reference_divergence(
    x: npt.NDArray[np.float64], k: int, exclusion: int = 3
) -> npt.NDArray[np.float64]:
    """The released Lyapunov loop, verbatim: the specification."""
    n = len(x)
    valid_end = n - k
    divergences: list[float] = []
    for i in range(valid_end):
        best_dist = np.inf
        best_idx = -1
        for j in range(valid_end):
            if abs(i - j) <= exclusion:
                continue
            dist = abs(x[i] - x[j])
            if dist < best_dist:
                best_dist = dist
                best_idx = j
        if best_idx < 0 or best_dist < EPSILON:
            divergences.append(0.0)
            continue
        future_dist = abs(x[i + k] - x[best_idx + k])
        if future_dist < EPSILON:
            divergences.append(0.0)
        else:
            divergences.append(float(np.log(future_dist / best_dist)) / k)
    return np.array(divergences, dtype=np.float64)


def _reference_similarities(
    x: npt.NDArray[np.float64], x_scaled: npt.NDArray[np.float64], w: int
) -> npt.NDArray[np.float64]:
    """The released per-window ``np.corrcoef`` loop, verbatim."""
    n = len(x)
    max_windows = min(n - w + 1, len(x_scaled) - w + 1)
    if max_windows <= 0:
        return np.zeros(n, dtype=np.float64)
    similarities: list[float] = []
    for i in range(max_windows):
        win_orig = x[i : i + w]
        win_scaled = x_scaled[i : i + w]
        std_o = float(np.std(win_orig))
        std_s = float(np.std(win_scaled))
        if std_o < EPSILON and std_s < EPSILON:
            similarities.append(1.0)
        elif std_o < EPSILON or std_s < EPSILON:
            similarities.append(0.0)
        else:
            corr = np.corrcoef(win_orig, win_scaled)[0, 1]
            similarities.append(float(np.nan_to_num(corr, nan=0.0)))
    result = np.array(similarities, dtype=np.float64)
    if len(result) < n:
        result = np.concatenate([result, np.zeros(n - len(result), dtype=np.float64)])
    return result[:n]


def _reference_ewma(x: npt.NDArray[np.float64], lam: float) -> npt.NDArray[np.float64]:
    """The released EWMA recursion, verbatim."""
    y = np.empty(len(x), dtype=np.float64)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = (1.0 - lam) * y[i - 1] + lam * x[i]
    return y


def _signals() -> dict[str, npt.NDArray[np.float64]]:
    rng = np.random.default_rng(7)
    base = np.sin(np.linspace(0.0, 24.0, 300)) + 0.1 * rng.standard_normal(300)
    spiked = base.copy()
    spiked[140:146] += 6.0
    ties = np.repeat(rng.standard_normal(30), 10)  # many exact-distance ties
    constant = np.ones(120, dtype=np.float64)
    short = rng.standard_normal(25)
    return {
        "noisy_sine": base,
        "spiked": spiked,
        "tied_values": ties.astype(np.float64),
        "constant": constant,
        "short": short,
    }


@pytest.mark.parametrize("name", sorted(_signals()))
def test_lyapunov_divergence_is_bit_identical_to_the_scalar_scan(name: str) -> None:
    """Same nearest neighbour, same tie-breaks, same guards, bit for bit."""
    x = _signals()[name]
    k = max(1, len(x) // 50)
    got = LyapunovChaosProbe._compute_divergence(x, k)
    expected = _reference_divergence(x, k)
    np.testing.assert_array_equal(got, expected)


def test_lyapunov_divergence_chunking_does_not_change_the_result() -> None:
    """The chunk size is a memory bound, not a parameter of the mathematics."""
    x = _signals()["noisy_sine"]
    k = max(1, len(x) // 50)
    whole = LyapunovChaosProbe._compute_divergence(x, k)
    original = LyapunovChaosProbe._CHUNK_ELEMENTS
    try:
        LyapunovChaosProbe._CHUNK_ELEMENTS = 17  # force many tiny chunks
        chunked = LyapunovChaosProbe._compute_divergence(x, k)
    finally:
        LyapunovChaosProbe._CHUNK_ELEMENTS = original
    np.testing.assert_array_equal(whole, chunked)


def test_lyapunov_divergence_handles_nan_like_the_scalar_scan() -> None:
    """A NaN sample is never selected as a neighbour, exactly as before."""
    x = _signals()["noisy_sine"].copy()
    x[50] = np.nan
    x[200] = np.nan
    k = max(1, len(x) // 50)
    got = LyapunovChaosProbe._compute_divergence(x, k)
    expected = _reference_divergence(x, k)
    # The reference emits NaN where the selected neighbour's future is NaN;
    # both paths are consumed through np.nan_to_num by the probe.
    np.testing.assert_array_equal(
        np.nan_to_num(got, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(expected, nan=0.0, posinf=0.0, neginf=0.0),
    )


@pytest.mark.parametrize("name", sorted(_signals()))
def test_fractal_similarities_match_the_per_window_corrcoef(name: str) -> None:
    x = _signals()[name]
    probe = FractalSelfSimilarityProbe()
    probe._window = max(5, len(x) // 20)
    x_scaled = probe._resample_at_phi(x)
    got = probe._compute_similarities(x)
    expected = _reference_similarities(x, x_scaled, probe._window)
    np.testing.assert_allclose(got, expected, atol=1e-9)


@pytest.mark.parametrize("lam", (0.001, 0.1, 0.9, 1.5, 1.999))
def test_ewma_is_bit_identical_to_the_recursion(lam: float) -> None:
    x = _signals()["noisy_sine"]
    np.testing.assert_array_equal(_ewma(x, lam), _reference_ewma(x, lam))
    single = np.array([3.5])
    np.testing.assert_array_equal(_ewma(single, lam), _reference_ewma(single, lam))


def test_the_lyapunov_hot_path_stays_vectorized() -> None:
    """The latency floor the scalar loop cannot meet.

    Measured on the development container: the released double loop took
    2.59 s at 4 096 samples; the chunked evaluation takes ~0.02 s. The bound
    leaves more than an order of magnitude for slower CI runners while
    remaining far below what any per-sample Python scan can reach.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4096)
    probe = LyapunovChaosProbe()
    probe.fit_trajectory(x[:1024])
    start = time.perf_counter()
    probe.deviation_score(x)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, (
        f"LyapunovChaosProbe.deviation_score took {elapsed:.2f}s at 4096 "
        "samples; the vectorized scan runs in milliseconds, so a per-sample "
        "Python loop has come back"
    )
