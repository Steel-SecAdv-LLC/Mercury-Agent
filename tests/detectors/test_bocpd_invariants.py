# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""BOCPD run-length posterior mass-conservation invariant tests.

Asserts the core BOCPD contract that the run-length posterior is a proper
probability distribution -- it sums to 1 at *every* step -- via the
:meth:`BOCPDDetector.run_length_posteriors` accessor, on fixed fixtures and on
Hypothesis-generated arbitrary (finite and non-finite) series.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.bocpd import BOCPDDetector


def _regime_shift(seed: int = 0, n: int = 240) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0.0, 1.0, n // 2), rng.normal(6.0, 1.0, n // 2)])


class TestMassConservation:
    def test_posterior_sums_to_one_each_step(self) -> None:
        series = _regime_shift()
        det = BOCPDDetector(max_run_length=80).fit(series)
        post = det.run_length_posteriors(series)
        assert post.shape == (series.size, det.max_run_length + 1)
        sums = post.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-9), "run-length posterior must sum to 1 each step"
        assert np.all(post >= 0.0), "posterior mass must be non-negative"

    def test_scores_are_posterior_tail_mass(self) -> None:
        series = _regime_shift(1)
        det = BOCPDDetector(change_grace=5, max_run_length=100).fit(series)
        post = det.run_length_posteriors(series)
        scores = np.asarray(det.detect(series)["scores"])
        grace = min(det.change_grace, det.max_run_length + 1)
        expected = post[:, :grace].sum(axis=1)
        np.testing.assert_allclose(scores, np.clip(expected, 0.0, 1.0), atol=1e-6)

    def test_constant_series_conserves_mass(self) -> None:
        det = BOCPDDetector(max_run_length=50).fit(np.full(120, 3.0))
        post = det.run_length_posteriors(np.full(120, 3.0))
        assert np.allclose(post.sum(axis=1), 1.0, atol=1e-9)

    def test_nonfinite_input_still_conserves_mass(self) -> None:
        # NaN policy sanitises the input, so the recursion still yields proper
        # distributions summing to 1.
        series = _regime_shift(2).copy()
        series[10] = np.nan
        series[20] = np.inf
        det = BOCPDDetector(max_run_length=60).fit(series)
        post = det.run_length_posteriors(series)
        assert np.all(np.isfinite(post))
        assert np.allclose(post.sum(axis=1), 1.0, atol=1e-9)


hyp = pytest.importorskip("hypothesis")
from hypothesis import (
    given,
    settings,
    strategies as st,
)
from hypothesis.extra import numpy as hnp


class TestMassConservationProperty:
    @settings(max_examples=60, deadline=None)
    @given(
        hnp.arrays(
            dtype=np.float64,
            shape=st.integers(12, 120),
            elements=st.floats(-50.0, 50.0, allow_nan=False, allow_infinity=False),
        )
    )
    def test_arbitrary_finite_series_conserve_mass(self, series: np.ndarray) -> None:
        det = BOCPDDetector(max_run_length=40).fit(series)
        post = det.run_length_posteriors(series)
        sums = post.sum(axis=1)
        assert np.all(np.isfinite(sums))
        assert np.allclose(sums, 1.0, atol=1e-8)

    @settings(max_examples=40, deadline=None)
    @given(
        hnp.arrays(
            dtype=np.float64,
            shape=st.integers(12, 80),
            elements=st.floats(allow_nan=True, allow_infinity=True, width=64),
        )
    )
    def test_arbitrary_nonfinite_series_conserve_mass(self, series: np.ndarray) -> None:
        det = BOCPDDetector(max_run_length=30).fit(series)
        post = det.run_length_posteriors(series)
        assert np.all(np.isfinite(post))
        assert np.allclose(post.sum(axis=1), 1.0, atol=1e-8)
        # And the emitted scores remain valid probabilities.
        scores = np.asarray(det.detect(series)["scores"])
        assert np.all((scores >= 0.0) & (scores <= 1.0))
