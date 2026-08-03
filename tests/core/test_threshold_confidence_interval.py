# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for :class:`ThresholdConfidenceIntervalCalculator`.

These lock in the bias-corrected and accelerated (BCa) bootstrap path, which
previously called the non-existent ``np.erfinv`` (NumPy exposes no ``erfinv`` --
it lives in :mod:`scipy.special`) and therefore raised ``AttributeError`` the
moment ``compute_bca`` was invoked. The path had no test coverage, so the crash
was latent. The fix uses the exact standard-normal quantile / CDF
(:func:`scipy.special.ndtri` / :func:`scipy.special.ndtr`), which also replaced
a ``tanh`` logistic *approximation* of the normal CDF that had been paired with
the exact probit, making the two BCa endpoints mutually inconsistent.

The tests below run purely on NumPy/SciPy (no torch), so they exercise the
calculator directly rather than through the engine.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import (
    HealthCheck,
    assume,
    given,
    settings,
    strategies as st,
)
from hypothesis.extra import numpy as hnp

from omni_mercury_engine.core.score_calibration import (
    ScoreDiagnostics,
    ThresholdConfidenceInterval,
    ThresholdConfidenceIntervalCalculator,
)


def _skewed_scores(n: int = 500, seed: int = 0) -> np.ndarray:
    """Mostly-normal scores with a contaminated, right-skewed upper tail."""
    rng = np.random.default_rng(seed)
    bulk = rng.normal(0.0, 1.0, n - n // 25)
    tail = rng.normal(6.0, 1.5, n // 25)
    return np.concatenate([bulk, tail]).astype(np.float64)


class TestComputeBca:
    """The BCa bootstrap interval must compute without crashing and be sane."""

    def test_runs_without_attribute_error_and_is_finite(self) -> None:
        """Direct regression: ``compute_bca`` must not raise on the erfinv path."""
        calc = ThresholdConfidenceIntervalCalculator(
            n_bootstrap=200, confidence_level=0.95, random_state=42
        )
        ci = calc.compute_bca(_skewed_scores())

        assert ci.method == "bootstrap_bca"
        assert np.isfinite([ci.lower, ci.threshold, ci.upper]).all()

    def test_interval_is_ordered(self) -> None:
        """Lower bound <= point estimate <= upper bound."""
        calc = ThresholdConfidenceIntervalCalculator(
            n_bootstrap=200, confidence_level=0.95, random_state=42
        )
        ci = calc.compute_bca(_skewed_scores(seed=3))

        assert ci.lower <= ci.threshold <= ci.upper

    def test_deterministic_for_fixed_random_state(self) -> None:
        """A fixed ``random_state`` yields byte-identical bounds across calls."""
        scores = _skewed_scores(seed=11)

        def make() -> ThresholdConfidenceInterval:
            return ThresholdConfidenceIntervalCalculator(
                n_bootstrap=200, confidence_level=0.95, random_state=7
            ).compute_bca(scores)

        a, b = make(), make()
        assert (a.lower, a.threshold, a.upper) == (b.lower, b.threshold, b.upper)

    def test_constant_scores_do_not_crash(self) -> None:
        """Degenerate (zero-variance) input must collapse to a point, not raise.

        With no spread the jackknife acceleration denominator is zero; the guard
        must keep ``a`` finite so the erfinv/ndtri call still receives a valid
        probability.
        """
        calc = ThresholdConfidenceIntervalCalculator(
            n_bootstrap=150, confidence_level=0.95, random_state=1
        )
        ci = calc.compute_bca(np.full(300, 3.0, dtype=np.float64))

        assert np.isfinite([ci.lower, ci.threshold, ci.upper]).all()
        assert ci.lower <= ci.threshold <= ci.upper

    def test_wider_confidence_level_gives_wider_interval(self) -> None:
        """Coverage monotonicity: a 99% BCa interval must not be narrower than 90%.

        This only holds if the endpoints come from a genuine, monotone normal
        quantile function -- it is a cheap guard against a future regression to a
        broken or approximate inverse-CDF.
        """
        scores = _skewed_scores(seed=5)
        narrow = ThresholdConfidenceIntervalCalculator(
            n_bootstrap=300, confidence_level=0.90, random_state=9
        ).compute_bca(scores)
        wide = ThresholdConfidenceIntervalCalculator(
            n_bootstrap=300, confidence_level=0.99, random_state=9
        ).compute_bca(scores)

        assert (wide.upper - wide.lower) >= (narrow.upper - narrow.lower)

    def test_symmetric_data_bca_close_to_percentile(self) -> None:
        """On symmetric, unbiased data BCa reduces to the percentile interval.

        With near-zero bias-correction (z0) and acceleration (a), the BCa
        endpoints collapse onto the plain percentile endpoints. A wrong normal
        CDF (e.g. the former ``tanh`` approximation, which maps the 95% level to
        ~88% central mass) would shift the endpoints well beyond this tolerance.
        """
        rng = np.random.default_rng(7)
        scores = rng.normal(0.0, 1.0, 800).astype(np.float64)
        calc = ThresholdConfidenceIntervalCalculator(
            n_bootstrap=400, confidence_level=0.95, random_state=1
        )

        percentile = calc.compute(scores)
        bca = calc.compute_bca(scores)

        assert abs(bca.lower - percentile.lower) < 0.15
        assert abs(bca.upper - percentile.upper) < 0.15


# =============================================================================
# Property-based numerical invariants for the BCa interval (the erfinv fix)
# =============================================================================
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    data=hnp.arrays(
        dtype=np.float64,
        shape=st.integers(min_value=20, max_value=60),
        elements=st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
    )
)
def test_compute_bca_is_always_finite_and_ordered(data: np.ndarray) -> None:
    """For any non-degenerate finite score array, the BCa interval is finite and
    correctly ordered (lower <= threshold <= upper). This is the invariant the
    non-existent ``np.erfinv`` used to break unconditionally."""
    assume(float(np.std(data)) > 1e-6)  # a spread must exist for a meaningful CI
    calc = ThresholdConfidenceIntervalCalculator(
        n_bootstrap=80, confidence_level=0.95, random_state=0
    )

    ci = calc.compute_bca(data)

    assert np.isfinite([ci.lower, ci.threshold, ci.upper]).all()
    assert ci.lower <= ci.threshold <= ci.upper


# =============================================================================
# Degenerate-resample regression: a descriptive statistic must not raise
# =============================================================================
@pytest.mark.parametrize(
    ("name", "scores"),
    [
        ("all identical", np.zeros(40)),
        ("subnormal span", np.concatenate([np.zeros(39), np.array([5e-324])])),
        ("tiny span", np.concatenate([np.zeros(39), np.array([1e-320])])),
        ("contains nan", np.concatenate([np.zeros(39), np.array([np.nan])])),
        ("contains inf", np.concatenate([np.zeros(39), np.array([np.inf])])),
    ],
)
def test_bimodality_on_a_degenerate_sample_answers_instead_of_raising(
    name: str, scores: np.ndarray
) -> None:
    """``np.histogram(scores, bins=50)`` raised on spans it cannot bin.

    Found by the property test above: ``compute_bca`` draws bootstrap
    resamples, and a resample of a well-spread parent can still come back
    all-but-constant. ``ValueError: Too many bins for data range`` then
    propagated out of ``_detect_bimodality`` and killed the whole confidence
    interval — a crash originating in a *diagnostic*. A sample with no spread
    has no two peaks, which is an answer, not an error.
    """
    assert ScoreDiagnostics._detect_bimodality(scores) is False, name


def test_bimodality_still_detects_two_real_peaks() -> None:
    """The other direction: the degenerate guard must not blind the heuristic."""
    rng = np.random.default_rng(0)
    two_peaks = np.concatenate([rng.normal(0.0, 0.1, 60), rng.normal(5.0, 0.1, 60)])
    assert ScoreDiagnostics._detect_bimodality(two_peaks) is True


def test_compute_bca_survives_a_near_constant_sample() -> None:
    """End-to-end: the interval computes rather than raising."""
    scores = np.concatenate([np.zeros(39), np.array([1e-9])])
    calc = ThresholdConfidenceIntervalCalculator(
        n_bootstrap=80, confidence_level=0.95, random_state=0
    )

    ci = calc.compute_bca(scores)

    assert np.isfinite([ci.lower, ci.threshold, ci.upper]).all()
    assert ci.lower <= ci.threshold <= ci.upper


def test_bca_interval_always_contains_its_own_point_estimate() -> None:
    """A CI that excludes its point estimate is not a confidence interval.

    Found by the property test above once the degenerate-sample crash stopped
    masking it. The percentile bounds are order statistics of the bootstrap
    distribution while the estimate is computed on the full sample, and
    ``AutoThresholdOptimizer`` is discontinuous -- so on low-cardinality scores
    the full sample can land on an optimum no resample reproduces. Measured on
    this exact input before the fix: ``threshold=1.0`` with
    ``CI [0.96975, 0.97299]``.
    """
    data = np.array(
        [
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
        ]
    )
    calc = ThresholdConfidenceIntervalCalculator(
        n_bootstrap=80, confidence_level=0.95, random_state=0
    )

    ci = calc.compute_bca(data)

    assert ci.lower <= ci.threshold <= ci.upper
    # The widening is disclosed rather than silently applied.
    assert ci.method == "bootstrap_bca_widened"


def test_a_clean_fit_is_not_labelled_as_widened() -> None:
    """The disclosure must be specific, or it means nothing."""
    calc = ThresholdConfidenceIntervalCalculator(
        n_bootstrap=200, confidence_level=0.95, random_state=0
    )

    ci = calc.compute_bca(_skewed_scores(400, seed=3))

    assert ci.method == "bootstrap_bca"
    assert ci.lower <= ci.threshold <= ci.upper
