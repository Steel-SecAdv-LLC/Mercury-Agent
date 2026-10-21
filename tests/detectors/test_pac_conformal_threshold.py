# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""``detection_tier.pac_conformal_threshold``: a training-conditional false-alarm bound.

With uniform calibration scores the false-positive rate of a threshold t is
exactly 1 - t, so each calibration's own rate is known without sampling.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.detection_tier import (
    conformal_threshold,
    pac_conformal_threshold,
    pac_minimum_calibration_size,
)


@pytest.mark.parametrize(
    ("n", "alpha", "delta"), [(300, 0.01, 0.1), (300, 0.05, 0.1), (1000, 0.01, 0.05)]
)
def test_own_rate_exceeds_alpha_with_probability_at_most_delta(
    n: int, alpha: float, delta: float
) -> None:
    """3 standard errors of 2,000 calibrations are allowed above delta."""
    rng = np.random.default_rng(0)
    rates = np.array(
        [1.0 - pac_conformal_threshold(rng.random(n), alpha, delta) for _ in range(2000)]
    )
    assert np.mean(rates > alpha) <= delta + 3 * np.sqrt(delta * (1 - delta) / 2000)


def test_the_marginal_threshold_gives_no_such_bound() -> None:
    """About half of single calibrations exceed alpha with the average-case threshold."""
    rng = np.random.default_rng(1)
    rates = np.array([1.0 - conformal_threshold(rng.random(300), 0.01) for _ in range(2000)])
    assert np.mean(rates > 0.01) > 0.3


def test_smallest_sufficient_sample_uses_the_maximum() -> None:
    """(1 - 0.01)^n <= 0.1 first holds at n = 230; there the maximum is the threshold."""
    scores = np.random.default_rng(2).random(230)
    assert pac_conformal_threshold(scores, 0.01, 0.1) == scores.max()
    with pytest.raises(ValueError, match="at least 230"):
        pac_conformal_threshold(scores[:229], 0.01, 0.1)


@pytest.mark.parametrize(("alpha", "n"), [(0.05, 59), (0.1, 10), (0.01, 230), (0.2, 7)])
def test_boundary_sample_size_never_returns_the_minimum(alpha: float, n: int) -> None:
    """delta = (1 - alpha)^n is the boundary; there the result is the maximum or an error.

    At (0.05, 59) the Beta CDF rounds below 1 - delta; the threshold once fell to the
    smallest score instead of raising.
    """
    scores = np.arange(1.0, n + 1.0)
    delta = (1.0 - alpha) ** n
    needed = pac_minimum_calibration_size(alpha, delta)
    if needed <= n:
        assert pac_conformal_threshold(scores, alpha, delta) == scores.max()
    else:
        with pytest.raises(ValueError, match=f"at least {needed} are needed"):
            pac_conformal_threshold(scores, alpha, delta)
        assert pac_conformal_threshold(np.arange(1.0, needed + 1.0), alpha, delta) == needed


@pytest.mark.parametrize(("alpha", "delta", "expected"), [(0.01, 0.1, 230), (0.05, 0.1, 45)])
def test_minimum_size_is_the_first_n_with_the_guarantee(
    alpha: float, delta: float, expected: int
) -> None:
    assert pac_minimum_calibration_size(alpha, delta) == expected
    assert (1.0 - alpha) ** expected <= delta < (1.0 - alpha) ** (expected - 1)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_scores_are_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        pac_conformal_threshold(np.r_[np.arange(300.0), bad], 0.01, 0.1)


@pytest.mark.parametrize(("alpha", "delta"), [(0.0, 0.1), (1.0, 0.1), (0.01, 0.0), (0.01, 1.0)])
def test_settings_are_validated(alpha: float, delta: float) -> None:
    with pytest.raises(ValueError, match="must be in"):
        pac_conformal_threshold(np.arange(1000.0), alpha, delta)
    with pytest.raises(ValueError, match="must be in"):
        pac_minimum_calibration_size(alpha, delta)
