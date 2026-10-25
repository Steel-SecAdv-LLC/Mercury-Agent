# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``core.conformal_martingale``: the conformal Shiryaev-Roberts change test."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from scipy.stats import kstest

from omni_mercury_engine.core.conformal_martingale import ConformalChangeDetector

if TYPE_CHECKING:
    from collections.abc import Callable


def _alarms(detector: ConformalChangeDetector, scores: np.ndarray) -> list[int]:
    """1-based positions in ``scores`` at which the test alarmed."""
    return [i for i, score in enumerate(scores, 1) if detector.update(score).alarm]


@pytest.mark.parametrize(
    "draw",
    [
        lambda rng, n: rng.normal(size=n),
        lambda rng, n: rng.pareto(1.5, n),
        lambda rng, n: rng.integers(0, 5, n).astype(float),
        lambda rng, n: (rng.random(n) < 0.9).astype(float),
    ],
    ids=["normal", "pareto", "five-level-ties", "bernoulli-ties"],
)
def test_mean_time_to_false_alarm_is_at_least_c(draw: Callable[..., np.ndarray]) -> None:
    """Under exchangeability alarms are at least c apart on average, for any scores.

    The bound is nearly tight (1.03-1.08 c measured), so the test allows for the
    sampling error of about 750 gaps: 0.85 c is over 5 standard errors below the
    expected gap, and 1.5 c catches a test that rarely alarms.
    """
    rng = np.random.default_rng(0)
    c = 50.0
    detector = ConformalChangeDetector(draw(rng, 300), mean_time_to_false_alarm=c, seed=0)
    alarms = _alarms(detector, draw(rng, 40_000))
    assert len(alarms) >= 500
    assert 0.85 * c <= alarms[-1] / len(alarms) <= 1.5 * c


def test_p_values_are_uniform_with_ties() -> None:
    """Smoothed p-values are uniform even when most scores tie."""
    rng = np.random.default_rng(1)
    scores = (rng.random(3000) < 0.9).astype(float)
    detector = ConformalChangeDetector(scores[:2], seed=1)
    p_values = [detector.update(score).p_value for score in scores[2:]]
    assert kstest(p_values, "uniform").pvalue > 0.01


def test_rise_is_detected_after_it_starts() -> None:
    rng = np.random.default_rng(2)
    detector = ConformalChangeDetector(rng.normal(size=300), mean_time_to_false_alarm=1000.0)
    for score in rng.normal(size=100):
        detector.update(score)
    detector.reset()
    after = [detector.update(score).alarm for score in rng.normal(2.0, 1.0, 200)]
    assert after.index(True) < 60


def test_a_fall_is_not_bet_on() -> None:
    """One-sided: a fall alarms no more often than the null allows; a rise far more."""
    c, n = 100.0, 2000

    def count(shift: float) -> int:
        rng = np.random.default_rng(3)
        detector = ConformalChangeDetector(rng.normal(size=300), mean_time_to_false_alarm=c, seed=3)
        return len(_alarms(detector, rng.normal(shift, 1.0, n)))

    assert count(-3.0) <= n / c
    assert count(3.0) >= 3 * n / c


def test_p_values_are_smoothed_ranks() -> None:
    detector = ConformalChangeDetector([1.0, 2.0, 3.0], seed=0)
    assert 0.0 < detector.update(10.0).p_value <= 1 / 4  # nothing at or above it
    assert 4 / 5 <= detector.update(-5.0).p_value <= 1.0  # everything above it
    assert 2 / 6 <= detector.update(2.0).p_value <= 4 / 6  # 3 and 10 above, one tie
    assert detector.bag_size == 6


def test_seeded_detector_is_reproducible() -> None:
    scores = np.random.default_rng(4).normal(size=200)
    a = ConformalChangeDetector(scores[:50], seed=9)
    b = ConformalChangeDetector(scores[:50], seed=9)
    assert [a.update(s) for s in scores[50:]] == [b.update(s) for s in scores[50:]]


def test_restarts_after_an_alarm_and_keeps_the_bag() -> None:
    detector = ConformalChangeDetector(np.zeros(50), mean_time_to_false_alarm=5.0)
    statuses = [detector.update(1.0) for _ in range(20)]
    first = next(i for i, status in enumerate(statuses) if status.alarm)
    assert statuses[first + 1].observations == 1
    assert detector.bag_size == 70


def test_log_space_reaches_a_threshold_near_the_float_limit() -> None:
    """R can reach 1.5 (c + 1), past the largest float for this c; log R cannot overflow."""
    detector = ConformalChangeDetector(np.zeros(10_000), mean_time_to_false_alarm=1.7e308)
    statuses = [detector.update(1.0 + i) for i in range(2500)]
    assert all(math.isfinite(status.log_statistic) for status in statuses)
    assert any(status.alarm for status in statuses)


@pytest.mark.parametrize(
    ("reference", "kwargs", "match"),
    [
        ([0.0, math.nan], {}, "reference_scores"),
        ([0.0, True], {}, "reference_scores"),
        ([0.0], {"mean_time_to_false_alarm": 1.0}, "> 1"),
        ([0.0], {"mean_time_to_false_alarm": math.inf}, "finite"),
        ([0.0], {"jump_rate": 0.0}, "jump_rate"),
        ([0.0], {"jump_rate": 1.5}, "jump_rate"),
    ],
)
def test_settings_are_validated(reference: list[float], kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ConformalChangeDetector(reference, **kwargs)


@pytest.mark.parametrize("score", [math.nan, math.inf, True, np.True_, 10**400])
def test_bad_scores_are_rejected(score: float) -> None:
    detector = ConformalChangeDetector([0.0])
    with pytest.raises(ValueError, match="score"):
        detector.update(score)
    assert detector.bag_size == 1
