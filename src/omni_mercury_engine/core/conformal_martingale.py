# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sequential change detection with a conformal test martingale.

Tests a stream of nonconformity scores (higher is stranger) online for a rise
away from a reference sample, with no model of either distribution:

- p-values: the smoothed conformal p-value p_n = (#{a in bag: a > a_n} +
  U_n·(#{a in bag: a = a_n} + 1)) / (|bag| + 1) with U_n uniform on [0, 1],
  where the bag holds the reference scores and every earlier stream score; a_n
  then joins the bag. Under exchangeability these p-values are independent and
  uniform, ties included (Vovk, Nouretdinov & Gammerman 2003, "Testing
  exchangeability on-line", ICML; Vovk, Gammerman & Shafer 2022, *Algorithmic
  Learning in a Random World*, 2nd ed.). Without U_n, tied scores can make
  p_n smaller than uniform given the past, and the guarantee below fails.
- Betting: the one-sided Simple Jumper (Vovk et al. 2021, "Retrain or not
  retrain: conformal test martingales for change-point detection", PMLR 152).
  Capital is split between f(p) = 1.5 - p, a bet on small p-values (high
  scores), and f(p) = 1, no bet; each step a fraction J of it is spread evenly
  over both, so it can move to the bet when a change starts. Each step's factor
  is fixed before p_n is seen and integrates to 1 over [0, 1], so the capital
  S_n is a test martingale.
- Alarm: the Shiryaev-Roberts statistic R_n = Σ_{k<n} S_n/S_k, the sum over k
  of the capital the same bets gained after observation k, computed as
  R_n = (S_n/S_{n-1})(R_{n-1} + 1). R_n - n is a martingale under
  exchangeability, so by optional stopping the mean time from a reset to an
  alarm R_n >= c is at least c observations, whatever the score distribution.
  Since S_n/S_{n-1} <= 1.5, an alarm overshoots c by little (R_n < 1.5(c + 1)),
  and the bound is nearly tight in simulation (see the tests).
  With known pre- and post-change densities this procedure is optimal for that
  constraint in repeated monitoring (Pollak & Tartakovsky 2009, *Stat. Sin.*
  19:1729); here the betting function stands in for the unknown likelihood
  ratio.

Limitations: the null is exchangeability, so drift, seasonality or
autocorrelation in the scores also raise the statistic. The bag grows by one
score per observation. After a change the bag absorbs post-change scores and
the p-values return toward uniform, so an alarm calls for a refit of the
reference, not for continued monitoring against it.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["ChangeStatus", "ConformalChangeDetector"]


@dataclass(frozen=True)
class ChangeStatus:
    """State of the change test after one observation.

    Attributes:
        p_value: Conformal p-value of the observation against the bag.
        log_statistic: Natural log of the Shiryaev-Roberts statistic R_n.
        alarm: Whether R_n reached the threshold; the test then restarts.
        observations: Observations since the last restart, this one included.
    """

    p_value: float
    log_statistic: float
    alarm: bool
    observations: int


class ConformalChangeDetector:
    """Conformal Shiryaev-Roberts test for a rise in a stream of scores."""

    def __init__(
        self,
        reference_scores: Iterable[float],
        *,
        mean_time_to_false_alarm: float = 1000.0,
        jump_rate: float = 0.01,
        seed: int | None = None,
    ) -> None:
        """Seed the bag with the reference scores and start the test.

        Args:
            reference_scores: Scores from the reference regime (for a detector, its
                held-out calibration scores); they seed the bag.
            mean_time_to_false_alarm: Threshold c of R_n. While the stream is
                exchangeable with the reference, alarms are at least this many
                observations apart on average.
            jump_rate: Fraction J of the capital redistributed each step.
            seed: Seed for the uniform draws of the smoothed p-values.

        Raises:
            ValueError: If a reference score is not a finite number, ``c <= 1`` or
                ``J`` is outside (0, 1].
        """
        bag = sorted(_finite_score(score, "reference_scores") for score in reference_scores)
        if not (math.isfinite(mean_time_to_false_alarm) and mean_time_to_false_alarm > 1.0):
            raise ValueError(
                f"mean_time_to_false_alarm must be finite and > 1, got {mean_time_to_false_alarm}"
            )
        if not 0.0 < jump_rate <= 1.0:
            raise ValueError(f"jump_rate must be in (0, 1], got {jump_rate}")
        self._bag = bag
        self.mean_time_to_false_alarm = float(mean_time_to_false_alarm)
        self.jump_rate = float(jump_rate)
        self._log_threshold = math.log(self.mean_time_to_false_alarm)
        self._rng = np.random.default_rng(seed)
        self.reset()

    @property
    def bag_size(self) -> int:
        """Reference scores plus every score observed so far."""
        return len(self._bag)

    def reset(self) -> None:
        """Restart the test; the bag is kept."""
        self._bet = 0.5  # share of the capital on f(p) = 1.5 - p
        self._log_statistic = -math.inf
        self._observations = 0

    def update(self, score: float) -> ChangeStatus:
        """Add one observation's score, oldest first, and test for a change.

        Raises:
            ValueError: If ``score`` is not a finite number.
        """
        value = _finite_score(score, "score")
        above = bisect.bisect_right(self._bag, value)
        greater = len(self._bag) - above
        equal = above - bisect.bisect_left(self._bag, value)
        p_value = (greater + float(self._rng.random()) * (equal + 1)) / (len(self._bag) + 1)
        bisect.insort(self._bag, value)

        bet = (1.0 - self.jump_rate) * self._bet + self.jump_rate / 2.0
        factor = bet * (1.5 - p_value) + (1.0 - bet)
        self._bet = bet * (1.5 - p_value) / factor
        log_factor = math.log(factor)
        self._log_statistic = log_factor + _log_add(self._log_statistic, 0.0)
        self._observations += 1

        status = ChangeStatus(
            p_value=p_value,
            log_statistic=self._log_statistic,
            alarm=self._log_statistic >= self._log_threshold,
            observations=self._observations,
        )
        if status.alarm:
            self.reset()
        return status


def _finite_score(value: float, name: str) -> float:
    """``value`` as a finite float; booleans are not scores."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numbers, not booleans")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be finite numbers, got {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite numbers, got {value!r}")
    return number


def _log_add(a: float, b: float) -> float:
    """log(e^a + e^b) without overflow."""
    if a == -math.inf:
        return b
    high, low = (a, b) if a >= b else (b, a)
    return high + math.log1p(math.exp(low - high))
