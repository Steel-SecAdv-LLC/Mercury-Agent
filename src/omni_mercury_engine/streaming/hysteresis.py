# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Two-threshold (Schmitt) verdict gate for streaming decision stability.

A streaming verdict that flips every few ticks is operationally unstable even
when each tick is individually defensible.  The decision layer measures that
churn (``EngineStreamSession``'s per-tick ``churn``, ``DecisionLedger``'s
summary churn); this module is the instrument that *reduces* it: a verdict
enters the anomalous state when the score reaches the enter threshold and
leaves it only when the score falls below the lower exit threshold, so scores
oscillating inside the band cannot flip the verdict.

Measured on the four committed NAB streams through the causal scoring
protocol (``benchmarks/nab_streaming_benchmark.py``), with enter at the
warmup scores' 0.95 quantile and exit at their 0.80 quantile: mean verdict
churn falls 0.0235 to 0.0153 (35% lower) and mean point F1 rises 0.3123 to
0.3722.  Leave-one-stream-out, the (0.95, 0.80) choice is selected on every
fold and improves the held-out stream's F1 on three of four (speed_7578 pays
0.051 F1 for its churn cut) — so the gate ships as a measured option, not a
silent default: the released single-threshold alarm semantics are unchanged
unless a caller opts in (``CausalMathArrest(exit_alpha=...)``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["HysteresisGate", "hysteresis_verdicts"]


class HysteresisGate:
    """Stateful two-threshold verdict gate.

    Args:
        enter_threshold: Score at or above which the verdict turns anomalous.
        exit_threshold: Score below which an anomalous verdict clears.  Must
            not exceed ``enter_threshold``.

    Raises:
        ValueError: If the thresholds are not finite or are mis-ordered.
    """

    def __init__(self, enter_threshold: float, exit_threshold: float) -> None:
        """Validate the band and start in the clear state."""
        enter = float(enter_threshold)
        leave = float(exit_threshold)
        if not (np.isfinite(enter) and np.isfinite(leave)):
            raise ValueError(f"thresholds must be finite, got ({enter}, {leave})")
        if leave > enter:
            raise ValueError(f"exit_threshold ({leave}) must not exceed enter_threshold ({enter})")
        self.enter_threshold = enter
        self.exit_threshold = leave
        self._active = False

    @property
    def active(self) -> bool:
        """Whether the gate currently holds the anomalous state."""
        return self._active

    def update(self, score: float) -> bool:
        """Advance the gate by one score and return the smoothed verdict.

        A non-finite score is a failed reading: it does not move the gate,
        and the verdict stands as it was.
        """
        value = float(score)
        if not np.isfinite(value):
            return self._active
        if not self._active and value > self.enter_threshold:
            self._active = True
        elif self._active and value <= self.exit_threshold:
            self._active = False
        return self._active

    def reset(self) -> None:
        """Clear the held state (e.g. after a refit)."""
        self._active = False


def hysteresis_verdicts(
    scores: npt.NDArray[Any],
    enter_threshold: float,
    exit_threshold: float,
) -> npt.NDArray[np.bool_]:
    """Apply the gate over a whole score array (benchmark/batch form).

    Semantically identical to feeding :class:`HysteresisGate` one score at a
    time from the clear state; kept as a function so harnesses measure the
    exact mathematics the streaming gate applies.

    Args:
        scores: 1-D score array.
        enter_threshold: See :class:`HysteresisGate`.
        exit_threshold: See :class:`HysteresisGate`.

    Returns:
        Boolean verdict per score.
    """
    gate = HysteresisGate(enter_threshold, exit_threshold)
    flat = np.asarray(scores, dtype=np.float64).ravel()
    return np.fromiter((gate.update(s) for s in flat), dtype=bool, count=flat.size)
