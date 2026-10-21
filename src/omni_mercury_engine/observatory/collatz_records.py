# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Collatz records instrument: exhaustive trajectories, record anomalies, refutation hook.

For every ``n`` up to the limit the instrument iterates the Collatz map
(``3n + 1`` for odd, ``n / 2`` for even) to 1, exactly -- no sampling, no
model.  Two record ladders are reported:

* **stopping-time records** -- the ``n`` whose total stopping time exceeds
  every smaller number's (the classic ladder: 27 takes 111 steps and climbs
  to 9,232 before falling);
* **excursion records** -- the ``n`` whose trajectory peak exceeds every
  smaller number's.

Each record carries an ``anomaly_score``: its stopping time divided by the
heuristic average ``2 / ln(4/3) * ln(n)`` (~6.952 ln n) from the standard
geometric-drift argument -- documented as the heuristic it is, not a theorem.
A score far above 1 marks a trajectory that resists the average drift, which
is exactly what makes record-setters objects of study.

The refutation hook: a trajectory that fails to reach 1 within the step
budget is a **counterexample candidate**, routable to
:meth:`omni_mercury_engine.verifiers.registry.MysteryRegistry.submit_collatz`
with a larger budget -- a genuine non-terminator or divergent cycle would
settle the conjecture.  (Every ``n`` below 2^68 is verified to reach 1,
Barina 2020; within that range the candidate list is empty by theorem.)

Pure Python and fully deterministic; no optional dependencies.
"""

from __future__ import annotations

import math
from typing import Any

#: Heuristic mean total stopping time is HEURISTIC_DRIFT * ln(n): each pair
#: of steps multiplies an odd n by ~3/4 on average, so ~2 / ln(4/3) steps
#: per factor of e.
HEURISTIC_DRIFT = 2.0 / math.log(4.0 / 3.0)


class CollatzRecordsInstrument:
    """Exhaustively measure Collatz trajectories and flag record anomalies."""

    def observe(self, limit: int = 50000, *, step_budget: int = 1_000_000) -> dict[str, Any]:
        """Iterate every trajectory for ``n`` in ``2..limit``.

        Args:
            limit: Largest starting value checked (>= 10; ~3 s at 50,000).
            step_budget: Steps allowed per trajectory before the number is
                reported as a counterexample candidate instead of forced.

        Returns:
            A dict with ``status``, ``oracle``, ``stopping_time_records``
            (each with ``n``, ``stopping_time``, ``peak``,
            ``anomaly_score``), ``excursion_records``, and
            ``counterexample_candidates``.

        Raises:
            ValueError: ``limit < 10`` or ``step_budget < 1`` -- refused.
        """
        if limit < 10:
            raise ValueError(f"limit must be >= 10, got {limit}")
        if step_budget < 1:
            raise ValueError("step_budget must be >= 1")

        stopping_records: list[dict[str, Any]] = []
        excursion_records: list[dict[str, Any]] = []
        counterexamples: list[int] = []
        best_steps = -1
        best_peak = -1

        for n in range(2, limit + 1):
            steps, peak, reached_one = self._trajectory(n, step_budget)
            if not reached_one:
                counterexamples.append(n)
                continue
            if steps > best_steps:
                best_steps = steps
                stopping_records.append(
                    {
                        "n": n,
                        "stopping_time": steps,
                        "peak": peak,
                        "anomaly_score": steps / (HEURISTIC_DRIFT * math.log(n)),
                    }
                )
            if peak > best_peak:
                best_peak = peak
                excursion_records.append({"n": n, "peak": peak, "stopping_time": steps})

        return {
            "status": "grounded",
            "oracle": "exhaustive_iteration",
            "limit": limit,
            "stopping_time_records": stopping_records,
            "excursion_records": excursion_records,
            "counterexample_candidates": counterexamples,
            "notes": [
                "anomaly_score divides the stopping time by the heuristic "
                f"average {HEURISTIC_DRIFT:.5f} * ln(n) from the geometric-"
                "drift argument -- a heuristic, not a theorem",
                "a counterexample candidate routes to "
                "MysteryRegistry.submit_collatz with a larger budget; every n "
                "below 2^68 is verified to reach 1 (Barina 2020)",
            ],
        }

    @staticmethod
    def _trajectory(n: int, step_budget: int) -> tuple[int, int, bool]:
        """Run one trajectory; return ``(steps, peak, reached_one)``."""
        steps = 0
        peak = n
        value = n
        while value != 1 and steps < step_budget:
            value = 3 * value + 1 if value % 2 else value // 2
            peak = max(peak, value)
            steps += 1
        return steps, peak, value == 1
