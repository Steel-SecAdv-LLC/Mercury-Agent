# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Goldbach comet instrument: measure structure, explain it, flag what's left.

For each even ``n`` the instrument counts the unordered Goldbach partitions
``r(n) = #{(p, q): p <= q, p + q = n, both prime}`` by exhaustive sieve --
these counts, plotted against ``n``, are the famous "Goldbach comet" with
its visible bands.  Three honest measurements follow:

* **The bands are real**: the raw density ``r(n) * ln(n)^2 / n`` is reported
  per residue class of ``n`` mod 3 -- numbers divisible by 3 sit on a band
  roughly twice as high.
* **The bands are explained**: the extended Goldbach conjecture of Hardy &
  Littlewood (1923) predicts, for unordered partitions,

      r(n) ~ C2 * prod_{p | n, p odd prime} (p - 1) / (p - 2) * n / ln(n)^2,

  with the twin-prime constant ``C2 = 0.6601618...``.  The singular-series
  product carries exactly the factor 2 for ``3 | n``, so the ratio
  ``r(n) / prediction`` collapses the bands -- an observed anomaly resolved
  by a published theory, which is the whole method in miniature.
* **What's left is flagged**: the largest ``|log(ratio)|`` outliers are
  reported as anomalies (clues), and any even ``n >= 4`` with ``r(n) = 0``
  is a **counterexample candidate** -- the one observation that would settle
  Goldbach, routable to
  :meth:`omni_mercury_engine.verifiers.registry.MysteryRegistry.submit_goldbach`,
  whose oracle can GROUND the refutation.  (None exists below 4e18,
  Oliveira e Silva et al. 2014.)

Pure Python and fully deterministic; no optional dependencies.
"""

from __future__ import annotations

import math
from typing import Any

#: Hardy--Littlewood twin-prime constant C2 = prod_{p > 2} (1 - 1/(p-1)^2).
TWIN_PRIME_CONSTANT = 0.6601618158468696


class GoldbachCometInstrument:
    """Count Goldbach partitions and measure them against Hardy--Littlewood."""

    #: Smallest even number with a partition (4 = 2 + 2).
    _FIRST_EVEN = 4

    def observe(self, limit: int = 10000, *, top_k: int = 5) -> dict[str, Any]:
        """Measure ``r(n)`` for all even ``n`` up to ``limit``.

        Args:
            limit: Upper bound (inclusive) for the even numbers checked;
                at least 100 so the statistics mean something.
            top_k: How many of the largest ``|log(r/prediction)|`` outliers
                to report as anomalies.

        Returns:
            A dict with ``status``, ``oracle``, the per-band raw densities
            (``band_density_by_mod3``), ratio statistics against the
            Hardy--Littlewood prediction, ``ratio_outliers``, and
            ``counterexample_candidates`` (even n with no partition).

        Raises:
            ValueError: ``limit < 100`` or ``top_k < 1`` -- refused, never
                silently adjusted.
        """
        if limit < 100:
            raise ValueError(f"limit must be >= 100, got {limit}")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        is_prime = self._sieve(limit)
        primes = [p for p in range(2, limit + 1) if is_prime[p]]

        counterexamples: list[int] = []
        density_sums = {0: 0.0, 1: 0.0, 2: 0.0}
        density_counts = {0: 0, 1: 0, 2: 0}
        log_ratios: list[tuple[float, int, int, float]] = []

        for n in range(self._FIRST_EVEN, limit + 1, 2):
            count = 0
            for p in primes:
                if p > n // 2:
                    break
                if is_prime[n - p]:
                    count += 1
            if count == 0:
                counterexamples.append(n)
                continue
            raw_density = count * math.log(n) ** 2 / n
            residue = n % 3
            density_sums[residue] += raw_density
            density_counts[residue] += 1
            prediction = self._hardy_littlewood(n)
            log_ratios.append((abs(math.log(count / prediction)), n, count, prediction))

        log_ratios.sort(reverse=True)
        outliers = [
            {
                "n": n,
                "partitions": count,
                "hardy_littlewood_prediction": prediction,
                "abs_log_ratio": abs_log,
            }
            for abs_log, n, count, prediction in log_ratios[:top_k]
        ]
        bands = {
            residue: (density_sums[residue] / density_counts[residue])
            for residue in (0, 1, 2)
            if density_counts[residue]
        }

        return {
            "status": "grounded",
            "oracle": "exhaustive_sieve",
            "limit": limit,
            "evens_checked": (limit - self._FIRST_EVEN) // 2 + 1,
            "band_density_by_mod3": bands,
            "ratio_outliers": outliers,
            "counterexample_candidates": counterexamples,
            "notes": [
                "band_density_by_mod3 shows the comet's bands (3 | n roughly "
                "doubles the density); the Hardy-Littlewood singular series "
                "carries exactly that factor, so the ratio collapses the bands",
                "a counterexample candidate (r(n) = 0) routes to "
                "MysteryRegistry.submit_goldbach, whose oracle can ground the "
                "refutation; none exists below 4e18 (Oliveira e Silva 2014)",
            ],
        }

    @staticmethod
    def _sieve(limit: int) -> list[bool]:
        """Sieve of Eratosthenes: ``result[k]`` is whether ``k`` is prime."""
        flags = [True] * (limit + 1)
        flags[0] = flags[1] = False
        for k in range(2, int(limit**0.5) + 1):
            if flags[k]:
                flags[k * k :: k] = [False] * len(flags[k * k :: k])
        return flags

    @staticmethod
    def _hardy_littlewood(n: int) -> float:
        """Unordered-partition prediction ``C2 * prod * n / ln(n)^2`` for even ``n``."""
        if n % 2 or n < 4:
            raise ValueError(f"the prediction is for even n >= 4, got {n}")
        product = 1.0
        remaining = n
        while remaining % 2 == 0:  # the singular series runs over ODD primes only
            remaining //= 2
        factor = 3
        while factor * factor <= remaining:
            if remaining % factor == 0:
                product *= (factor - 1) / (factor - 2)
                while remaining % factor == 0:
                    remaining //= factor
            factor += 2
        if remaining > 1:  # one odd prime factor above the square root remains
            product *= (remaining - 1) / (remaining - 2)
        return TWIN_PRIME_CONSTANT * product * n / math.log(n) ** 2
