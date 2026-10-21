# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Zeta zero-spacing instrument: GUE statistics and close-pair (Lehmer) hunting.

Montgomery's pair-correlation conjecture (1973), confirmed numerically by
Odlyzko (1987) on millions of zeros, says the normalized spacings between
consecutive non-trivial zeros of the Riemann zeta function follow the
statistics of eigenvalues of large random GUE matrices.  The Wigner surmise
for GUE spacings,

    P(s) = (32 / pi^2) * s^2 * exp(-4 s^2 / pi),

integrates to the closed-form CDF used here:

    F(s) = erf(2 s / sqrt(pi)) - (4 s / pi) * exp(-4 s^2 / pi).

The instrument computes genuine zeros with ``mpmath.zetazero`` (zeros of the
real Riemann--Siegel ``Z(t)``, exactly on ``Re(s) = 1/2``), unfolds the gaps
with the local mean spacing ``2 pi / log(gamma / 2 pi)``, measures the
Kolmogorov--Smirnov distance to the GUE law, and flags unusually close
pairs -- Lehmer's phenomenon (Lehmer 1956): a pair of zeros nearly merging
is exactly where the Riemann Hypothesis is stressed hardest, since a double
zero off the line is what a counterexample would look like.

An anomaly here is a *clue to investigate*, never a theorem.  The
counterexample path proper is the strip-count cross-check in
:meth:`omni_mercury_engine.models.simulation.SimulationModule._explore_riemann_hypothesis`.
Fails closed without mpmath.
"""

from __future__ import annotations

import math
from typing import Any

# Optional high-precision oracle (same convention as utils.constants).
try:
    import mpmath

    MPMATH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via monkeypatched flag
    MPMATH_AVAILABLE = False


def gue_spacing_cdf(s: float) -> float:
    """Closed-form CDF of the GUE Wigner surmise, ``F(s)`` as derived above.

    Args:
        s: Normalized spacing, ``s >= 0``.

    Returns:
        ``P(spacing <= s)`` under the Wigner surmise for GUE.

    Raises:
        ValueError: Negative ``s`` -- a spacing cannot be negative.
    """
    if s < 0:
        raise ValueError("a normalized spacing cannot be negative")
    return math.erf(2.0 * s / math.sqrt(math.pi)) - (4.0 * s / math.pi) * math.exp(
        -4.0 * s * s / math.pi
    )


class ZetaSpectrumInstrument:
    """Observe the zeta zero spectrum for GUE agreement and close pairs."""

    #: Hard cap on zeros per observation (~7 s of mpmath work at 100).
    MAX_ZEROS = 100

    def observe(
        self,
        num_zeros: int = 50,
        *,
        close_pair_threshold: float = 0.5,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Measure spacing statistics of the first ``num_zeros`` zeta zeros.

        Args:
            num_zeros: Zeros to compute (2..``MAX_ZEROS``; ~1.5 s at 50).
            close_pair_threshold: Normalized spacings below this are flagged
                as close-pair anomalies (the mean normalized spacing is 1).
            top_k: How many closest pairs to report regardless of threshold.

        Returns:
            A dict with ``status``, ``oracle``, spacing statistics
            (``mean_normalized_spacing``, ``ks_distance_to_gue``),
            ``closest_pairs``, ``close_pair_anomalies`` and ``notes``.

        Raises:
            ValueError: ``num_zeros`` outside 2..``MAX_ZEROS`` or a
                non-positive threshold/top_k -- refused, never clamped
                silently.
        """
        if not 2 <= num_zeros <= self.MAX_ZEROS:
            raise ValueError(f"num_zeros must be in 2..{self.MAX_ZEROS}, got {num_zeros}")
        if close_pair_threshold <= 0 or top_k < 1:
            raise ValueError("close_pair_threshold must be > 0 and top_k >= 1")
        if not MPMATH_AVAILABLE:
            return {
                "status": "unavailable",
                "oracle": "unavailable",
                "num_zeros": 0,
                "mean_normalized_spacing": float("nan"),
                "ks_distance_to_gue": float("nan"),
                "closest_pairs": [],
                "close_pair_anomalies": [],
                "notes": [
                    "mpmath is not installed: no zeta oracle, so no zeros "
                    "were computed (refusing to fabricate them)"
                ],
            }

        ordinates = [float(mpmath.zetazero(k).imag) for k in range(1, num_zeros + 1)]
        spacings = self._normalized_spacings(ordinates)
        pairs = sorted(
            (
                {
                    "normalized_spacing": s,
                    "lower_zero_index": i + 1,
                    "upper_zero_index": i + 2,
                    "lower_ordinate": ordinates[i],
                    "upper_ordinate": ordinates[i + 1],
                }
                for i, s in enumerate(spacings)
            ),
            key=lambda p: float(p["normalized_spacing"]),
        )
        anomalies = [p for p in pairs if float(p["normalized_spacing"]) < close_pair_threshold]

        return {
            "status": "grounded",
            "oracle": "mpmath",
            "num_zeros": num_zeros,
            "mean_normalized_spacing": sum(spacings) / len(spacings),
            "ks_distance_to_gue": self._ks_distance(spacings),
            "closest_pairs": pairs[:top_k],
            "close_pair_anomalies": anomalies,
            "notes": [
                "an unusually close pair is Lehmer's phenomenon -- a clue to "
                "investigate at higher precision, never a theorem",
                "zeros from mpmath.zetazero lie on Re(s)=1/2 by construction; "
                "the off-line counterexample check is the strip-count "
                "cross-check in SimulationModule._explore_riemann_hypothesis",
            ],
        }

    @staticmethod
    def _normalized_spacings(ordinates: list[float]) -> list[float]:
        """Unfold gaps by the local mean spacing ``2 pi / log(gamma / 2 pi)``."""
        return [
            (ordinates[i + 1] - ordinates[i])
            * math.log(ordinates[i] / (2 * math.pi))
            / (2 * math.pi)
            for i in range(len(ordinates) - 1)
        ]

    @staticmethod
    def _ks_distance(spacings: list[float]) -> float:
        """Kolmogorov--Smirnov distance of the sample to the GUE spacing CDF."""
        ordered = sorted(spacings)
        n = len(ordered)
        distance = 0.0
        for i, s in enumerate(ordered):
            cdf = gue_spacing_cdf(s)
            distance = max(distance, abs((i + 1) / n - cdf), abs(cdf - i / n))
        return distance
