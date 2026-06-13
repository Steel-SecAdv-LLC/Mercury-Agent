# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fibring Fusion Composer.

Fibring Fusion: hierarchical composition of three primitives that already
exist in Mercury but were never named as a single mode:

    1. Phi-weighted base   — golden-ratio split between neural / symbolic.
    2. Correlation-aware decorrelation — when neural and symbolic agree
       too consistently across a recent window, the redundant component's
       weight is reduced (echoing the math_arrest CorrelationAwareDecorrelator).
    3. Domain-affinity reordering — per-domain bias that favours the
       modality which is empirically stronger for that domain (medical
       favours symbolic; geomagnetic / earthquake favour neural).

The composition is stateful (running window) but pure: identical histories
and inputs yield identical weights. No randomness.

This module is taxonomy-faithful: in the NSAI literature (Garcez & Lamb 2020,
Sarker et al. 2021) "fibring" is the architectural pattern in which one
reasoning system is fibred over another rather than placed sequentially or
independently in parallel. Mercury's PHI / decorrelator / affinity stack
already implements that pattern; this module gives it its name and a single
entry point.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from omni_mercury_engine.core.centralized_constants import MATH

PHI: float = MATH.GOLDEN_RATIO

# Match the redundancy threshold used by the math_arrest decorrelator so the
# two layers behave consistently.
REDUNDANCY_THRESHOLD: float = 0.85
MIN_SAMPLES_FOR_DECORRELATION: int = 32
DEFAULT_WINDOW_SIZE: int = 128

# Per-domain bias toward (neural, symbolic). Sums to 0 within each tuple so
# the renormalisation step is a no-op when no decorrelation fires. Values
# chosen to match the affinity ranks already encoded in
# detectors/math_arrest/domain_affinity.py — domains whose top-affinity probe
# is a statistical/pattern probe lean neural; rule-rich domains lean symbolic.
DOMAIN_AFFINITY_BIAS: dict[str, tuple[float, float]] = {
    # Rule-driven / safety-critical domains: symbolic-favoured
    "medical": (-0.08, 0.08),
    "ethical": (-0.10, 0.10),
    "conflict": (-0.05, 0.05),
    # Signal / dynamics-driven domains: neural-favoured
    "earthquake": (0.06, -0.06),
    "tsunami": (0.06, -0.06),
    "marine": (0.05, -0.05),
    "geomagnetic": (0.07, -0.07),
    "pandemic": (0.04, -0.04),
    "financial": (0.05, -0.05),
    # Default: no bias
    "default": (0.0, 0.0),
    "general": (0.0, 0.0),
}


@dataclass
class FibringWeights:
    """Composed neural/symbolic weights and the diagnostics that produced them."""

    neural_weight: float
    symbolic_weight: float
    base_neural_weight: float
    base_symbolic_weight: float
    correlation: float | None
    decorrelation_applied: bool
    domain_bias_applied: tuple[float, float]
    window_size: int


def _phi_base() -> tuple[float, float]:
    """Phi-weighted neural/symbolic split (sums to 1)."""
    n = PHI / (1.0 + PHI)
    s = 1.0 / (1.0 + PHI)
    return n, s


class FibringComposer:
    """Stateful composer producing per-sample (neural, symbolic) fusion weights.

    The composer keeps a sliding window of the last N (neural_score,
    symbolic_score) pairs. On every call it:

        1. Starts from the Phi-weighted base.
        2. If the window has at least ``min_samples_for_decorrelation``
           entries and the absolute Pearson correlation between the
           neural and symbolic series exceeds ``redundancy_threshold``,
           it reduces the weight of the lower-variance (redundant)
           component by the symmetric factor ``1 / (1 + |r|)``.
        3. Adds the per-domain affinity bias.
        4. Renormalises and clips to ``[0, 1]``.
    """

    def __init__(
        self,
        domain: str | None = None,
        window_size: int = DEFAULT_WINDOW_SIZE,
        redundancy_threshold: float = REDUNDANCY_THRESHOLD,
        min_samples_for_decorrelation: int = MIN_SAMPLES_FOR_DECORRELATION,
    ) -> None:
        """Initialize the instance."""
        if window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {window_size}")
        if not 0.0 < redundancy_threshold <= 1.0:
            raise ValueError(f"redundancy_threshold must be in (0, 1], got {redundancy_threshold}")
        if min_samples_for_decorrelation < 2:
            raise ValueError(
                f"min_samples_for_decorrelation must be >= 2, "
                f"got {min_samples_for_decorrelation}"
            )

        self._domain = (domain or "default").lower()
        self._window_size = window_size
        self._redundancy_threshold = redundancy_threshold
        self._min_samples = min_samples_for_decorrelation

        self._neural_history: deque[float] = deque(maxlen=window_size)
        self._symbolic_history: deque[float] = deque(maxlen=window_size)

    @property
    def domain(self) -> str:
        """Return the affinity-bias domain key this composer was constructed for."""
        return self._domain

    @property
    def window_size(self) -> int:
        """Return the sliding-window size used by the decorrelation primitive."""
        return self._window_size

    @property
    def history_length(self) -> int:
        """Return the number of (neural, symbolic) pairs currently in the window."""
        return len(self._neural_history)

    def reset(self) -> None:
        """Clear the running window."""
        self._neural_history.clear()
        self._symbolic_history.clear()

    def observe(self, neural_score: float, symbolic_score: float) -> None:
        """Append a (neural, symbolic) pair to the running window."""
        self._neural_history.append(float(neural_score))
        self._symbolic_history.append(float(symbolic_score))

    def _domain_bias(self) -> tuple[float, float]:
        return DOMAIN_AFFINITY_BIAS.get(self._domain, DOMAIN_AFFINITY_BIAS["default"])

    def _compute_correlation(self) -> float | None:
        """Return Pearson correlation of the running window, or None if undefined."""
        if len(self._neural_history) < self._min_samples:
            return None

        n_arr = np.asarray(self._neural_history, dtype=np.float64)
        s_arr = np.asarray(self._symbolic_history, dtype=np.float64)

        # If either series is constant the correlation is undefined.
        if float(np.std(n_arr)) < 1e-12 or float(np.std(s_arr)) < 1e-12:
            return None

        # np.corrcoef returns a 2x2 matrix; we want the off-diagonal.
        r = float(np.corrcoef(n_arr, s_arr)[0, 1])
        if not np.isfinite(r):
            return None
        return r

    def compose(
        self,
        neural_score: float,
        symbolic_score: float,
        update_history: bool = True,
    ) -> FibringWeights:
        """Compose fusion weights for a single (neural, symbolic) pair.

        Args:
            neural_score: Neural component score in [0, 1].
            symbolic_score: Symbolic component score in [0, 1].
            update_history: If True, the pair is appended to the running
                window after composition (so the composition is causal —
                weights for sample t do not depend on sample t itself).

        Returns:
            FibringWeights with composed neural/symbolic weights and
            full diagnostics.
        """
        base_n, base_s = _phi_base()

        # Step 2: correlation-aware decorrelation.
        correlation = self._compute_correlation()
        decorrelated_n, decorrelated_s = base_n, base_s
        decorrelation_applied = False

        if correlation is not None and abs(correlation) >= self._redundancy_threshold:
            # The two streams are redundant; reduce the weight of the
            # lower-variance (less informative) component.
            n_arr = np.asarray(self._neural_history, dtype=np.float64)
            s_arr = np.asarray(self._symbolic_history, dtype=np.float64)
            var_n = float(np.var(n_arr))
            var_s = float(np.var(s_arr))

            shrink = 1.0 / (1.0 + abs(correlation))
            if var_n <= var_s:
                decorrelated_n = base_n * shrink
            else:
                decorrelated_s = base_s * shrink
            decorrelation_applied = True

        # Step 3: domain affinity.
        bias_n, bias_s = self._domain_bias()
        biased_n = decorrelated_n + bias_n
        biased_s = decorrelated_s + bias_s

        # Step 4: clip negatives, renormalise. Renormalisation guarantees
        # weights sum to exactly 1 even after bias.
        biased_n = max(biased_n, 0.0)
        biased_s = max(biased_s, 0.0)
        total = biased_n + biased_s
        if total <= 0.0:
            # Degenerate: fall back to equal weights.
            final_n = 0.5
            final_s = 0.5
        else:
            final_n = biased_n / total
            final_s = biased_s / total

        if update_history:
            self.observe(neural_score, symbolic_score)

        return FibringWeights(
            neural_weight=final_n,
            symbolic_weight=final_s,
            base_neural_weight=base_n,
            base_symbolic_weight=base_s,
            correlation=correlation,
            decorrelation_applied=decorrelation_applied,
            domain_bias_applied=(bias_n, bias_s),
            window_size=self._window_size,
        )

    def fuse(
        self,
        neural_score: float,
        symbolic_score: float,
        update_history: bool = True,
    ) -> tuple[float, FibringWeights]:
        """Convenience: compose weights and return the fused score."""
        weights = self.compose(neural_score, symbolic_score, update_history=update_history)
        fused = weights.neural_weight * neural_score + weights.symbolic_weight * symbolic_score
        return float(np.clip(fused, 0.0, 1.0)), weights


__all__ = [
    "DEFAULT_WINDOW_SIZE",
    "DOMAIN_AFFINITY_BIAS",
    "MIN_SAMPLES_FOR_DECORRELATION",
    "PHI",
    "REDUNDANCY_THRESHOLD",
    "FibringComposer",
    "FibringWeights",
]
