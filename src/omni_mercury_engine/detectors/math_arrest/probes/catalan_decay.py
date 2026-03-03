# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Merged probe: CatalanDecayProbe (CatalanOptimized + ExponentialDecay).

Combines Catalan-number-weighted structural detection with exponential-decay
temporal weighting so that recent structural violations are amplified while
older ones decay.

Unique signals preserved:
    * CatalanOptimized: structural/combinatorial pattern detection via
      Catalan number sequences (ballot sequences, tree structures,
      nested patterns).
    * ExponentialDecay: temporal weighting — recent observations carry
      higher weight via exponential decay.

Fusion:
    1. Apply exponential decay weights to the signal (ExponentialDecay).
    2. Score structural deviations at Catalan-indexed lags (CatalanOptimized).
    3. Per-sample score = weighted sum of Catalan-lagged differences.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.detectors.math_arrest.base_probe import (
    EPSILON,
    BaseEquationProbe,
    ProbeResult,
)


class CatalanDecayProbe(BaseEquationProbe):
    """Detect temporal-structural breaks via decay-weighted Catalan lag diffs.

    Equations:
        C(0) = 1,  C(i) = C(i-1) * 2*(2i-1) / (i+1)       (Catalan numbers)
        weights(t) = exp(-decay_rate * (n - 1 - t))           (recent = high)
        weighted(t) = x(t) * weights(t) / sum(weights)
        score(t) = sum_lag  cw[lag] * |weighted[t] - weighted[t-lag]|
    """

    def __init__(self) -> None:
        super().__init__(min_samples=8)
        self._decay_rate: float = 0.1
        self._catalan_depth: int = 8
        self._catalan_numbers: npt.NDArray[np.float64] = self._precompute_catalan(8)
        self._residual_std: float = 0.0
        self._fit_quality: float = 0.0

    @staticmethod
    def _precompute_catalan(n: int) -> npt.NDArray[np.float64]:
        """Precompute Catalan numbers C(0) through C(n)."""
        c = np.zeros(n + 1, dtype=np.float64)
        c[0] = 1.0
        for i in range(1, n + 1):
            c[i] = c[i - 1] * 2 * (2 * i - 1) / (i + 1)
        return c

    def _per_sample_scores(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute per-sample decay-weighted Catalan structural scores."""
        n = len(x)
        t = np.arange(n, dtype=np.float64)

        # Exponential decay weights (unique from ExponentialDecay)
        weights = np.exp(-self._decay_rate * t[::-1])  # recent = high weight
        weighted = x * weights / (weights.sum() + EPSILON)

        # Catalan structural scoring (unique from CatalanOptimized)
        depth = min(self._catalan_depth, n)
        catalan_weights = self._catalan_numbers[:depth] / (
            self._catalan_numbers[:depth].sum() + EPSILON
        )

        # Per-sample scores: accumulated Catalan-lagged differences
        scores = np.zeros(n, dtype=np.float64)
        for i, cw in enumerate(catalan_weights):
            lag = i + 1
            if lag < n:
                diff = np.abs(weighted[lag:] - weighted[:-lag])
                scores[lag:] += cw * diff

        return scores

    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Learn normalization statistics from training data."""
        x = self._to_1d(data)
        self._validate_data(x)

        train_scores = self._per_sample_scores(x)
        self._residual_std = float(np.std(train_scores)) + EPSILON

        # Fit quality: ratio of structured variance to total variance
        total_var = float(np.var(x)) + EPSILON
        residual_var = float(np.var(train_scores))
        self._fit_quality = float(np.clip(1.0 - residual_var / total_var, 0.0, 1.0))
        self._is_fitted = True

    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Score each sample by its Catalan-lagged structural deviation."""
        self._validate_fitted()
        x = self._to_1d(data)

        raw = self._per_sample_scores(x) / self._residual_std
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        scores = self._normalize_scores(raw)

        return ProbeResult(
            probe_name="catalan_decay",
            deviation_scores=scores,
            confidence=self._fit_quality,
            trajectory_fit_quality=self._fit_quality,
            anomaly_geometry="temporal_structural_break",
        )
