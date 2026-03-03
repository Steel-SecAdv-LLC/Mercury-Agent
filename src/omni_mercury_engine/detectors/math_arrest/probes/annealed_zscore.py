# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Merged probe: AnnealedZScoreProbe (ModifiedZScore + QuantumAnnealing).

Combines MAD-based robust z-scoring with simulated-annealing-based
adaptive threshold optimization.  Instead of a fixed z-score cutoff,
the threshold is found by minimizing an energy function that trades off
false-positive and false-negative proxy costs.

Unique signals preserved:
    * ModifiedZScore: MAD-based robust z-score, resistant to outlier
      contamination of the mean.
    * QuantumAnnealing: simulated annealing global optimization finds
      the decision threshold dynamically rather than using a fixed cutoff.

Enhancements:
    * Adaptive threshold: threshold is optimized per training dataset
      via simulated annealing, gaining domain-adaptive behaviour.

Fusion:
    1. Compute modified z-scores using median and MAD (ModifiedZScore).
    2. Run simulated annealing to find optimal threshold (QuantumAnnealing).
    3. Per-sample score = max(0, z_score - threshold), normalized.

Equation:
    median  = median(x_train)
    MAD     = median(|x_train - median|)
    z(t)    = |x(t) - median| / (consistency_factor * MAD + epsilon)
    threshold = anneal(z_train)      (simulated annealing search)
    score(t)  = max(0, z(t) - threshold)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.detectors.math_arrest.base_probe import (
    EPSILON,
    BaseEquationProbe,
    ProbeResult,
)


class AnnealedZScoreProbe(BaseEquationProbe):
    """Detect annealed location anomalies via MAD z-scores + SA threshold.

    Combines robust modified z-scores with simulated annealing threshold
    optimization for adaptive anomaly detection.
    """

    def __init__(self) -> None:
        super().__init__(min_samples=8)
        self._consistency_factor: float = 1.4826  # MAD -> std scaling
        self._n_anneal_steps: int = 100
        self._initial_temp: float = 1.0
        self._cooling_rate: float = 0.95
        self._median: float = 0.0
        self._mad_scaled: float = 0.0
        self._threshold: float = 0.0
        self._fit_quality: float = 0.0

    def _modified_z_scores(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute modified z-scores using pre-fitted median and MAD."""
        return np.abs(x - self._median) / self._mad_scaled

    @staticmethod
    def _energy(scores: npt.NDArray[np.float64], threshold: float) -> float:
        """Energy = cost of false positives + false negatives proxy."""
        above = scores[scores >= threshold]
        fp_cost = float(len(above)) / (len(scores) + EPSILON)
        fn_cost = 1.0 - (float(above.mean()) if len(above) > 0 else 0.0) / (
            float(scores.max()) + EPSILON
        )
        return fp_cost + fn_cost

    def _anneal_threshold(self, scores: npt.NDArray[np.float64]) -> float:
        """Simulated annealing to find optimal anomaly threshold."""
        scores_std = float(np.std(scores))
        current_thresh = float(np.median(scores)) + scores_std
        best_thresh = current_thresh
        best_energy = self._energy(scores, current_thresh)
        temp = self._initial_temp

        rng = np.random.default_rng(seed=42)  # deterministic
        for _ in range(self._n_anneal_steps):
            # Propose neighbor threshold
            candidate = current_thresh + rng.normal(0, temp * scores_std)
            candidate = float(np.clip(candidate, float(scores.min()), float(scores.max())))
            energy = self._energy(scores, candidate)

            delta = energy - best_energy
            # Accept if better, or probabilistically if worse (annealing)
            if delta < 0 or rng.random() < np.exp(-delta / (temp + EPSILON)):
                current_thresh = candidate
                if energy < best_energy:
                    best_energy = energy
                    best_thresh = candidate

            temp *= self._cooling_rate

        return best_thresh

    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Learn median, MAD, and annealed threshold from training data."""
        x = self._to_1d(data)
        self._validate_data(x)

        self._median = float(np.median(x))
        mad = float(np.median(np.abs(x - self._median)))
        self._mad_scaled = self._consistency_factor * mad + EPSILON

        # Compute training z-scores (ModifiedZScore pathway)
        train_scores = self._modified_z_scores(x)

        # Find optimal threshold (QuantumAnnealing pathway)
        self._threshold = self._anneal_threshold(train_scores)

        # Fit quality: MAD-based methods work on all distributions
        fq = 1.0 - mad / (abs(self._median) + EPSILON)
        self._fit_quality = max(0.5, float(np.clip(fq, 0.0, 1.0)))
        self._is_fitted = True

    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Score each sample by exceedance above annealed threshold."""
        self._validate_fitted()
        x = self._to_1d(data)

        z_scores = self._modified_z_scores(x)
        # Exceedance above annealed threshold
        raw = np.maximum(0.0, z_scores - self._threshold)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        scores = self._normalize_scores(raw)

        metadata: dict[str, Any] = {"annealed_threshold": self._threshold}
        return ProbeResult(
            probe_name="annealed_zscore",
            deviation_scores=scores,
            confidence=self._fit_quality,
            trajectory_fit_quality=self._fit_quality,
            anomaly_geometry="annealed_location_anomaly",
            metadata=metadata,
        )
