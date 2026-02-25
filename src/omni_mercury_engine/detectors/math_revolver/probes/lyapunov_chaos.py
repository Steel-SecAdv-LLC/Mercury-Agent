# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Probe 11: Lyapunov chaos probe for detecting chaos onset."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.detectors.math_revolver.base_probe import (
    EPSILON,
    BaseEquationProbe,
    ProbeResult,
)


class LyapunovChaosProbe(BaseEquationProbe):
    """Detect chaos onset via nearest-neighbor trajectory divergence.

    For each sample, finds the nearest neighbor (with an exclusion zone),
    then measures how quickly their trajectories diverge over a horizon k.
    """

    def __init__(self) -> None:
        super().__init__(min_samples=20)
        self._mu_lyap: float = 0.0
        self._sigma_lyap: float = 0.0
        self._horizon: int = 1
        self._fit_quality: float = 0.0

    @staticmethod
    def _compute_divergence(
        x: npt.NDArray[np.float64], k: int, exclusion: int = 3
    ) -> npt.NDArray[np.float64]:
        """Compute nearest-neighbor divergence rates.

        Args:
            x: 1-D signal.
            k: Prediction horizon.
            exclusion: Exclusion zone radius around each sample.

        Returns:
            Array of divergence rates.
        """
        n = len(x)
        valid_end = n - k
        divergences: list[float] = []

        for i in range(valid_end):
            # Find nearest neighbor outside exclusion zone
            best_dist = np.inf
            best_idx = -1
            for j in range(valid_end):
                if abs(i - j) <= exclusion:
                    continue
                dist = abs(x[i] - x[j])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = j

            if best_idx < 0 or best_dist < EPSILON:
                divergences.append(0.0)
                continue

            future_dist = abs(x[i + k] - x[best_idx + k])
            if future_dist < EPSILON:
                divergences.append(0.0)
            else:
                divergences.append(
                    float(np.log(future_dist / best_dist)) / k
                )

        return np.array(divergences, dtype=np.float64)

    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Compute Lyapunov divergence statistics from training data."""
        x = self._to_1d(data)
        self._validate_data(x)
        n = len(x)
        self._horizon = max(1, n // 50)

        divergences = self._compute_divergence(x, self._horizon)
        divergences = np.nan_to_num(divergences, nan=0.0, posinf=0.0, neginf=0.0)

        self._mu_lyap = float(np.mean(divergences))
        self._sigma_lyap = float(np.std(divergences)) + EPSILON

        cv = self._sigma_lyap / (abs(self._mu_lyap) + EPSILON)
        self._fit_quality = float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))
        self._is_fitted = True

    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Score each sample by its divergence deviation."""
        self._validate_fitted()
        x = self._to_1d(data)
        n = len(x)

        divergences = self._compute_divergence(x, self._horizon)
        divergences = np.nan_to_num(divergences, nan=0.0, posinf=0.0, neginf=0.0)

        raw_inner = np.abs(divergences - self._mu_lyap) / self._sigma_lyap
        pad_len = n - len(raw_inner)
        raw = np.concatenate([raw_inner, np.zeros(pad_len, dtype=np.float64)])

        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        scores = self._normalize_scores(raw)
        return ProbeResult(
            probe_name="lyapunov_chaos",
            deviation_scores=scores,
            confidence=self._fit_quality,
            trajectory_fit_quality=self._fit_quality,
            anomaly_geometry="chaos_onset",
        )
