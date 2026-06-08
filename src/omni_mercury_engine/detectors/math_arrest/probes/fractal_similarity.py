# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Probe 13: Fractal self-similarity probe for detecting scale-invariance loss."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.detectors.math_arrest.base_probe import (
    EPSILON,
    PHI,
    BaseEquationProbe,
    ProbeResult,
)


class FractalSelfSimilarityProbe(BaseEquationProbe):
    """Detect scale-invariance loss via cross-scale correlation at phi ratio.

    Compares windowed correlations between the original signal and a version resampled at the golden
    ratio scale.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__(min_samples=20)
        self._window: int = 5
        self._mu_sim: float = 0.0
        self._sigma_sim: float = 0.0
        self._fit_quality: float = 0.0

    @staticmethod
    def _resample_at_phi(
        x: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Resample the signal to length / PHI using linear interpolation."""
        n = len(x)
        new_len = max(2, int(n / PHI))
        old_indices = np.linspace(0, n - 1, n)
        new_indices = np.linspace(0, n - 1, new_len)
        resampled: npt.NDArray[np.float64] = np.interp(new_indices, old_indices, x)
        return resampled

    def _compute_similarities(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute windowed cross-scale correlations."""
        x_scaled = self._resample_at_phi(x)
        n = len(x)
        n_scaled = len(x_scaled)
        w = self._window
        max_windows = min(n - w + 1, n_scaled - w + 1)

        if max_windows <= 0:
            return np.zeros(n, dtype=np.float64)

        similarities: list[float] = []
        for i in range(max_windows):
            win_orig = x[i : i + w]
            win_scaled = x_scaled[i : i + w]
            std_o = float(np.std(win_orig))
            std_s = float(np.std(win_scaled))

            if std_o < EPSILON and std_s < EPSILON:
                similarities.append(1.0)
            elif std_o < EPSILON or std_s < EPSILON:
                similarities.append(0.0)
            else:
                corr = np.corrcoef(win_orig, win_scaled)[0, 1]
                similarities.append(float(np.nan_to_num(corr, nan=0.0)))

        result = np.array(similarities, dtype=np.float64)
        # Pad to full length
        if len(result) < n:
            pad = np.zeros(n - len(result), dtype=np.float64)
            result = np.concatenate([result, pad])
        return result[:n]

    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Learn cross-scale correlation statistics."""
        x = self._to_1d(data)
        self._validate_data(x)
        n = len(x)
        self._window = max(5, n // 20)

        sims = self._compute_similarities(x)
        self._mu_sim = float(np.mean(sims))
        self._sigma_sim = float(np.std(sims)) + EPSILON
        self._fit_quality = float(np.clip(abs(self._mu_sim), 0.0, 1.0))
        self._is_fitted = True

    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Score each sample by its cross-scale correlation deviation."""
        self._validate_fitted()
        x = self._to_1d(data)
        sims = self._compute_similarities(x)
        raw = np.abs(sims - self._mu_sim) / self._sigma_sim
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        scores = self._normalize_scores(raw)
        return ProbeResult(
            probe_name="fractal_self_similarity",
            deviation_scores=scores,
            confidence=self._fit_quality,
            trajectory_fit_quality=self._fit_quality,
            anomaly_geometry="scale_invariance_loss",
        )
