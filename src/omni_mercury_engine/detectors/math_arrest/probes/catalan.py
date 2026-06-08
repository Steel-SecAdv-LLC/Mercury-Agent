# Copyright (C) 2025 Steel Security Advisors LLC
"""Probe 6: Catalan-optimized AR(1) probe for detecting autocorrelation breaks."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.detectors.math_arrest.base_probe import (
    CATALAN_G,
    EPSILON,
    BaseEquationProbe,
    ProbeResult,
)


class CatalanOptimizedProbe(BaseEquationProbe):
    """Detect autocorrelation breaks using a Catalan-constant AR(1) model.

    Equation:
        x_hat(t) = mu + G * (x(t-1) - mu)

    where G = Catalan's constant (0.9159...).
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__(min_samples=8)
        self._mean: float = 0.0
        self._std: float = 0.0
        self._residual_std: float = 0.0
        self._fit_quality: float = 0.0

    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Learn mean, std, and AR(1) residual statistics."""
        x = self._to_1d(data)
        self._validate_data(x)
        self._mean = float(np.mean(x))
        self._std = float(np.std(x)) + EPSILON

        # AR(1) predictions on training data
        predicted = self._mean + CATALAN_G * (x[:-1] - self._mean)
        residuals = x[1:] - predicted
        self._residual_std = float(np.std(residuals)) + EPSILON
        self._fit_quality = self._r_squared(x[1:], predicted)
        self._is_fitted = True

    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Score each sample by its AR(1) prediction residual."""
        self._validate_fitted()
        x = self._to_1d(data)
        predicted = self._mean + CATALAN_G * (x[:-1] - self._mean)
        residuals = np.abs(x[1:] - predicted) / self._residual_std
        raw = np.concatenate([np.zeros(1, dtype=np.float64), residuals])
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        scores = self._normalize_scores(raw)
        return ProbeResult(
            probe_name="catalan_optimized",
            deviation_scores=scores,
            confidence=self._fit_quality,
            trajectory_fit_quality=self._fit_quality,
            anomaly_geometry="autocorrelation_break",
        )
