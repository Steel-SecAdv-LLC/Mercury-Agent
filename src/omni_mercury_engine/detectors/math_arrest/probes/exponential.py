# Copyright (C) 2025 Steel Security Advisors LLC
"""Probe 7: Exponential decay (EWMA) probe for detecting signal degradation."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar

from omni_mercury_engine.detectors.math_arrest.base_probe import (
    EPSILON,
    BaseEquationProbe,
    ProbeResult,
)


def _ewma(x: npt.NDArray[np.float64], lam: float) -> npt.NDArray[np.float64]:
    """Compute exponentially weighted moving average.

    Args:
        x: Input signal.
        lam: Smoothing factor in (0, 2).

    Returns:
        EWMA signal of same length.
    """
    n = len(x)
    y = np.empty(n, dtype=np.float64)
    y[0] = x[0]
    for i in range(1, n):
        y[i] = (1.0 - lam) * y[i - 1] + lam * x[i]
    return y


class ExponentialDecayProbe(BaseEquationProbe):
    """Detect signal degradation using optimal-lambda EWMA residuals.

    The optimal smoothing parameter lambda is found by minimizing training MSE via
    ``scipy.optimize.minimize_scalar``.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__(min_samples=10)
        self._lambda: float = 0.1
        self._residual_std: float = 0.0
        self._fit_quality: float = 0.0

    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Find optimal EWMA lambda and compute residual statistics."""
        x = self._to_1d(data)
        self._validate_data(x)

        def mse(lam: float) -> float:
            y = _ewma(x, lam)
            return float(np.mean((x - y) ** 2))

        try:
            result = minimize_scalar(mse, bounds=(0.001, 2.0), method="bounded")
            self._lambda = float(result.x)
        except (ValueError, RuntimeError):
            self._lambda = 0.1

        ewma_train = _ewma(x, self._lambda)
        residuals = np.abs(x - ewma_train)
        self._residual_std = float(np.std(residuals)) + EPSILON
        self._fit_quality = self._r_squared(x, ewma_train)
        self._is_fitted = True

    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Score each sample by its EWMA residual magnitude."""
        self._validate_fitted()
        x = self._to_1d(data)
        ewma_signal = _ewma(x, self._lambda)
        raw = np.abs(x - ewma_signal) / self._residual_std
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        scores = self._normalize_scores(raw)
        metadata: dict[str, Any] = {"lambda": self._lambda}
        return ProbeResult(
            probe_name="exponential_decay",
            deviation_scores=scores,
            confidence=self._fit_quality,
            trajectory_fit_quality=self._fit_quality,
            anomaly_geometry="signal_degradation",
            metadata=metadata,
        )
