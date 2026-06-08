# Copyright (C) 2025 Steel Security Advisors LLC
"""Probe 3: Momentum (second-difference) for detecting sudden acceleration."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.detectors.math_arrest.base_probe import (
    EPSILON,
    BaseEquationProbe,
    ProbeResult,
)


class MomentumProbe(BaseEquationProbe):
    """Detect sudden acceleration via second-order finite differences.

    Equation:
        deviation(t) = |Delta^2 x(t)| / sigma_{Delta^2 x}

    The first two positions are padded with 0.0 (causal constraint).
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__(min_samples=10)
        self._accel_std: float = 0.0
        self._fit_quality: float = 0.0

    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Compute second-difference statistics from training data."""
        x = self._to_1d(data)
        self._validate_data(x)
        accel = np.diff(x, n=2)
        self._accel_std = float(np.std(accel)) + EPSILON
        mean_accel = float(np.mean(np.abs(accel)))
        cv = self._accel_std / (mean_accel + EPSILON)
        self._fit_quality = float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))
        self._is_fitted = True

    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Score each sample by the magnitude of its second difference."""
        self._validate_fitted()
        x = self._to_1d(data)
        accel = np.diff(x, n=2)
        raw_inner = np.abs(accel) / self._accel_std
        raw = np.concatenate([np.zeros(2, dtype=np.float64), raw_inner])
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        scores = self._normalize_scores(raw)
        return ProbeResult(
            probe_name="momentum",
            deviation_scores=scores,
            confidence=self._fit_quality,
            trajectory_fit_quality=self._fit_quality,
            anomaly_geometry="sudden_acceleration",
        )
