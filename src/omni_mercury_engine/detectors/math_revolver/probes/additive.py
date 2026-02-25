# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Probe 1: Additive linear trend model for detecting level shifts."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.detectors.math_revolver.base_probe import (
    BaseEquationProbe,
    ProbeResult,
)


class AdditiveProbe(BaseEquationProbe):
    """Detect level shifts and trend breaks via a linear fit.

    Equation:
        x_hat = mu + alpha * t

    Residuals that deviate significantly from the fitted trend indicate
    level-shift anomalies.
    """

    def __init__(self) -> None:
        super().__init__(min_samples=8)
        self._slope: float = 0.0
        self._intercept: float = 0.0
        self._residual_std: float = 0.0
        self._fit_quality: float = 0.0

    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Fit a linear trend to *data*."""
        x = self._to_1d(data)
        self._validate_data(x)
        t = np.arange(len(x), dtype=np.float64)
        coeffs = np.polyfit(t, x, 1)
        self._slope = float(coeffs[0])
        self._intercept = float(coeffs[1])
        predicted = self._intercept + self._slope * t
        residuals = np.abs(x - predicted)
        self._residual_std = float(np.std(residuals)) + 1e-10
        self._fit_quality = self._r_squared(x, predicted)
        self._is_fitted = True

    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Score each sample by its deviation from the fitted linear trend."""
        self._validate_fitted()
        x = self._to_1d(data)
        t = np.arange(len(x), dtype=np.float64)
        predicted = self._intercept + self._slope * t
        raw = np.abs(x - predicted) / self._residual_std
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        scores = self._normalize_scores(raw)
        return ProbeResult(
            probe_name="additive",
            deviation_scores=scores,
            confidence=self._fit_quality,
            trajectory_fit_quality=self._fit_quality,
            anomaly_geometry="level_shift",
        )
