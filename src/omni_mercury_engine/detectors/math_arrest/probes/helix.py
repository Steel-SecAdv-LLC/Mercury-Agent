# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Probe 8: Helix multiplicative probe for detecting multiplicative shocks."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.stats import shapiro

from omni_mercury_engine.detectors.math_arrest.base_probe import (
    EPSILON,
    BaseEquationProbe,
    ProbeResult,
)


class HelixMultiplicativeProbe(BaseEquationProbe):
    """Detect multiplicative shocks via log-ratio analysis.

    Equation:
        log_ratio(t) = log(|x(t) / x(t-1)| + epsilon)
        deviation(t) = |log_ratio(t) - mu_logr| / sigma_logr
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__(min_samples=10)
        self._mu_logr: float = 0.0
        self._sigma_logr: float = 0.0
        self._fit_quality: float = 0.0

    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Learn log-ratio distribution from training data."""
        x = self._to_1d(data)
        self._validate_data(x)

        ratios = np.abs(x[1:] / (x[:-1] + EPSILON)) + EPSILON
        log_ratios = np.log(ratios)
        self._mu_logr = float(np.mean(log_ratios))
        self._sigma_logr = float(np.std(log_ratios)) + EPSILON

        # Fit quality: Shapiro-Wilk p-value (normality of log-ratios)
        if len(log_ratios) >= 3:
            try:
                sample = log_ratios[:5000] if len(log_ratios) > 5000 else log_ratios
                _, p_value = shapiro(sample)
                self._fit_quality = float(np.clip(p_value, 0.0, 1.0))
            except (ValueError, RuntimeError):
                # Kurtosis fallback
                kurt = float(np.mean(((log_ratios - self._mu_logr) / self._sigma_logr) ** 4))
                self._fit_quality = float(np.clip(1.0 / (1.0 + abs(kurt - 3.0)), 0.0, 1.0))
        else:
            self._fit_quality = 0.2

        self._is_fitted = True

    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Score each sample by its log-ratio deviation."""
        self._validate_fitted()
        x = self._to_1d(data)
        ratios = np.abs(x[1:] / (x[:-1] + EPSILON)) + EPSILON
        log_ratios = np.log(ratios)
        raw_inner = np.abs(log_ratios - self._mu_logr) / self._sigma_logr
        raw = np.concatenate([np.zeros(1, dtype=np.float64), raw_inner])
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        scores = self._normalize_scores(raw)
        return ProbeResult(
            probe_name="helix_multiplicative",
            deviation_scores=scores,
            confidence=self._fit_quality,
            trajectory_fit_quality=self._fit_quality,
            anomaly_geometry="multiplicative_shock",
        )
