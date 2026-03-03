# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Merged probe: EthicalIQRProbe (EthicalConstrained + IQRRobust).

Combines IQR-based robust outlier detection with ethical harm-weighted
scoring.  IQR fences provide robustness to heavy-tailed distributions;
the ethical layer applies asymmetric multipliers to breach magnitudes
so that safety-critical exceedances are amplified.

Unique signals preserved:
    * IQRRobust: Tukey IQR fences for robust outlier detection,
      resistant to heavy-tailed distributions.
    * EthicalConstrained: domain-aware safety bounds with asymmetric
      harm-weighted scoring.

Enhancements:
    * Asymmetric fence multipliers: upper breaches can be weighted
      differently from lower breaches (safety-critical domains often
      care more about upper exceedances).

Fusion:
    1. Compute IQR fences from training data (IQRRobust pathway).
    2. Apply asymmetric ethical multipliers to breach magnitudes
       (EthicalConstrained pathway).
    3. Per-sample score = ethically-weighted fence violation.

Equation:
    Q1, Q3 = percentile(x_train, [25, 75])
    IQR = Q3 - Q1 + epsilon
    lower_fence = Q1 - k * IQR
    upper_fence = Q3 + k * IQR
    For x < lower_fence:
        score(t) = (lower_fence - x(t)) / IQR * lower_multiplier
    For x > upper_fence:
        score(t) = (x(t) - upper_fence) / IQR * upper_multiplier
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.detectors.math_arrest.base_probe import (
    EPSILON,
    BaseEquationProbe,
    ProbeResult,
)


class EthicalIQRProbe(BaseEquationProbe):
    """Detect ethical fence violations via IQR fences + harm weighting.

    Combines Tukey IQR robustness with asymmetric ethical multipliers.
    """

    def __init__(self) -> None:
        super().__init__(min_samples=8)
        self._lower_fence_multiplier: float = 1.0
        self._upper_fence_multiplier: float = 1.5  # upper breaches often more critical
        self._k: float = 1.5  # IQR fence factor (Tukey default)
        self._q1: float = 0.0
        self._q3: float = 0.0
        self._iqr: float = 0.0
        self._lower_fence: float = 0.0
        self._upper_fence: float = 0.0
        self._fit_quality: float = 0.0

    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Compute IQR fences from training data."""
        x = self._to_1d(data)
        self._validate_data(x)

        self._q1 = float(np.percentile(x, 25))
        self._q3 = float(np.percentile(x, 75))
        self._iqr = self._q3 - self._q1 + EPSILON  # avoid zero IQR

        self._lower_fence = self._q1 - self._k * self._iqr
        self._upper_fence = self._q3 + self._k * self._iqr

        # Fit quality: fraction of training data inside fences
        inside = np.sum((x >= self._lower_fence) & (x <= self._upper_fence))
        self._fit_quality = float(inside / len(x))
        self._is_fitted = True

    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Score each sample by ethically-weighted fence violation."""
        self._validate_fitted()
        x = self._to_1d(data)

        scores = np.zeros_like(x, dtype=np.float64)

        # Lower breach: normalize by IQR, apply ethical weight
        lower_mask = x < self._lower_fence
        scores[lower_mask] = (
            (self._lower_fence - x[lower_mask]) / self._iqr
        ) * self._lower_fence_multiplier

        # Upper breach: normalize by IQR, apply ethical weight
        upper_mask = x > self._upper_fence
        scores[upper_mask] = (
            (x[upper_mask] - self._upper_fence) / self._iqr
        ) * self._upper_fence_multiplier

        raw = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        normalized = self._normalize_scores(raw)

        return ProbeResult(
            probe_name="ethical_iqr",
            deviation_scores=normalized,
            confidence=self._fit_quality,
            trajectory_fit_quality=self._fit_quality,
            anomaly_geometry="ethical_fence_violation",
        )
