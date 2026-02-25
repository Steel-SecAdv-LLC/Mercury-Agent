# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Probe 20: IQR-robust probe for detecting distribution tail anomalies."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.detectors.math_revolver.base_probe import (
    EPSILON,
    BaseEquationProbe,
    ProbeResult,
)


class IQRRobustProbe(BaseEquationProbe):
    """Detect distribution tail anomalies using Tukey IQR fences.

    Equation:
        Q1, Q3 = percentile(x, 25), percentile(x, 75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        deviation(t) = (max(0, x - upper) + max(0, lower - x)) / (IQR + eps)
    """

    def __init__(self) -> None:
        super().__init__(min_samples=8)
        self._q1: float = 0.0
        self._q3: float = 0.0
        self._iqr: float = 0.0
        self._lower: float = 0.0
        self._upper: float = 0.0
        self._divisor: float = 1.0
        self._fit_quality: float = 0.0

    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Compute IQR fences from training data."""
        x = self._to_1d(data)
        self._validate_data(x)

        self._q1 = float(np.percentile(x, 25))
        self._q3 = float(np.percentile(x, 75))
        self._iqr = self._q3 - self._q1

        if self._iqr < EPSILON:
            # Zero-IQR guard: fall back to std
            self._divisor = float(np.std(x)) + EPSILON
            self._lower = self._q1 - 1.5 * self._divisor
            self._upper = self._q3 + 1.5 * self._divisor
            self._fit_quality = 0.1
        else:
            self._divisor = self._iqr + EPSILON
            self._lower = self._q1 - 1.5 * self._iqr
            self._upper = self._q3 + 1.5 * self._iqr
            inside = np.sum((x >= self._lower) & (x <= self._upper))
            self._fit_quality = float(inside / len(x))

        self._is_fitted = True

    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Score each sample by its fence violation magnitude."""
        self._validate_fitted()
        x = self._to_1d(data)
        upper_violation = np.maximum(0.0, x - self._upper)
        lower_violation = np.maximum(0.0, self._lower - x)
        raw = (upper_violation + lower_violation) / self._divisor
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        scores = self._normalize_scores(raw)
        return ProbeResult(
            probe_name="iqr_robust",
            deviation_scores=scores,
            confidence=self._fit_quality,
            trajectory_fit_quality=self._fit_quality,
            anomaly_geometry="distribution_tail_anomaly",
        )
