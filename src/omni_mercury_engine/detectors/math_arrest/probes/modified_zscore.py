# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Probe 21: Modified Z-score probe for robust location anomaly detection."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.detectors.math_arrest.base_probe import (
    EPSILON,
    BaseEquationProbe,
    ProbeResult,
)


class ModifiedZScoreProbe(BaseEquationProbe):
    """Detect robust location anomalies using MAD-based modified Z-scores.

    Equation:
        median = median(x_train)
        MAD = median(|x_train - median|) + epsilon
        deviation(t) = 0.6745 * |x(t) - median| / MAD
    """

    def __init__(self) -> None:
        super().__init__(min_samples=8)
        self._median: float = 0.0
        self._mad: float = 0.0
        self._fit_quality: float = 0.0

    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Learn median and MAD from training data."""
        x = self._to_1d(data)
        self._validate_data(x)

        self._median = float(np.median(x))
        self._mad = float(np.median(np.abs(x - self._median))) + EPSILON

        # Fit quality: MAD-based methods work on all distributions
        fq = 1.0 - self._mad / (abs(self._median) + EPSILON)
        self._fit_quality = max(0.5, float(np.clip(fq, 0.0, 1.0)))
        self._is_fitted = True

    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Score each sample by its modified Z-score."""
        self._validate_fitted()
        x = self._to_1d(data)
        raw = 0.6745 * np.abs(x - self._median) / self._mad
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        scores = self._normalize_scores(raw)
        return ProbeResult(
            probe_name="modified_zscore",
            deviation_scores=scores,
            confidence=self._fit_quality,
            trajectory_fit_quality=self._fit_quality,
            anomaly_geometry="robust_location_anomaly",
        )
