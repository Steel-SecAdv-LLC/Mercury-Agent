# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Probe 18: Quantum annealing probe for detecting thermodynamic outliers."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.detectors.math_arrest.base_probe import (
    EPSILON,
    BaseEquationProbe,
    ProbeResult,
)


class QuantumAnnealingProbe(BaseEquationProbe):
    """Detect thermodynamic outliers via Boltzmann negative log-likelihood.

    Equation:
        T = var(x_train)   (temperature from training variance)
        deviation(t) = x(t)^2 / T
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__(min_samples=8)
        self._temperature: float = 1.0
        self._fit_quality: float = 0.0

    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Set temperature from training variance."""
        x = self._to_1d(data)
        self._validate_data(x)
        self._temperature = float(np.var(x)) + EPSILON

        # Fit quality: closeness of kurtosis to mesokurtic (3)
        std_x = float(np.std(x)) + EPSILON
        z = (x - np.mean(x)) / std_x
        kurt = float(np.mean(z**4))
        self._fit_quality = float(np.clip(1.0 - abs(kurt - 3.0) / 10.0, 0.0, 1.0))
        self._is_fitted = True

    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Score each sample by its Boltzmann NLL."""
        self._validate_fitted()
        x = self._to_1d(data)
        raw = x**2 / self._temperature
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        scores = self._normalize_scores(raw)
        return ProbeResult(
            probe_name="quantum_annealing",
            deviation_scores=scores,
            confidence=self._fit_quality,
            trajectory_fit_quality=self._fit_quality,
            anomaly_geometry="thermodynamic_outlier",
        )
