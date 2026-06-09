# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Probe 19: Boltzmann coupling probe for detecting coupling structure breaks."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.detectors.math_arrest.base_probe import (
    EPSILON,
    BaseEquationProbe,
    ProbeResult,
)


class BoltzmannCouplingProbe(BaseEquationProbe):
    """Detect coupling structure breaks via multi-lag autocorrelation energy.

    Equation:
        J[lag] = mean(x[lag:] * x[:-lag]) / var(x)
        CE(t) = sum_{lag=1}^{L} J[lag] * x(t) * x(t-lag)
        deviation(t) = |CE(t) - mu_CE| / sigma_CE
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__(min_samples=20)
        self._max_lag: int = 1
        self._j_coeffs: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self._mu_ce: float = 0.0
        self._sigma_ce: float = 0.0
        self._fit_quality: float = 0.0

    def _compute_coupling_energy(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute coupling energy for each sample."""
        n = len(x)
        ce = np.zeros(n, dtype=np.float64)
        for lag_idx in range(len(self._j_coeffs)):
            lag = lag_idx + 1
            if lag >= n:
                break
            j = self._j_coeffs[lag_idx]
            ce[lag:] += j * x[lag:] * x[:-lag]
        return ce

    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Learn coupling coefficients and energy statistics."""
        x = self._to_1d(data)
        self._validate_data(x)
        n = len(x)
        self._max_lag = min(10, n // 10)
        self._max_lag = max(self._max_lag, 1)

        var_x = float(np.var(x)) + EPSILON

        # Compute coupling coefficients J[lag]
        j_list: list[float] = []
        for lag in range(1, self._max_lag + 1):
            j = float(np.mean(x[lag:] * x[:-lag])) / var_x
            j_list.append(j)
        self._j_coeffs = np.array(j_list, dtype=np.float64)

        # Compute coupling energy
        ce = self._compute_coupling_energy(x)
        self._mu_ce = float(np.mean(ce))
        self._sigma_ce = float(np.std(ce)) + EPSILON
        self._fit_quality = float(np.clip(np.mean(np.abs(self._j_coeffs)), 0.0, 1.0))
        self._is_fitted = True

    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Score each sample by its coupling energy deviation."""
        self._validate_fitted()
        x = self._to_1d(data)
        ce = self._compute_coupling_energy(x)
        raw = np.abs(ce - self._mu_ce) / self._sigma_ce
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        scores = self._normalize_scores(raw)
        return ProbeResult(
            probe_name="boltzmann_coupling",
            deviation_scores=scores,
            confidence=self._fit_quality,
            trajectory_fit_quality=self._fit_quality,
            anomaly_geometry="coupling_break",
        )
