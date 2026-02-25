# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Probe 17: Energy minimization probe for detecting energy well escapes."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.detectors.math_arrest.base_probe import (
    EPSILON,
    BaseEquationProbe,
    ProbeResult,
)


class EnergyMinimizationProbe(BaseEquationProbe):
    """Detect energy well escapes via quadratic energy landscape.

    Equation:
        E(t) = -x(t)^2
        Delta_E(t) = E(t) - E(t-1)
        deviation(t) = |Delta_E(t)| / Delta_E_std
    """

    def __init__(self) -> None:
        super().__init__(min_samples=10)
        self._delta_e_std: float = 0.0
        self._fit_quality: float = 0.0

    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Learn energy change statistics from training data."""
        x = self._to_1d(data)
        self._validate_data(x)

        energy = -(x**2)
        delta_e = np.diff(energy)
        self._delta_e_std = float(np.std(delta_e)) + EPSILON

        var_delta_e = float(np.var(delta_e))
        var_e = float(np.var(energy)) + EPSILON
        self._fit_quality = float(np.clip(1.0 - var_delta_e / var_e, 0.0, 1.0))
        self._is_fitted = True

    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Score each sample by its energy change magnitude."""
        self._validate_fitted()
        x = self._to_1d(data)
        energy = -(x**2)
        delta_e = np.diff(energy)
        raw_inner = np.abs(delta_e) / self._delta_e_std
        raw = np.concatenate([np.zeros(1, dtype=np.float64), raw_inner])
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        scores = self._normalize_scores(raw)
        return ProbeResult(
            probe_name="energy_minimization",
            deviation_scores=scores,
            confidence=self._fit_quality,
            trajectory_fit_quality=self._fit_quality,
            anomaly_geometry="energy_well_escape",
        )
