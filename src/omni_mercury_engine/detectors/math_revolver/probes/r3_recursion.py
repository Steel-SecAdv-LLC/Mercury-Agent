# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Probe 9: R3 Recursion Resonance probe for detecting nonlinear saturation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.detectors.math_revolver.base_probe import (
    EPSILON,
    BaseEquationProbe,
    ProbeResult,
)


class R3RecursionResonanceProbe(BaseEquationProbe):
    """Detect nonlinear saturation via three fused nonlinear transforms.

    Equations:
        saturation(t) = x(t)^2 / (1 + |x(t)|)
        resonance(t)  = sin(pi * x(t))
        refactor(t)   = (x(t) - mu) / (sigma + epsilon)
        r3(t) = saturation(t) + resonance(t) + refactor(t)
        deviation(t) = |r3(t) - mu_r3| / (sigma_r3 + epsilon)
    """

    def __init__(self) -> None:
        super().__init__(min_samples=8)
        self._mean: float = 0.0
        self._std: float = 0.0
        self._mu_r3: float = 0.0
        self._sigma_r3: float = 0.0
        self._fit_quality: float = 0.0

    def _compute_r3(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute the R3 transform."""
        saturation = x**2 / (1.0 + np.abs(x))
        resonance = np.sin(np.pi * x)
        refactor = (x - self._mean) / (self._std + EPSILON)
        return saturation + resonance + refactor

    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Learn R3 transform statistics from training data."""
        x = self._to_1d(data)
        self._validate_data(x)
        self._mean = float(np.mean(x))
        self._std = float(np.std(x)) + EPSILON

        r3 = self._compute_r3(x)
        self._mu_r3 = float(np.mean(r3))
        self._sigma_r3 = float(np.std(r3)) + EPSILON

        cv = self._sigma_r3 / (abs(self._mu_r3) + EPSILON)
        self._fit_quality = float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))
        self._is_fitted = True

    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Score each sample by its R3 transform deviation."""
        self._validate_fitted()
        x = self._to_1d(data)
        r3 = self._compute_r3(x)
        raw = np.abs(r3 - self._mu_r3) / (self._sigma_r3 + EPSILON)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        scores = self._normalize_scores(raw)
        return ProbeResult(
            probe_name="r3_recursion_resonance",
            deviation_scores=scores,
            confidence=self._fit_quality,
            trajectory_fit_quality=self._fit_quality,
            anomaly_geometry="nonlinear_saturation",
        )
