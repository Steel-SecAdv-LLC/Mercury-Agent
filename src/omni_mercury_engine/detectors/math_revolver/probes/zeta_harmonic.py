# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Probe 14: Zeta harmonic probe for detecting phase coherence anomalies."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.detectors.math_revolver.base_probe import (
    EPSILON,
    BaseEquationProbe,
    ProbeResult,
)


class ZetaHarmonicProbe(BaseEquationProbe):
    """Detect phase coherence anomalies via sin/cos phase transform.

    Equation:
        z(t) = sin(2*pi*x(t)) + cos(2*pi*x(t))
        deviation(t) = |z(t) - mu_z| / sigma_z
    """

    def __init__(self) -> None:
        super().__init__(min_samples=8)
        self._mu_z: float = 0.0
        self._sigma_z: float = 0.0
        self._fit_quality: float = 0.0

    @staticmethod
    def _zeta_transform(
        x: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute sin(2*pi*x) + cos(2*pi*x)."""
        result: npt.NDArray[np.float64] = np.sin(2.0 * np.pi * x) + np.cos(
            2.0 * np.pi * x
        )
        return result

    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Learn zeta transform statistics."""
        x = self._to_1d(data)
        self._validate_data(x)
        z = self._zeta_transform(x)
        self._mu_z = float(np.mean(z))
        self._sigma_z = float(np.std(z)) + EPSILON

        cv = self._sigma_z / (abs(self._mu_z) + EPSILON)
        self._fit_quality = float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))
        self._is_fitted = True

    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Score each sample by its zeta transform deviation."""
        self._validate_fitted()
        x = self._to_1d(data)
        z = self._zeta_transform(x)
        raw = np.abs(z - self._mu_z) / self._sigma_z
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        scores = self._normalize_scores(raw)
        return ProbeResult(
            probe_name="zeta_harmonic",
            deviation_scores=scores,
            confidence=self._fit_quality,
            trajectory_fit_quality=self._fit_quality,
            anomaly_geometry="phase_coherence_anomaly",
        )
