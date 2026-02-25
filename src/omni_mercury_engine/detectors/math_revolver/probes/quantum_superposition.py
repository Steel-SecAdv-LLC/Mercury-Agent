# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Probe 16: Quantum superposition probe for detecting interference breaks."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.detectors.math_revolver.base_probe import (
    EPSILON,
    BaseEquationProbe,
    ProbeResult,
)


class QuantumSuperpositionProbe(BaseEquationProbe):
    """Detect interference pattern breaks via cosine fringe analysis.

    Equation:
        fringe(t) = cos(x(t))
        deviation(t) = |fringe(t) - mu_fringe| / sigma_fringe

    This differs from ZetaHarmonicProbe by using cos(x) directly
    rather than sin(2*pi*x) + cos(2*pi*x), probing a different
    frequency domain.
    """

    def __init__(self) -> None:
        super().__init__(min_samples=8)
        self._mu_fringe: float = 0.0
        self._sigma_fringe: float = 0.0
        self._fit_quality: float = 0.0

    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Learn fringe pattern statistics from training data."""
        x = self._to_1d(data)
        self._validate_data(x)
        fringe = np.cos(x)
        self._mu_fringe = float(np.mean(fringe))
        self._sigma_fringe = float(np.std(fringe)) + EPSILON

        cv = self._sigma_fringe / (abs(self._mu_fringe) + EPSILON)
        self._fit_quality = float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))
        self._is_fitted = True

    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Score each sample by its fringe pattern deviation."""
        self._validate_fitted()
        x = self._to_1d(data)
        fringe = np.cos(x)
        raw = np.abs(fringe - self._mu_fringe) / self._sigma_fringe
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        scores = self._normalize_scores(raw)
        return ProbeResult(
            probe_name="quantum_superposition",
            deviation_scores=scores,
            confidence=self._fit_quality,
            trajectory_fit_quality=self._fit_quality,
            anomaly_geometry="interference_break",
        )
