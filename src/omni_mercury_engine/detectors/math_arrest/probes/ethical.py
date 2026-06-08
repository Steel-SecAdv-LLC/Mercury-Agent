# Copyright (C) 2025 Steel Security Advisors LLC
"""Probe 5: Ethical-constrained boundary violation detector."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.detectors.math_arrest.base_probe import (
    EPSILON,
    BaseEquationProbe,
    ProbeResult,
)


class EthicalConstrainedProbe(BaseEquationProbe):
    """Detect boundary violations using percentile-based envelopes.

    Equation:
        deviation(t) = (max(0, x - B_hi - margin) +
                        max(0, B_lo - margin - x)) / range
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__(min_samples=8)
        self._b_lo: float = 0.0
        self._b_hi: float = 0.0
        self._margin: float = 0.0
        self._range: float = 0.0
        self._fit_quality: float = 0.0

    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Learn boundary envelope from training percentiles."""
        x = self._to_1d(data)
        self._validate_data(x)

        self._b_lo = float(np.percentile(x, 2.5))
        self._b_hi = float(np.percentile(x, 97.5))
        data_range = self._b_hi - self._b_lo

        if data_range < EPSILON:
            # Zero-range fallback: use MAD
            median = float(np.median(x))
            mad = float(np.median(np.abs(x - median))) + EPSILON
            self._b_lo = median - 3.0 * mad
            self._b_hi = median + 3.0 * mad
            data_range = self._b_hi - self._b_lo
            self._fit_quality = 0.1
        else:
            inside = np.sum((x >= self._b_lo) & (x <= self._b_hi))
            self._fit_quality = float(inside / len(x))

        self._margin = 0.1 * data_range
        self._range = data_range + EPSILON
        self._is_fitted = True

    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Score each sample by its boundary violation magnitude."""
        self._validate_fitted()
        x = self._to_1d(data)
        upper_violation = np.maximum(0.0, x - self._b_hi - self._margin)
        lower_violation = np.maximum(0.0, self._b_lo - self._margin - x)
        raw = (upper_violation + lower_violation) / self._range
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        scores = self._normalize_scores(raw)
        return ProbeResult(
            probe_name="ethical_constrained",
            deviation_scores=scores,
            confidence=self._fit_quality,
            trajectory_fit_quality=self._fit_quality,
            anomaly_geometry="boundary_violation",
        )
