# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Probe 15: Wave propagation probe for detecting wave equation violations."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d

from omni_mercury_engine.detectors.math_arrest.base_probe import (
    EPSILON,
    BaseEquationProbe,
    ProbeResult,
)


class WavePropagationProbe(BaseEquationProbe):
    """Detect wave equation violations via smoothed discrete Laplacian.

    Distinguished from MomentumProbe by Gaussian pre-smoothing (sigma=3),
    making this probe sensitive to meso-scale curvature rather than
    instantaneous acceleration.

    Equation:
        smoothed = gaussian_convolve(x, sigma=3)
        laplacian(t) = smoothed[t+1] + smoothed[t-1] - 2*smoothed[t]
        deviation(t) = |laplacian(t)| / laplacian_std
    """

    def __init__(self) -> None:
        super().__init__(min_samples=10)
        self._laplacian_std: float = 0.0
        self._fit_quality: float = 0.0

    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Learn Laplacian statistics from Gaussian-smoothed training data."""
        x = self._to_1d(data)
        self._validate_data(x)

        smoothed = gaussian_filter1d(x, sigma=3.0)
        laplacian = smoothed[2:] + smoothed[:-2] - 2.0 * smoothed[1:-1]
        self._laplacian_std = float(np.std(laplacian)) + EPSILON

        # Fit quality: closeness of kurtosis to mesokurtic (3)
        if len(laplacian) > 2:
            laplacian_z = (laplacian - np.mean(laplacian)) / (np.std(laplacian) + EPSILON)
            kurt = float(np.mean(laplacian_z**4))
            self._fit_quality = float(np.clip(1.0 / (1.0 + abs(kurt - 3.0)), 0.0, 1.0))
        else:
            self._fit_quality = 0.5

        self._is_fitted = True

    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Score each sample by its smoothed Laplacian magnitude."""
        self._validate_fitted()
        x = self._to_1d(data)
        n = len(x)

        smoothed = gaussian_filter1d(x, sigma=3.0)
        if n >= 3:
            laplacian = smoothed[2:] + smoothed[:-2] - 2.0 * smoothed[1:-1]
            raw_inner = np.abs(laplacian) / self._laplacian_std
            raw = np.concatenate(
                [
                    np.zeros(1, dtype=np.float64),
                    raw_inner,
                    np.zeros(1, dtype=np.float64),
                ]
            )
        else:
            raw = np.zeros(n, dtype=np.float64)

        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        scores = self._normalize_scores(raw)
        return ProbeResult(
            probe_name="wave_propagation",
            deviation_scores=scores,
            confidence=self._fit_quality,
            trajectory_fit_quality=self._fit_quality,
            anomaly_geometry="wave_equation_violation",
        )
