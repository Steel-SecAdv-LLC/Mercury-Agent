# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Probe 10: SVD Projection probe for detecting dimensional collapse."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.detectors.math_revolver.base_probe import (
    EPSILON,
    BaseEquationProbe,
    ProbeResult,
)


class SVDProjectionProbe(BaseEquationProbe):
    """Detect dimensional collapse via rank-1 SVD Hankel reconstruction.

    Builds a Hankel matrix from the time series, computes the SVD,
    reconstructs using only the first singular component, and measures
    the reconstruction residual per row.
    """

    def __init__(self) -> None:
        super().__init__(min_samples=20)
        self._d: int = 3
        self._residual_std: float = 0.0
        self._fit_quality: float = 0.0
        self._residual_mean: float = 0.0

    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Build Hankel matrix, compute SVD, and store residual statistics."""
        x = self._to_1d(data)
        self._validate_data(x)
        n = len(x)
        self._d = max(3, min(50, n // 4))

        hankel = np.lib.stride_tricks.sliding_window_view(x, self._d)
        u, s, vt = np.linalg.svd(hankel, full_matrices=False)

        # Rank-1 reconstruction
        rank1 = np.outer(u[:, 0], vt[0, :]) * s[0]
        residuals = np.sqrt(np.sum((hankel - rank1) ** 2, axis=1))

        self._residual_mean = float(np.mean(residuals))
        self._residual_std = float(np.std(residuals)) + EPSILON

        # Fit quality: fraction of variance explained by rank-1
        total_var = float(np.sum(s**2))
        if total_var > EPSILON:
            self._fit_quality = float(np.clip(s[0] ** 2 / total_var, 0.0, 1.0))
        else:
            self._fit_quality = 1.0

        self._is_fitted = True

    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Score each sample by its Hankel reconstruction residual."""
        self._validate_fitted()
        x = self._to_1d(data)
        n = len(x)

        if n >= self._d:
            hankel = np.lib.stride_tricks.sliding_window_view(x, self._d)
            u, s, vt = np.linalg.svd(hankel, full_matrices=False)
            rank1 = np.outer(u[:, 0], vt[0, :]) * s[0]
            residuals = np.sqrt(np.sum((hankel - rank1) ** 2, axis=1))
            raw_inner = np.abs(residuals - self._residual_mean) / self._residual_std
            pad_len = n - len(raw_inner)
            raw = np.concatenate(
                [np.zeros(pad_len, dtype=np.float64), raw_inner]
            )
        else:
            raw = np.zeros(n, dtype=np.float64)

        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        scores = self._normalize_scores(raw)
        return ProbeResult(
            probe_name="svd_projection",
            deviation_scores=scores,
            confidence=self._fit_quality,
            trajectory_fit_quality=self._fit_quality,
            anomaly_geometry="dimensional_collapse",
        )
