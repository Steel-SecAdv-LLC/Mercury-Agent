# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Probe 12: Topology homology probe for detecting symmetry breaks."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.detectors.math_revolver.base_probe import (
    EPSILON,
    BaseEquationProbe,
    ProbeResult,
)


class TopologyHomologyProbe(BaseEquationProbe):
    """Detect symmetry breaks via central finite differences.

    Equation:
        cd(t) = x(t+1) - x(t-1)
        deviation(t) = |cd(t) - mu_cd| / sigma_cd

    First and last positions are padded with 0.0.
    """

    def __init__(self) -> None:
        super().__init__(min_samples=10)
        self._mu_cd: float = 0.0
        self._sigma_cd: float = 0.0
        self._fit_quality: float = 0.0

    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Compute central difference statistics from training data."""
        x = self._to_1d(data)
        self._validate_data(x)

        cd = x[2:] - x[:-2]
        self._mu_cd = float(np.mean(cd))
        self._sigma_cd = float(np.std(cd)) + EPSILON

        # Fit quality: 1 - |acf_lag1(cd)| (low autocorrelation = stable)
        if len(cd) > 1:
            cd_centered = cd - self._mu_cd
            var_cd = float(np.var(cd_centered))
            if var_cd > EPSILON:
                acf1 = float(np.mean(cd_centered[:-1] * cd_centered[1:])) / var_cd
            else:
                acf1 = 0.0
            self._fit_quality = float(np.clip(1.0 - abs(acf1), 0.0, 1.0))
        else:
            self._fit_quality = 0.5

        self._is_fitted = True

    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Score each sample by its central difference deviation."""
        self._validate_fitted()
        x = self._to_1d(data)
        n = len(x)

        if n >= 3:
            cd = x[2:] - x[:-2]
            raw_inner = np.abs(cd - self._mu_cd) / self._sigma_cd
            raw = np.concatenate([
                np.zeros(1, dtype=np.float64),
                raw_inner,
                np.zeros(1, dtype=np.float64),
            ])
        else:
            raw = np.zeros(n, dtype=np.float64)

        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        scores = self._normalize_scores(raw)
        return ProbeResult(
            probe_name="topology_homology",
            deviation_scores=scores,
            confidence=self._fit_quality,
            trajectory_fit_quality=self._fit_quality,
            anomaly_geometry="symmetry_break",
        )
