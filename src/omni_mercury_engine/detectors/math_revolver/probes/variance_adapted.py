# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Probe 4: Rolling-variance probe for detecting volatility anomalies."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.detectors.math_revolver.base_probe import (
    EPSILON,
    BaseEquationProbe,
    ProbeResult,
)


class VarianceAdaptedProbe(BaseEquationProbe):
    """Detect volatility anomalies by comparing rolling variance to training.

    Equation:
        deviation(t) = |sigma^2_window(t) - sigma^2_train| / sigma_{sigma^2}
    """

    def __init__(self) -> None:
        super().__init__(min_samples=20)
        self._train_var: float = 0.0
        self._var_of_var: float = 0.0
        self._window: int = 10
        self._fit_quality: float = 0.0

    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Learn training variance and variance-of-rolling-variance."""
        x = self._to_1d(data)
        self._validate_data(x)
        n = len(x)
        self._window = max(10, n // 20)
        self._window = min(self._window, n)

        self._train_var = float(np.var(x))

        if n >= self._window:
            windows = np.lib.stride_tricks.sliding_window_view(x, self._window)
            rolling_var = np.var(windows, axis=1)
            self._var_of_var = float(np.std(rolling_var)) + EPSILON
        else:
            self._var_of_var = EPSILON

        fq = 1.0 - self._var_of_var / (self._train_var + EPSILON)
        self._fit_quality = float(np.clip(fq, 0.0, 1.0))
        self._is_fitted = True

    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Score each sample by its local variance deviation."""
        self._validate_fitted()
        x = self._to_1d(data)
        n = len(x)

        if n >= self._window:
            windows = np.lib.stride_tricks.sliding_window_view(x, self._window)
            rolling_var = np.var(windows, axis=1)
            raw_inner = np.abs(rolling_var - self._train_var) / self._var_of_var
            pad_len = n - len(raw_inner)
            raw = np.concatenate(
                [np.zeros(pad_len, dtype=np.float64), raw_inner]
            )
        else:
            local_var = float(np.var(x))
            raw_val = abs(local_var - self._train_var) / self._var_of_var
            raw = np.full(n, raw_val, dtype=np.float64)

        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        scores = self._normalize_scores(raw)
        return ProbeResult(
            probe_name="variance_adapted",
            deviation_scores=scores,
            confidence=self._fit_quality,
            trajectory_fit_quality=self._fit_quality,
            anomaly_geometry="volatility_anomaly",
        )
