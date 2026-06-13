# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Probe 2: Damped harmonic oscillator for detecting periodicity violations."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit

from omni_mercury_engine.detectors.math_arrest.base_probe import (
    EPSILON,
    BaseEquationProbe,
    ProbeResult,
)

logger = logging.getLogger(__name__)


def _damped_oscillator(
    t: npt.NDArray[np.float64],
    a: float,
    gamma: float,
    omega: float,
    phi: float,
) -> npt.NDArray[np.float64]:
    """Full damped oscillator: A * exp(-gamma*t) * cos(omega*t + phi)."""
    result: npt.NDArray[np.float64] = a * np.exp(-gamma * t) * np.cos(omega * t + phi)
    return result


def _pure_cosine(
    t: npt.NDArray[np.float64],
    a: float,
    omega: float,
    phi: float,
) -> npt.NDArray[np.float64]:
    """Pure cosine fallback: A * cos(omega*t + phi)."""
    result: npt.NDArray[np.float64] = a * np.cos(omega * t + phi)
    return result


class HarmonicOscillatorProbe(BaseEquationProbe):
    """Detect periodicity violations using a damped harmonic oscillator fit.

    Three-tier fallback chain:
        1. Full damped oscillator: A * exp(-gamma*t) * cos(omega*t + phi)
        2. Pure cosine: A * cos(omega*t + phi)
        3. Z-score fallback with fit_quality = 0.2
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__(min_samples=16)
        self._predicted: npt.NDArray[np.float64] | None = None
        self._residual_std: float = 0.0
        self._fit_quality: float = 0.0
        self._mode: str = "none"
        self._train_mean: float = 0.0
        self._train_std: float = 0.0

    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Fit a damped oscillator with fallback chain."""
        x = self._to_1d(data)
        self._validate_data(x)
        n = len(x)
        t = np.arange(n, dtype=np.float64)
        self._train_mean = float(np.mean(x))
        self._train_std = float(np.std(x)) + EPSILON

        # Estimate dominant frequency via FFT
        fft_vals = np.fft.rfft(x - self._train_mean)
        magnitudes = np.abs(fft_vals[1:])
        if len(magnitudes) > 0 and np.max(magnitudes) > EPSILON:
            dominant_idx = int(np.argmax(magnitudes)) + 1
            omega_est = 2.0 * np.pi * dominant_idx / n
        else:
            omega_est = 2.0 * np.pi / max(n, 1)

        amp_est = float(np.std(x)) * np.sqrt(2.0)

        # Tier 1: full damped oscillator
        try:
            p0 = [amp_est, 0.01, omega_est, 0.0]
            popt, _ = curve_fit(
                _damped_oscillator,
                t,
                x,
                p0=p0,
                maxfev=5000,
            )
            predicted = _damped_oscillator(t, *popt)
            self._fit_quality = self._r_squared(x, predicted)
            self._predicted = predicted
            self._mode = "damped_oscillator"
            residuals = np.abs(x - predicted)
            self._residual_std = float(np.std(residuals)) + EPSILON
            self._is_fitted = True
            return
        except (RuntimeError, ValueError):
            pass

        # Tier 2: pure cosine
        try:
            p0 = [amp_est, omega_est, 0.0]
            popt, _ = curve_fit(
                _pure_cosine,
                t,
                x,
                p0=p0,
                maxfev=5000,
            )
            predicted = _pure_cosine(t, *popt)
            self._fit_quality = self._r_squared(x, predicted)
            self._predicted = predicted
            self._mode = "pure_cosine"
            residuals = np.abs(x - predicted)
            self._residual_std = float(np.std(residuals)) + EPSILON
            self._is_fitted = True
            return
        except (RuntimeError, ValueError):
            pass

        # Tier 3: z-score fallback
        self._mode = "zscore_fallback"
        self._fit_quality = 0.2
        self._predicted = None
        self._residual_std = self._train_std
        self._is_fitted = True

    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Score each sample by deviation from the fitted oscillator."""
        self._validate_fitted()
        x = self._to_1d(data)

        if self._mode == "zscore_fallback" or self._predicted is None:
            raw = np.abs(x - self._train_mean) / self._train_std
        else:
            n = len(x)
            t = np.arange(n, dtype=np.float64)
            if self._mode == "damped_oscillator" and self._predicted is not None:
                predicted = np.interp(
                    t,
                    np.arange(len(self._predicted), dtype=np.float64),
                    self._predicted,
                )
            else:
                predicted = np.interp(
                    t,
                    np.arange(len(self._predicted), dtype=np.float64),
                    self._predicted,
                )
            raw = np.abs(x - predicted) / self._residual_std

        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        scores = self._normalize_scores(raw)
        metadata: dict[str, Any] = {"mode": self._mode}
        return ProbeResult(
            probe_name="harmonic_oscillator",
            deviation_scores=scores,
            confidence=self._fit_quality,
            trajectory_fit_quality=self._fit_quality,
            anomaly_geometry="periodicity_violation",
            metadata=metadata,
        )
