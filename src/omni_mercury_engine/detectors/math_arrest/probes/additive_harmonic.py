# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) Steel Security Advisors LLC
"""Merged probe: AdditiveHarmonicProbe (Additive + HarmonicOscillator).

Combines additive linear-superposition scoring with damped harmonic
oscillator frequency detection.  Also fixes the overflow bug from
the original HarmonicOscillatorProbe by clamping the exponent before
calling ``np.exp``.

Unique signals preserved:
    * Additive: linear superposition — detects when combined signal
      components exceed threshold.
    * HarmonicOscillator: damped oscillation model — detects
      frequency-domain anomalies and phase disruptions.

Fix applied:
    * Overflow bug: exponent clamped via ``np.clip`` before ``np.exp``
      (max_exponent = 700.0).

Fusion:
    1. Decompose signal into top-k harmonic components via FFT (Additive).
    2. Reconstruct expected signal as additive sum of damped harmonics
       (HarmonicOscillator).
    3. Per-sample score = absolute residual between actual and
       reconstructed signal.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.detectors.math_arrest.base_probe import (
    EPSILON,
    BaseEquationProbe,
    ProbeResult,
)


class AdditiveHarmonicProbe(BaseEquationProbe):
    """Detect additive harmonic deviations via FFT decomposition + damped fit.

    Equations:
        freqs = rfftfreq(n)
        fft_mag = |rfft(x)|
        top_k = argsort(fft_mag)[-n_harmonics:]
        reconstruction(t) = sum_k  A_k * exp(-clip(gamma*t, 0, 700)) *
                                   cos(omega_k * t + phi)
        score(t) = |x(t) - reconstruction(t)| / residual_std
    """

    def __init__(self) -> None:
        super().__init__(min_samples=16)
        self._n_harmonics: int = 4
        self._gamma: float = 0.1
        self._max_exponent: float = 700.0  # FIX: clamp before exp
        self._top_k_omegas: npt.NDArray[np.float64] | None = None
        self._top_k_amps: npt.NDArray[np.float64] | None = None
        self._residual_std: float = 0.0
        self._fit_quality: float = 0.0

    def _safe_damped_harmonic(
        self,
        a: float,
        gamma: float,
        t: npt.NDArray[np.float64],
        omega: float,
        phi: float,
    ) -> npt.NDArray[np.float64]:
        """Damped harmonic with overflow-safe exponent clipping."""
        # FIX: clip exponent argument to prevent overflow
        exponent = np.clip(gamma * t, 0.0, self._max_exponent)
        result: npt.NDArray[np.float64] = a * np.exp(-exponent) * np.cos(omega * t + phi)
        return result

    def _reconstruct(self, n: int, t: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Reconstruct signal as additive sum of damped harmonics."""
        assert self._top_k_omegas is not None
        assert self._top_k_amps is not None
        reconstruction = np.zeros(n, dtype=np.float64)
        for i in range(len(self._top_k_omegas)):
            omega = float(self._top_k_omegas[i])
            a = float(self._top_k_amps[i])
            reconstruction += self._safe_damped_harmonic(a, self._gamma, t, omega, phi=0.0)
        return reconstruction

    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Fit additive harmonic decomposition to training data."""
        x = self._to_1d(data)
        self._validate_data(x)
        n = len(x)
        t = np.arange(n, dtype=np.float64)

        # FFT to find top-k harmonic components (Additive pathway)
        freqs = np.fft.rfftfreq(n)
        fft_mag = np.abs(np.fft.rfft(x))
        top_k_idx = np.argsort(fft_mag)[-self._n_harmonics :]

        self._top_k_omegas = (2.0 * np.pi * freqs[top_k_idx]).astype(np.float64)
        self._top_k_amps = fft_mag[top_k_idx] / n

        # Reconstruct expected signal (HarmonicOscillator pathway)
        reconstruction = self._reconstruct(n, t)
        residuals = np.abs(x - reconstruction)
        self._residual_std = float(np.std(residuals)) + EPSILON
        self._fit_quality = self._r_squared(x, reconstruction)
        self._is_fitted = True

    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Score each sample by residual from additive harmonic model."""
        self._validate_fitted()
        x = self._to_1d(data)
        n = len(x)
        t = np.arange(n, dtype=np.float64)

        reconstruction = self._reconstruct(n, t)
        raw = np.abs(x - reconstruction) / self._residual_std
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        scores = self._normalize_scores(raw)

        return ProbeResult(
            probe_name="additive_harmonic",
            deviation_scores=scores,
            confidence=self._fit_quality,
            trajectory_fit_quality=self._fit_quality,
            anomaly_geometry="additive_harmonic_deviation",
        )
