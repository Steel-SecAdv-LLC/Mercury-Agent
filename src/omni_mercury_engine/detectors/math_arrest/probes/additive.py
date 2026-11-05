# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Probe 1: dispersion around a learned linear trend, re-anchored per window."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.detectors.math_arrest.base_probe import (
    BaseEquationProbe,
    ProbeResult,
)


class AdditiveProbe(BaseEquationProbe):
    """Dispersion around the learned trend rate, with the level re-anchored.

    Equation:
        x_hat = alpha * t + median(x - alpha * t)

    The fitted slope ``alpha`` carries the learned trend rate; the level is
    re-anchored per scored window because ``t`` is local to each array and the
    fit-time intercept does not transfer (see :meth:`deviation_score`).

    **This measures dispersion about a locally-levelled trend, not level
    shifts.** It was documented as a level-shift detector and scored
    ``|x - (intercept + alpha*t)|`` against the fit-time intercept, which made
    it *anti*-correlated with real anomalies (mean NAB AUROC 0.378; 0.032 on
    one file). Re-anchoring takes it to 0.685, but a shift that occupies part
    of the window is then partly absorbed by the anchor, so the asymmetric
    level-shift response is gone. Both alternatives were measured: anchoring to
    the window's leading quarter keeps that asymmetry but drops back to 0.403,
    and anchoring to the first sample to 0.364 -- i.e. preserving the stated
    purpose costs ~0.28 AUROC on real labels. The repaired probe is kept
    because it is a genuinely independent signal (max ``|r|`` 0.44 against its
    nearest peers, well under the 0.85 redundancy threshold), and renamed in
    its documentation and ``anomaly_geometry`` rather than left mislabelled.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__(min_samples=8)
        self._slope: float = 0.0
        self._intercept: float = 0.0
        self._residual_std: float = 0.0
        self._fit_quality: float = 0.0

    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Fit a linear trend to *data*."""
        x = self._to_1d(data)
        self._validate_data(x)
        t = np.arange(len(x), dtype=np.float64)
        coeffs = np.polyfit(t, x, 1)
        self._slope = float(coeffs[0])
        self._intercept = float(coeffs[1])
        predicted = self._intercept + self._slope * t
        residuals = np.abs(x - predicted)
        self._residual_std = float(np.std(residuals)) + 1e-10
        self._fit_quality = self._r_squared(x, predicted)
        self._is_fitted = True

    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Score each sample by its deviation from the fitted trend *rate*.

        The fitted ``slope`` is applied, but the level is re-anchored to the
        scored window instead of reusing ``self._intercept``.

        ``t`` is local to whatever array is passed, so it restarts at 0 on every
        call: the fit-time intercept belongs to the start of the fit series and
        is meaningless on a window drawn from later in it. Reusing it added a
        systematic offset of ``slope x (true position)``, which dominated the
        residual and made the probe anti-correlated with anomalies -- measured
        mean NAB AUROC 0.378, and 0.032 on
        ``realAWSCloudwatch/iio_us-east-1_i-a2eb1cd9_NetworkIn.csv``.

        Carrying the true time index was measured too and is *worse* (0.368):
        it extrapolates a linear fit far beyond its fitting range. The absolute
        level is not recoverable at score time, so only the rate and the
        residual scale are carried. Re-anchoring measures 0.685.
        """
        self._validate_fitted()
        x = self._to_1d(data)
        t = np.arange(len(x), dtype=np.float64)
        trend = self._slope * t
        predicted = trend + float(np.median(x - trend))
        raw = np.abs(x - predicted) / self._residual_std
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        scores = self._normalize_scores(raw)
        return ProbeResult(
            probe_name="additive",
            deviation_scores=scores,
            confidence=self._fit_quality,
            trajectory_fit_quality=self._fit_quality,
            anomaly_geometry="trend_dispersion",
        )
