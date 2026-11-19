# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Probe 11: Lyapunov chaos probe for detecting chaos onset."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.detectors.math_arrest.base_probe import (
    EPSILON,
    BaseEquationProbe,
    ProbeResult,
)


class LyapunovChaosProbe(BaseEquationProbe):
    """Detect chaos onset via nearest-neighbor trajectory divergence.

    For each sample, finds the nearest neighbor (with an exclusion zone), then measures how quickly
    their trajectories diverge over a horizon k.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__(min_samples=20)
        self._mu_lyap: float = 0.0
        self._sigma_lyap: float = 0.0
        self._horizon: int = 1
        self._fit_quality: float = 0.0

    #: Cap on the scratch distance block, in elements: a chunk is sized so one
    #: block stays within ~32 MB of float64 whatever the series length.
    _CHUNK_ELEMENTS: int = 4_000_000

    @staticmethod
    def _compute_divergence(
        x: npt.NDArray[np.float64], k: int, exclusion: int = 3
    ) -> npt.NDArray[np.float64]:
        """Compute nearest-neighbor divergence rates.

        The mathematics is the released per-sample scan, evaluated as chunked
        NumPy: for each sample the nearest neighbour outside the exclusion
        zone (first index on ties, exactly as the former Python loop's strict
        ``<`` chose it), then ``log(future_dist / best_dist) / k``, with zero
        where no neighbour exists or either distance is under ``EPSILON``.
        The former loop was the forward path's hot spot: 2.59 s of a 2.69 s
        ``detect`` at 4 096 samples, against 0.02 s for this evaluation
        (``tests/detectors/test_math_arrest_vectorized.py`` holds the
        equivalence to the scalar reference and the latency floor).

        Args:
            x: 1-D signal.
            k: Prediction horizon.
            exclusion: Exclusion zone radius around each sample.

        Returns:
            Array of divergence rates.
        """
        n = len(x)
        valid_end = n - k
        if valid_end <= 0:
            return np.array([], dtype=np.float64)

        values = x[:valid_end]
        divergences = np.zeros(valid_end, dtype=np.float64)
        chunk = max(1, LyapunovChaosProbe._CHUNK_ELEMENTS // valid_end)
        offsets = np.arange(valid_end)

        for start in range(0, valid_end, chunk):
            stop = min(start + chunk, valid_end)
            rows = np.arange(start, stop)
            # |x[i] - x[j]| with the exclusion zone and NaN distances masked to
            # +inf: the loop's ``dist < best_dist`` never selected either (a
            # NaN comparison is False), and argmin keeps its first-index ties.
            dist = np.abs(values[rows, None] - values[None, :])
            dist[np.isnan(dist)] = np.inf
            dist[np.abs(rows[:, None] - offsets[None, :]) <= exclusion] = np.inf

            best_idx = np.argmin(dist, axis=1)
            best_dist = dist[np.arange(len(rows)), best_idx]

            future_dist = np.abs(x[rows + k] - x[best_idx + k])
            with np.errstate(divide="ignore", invalid="ignore"):
                rates = np.log(future_dist / best_dist) / k
            usable = np.isfinite(best_dist) & (best_dist >= EPSILON) & (future_dist >= EPSILON)
            divergences[rows] = np.where(usable, rates, 0.0)

        return divergences

    def fit_trajectory(self, data: npt.NDArray[np.float64]) -> None:
        """Compute Lyapunov divergence statistics from training data."""
        x = self._to_1d(data)
        self._validate_data(x)
        n = len(x)
        self._horizon = max(1, n // 50)

        divergences = self._compute_divergence(x, self._horizon)
        divergences = np.nan_to_num(divergences, nan=0.0, posinf=0.0, neginf=0.0)

        self._mu_lyap = float(np.mean(divergences))
        self._sigma_lyap = float(np.std(divergences)) + EPSILON

        cv = self._sigma_lyap / (abs(self._mu_lyap) + EPSILON)
        self._fit_quality = float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))
        self._is_fitted = True

    def deviation_score(self, data: npt.NDArray[np.float64]) -> ProbeResult:
        """Score each sample by its divergence deviation."""
        self._validate_fitted()
        x = self._to_1d(data)
        n = len(x)

        divergences = self._compute_divergence(x, self._horizon)
        divergences = np.nan_to_num(divergences, nan=0.0, posinf=0.0, neginf=0.0)

        raw_inner = np.abs(divergences - self._mu_lyap) / self._sigma_lyap
        pad_len = n - len(raw_inner)
        raw = np.concatenate([raw_inner, np.zeros(pad_len, dtype=np.float64)])

        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        scores = self._normalize_scores(raw)
        return ProbeResult(
            probe_name="lyapunov_chaos",
            deviation_scores=scores,
            confidence=self._fit_quality,
            trajectory_fit_quality=self._fit_quality,
            anomaly_geometry="chaos_onset",
        )
