# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Particle-filter state-space residual detector for streaming series.

A bootstrap particle filter tracks the latent state of a state-space model with
a cloud of weighted samples ("particles"), giving a fully non-parametric
one-step-ahead predictive distribution for the next observation. Anomalies are
observations that fall in the tail of that predictive: the normalised innovation
``|y_t - E[y_t | past]| / std[y_t | past]`` is large when the series deviates
from the tracked dynamics (a level shift, spike, or variance change the filter
has not yet absorbed).

This module uses a local-level (random-walk) state model -- ``x_t = x_{t-1} +
process_noise``, ``y_t = x_t + observation_noise`` -- which is the standard,
assumption-light choice for univariate monitoring; the particle representation
means the detector degrades gracefully to non-Gaussian, heavy-tailed streams
where a Kalman filter's Gaussian predictive would be miscalibrated.

The detector conforms to the
:class:`~omni_mercury_engine.core.base.BaseDetector` contract. ``fit`` estimates
the process/observation noise and a robust innovation scale from training data;
``detect`` streams the filter and squashes normalised innovations into
``[0, 1]``. The filter is seeded deterministically (reproducible builds). It
depends only on NumPy and is registered as an opt-in BASE detector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.detectors._calibration import (
    bound_finite_config,
    finite_features,
    finite_scores,
    squash_scale,
)

if TYPE_CHECKING:
    import torch

__all__ = ["ParticleFilterDetector"]

# ``1 - exp(-r / scale) = 0.5`` at ``r = scale * ln 2``; anchoring ``scale`` to a
# high training quantile places the 0.5 anomaly boundary at that quantile for a
# controlled ``1 - calibration_quantile`` false-positive rate.
_LN2 = float(np.log(2.0))


class ParticleFilterDetector(BaseDetector):
    """Bootstrap particle-filter detector scoring predictive innovations.

    Tracks a local-level state-space model with ``n_particles`` weighted
    samples and scores each observation by its normalised one-step-ahead
    innovation. ``fit`` estimates the process noise (from training first
    differences), the observation noise, and a robust innovation scale used to
    squash scores into ``[0, 1]``. Resampling uses systematic resampling with a
    fixed seed so runs are reproducible.
    """

    def __init__(
        self,
        n_particles: int = 200,
        seed: int = 0,
        calibration_quantile: float = 0.98,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the particle-filter detector.

        Args:
            n_particles: Number of particles tracking the latent state. More
                particles give a smoother predictive at higher cost. Must be
                >= 8.
            seed: Seed for the internal RNG (process-noise sampling and
                resampling), pinned for deterministic, reproducible output.
            calibration_quantile: Training-innovation quantile placed at the 0.5
                anomaly boundary; ``1 - calibration_quantile`` is the resulting
                normal-regime false-positive rate. Must be in ``(0, 1)``.
            config: Optional ``BaseDetector`` config.

        Raises:
            ValueError: If ``n_particles < 8`` or ``calibration_quantile`` is
                out of ``(0, 1)``.
        """
        super().__init__(config)
        if n_particles < 8:
            raise ValueError(f"n_particles must be >= 8, got {n_particles}")
        if not 0.0 < calibration_quantile < 1.0:
            raise ValueError(f"calibration_quantile must be in (0, 1), got {calibration_quantile}")
        self.n_particles = int(n_particles)
        self.seed = int(seed)
        self.calibration_quantile = float(calibration_quantile)
        self._process_std: float = 1.0
        self._obs_std: float = 1.0
        self._scale: float = 1.0

    def is_fitted(self) -> bool:
        """Return ``True`` once :meth:`fit` has estimated the noise model."""
        return self._is_fitted

    def _squash_scale(self, raw: np.ndarray[Any, Any]) -> float:
        """Squash scale anchoring the ``calibration_quantile`` at score 0.5."""
        return squash_scale(raw, self.calibration_quantile)

    def _to_1d_f64(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Coerce numpy/torch input to a finite 1-D float64 series."""
        detach = getattr(data, "detach", None)
        if callable(detach):
            data = detach().cpu().numpy()
        return bound_finite_config(self, np.asarray(data, dtype=np.float64)).ravel()

    @staticmethod
    def _systematic_resample(
        weights: np.ndarray[Any, Any], rng: np.random.Generator
    ) -> np.ndarray[Any, Any]:
        """Systematic resampling: low-variance, ``O(n)`` index draw."""
        n = weights.size
        positions = (rng.random() + np.arange(n)) / n
        cumulative = np.cumsum(weights)
        cumulative[-1] = 1.0  # guard against floating-point round-off
        return np.searchsorted(cumulative, positions).astype(np.intp)

    def _innovations(self, series: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Normalised one-step-ahead innovations for each observation."""
        n = series.size
        out = np.zeros(n, dtype=np.float64)
        if n == 0:
            return out
        rng = np.random.default_rng(self.seed)
        # Initialise particles around the first observation.
        particles = series[0] + rng.normal(0.0, self._obs_std + 1e-8, self.n_particles)
        q = max(self._process_std, 1e-8)
        r = max(self._obs_std, 1e-8)
        inv_two_r2 = 1.0 / (2.0 * r * r)
        for t in range(n):
            # Predict: propagate the latent random walk.
            particles = particles + rng.normal(0.0, q, self.n_particles)
            # Predictive moments of y_t (state + observation noise).
            pred_mean = float(np.mean(particles))
            pred_var = float(np.var(particles)) + r * r
            pred_std = float(np.sqrt(max(pred_var, 1e-12)))
            out[t] = abs(series[t] - pred_mean) / pred_std
            # Update: weight particles by the observation likelihood.
            resid = series[t] - particles
            log_w = -resid * resid * inv_two_r2
            log_w -= log_w.max()
            weights = np.exp(log_w)
            total = float(np.sum(weights))
            if total <= 0.0 or not np.isfinite(total):
                weights = np.full(self.n_particles, 1.0 / self.n_particles)
            else:
                weights /= total
            particles = particles[self._systematic_resample(weights, rng)]
        return out

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> ParticleFilterDetector:
        """Estimate the noise model and robust innovation scale from data.

        The observation noise is a small fraction of the series spread; the
        process noise is taken from the robust scale of first differences (how
        fast the level legitimately moves). The innovation scale used to squash
        scores is the median-absolute-deviation of training innovations.

        Args:
            data: Training series (flattened to 1-D), predominantly normal.

        Returns:
            ``self``.
        """
        series = self._to_1d_f64(data)
        if series.size >= 2:
            diffs = np.diff(series)
            med = float(np.median(diffs))
            mad = float(np.median(np.abs(diffs - med)))
            self._process_std = max(1.4826 * mad, 1e-6)
            spread = float(np.std(series))
            self._obs_std = max(0.1 * spread, 1e-6)
        else:
            self._process_std = 1.0
            self._obs_std = 1.0

        raw = self._innovations(series)
        self._scale = self._squash_scale(raw)
        self._is_fitted = True
        return self

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Per-sample fusion feature: the normalised innovation.

        Args:
            data: Input series (flattened to 1-D).

        Returns:
            ``(n_samples, 1)`` float32 innovations.
        """
        raw = self._innovations(self._to_1d_f64(data))
        return finite_features(raw, detector=self.name).reshape(-1, 1)

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Per-sample anomaly scores in ``[0, 1]`` from predictive innovations.

        Innovations are squashed monotonically via ``1 - exp(-r / scale)`` using
        the training scale from :meth:`fit` (or the input's own robust scale when
        unfitted), so observations the filter tracks well score near 0 and
        deviations from the dynamics approach 1.
        """
        raw = self._innovations(self._to_1d_f64(data))
        scale = self._scale if self._is_fitted else self._squash_scale(raw)
        scores = finite_scores(1.0 - np.exp(-raw / scale), detector=self.name).astype(np.float32)
        return {
            "anomaly_score": float(scores.max()) if scores.size else 0.0,
            "scores": scores,
            "is_anomaly": scores > self.threshold,
            "confidence": scores,
        }
