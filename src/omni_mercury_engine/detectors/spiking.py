# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Spiking-neural-network (leaky integrate-and-fire) novelty detector.

Spiking neural networks encode information in the *timing and rate* of discrete
spikes emitted by neurons whose membrane potential integrates input and leaks
over time (Gerstner & Kistler, *Spiking Neuron Models*, 2002). A population of
leaky integrate-and-fire (LIF) neurons driven by a signal produces a
spike-rate signature that is characteristic of the signal's amplitude/dynamics;
when the input leaves the regime seen in training, the population spike rate
departs from its learned baseline.

This detector drives a fixed, seeded LIF population with the (normalised) series
and scores each observation by how far the instantaneous population spike rate
deviates (in robust z-units) from the training baseline, squashed into
``[0, 1]``. Pure NumPy (always importable); registered as an opt-in BASE
detector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.detectors._calibration import (
    bound_finite,
    finite_features,
    finite_scores,
    squash_scale,
)

if TYPE_CHECKING:
    import torch

__all__ = ["SpikingNetworkDetector"]

_LN2 = float(np.log(2.0))


class SpikingNetworkDetector(BaseDetector):
    """Leaky integrate-and-fire population spike-rate novelty detector.

    A population of LIF neurons with heterogeneous, seeded input gains and
    thresholds integrates the normalised input. :meth:`fit` records the baseline
    population spike-rate distribution; :meth:`detect` scores each point by the
    robust deviation of the local spike rate from that baseline, squashed into
    ``[0, 1]``.
    """

    def __init__(
        self,
        n_neurons: int = 64,
        leak: float = 0.9,
        threshold_mean: float = 1.0,
        refractory: int = 2,
        rate_window: int = 8,
        calibration_quantile: float = 0.98,
        seed: int = 0,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the spiking-network detector.

        Args:
            n_neurons: Number of LIF neurons in the population. Must be >= 4.
            leak: Membrane leak (decay) factor per step, in ``(0, 1)``.
            threshold_mean: Mean firing threshold; per-neuron thresholds are
                jittered around it. Must be > 0.
            refractory: Refractory period (steps a neuron is silent after
                firing). Must be >= 0.
            rate_window: Trailing window over which the population spike rate is
                measured. Must be >= 1.
            calibration_quantile: Training-deviation quantile at the 0.5 boundary;
                ``1 - calibration_quantile`` is the normal-regime FPR. In ``(0, 1)``.
            seed: RNG seed for the fixed neuron parameters (determinism).
            config: Optional ``BaseDetector`` config (``threshold`` ...).

        Raises:
            ValueError: If any parameter is out of its valid range.
        """
        super().__init__(config)
        if n_neurons < 4:
            raise ValueError(f"n_neurons must be >= 4, got {n_neurons}")
        if not 0.0 < leak < 1.0:
            raise ValueError(f"leak must be in (0, 1), got {leak}")
        if threshold_mean <= 0.0:
            raise ValueError("threshold_mean must be > 0")
        if refractory < 0:
            raise ValueError("refractory must be >= 0")
        if rate_window < 1:
            raise ValueError("rate_window must be >= 1")
        if not 0.0 < calibration_quantile < 1.0:
            raise ValueError(f"calibration_quantile must be in (0, 1), got {calibration_quantile}")
        self.n_neurons = int(n_neurons)
        self.leak = float(leak)
        self.threshold_mean = float(threshold_mean)
        self.refractory = int(refractory)
        self.rate_window = int(rate_window)
        self.calibration_quantile = float(calibration_quantile)
        self.seed = int(seed)
        self._gains: np.ndarray[Any, Any] | None = None
        self._thresholds: np.ndarray[Any, Any] | None = None
        self._mean: float = 0.0
        self._std: float = 1.0
        self._base_rate: float = 0.0
        self._base_scale: float = 1.0
        self._scale: float = 1.0

    def is_fitted(self) -> bool:
        """Return ``True`` once :meth:`fit` has learned the baseline/scale."""
        return self._is_fitted

    def _to_1d_f64(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Coerce numpy/torch input to a finite 1-D float64 series."""
        detach = getattr(data, "detach", None)
        if callable(detach):
            data = detach().cpu().numpy()
        return bound_finite(np.asarray(data, dtype=np.float64), detector=self.name).ravel()

    def _squash_scale(self, raw: np.ndarray[Any, Any]) -> float:
        """Squash scale anchoring the ``calibration_quantile`` at score 0.5."""
        return squash_scale(raw, self.calibration_quantile)

    def _build_population(self) -> None:
        """Instantiate the fixed, seeded LIF neuron parameters.

        Thresholds span a wide range so that, at the training amplitude, the
        high-threshold neurons stay near-silent — leaving spike-rate headroom
        that off-regime (higher-amplitude / bursty) inputs then recruit, giving
        a graded, sensitive population response instead of a saturated one.
        """
        rng = np.random.default_rng(self.seed)
        self._gains = rng.uniform(0.5, 1.5, self.n_neurons)
        self._thresholds = self.threshold_mean * rng.uniform(0.8, 3.5, self.n_neurons)

    def _spike_rate(self, series: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Windowed population spike rate driven by the rectified input."""
        if self._gains is None or self._thresholds is None:
            # Build the fixed population lazily so ``detect()`` before ``fit()``
            # (whose body already branches on ``_is_fitted``) works instead of
            # tripping an assertion — matching the graceful unfitted-detect
            # behaviour of the sibling detectors.
            self._build_population()
        assert self._gains is not None and self._thresholds is not None  # set by _build_population
        n = series.size
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        u = np.abs((series - self._mean) / self._std)  # drive by deviation magnitude
        v = np.zeros(self.n_neurons, dtype=np.float64)
        refr = np.zeros(self.n_neurons, dtype=np.intp)
        spike_count = np.zeros(n, dtype=np.float64)
        for t in range(n):
            active = refr <= 0
            v = self.leak * v
            v[active] += self._gains[active] * u[t]
            fired = active & (v >= self._thresholds)
            spike_count[t] = float(np.count_nonzero(fired))
            v[fired] = 0.0
            refr[fired] = self.refractory
            refr[~active] -= 1
        # Windowed rate (fraction of population firing), causal moving average.
        rate = spike_count / self.n_neurons
        kernel = np.ones(self.rate_window) / self.rate_window
        padded = np.concatenate([np.full(self.rate_window - 1, rate[0]), rate])
        return np.convolve(padded, kernel, mode="valid")

    def _deviations(self, series: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Robust deviation of the population spike rate from baseline."""
        rate = self._spike_rate(series)
        return np.abs(rate - self._base_rate) / max(self._base_scale, 1e-9)

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> SpikingNetworkDetector:
        """Learn input normalisation, the LIF population, and baseline rate.

        Args:
            data: Training series (flattened to 1-D), predominantly normal.

        Returns:
            ``self``.
        """
        series = self._to_1d_f64(data)
        self._mean = float(np.mean(series)) if series.size else 0.0
        self._std = float(np.std(series)) + 1e-9
        self._build_population()
        rate = self._spike_rate(series)
        self._base_rate = float(np.median(rate)) if rate.size else 0.0
        mad = float(np.median(np.abs(rate - self._base_rate))) if rate.size else 0.0
        self._base_scale = max(1.4826 * mad, 1e-6)
        raw = self._deviations(series)
        self._scale = self._squash_scale(raw)
        self._is_fitted = True
        return self

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Per-sample fusion feature: the spike-rate deviation.

        Args:
            data: Input series (flattened to 1-D).

        Returns:
            ``(n_samples, 1)`` float32 deviations.
        """
        raw = self._deviations(self._to_1d_f64(data))
        return finite_features(raw).reshape(-1, 1)

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Per-sample anomaly scores in ``[0, 1]`` from spike-rate deviation.

        Deviations are squashed monotonically via ``1 - exp(-r / scale)`` using
        the training scale from :meth:`fit`.
        """
        raw = self._deviations(self._to_1d_f64(data))
        scale = self._scale if self._is_fitted else self._squash_scale(raw)
        scores = finite_scores(1.0 - np.exp(-raw / scale)).astype(np.float32)
        return {
            "anomaly_score": float(scores.max()) if scores.size else 0.0,
            "scores": scores,
            "is_anomaly": scores > self.threshold,
            "confidence": scores,
        }
