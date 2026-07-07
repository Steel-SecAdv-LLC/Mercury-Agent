# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Echo-State-Network (reservoir-computing) predictive residual detector.

Reservoir computing / Echo State Networks (Jaeger, *The "echo state" approach to
analysing and training recurrent neural networks*, 2001) drive a large, fixed,
randomly-connected recurrent reservoir with the input signal and train only a
cheap linear readout. The reservoir's fading-memory dynamics embed the recent
history of the signal in a high-dimensional state, so a linear one-step-ahead
predictor over that state captures rich non-linear temporal structure without
back-propagation-through-time.

For anomaly detection the readout is trained (ridge regression) to predict the
next sample; the standardised prediction residual is the anomaly signal — small
while the dynamics match training, large when the signal leaves the learned
manifold. The reservoir is seeded for determinism. Pure NumPy (always
importable); registered as an opt-in BASE detector.
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

__all__ = ["EchoStateDetector"]

_LN2 = float(np.log(2.0))


class EchoStateDetector(BaseDetector):
    """Echo-State-Network one-step-ahead predictive residual detector.

    A fixed random reservoir (spectral-radius scaled for the echo-state
    property) is driven by the scalar series; a ridge-regression readout learned
    in :meth:`fit` predicts the next sample from the reservoir state. Each point
    is scored by the standardised one-step prediction residual squashed into
    ``[0, 1]``.
    """

    def __init__(
        self,
        reservoir_size: int = 100,
        spectral_radius: float = 0.9,
        leak_rate: float = 0.3,
        input_scaling: float = 1.0,
        ridge: float = 1e-4,
        calibration_quantile: float = 0.98,
        seed: int = 0,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the Echo-State detector.

        Args:
            reservoir_size: Number of reservoir units. Must be >= 8.
            spectral_radius: Target spectral radius of the reservoir weight
                matrix (echo-state property typically wants < 1). Must be > 0.
            leak_rate: Leaky-integrator rate in ``(0, 1]``.
            input_scaling: Scale applied to the input weights. Must be > 0.
            ridge: Tikhonov regularisation for the readout. Must be > 0.
            calibration_quantile: Training-residual quantile at the 0.5 boundary;
                ``1 - calibration_quantile`` is the normal-regime FPR. In ``(0, 1)``.
            seed: RNG seed for the fixed reservoir (determinism).
            config: Optional ``BaseDetector`` config (``threshold`` ...).

        Raises:
            ValueError: If any parameter is out of its valid range.
        """
        super().__init__(config)
        if reservoir_size < 8:
            raise ValueError(f"reservoir_size must be >= 8, got {reservoir_size}")
        if spectral_radius <= 0.0:
            raise ValueError("spectral_radius must be > 0")
        if not 0.0 < leak_rate <= 1.0:
            raise ValueError(f"leak_rate must be in (0, 1], got {leak_rate}")
        if input_scaling <= 0.0 or ridge <= 0.0:
            raise ValueError("input_scaling and ridge must be > 0")
        if not 0.0 < calibration_quantile < 1.0:
            raise ValueError(f"calibration_quantile must be in (0, 1), got {calibration_quantile}")
        self.reservoir_size = int(reservoir_size)
        self.spectral_radius = float(spectral_radius)
        self.leak_rate = float(leak_rate)
        self.input_scaling = float(input_scaling)
        self.ridge = float(ridge)
        self.calibration_quantile = float(calibration_quantile)
        self.seed = int(seed)
        self._w_res: np.ndarray[Any, Any] | None = None
        self._w_in: np.ndarray[Any, Any] | None = None
        self._w_out: np.ndarray[Any, Any] | None = None
        self._mean: float = 0.0
        self._std: float = 1.0
        self._scale: float = 1.0

    def is_fitted(self) -> bool:
        """Return ``True`` once :meth:`fit` has trained the readout/scale."""
        return self._is_fitted

    @staticmethod
    def _to_1d_f64(data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Coerce numpy/torch input to a finite 1-D float64 series."""
        detach = getattr(data, "detach", None)
        if callable(detach):
            data = detach().cpu().numpy()
        return bound_finite(np.asarray(data, dtype=np.float64)).ravel()

    def _squash_scale(self, raw: np.ndarray[Any, Any]) -> float:
        """Squash scale anchoring the ``calibration_quantile`` at score 0.5."""
        return squash_scale(raw, self.calibration_quantile)

    def _build_reservoir(self) -> None:
        """Instantiate the fixed, spectral-radius-scaled random reservoir."""
        rng = np.random.default_rng(self.seed)
        n = self.reservoir_size
        w = rng.uniform(-1.0, 1.0, (n, n))
        # Sparsify (~10% density) then rescale to the target spectral radius.
        mask = rng.uniform(0.0, 1.0, (n, n)) < 0.1
        w *= mask
        eigs = np.linalg.eigvals(w)
        radius = float(np.max(np.abs(eigs)))
        if radius > 0.0:
            w *= self.spectral_radius / radius
        self._w_res = w
        self._w_in = self.input_scaling * rng.uniform(-1.0, 1.0, (n, 1))

    def _run_states(self, series: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Collect leaky reservoir states driven by the (normalised) series."""
        assert self._w_res is not None and self._w_in is not None
        n = self.reservoir_size
        u = (series - self._mean) / self._std
        states = np.zeros((series.size, n), dtype=np.float64)
        x = np.zeros(n, dtype=np.float64)
        a = self.leak_rate
        for t in range(series.size):
            pre = self._w_in[:, 0] * u[t] + self._w_res @ x
            x = (1.0 - a) * x + a * np.tanh(pre)
            states[t] = x
        return states

    def _residuals(self, series: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Standardised one-step-ahead reservoir-readout residuals."""
        n = series.size
        out = np.zeros(n, dtype=np.float64)
        if n < 2 or self._w_out is None:
            return out
        states = self._run_states(series)
        u = (series - self._mean) / self._std
        # Predict u[t+1] from state[t]; residual aligned to the predicted index.
        bias = np.ones((states.shape[0], 1))
        aug = np.hstack([states, bias])
        pred = aug @ self._w_out  # predicts next normalised sample
        resid = np.zeros(n, dtype=np.float64)
        resid[1:] = np.abs(u[1:] - pred[:-1])
        resid[0] = resid[1] if n > 1 else 0.0
        out = resid
        return out

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> EchoStateDetector:
        """Build the reservoir and train the ridge readout + squash scale.

        Args:
            data: Training series (flattened to 1-D), predominantly normal.

        Returns:
            ``self``.
        """
        series = self._to_1d_f64(data)
        self._mean = float(np.mean(series)) if series.size else 0.0
        self._std = float(np.std(series)) + 1e-9
        self._build_reservoir()
        if series.size >= 2:
            states = self._run_states(series)
            u = (series - self._mean) / self._std
            bias = np.ones((states.shape[0], 1))
            aug = np.hstack([states, bias])
            # Ridge readout mapping state[t] -> u[t+1].
            x_train = aug[:-1]
            y_train = u[1:]
            d = x_train.shape[1]
            gram = x_train.T @ x_train + self.ridge * np.eye(d)
            self._w_out = np.linalg.solve(gram, x_train.T @ y_train)
        else:
            self._w_out = np.zeros((self.reservoir_size + 1,), dtype=np.float64)
        raw = self._residuals(series)
        self._scale = self._squash_scale(raw)
        self._is_fitted = True
        return self

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Per-sample fusion feature: the standardised readout residual.

        Args:
            data: Input series (flattened to 1-D).

        Returns:
            ``(n_samples, 1)`` float32 residuals.
        """
        raw = self._residuals(self._to_1d_f64(data))
        return finite_features(raw).reshape(-1, 1)

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Per-sample anomaly scores in ``[0, 1]`` from reservoir residuals.

        Residuals are squashed monotonically via ``1 - exp(-r / scale)`` using
        the training scale from :meth:`fit`.
        """
        raw = self._residuals(self._to_1d_f64(data))
        scale = self._scale if self._is_fitted else self._squash_scale(raw)
        scores = finite_scores(1.0 - np.exp(-raw / scale)).astype(np.float32)
        return {
            "anomaly_score": float(scores.max()) if scores.size else 0.0,
            "scores": scores,
            "is_anomaly": scores > self.threshold,
            "confidence": scores,
        }
