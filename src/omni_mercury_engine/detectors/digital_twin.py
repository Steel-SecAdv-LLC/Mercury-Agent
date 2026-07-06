# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Digital-twin simulation-residual detector.

A *digital twin* is an executable model of a system kept in step with its real
counterpart; monitoring the divergence between the observed signal and the
twin's simulation is a standard industrial anomaly-detection pattern (Grieves &
Vickers, *Digital Twin: Mitigating Unpredictable, Undesirable Emergent Behavior
in Complex Systems*, 2017). When the plant behaves as modelled, observed and
simulated trajectories track each other; a fault, attack, or regime change makes
them diverge.

This detector builds a self-contained twin — a linear auto-regressive forward
model identified from training data — and scores each observation two ways: the
one-step-ahead residual (observed vs. twin prediction) and the free-running
simulation divergence over a short horizon (how fast the twin and reality drift
apart when the twin is left to run open-loop). Both feed a single squashed
anomaly score. Pure NumPy (always importable); registered as an opt-in BASE
detector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.core.base import BaseDetector

if TYPE_CHECKING:
    import torch

__all__ = ["DigitalTwinResidualDetector"]

_LN2 = float(np.log(2.0))


class DigitalTwinResidualDetector(BaseDetector):
    """Observed-vs-simulated divergence detector backed by an AR digital twin.

    :meth:`fit` identifies an auto-regressive forward model (the twin) by least
    squares on training data. :meth:`detect` scores each point by combining the
    one-step-ahead residual with a short free-running simulation divergence,
    squashed into ``[0, 1]``.
    """

    def __init__(
        self,
        order: int = 4,
        horizon: int = 5,
        divergence_weight: float = 0.5,
        ridge: float = 1e-6,
        calibration_quantile: float = 0.98,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the digital-twin detector.

        Args:
            order: AR order ``p`` of the twin forward model. Must be >= 1.
            horizon: Free-running simulation horizon used for the divergence
                term. Must be >= 1.
            divergence_weight: Blend in ``[0, 1]`` between the one-step residual
                (weight ``1 - divergence_weight``) and the free-run divergence
                (weight ``divergence_weight``).
            ridge: Tikhonov regularisation for the least-squares identification.
                Must be > 0.
            calibration_quantile: Training-residual quantile at the 0.5 boundary;
                ``1 - calibration_quantile`` is the normal-regime FPR. In ``(0, 1)``.
            config: Optional ``BaseDetector`` config (``threshold`` ...).

        Raises:
            ValueError: If any parameter is out of its valid range.
        """
        super().__init__(config)
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if not 0.0 <= divergence_weight <= 1.0:
            raise ValueError(f"divergence_weight must be in [0, 1], got {divergence_weight}")
        if ridge <= 0.0:
            raise ValueError("ridge must be > 0")
        if not 0.0 < calibration_quantile < 1.0:
            raise ValueError(f"calibration_quantile must be in (0, 1), got {calibration_quantile}")
        self.order = int(order)
        self.horizon = int(horizon)
        self.divergence_weight = float(divergence_weight)
        self.ridge = float(ridge)
        self.calibration_quantile = float(calibration_quantile)
        self._coef: np.ndarray[Any, Any] | None = None
        self._intercept: float = 0.0
        self._resid_std: float = 1.0
        self._scale: float = 1.0

    def is_fitted(self) -> bool:
        """Return ``True`` once :meth:`fit` has identified the twin/scale."""
        return self._is_fitted

    @staticmethod
    def _to_1d_f64(data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Coerce numpy/torch input to a finite 1-D float64 series."""
        detach = getattr(data, "detach", None)
        if callable(detach):
            data = detach().cpu().numpy()
        return np.nan_to_num(np.asarray(data, dtype=np.float64)).ravel()

    def _squash_scale(self, raw: np.ndarray[Any, Any]) -> float:
        """Squash scale anchoring the ``calibration_quantile`` at score 0.5."""
        q = float(np.quantile(raw, self.calibration_quantile))
        if q < 1e-9:
            q = float(np.mean(raw)) + 1e-9
        return max(q / _LN2, 1e-9)

    def _design(
        self, series: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Build the AR design matrix ``X`` and target ``y`` (order lags)."""
        p = self.order
        n = series.size
        rows = n - p
        x = np.zeros((rows, p), dtype=np.float64)
        for i in range(p):
            x[:, i] = series[p - 1 - i : n - 1 - i]
        y = series[p:]
        return x, y

    def _twin_step(self, lags: np.ndarray[Any, Any]) -> float:
        """One twin forward step from the most-recent ``order`` values."""
        assert self._coef is not None
        return float(self._coef @ lags + self._intercept)

    def _residuals(self, series: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Blended one-step + free-run divergence residual per observation."""
        n = series.size
        out = np.zeros(n, dtype=np.float64)
        if self._coef is None or n <= self.order:
            return out
        p = self.order
        w = self.divergence_weight
        for t in range(p, n):
            lags = series[t - 1 :: -1][:p]  # [y_{t-1}, ..., y_{t-p}]
            one_step = abs(series[t] - self._twin_step(lags))
            # Free-running simulation seeded at t-p, run open-loop to t.
            sim = list(series[t - p : t])
            horizon = min(self.horizon, t - p + 1)
            for _ in range(horizon):
                nxt = self._twin_step(np.asarray(sim[-1 : -p - 1 : -1][:p]))
                sim.append(nxt)
            free_run = abs(series[t] - sim[min(horizon, len(sim) - 1)])
            out[t] = (1.0 - w) * one_step + w * free_run
        out[:p] = out[p] if n > p else 0.0
        return out / max(self._resid_std, 1e-9)

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> DigitalTwinResidualDetector:
        """Identify the AR twin (ridge least squares) and set the squash scale.

        Args:
            data: Training series (flattened to 1-D), predominantly normal.

        Returns:
            ``self``.
        """
        series = self._to_1d_f64(data)
        if series.size > self.order + 1:
            x, y = self._design(series)
            x_aug = np.hstack([x, np.ones((x.shape[0], 1))])
            d = x_aug.shape[1]
            gram = x_aug.T @ x_aug + self.ridge * np.eye(d)
            beta = np.linalg.solve(gram, x_aug.T @ y)
            self._coef = beta[:-1]
            self._intercept = float(beta[-1])
            pred = x_aug @ beta
            self._resid_std = float(np.std(y - pred)) + 1e-9
        else:
            self._coef = np.zeros(self.order, dtype=np.float64)
            self._intercept = float(np.mean(series)) if series.size else 0.0
            self._resid_std = 1.0
        raw = self._residuals(series)
        self._scale = self._squash_scale(raw)
        self._is_fitted = True
        return self

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Per-sample fusion feature: the blended twin-divergence residual.

        Args:
            data: Input series (flattened to 1-D).

        Returns:
            ``(n_samples, 1)`` float32 residuals.
        """
        raw = self._residuals(self._to_1d_f64(data))
        return raw.astype(np.float32).reshape(-1, 1)

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Per-sample anomaly scores in ``[0, 1]`` from twin divergence.

        Residuals are squashed monotonically via ``1 - exp(-r / scale)`` using
        the training scale from :meth:`fit`.
        """
        raw = self._residuals(self._to_1d_f64(data))
        scale = self._scale if self._is_fitted else self._squash_scale(raw)
        scores = np.clip(1.0 - np.exp(-raw / scale), 0.0, 1.0).astype(np.float32)
        return {
            "anomaly_score": float(scores.max()) if scores.size else 0.0,
            "scores": scores,
            "is_anomaly": scores > self.threshold,
            "confidence": scores,
        }
