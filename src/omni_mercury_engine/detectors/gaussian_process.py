# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gaussian-Process regression residual detector for streaming time series.

Gaussian-Process regression (Rasmussen & Williams, *Gaussian Processes for
Machine Learning*, 2006) is the Bayesian non-parametric workhorse for smooth
function estimation with calibrated predictive uncertainty. For anomaly
detection it gives exactly what a residual detector wants: a one-step-ahead
predictive **mean and variance**, so a point can be scored by how many
predictive standard deviations it sits from the model — a proper *standardised*
residual rather than a raw error.

To stay streaming-cheap (full GP inference is cubic in the number of points),
this detector fits an RBF-kernel GP over a short trailing window to predict each
next observation. Hyperparameters (signal variance, length scale, noise) are
learned in :meth:`fit` by maximising the average windowed marginal likelihood
over a deterministic grid. It is pure NumPy/SciPy-free linear algebra (always
importable) and registered as an opt-in BASE detector.
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

__all__ = ["GaussianProcessDetector"]

_LN2 = float(np.log(2.0))


class GaussianProcessDetector(BaseDetector):
    """Windowed RBF Gaussian-Process one-step-ahead residual detector.

    For each observation the detector conditions an RBF-kernel GP on the
    preceding ``window`` points (their integer time offsets are the inputs) and
    predicts the current point, yielding a posterior mean and variance. The
    anomaly signal is the standardised residual ``|y - mean| / sqrt(var)``
    squashed into ``[0, 1]``. Hyperparameters are learned by grid-maximising the
    windowed log marginal likelihood on training data.
    """

    def __init__(
        self,
        window: int = 24,
        calibration_quantile: float = 0.98,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the GP detector.

        Args:
            window: Number of trailing points conditioned on for each one-step
                prediction. Must be >= 3.
            calibration_quantile: Training-residual quantile placed at the 0.5
                anomaly boundary; ``1 - calibration_quantile`` is the resulting
                normal-regime false-positive rate. Must be in ``(0, 1)``.
            config: Optional ``BaseDetector`` config (``threshold`` ...).

        Raises:
            ValueError: If ``window`` < 3 or ``calibration_quantile`` is out of
                ``(0, 1)``.
        """
        super().__init__(config)
        if window < 3:
            raise ValueError(f"window must be >= 3, got {window}")
        if not 0.0 < calibration_quantile < 1.0:
            raise ValueError(f"calibration_quantile must be in (0, 1), got {calibration_quantile}")
        self.window = int(window)
        self.calibration_quantile = float(calibration_quantile)
        self._length_scale: float = 1.0
        self._signal_var: float = 1.0
        self._noise_var: float = 1e-2
        self._scale: float = 1.0

    def is_fitted(self) -> bool:
        """Return ``True`` once :meth:`fit` has learned hyperparameters/scale."""
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

    def _rbf(self, xa: np.ndarray[Any, Any], xb: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """RBF (squared-exponential) kernel matrix for scalar time inputs."""
        d = xa.reshape(-1, 1) - xb.reshape(1, -1)
        return self._signal_var * np.exp(-0.5 * (d * d) / (self._length_scale**2))

    def _predict_point(
        self, x_ctx: np.ndarray[Any, Any], y_ctx: np.ndarray[Any, Any], x_star: float
    ) -> tuple[float, float]:
        """GP posterior mean/variance at ``x_star`` given a context window."""
        k = self._rbf(x_ctx, x_ctx) + self._noise_var * np.eye(x_ctx.size)
        k_star = self._rbf(x_ctx, np.array([x_star]))[:, 0]  # (n,)
        # Solve rather than invert for numerical stability.
        alpha = np.linalg.solve(k, y_ctx)
        mean = float(k_star @ alpha)
        v = np.linalg.solve(k, k_star)
        var = float(self._signal_var + self._noise_var - (k_star @ v))
        return mean, max(var, 1e-9)

    def _residuals(self, series: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Standardised one-step-ahead GP residuals for each observation."""
        n = series.size
        out = np.zeros(n, dtype=np.float64)
        if n == 0:
            return out
        offsets = np.arange(self.window, dtype=np.float64)
        for t in range(n):
            lo = t - self.window
            if lo < 0:
                # Not enough history: fall back to a robust local z-score.
                ctx = series[:t] if t > 0 else series[:1]
                mu = float(np.mean(ctx))
                sd = float(np.std(ctx)) + 1e-9
                out[t] = abs(series[t] - mu) / sd
                continue
            y_ctx = series[lo:t]
            mean, var = self._predict_point(offsets, y_ctx, float(self.window))
            out[t] = abs(series[t] - mean) / np.sqrt(var)
        return out

    def _log_marginal(self, series: np.ndarray[Any, Any]) -> float:
        """Average windowed log marginal likelihood used for hyperparam search."""
        n = series.size
        if n <= self.window:
            return -np.inf
        offsets = np.arange(self.window, dtype=np.float64)
        total = 0.0
        count = 0
        step = max(1, (n - self.window) // 32)  # subsample windows for speed
        for t in range(self.window, n, step):
            y_ctx = series[t - self.window : t]
            mean, var = self._predict_point(offsets, y_ctx, float(self.window))
            resid = series[t] - mean
            total += -0.5 * (resid * resid / var + np.log(2.0 * np.pi * var))
            count += 1
        return total / count if count else -np.inf

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> GaussianProcessDetector:
        """Learn RBF hyperparameters (grid marginal likelihood) and squash scale.

        Args:
            data: Training series (flattened to 1-D), predominantly normal.

        Returns:
            ``self``.
        """
        series = self._to_1d_f64(data)
        spread = float(np.var(series)) if series.size else 1.0
        self._signal_var = max(spread, 1e-6)
        if series.size >= 2:
            diffs = np.diff(series)
            med = float(np.median(diffs))
            mad = float(np.median(np.abs(diffs - med)))
            self._noise_var = max((1.4826 * mad) ** 2, 1e-6)
        else:
            self._noise_var = 1e-2

        # Deterministic grid over the length scale; pick the max-marginal-lik one.
        best_ls, best_ll = 1.0, -np.inf
        for ls in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
            self._length_scale = ls
            ll = self._log_marginal(series)
            if ll > best_ll:
                best_ll, best_ls = ll, ls
        self._length_scale = best_ls

        raw = self._residuals(series)
        self._scale = self._squash_scale(raw)
        self._is_fitted = True
        return self

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Per-sample fusion feature: the standardised GP residual.

        Args:
            data: Input series (flattened to 1-D).

        Returns:
            ``(n_samples, 1)`` float32 residuals.
        """
        raw = self._residuals(self._to_1d_f64(data))
        return finite_features(raw).reshape(-1, 1)

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Per-sample anomaly scores in ``[0, 1]`` from GP predictive residuals.

        Residuals are squashed monotonically via ``1 - exp(-r / scale)`` using
        the training scale from :meth:`fit` (or the input's own robust scale when
        unfitted).
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
