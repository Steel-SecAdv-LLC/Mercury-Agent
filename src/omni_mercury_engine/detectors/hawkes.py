# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hawkes-process event-rate / burst detector for count streams.

A Hawkes process is a self-exciting point process: each event transiently
raises the intensity of further events, so the conditional intensity captures
*clustering* -- exactly the "bursts" that fixed-rate (Poisson) baselines miss.
For a stream of per-bin event counts ``n_t`` the intensity obeys the recursion

    ``lambda_t = mu + g_t``,  ``g_t = exp(-beta) * (g_{t-1} + alpha * n_{t-1})``,

where ``mu`` is the background rate, ``alpha`` the excitation gain and ``beta``
the exponential decay. An anomaly is a bin whose observed count departs sharply
from its Hawkes-predicted intensity -- a burst (count far above intensity) or an
unexpected silence (count far below an elevated intensity). The signal is the
absolute Pearson residual ``|n_t - lambda_t| / sqrt(lambda_t)``.

The detector conforms to the
:class:`~omni_mercury_engine.core.base.BaseDetector` contract. ``fit`` estimates
``mu`` and the excitation ``(alpha, beta)`` from training counts and learns a
robust residual scale; ``detect`` streams the intensity recursion and squashes
residuals into ``[0, 1]``. It depends only on NumPy and is registered as an
opt-in BASE detector.
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

__all__ = ["HawkesBurstDetector"]

# ``1 - exp(-r / scale) = 0.5`` at ``r = scale * ln 2``; anchoring ``scale`` to a
# high training quantile therefore places the 0.5 anomaly boundary at that
# quantile, giving a controlled ``1 - calibration_quantile`` false-positive rate.
_LN2 = float(np.log(2.0))


class HawkesBurstDetector(BaseDetector):
    """Self-exciting point-process detector for bursts in count streams.

    Models the conditional intensity of a per-bin count series with an
    exponential-kernel Hawkes process and scores each bin by how far its count
    deviates from the predicted intensity (absolute Pearson residual). Detects
    both bursts and anomalous silences. ``fit`` estimates the background rate,
    the excitation gain/decay, and a robust residual scale used to squash scores
    into ``[0, 1]``.
    """

    def __init__(
        self,
        beta: float = 1.0,
        alpha: float | None = None,
        calibration_quantile: float = 0.98,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the Hawkes burst detector.

        Args:
            beta: Exponential decay rate of the excitation kernel (larger =
                shorter memory). Must be > 0.
            alpha: Optional fixed excitation gain (branching contribution per
                event). When ``None`` it is estimated in :meth:`fit` from the
                lag-1 autocovariance of the training counts. Must be >= 0.
            calibration_quantile: Training-residual quantile placed at the 0.5
                anomaly boundary; ``1 - calibration_quantile`` is the resulting
                normal-regime false-positive rate. Must be in ``(0, 1)``.
            config: Optional ``BaseDetector`` config.

        Raises:
            ValueError: If ``beta <= 0``, ``alpha`` is negative, or
                ``calibration_quantile`` is out of ``(0, 1)``.
        """
        super().__init__(config)
        if beta <= 0.0:
            raise ValueError(f"beta must be > 0, got {beta}")
        if alpha is not None and alpha < 0.0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if not 0.0 < calibration_quantile < 1.0:
            raise ValueError(f"calibration_quantile must be in (0, 1), got {calibration_quantile}")
        self.beta = float(beta)
        self._alpha_override = None if alpha is None else float(alpha)
        self.calibration_quantile = float(calibration_quantile)
        self._mu: float = 1.0
        self._alpha: float = 0.0
        self._scale: float = 1.0

    def is_fitted(self) -> bool:
        """Return ``True`` once :meth:`fit` has estimated the process."""
        return self._is_fitted

    def _squash_scale(self, raw: np.ndarray[Any, Any]) -> float:
        """Squash scale anchoring the ``calibration_quantile`` at score 0.5."""
        return squash_scale(raw, self.calibration_quantile)

    def _to_1d_f64(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Coerce numpy/torch input to a finite, non-negative 1-D count series."""
        detach = getattr(data, "detach", None)
        if callable(detach):
            data = detach().cpu().numpy()
        arr = bound_finite(np.asarray(data, dtype=np.float64), detector=self.name).ravel()
        # Counts are non-negative; clamp defensively so residuals stay meaningful.
        return np.maximum(arr, 0.0)

    def _intensity(self, counts: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Conditional intensity ``lambda_t`` for each bin via the recursion."""
        n = counts.size
        lam = np.empty(n, dtype=np.float64)
        decay = float(np.exp(-self.beta))
        g = 0.0
        for t in range(n):
            lam[t] = self._mu + g
            g = decay * (g + self._alpha * counts[t])
        return lam

    def _residuals(self, counts: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Absolute Pearson residuals ``|n_t - lambda_t| / sqrt(lambda_t)``."""
        lam = self._intensity(counts)
        denom = np.sqrt(np.maximum(lam, 1e-8))
        return np.abs(counts - lam) / denom

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> HawkesBurstDetector:
        """Estimate ``mu``, the excitation gain, and the residual scale.

        The background rate ``mu`` is the mean training count. When ``alpha`` was
        not fixed at construction it is estimated from the normalised lag-1
        autocovariance (the empirical footprint of self-excitation), then clamped
        to keep the branching ratio ``alpha / (exp(beta) - 1)`` below 1 so the
        process stays sub-critical (stationary).

        Args:
            data: Training count series (flattened to 1-D, clamped non-negative).

        Returns:
            ``self``.
        """
        counts = self._to_1d_f64(data)
        self._mu = float(np.mean(counts)) if counts.size else 1.0
        self._mu = max(self._mu, 1e-8)

        if self._alpha_override is not None:
            self._alpha = self._alpha_override
        elif counts.size >= 3:
            centred = counts - counts.mean()
            var = float(np.mean(centred * centred))
            cov1 = float(np.mean(centred[1:] * centred[:-1]))
            rho1 = cov1 / var if var > 1e-12 else 0.0
            # Positive lag-1 correlation is the excitation signature; map it to a
            # gain and keep the branching ratio < 1 for stationarity.
            branch_cap = 0.9 * (float(np.exp(self.beta)) - 1.0)
            self._alpha = float(np.clip(max(rho1, 0.0) * self._mu, 0.0, branch_cap))
        else:
            self._alpha = 0.0

        raw = self._residuals(counts)
        self._scale = self._squash_scale(raw)
        self._is_fitted = True
        return self

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Per-sample fusion feature: the absolute Pearson residual.

        Args:
            data: Input count series (flattened to 1-D).

        Returns:
            ``(n_samples, 1)`` float32 residuals.
        """
        counts = self._to_1d_f64(data)
        raw = self._residuals(counts)
        return finite_features(raw, detector=self.name).reshape(-1, 1)

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Per-sample burst/silence anomaly scores in ``[0, 1]``.

        Residuals are squashed monotonically via ``1 - exp(-r / scale)`` using
        the training scale from :meth:`fit` (or the input's own robust scale when
        unfitted), so bins consistent with the Hawkes intensity score near 0 and
        bursts/silences approach 1.
        """
        counts = self._to_1d_f64(data)
        raw = self._residuals(counts)
        scale = self._scale if self._is_fitted else self._squash_scale(raw)
        scores = finite_scores(1.0 - np.exp(-raw / scale), detector=self.name).astype(np.float32)
        return {
            "anomaly_score": float(scores.max()) if scores.size else 0.0,
            "scores": scores,
            "is_anomaly": scores > self.threshold,
            "confidence": scores,
        }
