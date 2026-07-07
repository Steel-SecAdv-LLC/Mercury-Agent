# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Survival / time-to-event hazard detector for inter-event-time streams.

Survival analysis models the distribution of *time until an event* through the
survival function ``S(t) = P(T > t)`` and the hazard ``h(t)`` (Kaplan & Meier,
1958; Cox, *Regression Models and Life-Tables*, 1972). Many operational signals
are naturally inter-arrival times — request gaps, heartbeat intervals, retry
delays — and an anomaly is an event that occurs at a time the learned survival
model considers improbable: an unusually long gap (a stall) or an unusually
short one (a burst), i.e. a point in the tail of the fitted distribution.

This detector treats consecutive-sample gaps as durations, fits a non-parametric
Kaplan-Meier baseline survival curve on training durations, and layers a Cox
proportional-hazards deviation using a recent-rate covariate. Each observation is
scored by its tail-improbability under the fitted model, squashed into ``[0, 1]``.
Pure NumPy (always importable); registered as an opt-in BASE detector.
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

__all__ = ["SurvivalHazardDetector"]

_LN2 = float(np.log(2.0))


class SurvivalHazardDetector(BaseDetector):
    """Kaplan-Meier + Cox proportional-hazards inter-event-time detector.

    Durations are taken as ``|Δ signal|`` between consecutive samples (or the
    raw non-negative series when already a duration stream). :meth:`fit` builds
    the empirical Kaplan-Meier survival curve and a Cox hazard-ratio slope on a
    recent-rate covariate. The anomaly signal is ``-log`` of the two-sided tail
    probability of each duration (small tail probability → large score),
    squashed into ``[0, 1]``.
    """

    def __init__(
        self,
        covariate_window: int = 16,
        calibration_quantile: float = 0.98,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the survival detector.

        Args:
            covariate_window: Trailing window used to compute the recent-rate
                covariate for the Cox proportional-hazards adjustment. Must be
                >= 2.
            calibration_quantile: Training tail-surprisal quantile at the 0.5
                boundary; ``1 - calibration_quantile`` is the normal-regime FPR.
                Must be in ``(0, 1)``.
            config: Optional ``BaseDetector`` config (``threshold`` ...).

        Raises:
            ValueError: If ``covariate_window`` < 2 or ``calibration_quantile``
                is out of ``(0, 1)``.
        """
        super().__init__(config)
        if covariate_window < 2:
            raise ValueError(f"covariate_window must be >= 2, got {covariate_window}")
        if not 0.0 < calibration_quantile < 1.0:
            raise ValueError(f"calibration_quantile must be in (0, 1), got {calibration_quantile}")
        self.covariate_window = int(covariate_window)
        self.calibration_quantile = float(calibration_quantile)
        self._km_times: np.ndarray[Any, Any] | None = None
        self._km_surv: np.ndarray[Any, Any] | None = None
        self._cox_beta: float = 0.0
        self._cov_mean: float = 0.0
        self._scale: float = 1.0

    def is_fitted(self) -> bool:
        """Return ``True`` once :meth:`fit` has fitted the survival model."""
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

    def _durations(self, series: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Non-negative duration stream: |first difference| of the series."""
        if series.size < 2:
            return np.abs(series)
        d = np.abs(np.diff(series))
        return np.concatenate([[d[0]], d])

    def _km_fit(self, durations: np.ndarray[Any, Any]) -> None:
        """Fit the empirical Kaplan-Meier survival curve (no censoring)."""
        times = np.sort(durations)
        n = times.size
        uniq, counts = np.unique(times, return_counts=True)
        at_risk = n - np.concatenate([[0], np.cumsum(counts)[:-1]])
        surv = np.cumprod(1.0 - counts / np.maximum(at_risk, 1))
        self._km_times = uniq
        self._km_surv = surv

    def _survival_prob(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Right-tail survival ``S(t) = P(T > t)`` via the KM step function."""
        assert self._km_times is not None and self._km_surv is not None
        idx = np.searchsorted(self._km_times, x, side="right") - 1
        s = np.where(idx >= 0, self._km_surv[np.clip(idx, 0, self._km_surv.size - 1)], 1.0)
        return np.clip(s, 1e-9, 1.0)

    def _covariate(self, durations: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Recent-rate covariate: trailing mean of the last ``covariate_window`` durations.

        Computed in O(n) with a prefix sum. The window ending at ``t`` spans
        ``[t - w + 1, t]`` (clamped at the start), so it holds exactly ``w``
        samples once warmed up.
        """
        w = self.covariate_window
        n = durations.size
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        prefix = np.concatenate(([0.0], np.cumsum(durations, dtype=np.float64)))
        idx = np.arange(n)
        lo = np.maximum(0, idx - w + 1)
        counts = (idx - lo + 1).astype(np.float64)
        return (prefix[idx + 1] - prefix[lo]) / counts

    def _surprisal(self, series: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Two-sided tail surprisal of each duration under the fitted model."""
        durations = self._durations(series)
        if self._km_times is None:
            return np.zeros(series.size, dtype=np.float64)
        surv = self._survival_prob(durations)  # P(T > t): small => long gap
        left = 1.0 - surv  # P(T <= t): small => short gap
        tail = np.minimum(surv, left)  # two-sided tail probability
        base = -np.log(np.clip(2.0 * tail, 1e-9, 1.0))
        # Cox proportional-hazards adjustment: hazard ratio exp(beta * (cov-mean))
        # inflates surprisal when the local rate covariate departs from baseline.
        cov = self._covariate(durations)
        hazard_log_ratio = np.abs(self._cox_beta * (cov - self._cov_mean))
        return base + hazard_log_ratio

    def _fit_cox(self, durations: np.ndarray[Any, Any]) -> None:
        """Estimate a scalar Cox slope by one Newton step of the partial LL.

        A single covariate ``z`` (the recent-rate) is regressed against event
        ordering via the Cox partial likelihood; one Newton-Raphson step from
        ``beta=0`` yields a stable, closed-form slope estimate without external
        optimisers.
        """
        cov = self._covariate(durations)
        self._cov_mean = float(np.mean(cov))
        order = np.argsort(durations)
        z = cov[order]
        n = z.size
        # Risk sets are suffixes under ascending duration order.
        # Score U and information I of the Cox partial LL at beta = 0.
        suffix_sum = np.cumsum(z[::-1])[::-1]
        suffix_sq = np.cumsum((z * z)[::-1])[::-1]
        risk_n = np.arange(n, 0, -1)
        zbar = suffix_sum / risk_n
        u = float(np.sum(z - zbar))
        info = float(np.sum(suffix_sq / risk_n - zbar * zbar))
        self._cox_beta = u / info if info > 1e-9 else 0.0

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> SurvivalHazardDetector:
        """Fit the Kaplan-Meier baseline + Cox slope and the squash scale.

        Args:
            data: Training series (flattened to 1-D), predominantly normal.

        Returns:
            ``self``.
        """
        series = self._to_1d_f64(data)
        durations = self._durations(series)
        if durations.size >= 2:
            self._km_fit(durations)
            self._fit_cox(durations)
        else:
            self._km_times = np.array([0.0])
            self._km_surv = np.array([1.0])
            self._cox_beta = 0.0
        raw = self._surprisal(series)
        self._scale = self._squash_scale(raw)
        self._is_fitted = True
        return self

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Per-sample fusion feature: the tail surprisal.

        Args:
            data: Input series (flattened to 1-D).

        Returns:
            ``(n_samples, 1)`` float32 surprisal values.
        """
        raw = self._surprisal(self._to_1d_f64(data))
        return finite_features(raw).reshape(-1, 1)

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Per-sample anomaly scores in ``[0, 1]`` from survival tail surprisal.

        Surprisal is squashed monotonically via ``1 - exp(-r / scale)`` using
        the training scale from :meth:`fit`.
        """
        raw = self._surprisal(self._to_1d_f64(data))
        scale = self._scale if self._is_fitted else self._squash_scale(raw)
        scores = finite_scores(1.0 - np.exp(-raw / scale)).astype(np.float32)
        return {
            "anomaly_score": float(scores.max()) if scores.size else 0.0,
            "scores": scores,
            "is_anomaly": scores > self.threshold,
            "confidence": scores,
        }
