# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""SPOT / DSPOT Peaks-Over-Threshold (EVT) dynamic-threshold detector.

SPOT (Siffer et al., *Anomaly Detection in Streams with Extreme Value Theory*,
KDD 2017) sets streaming anomaly thresholds directly from Extreme Value Theory
rather than from a distributional assumption on the whole signal. The tail of a
stationary stream -- the excesses over a high initial quantile -- is modelled by
a Generalized Pareto Distribution (GPD); a target risk ``q`` (allowed false-
positive rate) is then mapped to a data-driven threshold ``z_q``. DSPOT is the
drift-aware variant that first removes a local moving-average trend so the
tail model applies to a de-trended, locally-stationary residual.

The detector conforms to the
:class:`~omni_mercury_engine.core.base.BaseDetector` contract. ``fit`` calibrates
the GPD tail on training data; ``detect`` streams new points, updates the tail
online, flags excesses over ``z_q`` as anomalies, and reports a calibrated score
that is the modelled tail probability mapped through the risk budget. It depends
only on NumPy and is registered as an opt-in BASE detector. Because the score is
an EVT tail probability, it gives the pipeline principled false-positive control
(the ``q`` budget) rather than an ad-hoc threshold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.core.base import BaseDetector

if TYPE_CHECKING:
    import torch

__all__ = ["SPOTDetector"]


class SPOTDetector(BaseDetector):
    """Streaming EVT detector with a data-driven Peaks-Over-Threshold threshold.

    Fits a Generalized Pareto tail to the excesses over a high empirical
    quantile and derives the anomaly threshold ``z_q`` from a target risk ``q``.
    Optionally de-trends the stream with a moving average first (DSPOT mode) to
    handle slow drift. Per-sample scores are the EVT tail probability of each
    observation, normalised into ``[0, 1]`` against the risk budget so an
    at-threshold point scores ~0.5.
    """

    def __init__(
        self,
        q: float = 1e-3,
        init_level: float = 0.98,
        depth: int = 0,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the SPOT/DSPOT detector.

        Args:
            q: Target risk -- the desired probability of exceeding the threshold
                under the normal regime (false-positive budget). Must be in
                ``(0, 1)``.
            init_level: Empirical quantile used as the peak-selection threshold
                ``t`` for the GPD tail fit. Must be in ``(0, 1)``.
            depth: DSPOT moving-average window for drift removal. ``0`` disables
                de-trending (plain SPOT); ``> 0`` selects DSPOT. Must be >= 0.
            config: Optional ``BaseDetector`` config.

        Raises:
            ValueError: If ``q`` or ``init_level`` is out of ``(0, 1)`` or
                ``depth`` is negative.
        """
        super().__init__(config)
        if not 0.0 < q < 1.0:
            raise ValueError(f"q must be in (0, 1), got {q}")
        if not 0.0 < init_level < 1.0:
            raise ValueError(f"init_level must be in (0, 1), got {init_level}")
        if depth < 0:
            raise ValueError(f"depth must be >= 0, got {depth}")
        self.q = float(q)
        self.init_level = float(init_level)
        self.depth = int(depth)
        # Fitted tail state.
        self._t: float = 0.0  # peak-selection threshold
        self._zq: float = 0.0  # anomaly threshold
        self._gamma: float = 0.0  # GPD shape
        self._sigma: float = 1.0  # GPD scale
        self._n: int = 0  # samples seen during calibration
        self._nt: int = 0  # number of peaks
        self._drift: float = 0.0  # last DSPOT trend estimate

    def is_fitted(self) -> bool:
        """Return ``True`` once :meth:`fit` has calibrated the EVT tail."""
        return self._is_fitted

    @staticmethod
    def _to_1d_f64(data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Coerce numpy/torch input to a finite 1-D float64 series."""
        detach = getattr(data, "detach", None)
        if callable(detach):
            data = detach().cpu().numpy()
        return np.nan_to_num(np.asarray(data, dtype=np.float64)).ravel()

    @staticmethod
    def _grimshaw(peaks: np.ndarray[Any, Any]) -> tuple[float, float]:
        """Fit GPD ``(gamma, sigma)`` to excesses by the method of moments.

        A full Grimshaw MLE is overkill for a streaming guard; the method of
        moments is stable, closed-form and adequate for setting a tail
        threshold. For excesses ``Y`` with mean ``m`` and variance ``v``:
        ``gamma = 0.5 * (1 - m^2 / v)`` and ``sigma = 0.5 * m * (m^2 / v + 1)``.
        Falls back to an exponential tail (``gamma = 0``) when the variance is
        degenerate.
        """
        if peaks.size < 2:
            m = float(np.mean(peaks)) if peaks.size else 1.0
            return 0.0, max(m, 1e-8)
        m = float(np.mean(peaks))
        v = float(np.var(peaks))
        if v < 1e-12 or m <= 0.0:
            return 0.0, max(m, 1e-8)
        ratio = m * m / v
        gamma = 0.5 * (1.0 - ratio)
        sigma = 0.5 * m * (ratio + 1.0)
        return float(gamma), float(max(sigma, 1e-8))

    def _threshold_from_tail(self, gamma: float, sigma: float) -> float:
        """Map the risk budget ``q`` to the EVT anomaly threshold ``z_q``."""
        if self._nt <= 0 or self._n <= 0:
            return self._t
        r = self.q * self._n / self._nt
        if abs(gamma) < 1e-8:
            return self._t - sigma * float(np.log(max(r, 1e-12)))
        return self._t + (sigma / gamma) * (r ** (-gamma) - 1.0)

    def _tail_probability(self, value: float) -> float:
        """Modelled probability of exceeding ``value`` under the GPD tail."""
        if value <= self._t or self._n <= 0:
            return 1.0
        excess = value - self._t
        base = self._nt / self._n  # P(X > t)
        if abs(self._gamma) < 1e-8:
            cond = float(np.exp(-excess / self._sigma))
        else:
            inner = 1.0 + self._gamma * excess / self._sigma
            cond = float(inner ** (-1.0 / self._gamma)) if inner > 0.0 else 0.0
        return float(np.clip(base * cond, 0.0, 1.0))

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> SPOTDetector:
        """Calibrate the GPD tail and initial threshold on training data.

        Args:
            data: Calibration series (flattened to 1-D). In DSPOT mode the local
                moving-average trend is removed before the tail is fit.

        Returns:
            ``self``.

        Raises:
            ValueError: If fewer than 10 samples are supplied (too few to
                estimate a tail).
        """
        series = self._to_1d_f64(data)
        if series.size < 10:
            raise ValueError("SPOT calibration needs at least 10 samples")
        residual = self._detrend(series)[0]
        self._t = float(np.quantile(residual, self.init_level))
        peaks = residual[residual > self._t] - self._t
        self._n = int(residual.size)
        self._nt = int(peaks.size)
        self._gamma, self._sigma = self._grimshaw(peaks)
        self._zq = self._threshold_from_tail(self._gamma, self._sigma)
        self._is_fitted = True
        return self

    def _detrend(
        self, series: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Return ``(residual, trend)``; identity when ``depth == 0``."""
        if self.depth <= 0:
            return series, np.zeros_like(series)
        window = min(self.depth, series.size)
        kernel = np.ones(window, dtype=np.float64) / float(window)
        pad = window
        padded = np.concatenate([np.full(pad, series[0]), series])
        trend_full = np.convolve(padded, kernel, mode="same")[pad:]
        return series - trend_full, trend_full

    def _stream_scores(self, series: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Stream points, updating the tail online, returning ``[0, 1]`` scores."""
        residual, _ = self._detrend(series)
        scores = np.zeros(series.size, dtype=np.float64)
        for i, value in enumerate(residual):
            v = float(value)
            if v > self._zq:
                # Anomaly: excess beyond the EVT threshold -> tail probability
                # below the risk budget maps to a high score in (0.5, 1].
                p = self._tail_probability(v)
                scores[i] = float(np.clip(1.0 - 0.5 * p / max(self.q, 1e-12), 0.5, 1.0))
                # SPOT does not feed anomalies back into the tail model.
            else:
                if v > self._t:
                    # Normal peak: update the tail and refit the threshold.
                    self._n += 1
                    self._nt += 1
                    self._sigma = self._sigma + (v - self._t - self._sigma) / self._nt
                    self._zq = self._threshold_from_tail(self._gamma, self._sigma)
                else:
                    self._n += 1
                p = self._tail_probability(v)
                # Below-threshold points score in [0, 0.5): closer to 0.5 as the
                # modelled exceedance probability approaches the budget.
                scores[i] = float(np.clip(0.5 * (1.0 - p / max(self.q, 1e-12)), 0.0, 0.5))
        return scores

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Per-sample fusion feature: the EVT-normalised tail score.

        Args:
            data: Input series (flattened to 1-D).

        Returns:
            ``(n_samples, 1)`` float32 scores.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if not self._is_fitted:
            raise RuntimeError("SPOTDetector must be fit before use")
        series = self._to_1d_f64(data)
        scores = self._stream_scores(series)
        return scores.astype(np.float32).reshape(-1, 1)

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Per-sample anomaly scores in ``[0, 1]`` with EVT thresholding.

        Points beyond the EVT threshold ``z_q`` score above 0.5 (``is_anomaly``
        true by default), giving the pipeline a false-positive rate bounded by
        the risk budget ``q``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if not self._is_fitted:
            raise RuntimeError("SPOTDetector must be fit before use")
        series = self._to_1d_f64(data)
        scores = self._stream_scores(series).astype(np.float32)
        return {
            "anomaly_score": float(scores.max()) if scores.size else 0.0,
            "scores": scores,
            "is_anomaly": scores > 0.5,
            "confidence": scores,
            "metadata": {"z_q": float(self._zq), "gamma": float(self._gamma)},
        }
