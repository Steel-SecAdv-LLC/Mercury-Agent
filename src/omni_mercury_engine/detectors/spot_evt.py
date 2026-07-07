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
the GPD tail on training data; ``detect`` streams new points, evolving the tail
online **on local state** (never mutating the fitted instance), flags excesses
over ``z_q`` as anomalies, and reports a calibrated score that is the modelled
tail probability mapped through the risk budget. It depends only on NumPy and is
registered as an opt-in BASE detector. Because the score is an EVT tail
probability, it gives the pipeline principled false-positive control (the ``q``
budget) rather than an ad-hoc threshold.

Purity & thread-safety
======================
The tail-model kernels :meth:`SPOTDetector._threshold_from_tail` and
:meth:`SPOTDetector._tail_probability` are **pure functions** of their explicit
arguments -- they read no instance attributes. After :meth:`SPOTDetector.fit`,
the fitted tail state (``_t``/``_zq``/``_gamma``/``_sigma``/``_n``/``_nt``) is
read-only; the online streaming update in :meth:`SPOTDetector._stream_scores`
runs entirely on *local* copies, so :meth:`SPOTDetector.detect` mutates nothing
on ``self``. That makes ``detect`` idempotent and safe to call concurrently on a
single fitted detector -- no snapshot/restore of instance state is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.detectors._calibration import bound_finite, finite_features
from omni_mercury_engine.detectors.detection_config import (
    DetectionConfig,
    guard_finite_scalar,
)

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
        self._detection_config = DetectionConfig.resolve(self._config)
        if not 0.0 < q < 1.0:
            raise ValueError(f"q must be in (0, 1), got {q}")
        if not 0.0 < init_level < 1.0:
            raise ValueError(f"init_level must be in (0, 1), got {init_level}")
        if depth < 0:
            raise ValueError(f"depth must be >= 0, got {depth}")
        self.q = float(q)
        self.init_level = float(init_level)
        self.depth = int(depth)
        # Fitted tail state -- set once in ``fit`` and thereafter READ-ONLY. The
        # streaming path (``_stream_scores``) copies these into locals and mutates
        # only the copies, so ``detect``/``extract_features`` never mutate the
        # instance and are safe to call concurrently on one detector (see the
        # module/``detect`` docstrings).
        self._t: float = 0.0  # peak-selection threshold
        self._zq: float = 0.0  # calibrated anomaly threshold (fitted)
        self._gamma: float = 0.0  # GPD shape
        self._sigma: float = 1.0  # GPD scale (fitted)
        self._n: int = 0  # samples seen during calibration
        self._nt: int = 0  # number of peaks at calibration

    def is_fitted(self) -> bool:
        """Return ``True`` once :meth:`fit` has calibrated the EVT tail."""
        return self._is_fitted

    def _to_1d_f64(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Coerce numpy/torch input to a finite 1-D float64 series (NaN policy applied)."""
        detach = getattr(data, "detach", None)
        if callable(detach):
            data = detach().cpu().numpy()
        # Honour this detector's fully-resolved config on the input path too, so the
        # documented precedence (defaults < file < env < per-detector ``config``)
        # holds for sanitisation -- not just the environment defaults ``bound_finite``
        # would read on its own. Identical to the env-only path when no file /
        # per-detector overrides are set.
        return bound_finite(
            np.asarray(data, dtype=np.float64),
            detector=self.name,
            policy=self._detection_config.nan_policy,
            max_magnitude=self._detection_config.max_magnitude,
        ).ravel()

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

    @staticmethod
    def _threshold_from_tail(
        t: float, q: float, n: int, nt: int, gamma: float, sigma: float
    ) -> float:
        """Map the risk budget ``q`` to the EVT anomaly threshold ``z_q``.

        A **pure** function of its explicit arguments -- it reads no instance
        state, so it can be evaluated against a caller's *local* streaming state
        without any snapshot/restore of ``self``. This is what makes
        :meth:`_stream_scores` (and therefore :meth:`detect`) mutation-free and
        safe for concurrent calls on one detector.
        """
        if nt <= 0 or n <= 0:
            return t
        r = q * n / nt
        if abs(gamma) < 1e-8:
            return t - sigma * float(np.log(max(r, 1e-12)))
        return t + (sigma / gamma) * (r ** (-gamma) - 1.0)

    @staticmethod
    def _tail_probability(
        value: float, t: float, n: int, nt: int, gamma: float, sigma: float
    ) -> float:
        """Modelled probability of exceeding ``value`` under the GPD tail.

        Pure function of its explicit arguments (see :meth:`_threshold_from_tail`);
        reads no instance state.
        """
        if value <= t or n <= 0:
            return 1.0
        excess = value - t
        base = nt / n  # P(X > t)
        if abs(gamma) < 1e-8:
            cond = float(np.exp(-excess / sigma))
        else:
            inner = 1.0 + gamma * excess / sigma
            cond = float(inner ** (-1.0 / gamma)) if inner > 0.0 else 0.0
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
        self._zq = self._threshold_from_tail(
            self._t, self.q, self._n, self._nt, self._gamma, self._sigma
        )
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
        """Stream points, updating the tail online, returning ``[0, 1]`` scores.

        The online tail update (``n``/``nt``/``sigma``/``zq``) runs entirely on
        **local** variables seeded from the read-only fitted state; ``self`` is
        never mutated. Two threads calling :meth:`detect` on the same instance
        therefore each get their own independent streaming state and cannot
        corrupt one another, and repeated calls are idempotent (a call no longer
        leaves an updated tail behind that shifts the next call's scores).
        """
        residual, _ = self._detrend(series)
        scores = np.zeros(series.size, dtype=np.float64)
        # Local streaming state -- copies of the fitted tail; only these mutate.
        t = self._t
        gamma = self._gamma
        n = self._n
        nt = self._nt
        sigma = self._sigma
        zq = self._zq
        q_floor = max(self.q, 1e-12)
        for i, value in enumerate(residual):
            v = float(value)
            if v > zq:
                # Anomaly: excess beyond the EVT threshold -> tail probability
                # below the risk budget maps to a high score in (0.5, 1].
                p = self._tail_probability(v, t, n, nt, gamma, sigma)
                scores[i] = float(np.clip(1.0 - 0.5 * p / q_floor, 0.5, 1.0))
                # SPOT does not feed anomalies back into the tail model.
            else:
                if v > t:
                    # Normal peak: update the (local) tail and refit the threshold.
                    n += 1
                    nt += 1
                    sigma = sigma + (v - t - sigma) / nt
                    zq = self._threshold_from_tail(t, self.q, n, nt, gamma, sigma)
                else:
                    n += 1
                p = self._tail_probability(v, t, n, nt, gamma, sigma)
                # Below-threshold points score in [0, 0.5): closer to 0.5 as the
                # modelled exceedance probability approaches the budget.
                scores[i] = float(np.clip(0.5 * (1.0 - p / q_floor), 0.0, 0.5))
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
        return finite_features(scores, detector=self.name).reshape(-1, 1)

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Per-sample anomaly scores in ``[0, 1]`` with EVT thresholding.

        Points beyond the EVT threshold ``z_q`` score above 0.5 (``is_anomaly``
        true by default), giving the pipeline a false-positive rate bounded by
        the risk budget ``q``.

        Thread-safety: this method mutates no instance state (the online tail
        update runs on locals in :meth:`_stream_scores`), so it is safe to call
        concurrently on a single fitted detector and is idempotent across
        repeated calls. The ``z_q`` / ``gamma`` metadata fields are guarded by the
        same finite/magnitude policy as the scores
        (:func:`~omni_mercury_engine.detectors.detection_config.guard_finite_scalar`).

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if not self._is_fitted:
            raise RuntimeError("SPOTDetector must be fit before use")
        series = self._to_1d_f64(data)
        scores = self._stream_scores(series).astype(np.float32)
        policy = self._detection_config.nan_policy
        max_mag = self._detection_config.max_magnitude
        return {
            "anomaly_score": float(scores.max()) if scores.size else 0.0,
            "scores": scores,
            "is_anomaly": scores > 0.5,
            "confidence": scores,
            "metadata": {
                "z_q": guard_finite_scalar(
                    self._zq, policy=policy, detector=self.name, field="z_q", max_magnitude=max_mag
                ),
                "gamma": guard_finite_scalar(
                    self._gamma,
                    policy=policy,
                    detector=self.name,
                    field="gamma",
                    max_magnitude=max_mag,
                ),
            },
        }
