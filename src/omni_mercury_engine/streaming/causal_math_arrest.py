# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Causal fixed-window execution of the 21-probe math-arrest ensemble.

The offline path fits the ensemble on a stream's probationary prefix and then
scores the remainder as one batch, so a point's score can depend on samples
that arrive after it (14 of the 21 probes compute window statistics). This
module runs the *identical instruments* — the same probes, the same
``PhiWeightedFusion`` with the same decorrelator, the same one-class
split-conformal alarm cut — in a strictly causal mode: each point is scored
as the last element of the fixed trailing window ending at it, so nothing
later than the point itself is ever read. No new algorithm is introduced;
only the execution order changes.

Measured on the four recorded NAB files against the offline instrument's
pinned triple (AUROC 0.749 / NAB standard score 33.7 / alarm rate 0.017 in
``tests/detectors/test_recorded_nab_series.py``), via
``benchmarks/nab_streaming_benchmark.py``; the causal pins live in
``tests/streaming/test_causal_math_arrest.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.detectors._calibration import conformal_cut
from omni_mercury_engine.detectors.math_arrest.arrest import AnomalyMathArrest

__all__ = ["CausalMathArrest"]

#: Default trailing-window length, matching ``EngineStreamSession``'s window.
DEFAULT_WINDOW = 256


class CausalMathArrest:
    """Fixed-window causal execution mode of :class:`AnomalyMathArrest`.

    Args:
        window: Trailing window length each point is scored within. The
            point's score is the window's last fused score, so only samples
            at or before the point are read.
        alpha: One-class split-conformal alarm level, the same promise the
            offline :class:`MathArrestDetector` makes (its
            ``CALIBRATION_ALPHA``).
        probes: Probe selection forwarded to the instrument (a preset name
            or probe class names); ``None`` is the full 21.
        weighting: Fusion weighting forwarded to the instrument.
    """

    def __init__(
        self,
        window: int = DEFAULT_WINDOW,
        alpha: float = 0.05,
        probes: str | list[str] | None = None,
        weighting: str = "uniform",
    ) -> None:
        """Initialize the instance."""
        if window < 8:
            raise ValueError(f"window must be >= 8 (the probes' minimum), got {window}")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self._window = window
        self._alpha = alpha
        self._arrest = AnomalyMathArrest(probes=probes, weighting=weighting)
        self._threshold: float | None = None
        self._history: list[float] = []
        self._is_fitted = False

    @property
    def threshold(self) -> float | None:
        """The fit-time conformal alarm cut; ``None`` before :meth:`fit`."""
        return self._threshold

    @property
    def window(self) -> int:
        """The trailing-window length each point is scored within."""
        return self._window

    def fit(self, warmup: npt.NDArray[np.float64]) -> CausalMathArrest:
        """Fit every probe on the warmup prefix and cut the alarm level.

        Identical to the offline adapter's fit: the probes learn their
        trajectories and normalizer scales from the warmup series, and the
        alarm threshold is the one-class split-conformal quantile of the
        warmup's own fused scores at ``alpha``.

        Args:
            warmup: The stream's leading, predominantly normal samples.

        Returns:
            ``self``.
        """
        series = np.asarray(warmup, dtype=np.float64).reshape(-1)
        self._arrest.fit(series)
        fused = np.asarray(self._arrest.detect(series), dtype=np.float64).reshape(-1)
        fused = fused[np.isfinite(fused)]
        cut = conformal_cut(np.sort(fused) if fused.size else None, self._alpha)
        self._threshold = float(cut) if cut is not None else float(self._arrest.threshold)
        self._history = list(series[-self._window :])
        self._is_fitted = True
        return self

    def update(self, value: float) -> dict[str, Any]:
        """Score one arriving sample from its trailing window only.

        Args:
            value: The newest sample.

        Returns:
            ``{"score", "is_anomaly", "threshold", "window_used"}`` for this
            point. The score is the fused score of the trailing window's last
            element — the arriving point — under the fitted instrument.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError("CausalMathArrest has not been fitted. Call fit() first.")
        self._history.append(float(value))
        if len(self._history) > self._window:
            del self._history[: len(self._history) - self._window]
        window = np.asarray(self._history, dtype=np.float64)
        fused = np.asarray(self._arrest.detect(window), dtype=np.float64).reshape(-1)
        score = float(fused[-1]) if fused.size else 0.0
        threshold = float(self._threshold) if self._threshold is not None else 0.5
        return {
            "score": score,
            "is_anomaly": bool(score > threshold),
            "threshold": threshold,
            "window_used": int(window.size),
        }

    def score_stream(
        self, values: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        """Feed a sequence of samples through :meth:`update`, in order.

        Args:
            values: Samples in arrival order.

        Returns:
            ``(scores, alarms)`` with one entry per sample, each computed
            from that sample's trailing window only.
        """
        scores = np.empty(len(values), dtype=np.float64)
        alarms = np.empty(len(values), dtype=bool)
        for i, value in enumerate(np.asarray(values, dtype=np.float64).reshape(-1)):
            result = self.update(float(value))
            scores[i] = result["score"]
            alarms[i] = result["is_anomaly"]
        return scores, alarms
