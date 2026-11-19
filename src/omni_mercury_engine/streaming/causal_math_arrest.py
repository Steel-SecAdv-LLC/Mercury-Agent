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

from omni_mercury_engine.core.centralized_constants import LYAPUNOV
from omni_mercury_engine.detectors._calibration import conformal_cut
from omni_mercury_engine.detectors.math_arrest.arrest import AnomalyMathArrest

__all__ = ["CausalMathArrest"]

#: Default trailing-window length, matching ``EngineStreamSession``'s window.
DEFAULT_WINDOW = 256
#: Steps after an excess onset before the decay envelope is enforced: the
#: envelope eps*exp(-lambda*t) only separates from eps meaningfully once a few
#: steps have passed, and flagging inside the grace would abstain on the very
#: alarm the detector exists to raise.
STABILITY_GRACE = 8
#: Consecutive envelope violations before the abstention flag raises: a single
#: stray sample above the envelope is noise, a run of them is a trajectory
#: that is not contracting.
STABILITY_PATIENCE = 4


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
        #: Trajectory-stability monitor state (the documented decay envelope
        #: ``eps * exp(-lambda t)`` with the system's own convergence rate).
        self._lambda = float(LYAPUNOV.LAMBDA_CONVERGENCE)
        self._excess_onset: float | None = None
        self._steps_since_onset = 0
        self._violation_streak = 0

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
        self._excess_onset = None
        self._steps_since_onset = 0
        self._violation_streak = 0
        self._is_fitted = True
        return self

    def _trajectory_check(self, score: float, threshold: float) -> bool:
        """Online decay-envelope check; returns ``True`` while stable.

        The documented Lyapunov schedule expects a disturbance to contract as
        ``eps * exp(-lambda t)`` (``OmniAvaEquation``; lambda is the system's
        ``LYAPUNOV.LAMBDA_CONVERGENCE``). Here the disturbance is the fused
        score's excess over the alarm cut: at an onset the excess sets
        ``eps``; afterwards each step's excess is compared against the
        envelope, and :data:`STABILITY_PATIENCE` consecutive samples above it
        once :data:`STABILITY_GRACE` steps have passed mean the trajectory is
        not contracting — the abstention flag the caller raises *before* any
        downstream gate. A score back at or under the cut resets the monitor.
        """
        excess = max(score - threshold, 0.0)
        if excess <= 0.0:
            self._excess_onset = None
            self._steps_since_onset = 0
            self._violation_streak = 0
            return True
        if self._excess_onset is None:
            self._excess_onset = excess
            self._steps_since_onset = 0
            self._violation_streak = 0
            return True
        self._steps_since_onset += 1
        envelope = self._excess_onset * float(np.exp(-self._lambda * self._steps_since_onset))
        if self._steps_since_onset >= STABILITY_GRACE and excess > envelope:
            self._violation_streak += 1
        else:
            self._violation_streak = 0
        return self._violation_streak < STABILITY_PATIENCE

    def update(self, value: float) -> dict[str, Any]:
        """Score one arriving sample from its trailing window only.

        Args:
            value: The newest sample.

        Returns:
            ``{"score", "is_anomaly", "threshold", "window_used",
            "trajectory_stable", "abstain"}`` for this point. The score is
            the fused score of the trailing window's last element — the
            arriving point — under the fitted instrument; ``abstain`` is the
            stability monitor's flag (see :meth:`_trajectory_check`).

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
        stable = self._trajectory_check(score, threshold)
        return {
            "score": score,
            "is_anomaly": bool(score > threshold),
            "threshold": threshold,
            "window_used": int(window.size),
            # Online Lyapunov-envelope check: False means the fused-score
            # trajectory has stopped contracting toward the cut, and the
            # caller should abstain or down-weight before any downstream
            # (ethical) gate consumes the verdict.
            "trajectory_stable": stable,
            "abstain": not stable,
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
        details = self.score_stream_detailed(values)
        return (
            np.asarray(details["scores"], dtype=np.float64),
            np.asarray(details["alarms"], dtype=bool),
        )

    def score_stream_detailed(
        self, values: npt.NDArray[np.float64]
    ) -> dict[str, npt.NDArray[np.float64] | npt.NDArray[np.bool_]]:
        """Like :meth:`score_stream`, also carrying the stability monitor.

        Args:
            values: Samples in arrival order.

        Returns:
            ``{"scores", "alarms", "abstain"}`` arrays, one entry per
            sample; ``abstain`` is ``True`` where the fused-score trajectory
            violated the decay envelope (:meth:`_trajectory_check`).
        """
        scores = np.empty(len(values), dtype=np.float64)
        alarms = np.empty(len(values), dtype=bool)
        abstain = np.empty(len(values), dtype=bool)
        for i, value in enumerate(np.asarray(values, dtype=np.float64).reshape(-1)):
            result = self.update(float(value))
            scores[i] = result["score"]
            alarms[i] = result["is_anomaly"]
            abstain[i] = result["abstain"]
        return {"scores": scores, "alarms": alarms, "abstain": abstain}
