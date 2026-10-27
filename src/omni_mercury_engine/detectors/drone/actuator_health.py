# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stuck-actuator health statistic - variance collapse against demonstrated activity.

A locked control surface or frozen servo channel is nearly invisible to
per-tick outlier scoring: the frozen value sits inside the channel's normal
range, and the autopilot's healthy surfaces compensate so the vehicle keeps
flying roughly on track. What the fault *does* change is the channel's
short-window variance - a servo that was dithering 20-80 PWM counts in
closed-loop control goes flat. This module scores exactly that signature:
the causal rolling standard deviation of each actuation channel, relative to
the activity the same channel has already demonstrated this flight.

Measured basis (ALFA, DOI 10.1184/R1/12707963 - 46 real fixed-wing flights
with mid-flight injected faults, ``datasets/uav_flight.py``): per-tick AUROC
of this statistic alone on servo channels is 0.907 for aileron locks, the
class where the per-flight Mercury ensemble scores 0.60; OR-fusing both
alarms (each thresholded at 0.5% FPR on the ten no-failure flights) detects
35/36 fault flights at 0.68% joint normal FPR with ~8 s median delay (the
stuck window length dominates the delay). See
``benchmarks/alfa_uav_benchmark.py`` for the reproducible protocol.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

#: Rolling window (ticks) over which variance collapse is judged. At the
#: ALFA grid rate of 4 Hz this is 8 s - long enough that steady cruise
#: never collapses a healthy closed-loop servo channel, short enough to
#: dominate detection delay at ~8 s.
DEFAULT_STUCK_WINDOW: int = 32

#: A channel participates only once its demonstrated activity (running
#: baseline std) exceeds this floor, in the channel's own units. Constant
#: channels (unused outputs, fixed gear) never participate.
DEFAULT_ACTIVE_FLOOR: float = 1.0

#: Alarm threshold on the stuck score. On the ALFA no-failure flights the
#: 99.5th percentile of the pooled statistic is 0.941, so 0.95 holds the
#: per-tick false-alarm rate under 0.5% there.
DEFAULT_STUCK_GATE: float = 0.95


def causal_rolling_std(values: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
    """Causal rolling standard deviation per channel.

    Args:
        values: ``(n_ticks, n_channels)`` array.
        window: Number of trailing ticks (including the current one) in
            each tick's window; early ticks use the partial prefix.

    Returns:
        ``(n_ticks, n_channels)`` population standard deviations.

    Raises:
        ValueError: ``window`` is not positive or ``values`` is not 2-D.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"values must be 2-D (ticks, channels); got shape {arr.shape}")
    n = len(arr)
    zeros = np.zeros((1, arr.shape[1]))
    cs = np.cumsum(np.vstack([zeros, arr]), axis=0)
    cs2 = np.cumsum(np.vstack([zeros, arr**2]), axis=0)
    lo = np.maximum(np.arange(n) - window + 1, 0)
    count = (np.arange(n) - lo + 1)[:, None].astype(np.float64)
    mean = (cs[np.arange(n) + 1] - cs[lo]) / count
    variance = np.maximum((cs2[np.arange(n) + 1] - cs2[lo]) / count - mean**2, 0.0)
    return np.asarray(np.sqrt(variance), dtype=np.float64)


def stuck_actuator_scores(
    values: npt.NDArray[np.float64],
    baseline_std: npt.NDArray[np.float64],
    window: int = DEFAULT_STUCK_WINDOW,
    active_floor: float = DEFAULT_ACTIVE_FLOOR,
) -> npt.NDArray[np.float64]:
    """Score variance collapse of actuation channels against a baseline.

    Args:
        values: ``(n_ticks, n_channels)`` actuation telemetry (servo PWM,
            motor RPM, ...).
        baseline_std: Per-channel activity reference, e.g. the std of a
            known-healthy window of the same flight, or the running maximum
            of the rolling std (demonstrated activity).
        window: Rolling window in ticks.
        active_floor: Channels whose baseline std is at or below this are
            ignored (never active, so never "stuck").

    Returns:
        ``(n_ticks,)`` scores in ``[0, 1]``: ``1 - min_c(rolling_std_c /
        baseline_std_c)`` over active channels, so 1.0 means some active
        channel's variance has fully collapsed. All-zero when no channel
        is active.

    Raises:
        ValueError: Shapes disagree, or inputs are malformed.
    """
    arr = np.asarray(values, dtype=np.float64)
    base = np.asarray(baseline_std, dtype=np.float64).ravel()
    if arr.ndim != 2 or len(base) != arr.shape[1]:
        raise ValueError(f"values {arr.shape} and baseline_std {base.shape} must agree on channels")
    active = base > float(active_floor)
    if not active.any():
        return np.zeros(len(arr), dtype=np.float64)
    ratio = causal_rolling_std(arr[:, active], window) / base[active]
    return np.asarray(1.0 - np.min(ratio, axis=1).clip(0.0, 1.0), dtype=np.float64)


class ActuatorHealthMonitor:
    """Streaming stuck-actuator monitor over per-tick actuation vectors.

    Keeps a bounded history of actuation vectors and a running *maximum*
    of each channel's rolling std (the activity the channel has
    demonstrated this flight). The running-max baseline is what makes the
    monitor streaming-safe: a later refit on a window that already
    contains the fault cannot drag the reference down and normalise the
    collapse away.

    The baseline deliberately never decays, so its scope is **one
    flight**: a monitor reused across flights would judge the new
    flight's actuation against the old flight's activity. Call
    :meth:`reset` between flights (one monitor instance per
    :class:`~omni_mercury_engine.detectors.drone.detector.DroneAnomalyDetector`
    per flight is the normal arrangement).
    """

    def __init__(
        self,
        n_channels: int,
        window: int = DEFAULT_STUCK_WINDOW,
        active_floor: float = DEFAULT_ACTIVE_FLOOR,
    ) -> None:
        """Initialize the instance.

        Raises:
            ValueError: Non-positive ``n_channels`` or ``window``.
        """
        if n_channels < 1:
            raise ValueError("n_channels must be >= 1")
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = int(window)
        self.active_floor = float(active_floor)
        self._history: list[npt.NDArray[np.float64]] = []
        self._baseline_std = np.zeros(n_channels, dtype=np.float64)

    def update(self, actuation: npt.NDArray[np.float64]) -> float:
        """Ingest one actuation vector and return the current stuck score.

        Args:
            actuation: ``(n_channels,)`` actuation values for this tick.

        Returns:
            The stuck score in ``[0, 1]`` for the current tick; 0.0 until
            a full window has been seen (no verdict on a partial window).

        Raises:
            ValueError: Wrong vector width.
        """
        vec = np.asarray(actuation, dtype=np.float64).ravel()
        if len(vec) != len(self._baseline_std):
            raise ValueError(
                f"actuation has {len(vec)} channels; monitor expects " f"{len(self._baseline_std)}"
            )
        self._history.append(vec)
        if len(self._history) > self.window:
            del self._history[: len(self._history) - self.window]
        if len(self._history) < self.window:
            return 0.0
        current_std = np.std(np.vstack(self._history), axis=0)
        self._baseline_std = np.maximum(self._baseline_std, current_std)
        active = self._baseline_std > self.active_floor
        if not active.any():
            return 0.0
        ratio = current_std[active] / self._baseline_std[active]
        return float(1.0 - float(np.min(ratio).clip(0.0, 1.0)))

    def reset(self) -> None:
        """Clear the history and baseline for a new flight.

        The demonstrated-activity reference is per-flight state; carrying
        it into the next flight would raise false stuck alarms whenever
        the new flight's servos are legitimately calmer than the old
        flight's.
        """
        self._history.clear()
        self._baseline_std = np.zeros_like(self._baseline_std)

    @property
    def baseline_std(self) -> npt.NDArray[np.float64]:
        """Per-channel demonstrated-activity reference (running max std)."""
        return self._baseline_std.copy()

    def get_state(self) -> dict[str, Any]:
        """Serializable monitor state (diagnostics)."""
        return {
            "window": self.window,
            "active_floor": self.active_floor,
            "n_ticks_held": len(self._history),
            "baseline_std": self._baseline_std.tolist(),
        }
