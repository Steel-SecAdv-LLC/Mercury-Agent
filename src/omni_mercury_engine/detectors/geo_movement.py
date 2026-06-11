# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Geographic movement-plausibility detector for (lat, lon, time) trajectories.

Flags physically implausible or statistically extreme movement in a
geographic track — the trajectory-level complement to
:class:`omni_mercury_engine.detectors.spatial.SpatialAnomalyDetector`,
which scores point clouds in abstract euclidean feature space and has no
notion of time or of the latitude-dependent scale of longitude degrees.

Channels (each in [0, 1]):
    velocity  -- implied speed of the step vs. a feasibility ceiling
    jump      -- step length vs. the distribution of fitted step lengths
    time_gap  -- silence duration vs. the expected reporting cadence

Channel fusion is noisy-OR, ``1 - prod(1 - s_i)``: a single saturated
channel alone exceeds any threshold below 1.0, so an impossible speed
fires even when the other channels are quiet.  (A weighted sum with
weights w_i makes every score <= max(w_i) unreachable for any single
channel — a structural failure mode this detector deliberately avoids.)

The jump statistic is scored against the distribution of *historical*
step lengths from ``fit()``; measuring spread from the candidate point
itself would suppress exactly the signal being tested.

Dependencies: numpy (torch optional, only for fusion feature tensors).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.exceptions import DetectorException
from omni_mercury_engine.utils.geo import haversine_km

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    import torch

    TORCH_AVAILABLE = True
else:
    try:
        import torch

        TORCH_AVAILABLE = True
    except ImportError:
        torch = None  # type: ignore[assignment, unused-ignore]
        TORCH_AVAILABLE = False

_FEATURE_DIM = 8
_GAP_SATURATION_MULTIPLE = 4.0
_MIN_DT_HOURS = 1e-9


@dataclass(frozen=True)
class MovementAssessment:
    """Result of a single movement-plausibility evaluation.

    Attributes:
        score: Noisy-OR fused anomaly score in [0, 1].
        is_anomalous: Whether the score exceeded the detector threshold.
        channels: Per-channel scores (velocity, jump, time_gap).
        reason: Dominant channel name when flagged, else "normal" or
            "insufficient_history".
    """

    score: float
    is_anomalous: bool
    channels: dict[str, float] = field(default_factory=dict)
    reason: str = ""


class GeoMovementAnomalyDetector(BaseDetector):
    """Movement-plausibility detector over (lat, lon, epoch-seconds) trajectories.

    Data contract (``fit`` / ``detect`` / ``extract_features``): a float
    array of shape ``[n, 3]`` whose columns are latitude in decimal
    degrees, longitude in decimal degrees, and timestamp in epoch seconds,
    sorted by non-decreasing time.

    Config keys (beyond the BaseDetector keys):
        - "max_feasible_kmh": speed ceiling in km/h above which the
          velocity channel saturates (default 130.0, sustained ground
          travel).
        - "jump_sigma_saturation": z-score at which the jump channel
          saturates (default 6.0).
        - "expected_gap_hours": expected reporting cadence; the time_gap
          channel saturates at 4x this silence (default 24.0).
        - "threshold": anomaly threshold on the fused score (default 0.7).

    Example:
        >>> import numpy as np
        >>> track = np.array([[40.0, -105.0, 0.0], [40.01, -105.01, 3600.0],
        ...                   [40.02, -105.02, 7200.0]])
        >>> det = GeoMovementAnomalyDetector().fit(track)
        >>> result = det.detect(track)
        >>> bool(result["is_anomaly"])
        False
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the detector with movement-channel configuration.

        Args:
            config: Optional configuration dict; see class docstring for
                recognized keys.

        Raises:
            ValueError: If a channel parameter is not strictly positive or
                the threshold is outside [0, 1].
        """
        cfg = dict(config or {})
        cfg.setdefault("threshold", 0.7)
        super().__init__(cfg)

        self.max_feasible_kmh = float(self.config.get("max_feasible_kmh", 130.0))
        self.jump_sigma_saturation = float(self.config.get("jump_sigma_saturation", 6.0))
        self.expected_gap_hours = float(self.config.get("expected_gap_hours", 24.0))
        for label, value in (
            ("max_feasible_kmh", self.max_feasible_kmh),
            ("jump_sigma_saturation", self.jump_sigma_saturation),
            ("expected_gap_hours", self.expected_gap_hours),
        ):
            if value <= 0.0:
                raise ValueError(f"{label} must be positive, got {value}")

        self._step_mu: float = 0.0
        self._step_sd: float = 0.0
        self._n_fit_steps: int = 0
        self._median_dt_h: float = 0.0

    # ------------------------------------------------------------------
    # Trajectory validation and channel math
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_trajectory(
        data: np.ndarray[Any, Any] | torch.Tensor,
        min_points: int,
    ) -> np.ndarray[Any, Any]:
        """Coerce input to a float64 ``[n, 3]`` trajectory and validate it.

        Args:
            data: Array or tensor of shape ``[n, 3]`` (lat, lon, epoch s).
            min_points: Minimum number of rows required.

        Returns:
            Validated float64 array of shape ``[n, 3]``.

        Raises:
            DetectorException: On wrong shape, non-finite values,
                out-of-range coordinates, or decreasing timestamps.
        """
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise DetectorException(
                f"Trajectory must have shape [n, 3] = (lat, lon, epoch_s), got {arr.shape}"
            )
        if arr.shape[0] < min_points:
            raise DetectorException(
                f"Trajectory needs at least {min_points} points, got {arr.shape[0]}"
            )
        if not np.all(np.isfinite(arr)):
            raise DetectorException("Trajectory contains non-finite values")
        if np.any(np.abs(arr[:, 0]) > 90.0) or np.any(np.abs(arr[:, 1]) > 180.0):
            raise DetectorException("Latitude/longitude out of range (|lat|<=90, |lon|<=180)")
        if np.any(np.diff(arr[:, 2]) < 0.0):
            raise DetectorException("Trajectory timestamps must be non-decreasing")
        return arr

    @staticmethod
    def _step_lengths_km(arr: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Great-circle length in km of each consecutive trajectory step.

        Args:
            arr: Validated ``[n, 3]`` trajectory array.

        Returns:
            1-D float64 array of ``n - 1`` step lengths.
        """
        steps = [
            haversine_km(arr[i, 0], arr[i, 1], arr[i + 1, 0], arr[i + 1, 1])
            for i in range(arr.shape[0] - 1)
        ]
        return np.asarray(steps, dtype=np.float64)

    def _channels(self, step_km: float, dt_h: float) -> dict[str, float]:
        """Per-channel scores for a single movement step.

        Args:
            step_km: Step length in kilometres.
            dt_h: Elapsed time in hours (floored away from zero).

        Returns:
            Dict with "velocity", "jump", and "time_gap" scores in [0, 1].
        """
        dt_h = max(dt_h, _MIN_DT_HOURS)
        velocity = min(1.0, (step_km / dt_h) / self.max_feasible_kmh)
        jump = min(1.0, (abs(step_km - self._step_mu) / self._step_sd) / self.jump_sigma_saturation)
        time_gap = min(1.0, dt_h / (_GAP_SATURATION_MULTIPLE * self.expected_gap_hours))
        return {"velocity": velocity, "jump": jump, "time_gap": time_gap}

    @staticmethod
    def _noisy_or(channels: dict[str, float]) -> float:
        """Noisy-OR fusion of channel scores.

        Args:
            channels: Per-channel scores in [0, 1].

        Returns:
            Fused score ``1 - prod(1 - s_i)`` in [0, 1].
        """
        return 1.0 - math.prod(1.0 - s for s in channels.values())

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> GeoMovementAnomalyDetector:
        """Fit step-length statistics from a reference trajectory.

        Args:
            data: ``[n, 3]`` trajectory (lat, lon, epoch seconds), n >= 2.

        Returns:
            Self for method chaining.
        """
        arr = self._validate_trajectory(data, min_points=2)
        steps = self._step_lengths_km(arr)
        self._step_mu = float(np.mean(steps))
        # Floor the spread so near-constant histories (sd -> 0) cannot
        # inflate the jump z-score arbitrarily: at least 15% of the mean
        # step or 100 m, whichever is larger.
        self._step_sd = max(float(np.std(steps)), 0.15 * self._step_mu, 0.1)
        self._n_fit_steps = int(steps.shape[0])
        dt_h = np.diff(arr[:, 2]) / 3600.0
        self._median_dt_h = float(np.median(dt_h))
        self._is_fitted = True
        return self

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Score every movement step of a trajectory for plausibility.

        Args:
            data: ``[m, 3]`` trajectory (lat, lon, epoch seconds).  The
                first point has no incoming step and scores 0.

        Returns:
            Dict with "anomaly_score" (max step score), "is_anomaly",
            "scores" (per-point), "reason" (dominant channel of the worst
            step, or "normal"), "channels" (worst step), "threshold", and
            "metadata" (fit statistics and worst step index).

        Raises:
            DetectorException: If the detector has not been fitted.
        """
        if not self._is_fitted:
            raise DetectorException("Detector must be fitted before detection")
        arr = self._validate_trajectory(data, min_points=1)

        scores = np.zeros(arr.shape[0], dtype=np.float64)
        per_step_channels: list[dict[str, float]] = [
            {"velocity": 0.0, "jump": 0.0, "time_gap": 0.0}
        ]
        for i in range(1, arr.shape[0]):
            step_km = haversine_km(arr[i - 1, 0], arr[i - 1, 1], arr[i, 0], arr[i, 1])
            dt_h = (arr[i, 2] - arr[i - 1, 2]) / 3600.0
            channels = self._channels(step_km, dt_h)
            per_step_channels.append(channels)
            scores[i] = self._noisy_or(channels)

        effective_threshold = self.threshold
        calibration_diagnostics = None
        if self._auto_calibrate and arr.shape[0] > 1:
            effective_threshold = self.calibrate_threshold(scores)
            calibration_diagnostics = self._last_diagnostics

        worst = int(np.argmax(scores))
        anomaly_score = float(scores[worst])
        is_anomaly = anomaly_score > effective_threshold
        worst_channels = per_step_channels[worst]
        if arr.shape[0] == 1:
            reason = "insufficient_history"
        elif is_anomaly:
            reason = max(worst_channels, key=lambda k: worst_channels[k])
        else:
            reason = "normal"

        return {
            "anomaly_score": round(anomaly_score, 4),
            "is_anomaly": is_anomaly,
            "scores": scores.tolist(),
            "channels": worst_channels,
            "reason": reason,
            "detector_type": "geo_movement",
            "threshold": effective_threshold,
            "uncertainty": 1.0 / (1.0 + self._n_fit_steps),
            "calibration_diagnostics": calibration_diagnostics,
            "metadata": {
                "worst_step_index": worst,
                "fit_step_mean_km": self._step_mu,
                "fit_step_sd_km": self._step_sd,
                "fit_median_dt_h": self._median_dt_h,
                "n_fit_steps": self._n_fit_steps,
            },
        }

    def extract_features(
        self,
        data: np.ndarray[Any, Any] | torch.Tensor,
    ) -> np.ndarray[Any, Any] | torch.Tensor:
        """Extract per-point movement features for ML fusion.

        Args:
            data: ``[m, 3]`` trajectory (lat, lon, epoch seconds).

        Returns:
            ``[m, 8]`` feature tensor (torch when available, else numpy):
            velocity, jump, and time_gap channels, fused score, saturated
            step length and time delta, and normalized lat/lon.
        """
        arr = self._validate_trajectory(data, min_points=1)
        if not self._is_fitted and arr.shape[0] >= 2:
            self.fit(arr)
        features = np.zeros((arr.shape[0], _FEATURE_DIM), dtype=np.float64)
        features[:, 6] = arr[:, 0] / 90.0
        features[:, 7] = arr[:, 1] / 180.0
        if self._is_fitted:
            for i in range(1, arr.shape[0]):
                step_km = haversine_km(arr[i - 1, 0], arr[i - 1, 1], arr[i, 0], arr[i, 1])
                dt_h = (arr[i, 2] - arr[i - 1, 2]) / 3600.0
                channels = self._channels(step_km, dt_h)
                features[i, 0] = channels["velocity"]
                features[i, 1] = channels["jump"]
                features[i, 2] = channels["time_gap"]
                features[i, 3] = self._noisy_or(channels)
                features[i, 4] = min(1.0, step_km / 500.0)
                features[i, 5] = min(
                    1.0, dt_h / (_GAP_SATURATION_MULTIPLE * self.expected_gap_hours)
                )
        if TORCH_AVAILABLE:
            return torch.tensor(features, dtype=torch.float32)
        return features.astype(np.float32)

    # ------------------------------------------------------------------
    # Casework convenience API
    # ------------------------------------------------------------------

    def assess(
        self,
        history: Sequence[tuple[float, float, datetime]],
        current: tuple[float, float],
        now: datetime,
    ) -> MovementAssessment:
        """Assess one candidate observation against a (lat, lon, time) history.

        Convenience wrapper over ``fit`` + ``detect`` for casework callers
        holding datetime-stamped sightings: fits step statistics on the
        history and scores only the candidate step from the last historical
        point to ``current``.

        Args:
            history: Chronological sightings as (lat, lon, datetime) tuples.
            current: Candidate (lat, lon) observation.
            now: Timestamp of the candidate observation.

        Returns:
            MovementAssessment for the candidate step.
        """
        if len(history) < 2:
            return MovementAssessment(0.0, False, {}, "insufficient_history")

        track = np.asarray(
            [[lat, lon, ts.timestamp()] for lat, lon, ts in history],
            dtype=np.float64,
        )
        self.fit(track)
        last = history[-1]
        step_km = haversine_km(last[0], last[1], current[0], current[1])
        dt_h = (now - last[2]).total_seconds() / 3600.0
        channels = self._channels(step_km, dt_h)
        score = self._noisy_or(channels)
        flagged = score > self.threshold
        reason = max(channels, key=lambda k: channels[k]) if flagged else "normal"
        return MovementAssessment(round(score, 4), flagged, channels, reason)


__all__ = ["GeoMovementAnomalyDetector", "MovementAssessment"]
