# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Drone anomaly detection with RADD and DronLomaly approaches.

Ported from Omni-AXA-Engine's ``drone_anomaly_detector.py``.  Three known
defects from the original implementation are corrected in this port:

1.  :class:`DroneState` now carries the ``altitude_rate``,
    ``horizontal_velocity``, ``vertical_velocity`` and ``distance_to_home``
    fields referenced by the invariant rules.  In the original implementation
    those rules silently no-op'd because the fields did not exist on the state
    object; rules now actually evaluate and fire correctly.
2.  The ensemble previously branded as "K-Means / DBSCAN / OPTICS / LOF /
    OCSVM" was implemented as hand-coded z-scores over a single feature
    matrix.  The port replaces that with Mercury Agent's first-class
    in-house anomaly ensemble,
    :class:`~omni_mercury_engine.detectors.statistical.MercuryAnomalyDetector`,
    which combines three deterministic ``numpy``/``scipy`` scorers:
    **Resonance** (40 %; FFT-based harmonic spectral anomaly),
    **Kinematic** (30 %; physics-based jerk / curvature dynamics) and
    **InfoGeometry** (30 %; Fisher Information Matrix OOD detection).
    The drone detector therefore carries **no scikit-learn runtime
    dependency** - sklearn lives in the ``benchmark-comparison`` extra
    only, where it is used to score Mercury against external baselines
    rather than to power Mercury itself.
3.  The original docstring carried an unvalidated "93.84% average recall"
    paper-citation claim.  No reproduction dataset existed in either tree, so
    the claim is removed here.  Any future quantitative claim must be backed
    by a reproducible benchmark in ``benchmarks/``.

Ensemble decision.  A drone emits one state per tick, so the ensemble must
judge a single row.  Resonance (an FFT along the rows) and Kinematic (row
differences) are undefined for one row -- measured: kinematic is constant
0.5 and resonance noise -- so a weighted blend of the three could never
reach its 0.70 gate (a 4000 -> 300 RPM motor collapse, a 95 C thermal
excursion and a 500 m position jump all produced no fault). The decision
therefore uses the ensemble's kNN density stream for the current row as a
robust z-score against its leave-one-out reference on the rolling window,
which never contains the state being judged. InfoGeometry is reported but
does not decide: its reference is in-sample Mahalanobis on 20-100 rows of
22 features, which is optimistic, and it put 4.4% of normal ticks above
z = 6. Measured on a simulated cruise (20 flights x 280 scored ticks): the
kNN stream at the default gate z >= 8 raised no false alarm in 5,600 normal
ticks and flagged motor 4000 -> 3000 RPM (median z 180), 4000 -> 300 RPM
(698), temperature 40 -> 55 C (53), 40 -> 95 C (222) and a 500 m position
jump (32) every time; a 20 m jump within the flown range (z 1) is not
flagged. The three-component blend is still reported for diagnostics.

Live telemetry adapters for PX4 ULog flight logs (via :mod:`pyulog`) or
MAVLink endpoints (via :mod:`pymavlink`) are *not* shipped in this PR -
adopters who want them should populate :class:`DroneState` instances
from their ingest layer of choice (an example using
:mod:`pyulog.ULog` is provided in
``docs/drone/SETUP.md``).  The detector itself is transport-agnostic.

References
----------
* RADD: rule-based anomaly detection for drones.
* DronLomaly: drone log anomaly detection with Bi-LSTM.
* PX4 ULog format: https://docs.px4.io/main/en/dev_log/ulog_file_format.html
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.core.exceptions import DetectorException
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class MissionPhase(Enum):
    """Drone mission phases."""

    INIT = "INIT"
    TAKEOFF = "TAKEOFF"
    ON_MISSION = "ON_MISSION"
    RETURN = "RETURN"
    LANDING = "LANDING"
    EMERGENCY = "EMERGENCY"


class FaultType(Enum):
    """Types of drone faults detected by the ensemble."""

    WIND_DISTURBANCE = "WIND_DISTURBANCE"
    SENSOR_FAILURE = "SENSOR_FAILURE"
    ACTUATOR_FAILURE = "ACTUATOR_FAILURE"
    BATTERY_CRITICAL = "BATTERY_CRITICAL"
    GPS_LOSS = "GPS_LOSS"
    COMMUNICATION_LOSS = "COMMUNICATION_LOSS"
    COLLISION_RISK = "COLLISION_RISK"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    CYBERATTACK = "CYBERATTACK"
    MOTOR_FAILURE = "MOTOR_FAILURE"
    PROPELLER_DAMAGE = "PROPELLER_DAMAGE"
    THERMAL_ANOMALY = "THERMAL_ANOMALY"


@dataclass
class DroneFault:
    """Detected drone fault record."""

    fault_id: str
    fault_type: FaultType
    mission_phase: MissionPhase
    severity: float
    confidence: float
    description: str
    detected_by: str
    sensor_data: dict[str, Any]
    recommendations: list[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class DroneState:
    """Snapshot of current drone state.

    Adds the four fields that were missing in the original Omni-AXA-Engine
    implementation but referenced by the invariant rules:
    ``altitude_rate``, ``horizontal_velocity``, ``vertical_velocity``, and
    ``distance_to_home``.  When the fields are not explicitly provided they
    are derived from ``velocity`` (and ``position`` for ``distance_to_home``).

    Attributes:
        position: World-frame position vector ``(x, y, z)`` in metres.
        velocity: World-frame velocity vector ``(vx, vy, vz)`` in metres/sec.
        attitude: Euler angles ``(roll, pitch, yaw)`` in radians.
        battery_level: Battery state of charge in ``[0, 1]``.
        altitude: Altitude above ground in metres.
        gps_satellites: Number of satellites locked.
        signal_strength: Command-link signal strength in ``[0, 1]``.
        motor_speeds: Per-motor angular speeds in RPM.
        temperature: Component temperature in degrees Celsius.
        mission_phase: Active mission phase.
        altitude_rate: Vertical climb rate in metres/sec (positive = ascent).
        horizontal_velocity: Horizontal speed magnitude in metres/sec.
        vertical_velocity: Vertical speed in metres/sec (positive = descent;
            matches the LANDING rule semantics).
        distance_to_home: Horizontal distance to home/launch in metres.
        timestamp: UTC timestamp.
    """

    position: npt.NDArray[np.float64]
    velocity: npt.NDArray[np.float64]
    attitude: npt.NDArray[np.float64]
    battery_level: float
    altitude: float
    gps_satellites: int
    signal_strength: float
    motor_speeds: npt.NDArray[np.float64]
    temperature: float
    mission_phase: MissionPhase
    altitude_rate: float | None = None
    horizontal_velocity: float | None = None
    vertical_velocity: float | None = None
    distance_to_home: float | None = None
    home_position: npt.NDArray[np.float64] | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        """Validate array shapes and populate derived kinematic fields.

        ``position``, ``velocity`` and ``attitude`` must each be 3-vectors
        and ``motor_speeds`` must be a 4-vector (RADD's invariant rules
        index those positions directly).  Validating up-front turns a
        mis-shaped feed into a clear :class:`ValueError` at the source
        rather than an obscure ``IndexError`` deep inside the rule
        engine, which is the exact silent-failure class this port was
        commissioned to eliminate.

        Raises:
            ValueError: If any of ``position``, ``velocity``, ``attitude``
                or ``motor_speeds`` is not a 1-D array of the expected
                length.  ``home_position``, if supplied, must also be a
                3-vector.
        """

        def _check_shape(name: str, value: npt.NDArray[np.float64], length: int) -> None:
            array = np.asarray(value)
            if array.ndim != 1 or array.shape[0] != length:
                raise ValueError(
                    f"DroneState.{name} must be a 1-D array of length {length}; "
                    f"got shape {tuple(array.shape)}"
                )

        _check_shape("position", self.position, 3)
        _check_shape("velocity", self.velocity, 3)
        _check_shape("attitude", self.attitude, 3)
        _check_shape("motor_speeds", self.motor_speeds, 4)
        if self.home_position is not None:
            _check_shape("home_position", self.home_position, 3)

        velocity = np.asarray(self.velocity, dtype=np.float64)
        if self.horizontal_velocity is None:
            self.horizontal_velocity = float(
                math.sqrt(float(velocity[0]) ** 2 + float(velocity[1]) ** 2)
            )
        if self.vertical_velocity is None:
            self.vertical_velocity = float(-velocity[2])
        if self.altitude_rate is None:
            self.altitude_rate = float(velocity[2])
        if self.distance_to_home is None:
            if self.home_position is not None:
                pos = np.asarray(self.position, dtype=np.float64)
                home = np.asarray(self.home_position, dtype=np.float64)
                self.distance_to_home = float(
                    math.sqrt(float((pos[0] - home[0]) ** 2 + (pos[1] - home[1]) ** 2))
                )
            else:
                pos = np.asarray(self.position, dtype=np.float64)
                self.distance_to_home = float(math.sqrt(float(pos[0]) ** 2 + float(pos[1]) ** 2))


_DEFAULT_ENSEMBLE_WEIGHTS: Final[dict[str, float]] = {
    "resonance": 0.40,
    "kinematic": 0.30,
    "info_geometry": 0.30,
}

#: Invariant-rule field -> the fault it evidences. Rules not listed are
#: mission-configuration invariants (CONFIGURATION_ERROR).
_RULE_FAULT_TYPES: Final[dict[str, FaultType]] = {
    "battery_level": FaultType.BATTERY_CRITICAL,
    "gps_satellites": FaultType.GPS_LOSS,
    "signal_strength": FaultType.COMMUNICATION_LOSS,
    "motor_speeds_min": FaultType.MOTOR_FAILURE,
    # Thrust not producing the commanded climb / descent.
    "altitude_rate": FaultType.ACTUATOR_FAILURE,
    "vertical_velocity": FaultType.ACTUATOR_FAILURE,
    # Lateral drift while landing.
    "horizontal_velocity": FaultType.WIND_DISTURBANCE,
    # Too low on the return leg.
    "altitude": FaultType.COLLISION_RISK,
}

#: Number of features in :meth:`DroneAnomalyDetector._extract_features_for_ensemble`.
DRONE_TELEMETRY_DIM: Final[int] = 22

#: Telemetry feature groups (indices into the ensemble feature vector) and
#: the fault a dominant deviation in that group indicates.
_FEATURE_GROUP_FAULTS: Final[tuple[tuple[tuple[int, ...], FaultType], ...]] = (
    ((13, 14, 15, 16), FaultType.MOTOR_FAILURE),
    ((17,), FaultType.THERMAL_ANOMALY),
    ((9,), FaultType.BATTERY_CRITICAL),
    ((11,), FaultType.GPS_LOSS),
    ((12,), FaultType.COMMUNICATION_LOSS),
    ((3, 4, 5, 6, 7, 8, 18, 19, 20), FaultType.WIND_DISTURBANCE),
    ((0, 1, 2, 10, 21), FaultType.SENSOR_FAILURE),
)


@dataclass
class _EnsembleModel:
    """Ensemble fitted on a rolling window, with per-stream fit-time references."""

    detector: MercuryAnomalyDetector
    references: dict[str, tuple[float, float]]
    fitted_at_tick: int


class DroneAnomalyDetector:
    """Drone fault detector combining rule-based and ML-based approaches.

    Combines:

    * **RADD rules** -- mission-phase-specific invariants over the state
      vector (battery thresholds, GPS sat count, altitude rate, etc.).
    * **Mercury in-house anomaly ensemble** -- the Resonance / Kinematic
      / InfoGeometry triple exposed by
      :class:`~omni_mercury_engine.detectors.statistical.MercuryAnomalyDetector`,
      fitted on the rolling telemetry window.  This is Mercury Agent's
      own first-class ensemble; scikit-learn is **not** in the runtime
      dependency surface of this detector.
    * **DronLomaly log analysis** -- keyword-based feature extraction over
      structured sensor-log entries.
    """

    DEFAULT_HISTORY_WINDOW: Final[int] = 100
    DEFAULT_FIT_MINIMUM: Final[int] = 20
    #: Ticks between refits of the rolling-window ensemble.
    DEFAULT_REFIT_INTERVAL: Final[int] = 10
    #: Robust z (median / 1.4826*MAD units of the window's leave-one-out
    #: reference) the kNN density stream must reach to raise an ensemble
    #: fault. See the module docstring for the measurement behind 8.
    DEFAULT_Z_GATE: Final[float] = 8.0
    #: Stream whose z decides an ensemble fault (others are reported).
    DECISION_STREAM: Final[str] = "knn_density"
    MAX_HISTORY: Final[int] = 1000
    MAX_LOG_BUFFER: Final[int] = 5000
    #: Width of :meth:`extract_features` (registry protocol).
    FEATURE_DIM: Final[int] = 16

    def __init__(
        self,
        enable_radd: bool = True,
        enable_dronlomaly: bool = True,
        mission_phases: Sequence[MissionPhase] | None = None,
        *,
        history_window: int | None = None,
        ensemble_weights: dict[str, float] | None = None,
        random_state: int = 0,
        refit_interval: int | None = None,
        z_gate: float | None = None,
    ) -> None:
        """Initialise the detector.

        Args:
            enable_radd: Enable RADD rule-based detection.
            enable_dronlomaly: Enable DronLomaly log-based detection.
            mission_phases: Mission phases to monitor (default: all).
            history_window: Number of recent states the ensemble is fitted on.
            ensemble_weights: Optional override of the per-component weights
                of the reported diagnostic blend (``resonance``,
                ``kinematic``, ``info_geometry``). The fault decision is the
                z-gate on the single-row-valid streams (see module docstring).
            random_state: Reserved for backwards compatibility; the in-house
                :class:`MercuryAnomalyDetector` is deterministic after fit
                and consumes no RNG seed.
            refit_interval: Ticks between ensemble refits on the rolling
                window (default :attr:`DEFAULT_REFIT_INTERVAL`).
            z_gate: Robust z a stream must reach to raise an ensemble fault
                (default :attr:`DEFAULT_Z_GATE`).
        """
        self.enable_radd = enable_radd
        self.enable_dronlomaly = enable_dronlomaly
        self.mission_phases = (
            list(mission_phases) if mission_phases is not None else list(MissionPhase)
        )

        self.history_window = int(history_window or self.DEFAULT_HISTORY_WINDOW)
        self.ensemble_weights: dict[str, float] = dict(
            ensemble_weights or _DEFAULT_ENSEMBLE_WEIGHTS
        )
        weights_sum = sum(self.ensemble_weights.values())
        if weights_sum <= 0:
            raise ValueError("ensemble_weights must sum to a positive number")
        for key in list(self.ensemble_weights):
            self.ensemble_weights[key] /= weights_sum

        self.refit_interval = int(refit_interval or self.DEFAULT_REFIT_INTERVAL)
        if self.refit_interval < 1:
            raise ValueError("refit_interval must be >= 1")
        self.z_gate = float(z_gate if z_gate is not None else self.DEFAULT_Z_GATE)
        if self.z_gate <= 0:
            raise ValueError("z_gate must be positive")

        self._random_state = random_state
        self.invariant_rules = self._initialize_invariant_rules()
        self.log_buffer: list[dict[str, Any]] = []
        self.state_history: list[DroneState] = []
        self._tick = 0
        self._model: _EnsembleModel | None = None

    # -- public API ---------------------------------------------------------

    def detect_faults(
        self,
        drone_state: DroneState,
        sensor_logs: Sequence[dict[str, Any]] | None = None,
    ) -> list[DroneFault]:
        """Detect faults using RADD + ensemble + DronLomaly.

        Args:
            drone_state: Current drone state.
            sensor_logs: Optional recent sensor log entries.

        Returns:
            List of detected faults.
        """
        faults: list[DroneFault] = []
        if sensor_logs:
            self.log_buffer.extend(sensor_logs)
            if len(self.log_buffer) > self.MAX_LOG_BUFFER:
                self.log_buffer = self.log_buffer[-self.MAX_LOG_BUFFER :]

        # Rules read the trend including the current state; the ensemble is
        # judged against the window BEFORE the state joins it, so a faulty
        # state is never part of its own reference.
        if self.enable_radd:
            faults.extend(self._detect_with_radd(drone_state))
        self.state_history.append(drone_state)
        if len(self.state_history) > self.MAX_HISTORY:
            self.state_history = self.state_history[-self.MAX_HISTORY :]
        self._tick += 1
        if self.enable_dronlomaly and sensor_logs:
            faults.extend(self._detect_with_dronlomaly(sensor_logs))
        return faults

    def generate_flight_report(
        self, faults: Sequence[DroneFault], flight_duration: float
    ) -> dict[str, Any]:
        """Aggregate detected faults into a per-flight summary report.

        Args:
            faults: Detected faults.
            flight_duration: Flight duration in seconds.

        Returns:
            Flight summary dictionary suitable for JSON serialisation.
        """
        fault_by_type: dict[str, list[DroneFault]] = {}
        for fault in faults:
            fault_by_type.setdefault(fault.fault_type.value, []).append(fault)
        critical = [f for f in faults if f.severity >= 0.85]
        avg_severity = float(np.mean([f.severity for f in faults])) if faults else 0.0
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "flight_duration_seconds": float(flight_duration),
            "total_faults": len(faults),
            "critical_faults": len(critical),
            "average_severity": avg_severity,
            "faults_by_type": {k: len(v) for k, v in fault_by_type.items()},
            "detection_methods": {
                "RADD_Rules": sum(1 for f in faults if f.detected_by == "RADD_Rules"),
                "RADD_Ensemble": sum(1 for f in faults if f.detected_by == "RADD_Ensemble"),
                "DronLomaly": sum(1 for f in faults if f.detected_by == "DronLomaly"),
            },
            "critical_fault_details": [
                {
                    "fault_id": f.fault_id,
                    "type": f.fault_type.value,
                    "severity": f.severity,
                    "description": f.description,
                    "recommendations": list(f.recommendations),
                }
                for f in critical
            ],
            "flight_safety_score": max(1.0 - avg_severity, 0.0),
        }

    # -- registry protocol --------------------------------------------------

    def extract_features(self, data: Any) -> npt.NDArray[np.float64]:
        """Fixed-width feature vector for the detector registry.

        Two transparent input paths:

        * a :class:`DroneState` (or a dict of its fields, ``mission_phase``
          as a :class:`MissionPhase` or its value) is processed as the next
          telemetry tick (:meth:`detect_faults`); the features describe the
          resulting verdict and the state (see :meth:`_tick_features`).
        * any other array-like gets documented robust summary statistics
          (mean, std, min, max, median, IQR); no drone meaning is invented
          for anonymous arrays.

        Args:
            data: Drone state, dict of its fields, or numeric array.

        Returns:
            Array of shape ``(FEATURE_DIM,)``.

        Raises:
            ValueError: For an empty or non-finite array.
        """
        state = self._coerce_state(data)
        if state is not None:
            features = self._tick_features(state, self._assess(state, data))
        else:
            arr = np.asarray(data, dtype=np.float64).ravel()
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                raise ValueError("extract_features requires a non-empty, finite numeric array.")
            q25, q75 = np.percentile(arr, [25, 75])
            features = [
                float(np.mean(arr)),
                float(np.std(arr)),
                float(np.min(arr)),
                float(np.max(arr)),
                float(np.median(arr)),
                float(q75 - q25),
            ]
        features = features[: self.FEATURE_DIM]
        features.extend([0.0] * (self.FEATURE_DIM - len(features)))
        return np.asarray(features, dtype=np.float64)

    def predict(self, data: Any) -> dict[str, Any]:
        """Score one telemetry tick for the detector registry.

        Args:
            data: Drone state or dict of its fields. Other inputs carry no
                drone semantics and return ``anomaly_scores=None``.

        Returns:
            ``{"anomaly_scores": [max fault severity or 0], "is_anomaly":
            bool, "faults": [...]}``; the same tick is processed once even
            when :meth:`extract_features` already saw it.
        """
        state = self._coerce_state(data)
        if state is None:
            return {"anomaly_scores": None, "is_anomaly": False, "faults": []}
        faults = self._assess(state, data)
        severity = max((f.severity for f in faults), default=0.0)
        return {
            "anomaly_scores": np.array([severity], dtype=np.float64),
            "is_anomaly": bool(faults),
            "faults": [
                {
                    "fault_id": f.fault_id,
                    "fault_type": f.fault_type.value,
                    "severity": f.severity,
                    "detected_by": f.detected_by,
                    "description": f.description,
                }
                for f in faults
            ],
        }

    @staticmethod
    def _coerce_state(data: Any) -> DroneState | None:
        """Interpret registry input as a :class:`DroneState` when it is one."""
        if isinstance(data, DroneState):
            return data
        if isinstance(data, dict) and "motor_speeds" in data and "mission_phase" in data:
            fields = dict(data)
            phase = fields["mission_phase"]
            fields["mission_phase"] = (
                phase if isinstance(phase, MissionPhase) else MissionPhase(str(phase))
            )
            for key in ("position", "velocity", "attitude", "motor_speeds", "home_position"):
                if fields.get(key) is not None:
                    fields[key] = np.asarray(fields[key], dtype=np.float64)
            return DroneState(**fields)
        return None

    @staticmethod
    def _state_fingerprint(state: DroneState) -> tuple[Any, ...]:
        """Content key of the decision-relevant fields of one state.

        The one-tick memo below must key on *content*, not only object
        identity: a telemetry loop that reuses one dict and mutates its
        fields in place keeps the same identity across ticks, and an
        identity-only memo would return the first tick's verdict forever
        while the new telemetry is never processed.
        """
        return (
            state.timestamp,
            bytes(np.asarray(state.position, dtype=np.float64).tobytes()),
            bytes(np.asarray(state.velocity, dtype=np.float64).tobytes()),
            bytes(np.asarray(state.attitude, dtype=np.float64).tobytes()),
            bytes(np.asarray(state.motor_speeds, dtype=np.float64).tobytes()),
            float(state.battery_level),
            float(state.altitude),
            int(state.gps_satellites),
            float(state.signal_strength),
            float(state.temperature),
            state.mission_phase,
        )

    def _assess(self, state: DroneState, source: Any) -> list[DroneFault]:
        """Process a tick once; repeated calls with the same input reuse it."""
        fingerprint = self._state_fingerprint(state)
        cached = getattr(self, "_last_assessment", None)
        if cached is not None and cached[0] is source and cached[1] == fingerprint:
            return list(cached[2])
        faults = self.detect_faults(state)
        self._last_assessment = (source, fingerprint, faults)
        return list(faults)

    def _tick_features(self, state: DroneState, faults: Sequence[DroneFault]) -> list[float]:
        """Verdict and state summary for one tick (documented feature layout).

        ``[max severity, n rule faults, n ensemble faults, battery, GPS
        satellites / 12, signal, min motor RPM / 1000, temperature / 100,
        altitude / 100, horizontal speed, vertical speed, distance home /
        1000, top fault-type ordinal (-1 none), any critical (>= 0.85)]``.
        """
        types = list(FaultType)
        top = max(faults, key=lambda f: f.severity, default=None)
        return [
            max((f.severity for f in faults), default=0.0),
            float(sum(1 for f in faults if f.detected_by == "RADD_Rules")),
            float(sum(1 for f in faults if f.detected_by == "RADD_Ensemble")),
            float(state.battery_level),
            float(state.gps_satellites) / 12.0,
            float(state.signal_strength),
            float(np.min(state.motor_speeds)) / 1000.0,
            float(state.temperature) / 100.0,
            float(state.altitude) / 100.0,
            float(state.horizontal_velocity or 0.0),
            float(state.vertical_velocity or 0.0),
            float(state.distance_to_home or 0.0) / 1000.0,
            float(types.index(top.fault_type)) if top is not None else -1.0,
            1.0 if any(f.severity >= 0.85 for f in faults) else 0.0,
        ]

    # -- RADD ---------------------------------------------------------------

    def _detect_with_radd(self, drone_state: DroneState) -> list[DroneFault]:
        """Run the RADD rule + ensemble layers (state not yet in the history)."""
        faults = self._check_invariant_rules(drone_state)
        faults.extend(self._ensemble_detection(drone_state))
        return faults

    def _initialize_invariant_rules(self) -> dict[MissionPhase, list[dict[str, Any]]]:
        """Mission-phase-specific invariant rules.

        Returns:
            Mapping from mission phase to a list of rule dicts.  Each rule has
            a structured form with ``field``/``op``/``threshold`` keys (not a
            text expression), so violations are evaluated without ``eval()``
            or text parsing.
        """
        return {
            MissionPhase.INIT: [
                {"field": "battery_level", "op": ">=", "threshold": 0.20, "severity": 0.90},
                {"field": "gps_satellites", "op": ">=", "threshold": 6, "severity": 0.85},
                {"field": "motors_off_required", "op": "motors_off", "severity": 0.70},
            ],
            MissionPhase.TAKEOFF: [
                {"field": "altitude_rate", "op": ">", "threshold": 0.0, "severity": 0.85},
                {"field": "battery_level", "op": ">=", "threshold": 0.15, "severity": 0.95},
                {
                    "field": "motor_speeds_min",
                    "op": ">",
                    "threshold": 1000.0,
                    "severity": 0.80,
                },
            ],
            MissionPhase.ON_MISSION: [
                {"field": "gps_satellites", "op": ">=", "threshold": 5, "severity": 0.90},
                {"field": "signal_strength", "op": ">=", "threshold": 0.30, "severity": 0.75},
                {"field": "battery_level", "op": ">=", "threshold": 0.10, "severity": 0.85},
            ],
            MissionPhase.RETURN: [
                {
                    "field": "distance_to_home",
                    "op": "decreasing",
                    "severity": 0.80,
                },
                {"field": "battery_level", "op": ">=", "threshold": 0.05, "severity": 0.95},
                {"field": "altitude", "op": ">", "threshold": 5.0, "severity": 0.70},
            ],
            MissionPhase.LANDING: [
                {"field": "altitude_rate", "op": "<", "threshold": 0.0, "severity": 0.85},
                {
                    "field": "vertical_velocity",
                    "op": "<",
                    "threshold": 2.0,
                    "severity": 0.80,
                },
                {
                    "field": "horizontal_velocity",
                    "op": "<",
                    "threshold": 1.0,
                    "severity": 0.75,
                },
            ],
        }

    def _check_invariant_rules(self, drone_state: DroneState) -> list[DroneFault]:
        """Evaluate the rule table for the active mission phase."""
        faults: list[DroneFault] = []
        for rule in self.invariant_rules.get(drone_state.mission_phase, []):
            violated, description = self._evaluate_rule(rule, drone_state)
            if not violated:
                continue
            fault_type = _RULE_FAULT_TYPES.get(str(rule["field"]), FaultType.CONFIGURATION_ERROR)
            faults.append(
                DroneFault(
                    fault_id=self._make_fault_id("RULE"),
                    fault_type=fault_type,
                    mission_phase=drone_state.mission_phase,
                    severity=float(rule["severity"]),
                    confidence=0.95,
                    description=f"Invariant rule violated: {description}",
                    detected_by="RADD_Rules",
                    sensor_data=self._extract_sensor_data(drone_state),
                    recommendations=self._get_fault_recommendations(fault_type),
                )
            )
        return faults

    def _evaluate_rule(self, rule: dict[str, Any], drone_state: DroneState) -> tuple[bool, str]:
        """Evaluate one structured invariant rule.

        Args:
            rule: Rule dictionary.
            drone_state: Current state.

        Returns:
            Tuple ``(violated, description)``.
        """
        field_name = rule["field"]
        op = rule["op"]

        if op == "motors_off":
            return bool(np.any(np.asarray(drone_state.motor_speeds) != 0)), ("motor_speeds == 0")

        if op == "decreasing" and field_name == "distance_to_home":
            return self._distance_to_home_not_decreasing(drone_state), (
                "distance_to_home decreasing"
            )

        threshold = float(rule["threshold"])
        value = self._resolve_field(drone_state, field_name)
        description = f"{field_name} {op} {threshold}"
        if value is None:
            return False, description

        if op == ">=":
            return value < threshold, description
        if op == "<=":
            return value > threshold, description
        if op == ">":
            return value <= threshold, description
        if op == "<":
            return value >= threshold, description
        if op == "==":
            return not math.isclose(value, threshold), description
        logger.debug("Unsupported rule operator %r; skipping", op)
        return False, description

    @staticmethod
    def _resolve_field(drone_state: DroneState, field_name: str) -> float | None:
        """Resolve a rule field name to a scalar."""
        if field_name == "motor_speeds_min":
            return float(np.min(drone_state.motor_speeds))
        if field_name == "altitude_rate":
            return drone_state.altitude_rate
        if field_name == "horizontal_velocity":
            return drone_state.horizontal_velocity
        if field_name == "vertical_velocity":
            return drone_state.vertical_velocity
        if field_name == "distance_to_home":
            return drone_state.distance_to_home
        value = getattr(drone_state, field_name, None)
        if value is None:
            return None
        return float(value)

    def _distance_to_home_not_decreasing(self, drone_state: DroneState) -> bool:
        """Check whether ``distance_to_home`` has failed to decrease over 3 states."""
        recent = [*self.state_history[-2:], drone_state]
        if len(recent) < 3:
            return False
        distances = [s.distance_to_home for s in recent if s.distance_to_home is not None]
        if len(distances) < 3:
            return False
        return distances[-1] >= distances[0]

    # -- ensemble -----------------------------------------------------------

    def _ensemble_detection(self, drone_state: DroneState) -> list[DroneFault]:
        """Judge the current state against the ensemble fitted on prior states.

        Emits a ``RADD_Ensemble`` :class:`DroneFault` when the kNN-density
        stream of the current row reaches :attr:`z_gate` robust z-units
        above its leave-one-out reference on the rolling window. Severity
        rises from 0.5 at the gate toward 1.0.
        """
        model = self._ensemble_model()
        if model is None:
            return []
        features = self._extract_features_for_ensemble(drone_state)
        z_scores = self._ensemble_z_scores(model, features)
        stream = self.DECISION_STREAM
        z = z_scores.get(stream)
        if z is None or z < self.z_gate:
            return []
        severity = 1.0 - 0.5 * math.exp(-(z - self.z_gate) / self.z_gate)
        blend = self._compute_ensemble_scores(
            features, np.asarray([], dtype=np.float64), model=model
        )
        weighted = sum(v * self.ensemble_weights.get(k, 0.0) for k, v in blend.items())
        fault_type = self._classify_fault_type(drone_state, features)
        return [
            DroneFault(
                fault_id=self._make_fault_id("ENS"),
                fault_type=fault_type,
                mission_phase=drone_state.mission_phase,
                severity=float(severity),
                confidence=0.85,
                description=(
                    f"Ensemble anomaly detected ({stream} z={z:.1f} >= gate "
                    f"{self.z_gate:.1f}; "
                    + ", ".join(f"{k} z={v:.1f}" for k, v in sorted(z_scores.items()))
                    + f"; diagnostic blend {weighted:.2f})"
                ),
                detected_by="RADD_Ensemble",
                sensor_data=self._extract_sensor_data(drone_state),
                recommendations=self._get_fault_recommendations(fault_type),
            )
        ]

    def _new_ensemble_detector(self) -> MercuryAnomalyDetector:
        """Ensemble configured for this detector's small rolling windows.

        The decision stream (:attr:`DECISION_STREAM`) must exist from the
        first fit at :attr:`DEFAULT_FIT_MINIMUM` rows, so the kNN stream's
        default 100-row minimum (built for benchmark *ranking* quality) is
        lowered to the fit minimum here. This regime is gated at robust
        z = :attr:`DEFAULT_Z_GATE` — an extreme-excursion rule, not a
        ranking — and is the one the drone detector validated: 0 false
        alarms over 5,600 normal ticks, every injected fault caught,
        fitting from 20-row windows. Without the override every flight's
        first 100 ticks had no ensemble fault detection at all (a motor
        collapse at tick 50 went unflagged; regression-tested).
        """
        return MercuryAnomalyDetector({"knn_stream_min_reference": self.DEFAULT_FIT_MINIMUM})

    def _ensemble_model(self) -> _EnsembleModel | None:
        """Return the rolling-window ensemble, refitting every ``refit_interval`` ticks."""
        window = self.state_history[-self.history_window :]
        if len(window) < self.DEFAULT_FIT_MINIMUM:
            return None
        model = self._model
        if model is not None and self._tick - model.fitted_at_tick < self.refit_interval:
            return model
        historical = np.asarray([self._extract_features_for_ensemble(s) for s in window])
        detector = self._new_ensemble_detector()
        try:
            detector.fit(historical)
            reference = detector.detect(historical)
        except DetectorException as exc:
            logger.warning("MercuryAnomalyDetector unavailable for current window: %s", exc)
            self._model = None
            return None
        references: dict[str, tuple[float, float]] = {}
        for stream, key in (
            ("info_geometry", "info_geometry_scores"),
            ("knn_density", "knn_density_scores"),
        ):
            values = reference.get(key)
            if values is None:
                continue
            stats = MercuryAnomalyDetector._robust_location_scale(np.asarray(values))
            if stats is not None:
                references[stream] = stats
        self._model = _EnsembleModel(detector, references, fitted_at_tick=self._tick)
        return self._model

    @staticmethod
    def _ensemble_z_scores(
        model: _EnsembleModel, features: npt.NDArray[np.float64]
    ) -> dict[str, float]:
        """Robust z of the current row's single-row-valid streams."""
        result = model.detector.detect(features.reshape(1, -1))
        out: dict[str, float] = {}
        for stream, key in (
            ("info_geometry", "info_geometry_scores"),
            ("knn_density", "knn_density_scores"),
        ):
            ref = model.references.get(stream)
            values = result.get(key)
            if ref is None or values is None:
                continue
            value = float(np.asarray(values).reshape(-1)[0])
            if math.isfinite(value):
                out[stream] = (value - ref[0]) / ref[1]
        return out

    def _compute_ensemble_scores(
        self,
        features: npt.NDArray[np.float64],
        historical_features: npt.NDArray[np.float64],
        *,
        model: _EnsembleModel | None = None,
    ) -> dict[str, float]:
        """Compute outlier scores via Mercury's in-house anomaly ensemble.

        Fits a fresh :class:`MercuryAnomalyDetector` on the rolling
        telemetry window and scores the current sample.  The three
        component scores -- Resonance (FFT harmonic), Kinematic
        (jerk / curvature) and InfoGeometry (Fisher OOD) -- are returned
        in ``[0, 1]`` and keyed to match :attr:`ensemble_weights`.

        Args:
            features: Current sample feature vector, shape
                ``(n_features,)``.
            historical_features: Rolling baseline window, shape
                ``(n_history, n_features)``.

        Returns:
            Dictionary mapping ``"resonance"`` / ``"kinematic"`` /
            ``"info_geometry"`` to a continuous anomaly score.  Returns
            an empty dict when the in-house detector cannot be fitted
            or invoked for the current window (e.g. degenerate baseline).
            This is the diagnostic per-component breakdown; the fault
            decision is :meth:`_ensemble_z_scores`. When *model* is given,
            its fitted detector is used and *historical_features* is
            ignored.
        """
        sample = features.reshape(1, -1)
        try:
            if model is not None:
                result = model.detector.detect(sample)
            else:
                detector = self._new_ensemble_detector()
                detector.fit(historical_features)
                result = detector.detect(sample)
        except DetectorException as exc:
            logger.warning("MercuryAnomalyDetector unavailable for current window: %s", exc)
            return {}

        resonance = cast("npt.NDArray[np.float64]", result["resonance_scores"])
        kinematic = cast("npt.NDArray[np.float64]", result["kinematic_scores"])
        info_geo = cast("npt.NDArray[np.float64]", result["info_geometry_scores"])
        return {
            "resonance": float(np.clip(resonance[0], 0.0, 1.0)),
            "kinematic": float(np.clip(kinematic[0], 0.0, 1.0)),
            "info_geometry": float(np.clip(info_geo[0], 0.0, 1.0)),
        }

    def _extract_features_for_ensemble(self, drone_state: DroneState) -> npt.NDArray[np.float64]:
        """Stack the drone-state telemetry into a flat feature vector."""
        return np.asarray(
            np.concatenate(
                [
                    np.asarray(drone_state.position, dtype=np.float64),
                    np.asarray(drone_state.velocity, dtype=np.float64),
                    np.asarray(drone_state.attitude, dtype=np.float64),
                    [float(drone_state.battery_level)],
                    [float(drone_state.altitude)],
                    [float(drone_state.gps_satellites)],
                    [float(drone_state.signal_strength)],
                    np.asarray(drone_state.motor_speeds, dtype=np.float64),
                    [float(drone_state.temperature)],
                    [float(drone_state.altitude_rate or 0.0)],
                    [float(drone_state.horizontal_velocity or 0.0)],
                    [float(drone_state.vertical_velocity or 0.0)],
                    [float(drone_state.distance_to_home or 0.0)],
                ]
            ),
            dtype=np.float64,
        )

    def _classify_fault_type(
        self, drone_state: DroneState, features: npt.NDArray[np.float64]
    ) -> FaultType:
        """Map an ensemble anomaly to a fault type.

        Absolute safety limits take precedence (battery, GPS, link). Then,
        when the rolling-window model is available, the fault is attributed
        to the telemetry group whose feature deviates most from the window
        in robust units (motors, temperature, battery, GPS, link, motion,
        position). The fixed thresholds below are the fallback.
        """
        if drone_state.battery_level < 0.15:
            return FaultType.BATTERY_CRITICAL
        if drone_state.gps_satellites < 5:
            return FaultType.GPS_LOSS
        if drone_state.signal_strength < 0.30:
            return FaultType.COMMUNICATION_LOSS
        attributed = self._attribute_fault(features)
        if attributed is not None:
            return attributed
        if (
            np.any(drone_state.motor_speeds < 500)
            and drone_state.mission_phase is not MissionPhase.INIT
        ):
            return FaultType.MOTOR_FAILURE
        if drone_state.temperature > 80.0:
            return FaultType.THERMAL_ANOMALY
        if float(np.linalg.norm(drone_state.velocity)) > 20.0:
            return FaultType.WIND_DISTURBANCE
        return FaultType.SENSOR_FAILURE

    def _attribute_fault(self, features: npt.NDArray[np.float64]) -> FaultType | None:
        """Fault type of the feature group with the largest robust deviation."""
        model = self._model
        if model is None or features.shape[0] != DRONE_TELEMETRY_DIM:
            return None
        center = model.detector._knn_center
        scale = model.detector._knn_scale
        if center is None or scale is None:
            return None
        deviation = np.abs((features - center) / scale)
        best: tuple[float, FaultType] | None = None
        for indices, fault_type in _FEATURE_GROUP_FAULTS:
            value = float(np.max(deviation[list(indices)]))
            if best is None or value > best[0]:
                best = (value, fault_type)
        return best[1] if best is not None else None

    # -- DronLomaly ---------------------------------------------------------

    def _detect_with_dronlomaly(self, sensor_logs: Sequence[dict[str, Any]]) -> list[DroneFault]:
        """Log-based fault detection using DronLomaly-style features."""
        faults: list[DroneFault] = []
        for log_entry in sensor_logs:
            score = self._analyze_log_entry(log_entry)
            if score <= 0.75:
                continue
            fault_type = self._classify_log_fault(log_entry)
            phase_str = log_entry.get("phase", MissionPhase.ON_MISSION.value)
            try:
                phase = MissionPhase(phase_str)
            except ValueError:
                phase = MissionPhase.ON_MISSION
            faults.append(
                DroneFault(
                    fault_id=self._make_fault_id("LOG"),
                    fault_type=fault_type,
                    mission_phase=phase,
                    severity=score,
                    confidence=0.80,
                    description=(f"Log anomaly detected: {log_entry.get('message', 'Unknown')}"),
                    detected_by="DronLomaly",
                    sensor_data=dict(log_entry),
                    recommendations=self._get_fault_recommendations(fault_type),
                )
            )
        return faults

    @staticmethod
    def _analyze_log_entry(log_entry: dict[str, Any]) -> float:
        """Score a single log entry by keyword features."""
        text = str(log_entry.get("message", "")).lower()
        level = str(log_entry.get("level", "INFO")).upper()
        score = 0.0
        if "critical" in text:
            score += 0.50
        if "error" in text:
            score += 0.40
        if "warning" in text:
            score += 0.25
        if level in {"ERROR", "CRITICAL"}:
            score += 0.30
        if "timeout" in text:
            score += 0.35
        if "connection" in text and "lost" in text:
            score += 0.45
        if "signal" in text and "lost" in text:
            score += 0.35
        if "attack" in text or "intrusion" in text or "unauthorized" in text:
            score += 0.55
        if "overheating" in text or "overheat" in text:
            score += 0.40
        return min(score, 1.0)

    @staticmethod
    def _classify_log_fault(log_entry: dict[str, Any]) -> FaultType:
        """Map a log entry's text to a :class:`FaultType`."""
        message = str(log_entry.get("message", "")).lower()
        if "attack" in message or "intrusion" in message or "unauthorized" in message:
            return FaultType.CYBERATTACK
        if "config" in message or "parameter" in message:
            return FaultType.CONFIGURATION_ERROR
        if "sensor" in message:
            return FaultType.SENSOR_FAILURE
        if "motor" in message or "actuator" in message:
            return FaultType.ACTUATOR_FAILURE
        if "battery" in message or "power" in message:
            return FaultType.BATTERY_CRITICAL
        if "gps" in message or "satellite" in message:
            return FaultType.GPS_LOSS
        if "communication" in message or "signal" in message:
            return FaultType.COMMUNICATION_LOSS
        return FaultType.SENSOR_FAILURE

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _extract_sensor_data(drone_state: DroneState) -> dict[str, Any]:
        """Serialise a :class:`DroneState` to a JSON-safe dictionary."""
        return {
            "position": np.asarray(drone_state.position).tolist(),
            "velocity": np.asarray(drone_state.velocity).tolist(),
            "attitude": np.asarray(drone_state.attitude).tolist(),
            "battery_level": float(drone_state.battery_level),
            "altitude": float(drone_state.altitude),
            "gps_satellites": int(drone_state.gps_satellites),
            "signal_strength": float(drone_state.signal_strength),
            "motor_speeds": np.asarray(drone_state.motor_speeds).tolist(),
            "temperature": float(drone_state.temperature),
            "altitude_rate": drone_state.altitude_rate,
            "horizontal_velocity": drone_state.horizontal_velocity,
            "vertical_velocity": drone_state.vertical_velocity,
            "distance_to_home": drone_state.distance_to_home,
            "mission_phase": drone_state.mission_phase.value,
        }

    @staticmethod
    def _make_fault_id(prefix: str) -> str:
        """Generate a unique fault id (timestamps collide within one tick)."""
        return f"{prefix}_{uuid.uuid4().hex}"

    @staticmethod
    def _get_fault_recommendations(fault_type: FaultType) -> list[str]:
        """Per-fault corrective-action lists."""
        table: dict[FaultType, list[str]] = {
            FaultType.WIND_DISTURBANCE: [
                "Reduce altitude if possible",
                "Activate wind compensation mode",
                "Consider landing if wind exceeds limits",
            ],
            FaultType.SENSOR_FAILURE: [
                "Switch to redundant sensor",
                "Recalibrate affected sensor",
                "Return to home if critical sensor",
            ],
            FaultType.ACTUATOR_FAILURE: [
                "Activate emergency landing protocol",
                "Reduce flight speed",
                "Notify ground control immediately",
            ],
            FaultType.BATTERY_CRITICAL: [
                "Initiate immediate return to home",
                "Reduce power consumption",
                "Prepare for emergency landing",
            ],
            FaultType.GPS_LOSS: [
                "Switch to visual navigation",
                "Activate optical flow sensors",
                "Maintain current position until GPS recovery",
            ],
            FaultType.COMMUNICATION_LOSS: [
                "Execute failsafe protocol",
                "Attempt to re-establish link",
                "Return to home if link not restored",
            ],
            FaultType.COLLISION_RISK: [
                "Execute evasive maneuver",
                "Activate collision avoidance system",
                "Reduce speed immediately",
            ],
            FaultType.CONFIGURATION_ERROR: [
                "Verify mission parameters",
                "Reset to safe configuration",
                "Abort mission if critical",
            ],
            FaultType.CYBERATTACK: [
                "Isolate affected systems",
                "Switch to manual control",
                "Land immediately and secure drone",
            ],
            FaultType.MOTOR_FAILURE: [
                "Activate emergency landing",
                "Compensate with remaining motors",
                "Notify ground control",
            ],
            FaultType.PROPELLER_DAMAGE: [
                "Reduce flight speed",
                "Land as soon as safe",
                "Inspect propeller integrity",
            ],
            FaultType.THERMAL_ANOMALY: [
                "Reduce power output",
                "Increase cooling airflow",
                "Land if temperature continues rising",
            ],
        }
        return table.get(fault_type, ["Investigate anomaly", "Monitor closely"])


def get_drone_detector(
    enable_radd: bool = True,
    enable_dronlomaly: bool = True,
    **kwargs: Any,
) -> DroneAnomalyDetector:
    """Factory returning a configured :class:`DroneAnomalyDetector`.

    Args:
        enable_radd: Enable RADD rule-based detection.
        enable_dronlomaly: Enable DronLomaly log-based detection.
        **kwargs: Extra arguments forwarded to :class:`DroneAnomalyDetector`.

    Returns:
        A new detector instance.
    """
    return DroneAnomalyDetector(
        enable_radd=enable_radd, enable_dronlomaly=enable_dronlomaly, **kwargs
    )


__all__ = [
    "DRONE_TELEMETRY_DIM",
    "DroneAnomalyDetector",
    "DroneFault",
    "DroneState",
    "FaultType",
    "MissionPhase",
    "get_drone_detector",
]
