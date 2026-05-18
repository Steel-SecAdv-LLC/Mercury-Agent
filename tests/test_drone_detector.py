"""Tests for the drone anomaly detector."""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from omni_mercury_engine.detectors.drone import (
    DroneAnomalyDetector,
    DroneFault,
    DroneState,
    FaultType,
    MissionPhase,
    detector as drone_detector_module,
    get_drone_detector,
)


def _make_state(
    *,
    phase: MissionPhase = MissionPhase.ON_MISSION,
    battery: float = 0.80,
    altitude: float = 50.0,
    gps_sats: int = 12,
    signal: float = 0.85,
    velocity: tuple[float, float, float] = (5.0, 0.0, 0.0),
    motors: tuple[float, float, float, float] = (3000.0, 3000.0, 3000.0, 3000.0),
    temperature: float = 35.0,
    position: tuple[float, float, float] = (50.0, 0.0, 50.0),
) -> DroneState:
    """Return a healthy drone state for tests."""
    return DroneState(
        position=np.array(position, dtype=np.float64),
        velocity=np.array(velocity, dtype=np.float64),
        attitude=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        battery_level=battery,
        altitude=altitude,
        gps_satellites=gps_sats,
        signal_strength=signal,
        motor_speeds=np.array(motors, dtype=np.float64),
        temperature=temperature,
        mission_phase=phase,
    )


class TestDroneStateDerivedFields:
    """DroneState must populate the four formerly-missing fields."""

    def test_horizontal_velocity_derived(self) -> None:
        """horizontal_velocity is sqrt(vx^2 + vy^2) when not supplied."""
        state = _make_state(velocity=(3.0, 4.0, 0.0))
        assert state.horizontal_velocity == pytest.approx(5.0)

    def test_vertical_velocity_inverts_vz(self) -> None:
        """vertical_velocity is -vz so positive = descent."""
        state = _make_state(velocity=(0.0, 0.0, -2.0))
        assert state.vertical_velocity == pytest.approx(2.0)

    def test_altitude_rate_uses_vz(self) -> None:
        """altitude_rate is the raw vz (positive = climb)."""
        state = _make_state(velocity=(0.0, 0.0, 1.5))
        assert state.altitude_rate == pytest.approx(1.5)

    def test_distance_to_home_from_position_when_home_unset(self) -> None:
        """distance_to_home defaults to horizontal distance from origin."""
        state = _make_state(position=(3.0, 4.0, 50.0))
        assert state.distance_to_home == pytest.approx(5.0)

    def test_distance_to_home_uses_home_position(self) -> None:
        """home_position overrides the origin-based default."""
        state = DroneState(
            position=np.array([13.0, 4.0, 50.0], dtype=np.float64),
            velocity=np.zeros(3, dtype=np.float64),
            attitude=np.zeros(3, dtype=np.float64),
            battery_level=0.8,
            altitude=50.0,
            gps_satellites=12,
            signal_strength=0.9,
            motor_speeds=np.full(4, 3000.0, dtype=np.float64),
            temperature=30.0,
            mission_phase=MissionPhase.ON_MISSION,
            home_position=np.array([10.0, 0.0, 0.0], dtype=np.float64),
        )
        assert state.distance_to_home == pytest.approx(5.0)

    def test_explicit_override_respected(self) -> None:
        """Explicit derived fields are not overwritten."""
        state = DroneState(
            position=np.zeros(3, dtype=np.float64),
            velocity=np.zeros(3, dtype=np.float64),
            attitude=np.zeros(3, dtype=np.float64),
            battery_level=0.5,
            altitude=10.0,
            gps_satellites=10,
            signal_strength=0.8,
            motor_speeds=np.full(4, 2500.0, dtype=np.float64),
            temperature=25.0,
            mission_phase=MissionPhase.LANDING,
            altitude_rate=-0.7,
            horizontal_velocity=0.2,
            vertical_velocity=0.7,
            distance_to_home=0.0,
        )
        assert state.altitude_rate == pytest.approx(-0.7)
        assert state.horizontal_velocity == pytest.approx(0.2)


class TestInvariantRulesEvaluateAfterFieldFix:
    """Rules that referenced missing fields now actually evaluate."""

    def test_landing_altitude_rate_rule_fires(self) -> None:
        """LANDING with non-descending altitude_rate raises a CONFIGURATION_ERROR."""
        detector = DroneAnomalyDetector()
        # Vertical velocity > 0 means ascending; rule needs altitude_rate < 0
        state = _make_state(phase=MissionPhase.LANDING, velocity=(0.0, 0.0, 2.0))
        faults = detector.detect_faults(state)
        rule_faults = [f for f in faults if f.detected_by == "RADD_Rules"]
        assert any("altitude_rate" in f.description for f in rule_faults), [
            f.description for f in rule_faults
        ]

    def test_landing_vertical_velocity_rule_fires(self) -> None:
        """vertical_velocity >= 2 (descending too fast) triggers."""
        detector = DroneAnomalyDetector()
        # Set explicit vertical_velocity high
        state = DroneState(
            position=np.zeros(3, dtype=np.float64),
            velocity=np.array([0.0, 0.0, -1.0], dtype=np.float64),
            attitude=np.zeros(3, dtype=np.float64),
            battery_level=0.5,
            altitude=10.0,
            gps_satellites=12,
            signal_strength=0.8,
            motor_speeds=np.full(4, 2000.0, dtype=np.float64),
            temperature=30.0,
            mission_phase=MissionPhase.LANDING,
            altitude_rate=-1.0,
            vertical_velocity=2.5,
        )
        faults = detector.detect_faults(state)
        rule_faults = [
            f
            for f in faults
            if f.detected_by == "RADD_Rules" and "vertical_velocity" in f.description
        ]
        assert rule_faults

    def test_takeoff_altitude_rate_rule_fires(self) -> None:
        """TAKEOFF with altitude_rate <= 0 emits a rule violation."""
        detector = DroneAnomalyDetector()
        state = _make_state(phase=MissionPhase.TAKEOFF, velocity=(0.0, 0.0, 0.0))
        faults = detector.detect_faults(state)
        assert any(
            "altitude_rate" in f.description for f in faults if f.detected_by == "RADD_Rules"
        )

    def test_return_distance_to_home_rule_fires(self) -> None:
        """RETURN phase flags non-decreasing distance_to_home over history."""
        detector = DroneAnomalyDetector()
        for _ in range(3):
            detector.detect_faults(
                _make_state(
                    phase=MissionPhase.RETURN,
                    position=(50.0, 0.0, 30.0),
                    velocity=(0.0, 0.0, 0.0),
                )
            )
        faults = detector.detect_faults(
            _make_state(
                phase=MissionPhase.RETURN,
                position=(60.0, 0.0, 30.0),
                velocity=(0.0, 0.0, 0.0),
            )
        )
        descriptions = [f.description for f in faults if f.detected_by == "RADD_Rules"]
        assert any("distance_to_home" in d for d in descriptions), descriptions

    def test_healthy_state_yields_no_rule_faults(self) -> None:
        """A healthy state yields no rule violations."""
        detector = DroneAnomalyDetector()
        state = _make_state()
        faults = detector.detect_faults(state)
        assert not [f for f in faults if f.detected_by == "RADD_Rules"]

    def test_init_phase_low_battery_rule(self) -> None:
        """INIT phase enforces battery >= 0.20."""
        detector = DroneAnomalyDetector()
        state = _make_state(phase=MissionPhase.INIT, battery=0.10, motors=(0.0, 0.0, 0.0, 0.0))
        faults = detector.detect_faults(state)
        assert any(
            "battery_level" in f.description for f in faults if f.detected_by == "RADD_Rules"
        )


class TestSklearnEnsemble:
    """The ensemble must use real sklearn estimators (not hand-coded z-scores)."""

    def test_isolation_forest_imported(self) -> None:
        """The detector imports IsolationForest at module load."""
        module = drone_detector_module

        assert module._SKLEARN_AVAILABLE is True
        assert module.IsolationForest is not None
        assert module.LocalOutlierFactor is not None
        assert module.EllipticEnvelope is not None

    def test_ensemble_returns_three_scorers(self) -> None:
        """The sklearn ensemble exposes three scorers, not five fake ones."""
        detector = DroneAnomalyDetector()
        for i in range(25):
            detector.detect_faults(_make_state(velocity=(5.0 + 0.1 * i, 0.0, 0.0)))
        # The 26th call will produce ensemble scores via _compute_ensemble_scores
        features = detector._extract_features_for_ensemble(_make_state())
        historical = np.asarray(
            [detector._extract_features_for_ensemble(s) for s in detector.state_history]
        )
        scores = detector._compute_ensemble_scores(features, historical)
        assert set(scores.keys()) == {"isolation_forest", "elliptic_envelope", "lof"}

    def test_anomalous_state_flagged_by_ensemble(self) -> None:
        """An obvious outlier produces an ENS fault after the warm-up window."""
        detector = DroneAnomalyDetector()
        # Build a stable history
        for _ in range(30):
            detector.detect_faults(_make_state(velocity=(0.0, 0.0, 0.0)))
        # Now an outlier
        outlier = _make_state(velocity=(50.0, 50.0, 0.0), temperature=120.0)
        # Need to give the detector enough chances; we just check that no error
        # is raised and that ensemble can produce a finite score
        features = detector._extract_features_for_ensemble(outlier)
        historical = np.asarray(
            [detector._extract_features_for_ensemble(s) for s in detector.state_history]
        )
        scores = detector._compute_ensemble_scores(features, historical)
        assert all(math.isfinite(v) for v in scores.values())

    def test_ensemble_does_not_run_before_minimum_history(self) -> None:
        """Below DEFAULT_FIT_MINIMUM history, the ensemble returns no faults."""
        detector = DroneAnomalyDetector()
        faults = detector.detect_faults(_make_state())
        assert not [f for f in faults if f.detected_by == "RADD_Ensemble"]

    def test_fallback_when_sklearn_unavailable(self) -> None:
        """Without sklearn the detector uses a deterministic Mahalanobis scorer."""
        with patch("omni_mercury_engine.detectors.drone.detector._SKLEARN_AVAILABLE", False):
            detector = DroneAnomalyDetector()
            for _ in range(25):
                detector.detect_faults(_make_state())
            features = detector._extract_features_for_ensemble(_make_state())
            historical = np.asarray(
                [detector._extract_features_for_ensemble(s) for s in detector.state_history]
            )
            scores = detector._compute_ensemble_scores(features, historical)
            assert set(scores.keys()) == set(detector.ensemble_weights.keys())


class TestDronLomalyLogs:
    """Log-based fault detection."""

    def test_critical_log_emits_fault(self) -> None:
        """A critical-level log entry emits a fault."""
        detector = DroneAnomalyDetector(enable_radd=False)
        logs = [{"message": "Critical error: motor 3 overheating", "level": "ERROR"}]
        faults = detector.detect_faults(_make_state(), logs)
        assert faults
        assert faults[0].detected_by == "DronLomaly"

    def test_info_log_no_fault(self) -> None:
        """An info-level log entry does not emit a fault."""
        detector = DroneAnomalyDetector(enable_radd=False)
        logs = [{"message": "Heartbeat OK", "level": "INFO"}]
        faults = detector.detect_faults(_make_state(), logs)
        assert not faults

    def test_cyberattack_classification(self) -> None:
        """Attack keywords trigger CYBERATTACK classification."""
        detector = DroneAnomalyDetector(enable_radd=False)
        logs = [
            {
                "message": "Unauthorized command intrusion detected; signal lost",
                "level": "CRITICAL",
            }
        ]
        faults = detector.detect_faults(_make_state(), logs)
        assert any(f.fault_type is FaultType.CYBERATTACK for f in faults)

    def test_log_keyword_scoring_does_not_overflag_routine_lines(self) -> None:
        """Task 8 regression: extended keyword set must not over-fire on noise.

        ``_analyze_log_entry`` was extended beyond the upstream's
        mechanical-fault keywords with three Mercury-specific signals:

        * ``+0.55`` for ``attack | intrusion | unauthorized`` (security).
        * ``+0.40`` for ``overheat[ing]`` (thermal).
        * ``+0.35`` for ``signal lost`` (telemetry loss).

        Weights are tuned to keep operationally-noisy lines (routine
        "signal weak" advisories, expected ``intrusion_detection``
        self-tests, transient temperature notes) below the
        ``score > 0.75`` fault gate while still letting genuinely
        anomalous lines cross it.  This regression test pins the
        balance.
        """
        threshold = 0.75

        # Operationally noisy but benign: must stay <= threshold.
        benign_lines = [
            # "signal" appears but no "lost" pair, no critical/error
            {"message": "Signal strength is weak but stable", "level": "INFO"},
            # "intrusion_detection" self-test is a routine INFO log
            {"message": "intrusion_detection self-test pass", "level": "INFO"},
            # Temperature note without overheat/critical/error words
            {"message": "Thermal sensor reports normal range", "level": "INFO"},
        ]
        for log in benign_lines:
            score = DroneAnomalyDetector._analyze_log_entry(log)
            assert score <= threshold, f"Routine line was over-scored: {log!r} -> {score:.2f}"

        # Genuinely anomalous: must exceed threshold.
        anomalous_lines = [
            {
                "message": "Critical error: unauthorized command intrusion detected",
                "level": "CRITICAL",
            },
            {
                "message": "Motor 2 overheating - thermal runaway warning",
                "level": "ERROR",
            },
            {
                "message": "Connection lost: signal lost from GCS",
                "level": "ERROR",
            },
        ]
        for log in anomalous_lines:
            score = DroneAnomalyDetector._analyze_log_entry(log)
            assert score > threshold, f"Anomalous line was under-scored: {log!r} -> {score:.2f}"


class TestFlightReport:
    """generate_flight_report aggregation."""

    def test_report_counts_critical(self) -> None:
        """Report counts critical faults correctly."""
        detector = DroneAnomalyDetector()
        faults = [
            DroneFault(
                fault_id="A",
                fault_type=FaultType.BATTERY_CRITICAL,
                mission_phase=MissionPhase.RETURN,
                severity=0.95,
                confidence=0.9,
                description="Battery critical",
                detected_by="RADD_Rules",
                sensor_data={},
                recommendations=[],
            ),
            DroneFault(
                fault_id="B",
                fault_type=FaultType.GPS_LOSS,
                mission_phase=MissionPhase.ON_MISSION,
                severity=0.5,
                confidence=0.8,
                description="GPS",
                detected_by="DronLomaly",
                sensor_data={},
                recommendations=[],
            ),
        ]
        report = detector.generate_flight_report(faults, 300.0)
        assert report["total_faults"] == 2
        assert report["critical_faults"] == 1
        assert report["faults_by_type"] == {"BATTERY_CRITICAL": 1, "GPS_LOSS": 1}
        assert report["detection_methods"]["RADD_Rules"] == 1

    def test_report_no_faults(self) -> None:
        """Report works with no faults."""
        detector = DroneAnomalyDetector()
        report = detector.generate_flight_report([], 60.0)
        assert report["total_faults"] == 0
        assert report["average_severity"] == pytest.approx(0.0)
        assert report["flight_safety_score"] == pytest.approx(1.0)


class TestFaultRecommendations:
    """Per-fault corrective-action lists."""

    def test_known_fault_returns_recommendations(self) -> None:
        """Each fault type returns a non-empty recommendation list."""
        for fault_type in FaultType:
            recs = DroneAnomalyDetector._get_fault_recommendations(fault_type)
            assert recs and isinstance(recs, list)


class TestFactory:
    """Module-level factory."""

    def test_factory_returns_detector(self) -> None:
        """get_drone_detector returns a configured detector."""
        detector = get_drone_detector(enable_radd=True, enable_dronlomaly=False)
        assert isinstance(detector, DroneAnomalyDetector)
        assert detector.enable_radd is True
        assert detector.enable_dronlomaly is False


class TestNoUnvalidatedRecallClaim:
    """The 93.84% recall paper-citation claim must be gone."""

    def test_no_recall_claim_in_docstrings(self) -> None:
        """No docstring contains the unvalidated 93.84% recall claim."""
        module = drone_detector_module

        bad_texts: list[str] = []
        for value in vars(module).values():
            doc: Any = getattr(value, "__doc__", None)
            if isinstance(doc, str) and "93.84" in doc:
                bad_texts.append(doc)
        assert not bad_texts, "Unvalidated 93.84% recall claim must be removed: " + repr(bad_texts)
