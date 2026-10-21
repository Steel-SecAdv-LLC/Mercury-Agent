# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the drone anomaly detector."""

from __future__ import annotations

import importlib
import math
from typing import Any

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

# The drone ensemble is wired to Mercury Agent's in-house
# :class:`MercuryAnomalyDetector` (Resonance / Kinematic /
# InfoGeometry) rather than scikit-learn.  These tests therefore
# require **no** optional dependency -- they run unconditionally and
# exercise the real ensemble path.


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


class TestDroneStateShapeValidation:
    """``DroneState.__post_init__`` rejects malformed input vectors.

    The RADD invariant rules index ``position[0..2]``, ``velocity[0..2]``,
    ``attitude[0..2]`` and ``motor_speeds[0..3]`` directly.  A mis-shaped
    feed previously bubbled up as an obscure ``IndexError`` deep inside
    the rule loop; the port now raises a clear :class:`ValueError` at
    state construction so the failure surfaces at the source.
    """

    def _kwargs(self) -> dict[str, Any]:
        return {
            "position": np.zeros(3, dtype=np.float64),
            "velocity": np.zeros(3, dtype=np.float64),
            "attitude": np.zeros(3, dtype=np.float64),
            "battery_level": 0.5,
            "altitude": 10.0,
            "gps_satellites": 10,
            "signal_strength": 0.8,
            "motor_speeds": np.full(4, 2500.0, dtype=np.float64),
            "temperature": 25.0,
            "mission_phase": MissionPhase.ON_MISSION,
        }

    def test_position_wrong_length_raises(self) -> None:
        kw = self._kwargs()
        kw["position"] = np.zeros(2, dtype=np.float64)
        with pytest.raises(ValueError, match="position"):
            DroneState(**kw)

    def test_velocity_wrong_length_raises(self) -> None:
        kw = self._kwargs()
        kw["velocity"] = np.zeros(4, dtype=np.float64)
        with pytest.raises(ValueError, match="velocity"):
            DroneState(**kw)

    def test_attitude_wrong_dim_raises(self) -> None:
        kw = self._kwargs()
        # 2-D array is invalid even if total element count matches.
        kw["attitude"] = np.zeros((1, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="attitude"):
            DroneState(**kw)

    def test_motor_speeds_wrong_length_raises(self) -> None:
        kw = self._kwargs()
        kw["motor_speeds"] = np.full(3, 2500.0, dtype=np.float64)
        with pytest.raises(ValueError, match="motor_speeds"):
            DroneState(**kw)

    def test_home_position_wrong_length_raises(self) -> None:
        kw = self._kwargs()
        kw["home_position"] = np.zeros(2, dtype=np.float64)
        with pytest.raises(ValueError, match="home_position"):
            DroneState(**kw)


class TestInvariantRulesEvaluateAfterFieldFix:
    """Rules that referenced missing fields now actually evaluate."""

    def test_landing_altitude_rate_rule_fires(self) -> None:
        """LANDING with non-descending altitude_rate raises an ACTUATOR_FAILURE."""
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


class TestMercuryEnsemble:
    """The ensemble must use Mercury's in-house anomaly detector.

    The drone detector previously delegated ensemble scoring to
    scikit-learn (``IsolationForest`` / ``EllipticEnvelope`` /
    ``LocalOutlierFactor``).  Per the architectural correction in
    ``CHANGELOG.md`` (PR #224, "MercuryAnomalyDetector adoption"),
    scoring is now done by Mercury Agent's first-class
    :class:`~omni_mercury_engine.detectors.statistical.MercuryAnomalyDetector`,
    which combines three deterministic ``numpy``/``scipy`` scorers --
    **Resonance**, **Kinematic**, and **InfoGeometry**.  scikit-learn
    is no longer in the runtime dependency surface; it lives only in
    the ``benchmark-comparison`` extra.
    """

    def test_no_sklearn_runtime_import(self) -> None:
        """The drone detector module does not import sklearn at load."""
        module = drone_detector_module

        # The in-house ensemble is the only ensemble path; no sklearn
        # estimators may be referenced as module-level symbols.
        assert not hasattr(module, "IsolationForest")
        assert not hasattr(module, "EllipticEnvelope")
        assert not hasattr(module, "LocalOutlierFactor")
        assert not hasattr(module, "_SKLEARN_AVAILABLE")
        assert not hasattr(module, "_compute_sklearn_scores")
        assert not hasattr(module, "_compute_fallback_scores")

    def test_ensemble_returns_mercury_components(self) -> None:
        """The in-house ensemble exposes the three Mercury components."""
        detector = DroneAnomalyDetector()
        for i in range(25):
            detector.detect_faults(_make_state(velocity=(5.0 + 0.1 * i, 0.0, 0.0)))
        features = detector._extract_features_for_ensemble(_make_state())
        historical = np.asarray(
            [detector._extract_features_for_ensemble(s) for s in detector.state_history]
        )
        scores = detector._compute_ensemble_scores(features, historical)
        assert set(scores.keys()) == {"resonance", "kinematic", "info_geometry"}
        for component, value in scores.items():
            assert 0.0 <= value <= 1.0, f"{component}={value!r} outside [0, 1]"

    def test_anomalous_state_flagged_by_ensemble(self) -> None:
        """An obvious outlier produces finite Mercury scores after warm-up."""
        detector = DroneAnomalyDetector()
        # Build a stable history.
        for _ in range(30):
            detector.detect_faults(_make_state(velocity=(0.0, 0.0, 0.0)))
        # Now an outlier; we just check that the in-house ensemble produces
        # finite scores rather than raising.
        outlier = _make_state(velocity=(50.0, 50.0, 0.0), temperature=120.0)
        features = detector._extract_features_for_ensemble(outlier)
        historical = np.asarray(
            [detector._extract_features_for_ensemble(s) for s in detector.state_history]
        )
        scores = detector._compute_ensemble_scores(features, historical)
        assert scores, "Mercury ensemble unexpectedly returned no components"
        assert all(math.isfinite(v) for v in scores.values())

    def test_ensemble_does_not_run_before_minimum_history(self) -> None:
        """Below DEFAULT_FIT_MINIMUM history, the ensemble returns no faults."""
        detector = DroneAnomalyDetector()
        faults = detector.detect_faults(_make_state())
        assert not [f for f in faults if f.detected_by == "RADD_Ensemble"]

    def test_default_weights_match_mercury_ratio(self) -> None:
        """Default weights mirror the 40/30/30 Resonance/Kinematic/InfoGeo ratio."""
        detector = DroneAnomalyDetector()
        assert detector.ensemble_weights["resonance"] == pytest.approx(0.40)
        assert detector.ensemble_weights["kinematic"] == pytest.approx(0.30)
        assert detector.ensemble_weights["info_geometry"] == pytest.approx(0.30)

    def test_degenerate_history_returns_empty_scores(self) -> None:
        """A zero-variance baseline yields an empty score dict, not a crash.

        ``MercuryAnomalyDetector.fit`` raises ``DetectorException`` when
        the training data contains only NaN/Inf, but a constant feed
        passes fit and exercises the surrounding numerical safeguards.
        Either way, the ensemble must surface a clean empty mapping
        rather than propagating a NumPy warning or raising.
        """
        detector = DroneAnomalyDetector()
        # All-zero rows -- well-defined statistics but no variance.
        # ``_extract_features_for_ensemble`` returns a 1-D ``NDArray[float64]``;
        # ``len(...)`` is unambiguously typed as ``int`` and avoids the
        # numpy-stub-driven ``tuple[int, ...]`` indexing error that mypy
        # ``--strict`` flags on Python 3.11.
        n_features = len(detector._extract_features_for_ensemble(_make_state()))
        features = np.zeros(n_features, dtype=np.float64)
        historical = np.zeros((25, n_features), dtype=np.float64)
        scores = detector._compute_ensemble_scores(features, historical)
        # The in-house detector should either return three finite component
        # scores or an empty dict; both outcomes are well-formed.
        if scores:
            assert set(scores.keys()) == {"resonance", "kinematic", "info_geometry"}
            assert all(math.isfinite(v) for v in scores.values())


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


def _cruise_state(
    rng: np.random.Generator,
    tick: int,
    *,
    motor: float = 4000.0,
    temperature: float = 40.0,
    position_jump: float = 0.0,
) -> DroneState:
    """Noisy straight-line cruise at 1 m/tick (a realistic trending feed)."""
    return DroneState(
        position=np.array([tick + position_jump, 0.0, 50.0]) + rng.normal(0, 0.1, 3),
        velocity=np.array([1.0, 0.0, 0.0]) + rng.normal(0, 0.05, 3),
        attitude=rng.normal(0, 0.01, 3),
        battery_level=0.9 - tick * 0.001,
        altitude=50.0 + float(rng.normal(0, 0.1)),
        gps_satellites=12,
        signal_strength=0.9,
        motor_speeds=np.full(4, motor) + rng.normal(0, 20, 4),
        temperature=temperature + float(rng.normal(0, 0.5)),
        mission_phase=MissionPhase.ON_MISSION,
    )


def _flown_detector(
    seed: int, ticks: int = 120
) -> tuple[DroneAnomalyDetector, np.random.Generator]:
    rng = np.random.default_rng(seed)
    detector = DroneAnomalyDetector()
    for tick in range(ticks):
        faults = detector.detect_faults(_cruise_state(rng, tick))
        assert not [f for f in faults if f.detected_by == "RADD_Ensemble"], tick
    return detector, rng


class TestSingleRowEnsembleDecision:
    """The ensemble judges one tick against a window that excludes it."""

    @pytest.mark.parametrize(
        ("kwargs", "fault_type"),
        [
            ({"motor": 300.0}, FaultType.MOTOR_FAILURE),
            ({"motor": 3000.0}, FaultType.MOTOR_FAILURE),
            ({"temperature": 95.0}, FaultType.THERMAL_ANOMALY),
            ({"position_jump": 500.0}, FaultType.SENSOR_FAILURE),
        ],
    )
    def test_injected_fault_is_flagged(
        self, kwargs: dict[str, float], fault_type: FaultType
    ) -> None:
        detector, rng = _flown_detector(seed=1)
        faults = detector.detect_faults(_cruise_state(rng, 120, **kwargs))
        ensemble = [f for f in faults if f.detected_by == "RADD_Ensemble"]
        assert len(ensemble) == 1
        assert ensemble[0].fault_type is fault_type
        assert 0.5 <= ensemble[0].severity < 1.0 or ensemble[0].severity == pytest.approx(1.0)
        assert "knn_density z=" in ensemble[0].description

    def test_judged_state_is_not_in_its_own_reference(self) -> None:
        detector, rng = _flown_detector(seed=2, ticks=40)
        faulty = _cruise_state(rng, 40, motor=300.0)
        row = detector._extract_features_for_ensemble(faulty)
        detector.detect_faults(faulty)
        model = detector._model
        assert model is not None
        train = model.detector._train_data
        assert train is not None
        assert not np.any(np.all(np.isclose(train, row), axis=1))
        assert detector.state_history[-1] is faulty

    def test_refits_every_interval(self) -> None:
        detector, rng = _flown_detector(seed=3, ticks=30)
        model = detector._model
        assert model is not None
        for tick in range(30, 30 + detector.refit_interval - 1):
            detector.detect_faults(_cruise_state(rng, tick))
            if detector._tick - model.fitted_at_tick < detector.refit_interval:
                assert detector._model is model
        for tick in range(40, 40 + detector.refit_interval):
            detector.detect_faults(_cruise_state(rng, tick))
        assert detector._model is not model

    def test_invalid_gate_and_interval_rejected(self) -> None:
        with pytest.raises(ValueError, match="z_gate"):
            DroneAnomalyDetector(z_gate=0.0)
        with pytest.raises(ValueError, match="refit_interval"):
            DroneAnomalyDetector(refit_interval=-1)


class TestRuleFaultTyping:
    """Rule violations carry the fault their field evidences."""

    def test_init_battery_and_gps_types(self) -> None:
        detector = DroneAnomalyDetector()
        state = _make_state(
            phase=MissionPhase.INIT, battery=0.05, gps_sats=2, motors=(0.0, 0.0, 0.0, 0.0)
        )
        types = {f.fault_type for f in detector.detect_faults(state)}
        assert FaultType.BATTERY_CRITICAL in types
        assert FaultType.GPS_LOSS in types

    def test_signal_and_motor_types(self) -> None:
        detector = DroneAnomalyDetector()
        on_mission = detector.detect_faults(_make_state(signal=0.1))
        assert FaultType.COMMUNICATION_LOSS in {f.fault_type for f in on_mission}
        takeoff = detector.detect_faults(
            _make_state(
                phase=MissionPhase.TAKEOFF,
                velocity=(0.0, 0.0, 1.0),
                motors=(3000.0, 3000.0, 3000.0, 200.0),
            )
        )
        assert FaultType.MOTOR_FAILURE in {f.fault_type for f in takeoff}

    def test_motors_spinning_at_init_is_configuration_error(self) -> None:
        detector = DroneAnomalyDetector()
        faults = detector.detect_faults(_make_state(phase=MissionPhase.INIT))
        assert {f.fault_type for f in faults} == {FaultType.CONFIGURATION_ERROR}

    def test_fault_ids_are_unique(self) -> None:
        detector = DroneAnomalyDetector()
        state = _make_state(
            phase=MissionPhase.INIT, battery=0.05, gps_sats=2, motors=(1.0, 1.0, 1.0, 1.0)
        )
        ids = [f.fault_id for f in detector.detect_faults(state)]
        assert len(ids) == 3
        assert len(set(ids)) == 3


class TestRegistryProtocol:
    """The detector is discoverable and speaks the registry protocol."""

    def test_manifest_entry_resolves(self) -> None:
        from omni_mercury_engine.core.detector_registry import (
            DETECTOR_MANIFEST,
            DetectorCategory,
        )

        entry = next(e for e in DETECTOR_MANIFEST if e.name == "drone")
        assert entry.category is DetectorCategory.UAV
        module = importlib.import_module(entry.module_path)
        assert getattr(module, entry.class_name) is DroneAnomalyDetector
        assert entry.feature_dim == DroneAnomalyDetector.FEATURE_DIM

    def test_state_tick_is_processed_once(self) -> None:
        detector = DroneAnomalyDetector()
        state = _make_state(signal=0.1)
        features = detector.extract_features(state)
        result = detector.predict(state)
        assert features.shape == (DroneAnomalyDetector.FEATURE_DIM,)
        assert len(detector.state_history) == 1
        assert result["is_anomaly"] is True
        assert result["anomaly_scores"][0] == pytest.approx(features[0])
        assert any(f["fault_type"] == "COMMUNICATION_LOSS" for f in result["faults"])

    def test_dict_state_is_accepted(self) -> None:
        detector = DroneAnomalyDetector()
        state = _make_state()
        payload = {
            "position": state.position.tolist(),
            "velocity": state.velocity.tolist(),
            "attitude": state.attitude.tolist(),
            "battery_level": state.battery_level,
            "altitude": state.altitude,
            "gps_satellites": state.gps_satellites,
            "signal_strength": state.signal_strength,
            "motor_speeds": state.motor_speeds.tolist(),
            "temperature": state.temperature,
            "mission_phase": "ON_MISSION",
        }
        assert detector.predict(payload)["anomaly_scores"][0] == 0.0

    def test_anonymous_array_gets_summary_statistics_only(self) -> None:
        detector = DroneAnomalyDetector()
        data = np.arange(12.0).reshape(3, 4)
        features = detector.extract_features(data)
        assert features[:6] == pytest.approx([5.5, np.std(data), 0.0, 11.0, 5.5, 5.5])
        assert detector.predict(data)["anomaly_scores"] is None
        assert detector.state_history == []

    def test_registry_extracts_drone_features(self) -> None:
        from omni_mercury_engine.core.detector_registry import DetectorRegistry

        registry = DetectorRegistry()
        registry.register("drone", DroneAnomalyDetector(), feature_dim=16)
        result = registry.extract_features("drone", np.ones((4, 22)))
        assert result.success
        assert np.asarray(result.features).shape == (16,)


class TestEnsembleLiveFromFitMinimum:
    """The decision stream must be live from the first ensemble fit.

    Regression: the kNN density stream's default minimum (100 reference
    rows, built for benchmark ranking quality) silently disabled the
    drone's only ensemble decision stream for every window smaller than
    100 states, although the ensemble fits from ``DEFAULT_FIT_MINIMUM``
    (20). A total motor collapse plus thermal excursion at tick 50
    produced zero faults; the identical fault at tick 120 was flagged.
    The drone now constructs its ensemble with
    ``knn_stream_min_reference = DEFAULT_FIT_MINIMUM`` (the regime its
    own validation measured: z-gate 8.0, 0 false alarms / 5,600 ticks).
    """

    @staticmethod
    def _fly(detector: DroneAnomalyDetector, tick: int, *, faulty: bool) -> list[DroneFault]:
        rng = np.random.default_rng(tick)

        def rpm() -> float:
            return float(300 + rng.normal(0, 10)) if faulty else float(4000 + rng.normal(0, 30))

        state = _make_state(
            motors=(rpm(), rpm(), rpm(), rpm()),
            temperature=95.0 if faulty else float(40 + rng.normal(0, 0.5)),
        )
        return detector.detect_faults(state)

    def test_motor_collapse_at_tick_50_is_flagged(self) -> None:
        detector = DroneAnomalyDetector()
        for tick in range(50):
            assert not self._fly(detector, tick, faulty=False)
        assert self._fly(
            detector, 50, faulty=True
        ), "ensemble decision stream dead inside the first history window"

    def test_ensemble_config_lowers_knn_minimum(self) -> None:
        detector = DroneAnomalyDetector()
        ensemble = detector._new_ensemble_detector()
        assert ensemble._knn_min_reference == DroneAnomalyDetector.DEFAULT_FIT_MINIMUM


class TestAssessMemoKeysOnContent:
    """One mutated-in-place payload dict must not pin the first verdict.

    Regression: ``_assess`` memoized on input object identity alone, so a
    telemetry loop reusing one dict got the first tick's faults forever and
    no later telemetry was processed.
    """

    def test_mutated_payload_is_reprocessed(self) -> None:
        detector = DroneAnomalyDetector()
        healthy = _make_state()
        payload = {"shared": "payload-object"}

        first = detector._assess(healthy, payload)
        assert detector._assess(healthy, payload) == first  # memo hit

        failing = _make_state(motors=(0.0, 0.0, 0.0, 0.0), battery=0.02)
        verdict = detector._assess(failing, payload)
        assert verdict, "mutated payload content returned the stale empty verdict"
