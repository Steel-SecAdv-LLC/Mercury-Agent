# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stuck-actuator health statistic and its DroneAnomalyDetector wiring.

Pins the ALFA-measured behaviour (benchmarks/alfa_uav_benchmark.py): a
frozen actuation channel is invisible to per-tick range checks but
collapses its rolling variance, and the running-max baseline keeps the
collapse visible even after the fault fills the rolling window.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.drone.actuator_health import (
    ActuatorHealthMonitor,
    causal_rolling_std,
    stuck_actuator_scores,
)
from omni_mercury_engine.detectors.drone.detector import (
    DroneAnomalyDetector,
    DroneState,
    MissionPhase,
)


def _dithering_pwm(n: int, seed: int = 0, std: float = 30.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 1500.0 + rng.normal(0.0, std, size=(n, 4))


class TestCausalRollingStd:
    def test_matches_naive_windows(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(size=(50, 3))
        out = causal_rolling_std(x, window=8)
        for t in (0, 3, 7, 8, 25, 49):
            lo = max(t - 7, 0)
            expected = x[lo : t + 1].std(axis=0)
            np.testing.assert_allclose(out[t], expected, atol=1e-9)

    def test_rejects_bad_inputs(self) -> None:
        with pytest.raises(ValueError, match="window"):
            causal_rolling_std(np.zeros((5, 2)), window=0)
        with pytest.raises(ValueError, match="2-D"):
            causal_rolling_std(np.zeros(5), window=3)


class TestStuckActuatorScores:
    def test_frozen_channel_scores_high_healthy_stays_low(self) -> None:
        pwm = _dithering_pwm(200)
        pwm[120:, 2] = 1500.0  # channel 2 freezes mid-flight
        baseline = pwm[:60].std(axis=0)
        scores = stuck_actuator_scores(pwm, baseline, window=32)
        assert scores[119] < 0.8  # healthy dithering never collapses
        assert scores[119 + 32 :].min() > 0.95  # full window after freeze

    def test_inactive_channels_never_stuck(self) -> None:
        pwm = np.full((100, 4), 1500.0)  # constant = never demonstrated activity
        scores = stuck_actuator_scores(pwm, pwm[:30].std(axis=0), window=16)
        assert np.all(scores == 0.0)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="channels"):
            stuck_actuator_scores(np.zeros((10, 4)), np.zeros(3))


class TestActuatorHealthMonitor:
    def test_streaming_matches_fault_semantics(self) -> None:
        monitor = ActuatorHealthMonitor(n_channels=4, window=32)
        pwm = _dithering_pwm(200, seed=2)
        pwm[120:, 1] = 1500.0
        scores = [monitor.update(row) for row in pwm]
        assert max(scores[:120]) < 0.9
        assert min(scores[120 + 32 :]) > 0.95

    def test_running_max_baseline_survives_fault_filled_window(self) -> None:
        """A fault inside the window cannot drag the reference down."""
        monitor = ActuatorHealthMonitor(n_channels=2, window=16)
        pwm = _dithering_pwm(300, seed=3)[:, :2]
        pwm[100:, 0] = 1500.0
        for row in pwm:
            score = monitor.update(row)
        # 184 ticks after the freeze the window holds only frozen samples,
        # yet the demonstrated-activity baseline keeps the alarm up.
        assert score > 0.95
        assert monitor.baseline_std[0] > 10.0

    def test_reset_clears_flight_scoped_state(self) -> None:
        """A calmer second flight must not alarm against the first's baseline."""
        monitor = ActuatorHealthMonitor(n_channels=2, window=16)
        for row in _dithering_pwm(80, seed=7, std=60.0)[:, :2]:  # lively flight
            monitor.update(row)
        monitor.reset()
        assert monitor.baseline_std.tolist() == [0.0, 0.0]
        calm = [monitor.update(row) for row in _dithering_pwm(80, seed=8, std=5.0)[:, :2]]
        assert max(calm) < 0.9  # judged against its own activity, not flight 1's

    def test_partial_window_gives_no_verdict(self) -> None:
        monitor = ActuatorHealthMonitor(n_channels=2, window=10)
        for row in _dithering_pwm(9, seed=4)[:, :2]:
            assert monitor.update(row) == 0.0

    def test_rejects_bad_construction(self) -> None:
        with pytest.raises(ValueError):
            ActuatorHealthMonitor(n_channels=0)
        with pytest.raises(ValueError):
            ActuatorHealthMonitor(n_channels=2, window=0)


class TestDroneDetectorWiring:
    def _state(self, motor_speeds: np.ndarray) -> DroneState:
        return DroneState(
            position=np.zeros(3),
            velocity=np.array([15.0, 0.0, 0.0]),
            attitude=np.zeros(3),
            battery_level=0.9,
            altitude=100.0,
            gps_satellites=10,
            signal_strength=0.9,
            motor_speeds=motor_speeds,
            temperature=40.0,
            mission_phase=MissionPhase.ON_MISSION,
        )

    def test_stuck_channel_raises_actuator_fault_and_healthy_does_not(self) -> None:
        detector = DroneAnomalyDetector(enable_dronlomaly=False)
        pwm = _dithering_pwm(120, seed=5)
        pwm[60:, 2] = 1500.0
        pre_fault_alarms = 0
        first_alarm_tick = None
        for t, row in enumerate(pwm):
            faults = detector.detect_faults(self._state(row))
            stuck = [f for f in faults if f.detected_by == "ActuatorHealth"]
            if t < 60:
                pre_fault_alarms += len(stuck)
            elif stuck and first_alarm_tick is None:
                first_alarm_tick = t
                assert stuck[0].severity >= 0.85
                assert stuck[0].fault_type.value == "ACTUATOR_FAILURE"
        assert pre_fault_alarms == 0
        assert first_alarm_tick is not None and first_alarm_tick <= 60 + 32

    def test_disable_flag_silences_monitor(self) -> None:
        detector = DroneAnomalyDetector(enable_dronlomaly=False, enable_actuator_health=False)
        pwm = _dithering_pwm(100, seed=6)
        pwm[40:, 0] = 1500.0
        for row in pwm:
            faults = detector.detect_faults(self._state(row))
            assert not [f for f in faults if f.detected_by == "ActuatorHealth"]
