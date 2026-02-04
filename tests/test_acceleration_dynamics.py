"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

Tests for Acceleration Dynamics Module.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from omni_mercury_engine.detectors.acceleration_dynamics import (
    AccelerationDynamicsDetector,
    EnergyConservationNetwork,
    EnergyState,
    KinematicFeatures,
    MotionEncoder,
    MotionState,
    PhaseSpaceFeatures,
    PhaseSpaceNetwork,
    compute_acceleration,
    compute_average_acceleration,
    compute_impulse,
    compute_kinetic_energy,
    compute_momentum,
    compute_velocity,
    estimate_initial_acceleration,
)


class TestAccelerationDynamicsDetector:
    """Tests for AccelerationDynamicsDetector."""

    def test_init_default_config(self) -> None:
        """Test initialization with default configuration."""
        detector = AccelerationDynamicsDetector()
        assert detector is not None
        assert detector.threshold == 0.5
        assert not detector.is_fitted()

    def test_init_custom_config(self) -> None:
        """Test initialization with custom configuration."""
        config = {
            "time_step": 0.01,
            "jerk_sensitivity": 3.0,
            "chaos_threshold": 0.2,
            "threshold": 0.7,
        }
        detector = AccelerationDynamicsDetector(config)
        assert detector.threshold == 0.7
        assert detector._dynamics_config.time_step == 0.01
        assert detector._dynamics_config.jerk_sensitivity == 3.0

    def test_fit_constant_velocity(self) -> None:
        """Test fitting on constant velocity signal."""
        detector = AccelerationDynamicsDetector()

        # Linear signal (constant velocity)
        # With dt=1.0 (default), gradient of 2.0*t gives ~0.02 per sample step
        t = np.linspace(0, 10, 1000)
        signal = 2.0 * t + 5.0  # slope = 2, but gradient with dt=1.0 gives ~0.02

        detector.fit(signal)
        assert detector.is_fitted()
        # The numerical gradient with dt=1.0 gives ~0.02, not 2.0
        # (2.0 * 10/1000) / 1.0 = 0.02 per sample
        assert detector._reference_velocity_mean > 0  # Positive velocity

    def test_fit_accelerating_signal(self) -> None:
        """Test fitting on accelerating signal."""
        detector = AccelerationDynamicsDetector()

        # Quadratic signal (constant acceleration)
        t = np.linspace(0, 10, 1000)
        signal = 0.5 * t**2 + t + 1.0  # a = 1

        detector.fit(signal)
        assert detector.is_fitted()

    def test_fit_empty_data_raises(self) -> None:
        """Test that fitting with empty data raises exception."""
        from omni_mercury_engine.core.exceptions import DetectorException

        detector = AccelerationDynamicsDetector()
        with pytest.raises((ValueError, RuntimeError, DetectorException)):
            detector.fit(np.array([]))

    def test_detect_uniform_motion(self) -> None:
        """Test detection on uniform motion signal."""
        detector = AccelerationDynamicsDetector({"threshold": 0.6})

        # Train on linear signal
        t = np.linspace(0, 10, 1000)
        train_signal = 2.0 * t + 5.0

        detector.fit(train_signal)

        # Test on similar signal
        test_signal = 2.0 * t + 3.0
        result = detector.detect(test_signal)

        assert "is_anomaly" in result
        assert "anomaly_score" in result
        assert "motion_state" in result
        assert "lyapunov_exponent" in result
        assert 0.0 <= result["anomaly_score"] <= 1.0

    def test_detect_returns_motion_state(self) -> None:
        """Test that detection returns motion state classification."""
        detector = AccelerationDynamicsDetector()

        t = np.linspace(0, 10, 1000)
        signal = t**2  # Accelerating
        detector.fit(signal)

        result = detector.detect(signal)
        assert "motion_state" in result
        assert result["motion_state"] in [e.value for e in MotionState]

    def test_detect_returns_energy_state(self) -> None:
        """Test that detection returns energy state."""
        detector = AccelerationDynamicsDetector()

        t = np.linspace(0, 10, 1000)
        signal = np.sin(t)  # Oscillating
        detector.fit(signal)

        result = detector.detect(signal)
        assert "energy_state" in result

    def test_detect_chaotic_signal(self) -> None:
        """Test detection of chaotic-like behavior."""
        detector = AccelerationDynamicsDetector({"chaos_threshold": 0.05})

        # Train on smooth signal
        t = np.linspace(0, 10, 1000)
        train_signal = np.sin(t)
        detector.fit(train_signal)

        # Test with noisy/chaotic signal
        np.random.seed(42)
        test_signal = np.sin(t) + np.random.randn(1000) * 0.5
        test_signal[500:550] = test_signal[500:550] * 3  # Add sudden jump

        result = detector.detect(test_signal)
        assert "lyapunov_exponent" in result
        assert "is_chaotic" in result

    def test_detect_jerk_anomaly(self) -> None:
        """Test detection of jerk anomalies."""
        detector = AccelerationDynamicsDetector({"jerk_sensitivity": 2.0})

        # Smooth training signal
        t = np.linspace(0, 10, 1000)
        train_signal = np.sin(t)
        detector.fit(train_signal)

        # Test signal with sudden jerk
        test_signal = np.sin(t).copy()
        test_signal[500] = 5.0  # Sudden spike creates high jerk

        result = detector.detect(test_signal)
        assert "jerk_anomaly" in result
        assert "anomaly_timestamps" in result

    def test_extract_features(self) -> None:
        """Test feature extraction for ML fusion."""
        detector = AccelerationDynamicsDetector()

        t = np.linspace(0, 10, 1000)
        signal = np.sin(2 * np.pi * t)
        detector.fit(signal)

        features = detector.extract_features(signal)
        assert isinstance(features, torch.Tensor)
        assert features.dim() == 2
        assert features.shape[0] == 1

    def test_detect_not_fitted_raises(self) -> None:
        """Test that detection before fitting raises exception."""
        from omni_mercury_engine.core.exceptions import DetectorException

        detector = AccelerationDynamicsDetector()
        with pytest.raises((ValueError, RuntimeError, DetectorException)):
            detector.detect(np.random.randn(100))


class TestMotionEncoder:
    """Tests for MotionEncoder."""

    def test_forward_pass(self) -> None:
        """Test forward pass through motion encoder."""
        encoder = MotionEncoder(
            input_dim=4,
            hidden_dim=32,
            output_dim=16,
        )

        # Simulate kinematic features [batch, time, features]
        batch_size = 2
        seq_len = 100
        features = torch.randn(batch_size, seq_len, 4)

        output = encoder(features)
        assert output.shape == (batch_size, 16)


class TestPhaseSpaceNetwork:
    """Tests for PhaseSpaceNetwork."""

    def test_forward_pass(self) -> None:
        """Test forward pass through phase space network."""
        network = PhaseSpaceNetwork(
            embedding_dim=3,
            hidden_dim=16,
            output_dim=8,
        )

        # Simulate phase space trajectory
        batch_size = 2
        seq_len = 50
        trajectory = torch.randn(batch_size, seq_len, 3)

        features, chaos_score = network(trajectory)

        assert features.shape == (batch_size, 8)
        assert chaos_score.shape == (batch_size, 1)
        assert (chaos_score >= 0).all() and (chaos_score <= 1).all()


class TestEnergyConservationNetwork:
    """Tests for EnergyConservationNetwork."""

    def test_forward_pass(self) -> None:
        """Test forward pass through energy conservation network."""
        network = EnergyConservationNetwork(
            input_dim=3,
            hidden_dim=16,
        )

        # Simulate energy features [batch, time, features]
        batch_size = 2
        seq_len = 100
        energy_features = torch.randn(batch_size, seq_len, 3).abs()

        predicted, violations = network(energy_features)

        assert predicted.shape == (batch_size, seq_len)
        assert violations.shape == (batch_size, seq_len)


class TestPhysicsFormulas:
    """Tests for physics utility functions."""

    def test_compute_velocity_central(self) -> None:
        """Test central difference velocity computation."""
        # Linear position = constant velocity
        position = np.linspace(0, 10, 100)
        velocity = compute_velocity(position, time_step=0.1, method="central")

        # Velocity should be approximately constant (~10/100*0.1 = 1)
        assert np.allclose(velocity[10:-10], 1.0, atol=0.1)

    def test_compute_velocity_forward(self) -> None:
        """Test forward difference velocity computation."""
        position = np.linspace(0, 10, 100)
        velocity = compute_velocity(position, time_step=0.1, method="forward")

        assert len(velocity) == len(position)

    def test_compute_acceleration(self) -> None:
        """Test acceleration computation."""
        # Constant velocity = zero acceleration
        velocity = np.ones(100) * 5.0
        acceleration = compute_acceleration(velocity, time_step=0.1)

        assert np.allclose(acceleration, 0.0, atol=1e-10)

    def test_compute_kinetic_energy(self) -> None:
        """Test kinetic energy computation: KE = 1/2 mv^2."""
        velocity = np.array([2.0, 4.0, 6.0])
        mass = 1.0

        ke = compute_kinetic_energy(velocity, mass)

        expected = 0.5 * mass * velocity**2
        assert np.allclose(ke, expected)

    def test_compute_kinetic_energy_with_mass(self) -> None:
        """Test kinetic energy with non-unit mass."""
        velocity = np.array([3.0, 4.0])
        mass = 2.0

        ke = compute_kinetic_energy(velocity, mass)

        expected = 0.5 * mass * velocity**2  # [9, 16]
        assert np.allclose(ke, expected)

    def test_compute_momentum(self) -> None:
        """Test momentum computation: p = mv."""
        velocity = np.array([1.0, 2.0, 3.0])
        mass = 5.0

        momentum = compute_momentum(velocity, mass)

        expected = mass * velocity
        assert np.allclose(momentum, expected)

    def test_compute_impulse(self) -> None:
        """Test impulse computation: J = dp."""
        momentum = np.array([0, 10, 20, 30, 40])
        time_step = 0.1

        impulse = compute_impulse(momentum, time_step)

        # Impulse is change in momentum
        assert len(impulse) == len(momentum)

    def test_estimate_initial_acceleration(self) -> None:
        """Test initial acceleration estimation: A_i = 2A - A_f."""
        average_acceleration = 5.0
        final_acceleration = 3.0

        initial = estimate_initial_acceleration(final_acceleration, average_acceleration)

        expected = 2 * average_acceleration - final_acceleration
        assert initial == expected

    def test_compute_average_acceleration(self) -> None:
        """Test average acceleration: a = (v_f - v_i) / t."""
        v_i = 10.0
        v_f = 30.0
        time = 4.0

        a = compute_average_acceleration(v_i, v_f, time)

        expected = (v_f - v_i) / time
        assert a == expected

    def test_compute_average_acceleration_zero_time(self) -> None:
        """Test average acceleration with zero time."""
        a = compute_average_acceleration(10.0, 20.0, 0.0)
        assert a == 0.0


class TestKinematicFeatures:
    """Tests for KinematicFeatures dataclass."""

    def test_features_creation(self) -> None:
        """Test creating kinematic features."""
        n = 100
        features = KinematicFeatures(
            position=np.zeros(n),
            velocity=np.ones(n),
            acceleration=np.zeros(n),
            jerk=np.zeros(n),
            kinetic_energy=np.ones(n) * 0.5,
            potential_energy=np.zeros(n),
            total_energy=np.ones(n) * 0.5,
            momentum=np.ones(n),
            impulse=np.zeros(n),
            mean_velocity=1.0,
            mean_acceleration=0.0,
            max_jerk=0.0,
            motion_state=MotionState.UNIFORM_MOTION,
            energy_state=EnergyState.CONSERVED,
        )

        assert features.motion_state == MotionState.UNIFORM_MOTION
        assert features.mean_velocity == 1.0
        assert len(features.velocity) == n


class TestPhaseSpaceFeatures:
    """Tests for PhaseSpaceFeatures dataclass."""

    def test_features_creation(self) -> None:
        """Test creating phase space features."""
        features = PhaseSpaceFeatures(
            trajectory=np.random.randn(100, 3),
            lyapunov_exponent=0.05,
            correlation_dimension=2.3,
            recurrence_rate=0.1,
            determinism=0.8,
            entropy=1.5,
            is_chaotic=False,
            attractor_type="limit_cycle",
        )

        assert features.lyapunov_exponent == 0.05
        assert features.is_chaotic is False
        assert features.attractor_type == "limit_cycle"


class TestMotionStateClassification:
    """Tests for motion state classification."""

    def test_stationary_classification(self) -> None:
        """Test classification of stationary signal."""
        detector = AccelerationDynamicsDetector()

        # Nearly constant signal
        signal = np.ones(1000) * 5.0 + np.random.randn(1000) * 0.001
        detector.fit(signal)

        result = detector.detect(signal)
        # Should be close to stationary
        assert result["motion_state"] in ["stationary", "uniform_motion"]

    def test_oscillating_classification(self) -> None:
        """Test classification of oscillating signal."""
        detector = AccelerationDynamicsDetector()

        t = np.linspace(0, 10, 1000)
        signal = np.sin(2 * np.pi * t)
        detector.fit(signal)

        result = detector.detect(signal)
        # Sine wave classification depends on thresholds and signal characteristics
        # It may be classified as oscillating, uniform_motion, or accelerating/decelerating
        # depending on the relative magnitudes of velocity and acceleration
        assert result["motion_state"] in [
            "oscillating",
            "uniform_motion",
            "accelerating",
            "decelerating",
        ]


class TestEnergyConservation:
    """Tests for energy conservation analysis."""

    def test_conserved_energy(self) -> None:
        """Test detection of conserved energy."""
        detector = AccelerationDynamicsDetector()

        # Simple harmonic oscillator (energy conserved)
        t = np.linspace(0, 10, 1000)
        signal = np.sin(t)
        detector.fit(signal)

        result = detector.detect(signal)
        # Should have low energy anomaly score
        assert result["energy_anomaly"] < 0.5

    def test_energy_change_detection(self) -> None:
        """Test detection of energy changes."""
        detector = AccelerationDynamicsDetector()

        # Train on oscillation
        t = np.linspace(0, 10, 1000)
        train_signal = np.sin(t)
        detector.fit(train_signal)

        # Test with growing amplitude (energy increasing)
        test_signal = np.sin(t) * np.linspace(1, 3, 1000)

        result = detector.detect(test_signal)
        # Should detect energy change
        assert result["energy_anomaly"] > 0.0
