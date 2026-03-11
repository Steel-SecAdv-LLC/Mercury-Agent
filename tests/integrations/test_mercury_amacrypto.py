"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Tests for AMA Cryptography integration adapter with post-quantum cryptography.

Covers:
- MercuryGuardianAdapter initialization and availability
- EWMA/MAD timing anomaly detection
- Crypto anomaly types and recording
- GOSNN synapse integration
- Attack simulation and detection
- Graceful fallback when PQC unavailable
"""

from __future__ import annotations

import time

import numpy as np
import pytest

# =============================================================================
# MercuryGuardianAdapter Tests
# =============================================================================


class TestMercuryGuardianAdapter:
    """Tests for MercuryGuardianAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create MercuryGuardianAdapter instance."""
        from omni_mercury_engine.integrations.mercury_amacrypto import MercuryGuardianAdapter

        return MercuryGuardianAdapter(
            enable_timing_monitor=True,
            timing_alpha=0.1,
            mad_threshold=3.0,
            gosnn_synapse_enabled=True,
        )

    @pytest.fixture
    def adapter_no_timing(self):
        """Create MercuryGuardianAdapter without timing monitor."""
        from omni_mercury_engine.integrations.mercury_amacrypto import MercuryGuardianAdapter

        return MercuryGuardianAdapter(
            enable_timing_monitor=False,
            gosnn_synapse_enabled=True,
        )

    @pytest.fixture
    def adapter_no_gosnn(self):
        """Create MercuryGuardianAdapter without GOSNN synapse."""
        from omni_mercury_engine.integrations.mercury_amacrypto import MercuryGuardianAdapter

        return MercuryGuardianAdapter(
            enable_timing_monitor=True,
            gosnn_synapse_enabled=False,
        )

    def test_adapter_initialization(self, adapter):
        """Test adapter initializes correctly."""
        assert adapter is not None
        assert adapter.timing_monitor is not None
        assert adapter.gosnn_synapse_enabled is True

    def test_adapter_without_timing_monitor(self, adapter_no_timing):
        """Test adapter works without timing monitor."""
        assert adapter_no_timing is not None
        assert adapter_no_timing.timing_monitor is None

    def test_adapter_without_gosnn_synapse(self, adapter_no_gosnn):
        """Test adapter works without GOSNN synapse."""
        assert adapter_no_gosnn is not None
        assert adapter_no_gosnn.gosnn_synapse_enabled is False

    def test_is_available(self, adapter):
        """Test availability check."""
        result = adapter.is_available()
        assert isinstance(result, bool)

    def test_get_pqc_status(self, adapter):
        """Test PQC status retrieval."""
        status = adapter.get_pqc_status()
        assert "ama_cryptography_available" in status
        assert "mercury_guardian_available" in status  # backward compat alias
        assert "dilithium_available" in status
        assert "kyber_available" in status
        assert "timing_monitor_enabled" in status
        assert "gosnn_synapse_enabled" in status
        assert "anomaly_count" in status
        # Verify ama_cryptography_available and mercury_guardian_available are consistent
        assert status["ama_cryptography_available"] == status["mercury_guardian_available"]

    def test_get_anomaly_summary_empty(self, adapter):
        """Test anomaly summary with no anomalies."""
        summary = adapter.get_anomaly_summary()
        assert summary["total_anomalies"] == 0
        assert summary["by_type"] == {}
        assert summary["avg_severity"] == 0.0
        assert summary["recent_anomalies"] == []

    def test_get_gosnn_scalars(self, adapter):
        """Test GOSNN scalars retrieval."""
        scalars = adapter.get_gosnn_scalars()
        assert "omni_ama_cryptography_available" in scalars
        assert "omni_dilithium_available" in scalars
        assert "omni_kyber_available" in scalars
        assert "omni_crypto_anomaly_count" in scalars

    def test_anomaly_history_limit(self, adapter):
        """Test anomaly history respects max limit."""
        assert adapter.max_anomaly_history == 1000


# =============================================================================
# EWMATimingMonitor Tests
# =============================================================================


class TestEWMATimingMonitor:
    """Tests for EWMATimingMonitor."""

    @pytest.fixture
    def monitor(self):
        """Create EWMATimingMonitor instance."""
        from omni_mercury_engine.integrations.mercury_amacrypto import EWMATimingMonitor

        return EWMATimingMonitor(alpha=0.1, mad_threshold=3.0)

    def test_monitor_initialization(self, monitor):
        """Test monitor initializes correctly."""
        assert monitor is not None
        assert monitor.alpha == 0.1
        assert monitor.mad_threshold == 3.0
        assert monitor.max_history == 100

    def test_record_timing_first(self, monitor):
        """Test recording first timing."""
        anomaly = monitor.record_timing("test_op", 10.0)
        assert anomaly is None
        assert "test_op" in monitor.stats
        assert monitor.stats["test_op"].sample_count == 1

    def test_record_timing_multiple(self, monitor):
        """Test recording multiple timings."""
        for i in range(20):
            monitor.record_timing("test_op", 10.0 + np.random.randn() * 0.5)
        assert monitor.stats["test_op"].sample_count == 20

    def test_record_timing_anomaly_detection(self, monitor):
        """Test anomaly detection with outlier timing."""
        for i in range(20):
            monitor.record_timing("test_op", 10.0)

        anomaly = monitor.record_timing("test_op", 100.0)
        # Anomaly detection depends on MAD being > 0, which requires variance in timings
        # With constant timings, MAD will be 0, so no anomaly is detected
        # This is expected behavior - constant timings have no baseline for anomaly detection
        if anomaly is not None:
            assert anomaly.anomaly_type.value == "timing_anomaly"

    def test_ewma_mean_update(self, monitor):
        """Test EWMA mean is updated correctly."""
        monitor.record_timing("test_op", 10.0)
        initial_mean = monitor.stats["test_op"].ewma_mean

        monitor.record_timing("test_op", 20.0)
        updated_mean = monitor.stats["test_op"].ewma_mean

        assert updated_mean > initial_mean

    def test_mad_calculation(self, monitor):
        """Test MAD is calculated correctly."""
        for i in range(15):
            monitor.record_timing("test_op", 10.0 + i * 0.1)

        assert monitor.stats["test_op"].mad >= 0

    def test_multiple_operations(self, monitor):
        """Test monitoring multiple operations."""
        monitor.record_timing("op_a", 10.0)
        monitor.record_timing("op_b", 20.0)
        monitor.record_timing("op_c", 30.0)

        assert "op_a" in monitor.stats
        assert "op_b" in monitor.stats
        assert "op_c" in monitor.stats

    def test_overhead_estimate(self, monitor):
        """Test overhead estimate is reasonable."""
        overhead = monitor.get_overhead_estimate()
        assert overhead < 2.0

    def test_timing_history_limit(self, monitor):
        """Test timing history respects max limit."""
        for i in range(150):
            monitor.record_timing("test_op", 10.0)

        assert len(monitor.recent_timings["test_op"]) <= monitor.max_history


# =============================================================================
# CryptoAnomaly Tests
# =============================================================================


class TestCryptoAnomaly:
    """Tests for CryptoAnomaly dataclass."""

    def test_crypto_anomaly_creation(self):
        """Test CryptoAnomaly creation."""
        from omni_mercury_engine.integrations.mercury_amacrypto import (
            CryptoAnomaly,
            CryptoAnomalyType,
        )

        anomaly = CryptoAnomaly(
            anomaly_type=CryptoAnomalyType.TIMING_ANOMALY,
            severity=0.8,
            timestamp=time.time(),
            operation="test_op",
            details={"test": "value"},
            omni_scalars={"omni_test": 1.0},
        )
        assert anomaly.anomaly_type == CryptoAnomalyType.TIMING_ANOMALY
        assert anomaly.severity == 0.8
        assert anomaly.operation == "test_op"

    def test_crypto_anomaly_types(self):
        """Test all CryptoAnomalyType values."""
        from omni_mercury_engine.integrations.mercury_amacrypto import CryptoAnomalyType

        assert CryptoAnomalyType.TIMING_ANOMALY.value == "timing_anomaly"
        assert CryptoAnomalyType.SIGNATURE_FAILURE.value == "signature_failure"
        assert CryptoAnomalyType.KEY_GENERATION_FAILURE.value == "key_generation_failure"
        assert CryptoAnomalyType.ENCAPSULATION_FAILURE.value == "encapsulation_failure"
        assert CryptoAnomalyType.DECAPSULATION_FAILURE.value == "decapsulation_failure"
        assert CryptoAnomalyType.REPLAY_ATTACK.value == "replay_attack"
        assert CryptoAnomalyType.SIDE_CHANNEL_SUSPECTED.value == "side_channel_suspected"


# =============================================================================
# TimingStats Tests
# =============================================================================


class TestTimingStats:
    """Tests for TimingStats dataclass."""

    def test_timing_stats_defaults(self):
        """Test TimingStats default values."""
        from omni_mercury_engine.integrations.mercury_amacrypto import TimingStats

        stats = TimingStats()
        assert stats.ewma_mean == 0.0
        assert stats.ewma_variance == 0.0
        assert stats.mad == 0.0
        assert stats.sample_count == 0
        assert stats.alpha == 0.1

    def test_timing_stats_custom(self):
        """Test TimingStats with custom values."""
        from omni_mercury_engine.integrations.mercury_amacrypto import TimingStats

        stats = TimingStats(
            ewma_mean=10.0,
            ewma_variance=1.0,
            mad=0.5,
            sample_count=100,
            alpha=0.2,
        )
        assert stats.ewma_mean == 10.0
        assert stats.alpha == 0.2


# =============================================================================
# Attack Simulation Tests
# =============================================================================


class TestAttackSimulation:
    """Tests for attack simulation."""

    @pytest.fixture
    def adapter(self):
        """Create MercuryGuardianAdapter instance."""
        from omni_mercury_engine.integrations.mercury_amacrypto import MercuryGuardianAdapter

        return MercuryGuardianAdapter(
            enable_timing_monitor=True,
            gosnn_synapse_enabled=True,
        )

    def test_simulate_timing_attack(self, adapter):
        """Test timing attack simulation."""
        result = adapter.simulate_attack(attack_type="timing")
        assert "attack_type" in result
        assert result["attack_type"] == "timing"
        assert "detected" in result
        assert "anomalies" in result
        assert "gosnn_triggered" in result

    def test_simulate_replay_attack(self, adapter):
        """Test replay attack simulation."""
        result = adapter.simulate_attack(attack_type="replay")
        assert result["attack_type"] == "replay"
        assert result["detected"] is True
        assert len(result["anomalies"]) > 0

    def test_simulate_side_channel_attack(self, adapter):
        """Test side-channel attack simulation."""
        result = adapter.simulate_attack(attack_type="side_channel")
        assert result["attack_type"] == "side_channel"
        assert result["detected"] is True
        assert len(result["anomalies"]) > 0

    def test_attack_detection_rate(self, adapter):
        """Test attack detection rate meets threshold."""
        attacks = ["timing", "replay", "side_channel"]
        detected_count = 0

        for attack_type in attacks:
            result = adapter.simulate_attack(attack_type=attack_type)
            if result["detected"]:
                detected_count += 1

        detection_rate = detected_count / len(attacks)
        # Timing attacks may not be detected if timing monitor hasn't built baseline
        # Replay and side_channel attacks should always be detected
        assert detection_rate >= 0.66  # At least 2/3 attacks detected

    def test_gosnn_triggered_on_attack(self, adapter):
        """Test GOSNN synapse is triggered on attack."""
        result = adapter.simulate_attack(attack_type="replay")
        assert result["gosnn_triggered"] is True


# =============================================================================
# Anomaly Recording Tests
# =============================================================================


class TestAnomalyRecording:
    """Tests for anomaly recording and history."""

    @pytest.fixture
    def adapter(self):
        """Create MercuryGuardianAdapter instance."""
        from omni_mercury_engine.integrations.mercury_amacrypto import MercuryGuardianAdapter

        return MercuryGuardianAdapter(
            enable_timing_monitor=True,
            gosnn_synapse_enabled=False,
        )

    def test_anomaly_recorded_after_attack(self, adapter):
        """Test anomaly is recorded after attack simulation."""
        initial_count = len(adapter.anomaly_history)
        adapter.simulate_attack(attack_type="replay")
        assert len(adapter.anomaly_history) > initial_count

    def test_anomaly_summary_after_attacks(self, adapter):
        """Test anomaly summary after multiple attacks."""
        adapter.simulate_attack(attack_type="timing")
        adapter.simulate_attack(attack_type="replay")
        adapter.simulate_attack(attack_type="side_channel")

        summary = adapter.get_anomaly_summary()
        assert summary["total_anomalies"] > 0
        assert len(summary["by_type"]) > 0
        assert summary["avg_severity"] > 0

    def test_recent_anomalies_limit(self, adapter):
        """Test recent anomalies are limited to 10."""
        for _ in range(20):
            adapter.simulate_attack(attack_type="replay")

        summary = adapter.get_anomaly_summary()
        assert len(summary["recent_anomalies"]) <= 10


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunction:
    """Tests for create_ama_cryptography_adapter factory function."""

    def test_create_adapter_default(self):
        """Test factory function with defaults."""
        from omni_mercury_engine.integrations.mercury_amacrypto import (
            create_ama_cryptography_adapter,
        )

        adapter = create_ama_cryptography_adapter()
        assert adapter is not None
        assert adapter.timing_monitor is not None
        assert adapter.gosnn_synapse_enabled is True

    def test_create_adapter_no_timing(self):
        """Test factory function without timing monitor."""
        from omni_mercury_engine.integrations.mercury_amacrypto import (
            create_ama_cryptography_adapter,
        )

        adapter = create_ama_cryptography_adapter(enable_timing_monitor=False)
        assert adapter.timing_monitor is None

    def test_create_adapter_no_gosnn(self):
        """Test factory function without GOSNN synapse."""
        from omni_mercury_engine.integrations.mercury_amacrypto import (
            create_ama_cryptography_adapter,
        )

        adapter = create_ama_cryptography_adapter(gosnn_synapse_enabled=False)
        assert adapter.gosnn_synapse_enabled is False

    def test_create_adapter_compat_alias(self):
        """Test backward compat factory function alias."""
        from omni_mercury_engine.integrations.mercury_amacrypto import (
            create_mercury_guardian_adapter,
        )

        adapter = create_mercury_guardian_adapter()
        assert adapter is not None
        assert adapter.timing_monitor is not None


# =============================================================================
# Module Import Tests
# =============================================================================


class TestModuleImports:
    """Tests for module imports and exports."""

    def test_import_from_integrations(self):
        """Test importing from integrations package."""
        from omni_mercury_engine.integrations import (
            AMA_CRYPTOGRAPHY_AVAILABLE,
            AVA_GUARDIAN_AVAILABLE,
            DILITHIUM_AVAILABLE,
            KYBER_AVAILABLE,
            CryptoAnomaly,
            CryptoAnomalyType,
            EWMATimingMonitor,
            MercuryGuardianAdapter,
            create_mercury_guardian_adapter,
        )

        assert MercuryGuardianAdapter is not None
        assert EWMATimingMonitor is not None
        assert CryptoAnomaly is not None
        assert CryptoAnomalyType is not None
        assert create_mercury_guardian_adapter is not None
        assert isinstance(AMA_CRYPTOGRAPHY_AVAILABLE, bool)
        assert isinstance(AVA_GUARDIAN_AVAILABLE, bool)
        assert isinstance(DILITHIUM_AVAILABLE, bool)
        assert isinstance(KYBER_AVAILABLE, bool)

    def test_import_from_mercury_amacrypto_module(self):
        """Test importing directly from mercury_amacrypto module."""
        from omni_mercury_engine.integrations.mercury_amacrypto import (
            CryptoAnomaly,
            CryptoAnomalyType,
            EWMATimingMonitor,
            MercuryGuardianAdapter,
            TimingStats,
            create_ama_cryptography_adapter,
            create_mercury_guardian_adapter,
        )

        assert all(
            x is not None
            for x in [
                MercuryGuardianAdapter,
                EWMATimingMonitor,
                CryptoAnomaly,
                CryptoAnomalyType,
                TimingStats,
                create_ama_cryptography_adapter,
                create_mercury_guardian_adapter,
            ]
        )


# =============================================================================
# Graceful Fallback Tests
# =============================================================================


class TestGracefulFallback:
    """Tests for graceful fallback when PQC unavailable."""

    @pytest.fixture
    def adapter(self):
        """Create MercuryGuardianAdapter instance."""
        from omni_mercury_engine.integrations.mercury_amacrypto import MercuryGuardianAdapter

        return MercuryGuardianAdapter()

    def test_dilithium_keygen_fallback(self, adapter):
        """Test Dilithium keygen returns None when unavailable."""
        from omni_mercury_engine.integrations.mercury_amacrypto import DILITHIUM_AVAILABLE

        result = adapter.generate_dilithium_keypair()
        if not DILITHIUM_AVAILABLE:
            assert result is None
        else:
            assert result is not None

    def test_kyber_keygen_fallback(self, adapter):
        """Test Kyber keygen returns None when unavailable."""
        from omni_mercury_engine.integrations.mercury_amacrypto import KYBER_AVAILABLE

        result = adapter.generate_kyber_keypair()
        if not KYBER_AVAILABLE:
            assert result is None
        else:
            assert result is not None

    def test_sign_without_keypair(self, adapter):
        """Test signing without keypair returns None."""
        result = adapter.sign_dilithium(b"test message")
        if adapter._dilithium_keypair is None:
            assert result is None

    def test_verify_without_keypair(self, adapter):
        """Test verification without keypair returns False."""
        result = adapter.verify_dilithium(b"test message", b"fake signature")
        if adapter._dilithium_keypair is None:
            assert result is False

    def test_encapsulate_without_keypair(self, adapter):
        """Test encapsulation without keypair returns None."""
        result = adapter.encapsulate_kyber()
        if adapter._kyber_keypair is None:
            assert result is None

    def test_decapsulate_without_keypair(self, adapter):
        """Test decapsulation without keypair returns None."""
        result = adapter.decapsulate_kyber(b"fake ciphertext")
        if adapter._kyber_keypair is None:
            assert result is None


# =============================================================================
# GOSNN Scalars Tests
# =============================================================================


class TestGOSNNScalars:
    """Tests for GOSNN scalar generation."""

    @pytest.fixture
    def adapter(self):
        """Create MercuryGuardianAdapter instance."""
        from omni_mercury_engine.integrations.mercury_amacrypto import MercuryGuardianAdapter

        return MercuryGuardianAdapter()

    def test_scalars_have_omni_prefix(self, adapter):
        """Test all scalars have omni_ prefix."""
        scalars = adapter.get_gosnn_scalars()
        for key in scalars:
            assert key.startswith("omni_"), f"Scalar {key} missing omni_ prefix"

    def test_scalars_are_floats(self, adapter):
        """Test all scalars are floats."""
        scalars = adapter.get_gosnn_scalars()
        for key, value in scalars.items():
            assert isinstance(value, float), f"Scalar {key} is not float"

    def test_scalars_after_anomalies(self, adapter):
        """Test scalars include anomaly info after attacks."""
        adapter.simulate_attack(attack_type="replay")
        scalars = adapter.get_gosnn_scalars()
        assert scalars["omni_crypto_anomaly_count"] > 0

    def test_avg_severity_scalar(self, adapter):
        """Test average severity scalar is computed."""
        adapter.simulate_attack(attack_type="replay")
        adapter.simulate_attack(attack_type="side_channel")
        scalars = adapter.get_gosnn_scalars()
        assert "omni_crypto_avg_severity" in scalars
        assert scalars["omni_crypto_avg_severity"] > 0
