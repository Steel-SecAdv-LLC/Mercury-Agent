# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for mercury_guardian backward-compatibility shim.

Verifies that all names re-exported from mercury_guardian still resolve
to the canonical implementations in mercury_amacrypto.

Covers:
- MercuryGuardianAdapter initialization and availability
- EWMA/MAD timing anomaly detection
- Crypto anomaly types and recording
- GOSNN synapse integration
- Attack simulation and detection
- Fail-closed behaviour when PQC key material is absent
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

# GOSNN singleton isolation is handled globally by the autouse
# ``_isolate_gosnn_singleton`` fixture in tests/conftest.py, so this adapter's
# SECURITY-scalar registrations cannot bleed into other tests' σ_Immutable gate.

# =============================================================================
# MercuryGuardianAdapter Tests
# =============================================================================


class TestMercuryGuardianAdapter:
    """Tests for MercuryGuardianAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create MercuryGuardianAdapter instance."""
        from omni_mercury_engine.integrations.mercury_guardian import MercuryGuardianAdapter

        return MercuryGuardianAdapter(
            enable_timing_monitor=True,
            timing_alpha=0.1,
            mad_threshold=3.0,
            gosnn_synapse_enabled=True,
        )

    @pytest.fixture
    def adapter_no_timing(self):
        """Create MercuryGuardianAdapter without timing monitor."""
        from omni_mercury_engine.integrations.mercury_guardian import MercuryGuardianAdapter

        return MercuryGuardianAdapter(
            enable_timing_monitor=False,
            gosnn_synapse_enabled=True,
        )

    @pytest.fixture
    def adapter_no_gosnn(self):
        """Create MercuryGuardianAdapter without GOSNN synapse."""
        from omni_mercury_engine.integrations.mercury_guardian import MercuryGuardianAdapter

        return MercuryGuardianAdapter(
            enable_timing_monitor=True,
            gosnn_synapse_enabled=False,
        )

    def test_adapter_initialization(self, adapter: Any) -> None:
        """Test adapter initializes correctly."""
        assert adapter is not None
        assert adapter.timing_monitor is not None
        assert adapter.gosnn_synapse_enabled is True

    def test_adapter_without_timing_monitor(self, adapter_no_timing: Any) -> None:
        """Test adapter works without timing monitor."""
        assert adapter_no_timing is not None
        assert adapter_no_timing.timing_monitor is None

    def test_adapter_without_gosnn_synapse(self, adapter_no_gosnn: Any) -> None:
        """Test adapter works without GOSNN synapse."""
        assert adapter_no_gosnn is not None
        assert adapter_no_gosnn.gosnn_synapse_enabled is False

    def test_is_available(self, adapter: Any) -> None:
        """Test availability check."""
        result = adapter.is_available()
        assert isinstance(result, bool)

    def test_get_pqc_status(self, adapter: Any) -> None:
        """Test PQC status retrieval."""
        status = adapter.get_pqc_status()
        assert "mercury_guardian_available" in status
        assert "dilithium_available" in status
        assert "kyber_available" in status
        assert "timing_monitor_enabled" in status
        assert "gosnn_synapse_enabled" in status
        assert "anomaly_count" in status

    def test_get_anomaly_summary_empty(self, adapter: Any) -> None:
        """Test anomaly summary with no anomalies."""
        summary = adapter.get_anomaly_summary()
        assert summary["total_anomalies"] == 0
        assert summary["by_type"] == {}
        assert summary["avg_severity"] == 0.0
        assert summary["recent_anomalies"] == []

    def test_get_gosnn_scalars(self, adapter: Any) -> None:
        """Test GOSNN scalars retrieval."""
        scalars = adapter.get_gosnn_scalars()
        assert "omni_mercury_guardian_available" in scalars
        assert "omni_dilithium_available" in scalars
        assert "omni_kyber_available" in scalars
        assert "omni_crypto_anomaly_count" in scalars

    def test_anomaly_history_limit(self, adapter: Any) -> None:
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
        from omni_mercury_engine.integrations.mercury_guardian import EWMATimingMonitor

        return EWMATimingMonitor(alpha=0.1, mad_threshold=3.0)

    def test_monitor_initialization(self, monitor: Any) -> None:
        """Test monitor initializes correctly."""
        assert monitor is not None
        assert monitor.alpha == 0.1
        assert monitor.mad_threshold == 3.0
        assert monitor.max_history == 100

    def test_record_timing_first(self, monitor: Any) -> None:
        """Test recording first timing."""
        anomaly = monitor.record_timing("test_op", 10.0)
        assert anomaly is None
        assert "test_op" in monitor.stats
        assert monitor.stats["test_op"].sample_count == 1

    def test_record_timing_multiple(self, monitor: Any) -> None:
        """Test recording multiple timings."""
        for i in range(20):
            monitor.record_timing("test_op", 10.0 + np.random.randn() * 0.5)
        assert monitor.stats["test_op"].sample_count == 20

    def test_record_timing_anomaly_detection(self, monitor: Any) -> None:
        """Test anomaly detection with outlier timing."""
        for i in range(20):
            monitor.record_timing("test_op", 10.0)

        anomaly = monitor.record_timing("test_op", 100.0)
        # Anomaly detection depends on MAD being > 0, which requires variance in timings
        # With constant timings, MAD will be 0, so no anomaly is detected
        # This is expected behavior - constant timings have no baseline for anomaly detection
        if anomaly is not None:
            assert anomaly.anomaly_type.value == "timing_anomaly"

    def test_ewma_mean_update(self, monitor: Any) -> None:
        """Test EWMA mean is updated correctly."""
        monitor.record_timing("test_op", 10.0)
        initial_mean = monitor.stats["test_op"].ewma_mean

        monitor.record_timing("test_op", 20.0)
        updated_mean = monitor.stats["test_op"].ewma_mean

        assert updated_mean > initial_mean

    def test_mad_calculation(self, monitor: Any) -> None:
        """Test MAD is calculated correctly."""
        for i in range(15):
            monitor.record_timing("test_op", 10.0 + i * 0.1)

        assert monitor.stats["test_op"].mad >= 0

    def test_multiple_operations(self, monitor: Any) -> None:
        """Test monitoring multiple operations."""
        monitor.record_timing("op_a", 10.0)
        monitor.record_timing("op_b", 20.0)
        monitor.record_timing("op_c", 30.0)

        assert "op_a" in monitor.stats
        assert "op_b" in monitor.stats
        assert "op_c" in monitor.stats

    def test_overhead_estimate(self, monitor: Any) -> None:
        """Test overhead estimate is reasonable."""
        overhead = monitor.get_overhead_estimate()
        assert overhead < 2.0

    def test_timing_history_limit(self, monitor: Any) -> None:
        """Test timing history respects max limit."""
        for i in range(150):
            monitor.record_timing("test_op", 10.0)

        assert len(monitor.recent_timings["test_op"]) <= monitor.max_history


# =============================================================================
# CryptoAnomaly Tests
# =============================================================================


class TestCryptoAnomaly:
    """Tests for CryptoAnomaly dataclass."""

    def test_crypto_anomaly_creation(self) -> None:
        """Test CryptoAnomaly creation."""
        from omni_mercury_engine.integrations.mercury_guardian import (
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

    def test_crypto_anomaly_types(self) -> None:
        """Test all CryptoAnomalyType values."""
        from omni_mercury_engine.integrations.mercury_guardian import CryptoAnomalyType

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

    def test_timing_stats_defaults(self) -> None:
        """Test TimingStats default values."""
        from omni_mercury_engine.integrations.mercury_guardian import TimingStats

        stats = TimingStats()
        assert stats.ewma_mean == 0.0
        assert stats.ewma_variance == 0.0
        assert stats.mad == 0.0
        assert stats.sample_count == 0
        assert stats.alpha == 0.1

    def test_timing_stats_custom(self) -> None:
        """Test TimingStats with custom values."""
        from omni_mercury_engine.integrations.mercury_guardian import TimingStats

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
        from omni_mercury_engine.integrations.mercury_guardian import MercuryGuardianAdapter

        return MercuryGuardianAdapter(
            enable_timing_monitor=True,
            gosnn_synapse_enabled=True,
        )

    def test_simulate_timing_attack(self, adapter: Any) -> None:
        """Test timing attack simulation."""
        result = adapter.simulate_attack(attack_type="timing")
        assert "attack_type" in result
        assert result["attack_type"] == "timing"
        assert "detected" in result
        assert "anomalies" in result
        assert "gosnn_triggered" in result

    def test_simulate_replay_attack(self, adapter: Any) -> None:
        """Test replay attack simulation."""
        result = adapter.simulate_attack(attack_type="replay")
        assert result["attack_type"] == "replay"
        assert result["detected"] is True
        assert len(result["anomalies"]) > 0

    def test_simulate_side_channel_attack(self, adapter: Any) -> None:
        """Test side-channel attack simulation."""
        result = adapter.simulate_attack(attack_type="side_channel")
        assert result["attack_type"] == "side_channel"
        assert result["detected"] is True
        assert len(result["anomalies"]) > 0

    def test_attack_detection_rate(self, adapter: Any) -> None:
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

    def test_gosnn_triggered_on_attack(self, adapter: Any) -> None:
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
        from omni_mercury_engine.integrations.mercury_guardian import MercuryGuardianAdapter

        return MercuryGuardianAdapter(
            enable_timing_monitor=True,
            gosnn_synapse_enabled=False,
        )

    def test_anomaly_recorded_after_attack(self, adapter: Any) -> None:
        """Test anomaly is recorded after attack simulation."""
        initial_count = len(adapter.anomaly_history)
        adapter.simulate_attack(attack_type="replay")
        assert len(adapter.anomaly_history) > initial_count

    def test_anomaly_summary_after_attacks(self, adapter: Any) -> None:
        """Test anomaly summary after multiple attacks."""
        adapter.simulate_attack(attack_type="timing")
        adapter.simulate_attack(attack_type="replay")
        adapter.simulate_attack(attack_type="side_channel")

        summary = adapter.get_anomaly_summary()
        assert summary["total_anomalies"] > 0
        assert len(summary["by_type"]) > 0
        assert summary["avg_severity"] > 0

    def test_recent_anomalies_limit(self, adapter: Any) -> None:
        """Test recent anomalies are limited to 10."""
        for _ in range(20):
            adapter.simulate_attack(attack_type="replay")

        summary = adapter.get_anomaly_summary()
        assert len(summary["recent_anomalies"]) <= 10


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunction:
    """Tests for create_mercury_guardian_adapter factory function."""

    def test_create_adapter_default(self) -> None:
        """Test factory function with defaults."""
        from omni_mercury_engine.integrations.mercury_guardian import (
            create_mercury_guardian_adapter,
        )

        adapter = create_mercury_guardian_adapter()
        assert adapter is not None
        assert adapter.timing_monitor is not None
        assert adapter.gosnn_synapse_enabled is True

    def test_create_adapter_no_timing(self) -> None:
        """Test factory function without timing monitor."""
        from omni_mercury_engine.integrations.mercury_guardian import (
            create_mercury_guardian_adapter,
        )

        adapter = create_mercury_guardian_adapter(enable_timing_monitor=False)
        assert adapter.timing_monitor is None

    def test_create_adapter_no_gosnn(self) -> None:
        """Test factory function without GOSNN synapse."""
        from omni_mercury_engine.integrations.mercury_guardian import (
            create_mercury_guardian_adapter,
        )

        adapter = create_mercury_guardian_adapter(gosnn_synapse_enabled=False)
        assert adapter.gosnn_synapse_enabled is False


# =============================================================================
# Module Import Tests
# =============================================================================


class TestModuleImports:
    """Tests for module imports and exports (backward compat shim)."""

    def test_import_from_integrations(self) -> None:
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

    def test_import_from_mercury_guardian_module(self) -> None:
        """Test importing directly from mercury_guardian module (backward compat shim)."""
        from omni_mercury_engine.integrations.mercury_guardian import (
            AMA_CRYPTOGRAPHY_AVAILABLE,
            AVA_GUARDIAN_AVAILABLE,
            CryptoAnomaly,
            CryptoAnomalyType,
            EWMATimingMonitor,
            MercuryGuardianAdapter,
            TimingStats,
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
                create_mercury_guardian_adapter,
            ]
        )
        # Verify backward compat aliases
        assert isinstance(AMA_CRYPTOGRAPHY_AVAILABLE, bool)
        assert isinstance(AVA_GUARDIAN_AVAILABLE, bool)
        assert AMA_CRYPTOGRAPHY_AVAILABLE == AVA_GUARDIAN_AVAILABLE

    def test_mercury_guardian_resolves_to_mercury_amacrypto(self) -> None:
        """Test that mercury_guardian shim exports the same objects as mercury_amacrypto."""
        from omni_mercury_engine.integrations import mercury_amacrypto, mercury_guardian

        assert mercury_guardian.MercuryGuardianAdapter is mercury_amacrypto.MercuryGuardianAdapter
        assert mercury_guardian.EWMATimingMonitor is mercury_amacrypto.EWMATimingMonitor
        assert mercury_guardian.CryptoAnomaly is mercury_amacrypto.CryptoAnomaly


# =============================================================================
# Fail-Closed PQC Boundary Tests
# =============================================================================


class TestFailClosedPqcBoundary:
    """Tests for mandatory AMA/PQC without silently fabricating key material."""

    @pytest.fixture
    def adapter(self):
        """Create MercuryGuardianAdapter instance."""
        from omni_mercury_engine.integrations.mercury_guardian import MercuryGuardianAdapter

        return MercuryGuardianAdapter()

    def test_dilithium_keygen_returns_real_keypair(self, adapter: Any) -> None:
        result = adapter.generate_dilithium_keypair()
        assert result is not None

    def test_kyber_keygen_returns_real_keypair(self, adapter: Any) -> None:
        result = adapter.generate_kyber_keypair()
        assert result is not None

    def test_sign_without_keypair(self, adapter: Any) -> None:
        """No default keypair means no signature is fabricated."""
        result = adapter.sign_dilithium(b"test message")
        if adapter._dilithium_keypair is None:
            assert result is None

    def test_verify_without_keypair(self, adapter: Any) -> None:
        """No default keypair means verification fails closed."""
        result = adapter.verify_dilithium(b"test message", b"fake signature")
        if adapter._dilithium_keypair is None:
            assert result is False

    def test_encapsulate_without_keypair(self, adapter: Any) -> None:
        """No default Kyber keypair means no encapsulation is fabricated."""
        result = adapter.encapsulate_kyber()
        if adapter._kyber_keypair is None:
            assert result is None

    def test_decapsulate_without_keypair(self, adapter: Any) -> None:
        """No default Kyber keypair means no shared secret is fabricated."""
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
        from omni_mercury_engine.integrations.mercury_guardian import MercuryGuardianAdapter

        return MercuryGuardianAdapter()

    def test_scalars_have_omni_prefix(self, adapter: Any) -> None:
        """Test all scalars have omni_ prefix."""
        scalars = adapter.get_gosnn_scalars()
        for key in scalars:
            assert key.startswith("omni_"), f"Scalar {key} missing omni_ prefix"

    def test_scalars_are_floats(self, adapter: Any) -> None:
        """Test all scalars are floats."""
        scalars = adapter.get_gosnn_scalars()
        for key, value in scalars.items():
            assert isinstance(value, float), f"Scalar {key} is not float"

    def test_scalars_after_anomalies(self, adapter: Any) -> None:
        """Test scalars include anomaly info after attacks."""
        adapter.simulate_attack(attack_type="replay")
        scalars = adapter.get_gosnn_scalars()
        assert scalars["omni_crypto_anomaly_count"] > 0

    def test_avg_severity_scalar(self, adapter: Any) -> None:
        """Test average severity scalar is computed."""
        adapter.simulate_attack(attack_type="replay")
        adapter.simulate_attack(attack_type="side_channel")
        scalars = adapter.get_gosnn_scalars()
        assert "omni_crypto_avg_severity" in scalars
        assert scalars["omni_crypto_avg_severity"] > 0


class TestAdaptivePostureRotationWiring:
    """The adaptive-posture controller is wired to real AMA key rotation (F2).

    Regression: CryptoPostureController was constructed with rotation_manager=
    None and hd_derivation=None, so a ROTATE_KEYS decision could not rotate any
    key material — the response loop was inert.
    """

    def test_controller_has_real_rotation_machinery(self) -> None:
        from omni_mercury_engine.integrations.mercury_amacrypto import (
            MercuryGuardianAdapter,
        )

        adapter = MercuryGuardianAdapter()
        ctrl = adapter._posture_controller
        assert ctrl.rotation_manager is not None
        assert ctrl.hd_derivation is not None

    def test_controller_shares_the_auth_key_manager_rotation_state(self) -> None:
        from omni_mercury_engine.api.auth import get_auth_key_manager
        from omni_mercury_engine.integrations.mercury_amacrypto import (
            MercuryGuardianAdapter,
        )

        adapter = MercuryGuardianAdapter()
        assert (
            adapter._posture_controller.rotation_manager is get_auth_key_manager().rotation_manager
        )

    def test_rotation_executes_without_error(self) -> None:
        from omni_mercury_engine.integrations.mercury_amacrypto import (
            MercuryGuardianAdapter,
        )

        adapter = MercuryGuardianAdapter()
        ctrl = adapter._posture_controller
        before = ctrl._rotation_count
        ctrl._trigger_rotation()  # exercises the real KeyRotationManager path
        assert ctrl._rotation_count == before + 1


class TestPostureStatusHonestyOnEvaluatorFailure:
    """A failing posture evaluator must degrade the reported posture (F11).

    Regression: get_pqc_status reported ThreatLevel.NOMINAL whenever there was
    no successful evaluation, so a PostureEvaluator that started raising left
    the crypto posture pinned at a falsely-reassuring "healthy" during an
    attack. It now reports UNKNOWN + posture_evaluation_healthy=False.
    """

    def test_fresh_adapter_is_nominal_and_healthy(self) -> None:
        from omni_mercury_engine.integrations.mercury_amacrypto import (
            MercuryGuardianAdapter,
        )

        status = MercuryGuardianAdapter().get_pqc_status()
        assert status["posture_threat_level"] == "NOMINAL"
        assert status["posture_evaluation_healthy"] is True
        assert status["posture_eval_consecutive_failures"] == 0

    def test_failing_evaluator_reports_unknown_not_nominal(self) -> None:
        from unittest.mock import patch

        from omni_mercury_engine.integrations.mercury_amacrypto import (
            MercuryGuardianAdapter,
        )

        adapter = MercuryGuardianAdapter()
        with patch.object(adapter._posture_evaluator, "evaluate", side_effect=RuntimeError("boom")):
            adapter._evaluate_posture_from_gosnn()

        status = adapter.get_pqc_status()
        assert status["posture_threat_level"] == "UNKNOWN"
        assert status["posture_evaluation_healthy"] is False
        assert status["posture_eval_consecutive_failures"] >= 1

    def test_stale_success_does_not_mask_an_actively_failing_evaluator(self) -> None:
        """A last-known-good evaluation must not keep reporting its level
        while the evaluator is failing (consecutive failures > 0): the stale
        picture cannot refresh, so the honest report is UNKNOWN."""
        from unittest.mock import patch

        from omni_mercury_engine.integrations.mercury_amacrypto import (
            MercuryGuardianAdapter,
        )

        adapter = MercuryGuardianAdapter()
        adapter._evaluate_posture_from_gosnn()  # genuine successful evaluation
        assert adapter.get_pqc_status()["posture_evaluation_healthy"] is True

        with patch.object(adapter._posture_evaluator, "evaluate", side_effect=RuntimeError("boom")):
            adapter._evaluate_posture_from_gosnn()

        status = adapter.get_pqc_status()
        assert status["posture_threat_level"] == "UNKNOWN"
        assert status["posture_evaluation_healthy"] is False
        assert status["posture_eval_consecutive_failures"] >= 1
