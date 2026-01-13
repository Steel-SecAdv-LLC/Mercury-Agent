"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisory LLC

Comprehensive tests for resilience/self_healing.py module.
Targets coverage improvement for AdaptiveDefenseSystem and SelfHealingEngine.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omni_mercury_engine.resilience.self_healing import (
    AdaptiveDefenseSystem,
    AnomalySignature,
    SelfHealingEngine,
)


class TestAnomalySignature:
    """Tests for AnomalySignature dataclass."""

    def test_signature_creation(self):
        """Test creating anomaly signature."""
        feature_vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sig = AnomalySignature(
            signature_id="test_001",
            feature_vector=feature_vec,
            timestamp=1234567890.0,
            detection_count=5,
            confidence=0.95,
            metadata={"source": "test"},
        )

        assert sig.signature_id == "test_001"
        assert len(sig.feature_vector) == 5
        assert sig.detection_count == 5
        assert sig.confidence == 0.95

    def test_signature_defaults(self):
        """Test default values for signature."""
        sig = AnomalySignature(
            signature_id="test",
            feature_vector=np.array([1.0]),
            timestamp=0.0,
        )

        assert sig.detection_count == 0
        assert sig.confidence == 0.0
        assert sig.metadata == {}


class TestAdaptiveDefenseSystem:
    """Tests for AdaptiveDefenseSystem (CRISPR-inspired)."""

    def test_init(self):
        """Test system initialization."""
        system = AdaptiveDefenseSystem(max_signatures=500, similarity_threshold=0.9)
        assert system.max_signatures == 500
        assert system.similarity_threshold == 0.9
        assert len(system.signature_library) == 0

    def test_stage_1_acquisition(self):
        """Test Stage 1: Acquisition of anomaly signature."""
        system = AdaptiveDefenseSystem()
        data = np.random.randn(100)

        signature = system.stage_1_acquisition(data, metadata={"type": "test"})

        assert signature.signature_id in system.signature_library
        assert signature.detection_count == 1
        assert signature.confidence == 0.95
        assert signature.metadata["type"] == "test"

    def test_stage_2_expression(self):
        """Test Stage 2: Expression of signature to detection pattern."""
        system = AdaptiveDefenseSystem()
        data = np.random.randn(100)
        signature = system.stage_1_acquisition(data)

        pattern = system.stage_2_expression(signature)

        # Pattern should be normalized (unit vector)
        norm = np.linalg.norm(pattern)
        assert abs(norm - 1.0) < 1e-6

    def test_stage_2_zero_vector(self):
        """Test Stage 2 with zero vector."""
        system = AdaptiveDefenseSystem()

        # Create signature with zero vector
        sig = AnomalySignature(
            signature_id="zero",
            feature_vector=np.zeros(7),
            timestamp=0.0,
        )

        pattern = system.stage_2_expression(sig)
        assert np.allclose(pattern, np.zeros(7))

    def test_stage_3_interference_no_library(self):
        """Test Stage 3: Interference with empty library."""
        system = AdaptiveDefenseSystem()
        data = np.random.randn(100)

        is_anomaly, confidence, sig_id = system.stage_3_interference(data)

        assert is_anomaly is False
        assert confidence == 0.0
        assert sig_id is None

    def test_stage_3_interference_with_match(self):
        """Test Stage 3: Interference with matching pattern."""
        system = AdaptiveDefenseSystem(similarity_threshold=0.5)

        # Acquire a signature
        original_data = np.random.randn(100)
        system.stage_1_acquisition(original_data)

        # Test with similar data
        is_anomaly, confidence, sig_id = system.stage_3_interference(original_data)

        # Should match itself
        assert is_anomaly is True
        assert confidence > 0.5
        assert sig_id is not None

    def test_library_pruning(self):
        """Test automatic pruning when library is full."""
        system = AdaptiveDefenseSystem(max_signatures=3)

        # Add 4 signatures (exceeds max)
        for i in range(4):
            data = np.random.randn(100) * (i + 1)  # Different data
            system.stage_1_acquisition(data, metadata={"index": i})

        # Library should be at max size
        assert len(system.signature_library) == 3

    def test_save_load_library(self):
        """Test saving and loading signature library."""
        system = AdaptiveDefenseSystem()

        # Add some signatures
        for i in range(3):
            data = np.random.randn(100) * (i + 1)
            system.stage_1_acquisition(data, metadata={"index": i})

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "signatures.json"
            system.save_library(str(filepath))

            assert filepath.exists()

            # Load into new system
            new_system = AdaptiveDefenseSystem()
            new_system.load_library(str(filepath))

            assert len(new_system.signature_library) == 3

    def test_online_learning_disabled(self):
        """Test that online learning is disabled by default."""
        system = AdaptiveDefenseSystem(enable_online_learning=False)
        data = np.random.randn(100)

        # Should do nothing when disabled
        system.update_online_statistics(data)

        assert system._sample_count == 0

    def test_online_learning_enabled(self):
        """Test online learning with incremental statistics."""
        system = AdaptiveDefenseSystem(
            enable_online_learning=True,
            sliding_window_size=10,
            forgetting_factor=0.99,
        )

        # Update with multiple samples
        for _ in range(5):
            data = np.random.randn(100)
            system.update_online_statistics(data)

        assert system._sample_count == 5
        assert system._running_mean is not None
        assert len(system._sliding_window) == 5

    def test_concept_drift_detection(self):
        """Test concept drift detection."""
        system = AdaptiveDefenseSystem(
            enable_online_learning=True,
            sliding_window_size=20,
        )

        # Add samples from one distribution
        for _ in range(10):
            data = np.random.randn(100)
            system.update_online_statistics(data)

        # Add samples from different distribution (simulate drift)
        for _ in range(15):
            data = np.random.randn(100) * 10 + 50  # Shifted distribution
            system.update_online_statistics(data)

        stats = system.get_online_learning_stats()
        assert "concept_drift_detected" in stats
        assert "drift_magnitude" in stats

    def test_adapt_signature(self):
        """Test incremental signature adaptation."""
        system = AdaptiveDefenseSystem(
            enable_online_learning=True,
            adaptation_rate=0.1,
        )

        # Create signature
        data = np.random.randn(100)
        signature = system.stage_1_acquisition(data)
        sig_id = signature.signature_id

        # Adapt with new data
        new_data = np.random.randn(100)
        result = system.adapt_signature(sig_id, new_data)

        assert result is True
        assert system.signature_library[sig_id].detection_count == 2

    def test_adapt_signature_not_found(self):
        """Test adaptation with non-existent signature."""
        system = AdaptiveDefenseSystem(enable_online_learning=True)

        result = system.adapt_signature("nonexistent", np.random.randn(100))
        assert result is False

    def test_adapt_signature_learning_disabled(self):
        """Test adaptation when online learning is disabled."""
        system = AdaptiveDefenseSystem(enable_online_learning=False)

        data = np.random.randn(100)
        signature = system.stage_1_acquisition(data)

        result = system.adapt_signature(signature.signature_id, np.random.randn(100))
        assert result is False

    def test_get_statistics_empty(self):
        """Test statistics with empty library."""
        system = AdaptiveDefenseSystem()

        stats = system.get_statistics()

        assert stats["total_signatures"] == 0
        assert stats["total_detections"] == 0
        assert stats["average_confidence"] == 0.0

    def test_get_statistics_with_signatures(self):
        """Test statistics with populated library."""
        system = AdaptiveDefenseSystem()

        for _ in range(3):
            system.stage_1_acquisition(np.random.randn(100))

        stats = system.get_statistics()

        assert stats["total_signatures"] == 3
        assert stats["total_detections"] == 3  # Each signature counted once
        assert stats["average_confidence"] > 0

    def test_get_statistics_with_online_learning(self):
        """Test statistics include online learning info."""
        system = AdaptiveDefenseSystem(enable_online_learning=True)

        for _ in range(3):
            system.update_online_statistics(np.random.randn(100))

        stats = system.get_statistics()
        assert "online_learning" in stats

    def test_backward_compat_aliases(self):
        """Test backward compatibility aliases."""
        system = AdaptiveDefenseSystem()
        system.stage_1_acquisition(np.random.randn(100))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"

            # Use deprecated aliases
            system.save_signature_library(str(filepath))
            assert filepath.exists()

            new_system = AdaptiveDefenseSystem()
            new_system.load_signature_library(str(filepath))
            assert len(new_system.signature_library) == 1


class TestSelfHealingEngine:
    """Tests for SelfHealingEngine."""

    def test_init(self):
        """Test engine initialization."""
        engine = SelfHealingEngine(max_signatures=500, similarity_threshold=0.9)

        assert len(engine.components) == 0
        assert engine.adaptive_defense is not None
        assert engine.adaptive_defense.max_signatures == 500

    def test_register_component(self):
        """Test component registration."""
        engine = SelfHealingEngine()

        def health_check():
            return True

        def recovery():
            pass

        engine.register_component("test_component", health_check, recovery)

        assert "test_component" in engine.components
        assert "test_component" in engine.circuit_breakers

    def test_check_health_healthy(self):
        """Test health check for healthy component."""
        engine = SelfHealingEngine()

        def health_check():
            return True

        engine.register_component("healthy", health_check)

        result = engine.check_health("healthy")
        assert result is True
        assert engine.components["healthy"]["status"] == "healthy"

    def test_check_health_unhealthy(self):
        """Test health check for unhealthy component."""
        engine = SelfHealingEngine()

        def health_check():
            return False

        engine.register_component("unhealthy", health_check)

        result = engine.check_health("unhealthy")
        assert result is False
        assert engine.components["unhealthy"]["status"] == "unhealthy"

    def test_check_health_exception(self):
        """Test health check that raises exception."""
        engine = SelfHealingEngine()

        def health_check():
            raise RuntimeError("Health check failed")

        engine.register_component("failing", health_check)

        result = engine.check_health("failing")
        assert result is False
        assert engine.components["failing"]["status"] == "unhealthy"

    def test_check_health_unknown_component(self):
        """Test health check for unknown component."""
        engine = SelfHealingEngine()

        result = engine.check_health("unknown")
        assert result is False

    def test_attempt_recovery_success(self):
        """Test successful recovery."""
        engine = SelfHealingEngine()

        state = {"healthy": False}

        def health_check():
            return state["healthy"]

        def recovery():
            state["healthy"] = True

        engine.register_component("recoverable", health_check, recovery)

        # Initially unhealthy
        assert engine.check_health("recoverable") is False

        # Recovery should succeed
        result = engine.attempt_recovery("recoverable")
        assert result is True
        assert engine.components["recoverable"]["status"] == "healthy"

    def test_attempt_recovery_no_action(self):
        """Test recovery with no recovery action."""
        engine = SelfHealingEngine()

        def health_check():
            return False

        engine.register_component("no_recovery", health_check, recovery_action=None)

        result = engine.attempt_recovery("no_recovery")
        assert result is False

    def test_attempt_recovery_unknown_component(self):
        """Test recovery for unknown component."""
        engine = SelfHealingEngine()

        result = engine.attempt_recovery("unknown")
        assert result is False

    def test_attempt_recovery_exception(self):
        """Test recovery that raises exception."""
        engine = SelfHealingEngine()

        def health_check():
            return False

        def recovery():
            raise RuntimeError("Recovery failed")

        engine.register_component("failing_recovery", health_check, recovery)

        result = engine.attempt_recovery("failing_recovery")
        assert result is False

    def test_learn_anomaly(self):
        """Test learning new anomaly pattern."""
        engine = SelfHealingEngine()

        data = np.random.randn(100)
        signature = engine.learn_anomaly(data, metadata={"source": "test"})

        assert signature is not None
        assert signature.metadata["source"] == "test"

    def test_check_known_anomaly(self):
        """Test checking for known anomaly patterns."""
        engine = SelfHealingEngine(similarity_threshold=0.5)

        # Learn an anomaly
        data = np.random.randn(100)
        engine.learn_anomaly(data)

        # Check same data
        is_anomaly, confidence, sig_id = engine.check_known_anomaly(data)

        assert is_anomaly is True
        assert confidence > 0.5

    def test_get_system_health_all_healthy(self):
        """Test system health when all components healthy."""
        engine = SelfHealingEngine()

        def healthy_check():
            return True

        engine.register_component("comp1", healthy_check)
        engine.register_component("comp2", healthy_check)

        health = engine.get_system_health()

        assert health["overall_health"] == "healthy"
        assert health["components"]["comp1"]["is_healthy"] is True
        assert health["components"]["comp2"]["is_healthy"] is True

    def test_get_system_health_degraded(self):
        """Test system health when some components unhealthy."""
        engine = SelfHealingEngine()

        def healthy_check():
            return True

        def unhealthy_check():
            return False

        engine.register_component("healthy_comp", healthy_check)
        engine.register_component("unhealthy_comp", unhealthy_check)

        health = engine.get_system_health()

        assert health["overall_health"] == "degraded"
        assert "adaptive_defense" in health

    def test_get_system_health_empty(self):
        """Test system health with no components."""
        engine = SelfHealingEngine()

        health = engine.get_system_health()

        # No components means "healthy" (vacuous truth)
        assert health["overall_health"] == "healthy"
        assert len(health["components"]) == 0
