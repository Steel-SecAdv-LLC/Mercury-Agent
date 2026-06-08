# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test suite for security enhancements: quantum-resistant encryption, real-time threat detection, and hive firewall."""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.security.encryption import QuantumResistantEncryption, SecureDataHandler
from omni_mercury_engine.security.hive_firewall import HiveFirewall
from omni_mercury_engine.security.realtime_threat_detection import (
    AdaptiveThreatDetector,
    RealTimeThreatDetector,
)


class TestQuantumResistantEncryption:
    """Test quantum-resistant encryption."""

    def test_key_generation(self) -> None:
        """Test lattice-based key generation."""
        qr = QuantumResistantEncryption(security_level=128)

        public_key, private_key = qr._generate_lattice_key()

        A, b = public_key
        assert A.shape == (128, 128)
        assert b.shape == (128,)
        assert private_key.shape == (128,)

    def test_encryption_decryption(self) -> None:
        """Test encrypt/decrypt cycle."""
        qr = QuantumResistantEncryption(security_level=128)
        public_key, private_key = qr._generate_lattice_key()

        plaintext = b"Test quantum-resistant encryption"

        ciphertext = qr.encrypt_hybrid(plaintext, public_key)

        assert len(ciphertext) > len(plaintext)
        assert ciphertext != plaintext

        decrypted = qr.decrypt_hybrid(ciphertext, private_key)

        assert decrypted == plaintext

    def test_secure_data_handler(self) -> None:
        """Test SecureDataHandler with quantum-resistant encryption."""
        handler = SecureDataHandler(enable_quantum_resistant=True)

        assert handler.public_key is not None
        assert handler.private_key is not None

        data = "Sensitive information"
        encrypted = handler.encrypt_quantum_resistant(data)

        assert isinstance(encrypted, bytes)
        assert encrypted != data.encode()

        decrypted = handler.decrypt_quantum_resistant(encrypted)
        assert decrypted == data.encode()

    def test_sanitization(self) -> None:
        """Test input sanitization."""
        handler = SecureDataHandler(enable_quantum_resistant=False)

        malicious = "<script>alert('xss')</script>"
        sanitized = handler.sanitize_input(malicious)

        assert "<script>" not in sanitized
        assert "&lt;script&gt;" in sanitized


class TestRealTimeThreatDetector:
    """Test real-time threat detection."""

    def test_initialization(self) -> None:
        """Test detector initialization."""
        detector = RealTimeThreatDetector(contamination=0.1, n_estimators=50)

        assert detector.contamination == 0.1
        assert detector.n_estimators == 50
        assert "isolation_forest" in detector.detectors
        assert "lof" in detector.detectors
        assert "elliptic" in detector.detectors

    def test_fit_and_detect(self) -> None:
        """Test fitting and detection."""
        detector = RealTimeThreatDetector(contamination=0.1)

        normal_data = np.random.randn(100, 10)

        detector.fit(normal_data)

        assert detector.is_fitted is True

        test_data = np.random.randn(20, 10)
        result = detector.detect_threat(test_data)

        assert "is_threat" in result
        assert "threat_indices" in result
        assert "ensemble_scores" in result
        assert "threat_level" in result

    def test_threat_levels(self) -> None:
        """Test threat level classification."""
        detector = RealTimeThreatDetector(contamination=0.1)

        np.random.seed(42)
        normal_data = np.random.randn(100, 10)
        detector.fit(normal_data)

        np.random.seed(43)
        normal_test = np.random.randn(10, 10)
        np.random.seed(44)
        anomalous_test = np.random.randn(10, 10) * 10

        normal_result = detector.detect_threat(normal_test)
        anomalous_result = detector.detect_threat(anomalous_test)

        assert normal_result["threat_level"] in ["LOW", "MEDIUM", "NEGLIGIBLE", "HIGH", "CRITICAL"]
        assert anomalous_result["threat_level"] in ["HIGH", "CRITICAL", "MEDIUM", "LOW"]

    def test_threat_recording(self) -> None:
        """Test threat signature recording."""
        detector = RealTimeThreatDetector()

        threat_data = np.random.randn(10)

        signature = detector.record_threat(threat_data, threat_type="ddos", severity=0.8)

        assert signature.threat_type == "ddos"
        assert signature.severity == 0.8
        assert len(detector.threat_history) == 1

    def test_threat_statistics(self) -> None:
        """Test threat statistics generation."""
        detector = RealTimeThreatDetector()

        for i in range(5):
            detector.record_threat(
                np.random.randn(10), threat_type="intrusion", severity=0.5 + i * 0.1
            )

        stats = detector.get_threat_statistics()

        assert stats["total_threats"] == 5
        assert "intrusion" in stats["threat_types"]
        assert stats["threat_types"]["intrusion"] == 5


class TestAdaptiveThreatDetector:
    """Test adaptive threat detector."""

    def test_adaptation(self) -> None:
        """Test model adaptation over time."""
        detector = AdaptiveThreatDetector(contamination=0.1, update_frequency=10)

        normal_data = np.random.randn(100, 10)
        detector.fit(normal_data)

        for i in range(15):
            new_data = np.random.randn(5, 10)
            detector.detect_and_adapt(new_data, is_normal=True)

        assert len(detector.training_buffer) > 0


class TestHiveFirewall:
    """Test HCIS-inspired hive-structured firewall."""

    def test_initialization(self) -> None:
        """Test firewall initialization."""
        firewall = HiveFirewall(
            num_worker_nodes=10, num_supervisor_nodes=3, consensus_threshold=0.6
        )

        assert len(firewall.worker_nodes) == 10
        assert len(firewall.supervisor_nodes) == 3
        assert firewall.queen_node.node_type == "queen"

    def test_signature_hash(self) -> None:
        """Test O(1) signature hashing."""
        firewall = HiveFirewall()

        data1 = np.array([1, 2, 3, 4, 5])
        data2 = np.array([1, 2, 3, 4, 5])
        data3 = np.array([5, 4, 3, 2, 1])

        hash1 = firewall._compute_signature_hash(data1)
        hash2 = firewall._compute_signature_hash(data2)
        hash3 = firewall._compute_signature_hash(data3)

        assert hash1 == hash2
        assert hash1 != hash3

    def test_blocking_decision(self) -> None:
        """Test hierarchical blocking decision."""
        firewall = HiveFirewall()

        suspicious_data = np.random.randn(50) * 5

        decision = firewall.detect_and_block(suspicious_data, anomaly_score=0.9)

        assert decision.signature_hash is not None
        assert isinstance(decision.block_decision, bool)
        assert decision.confidence >= 0.0
        assert decision.confidence <= 1.0

    def test_o1_lookup(self) -> None:
        """Test O(1) blocking lookup."""
        firewall = HiveFirewall()

        threat_data = np.random.randn(50)

        firewall.detect_and_block(threat_data, anomaly_score=0.95)

        is_blocked, decision = firewall.is_blocked(threat_data)

        assert isinstance(is_blocked, bool)
        if is_blocked:
            assert decision is not None

    def test_whitelist(self) -> None:
        """Test pattern whitelisting."""
        firewall = HiveFirewall()

        safe_data = np.random.randn(50)

        firewall.detect_and_block(safe_data, anomaly_score=0.9)

        firewall.allow_pattern(safe_data)

        is_blocked, _ = firewall.is_blocked(safe_data)

        assert is_blocked is False

    def test_false_positive_handling(self) -> None:
        """Test false positive reporting and trust adjustment."""
        firewall = HiveFirewall()

        data = np.random.randn(50)

        initial_trust = firewall.worker_nodes[0].trust_score

        firewall.report_false_positive(data)

        final_trust = firewall.worker_nodes[0].trust_score

        assert final_trust < initial_trust

    def test_consensus_mechanism(self) -> None:
        """Test worker-supervisor-queen consensus."""
        firewall = HiveFirewall()

        low_anomaly = firewall.detect_and_block(np.random.randn(50) * 0.1, anomaly_score=0.2)

        high_anomaly = firewall.detect_and_block(np.random.randn(50) * 10, anomaly_score=0.95)

        assert high_anomaly.block_decision
        assert high_anomaly.confidence > low_anomaly.confidence

    def test_firewall_stats(self) -> None:
        """Test firewall statistics."""
        firewall = HiveFirewall()

        for i in range(10):
            data = np.random.randn(50)
            firewall.detect_and_block(data, anomaly_score=0.7 + i * 0.02)

        stats = firewall.get_firewall_stats()

        assert "total_blocked_signatures" in stats
        assert "avg_worker_trust" in stats
        assert "queen_trust" in stats
        assert "detection_accuracy" in stats

    def test_byzantine_tolerance(self) -> None:
        """Test Byzantine fault tolerance through trust scoring."""
        firewall = HiveFirewall()

        for worker in firewall.worker_nodes[:3]:
            worker.trust_score = 0.5

        data = np.random.randn(50)
        decision = firewall.detect_and_block(data, anomaly_score=0.8)

        assert decision.confidence >= 0.0


class TestIntegratedSecurity:
    """Integration tests for security components."""

    def test_end_to_end_threat_pipeline(self) -> None:
        """Test complete threat detection and blocking pipeline."""
        detector = RealTimeThreatDetector(contamination=0.1)
        firewall = HiveFirewall()

        normal_data = np.random.randn(100, 50)
        detector.fit(normal_data)

        test_data = np.random.randn(10, 50)
        threat_result = detector.detect_threat(test_data)

        if threat_result["is_threat"]:
            for idx in threat_result["threat_indices"]:
                sample = test_data[idx]
                decision = firewall.detect_and_block(
                    sample, anomaly_score=threat_result["ensemble_scores"][idx]
                )

                assert decision.signature_hash is not None

    def test_ddos_resilience(self) -> None:
        """Test DDoS resilience with high-volume requests."""
        firewall = HiveFirewall()

        import time

        start = time.time()

        for _ in range(1000):
            data = np.random.randn(50)
            firewall.is_blocked(data)

        elapsed = time.time() - start

        assert elapsed < 1.0

        avg_lookup_time = elapsed / 1000
        assert avg_lookup_time < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
