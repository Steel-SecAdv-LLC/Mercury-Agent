# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the hive-structured firewall.

The first two classes are the released tests (restored with the module).
``TestFeedbackDirection`` pins the defects that got the module removed: a
reported false positive lowered every threshold (a 0.68 score allowed before
the report was blocked after it), and identical workers reduced the whole
hierarchy to ``score > 0.7`` with a confidence of only 0 or 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.security.hive_firewall import HiveFirewall
from omni_mercury_engine.security.realtime_threat_detection import RealTimeThreatDetector


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

        # Feedback reaches the nodes that voted: decide first. (Penalising
        # every node for a pattern the hive never decided was the defect.)
        firewall.detect_and_block(data, anomaly_score=0.95)
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
                # The firewall scores on [0, 1]: the detector's probability,
                # not its standardized ensemble score (4.5 here).
                decision = firewall.detect_and_block(
                    sample, anomaly_score=threat_result["anomaly_probabilities"][idx]
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


class TestFeedbackDirection:
    """A false positive makes blocking less likely; a missed threat more likely."""

    def test_false_positive_never_makes_a_borderline_score_blocked(self) -> None:
        firewall = HiveFirewall()
        borderline = np.arange(50.0)
        assert firewall.detect_and_block(borderline, anomaly_score=0.68).block_decision is False
        blocked = np.arange(50.0) + 100.0
        assert firewall.detect_and_block(blocked, anomaly_score=0.9).block_decision is True
        before = [w.threshold for w in firewall.worker_nodes]

        firewall.report_false_positive(blocked)

        after = [w.threshold for w in firewall.worker_nodes]
        assert all(a >= b for a, b in zip(after, before, strict=True))
        assert firewall.detect_and_block(np.arange(50.0) + 7.0, 0.68).block_decision is False
        assert firewall.is_blocked(blocked)[0] is False

    def test_missed_threat_lowers_the_thresholds_that_allowed_it(self) -> None:
        firewall = HiveFirewall()
        sample = np.arange(50.0) * 0.5
        assert firewall.detect_and_block(sample, anomaly_score=0.7).block_decision is False
        for k in range(3):
            firewall.report_missed_threat(np.arange(50.0) + 1000.0 * (k + 1), anomaly_score=0.7)
        assert firewall.is_blocked(np.arange(50.0) + 1000.0)[0] is True
        assert firewall.detect_and_block(np.arange(50.0) - 5.0, 0.7).block_decision is True

    def test_consensus_is_graded_in_the_score(self) -> None:
        firewall = HiveFirewall()
        confidences = [
            firewall.detect_and_block(np.full(8, s), anomaly_score=s).confidence
            for s in (0.5, 0.6, 0.7, 0.8, 0.9)
        ]
        assert confidences == sorted(confidences)
        assert any(0.0 < c < 1.0 for c in confidences)
        assert confidences[0] == 0.0 and confidences[-1] == 1.0

    def test_thresholds_stay_bounded(self) -> None:
        firewall = HiveFirewall()
        for k in range(200):
            data = np.full(4, float(k))
            firewall.detect_and_block(data, anomaly_score=1.0)
            firewall.report_false_positive(data)
        assert max(w.threshold for w in firewall.worker_nodes) <= 0.99
        assert min(w.trust_score for w in firewall.worker_nodes) > 0.0

    def test_signature_ignores_dtype(self) -> None:
        firewall = HiveFirewall()
        assert firewall._compute_signature_hash(np.array([1, 2, 3])) == (
            firewall._compute_signature_hash(np.array([1.0, 2.0, 3.0]))
        )

    @pytest.mark.parametrize("score", [float("nan"), -0.1, 1.5])
    def test_score_outside_unit_interval_is_refused(self, score: float) -> None:
        with pytest.raises(ValueError, match="anomaly_score"):
            HiveFirewall().detect_and_block(np.zeros(4), anomaly_score=score)

    def test_stats_accuracy_counts_reported_false_blocks(self) -> None:
        firewall = HiveFirewall()
        a, b = np.zeros(4), np.ones(4)
        firewall.detect_and_block(a, 0.95)
        firewall.detect_and_block(b, 0.95)
        firewall.report_false_positive(b)
        assert firewall.get_firewall_stats()["detection_accuracy"] == pytest.approx(0.5)
