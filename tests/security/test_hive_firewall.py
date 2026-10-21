# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the hive-structured firewall.

The first two classes are the released tests (restored with the module).
``TestFeedbackDirection`` pins the defects that got the module removed: a
reported false positive lowered every threshold (a 0.68 score allowed before
the report was blocked after it), and identical workers reduced the whole
hierarchy to ``score > 0.7`` with a confidence of only 0 or 1.

The last three classes pin defects found in the restored module: an
allowlisted pattern was voted (and blocked) again, repeated feedback for one
decision compounded, operator blocks drove ``detection_accuracy`` negative,
``report_missed_threat`` took NaN scores, the float64 signature refused
strings and collided large integers and complex values, trust decayed to 0.0
(NaN shares), and the constructor refused released ``trust_decay=1.0``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.security import hive_firewall
from omni_mercury_engine.security.hive_firewall import HiveFirewall
from omni_mercury_engine.security.realtime_threat_detection import RealTimeThreatDetector


def _node_state(firewall: HiveFirewall) -> tuple[list[float], list[float], list[int]]:
    """Worker thresholds, worker and supervisor trust, worker false-positive counts."""
    return (
        [w.threshold for w in firewall.worker_nodes],
        [n.trust_score for n in firewall.worker_nodes + firewall.supervisor_nodes],
        [w.false_positive_count for w in firewall.worker_nodes],
    )


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


class TestAllowlistAndFeedback:
    """An allowlisted pattern is not voted again; one decision gets feedback once."""

    def test_allowlisted_pattern_is_allowed_without_a_vote(self) -> None:
        firewall = HiveFirewall()
        reported, allowed = np.arange(8.0), np.ones(8)
        assert firewall.detect_and_block(reported, anomaly_score=0.95).block_decision is True
        firewall.report_false_positive(reported)
        firewall.allow_pattern(allowed)
        detections = sum(w.detection_count for w in firewall.worker_nodes)

        for data in (reported, allowed):
            decision = firewall.detect_and_block(data, anomaly_score=0.95)
            assert decision.block_decision is False
            assert (decision.confidence, decision.consensus_votes) == (0.0, 0)
            assert "allowlisted" in decision.reasoning.lower()
            assert firewall.is_blocked(data) == (False, None)

        assert sum(w.detection_count for w in firewall.worker_nodes) == detections
        assert firewall.get_firewall_stats()["total_blocked_signatures"] == 0
        firewall.report_missed_threat(allowed, anomaly_score=0.95)
        assert firewall.detect_and_block(allowed, anomaly_score=0.95).block_decision is True

    @pytest.mark.parametrize("feedback", ["false_positive", "missed_decided", "missed_undecided"])
    def test_repeated_feedback_for_one_decision_applies_once(self, feedback: str) -> None:
        data = np.arange(8.0)

        def state_after(reports: int) -> tuple[list[float], list[float], list[int]]:
            firewall = HiveFirewall()
            if feedback == "false_positive":
                firewall.detect_and_block(data, anomaly_score=0.9)
            elif feedback == "missed_decided":
                firewall.detect_and_block(data, anomaly_score=0.6)
            for _ in range(reports):
                if feedback == "false_positive":
                    firewall.report_false_positive(data)
                else:
                    firewall.report_missed_threat(data, anomaly_score=0.6)
            return _node_state(firewall)

        assert state_after(1) != state_after(0)
        assert state_after(25) == state_after(1)

    def test_accuracy_counts_only_the_blocks_the_hive_issued(self) -> None:
        firewall = HiveFirewall()
        a, b, c = np.zeros(4), np.ones(4), np.full(4, 2.0)
        firewall.detect_and_block(a, 0.95)
        firewall.detect_and_block(b, 0.95)
        firewall.report_false_positive(b)
        firewall.report_missed_threat(c, anomaly_score=0.5)  # the operator's block ...
        firewall.report_false_positive(c)  # ... retracted: not the hive's error
        assert firewall.get_firewall_stats()["detection_accuracy"] == pytest.approx(0.5)

        firewall.report_false_positive(a)
        firewall.report_missed_threat(b, anomaly_score=0.5)
        firewall.report_false_positive(b)
        assert firewall.get_firewall_stats()["detection_accuracy"] == 0.0

    @pytest.mark.parametrize("score", [float("nan"), -0.1, 1.5])
    def test_missed_threat_score_outside_unit_interval_is_refused(self, score: float) -> None:
        firewall = HiveFirewall()
        before = _node_state(firewall)
        with pytest.raises(ValueError, match="anomaly_score"):
            firewall.report_missed_threat(np.zeros(4), anomaly_score=score)
        assert _node_state(firewall) == before
        assert firewall.is_blocked(np.zeros(4))[0] is False


@pytest.mark.filterwarnings("error")
class TestSignature:
    """Exact numeric values share a signature; nothing else is cast to reach one."""

    def test_strings_and_bytes_are_hashed(self) -> None:
        firewall = HiveFirewall()
        sig = firewall._compute_signature_hash
        assert sig(np.array(["GET /admin"])) == sig(np.array(["GET /admin"]))
        assert sig(np.array(["GET /admin"])) != sig(np.array(["GET /index"]))
        assert sig(np.array([b"GET /admin"])) != sig(np.array([b"GET /index"]))
        request = np.array([b"DROP TABLE users"])
        assert firewall.detect_and_block(request, anomaly_score=0.95).block_decision is True
        assert firewall.is_blocked(np.array([b"DROP TABLE users"]))[0] is True

    def test_integers_float64_would_round_do_not_collide(self) -> None:
        sig = HiveFirewall()._compute_signature_hash
        stamp = np.array([1_760_000_000_000_000_000, 7], dtype=np.int64)
        later = stamp.copy()
        later[0] += 100
        assert sig(stamp) != sig(later)
        assert sig(np.array([2**63 - 1], dtype=np.int64)) != sig(np.array([2.0**63]))
        assert sig(np.array([2**64 - 1], dtype=np.uint64)) != sig(np.array([2.0**64]))
        # A large integer float64 holds exactly still matches its float.
        assert sig(np.array([2**60], dtype=np.int64)) == sig(np.array([2.0**60]))

    def test_complex_values_keep_their_imaginary_part(self) -> None:
        sig = HiveFirewall()._compute_signature_hash
        assert sig(np.array([1 + 2j, 3 + 4j])) != sig(np.array([1 - 9j, 3 + 0j]))
        assert sig(np.array([1 + 2j, 3 + 4j])) == sig(np.array([1 + 2j, 3 + 4j]))

    def test_signature_ignores_byte_order_and_memory_layout(self) -> None:
        sig = HiveFirewall()._compute_signature_hash
        big = np.array([2**62 + 1, 3], dtype=">i8")
        assert sig(big) == sig(big.astype("<i8")) != sig(np.array([2**62, 3], dtype="<i8"))
        assert sig(np.array(["ab"], dtype=">U2")) == sig(np.array(["ab"], dtype="<U2"))
        grid = np.arange(6.0).reshape(2, 3)
        assert sig(grid) == sig(np.asfortranarray(grid))

    def test_object_array_of_real_numbers_shares_the_float_signature(self) -> None:
        sig = HiveFirewall()._compute_signature_hash
        mixed = np.array([1, 2.5, True, np.int32(-4), np.float32(0.5), 2**60], dtype=object)
        assert sig(mixed) == sig(np.array([1.0, 2.5, 1.0, -4.0, 0.5, 2.0**60]))

    @pytest.mark.parametrize(
        "items",
        [
            [1.5, "x"],
            [1 + 2j],
            [2**53 + 1],
            [np.uint64(2**64 - 1)],
            [2**1100],
            [np.timedelta64(5, "s")],
        ],
    )
    def test_other_object_arrays_are_refused_with_the_reason(self, items: list[Any]) -> None:
        with pytest.raises(TypeError, match="memory addresses"):
            HiveFirewall()._compute_signature_hash(np.array(items, dtype=object))


class TestTrustAndParameters:
    """Trust stays positive; every parameter the hive can honour is accepted."""

    def test_trust_never_reaches_zero(self) -> None:
        firewall = HiveFirewall(trust_decay=0.01)
        for k in range(200):
            data = np.full(4, float(k))
            firewall.detect_and_block(data, anomaly_score=1.0)
            firewall.report_false_positive(data)
        for _ in range(200):
            firewall.update_node_trust("queen", success=False)
        nodes = firewall.worker_nodes + firewall.supervisor_nodes + [firewall.queen_node]
        assert min(n.trust_score for n in nodes) > 0.0

        decision = firewall.detect_and_block(np.full(4, -1.0), anomaly_score=1.0)
        assert decision.block_decision is True
        assert decision.confidence == pytest.approx(1.0)
        assert min(n.trust_score for n in nodes) == hive_firewall.TRUST_FLOOR

    def test_trust_decay_of_one_keeps_trust_fixed(self) -> None:
        firewall = HiveFirewall(trust_decay=1.0)
        data = np.arange(4.0)
        assert firewall.detect_and_block(data, anomaly_score=0.95).block_decision is True
        before = [w.threshold for w in firewall.worker_nodes]
        firewall.report_false_positive(data)
        firewall.update_node_trust("worker_0", success=False)
        nodes = firewall.worker_nodes + firewall.supervisor_nodes + [firewall.queen_node]
        assert all(n.trust_score == 1.0 for n in nodes)
        after = [w.threshold for w in firewall.worker_nodes]
        assert all(a > b for a, b in zip(after, before, strict=True))

    def test_consensus_threshold_zero_needs_one_block_vote_per_supervisor(self) -> None:
        firewall = HiveFirewall(consensus_threshold=0.0)
        assert firewall.detect_and_block(np.zeros(4), anomaly_score=0.0).block_decision is False
        assert firewall.detect_and_block(np.ones(4), anomaly_score=0.5).block_decision is False
        # Workers 0-2 hold the three lowest thresholds, one in each supervisor's group.
        score = firewall.worker_nodes[2].threshold
        decision = firewall.detect_and_block(np.full(4, 2.0), anomaly_score=score)
        assert (decision.block_decision, decision.consensus_votes) == (True, 3)
        assert HiveFirewall().detect_and_block(np.full(4, 2.0), score).block_decision is False

    @pytest.mark.parametrize(
        ("kwargs", "reason"),
        [
            ({"num_worker_nodes": 2, "num_supervisor_nodes": 3}, "own group of workers"),
            ({"consensus_threshold": -0.1}, "share of a supervisor's workers"),
            ({"consensus_threshold": 1.1}, "share of a supervisor's workers"),
            ({"consensus_threshold": float("nan")}, "share of a supervisor's workers"),
            ({"trust_decay": 0.0}, "divided by it"),
            ({"trust_decay": 1.5}, "divided by it"),
            ({"trust_decay": float("nan")}, "divided by it"),
        ],
    )
    def test_parameters_the_hive_cannot_honour_are_refused_with_the_reason(
        self, kwargs: dict[str, Any], reason: str
    ) -> None:
        with pytest.raises(ValueError, match=reason):
            HiveFirewall(**kwargs)
