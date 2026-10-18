# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""HCIS-Inspired Hive-Structured Firewall for Anomaly Blocking.

Implements a hierarchical, distributed firewall inspired by HCIS (Hierarchical
Cognitive Immune System) principles with hive-like collaborative defense.

Architecture:
- Worker nodes: Front-line anomaly detection. Each worker has its own block
  threshold, spread evenly over ``base_threshold +/- threshold_spread``, so
  the hive's consensus is graded in the anomaly score rather than a single
  cut (identical workers made the whole hierarchy equal to ``score > 0.7``).
- Supervisor nodes: Each supervises its own group of workers (round robin)
  and votes block when the trust-weighted share of its workers voting block
  reaches ``consensus_threshold``.
- Queen node: Blocks when the trust-weighted majority of supervisors votes
  block.

Feedback moves each node the way the evidence points. A reported false
positive raises the thresholds and lowers the trust of the nodes that voted
to block it (it used to lower every threshold, so a borderline score that
was allowed before a false-positive report was blocked after it); a reported
missed threat lowers the thresholds of the nodes that voted to allow it.
Thresholds stay within [0.01, 0.99] and trust within (0, 1]. Votes are kept
per signature (bounded) so feedback reaches the nodes that cast them.

Reference: Immune system-inspired computing architectures.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

#: Bounds on an adapted worker threshold.
THRESHOLD_BOUNDS = (0.01, 0.99)


@dataclass
class HiveNode:
    """Individual node in hive firewall."""

    node_id: str
    node_type: str
    trust_score: float = 1.0
    blocked_signatures: set[str] = field(default_factory=set)
    detection_count: int = 0
    false_positive_count: int = 0
    #: Anomaly score at or above which a worker votes block (workers only).
    threshold: float = 0.7


@dataclass
class ThreatBlocking:
    """Threat blocking decision with reasoning."""

    signature_hash: str
    block_decision: bool
    confidence: float
    reasoning: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    consensus_votes: int = 0


@dataclass
class _VoteRecord:
    """The votes behind one decision, kept so feedback reaches their voters."""

    anomaly_score: float
    worker_votes: list[bool]
    supervisor_votes: list[bool]


class HiveFirewall:
    """HCIS-inspired hive-structured firewall.

    Implements distributed, collaborative anomaly blocking with:
    - O(1) lookup efficiency via hash-based blocking
    - Hierarchical decision-making
    - Consensus-based threat response
    - Adaptive trust scoring and per-worker thresholds
    """

    def __init__(
        self,
        num_worker_nodes: int = 10,
        num_supervisor_nodes: int = 3,
        consensus_threshold: float = 0.6,
        trust_decay: float = 0.95,
        base_threshold: float = 0.7,
        threshold_spread: float = 0.15,
        adaptation_step: float = 0.02,
        max_tracked_signatures: int = 100_000,
    ):
        """Initialize hive firewall.

        Args:
            num_worker_nodes: Number of worker detection nodes
            num_supervisor_nodes: Number of supervisor aggregation nodes
            consensus_threshold: Trust-weighted share of a supervisor's
                workers that must vote block for it to vote block (0.0-1.0)
            trust_decay: Factor applied to a node's trust when it votes wrong
            base_threshold: Centre of the worker thresholds
            threshold_spread: Half-width of the worker thresholds around
                ``base_threshold``
            adaptation_step: Threshold change per feedback report
            max_tracked_signatures: Vote records kept for feedback (oldest
                dropped first)

        Raises:
            ValueError: On a non-positive node count or a parameter outside
                its range.
        """
        if num_worker_nodes < 1 or num_supervisor_nodes < 1:
            raise ValueError("the hive needs at least one worker and one supervisor")
        if num_supervisor_nodes > num_worker_nodes:
            raise ValueError("each supervisor needs at least one worker")
        if not 0.0 < consensus_threshold <= 1.0 or not 0.0 < trust_decay < 1.0:
            raise ValueError("consensus_threshold must be in (0, 1] and trust_decay in (0, 1)")
        lo, hi = base_threshold - threshold_spread, base_threshold + threshold_spread
        if threshold_spread < 0.0 or lo < THRESHOLD_BOUNDS[0] or hi > THRESHOLD_BOUNDS[1]:
            raise ValueError(f"worker thresholds [{lo:g}, {hi:g}] must lie in {THRESHOLD_BOUNDS}")

        self.num_worker_nodes = num_worker_nodes
        self.num_supervisor_nodes = num_supervisor_nodes
        self.consensus_threshold = consensus_threshold
        self.trust_decay = trust_decay
        self.adaptation_step = adaptation_step
        self.max_tracked_signatures = max_tracked_signatures

        self._initial_thresholds = (
            [float(t) for t in np.linspace(lo, hi, num_worker_nodes)]
            if num_worker_nodes > 1
            else [float(base_threshold)]
        )
        self.worker_nodes = [
            HiveNode(node_id=f"worker_{i}", node_type="worker", threshold=t)
            for i, t in enumerate(self._initial_thresholds)
        ]

        self.supervisor_nodes = [
            HiveNode(node_id=f"supervisor_{i}", node_type="supervisor")
            for i in range(num_supervisor_nodes)
        ]
        #: Worker indices each supervisor aggregates (round robin).
        self.supervisor_groups = [
            list(range(j, num_worker_nodes, num_supervisor_nodes))
            for j in range(num_supervisor_nodes)
        ]

        self.queen_node = HiveNode(node_id="queen", node_type="queen")

        self.blocked_threats: dict[str, ThreatBlocking] = {}

        self.allowed_patterns: set[str] = set()

        self.threat_stats = defaultdict[str, int](int)

        self._vote_log: dict[str, _VoteRecord] = {}
        self._blocks_issued = 0
        self._blocks_reported_false = 0

    def _compute_signature_hash(self, data: np.ndarray[Any, Any]) -> str:
        """Compute O(1) signature hash for threat data (dtype-independent)."""
        data_bytes = np.ascontiguousarray(data, dtype=np.float64).tobytes()
        return hashlib.sha3_256(data_bytes).hexdigest()[:16]

    def is_blocked(self, data: np.ndarray[Any, Any]) -> tuple[bool, ThreatBlocking | None]:
        """Check if data matches blocked threat signature (O(1) lookup).

        Args:
            data: Input data to check

        Returns:
            Tuple of (is_blocked, blocking_decision)
        """
        signature = self._compute_signature_hash(data)

        if signature in self.allowed_patterns:
            return False, None

        if signature in self.blocked_threats:
            decision = self.blocked_threats[signature]
            return decision.block_decision, decision

        return False, None

    def detect_and_block(self, data: np.ndarray[Any, Any], anomaly_score: float) -> ThreatBlocking:
        """Hierarchical threat detection and blocking decision.

        Process:
        1. Workers vote against their own thresholds
        2. Supervisors aggregate their worker groups (trust-weighted)
        3. Queen blocks on a trust-weighted supervisor majority

        Args:
            data: Input data
            anomaly_score: Anomaly score from detector (0.0-1.0)

        Returns:
            ThreatBlocking decision

        Raises:
            ValueError: If the score is not finite or outside [0, 1].
        """
        score = float(anomaly_score)
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"anomaly_score must be in [0, 1], got {anomaly_score!r}")
        signature = self._compute_signature_hash(data)

        existing = self.is_blocked(data)
        if existing[0] and existing[1] is not None:
            return existing[1]

        worker_votes = self._worker_consensus(score)

        supervisor_votes = self._supervisor_consensus(worker_votes)

        block_decision = self._queen_decision(supervisor_votes, score)

        threat_block = ThreatBlocking(
            signature_hash=signature,
            block_decision=block_decision,
            confidence=float(worker_votes["confidence"]),
            reasoning=supervisor_votes["reasoning"],
            consensus_votes=supervisor_votes["votes"],
        )
        self._record_votes(
            signature, _VoteRecord(score, worker_votes["vote_list"], supervisor_votes["vote_list"])
        )

        if block_decision:
            self.blocked_threats[signature] = threat_block
            self.threat_stats[signature] += 1
            self._blocks_issued += 1

        return threat_block

    def _record_votes(self, signature: str, record: _VoteRecord) -> None:
        self._vote_log.pop(signature, None)
        self._vote_log[signature] = record
        while len(self._vote_log) > self.max_tracked_signatures:
            self._vote_log.pop(next(iter(self._vote_log)))

    @staticmethod
    def _weighted_share(nodes: list[HiveNode], votes: list[bool]) -> float:
        weights = np.array([n.trust_score for n in nodes])
        return float(np.dot(weights, votes) / weights.sum())

    def _worker_consensus(self, anomaly_score: float) -> dict[str, Any]:
        """Worker nodes vote on threat.

        Args:
            anomaly_score: Anomaly score

        Returns:
            Worker consensus results; ``confidence`` is the trust-weighted
            share of workers voting block.
        """
        votes = [anomaly_score >= worker.threshold for worker in self.worker_nodes]
        for worker, vote in zip(self.worker_nodes, votes, strict=True):
            if vote:
                worker.detection_count += 1

        consensus = self._weighted_share(self.worker_nodes, votes)

        return {
            "consensus": consensus,
            "votes": int(np.sum(votes)),
            "total": len(votes),
            "confidence": consensus,
            "vote_list": votes,
        }

    def _supervisor_consensus(self, worker_results: dict[str, Any]) -> dict[str, Any]:
        """Supervisor nodes aggregate their worker groups.

        Args:
            worker_results: Results from worker consensus

        Returns:
            Supervisor consensus results
        """
        worker_votes: list[bool] = worker_results["vote_list"]
        supervisor_votes = []
        for group in self.supervisor_groups:
            share = self._weighted_share(
                [self.worker_nodes[i] for i in group], [worker_votes[i] for i in group]
            )
            supervisor_votes.append(share >= self.consensus_threshold)

        supervisor_consensus = self._weighted_share(self.supervisor_nodes, supervisor_votes)

        if supervisor_consensus > 0.7:
            reasoning = "High supervisor consensus: Significant threat detected"
        elif supervisor_consensus > 0.5:
            reasoning = "Moderate supervisor consensus: Potential threat"
        else:
            reasoning = "Low supervisor consensus: Likely benign"

        return {
            "consensus": supervisor_consensus,
            "votes": int(np.sum(supervisor_votes)),
            "total": len(supervisor_votes),
            "confidence": supervisor_consensus,
            "reasoning": reasoning,
            "vote_list": supervisor_votes,
        }

    def _queen_decision(self, supervisor_results: dict[str, Any], anomaly_score: float) -> bool:
        """Queen node makes final blocking decision: a trust-weighted supervisor majority.

        Args:
            supervisor_results: Results from supervisor consensus
            anomaly_score: Original anomaly score (already weighed by the workers)

        Returns:
            Block decision
        """
        block = bool(supervisor_results["consensus"] > 0.5)

        if block:
            self.queen_node.detection_count += 1

        return block

    def allow_pattern(self, data: np.ndarray[Any, Any]) -> None:
        """Whitelist a pattern (add to allowed patterns).

        Args:
            data: Data pattern to allow
        """
        signature = self._compute_signature_hash(data)
        self.allowed_patterns.add(signature)

        if signature in self.blocked_threats:
            del self.blocked_threats[signature]

    def _adapt(self, record: _VoteRecord, *, blocking_was_wrong: bool) -> None:
        """Move each voter toward the reported truth."""
        lo, hi = THRESHOLD_BOUNDS
        for worker, voted_block in zip(self.worker_nodes, record.worker_votes, strict=True):
            if voted_block == blocking_was_wrong:
                # Wrong vote: lose trust; move the threshold away from the error.
                worker.trust_score *= self.trust_decay
                step = self.adaptation_step if blocking_was_wrong else -self.adaptation_step
                worker.threshold = float(np.clip(worker.threshold + step, lo, hi))
                if blocking_was_wrong:
                    worker.false_positive_count += 1
            else:
                worker.trust_score = min(1.0, worker.trust_score / self.trust_decay)
        for supervisor, voted_block in zip(
            self.supervisor_nodes, record.supervisor_votes, strict=True
        ):
            if voted_block == blocking_was_wrong:
                supervisor.trust_score *= self.trust_decay
            else:
                supervisor.trust_score = min(1.0, supervisor.trust_score / self.trust_decay)

    def report_false_positive(self, data: np.ndarray[Any, Any]) -> None:
        """Report false positive and update trust scores.

        The nodes that voted to block lose trust and raise their thresholds;
        those that voted to allow gain trust. A pattern the hive never
        decided is only allowlisted: there are no votes to correct.

        Args:
            data: Data that was incorrectly blocked
        """
        signature = self._compute_signature_hash(data)

        if signature in self.blocked_threats:
            del self.blocked_threats[signature]
            self._blocks_reported_false += 1

        self.allowed_patterns.add(signature)

        record = self._vote_log.get(signature)
        if record is not None:
            self._adapt(record, blocking_was_wrong=True)

    def report_missed_threat(self, data: np.ndarray[Any, Any], anomaly_score: float) -> None:
        """Report a threat the hive allowed, and block its signature.

        The nodes that voted to allow it lose trust and lower their
        thresholds; those that voted to block gain trust. Without a recorded
        decision the votes are those the hive would cast at ``anomaly_score``.

        Args:
            data: Data that should have been blocked
            anomaly_score: Its anomaly score (0.0-1.0)
        """
        signature = self._compute_signature_hash(data)
        record = self._vote_log.get(signature)
        if record is None:
            score = float(anomaly_score)
            worker_votes = [score >= w.threshold for w in self.worker_nodes]
            supervisor_votes = self._supervisor_consensus(
                {"vote_list": worker_votes, "consensus": 0.0}
            )["vote_list"]
            record = _VoteRecord(score, worker_votes, supervisor_votes)
        self._adapt(record, blocking_was_wrong=False)

        self.allowed_patterns.discard(signature)
        self.blocked_threats[signature] = ThreatBlocking(
            signature_hash=signature,
            block_decision=True,
            confidence=1.0,
            reasoning="Reported threat",
        )
        self.threat_stats[signature] += 1

    def update_node_trust(self, node_id: str, success: bool) -> None:
        """Update trust score for specific node.

        Args:
            node_id: Node identifier
            success: Whether detection was successful
        """
        all_nodes = self.worker_nodes + self.supervisor_nodes + [self.queen_node]

        for node in all_nodes:
            if node.node_id == node_id:
                if success:
                    node.trust_score = min(1.0, node.trust_score * 1.05)
                else:
                    node.trust_score *= self.trust_decay
                break

    def get_firewall_stats(self) -> dict[str, Any]:
        """Get comprehensive firewall statistics.

        ``detection_accuracy`` is the share of issued blocks not reported as
        false positives (1.0 when nothing was blocked is not claimed: 0.0).
        """
        total_blocked = len(self.blocked_threats)
        total_allowed = len(self.allowed_patterns)

        avg_worker_trust = np.mean([w.trust_score for w in self.worker_nodes])
        avg_supervisor_trust = np.mean([s.trust_score for s in self.supervisor_nodes])

        total_detections = sum(w.detection_count for w in self.worker_nodes)
        total_false_positives = sum(w.false_positive_count for w in self.worker_nodes)

        return {
            "total_blocked_signatures": total_blocked,
            "total_allowed_patterns": total_allowed,
            "avg_worker_trust": float(avg_worker_trust),
            "avg_supervisor_trust": float(avg_supervisor_trust),
            "queen_trust": self.queen_node.trust_score,
            "total_detections": total_detections,
            "total_false_positives": total_false_positives,
            "detection_accuracy": (
                (self._blocks_issued - self._blocks_reported_false) / self._blocks_issued
                if self._blocks_issued > 0
                else 0.0
            ),
            "worker_thresholds": [w.threshold for w in self.worker_nodes],
            "most_blocked_threats": sorted(
                self.threat_stats.items(), key=lambda x: x[1], reverse=True
            )[:10],
        }

    def reset_firewall(self) -> None:
        """Reset firewall state (emergency use only)."""
        self.blocked_threats.clear()
        self.allowed_patterns.clear()
        self.threat_stats.clear()
        self._vote_log.clear()
        self._blocks_issued = 0
        self._blocks_reported_false = 0

        for worker, threshold in zip(self.worker_nodes, self._initial_thresholds, strict=True):
            worker.trust_score = 1.0
            worker.detection_count = 0
            worker.false_positive_count = 0
            worker.threshold = threshold

        for supervisor in self.supervisor_nodes:
            supervisor.trust_score = 1.0
            supervisor.detection_count = 0
            supervisor.false_positive_count = 0

        self.queen_node.trust_score = 1.0
        self.queen_node.detection_count = 0
