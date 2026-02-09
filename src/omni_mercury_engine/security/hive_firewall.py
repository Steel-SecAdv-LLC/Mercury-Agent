"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
HCIS-Inspired Hive-Structured Firewall for Anomaly Blocking

Implements a hierarchical, distributed firewall inspired by HCIS (Hierarchical
Cognitive Immune System) principles with hive-like collaborative defense.

Architecture:
- Worker nodes: Front-line anomaly detection
- Supervisor nodes: Aggregation and decision-making
- Queen node: Global policy and coordination

Reference: Immune system-inspired computing architectures
MIT-compatible implementation.
"""

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class HiveNode:
    """Individual node in hive firewall."""

    node_id: str
    node_type: str
    trust_score: float = 1.0
    blocked_signatures: set[str] = field(default_factory=set)
    detection_count: int = 0
    false_positive_count: int = 0


@dataclass
class ThreatBlocking:
    """Threat blocking decision with reasoning."""

    signature_hash: str
    block_decision: bool
    confidence: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)
    consensus_votes: int = 0


class HiveFirewall:
    """
    HCIS-inspired hive-structured firewall.

    Implements distributed, collaborative anomaly blocking with:
    - O(1) lookup efficiency via hash-based blocking
    - Hierarchical decision-making
    - Consensus-based threat response
    - Adaptive trust scoring
    """

    def __init__(
        self,
        num_worker_nodes: int = 10,
        num_supervisor_nodes: int = 3,
        consensus_threshold: float = 0.6,
        trust_decay: float = 0.95,
    ):
        """
        Initialize hive firewall.

        Args:
            num_worker_nodes: Number of worker detection nodes
            num_supervisor_nodes: Number of supervisor aggregation nodes
            consensus_threshold: Threshold for consensus blocking (0.0-1.0)
            trust_decay: Decay factor for node trust scores
        """
        self.num_worker_nodes = num_worker_nodes
        self.num_supervisor_nodes = num_supervisor_nodes
        self.consensus_threshold = consensus_threshold
        self.trust_decay = trust_decay

        self.worker_nodes = [
            HiveNode(node_id=f"worker_{i}", node_type="worker") for i in range(num_worker_nodes)
        ]

        self.supervisor_nodes = [
            HiveNode(node_id=f"supervisor_{i}", node_type="supervisor")
            for i in range(num_supervisor_nodes)
        ]

        self.queen_node = HiveNode(node_id="queen", node_type="queen")

        self.blocked_threats: dict[str, ThreatBlocking] = {}

        self.allowed_patterns: set[str] = set()

        self.threat_stats = defaultdict[str, int](int)

    def _compute_signature_hash(self, data: np.ndarray[Any, Any]) -> str:
        """Compute O(1) signature hash for threat data."""
        data_bytes = data.tobytes()
        return hashlib.sha256(data_bytes).hexdigest()[:16]

    def is_blocked(self, data: np.ndarray[Any, Any]) -> tuple[bool, ThreatBlocking | None]:
        """
        Check if data matches blocked threat signature (O(1) lookup).

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
        """
        Hierarchical threat detection and blocking decision.

        Process:
        1. Workers detect anomaly
        2. Supervisors aggregate decisions
        3. Queen makes final blocking decision

        Args:
            data: Input data
            anomaly_score: Anomaly score from detector (0.0-1.0)

        Returns:
            ThreatBlocking decision
        """
        signature = self._compute_signature_hash(data)

        existing = self.is_blocked(data)
        if existing[0] and existing[1] is not None:
            return existing[1]

        worker_votes = self._worker_consensus(anomaly_score)

        supervisor_votes = self._supervisor_consensus(worker_votes)

        block_decision = self._queen_decision(supervisor_votes, anomaly_score)

        threat_block = ThreatBlocking(
            signature_hash=signature,
            block_decision=block_decision,
            confidence=float(supervisor_votes["confidence"]),
            reasoning=supervisor_votes["reasoning"],
            consensus_votes=supervisor_votes["votes"],
        )

        if block_decision:
            self.blocked_threats[signature] = threat_block
            self.threat_stats[signature] += 1

        return threat_block

    def _worker_consensus(self, anomaly_score: float) -> dict[str, Any]:
        """
        Worker nodes vote on threat.

        Args:
            anomaly_score: Anomaly score

        Returns:
            Worker consensus results
        """
        votes = []

        for worker in self.worker_nodes:
            threshold = 0.7 * worker.trust_score
            vote = anomaly_score > threshold
            votes.append(vote)

            if vote:
                worker.detection_count += 1

        consensus = np.mean(votes)

        return {
            "consensus": consensus,
            "votes": int(np.sum(votes)),
            "total": len(votes),
            "confidence": consensus,
        }

    def _supervisor_consensus(self, worker_results: dict[str, Any]) -> dict[str, Any]:
        """
        Supervisor nodes aggregate worker decisions.

        Args:
            worker_results: Results from worker consensus

        Returns:
            Supervisor consensus results
        """
        supervisor_votes = []

        for supervisor in self.supervisor_nodes:
            vote = worker_results["consensus"] > (self.consensus_threshold * supervisor.trust_score)
            supervisor_votes.append(vote)

        supervisor_consensus = np.mean(supervisor_votes)

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
        }

    def _queen_decision(self, supervisor_results: dict[str, Any], anomaly_score: float) -> bool:
        """
        Queen node makes final blocking decision.

        Args:
            supervisor_results: Results from supervisor consensus
            anomaly_score: Original anomaly score

        Returns:
            Block decision
        """
        weighted_decision = (
            0.6 * supervisor_results["consensus"]
            + 0.3 * anomaly_score
            + 0.1 * self.queen_node.trust_score
        )

        block = bool(weighted_decision > 0.7)

        if block:
            self.queen_node.detection_count += 1

        return block

    def allow_pattern(self, data: np.ndarray[Any, Any]) -> None:
        """
        Whitelist a pattern (add to allowed patterns).

        Args:
            data: Data pattern to allow
        """
        signature = self._compute_signature_hash(data)
        self.allowed_patterns.add(signature)

        if signature in self.blocked_threats:
            del self.blocked_threats[signature]

    def report_false_positive(self, data: np.ndarray[Any, Any]) -> None:
        """
        Report false positive and update trust scores.

        Args:
            data: Data that was incorrectly blocked
        """
        signature = self._compute_signature_hash(data)

        if signature in self.blocked_threats:
            del self.blocked_threats[signature]

        self.allowed_patterns.add(signature)

        for worker in self.worker_nodes:
            worker.trust_score *= self.trust_decay
            worker.false_positive_count += 1

        for supervisor in self.supervisor_nodes:
            supervisor.trust_score *= self.trust_decay

    def update_node_trust(self, node_id: str, success: bool) -> None:
        """
        Update trust score for specific node.

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
        """Get comprehensive firewall statistics."""
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
                (total_detections - total_false_positives) / total_detections
                if total_detections > 0
                else 0.0
            ),
            "most_blocked_threats": sorted(
                self.threat_stats.items(), key=lambda x: x[1], reverse=True
            )[:10],
        }

    def reset_firewall(self) -> None:
        """Reset firewall state (emergency use only)."""
        self.blocked_threats.clear()
        self.allowed_patterns.clear()
        self.threat_stats.clear()

        for worker in self.worker_nodes:
            worker.trust_score = 1.0
            worker.detection_count = 0
            worker.false_positive_count = 0

        for supervisor in self.supervisor_nodes:
            supervisor.trust_score = 1.0
            supervisor.detection_count = 0
            supervisor.false_positive_count = 0

        self.queen_node.trust_score = 1.0
        self.queen_node.detection_count = 0
