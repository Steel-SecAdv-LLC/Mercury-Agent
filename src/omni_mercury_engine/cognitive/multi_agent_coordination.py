"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Multi-Agent Coordination Protocol for Mercury Agent.

Implements coordination between multiple detection agents inspired by:
- "Multi-Agent Reinforcement Learning: A Survey" (Hernandez-Leal et al., 2019)
- "Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design"
- "QMIX: Monotonic Value Function Factorization" (Rashid et al., 2018)
- "MAPPO: Multi-Agent PPO" (Yu et al., 2022)

Multi-agent coordination enables:
1. Distributed anomaly detection across domains
2. Consensus building for uncertain detections
3. Specialization of agents for different anomaly types
4. Collaborative investigation of complex anomalies
5. Robust detection through agent redundancy

Key Concepts:
- Agent: Independent detection unit with specific capabilities
- Coalition: Temporary group of agents for joint task
- Consensus: Agreement mechanism for collective decisions
- Coordination Protocol: Rules for agent interaction
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from queue import Empty, Queue
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio

# Coordination parameters
MAX_AGENTS = 50
MAX_COALITION_SIZE = 10
DEFAULT_CONSENSUS_THRESHOLD = 0.7
MESSAGE_TIMEOUT = 5.0


class AgentRole(Enum):
    """Roles for detection agents."""

    STATISTICAL = "statistical"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    BEHAVIORAL = "behavioral"
    DIMENSIONAL = "dimensional"
    SPECIALIST = "specialist"
    COORDINATOR = "coordinator"
    VALIDATOR = "validator"


class AgentStatus(Enum):
    """Agent operational status."""

    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    COORDINATING = "coordinating"
    OFFLINE = "offline"


class MessageType(Enum):
    """Types of inter-agent messages."""

    DETECTION_REQUEST = "detection_request"
    DETECTION_RESULT = "detection_result"
    CONSENSUS_REQUEST = "consensus_request"
    CONSENSUS_VOTE = "consensus_vote"
    COALITION_INVITE = "coalition_invite"
    COALITION_ACCEPT = "coalition_accept"
    COALITION_REJECT = "coalition_reject"
    COALITION_DISSOLVE = "coalition_dissolve"
    STATUS_UPDATE = "status_update"
    CAPABILITY_QUERY = "capability_query"
    CAPABILITY_RESPONSE = "capability_response"
    TASK_ASSIGNMENT = "task_assignment"
    TASK_COMPLETE = "task_complete"


class ConsensusMethod(Enum):
    """Methods for reaching consensus."""

    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    UNANIMOUS = "unanimous"
    BYZANTINE_TOLERANT = "byzantine_tolerant"
    CONFIDENCE_WEIGHTED = "confidence_weighted"


class CoordinationStrategy(Enum):
    """Strategies for multi-agent coordination."""

    CENTRALIZED = "centralized"  # Single coordinator
    DISTRIBUTED = "distributed"  # No coordinator
    HIERARCHICAL = "hierarchical"  # Multi-level coordination
    MARKET_BASED = "market_based"  # Task auction


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class Message:
    """Inter-agent message.

    Attributes:
        message_id: Unique identifier
        sender_id: Sending agent ID
        receiver_id: Receiving agent ID (or "broadcast")
        message_type: Type of message
        content: Message content
        timestamp: Creation timestamp
        priority: Message priority (higher = more urgent)
        requires_response: Whether response is expected
        correlation_id: ID for request-response correlation
    """

    message_id: str
    sender_id: str
    receiver_id: str
    message_type: MessageType
    content: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: int = 5
    requires_response: bool = False
    correlation_id: str | None = None


@dataclass
class AgentCapability:
    """
    Description of agent capabilities.

    Attributes:
        capability_id: Unique identifier
        name: Capability name
        domains: Applicable domains
        anomaly_types: Detectable anomaly types
        accuracy: Historical accuracy
        speed: Processing speed (samples/sec)
        confidence_calibration: How well-calibrated is confidence
    """

    capability_id: str
    name: str
    domains: list[str]
    anomaly_types: list[str]
    accuracy: float = 0.8
    speed: float = 100.0
    confidence_calibration: float = 0.9


@dataclass
class DetectionResult:
    """
    Result from an agent's detection.

    Attributes:
        agent_id: Detecting agent ID
        anomaly_score: Anomaly score (0-1)
        is_anomaly: Boolean classification
        confidence: Confidence in result
        features_used: Features used for detection
        reasoning: Reasoning for result
        timestamp: Result timestamp
        metadata: Additional metadata
    """

    agent_id: str
    anomaly_score: float
    is_anomaly: bool
    confidence: float
    features_used: list[str] = field(default_factory=list)
    reasoning: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsensusResult:
    """
    Result of consensus process.

    Attributes:
        consensus_id: Unique identifier
        final_decision: Agreed decision
        confidence: Collective confidence
        agreement_ratio: Ratio of agents agreeing
        participant_count: Number of participating agents
        method_used: Consensus method used
        dissenting_agents: IDs of dissenting agents
        timestamp: Consensus timestamp
    """

    consensus_id: str
    final_decision: bool
    confidence: float
    agreement_ratio: float
    participant_count: int
    method_used: ConsensusMethod
    dissenting_agents: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Coalition:
    """
    A coalition of cooperating agents.

    Attributes:
        coalition_id: Unique identifier
        leader_id: Coalition leader agent ID
        member_ids: Member agent IDs
        members: List of agent objects (alternative to member_ids)
        task: Task the coalition is addressing
        objective: Coalition objective (alternative to task)
        created_at: Creation timestamp
        status: Coalition status
        results: Collected results
    """

    coalition_id: str
    leader_id: str | None = None
    member_ids: list[str] = field(default_factory=list)
    members: list[Any] = field(default_factory=list)
    task: dict[str, Any] = field(default_factory=dict)
    objective: str = ""
    created_at: float = field(default_factory=time.time)
    status: str = "active"
    results: list[DetectionResult] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Post-initialization to handle alternative parameters."""
        # If members provided but not member_ids, extract IDs
        if self.members and not self.member_ids:
            self.member_ids = [
                m.agent_id if hasattr(m, "agent_id") else str(m) for m in self.members
            ]
        # If leader_id not set, use first member
        if self.leader_id is None and self.member_ids:
            self.leader_id = self.member_ids[0]
        # If objective provided but not task, create task from objective
        if self.objective and not self.task:
            self.task = {"objective": self.objective}


# =============================================================================
# Agent Interface
# =============================================================================


class DetectionAgent(ABC):
    """Abstract base class for detection agents."""

    def __init__(
        self,
        agent_id: str,
        role: AgentRole | None = None,
        capabilities: list[AgentCapability] | list[str] | None = None,
    ):
        """
        Initialize detection agent.

        Args:
            agent_id: Unique agent identifier
            role: Agent role (optional, defaults to STATISTICAL)
            capabilities: Agent capabilities (AgentCapability objects or strings)
        """
        self.agent_id = agent_id
        self.role = role or AgentRole.STATISTICAL

        # Handle both AgentCapability objects and string capabilities
        if capabilities is None:
            self.capabilities: list[AgentCapability] = []
            self._capability_names: list[str] = []
        elif capabilities and isinstance(capabilities[0], str):
            self._capability_names = [c for c in capabilities if isinstance(c, str)]
            self.capabilities = []
        else:
            cap_objects = [c for c in capabilities if isinstance(c, AgentCapability)]
            self.capabilities = cap_objects
            self._capability_names = [c.name for c in cap_objects]

        self.status = AgentStatus.IDLE

        self._message_queue: Queue[Message] = Queue()
        self._message_counter = 0

    @abstractmethod
    def detect(
        self,
        data: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> DetectionResult:
        """
        Perform anomaly detection.

        Args:
            data: Input data
            context: Detection context

        Returns:
            Detection result
        """
        pass

    def receive_message(self, message: Message) -> None:
        """Receive a message."""
        self._message_queue.put(message)

    def get_pending_messages(self) -> list[Message]:
        """Get all pending messages."""
        messages = []
        while True:
            try:
                messages.append(self._message_queue.get_nowait())
            except Empty:
                break
        return messages

    def send_message(
        self,
        receiver_id: str,
        message_type: MessageType,
        content: dict[str, Any],
        coordinator: AgentCoordinator,
        priority: int = 5,
    ) -> Message:
        """
        Send a message to another agent.

        Args:
            receiver_id: Receiving agent ID
            message_type: Type of message
            content: Message content
            coordinator: Coordinator for routing
            priority: Message priority

        Returns:
            Sent message
        """
        self._message_counter += 1
        message_id = f"{self.agent_id}_msg_{self._message_counter:06d}"

        message = Message(
            message_id=message_id,
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            priority=priority,
        )

        coordinator.route_message(message)
        return message


class SimpleDetectionAgent(DetectionAgent):
    """Simple detection agent implementation."""

    def __init__(
        self,
        agent_id: str,
        role: AgentRole = AgentRole.STATISTICAL,
        threshold: float = 0.5,
    ):
        """Initialize simple agent."""
        super().__init__(agent_id, role)
        self.threshold = threshold

    def detect(
        self,
        data: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> DetectionResult:
        """Perform simple threshold-based detection."""
        # Compute anomaly score based on role
        if self.role == AgentRole.STATISTICAL:
            score = self._statistical_score(data)
        elif self.role == AgentRole.TEMPORAL:
            score = self._temporal_score(data)
        elif self.role == AgentRole.DIMENSIONAL:
            score = self._dimensional_score(data)
        else:
            score = self._generic_score(data)

        is_anomaly = score > self.threshold
        confidence = abs(score - 0.5) * 2

        return DetectionResult(
            agent_id=self.agent_id,
            anomaly_score=float(score),
            is_anomaly=is_anomaly,
            confidence=confidence,
            features_used=[self.role.value],
            reasoning=f"{self.role.value} detection with score {score:.3f}",
        )

    def _statistical_score(self, data: np.ndarray) -> float:
        """Compute statistical anomaly score."""
        mean = np.mean(data)
        std = np.std(data) + 1e-8
        z_scores = np.abs((data - mean) / std)
        return float(np.clip(np.max(z_scores) / 3, 0, 1))

    def _temporal_score(self, data: np.ndarray) -> float:
        """Compute temporal anomaly score."""
        if len(data) < 2:
            return 0.5
        diffs = np.diff(data)
        mean_diff = np.mean(np.abs(diffs))
        max_diff = np.max(np.abs(diffs))
        return float(np.clip(max_diff / (mean_diff + 1e-8) / 5, 0, 1))

    def _dimensional_score(self, data: np.ndarray) -> float:
        """Compute dimensional anomaly score."""
        if data.ndim == 1:
            data = data.reshape(1, -1)
        # Use distance from centroid
        centroid = np.mean(data, axis=0)
        distances = np.linalg.norm(data - centroid, axis=-1)
        max_dist = np.max(distances)
        mean_dist = np.mean(distances)
        return float(np.clip((max_dist - mean_dist) / (mean_dist + 1e-8), 0, 1))

    def _generic_score(self, data: np.ndarray) -> float:
        """Generic anomaly score."""
        return float(np.clip(np.max(np.abs(data)) / 10, 0, 1))


# =============================================================================
# Consensus Protocol
# =============================================================================


class ConsensusProtocol:
    """
    Protocol for reaching consensus among agents.

    Supports multiple consensus methods from simple voting to Byzantine fault-tolerant consensus.
    """

    def __init__(
        self,
        method: ConsensusMethod | str = ConsensusMethod.CONFIDENCE_WEIGHTED,
        threshold: float = DEFAULT_CONSENSUS_THRESHOLD,
        min_participants: int = 3,
    ):
        """
        Initialize consensus protocol.

        Args:
            method: Consensus method to use (ConsensusMethod or string)
            threshold: Agreement threshold
            min_participants: Minimum participants required
        """
        # Handle string method names
        if isinstance(method, str):
            method_map = {
                "majority_vote": ConsensusMethod.MAJORITY_VOTE,
                "weighted_vote": ConsensusMethod.WEIGHTED_VOTE,
                "unanimous": ConsensusMethod.UNANIMOUS,
                "byzantine_tolerant": ConsensusMethod.BYZANTINE_TOLERANT,
                "confidence_weighted": ConsensusMethod.CONFIDENCE_WEIGHTED,
            }
            self.method = method_map.get(method.lower(), ConsensusMethod.MAJORITY_VOTE)
            self._method_str = method.lower()
        else:
            self.method = method
            self._method_str = method.value

        self.threshold = threshold
        self.min_participants = min_participants
        self._consensus_counter = 0

    def reach_consensus(
        self,
        results: list[DetectionResult] | list[dict[str, Any]],
    ) -> ConsensusResult | dict[str, Any]:
        """
        Reach consensus on detection results.

        Args:
            results: Results from multiple agents (DetectionResult objects or vote dicts)

        Returns:
            Consensus result (ConsensusResult or dict)
        """
        self._consensus_counter += 1
        consensus_id = f"consensus_{self._consensus_counter:06d}"

        # Handle list of vote dicts (test API)
        if results and isinstance(results[0], dict):
            vote_list: list[dict[str, Any]] = [r for r in results if isinstance(r, dict)]
            return self._reach_consensus_from_votes(consensus_id, vote_list)

        # Original DetectionResult handling
        detection_results: list[DetectionResult] = [
            r for r in results if isinstance(r, DetectionResult)
        ]

        if len(detection_results) < self.min_participants:
            return ConsensusResult(
                consensus_id=consensus_id,
                final_decision=False,
                confidence=0.0,
                agreement_ratio=0.0,
                participant_count=len(detection_results),
                method_used=self.method,
            )

        if self.method == ConsensusMethod.MAJORITY_VOTE:
            return self._majority_vote(consensus_id, detection_results)
        elif self.method == ConsensusMethod.WEIGHTED_VOTE:
            return self._weighted_vote(consensus_id, detection_results)
        elif self.method == ConsensusMethod.UNANIMOUS:
            return self._unanimous(consensus_id, detection_results)
        elif self.method == ConsensusMethod.BYZANTINE_TOLERANT:
            return self._byzantine_tolerant(consensus_id, detection_results)
        elif self.method == ConsensusMethod.CONFIDENCE_WEIGHTED:
            return self._confidence_weighted(consensus_id, detection_results)
        else:
            return self._majority_vote(consensus_id, detection_results)

    def _reach_consensus_from_votes(
        self,
        consensus_id: str,
        votes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Reach consensus from vote dicts (test API).

        Args:
            consensus_id: Consensus ID
            votes: List of vote dicts with agent, decision, confidence, weight

        Returns:
            Consensus result dict
        """
        if not votes:
            return {"decision": False, "agreement_ratio": 0.0, "is_byzantine_safe": True}

        if self._method_str == "majority_vote":
            votes_for = sum(1 for v in votes if v.get("decision", False))
            votes_against = len(votes) - votes_for
            decision = votes_for > votes_against
            agreement_ratio = max(votes_for, votes_against) / len(votes)

        elif self._method_str == "weighted_vote":
            weighted_for = sum(
                v.get("confidence", 0.5) * v.get("weight", 1.0)
                for v in votes
                if v.get("decision", False)
            )
            weighted_against = sum(
                v.get("confidence", 0.5) * v.get("weight", 1.0)
                for v in votes
                if not v.get("decision", False)
            )
            decision = weighted_for > weighted_against
            total_weight = weighted_for + weighted_against
            agreement_ratio = (
                max(weighted_for, weighted_against) / total_weight if total_weight > 0 else 0.5
            )

        elif self._method_str == "byzantine_tolerant":
            # Byzantine fault tolerance: can tolerate f < n/3 faulty nodes
            n = len(votes)
            f_max = (n - 1) // 3  # Maximum faulty nodes we can tolerate

            # Sort by confidence to identify potential outliers
            sorted_votes = sorted(votes, key=lambda v: v.get("confidence", 0.5), reverse=True)

            # Use top 2/3 of votes (excluding potential Byzantine nodes)
            reliable_count = n - f_max
            reliable_votes = sorted_votes[:reliable_count]

            votes_for = sum(1 for v in reliable_votes if v.get("decision", False))
            decision = votes_for > reliable_count / 2
            agreement_ratio = votes_for / reliable_count if reliable_count > 0 else 0.5

            return {
                "decision": decision,
                "agreement_ratio": agreement_ratio,
                "is_byzantine_safe": True,
                "reliable_votes": reliable_count,
                "total_votes": n,
            }

        else:
            # Default to majority
            votes_for = sum(1 for v in votes if v.get("decision", False))
            decision = votes_for > len(votes) / 2
            agreement_ratio = max(votes_for, len(votes) - votes_for) / len(votes)

        return {
            "decision": decision,
            "agreement_ratio": agreement_ratio,
            "is_byzantine_safe": self._method_str == "byzantine_tolerant",
        }

    def _majority_vote(
        self,
        consensus_id: str,
        results: list[DetectionResult],
    ) -> ConsensusResult:
        """Simple majority voting."""
        votes_for = sum(1 for r in results if r.is_anomaly)
        votes_against = len(results) - votes_for

        final_decision = votes_for > votes_against
        agreement_ratio = max(votes_for, votes_against) / len(results)

        # Average confidence
        confidence = np.mean([r.confidence for r in results])

        # Find dissenters
        dissenting = [r.agent_id for r in results if r.is_anomaly != final_decision]

        return ConsensusResult(
            consensus_id=consensus_id,
            final_decision=final_decision,
            confidence=float(confidence),
            agreement_ratio=agreement_ratio,
            participant_count=len(results),
            method_used=ConsensusMethod.MAJORITY_VOTE,
            dissenting_agents=dissenting,
        )

    def _weighted_vote(
        self,
        consensus_id: str,
        results: list[DetectionResult],
    ) -> ConsensusResult:
        """Weighted voting by anomaly score."""
        total_score = sum(r.anomaly_score for r in results)
        avg_score = total_score / len(results)

        final_decision = avg_score > 0.5

        # Agreement is how far from 0.5 (uncertainty)
        agreement_ratio = abs(avg_score - 0.5) * 2

        confidence = np.mean([r.confidence for r in results]) * agreement_ratio

        dissenting = [r.agent_id for r in results if r.is_anomaly != final_decision]

        return ConsensusResult(
            consensus_id=consensus_id,
            final_decision=final_decision,
            confidence=float(confidence),
            agreement_ratio=agreement_ratio,
            participant_count=len(results),
            method_used=ConsensusMethod.WEIGHTED_VOTE,
            dissenting_agents=dissenting,
        )

    def _unanimous(
        self,
        consensus_id: str,
        results: list[DetectionResult],
    ) -> ConsensusResult:
        """Unanimous consensus required."""
        decisions = [r.is_anomaly for r in results]
        unanimous = len(set(decisions)) == 1

        if unanimous:
            final_decision = decisions[0]
            agreement_ratio = 1.0
            confidence = float(np.mean([r.confidence for r in results]))
            dissenting = []
        else:
            # No consensus - default to anomaly for safety
            final_decision = any(decisions)
            agreement_ratio = max(decisions.count(True), decisions.count(False)) / len(decisions)
            confidence = 0.5  # type: ignore[assignment, unused-ignore]
            dissenting = [r.agent_id for r in results if r.is_anomaly != final_decision]

        return ConsensusResult(
            consensus_id=consensus_id,
            final_decision=final_decision,
            confidence=float(confidence),
            agreement_ratio=agreement_ratio,
            participant_count=len(results),
            method_used=ConsensusMethod.UNANIMOUS,
            dissenting_agents=dissenting,
        )

    def _byzantine_tolerant(
        self,
        consensus_id: str,
        results: list[DetectionResult],
    ) -> ConsensusResult:
        """
        Byzantine fault-tolerant consensus.

        Tolerates up to f faulty agents with n >= 3f + 1 total.
        """
        n = len(results)
        f = (n - 1) // 3  # Maximum faulty agents

        # Need at least 2f + 1 agreeing
        required_agreement = 2 * f + 1

        votes_for = sum(1 for r in results if r.is_anomaly)
        votes_against = n - votes_for

        if votes_for >= required_agreement:
            final_decision = True
            agreement_ratio = votes_for / n
        elif votes_against >= required_agreement:
            final_decision = False
            agreement_ratio = votes_against / n
        else:
            # No Byzantine agreement - default to cautious
            final_decision = votes_for >= votes_against
            agreement_ratio = max(votes_for, votes_against) / n

        confidence = np.mean([r.confidence for r in results])

        dissenting = [r.agent_id for r in results if r.is_anomaly != final_decision]

        return ConsensusResult(
            consensus_id=consensus_id,
            final_decision=final_decision,
            confidence=float(confidence),
            agreement_ratio=agreement_ratio,
            participant_count=n,
            method_used=ConsensusMethod.BYZANTINE_TOLERANT,
            dissenting_agents=dissenting,
        )

    def _confidence_weighted(
        self,
        consensus_id: str,
        results: list[DetectionResult],
    ) -> ConsensusResult:
        """Confidence-weighted consensus."""
        # Weight votes by confidence
        weighted_scores = []
        total_confidence = 0.0

        for r in results:
            weight = r.confidence
            score = r.anomaly_score * weight
            weighted_scores.append(score)
            total_confidence += weight

        if total_confidence > 0:
            avg_weighted_score = sum(weighted_scores) / total_confidence
        else:
            avg_weighted_score = 0.5

        final_decision = avg_weighted_score > 0.5

        # Calculate agreement weighted by confidence
        agreements = []
        for r in results:
            agrees = r.is_anomaly == final_decision
            agreements.append(r.confidence if agrees else 0)

        agreement_ratio = sum(agreements) / total_confidence if total_confidence > 0 else 0

        confidence = avg_weighted_score if final_decision else (1 - avg_weighted_score)

        dissenting = [r.agent_id for r in results if r.is_anomaly != final_decision]

        return ConsensusResult(
            consensus_id=consensus_id,
            final_decision=final_decision,
            confidence=float(confidence),
            agreement_ratio=agreement_ratio,
            participant_count=len(results),
            method_used=ConsensusMethod.CONFIDENCE_WEIGHTED,
            dissenting_agents=dissenting,
        )


# =============================================================================
# Agent Coordinator
# =============================================================================


class AgentCoordinator:
    """
    Coordinator for multi-agent system.

    Manages agent registration, message routing, coalition formation, and consensus building.
    """

    def __init__(
        self,
        strategy: CoordinationStrategy = CoordinationStrategy.DISTRIBUTED,
        consensus_method: ConsensusMethod = ConsensusMethod.CONFIDENCE_WEIGHTED,
        max_agents: int = MAX_AGENTS,
    ):
        """
        Initialize agent coordinator.

        Args:
            strategy: Coordination strategy
            consensus_method: Default consensus method
            max_agents: Maximum number of agents
        """
        self.strategy = strategy
        self.consensus_method = consensus_method
        self.max_agents = max_agents

        # Agent registry
        self._agents: dict[str, DetectionAgent] = {}

        # Coalition management
        self._coalitions: dict[str, Coalition] = {}
        self._coalition_counter = 0

        # Consensus protocol
        self.consensus_protocol = ConsensusProtocol(method=consensus_method)

        # Message queue
        self._broadcast_queue: Queue[Message] = Queue()

        # Statistics
        self._stats = {
            "messages_routed": 0,
            "coalitions_formed": 0,
            "consensus_reached": 0,
            "detections_coordinated": 0,
        }

        logger.info(
            f"AgentCoordinator initialized (strategy={strategy.value}, "
            f"consensus={consensus_method.value})"
        )

    def register_agent(self, agent: DetectionAgent) -> bool:
        """
        Register an agent with the coordinator.

        Args:
            agent: Agent to register

        Returns:
            True if registered successfully
        """
        if len(self._agents) >= self.max_agents:
            logger.warning(f"Cannot register agent {agent.agent_id}: max agents reached")
            return False

        if agent.agent_id in self._agents:
            logger.warning(f"Agent {agent.agent_id} already registered")
            return False

        self._agents[agent.agent_id] = agent
        agent.status = AgentStatus.IDLE
        logger.info(f"Registered agent: {agent.agent_id} (role={agent.role.value})")
        return True

    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent.

        Args:
            agent_id: Agent ID to unregister

        Returns:
            True if unregistered successfully
        """
        if agent_id not in self._agents:
            return False

        # Remove from any coalitions
        for coalition in self._coalitions.values():
            if agent_id in coalition.member_ids:
                coalition.member_ids.remove(agent_id)

        del self._agents[agent_id]
        logger.info(f"Unregistered agent: {agent_id}")
        return True

    def route_message(self, message: Message) -> None:
        """
        Route a message to its recipient.

        Args:
            message: Message to route
        """
        self._stats["messages_routed"] += 1

        if message.receiver_id == "broadcast":
            # Send to all agents
            for agent in self._agents.values():
                if agent.agent_id != message.sender_id:
                    agent.receive_message(message)
        elif message.receiver_id in self._agents:
            self._agents[message.receiver_id].receive_message(message)
        else:
            logger.warning(f"Unknown recipient: {message.receiver_id}")

    def form_coalition(
        self,
        task: dict[str, Any],
        leader_id: str | None = None,
        required_roles: list[AgentRole] | None = None,
        max_size: int = MAX_COALITION_SIZE,
    ) -> Coalition | None:
        """
        Form a coalition for a task.

        Args:
            task: Task description
            leader_id: Preferred leader (or auto-select)
            required_roles: Required agent roles
            max_size: Maximum coalition size

        Returns:
            Formed coalition or None
        """
        self._coalition_counter += 1
        coalition_id = f"coalition_{self._coalition_counter:06d}"

        # Select leader
        if leader_id is None or leader_id not in self._agents:
            # Select coordinator agent if available
            coordinator_agents = [
                a
                for a in self._agents.values()
                if a.role == AgentRole.COORDINATOR and a.status == AgentStatus.IDLE
            ]
            if coordinator_agents:
                leader_id = coordinator_agents[0].agent_id
            else:
                # Select any idle agent
                idle_agents = [a for a in self._agents.values() if a.status == AgentStatus.IDLE]
                if idle_agents:
                    leader_id = idle_agents[0].agent_id
                else:
                    return None

        # Select members based on required roles
        member_ids = [leader_id]

        if required_roles:
            for role in required_roles:
                role_agents = [
                    a
                    for a in self._agents.values()
                    if a.role == role
                    and a.agent_id not in member_ids
                    and a.status == AgentStatus.IDLE
                ]
                if role_agents:
                    member_ids.append(role_agents[0].agent_id)
        else:
            # Add available idle agents up to max_size
            idle_agents = [
                a
                for a in self._agents.values()
                if a.agent_id not in member_ids and a.status == AgentStatus.IDLE
            ]
            for agent in idle_agents[: max_size - len(member_ids)]:
                member_ids.append(agent.agent_id)

        # Update agent statuses
        for agent_id in member_ids:
            self._agents[agent_id].status = AgentStatus.COORDINATING

        coalition = Coalition(
            coalition_id=coalition_id,
            leader_id=leader_id,
            member_ids=member_ids,
            task=task,
        )

        self._coalitions[coalition_id] = coalition
        self._stats["coalitions_formed"] += 1

        logger.info(
            f"Formed coalition {coalition_id} with {len(member_ids)} members (leader={leader_id})"
        )

        return coalition

    def dissolve_coalition(self, coalition_id: str) -> None:
        """
        Dissolve a coalition.

        Args:
            coalition_id: Coalition to dissolve
        """
        if coalition_id not in self._coalitions:
            return

        coalition = self._coalitions[coalition_id]

        # Reset agent statuses
        for agent_id in coalition.member_ids:
            if agent_id in self._agents:
                self._agents[agent_id].status = AgentStatus.IDLE

        coalition.status = "dissolved"
        del self._coalitions[coalition_id]

        logger.info(f"Dissolved coalition {coalition_id}")

    def coordinate_detection(
        self,
        data: np.ndarray,
        context: dict[str, Any] | None = None,
        agent_ids: list[str] | None = None,
    ) -> ConsensusResult:
        """
        Coordinate detection across multiple agents.

        Args:
            data: Input data for detection
            context: Detection context
            agent_ids: Specific agents to use (or all available)

        Returns:
            Consensus result from coordinated detection
        """
        self._stats["detections_coordinated"] += 1

        # Select agents
        if agent_ids:
            agents = [self._agents[aid] for aid in agent_ids if aid in self._agents]
        else:
            agents = [
                a
                for a in self._agents.values()
                if a.status in [AgentStatus.IDLE, AgentStatus.ACTIVE]
            ]

        if not agents:
            logger.warning("No available agents for detection")
            return ConsensusResult(
                consensus_id="none",
                final_decision=False,
                confidence=0.0,
                agreement_ratio=0.0,
                participant_count=0,
                method_used=self.consensus_method,
            )

        # Collect results from all agents
        results: list[DetectionResult] = []
        for agent in agents:
            try:
                result = agent.detect(data, context)
                results.append(result)
            except Exception as e:
                logger.error(f"Agent {agent.agent_id} detection error: {e}")

        # Reach consensus
        consensus_result = self.consensus_protocol.reach_consensus(results)
        if not isinstance(consensus_result, ConsensusResult):
            raise TypeError("Expected ConsensusResult from reach_consensus")
        consensus = consensus_result
        self._stats["consensus_reached"] += 1

        return consensus

    def get_agent_by_role(self, role: AgentRole) -> list[DetectionAgent]:
        """
        Get agents by role.

        Args:
            role: Agent role to filter by

        Returns:
            List of agents with specified role
        """
        return [a for a in self._agents.values() if a.role == role]

    def get_available_agents(self) -> list[DetectionAgent]:
        """Get all available (idle) agents."""
        return [a for a in self._agents.values() if a.status == AgentStatus.IDLE]

    def get_statistics(self) -> dict[str, Any]:
        """Get coordinator statistics."""
        return {
            **self._stats,
            "registered_agents": len(self._agents),
            "active_coalitions": len(self._coalitions),
            "agents_by_role": {role.value: len(self.get_agent_by_role(role)) for role in AgentRole},
            "agents_by_status": {
                status.value: sum(1 for a in self._agents.values() if a.status == status)
                for status in AgentStatus
            },
        }


# =============================================================================
# Multi-Agent Detection System
# =============================================================================


class MultiAgentDetectionSystem:
    """
    Complete multi-agent anomaly detection system.

    Provides high-level interface for multi-agent detection with automatic agent management and
    coordination.
    """

    def __init__(
        self,
        num_agents: int = 5,
        strategy: CoordinationStrategy = CoordinationStrategy.DISTRIBUTED,
        consensus_method: ConsensusMethod = ConsensusMethod.CONFIDENCE_WEIGHTED,
        seed: int | None = None,
    ):
        """
        Initialize multi-agent detection system.

        Args:
            num_agents: Number of detection agents
            strategy: Coordination strategy
            consensus_method: Consensus method
            seed: Optional seed for the per-instance ``Generator`` driving
                agent threshold jitter. ``None`` (default) uses an
                OS-seeded ``Generator`` — same effective behavior as
                before.
        """
        self.coordinator = AgentCoordinator(
            strategy=strategy,
            consensus_method=consensus_method,
        )
        self._rng: np.random.Generator = np.random.default_rng(seed)

        # Create diverse set of agents
        self._create_agents(num_agents)

        self._detection_counter = 0

        logger.info(f"MultiAgentDetectionSystem initialized with {num_agents} agents")

    def _create_agents(self, num_agents: int) -> None:
        """Create diverse set of detection agents."""
        roles = [
            AgentRole.STATISTICAL,
            AgentRole.TEMPORAL,
            AgentRole.DIMENSIONAL,
            AgentRole.BEHAVIORAL,
            AgentRole.VALIDATOR,
        ]

        for i in range(num_agents):
            role = roles[i % len(roles)]
            agent = SimpleDetectionAgent(
                agent_id=f"agent_{i:03d}",
                role=role,
                threshold=0.5 + self._rng.uniform(-0.1, 0.1),
            )
            self.coordinator.register_agent(agent)

    def detect(
        self,
        data: np.ndarray,
        context: dict[str, Any] | None = None,
        use_coalition: bool = False,
    ) -> dict[str, Any]:
        """
        Perform multi-agent detection.

        Args:
            data: Input data
            context: Detection context
            use_coalition: Whether to form coalition for task

        Returns:
            Detection result with consensus information
        """
        self._detection_counter += 1

        if use_coalition:
            # Form coalition for this detection
            coalition = self.coordinator.form_coalition(
                task={"type": "detection", "data_shape": data.shape},
                required_roles=[AgentRole.STATISTICAL, AgentRole.TEMPORAL],
            )

            if coalition:
                result = self.coordinator.coordinate_detection(data, context, coalition.member_ids)
                self.coordinator.dissolve_coalition(coalition.coalition_id)
            else:
                result = self.coordinator.coordinate_detection(data, context)
        else:
            result = self.coordinator.coordinate_detection(data, context)

        return {
            "is_anomaly": result.final_decision,
            "confidence": result.confidence,
            "agreement_ratio": result.agreement_ratio,
            "participant_count": result.participant_count,
            "consensus_method": result.method_used.value,
            "dissenting_agents": result.dissenting_agents,
            "detection_id": f"detection_{self._detection_counter:06d}",
            "consensus_decision": result.final_decision,
            "individual_results": (
                [
                    {"agent_id": agent_id, "decision": vote}
                    for agent_id, vote in getattr(result, "votes", {}).items()
                ]
                if hasattr(result, "votes")
                else []
            ),
        }

    def register_agent(self, agent: DetectionAgent) -> bool:
        """
        Register a detection agent (simplified API).

        Args:
            agent: Detection agent to register

        Returns:
            True if registered successfully
        """
        return self.coordinator.register_agent(agent)

    def add_agent(
        self,
        role: AgentRole = AgentRole.STATISTICAL,
        threshold: float = 0.5,
    ) -> str | None:
        """
        Add a new detection agent.

        Args:
            role: Agent role
            threshold: Detection threshold

        Returns:
            Agent ID or None if failed
        """
        agent_id = f"agent_{len(self.coordinator._agents):03d}"
        agent = SimpleDetectionAgent(agent_id=agent_id, role=role, threshold=threshold)

        if self.coordinator.register_agent(agent):
            return agent_id
        return None

    def remove_agent(self, agent_id: str) -> bool:
        """
        Remove a detection agent.

        Args:
            agent_id: Agent ID to remove

        Returns:
            True if removed successfully
        """
        return self.coordinator.unregister_agent(agent_id)

    def get_statistics(self) -> dict[str, Any]:
        """Get system statistics."""
        return {
            "total_detections": self._detection_counter,
            "coordinator": self.coordinator.get_statistics(),
        }
