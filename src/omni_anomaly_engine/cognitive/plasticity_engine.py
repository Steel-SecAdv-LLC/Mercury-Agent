"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

"""
Plasticity Engine - Dynamic Knowledge Adaptation

Inspired by Nucleoid's plasticity framework, this module provides:
- Dynamic knowledge base updates when encountering new information
- Adaptive logic that modifies reasoning strategies based on outcomes
- Generalization allowing transfer to unseen scenarios
- Hebbian-inspired learning: "neurons that fire together wire together"

Research Sources:
- Nucleoid: https://github.com/NucleoidAI/Nucleoid
- DARPA ANSR: Adaptive representations
- Hebbian Learning: Connection strength adaptation
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


class AdaptationType(Enum):
    """Types of knowledge adaptation."""

    STRUCTURAL = "structural"  # New connections/nodes
    PARAMETRIC = "parametric"  # Weight adjustments
    SEMANTIC = "semantic"  # Meaning refinement
    TEMPORAL = "temporal"  # Time-dependent changes
    CONTEXTUAL = "contextual"  # Context-specific adaptations


class PlasticityMode(Enum):
    """Operating modes for plasticity."""

    LEARNING = "learning"  # Active learning mode
    CONSOLIDATION = "consolidation"  # Memory consolidation
    RECALL = "recall"  # Fast retrieval mode
    ADAPTATION = "adaptation"  # Active schema modification


@dataclass
class AdaptationEvent:
    """Record of a knowledge adaptation event."""

    event_id: str
    timestamp: float
    adaptation_type: AdaptationType
    source_pattern: str
    target_pattern: str
    strength_delta: float
    confidence: float
    context: dict[str, Any] = field(default_factory=dict)
    outcome: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "type": self.adaptation_type.value,
            "source": self.source_pattern,
            "target": self.target_pattern,
            "delta": self.strength_delta,
            "confidence": self.confidence,
            "context": self.context,
            "outcome": self.outcome,
        }


@dataclass
class PlasticConnection:
    """A plastic (adaptable) connection between knowledge elements."""

    source_id: str
    target_id: str
    weight: float
    activation_count: int = 0
    last_activated: float = 0.0
    creation_time: float = field(default_factory=time.time)
    decay_rate: float = 0.01  # Forgetting rate
    consolidation_level: float = 0.0  # 0=volatile, 1=consolidated

    def activate(self, strength: float = 1.0) -> None:
        """Activate this connection (Hebbian strengthening)."""
        self.activation_count += 1
        self.last_activated = time.time()
        # Hebbian update: strengthen connections that fire together
        self.weight = min(1.0, self.weight + 0.1 * strength * (1 - self.weight))
        # Consolidation increases with repeated activation
        self.consolidation_level = min(
            1.0, self.consolidation_level + 0.05 * self.activation_count
        )

    def decay(self, time_delta: float) -> None:
        """Apply decay based on time since last activation."""
        if self.consolidation_level < 0.5:  # Only decay unconsolidated connections
            decay_factor = np.exp(-self.decay_rate * time_delta)
            self.weight *= decay_factor


class PlasticityEngine:
    """
    Dynamic knowledge adaptation engine inspired by Nucleoid.

    Implements neural plasticity principles:
    - Hebbian learning: Co-activated patterns strengthen connections
    - Synaptic plasticity: Connection weights adapt based on outcomes
    - Structural plasticity: New connections form, weak ones prune
    - Metaplasticity: Learning rate adapts based on experience

    This enables the system to:
    - Learn new patterns from experience
    - Adapt reasoning strategies based on outcomes
    - Generalize to unseen scenarios
    - Forget irrelevant information over time
    """

    # Golden ratio for optimal information distribution
    PHI = (1 + np.sqrt(5)) / 2

    def __init__(
        self,
        learning_rate: float = 0.1,
        decay_rate: float = 0.01,
        consolidation_threshold: float = 0.7,
        max_connections: int = 10000,
        enable_pruning: bool = True,
        enable_consolidation: bool = True,
    ):
        """
        Initialize Plasticity Engine.

        Args:
            learning_rate: Base learning rate for adaptation
            decay_rate: Rate of forgetting for unconsolidated memories
            consolidation_threshold: Threshold for memory consolidation
            max_connections: Maximum number of plastic connections
            enable_pruning: Enable weak connection pruning
            enable_consolidation: Enable memory consolidation
        """
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.consolidation_threshold = consolidation_threshold
        self.max_connections = max_connections
        self.enable_pruning = enable_pruning
        self.enable_consolidation = enable_consolidation

        # Core data structures
        self._connections: dict[tuple[str, str], PlasticConnection] = {}
        self._node_activations: dict[str, list[float]] = defaultdict(list)
        self._adaptation_history: list[AdaptationEvent] = []
        self._outcome_feedback: dict[str, list[tuple[bool, float]]] = defaultdict(list)

        # Metaplasticity state
        self._global_learning_rate = learning_rate
        self._recent_error_rate = 0.0
        self._mode = PlasticityMode.LEARNING

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            "total_adaptations": 0,
            "connections_created": 0,
            "connections_pruned": 0,
            "consolidations": 0,
        }

        logger.info(
            f"PlasticityEngine initialized (lr={learning_rate}, "
            f"decay={decay_rate}, max_conn={max_connections})"
        )

    def adapt(
        self,
        source_pattern: str,
        target_pattern: str,
        strength: float = 1.0,
        adaptation_type: AdaptationType = AdaptationType.PARAMETRIC,
        context: dict[str, Any] | None = None,
    ) -> AdaptationEvent:
        """
        Adapt knowledge by strengthening/creating a connection.

        Implements Hebbian learning: patterns that occur together
        strengthen their connection.

        Args:
            source_pattern: Source pattern/concept identifier
            target_pattern: Target pattern/concept identifier
            strength: Strength of the adaptation (0-1)
            adaptation_type: Type of adaptation
            context: Optional context for this adaptation

        Returns:
            AdaptationEvent recording this adaptation
        """
        with self._lock:
            connection_key = (source_pattern, target_pattern)

            # Get or create connection
            if connection_key not in self._connections:
                if len(self._connections) >= self.max_connections:
                    self._prune_weakest()

                self._connections[connection_key] = PlasticConnection(
                    source_id=source_pattern,
                    target_id=target_pattern,
                    weight=0.1,  # Start with weak connection
                    decay_rate=self.decay_rate,
                )
                self._stats["connections_created"] += 1

            connection = self._connections[connection_key]
            old_weight = connection.weight

            # Hebbian update with metaplasticity
            effective_lr = self._global_learning_rate * strength
            connection.activate(effective_lr)

            # Record activation for pattern nodes
            self._node_activations[source_pattern].append(time.time())
            self._node_activations[target_pattern].append(time.time())

            # Create adaptation event
            event = AdaptationEvent(
                event_id=f"adapt_{self._stats['total_adaptations']}",
                timestamp=time.time(),
                adaptation_type=adaptation_type,
                source_pattern=source_pattern,
                target_pattern=target_pattern,
                strength_delta=connection.weight - old_weight,
                confidence=connection.weight,
                context=context or {},
            )

            self._adaptation_history.append(event)
            self._stats["total_adaptations"] += 1

            # Consolidation check
            if (
                self.enable_consolidation
                and connection.consolidation_level >= self.consolidation_threshold
                and connection.weight >= 0.8
            ):
                self._consolidate_connection(connection_key)

            return event

    def feedback(
        self,
        pattern_id: str,
        success: bool,
        magnitude: float = 1.0,
    ) -> None:
        """
        Provide outcome feedback for a pattern (reinforcement learning).

        Adjusts metaplasticity based on prediction accuracy.

        Args:
            pattern_id: Pattern that was used
            success: Whether the prediction/action was successful
            magnitude: Magnitude of success/failure
        """
        with self._lock:
            self._outcome_feedback[pattern_id].append((success, magnitude))

            # Update metaplasticity based on recent outcomes
            recent_outcomes = []
            for outcomes in self._outcome_feedback.values():
                recent_outcomes.extend(outcomes[-10:])

            if recent_outcomes:
                successes = sum(1 for s, _ in recent_outcomes if s)
                self._recent_error_rate = 1 - (successes / len(recent_outcomes))

                # Metaplasticity: increase learning rate when error is high
                # decrease when performance is good
                if self._recent_error_rate > 0.3:
                    self._global_learning_rate = min(
                        0.5, self.learning_rate * (1 + self._recent_error_rate)
                    )
                else:
                    self._global_learning_rate = max(
                        0.01, self.learning_rate * (1 - self._recent_error_rate * 0.5)
                    )

    def query_association(
        self,
        source_pattern: str,
        depth: int = 1,
        min_strength: float = 0.1,
    ) -> dict[str, float]:
        """
        Query associated patterns with their connection strengths.

        Args:
            source_pattern: Starting pattern
            depth: How many hops to follow
            min_strength: Minimum connection strength to include

        Returns:
            Dictionary mapping pattern IDs to association strengths
        """
        with self._lock:
            associations: dict[str, float] = {}
            visited = {source_pattern}
            current_level = {source_pattern: 1.0}

            for _ in range(depth):
                next_level: dict[str, float] = {}

                for pattern, incoming_strength in current_level.items():
                    for (src, tgt), conn in self._connections.items():
                        if src == pattern and conn.weight >= min_strength:
                            if tgt not in visited:
                                combined = incoming_strength * conn.weight
                                if tgt in next_level:
                                    next_level[tgt] = max(next_level[tgt], combined)
                                else:
                                    next_level[tgt] = combined
                                visited.add(tgt)

                associations.update(next_level)
                current_level = next_level

            return associations

    def generalize(
        self,
        novel_pattern: str,
        feature_vector: np.ndarray,
        existing_patterns: dict[str, np.ndarray],
        similarity_threshold: float = 0.7,
    ) -> list[tuple[str, float]]:
        """
        Generalize from known patterns to novel pattern.

        Uses similarity-based transfer to connect novel patterns
        to existing knowledge.

        Args:
            novel_pattern: ID for the novel pattern
            feature_vector: Feature representation of novel pattern
            existing_patterns: Known patterns with their features
            similarity_threshold: Minimum similarity to create connection

        Returns:
            List of (pattern_id, similarity) for created connections
        """
        with self._lock:
            connections_made = []

            for known_id, known_vector in existing_patterns.items():
                # Cosine similarity
                if np.linalg.norm(feature_vector) > 0 and np.linalg.norm(known_vector) > 0:
                    similarity = np.dot(feature_vector, known_vector) / (
                        np.linalg.norm(feature_vector) * np.linalg.norm(known_vector)
                    )

                    if similarity >= similarity_threshold:
                        # Create bidirectional connections
                        self.adapt(
                            novel_pattern,
                            known_id,
                            strength=similarity,
                            adaptation_type=AdaptationType.STRUCTURAL,
                            context={"generalization": True, "similarity": float(similarity)},
                        )
                        self.adapt(
                            known_id,
                            novel_pattern,
                            strength=similarity * 0.5,  # Weaker reverse connection
                            adaptation_type=AdaptationType.STRUCTURAL,
                        )
                        connections_made.append((known_id, float(similarity)))

            logger.debug(
                f"Generalized {novel_pattern}: {len(connections_made)} connections"
            )
            return connections_made

    def apply_decay(self) -> int:
        """
        Apply time-based decay to all connections.

        Unconsolidated memories decay over time, while consolidated
        memories remain stable.

        Returns:
            Number of connections that were pruned due to decay
        """
        with self._lock:
            current_time = time.time()
            pruned = 0

            to_prune = []
            for key, conn in self._connections.items():
                time_delta = current_time - conn.last_activated
                conn.decay(time_delta)

                if self.enable_pruning and conn.weight < 0.05:
                    to_prune.append(key)

            for key in to_prune:
                del self._connections[key]
                pruned += 1

            self._stats["connections_pruned"] += pruned
            return pruned

    def _prune_weakest(self) -> None:
        """Prune the weakest 10% of connections."""
        if not self._connections:
            return

        # Sort by weight and consolidation
        sorted_conns = sorted(
            self._connections.items(),
            key=lambda x: x[1].weight * (1 + x[1].consolidation_level),
        )

        prune_count = max(1, len(sorted_conns) // 10)
        for key, _ in sorted_conns[:prune_count]:
            del self._connections[key]
            self._stats["connections_pruned"] += 1

    def _consolidate_connection(self, connection_key: tuple[str, str]) -> None:
        """Consolidate a connection (make it permanent)."""
        if connection_key in self._connections:
            conn = self._connections[connection_key]
            conn.consolidation_level = 1.0
            conn.decay_rate = 0.001  # Much slower decay
            self._stats["consolidations"] += 1
            logger.debug(f"Consolidated: {connection_key}")

    def set_mode(self, mode: PlasticityMode) -> None:
        """Set the operating mode."""
        with self._lock:
            old_mode = self._mode
            self._mode = mode

            if mode == PlasticityMode.CONSOLIDATION:
                # Trigger consolidation process
                self._consolidate_strong_connections()
            elif mode == PlasticityMode.RECALL:
                # Optimize for fast retrieval
                self._global_learning_rate = 0.01

            logger.info(f"PlasticityEngine mode: {old_mode.value} -> {mode.value}")

    def _consolidate_strong_connections(self) -> None:
        """Consolidate all connections above threshold."""
        with self._lock:
            for key, conn in self._connections.items():
                if conn.weight >= 0.8 and conn.activation_count >= 5:
                    self._consolidate_connection(key)

    def get_statistics(self) -> dict[str, Any]:
        """Get engine statistics."""
        with self._lock:
            return {
                **self._stats,
                "active_connections": len(self._connections),
                "global_learning_rate": self._global_learning_rate,
                "recent_error_rate": self._recent_error_rate,
                "mode": self._mode.value,
                "consolidated_connections": sum(
                    1 for c in self._connections.values() if c.consolidation_level >= 1.0
                ),
            }

    def export_graph(self) -> dict[str, Any]:
        """Export the plasticity graph for visualization."""
        with self._lock:
            nodes = set()
            edges = []

            for (src, tgt), conn in self._connections.items():
                nodes.add(src)
                nodes.add(tgt)
                edges.append(
                    {
                        "source": src,
                        "target": tgt,
                        "weight": conn.weight,
                        "consolidated": conn.consolidation_level >= 1.0,
                        "activations": conn.activation_count,
                    }
                )

            return {
                "nodes": list(nodes),
                "edges": edges,
                "statistics": self.get_statistics(),
            }
