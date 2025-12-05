"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

"""
Plasticity Engine - Production Implementation

Neural plasticity mechanisms for dynamic knowledge adaptation:
- STDP: Spike-Timing Dependent Plasticity
- BCM Theory: Bienenstock-Cooper-Munro sliding threshold
- Hebbian Learning: Correlation-based strengthening
- Metaplasticity: Learning rate adaptation
- Eligibility Traces: Temporal credit assignment
- Synaptic Tagging: Memory consolidation

Research Sources:
- Bi & Poo (1998): STDP in hippocampal neurons
- Bienenstock, Cooper, Munro (1982): BCM theory
- Frey & Morris (1997): Synaptic tagging and capture
- Redish (2004): Eligibility traces in RL
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

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


class PlasticityRule(Enum):
    """Synaptic plasticity rules."""

    HEBBIAN = "hebbian"  # Classic fire-together-wire-together
    STDP = "stdp"  # Spike-timing dependent
    BCM = "bcm"  # Bienenstock-Cooper-Munro
    OJA = "oja"  # Oja's normalized Hebbian


@dataclass
class STDPParameters:
    """Parameters for Spike-Timing Dependent Plasticity."""

    a_plus: float = 0.1  # LTP amplitude
    a_minus: float = 0.12  # LTD amplitude (slightly stronger)
    tau_plus: float = 20.0  # LTP time constant (ms)
    tau_minus: float = 20.0  # LTD time constant (ms)
    w_max: float = 1.0  # Maximum weight
    w_min: float = 0.0  # Minimum weight

    def compute_delta_w(self, delta_t: float, w_current: float) -> float:
        """
        Compute weight change based on spike timing.

        Args:
            delta_t: t_post - t_pre (positive = pre before post = LTP)
            w_current: Current weight

        Returns:
            Weight change (can be positive or negative)
        """
        if delta_t > 0:
            # Pre before post: LTP (potentiation)
            # Weight-dependent scaling (soft bounds)
            return self.a_plus * (self.w_max - w_current) * np.exp(-delta_t / self.tau_plus)
        else:
            # Post before pre: LTD (depression)
            return -self.a_minus * (w_current - self.w_min) * np.exp(delta_t / self.tau_minus)


@dataclass
class BCMParameters:
    """Parameters for BCM (Bienenstock-Cooper-Munro) theory."""

    eta: float = 0.01  # Learning rate
    tau_theta: float = 100.0  # Time constant for threshold adaptation
    theta_init: float = 0.5  # Initial modification threshold
    p: float = 2.0  # Power for threshold computation

    def compute_theta(
        self,
        activity_history: list[float],
        current_theta: float,
    ) -> float:
        """
        Update modification threshold based on activity history.

        BCM: theta = E[y^p] (sliding threshold based on recent activity)
        """
        if not activity_history:
            return current_theta

        # Exponential moving average of y^p
        recent = activity_history[-100:]  # Use last 100 activities
        mean_activity_p = np.mean([y**self.p for y in recent])

        # Update theta toward this value
        alpha = 1.0 / self.tau_theta
        return (1 - alpha) * current_theta + alpha * mean_activity_p

    def compute_delta_w(
        self,
        pre_activity: float,
        post_activity: float,
        theta: float,
    ) -> float:
        """
        Compute BCM weight change.

        phi(y, theta) = y(y - theta)
        Delta_w = eta * phi(post, theta) * pre
        """
        phi = post_activity * (post_activity - theta)
        return self.eta * phi * pre_activity


@dataclass
class EligibilityTrace:
    """Eligibility trace for temporal credit assignment."""

    trace: float = 0.0
    decay_rate: float = 0.95  # Trace decay per timestep
    last_update: float = field(default_factory=time.time)

    def update(self, activation: float) -> None:
        """Update trace with new activation."""
        # Apply decay since last update
        elapsed = time.time() - self.last_update
        decay_steps = int(elapsed * 10)  # 10 steps per second
        self.trace *= self.decay_rate**decay_steps

        # Add new activation
        self.trace += activation
        self.last_update = time.time()

    def get_trace(self) -> float:
        """Get current trace value with decay applied."""
        elapsed = time.time() - self.last_update
        decay_steps = int(elapsed * 10)
        return self.trace * (self.decay_rate**decay_steps)


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
    rule_used: PlasticityRule = PlasticityRule.HEBBIAN

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
            "rule": self.rule_used.value,
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
    eligibility: EligibilityTrace = field(default_factory=EligibilityTrace)
    # Timing for STDP
    last_pre_spike: float = 0.0
    last_post_spike: float = 0.0
    # Tag for synaptic tagging and capture
    tagged: bool = False
    tag_time: float = 0.0

    def activate_pre(self, strength: float = 1.0) -> None:
        """Record presynaptic activation."""
        self.last_pre_spike = time.time()
        self.eligibility.update(strength)

    def activate_post(self, strength: float = 1.0) -> None:
        """Record postsynaptic activation."""
        self.last_post_spike = time.time()
        self.activation_count += 1
        self.last_activated = time.time()

    def compute_stdp_delta(self, params: STDPParameters) -> float:
        """Compute STDP weight change based on spike times."""
        if self.last_pre_spike == 0 or self.last_post_spike == 0:
            return 0.0

        # Delta t in milliseconds
        delta_t = (self.last_post_spike - self.last_pre_spike) * 1000
        return params.compute_delta_w(delta_t, self.weight)

    def apply_hebbian_update(self, strength: float, learning_rate: float) -> float:
        """
        Apply Hebbian weight update with soft bounds.

        Returns the weight change.
        """
        old_weight = self.weight
        # Soft bound Hebbian: dw = lr * x * (w_max - w)
        delta = learning_rate * strength * (1.0 - self.weight)
        self.weight = min(1.0, max(0.0, self.weight + delta))

        # Update consolidation
        self.consolidation_level = min(1.0, self.consolidation_level + 0.02 * self.activation_count)

        return self.weight - old_weight

    def decay(self, time_delta: float) -> None:
        """Apply decay based on time since last activation."""
        if self.consolidation_level < 0.5:  # Only decay unconsolidated
            decay_factor = np.exp(-self.decay_rate * time_delta)
            self.weight *= decay_factor

    def tag_for_consolidation(self) -> None:
        """Tag this synapse for potential late-phase consolidation."""
        self.tagged = True
        self.tag_time = time.time()


class CompetitiveLearning:
    """
    Competitive learning with lateral inhibition.

    Implements winner-take-all dynamics for pattern separation.
    """

    def __init__(self, n_units: int = 100, inhibition_strength: float = 0.1):
        self.n_units = n_units
        self.inhibition_strength = inhibition_strength
        self.activities = np.zeros(n_units)
        self.winner_history: list[int] = []

    def compete(self, inputs: np.ndarray) -> np.ndarray:
        """
        Apply competitive dynamics.

        Args:
            inputs: Input activations for each unit

        Returns:
            Post-competition activities
        """
        # Initial activities
        activities = inputs.copy()

        # Lateral inhibition (soft winner-take-all)
        for _ in range(10):  # Iterate until convergence
            # Inhibition proportional to total activity
            total_activity = np.sum(activities)
            inhibition = self.inhibition_strength * total_activity / self.n_units

            # Apply inhibition
            activities = np.maximum(0, inputs - inhibition)

            # Normalize
            if np.sum(activities) > 0:
                activities = activities / np.sum(activities)

        # Track winner
        winner = int(np.argmax(activities))
        self.winner_history.append(winner)

        self.activities = activities
        return activities

    def get_winner(self) -> int:
        """Get the current winner."""
        return int(np.argmax(self.activities))


class PlasticityEngine:
    """
    Production Neural Plasticity Engine.

    Implements biologically-inspired learning rules:

    1. STDP (Spike-Timing Dependent Plasticity)
       - Pre-before-post: LTP (strengthening)
       - Post-before-pre: LTD (weakening)
       - Asymmetric learning window

    2. BCM Theory (Bienenstock-Cooper-Munro)
       - Sliding modification threshold
       - Activity-dependent metaplasticity
       - Prevents runaway excitation/inhibition

    3. Hebbian Learning
       - "Fire together, wire together"
       - Correlation-based strengthening
       - Soft weight bounds

    4. Eligibility Traces
       - Temporal credit assignment
       - Bridge gap between action and reward
       - Enable three-factor learning

    5. Synaptic Tagging and Capture
       - Two-stage memory consolidation
       - Tag-dependent late-phase LTP
       - Protein synthesis simulation

    6. Competitive Learning
       - Lateral inhibition
       - Winner-take-all dynamics
       - Pattern separation
    """

    PHI = (1 + np.sqrt(5)) / 2  # Golden ratio

    def __init__(
        self,
        learning_rate: float = 0.1,
        decay_rate: float = 0.01,
        consolidation_threshold: float = 0.7,
        max_connections: int = 10000,
        enable_pruning: bool = True,
        enable_consolidation: bool = True,
        plasticity_rule: PlasticityRule = PlasticityRule.STDP,
        enable_competition: bool = True,
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
            plasticity_rule: Primary plasticity rule to use
            enable_competition: Enable competitive learning
        """
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.consolidation_threshold = consolidation_threshold
        self.max_connections = max_connections
        self.enable_pruning = enable_pruning
        self.enable_consolidation = enable_consolidation
        self.plasticity_rule = plasticity_rule
        self.enable_competition = enable_competition

        # Plasticity parameters
        self.stdp_params = STDPParameters()
        self.bcm_params = BCMParameters(eta=learning_rate)

        # Core data structures
        self._connections: dict[tuple[str, str], PlasticConnection] = {}
        self._node_activations: dict[str, list[float]] = defaultdict(list)
        self._node_spike_times: dict[str, list[float]] = defaultdict(list)
        self._adaptation_history: list[AdaptationEvent] = []
        self._outcome_feedback: dict[str, list[tuple[bool, float]]] = defaultdict(list)

        # BCM state: sliding thresholds per node
        self._bcm_thresholds: dict[str, float] = defaultdict(lambda: 0.5)

        # Competitive learning
        self._competition = CompetitiveLearning() if enable_competition else None

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
            "stdp_updates": 0,
            "bcm_updates": 0,
            "tagged_synapses": 0,
        }

        logger.info(
            f"PlasticityEngine initialized (rule={plasticity_rule.value}, lr={learning_rate})"
        )

    def adapt(
        self,
        source_pattern: str,
        target_pattern: str,
        strength: float = 1.0,
        adaptation_type: AdaptationType = AdaptationType.PARAMETRIC,
        context: dict[str, Any] | None = None,
        use_stdp: bool = True,
    ) -> AdaptationEvent:
        """
        Adapt knowledge by strengthening/creating a connection.

        Applies the configured plasticity rule (STDP, BCM, or Hebbian).

        Args:
            source_pattern: Source pattern/concept identifier
            target_pattern: Target pattern/concept identifier
            strength: Strength of the adaptation (0-1)
            adaptation_type: Type of adaptation
            context: Optional context for this adaptation
            use_stdp: Whether to use STDP timing

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
                    weight=0.1,
                    decay_rate=self.decay_rate,
                )
                self._stats["connections_created"] += 1

            connection = self._connections[connection_key]
            _old_weight = connection.weight

            # Record spike times
            current_time = time.time()
            self._node_spike_times[source_pattern].append(current_time)
            self._node_spike_times[target_pattern].append(current_time)

            # Record activations for BCM
            self._node_activations[source_pattern].append(strength)
            self._node_activations[target_pattern].append(strength)

            # Apply plasticity rule
            rule_used = self.plasticity_rule
            if self.plasticity_rule == PlasticityRule.STDP and use_stdp:
                delta_w = self._apply_stdp(connection, source_pattern, target_pattern, strength)
                self._stats["stdp_updates"] += 1

            elif self.plasticity_rule == PlasticityRule.BCM:
                delta_w = self._apply_bcm(connection, source_pattern, target_pattern, strength)
                self._stats["bcm_updates"] += 1

            else:  # Hebbian
                effective_lr = self._global_learning_rate * strength
                delta_w = connection.apply_hebbian_update(strength, effective_lr)

            # Record post-activation
            connection.activate_post(strength)

            # Check for synaptic tagging
            if abs(delta_w) > 0.1 and not connection.tagged:
                connection.tag_for_consolidation()
                self._stats["tagged_synapses"] += 1

            # Create adaptation event
            event = AdaptationEvent(
                event_id=f"adapt_{self._stats['total_adaptations']}",
                timestamp=current_time,
                adaptation_type=adaptation_type,
                source_pattern=source_pattern,
                target_pattern=target_pattern,
                strength_delta=delta_w,
                confidence=connection.weight,
                context=context or {},
                rule_used=rule_used,
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

    def _apply_stdp(
        self,
        connection: PlasticConnection,
        source: str,
        target: str,
        strength: float,
    ) -> float:
        """Apply STDP learning rule."""
        # Get recent spike times
        pre_times = self._node_spike_times.get(source, [])
        post_times = self._node_spike_times.get(target, [])

        if not pre_times or not post_times:
            # Fall back to Hebbian
            return connection.apply_hebbian_update(strength, self._global_learning_rate)

        # Use most recent spikes
        t_pre = pre_times[-1]
        t_post = post_times[-1] if len(post_times) > 1 else time.time()

        # Record spike times for future STDP
        connection.activate_pre(strength)

        # Compute STDP delta
        delta_t = (t_post - t_pre) * 1000  # Convert to ms

        # Only apply if spikes are close enough in time
        if abs(delta_t) < 100:  # 100ms window
            delta_w = self.stdp_params.compute_delta_w(delta_t, connection.weight)

            # Apply with eligibility trace modulation
            trace = connection.eligibility.get_trace()
            delta_w *= 1 + trace

            connection.weight = np.clip(
                connection.weight + delta_w, self.stdp_params.w_min, self.stdp_params.w_max
            )
            return delta_w

        return 0.0

    def _apply_bcm(
        self,
        connection: PlasticConnection,
        source: str,
        target: str,
        strength: float,
    ) -> float:
        """Apply BCM learning rule."""
        # Update sliding threshold for target node
        post_history = self._node_activations.get(target, [])
        current_theta = self._bcm_thresholds[target]
        new_theta = self.bcm_params.compute_theta(post_history, current_theta)
        self._bcm_thresholds[target] = new_theta

        # Compute BCM weight change
        pre_activity = strength
        post_activity = post_history[-1] if post_history else strength

        delta_w = self.bcm_params.compute_delta_w(pre_activity, post_activity, new_theta)

        # Apply with bounds
        old_weight = connection.weight
        connection.weight = np.clip(connection.weight + delta_w, 0.0, 1.0)

        return connection.weight - old_weight

    def feedback(
        self,
        pattern_id: str,
        success: bool,
        magnitude: float = 1.0,
    ) -> None:
        """
        Provide outcome feedback for a pattern (reinforcement learning).

        Uses eligibility traces to assign credit to recent adaptations.

        Args:
            pattern_id: Pattern that was used
            success: Whether the prediction/action was successful
            magnitude: Magnitude of success/failure
        """
        with self._lock:
            self._outcome_feedback[pattern_id].append((success, magnitude))

            # Apply three-factor learning: eligibility * reward
            reward_signal = magnitude if success else -magnitude * 0.5

            for (src, tgt), conn in self._connections.items():
                if src == pattern_id or tgt == pattern_id:
                    trace = conn.eligibility.get_trace()
                    if trace > 0.1:
                        # Modulate weight by reward * eligibility
                        delta = self._global_learning_rate * reward_signal * trace
                        conn.weight = np.clip(conn.weight + delta, 0.0, 1.0)

            # Update metaplasticity
            self._update_metaplasticity()

    def _update_metaplasticity(self) -> None:
        """Update global learning rate based on recent performance."""
        recent_outcomes = []
        for outcomes in self._outcome_feedback.values():
            recent_outcomes.extend(outcomes[-20:])

        if recent_outcomes:
            successes = sum(1 for s, _ in recent_outcomes if s)
            self._recent_error_rate = 1 - (successes / len(recent_outcomes))

            # BCM-inspired: high error -> increase LR, low error -> decrease
            if self._recent_error_rate > 0.3:
                self._global_learning_rate = min(
                    0.5, self.learning_rate * (1 + self._recent_error_rate)
                )
            else:
                self._global_learning_rate = max(
                    0.01, self.learning_rate * (1 - self._recent_error_rate * 0.5)
                )

            # Update BCM learning rate
            self.bcm_params.eta = self._global_learning_rate

    def query_association(
        self,
        source_pattern: str,
        depth: int = 1,
        min_strength: float = 0.1,
    ) -> dict[str, float]:
        """
        Query associated patterns with their connection strengths.

        Uses spreading activation with eligibility trace weighting.

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
                                # Weight by eligibility trace
                                trace_factor = 1 + conn.eligibility.get_trace()
                                combined = incoming_strength * conn.weight * trace_factor
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

        Uses competitive learning if enabled.

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

            # Compute similarities
            similarities = []
            for known_id, known_vector in existing_patterns.items():
                if np.linalg.norm(feature_vector) > 0 and np.linalg.norm(known_vector) > 0:
                    similarity = np.dot(feature_vector, known_vector) / (
                        np.linalg.norm(feature_vector) * np.linalg.norm(known_vector)
                    )
                    similarities.append((known_id, float(similarity)))

            # Apply competition if enabled
            if self._competition and similarities:
                sim_array = np.array([s for _, s in similarities])
                competed = self._competition.compete(sim_array)

                # Use competed values
                for i, (known_id, _) in enumerate(similarities):
                    if competed[i] >= similarity_threshold:
                        self.adapt(
                            novel_pattern,
                            known_id,
                            strength=competed[i],
                            adaptation_type=AdaptationType.STRUCTURAL,
                            context={"generalization": True, "competed": True},
                        )
                        self.adapt(
                            known_id,
                            novel_pattern,
                            strength=competed[i] * 0.5,
                            adaptation_type=AdaptationType.STRUCTURAL,
                        )
                        connections_made.append((known_id, competed[i]))
            else:
                # Standard generalization without competition
                for known_id, similarity in similarities:
                    if similarity >= similarity_threshold:
                        self.adapt(
                            novel_pattern,
                            known_id,
                            strength=similarity,
                            adaptation_type=AdaptationType.STRUCTURAL,
                            context={"generalization": True},
                        )
                        self.adapt(
                            known_id,
                            novel_pattern,
                            strength=similarity * 0.5,
                            adaptation_type=AdaptationType.STRUCTURAL,
                        )
                        connections_made.append((known_id, similarity))

            logger.debug(f"Generalized {novel_pattern}: {len(connections_made)} connections")
            return connections_made

    def consolidate_tagged(self) -> int:
        """
        Consolidate tagged synapses (synaptic tagging and capture).

        Implements late-phase LTP for tagged synapses.

        Returns:
            Number of synapses consolidated
        """
        with self._lock:
            consolidated = 0
            current_time = time.time()

            for key, conn in self._connections.items():
                if conn.tagged:
                    # Check if tag is still valid (decays over time)
                    tag_age = current_time - conn.tag_time
                    if tag_age < 3600:  # 1 hour window
                        # Consolidate: increase weight and permanence
                        conn.weight = min(1.0, conn.weight * 1.2)
                        conn.consolidation_level = 1.0
                        conn.decay_rate = 0.001
                        conn.tagged = False
                        consolidated += 1
                        self._stats["consolidations"] += 1
                    else:
                        # Tag expired
                        conn.tagged = False

            return consolidated

    def apply_decay(self) -> int:
        """
        Apply time-based decay to all connections.

        Returns:
            Number of connections that were pruned
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

        sorted_conns = sorted(
            self._connections.items(),
            key=lambda x: x[1].weight * (1 + x[1].consolidation_level),
        )

        prune_count = max(1, len(sorted_conns) // 10)
        for key, _ in sorted_conns[:prune_count]:
            del self._connections[key]
            self._stats["connections_pruned"] += 1

    def _consolidate_connection(self, connection_key: tuple[str, str]) -> None:
        """Consolidate a connection."""
        if connection_key in self._connections:
            conn = self._connections[connection_key]
            conn.consolidation_level = 1.0
            conn.decay_rate = 0.001
            self._stats["consolidations"] += 1
            logger.debug(f"Consolidated: {connection_key}")

    def set_mode(self, mode: PlasticityMode) -> None:
        """Set the operating mode."""
        with self._lock:
            old_mode = self._mode
            self._mode = mode

            if mode == PlasticityMode.CONSOLIDATION:
                self.consolidate_tagged()
                self._consolidate_strong_connections()
            elif mode == PlasticityMode.RECALL:
                self._global_learning_rate = 0.01
                self.bcm_params.eta = 0.01

            logger.info(f"PlasticityEngine mode: {old_mode.value} -> {mode.value}")

    def _consolidate_strong_connections(self) -> None:
        """Consolidate all connections above threshold."""
        with self._lock:
            for key, conn in self._connections.items():
                if conn.weight >= 0.8 and conn.activation_count >= 5:
                    self._consolidate_connection(key)

    def get_connection_stats(self) -> dict[str, Any]:
        """Get detailed connection statistics."""
        with self._lock:
            weights = [c.weight for c in self._connections.values()]
            consolidation = [c.consolidation_level for c in self._connections.values()]

            return {
                "total": len(self._connections),
                "mean_weight": float(np.mean(weights)) if weights else 0,
                "std_weight": float(np.std(weights)) if weights else 0,
                "max_weight": float(np.max(weights)) if weights else 0,
                "min_weight": float(np.min(weights)) if weights else 0,
                "consolidated_pct": (
                    100 * sum(1 for c in consolidation if c >= 1.0) / len(consolidation)
                    if consolidation
                    else 0
                ),
                "tagged_count": sum(1 for c in self._connections.values() if c.tagged),
            }

    def get_statistics(self) -> dict[str, Any]:
        """Get engine statistics."""
        with self._lock:
            conn_stats = self.get_connection_stats()
            return {
                **self._stats,
                "active_connections": len(self._connections),
                "global_learning_rate": self._global_learning_rate,
                "recent_error_rate": self._recent_error_rate,
                "mode": self._mode.value,
                "plasticity_rule": self.plasticity_rule.value,
                "connection_stats": conn_stats,
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
                        "tagged": conn.tagged,
                        "eligibility": conn.eligibility.get_trace(),
                    }
                )

            return {
                "nodes": list(nodes),
                "edges": edges,
                "statistics": self.get_statistics(),
            }
