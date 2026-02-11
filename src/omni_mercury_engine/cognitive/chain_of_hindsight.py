"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisors LLC

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
Chain of Hindsight (CoH) Learning Module for Mercury Agent.

Implements learning from historical sequences with feedback, inspired by:
- "Chain of Hindsight Aligns Language Models with Feedback" (Liu et al., 2023)
- "Learning from Feedback in Language Models" (Scheurer et al., 2022)

Chain of Hindsight (CoH) trains models on sequences of model outputs paired
with feedback. The key insight is that models can learn to condition on
feedback to generate better outputs.

CoH Objective: log p(x) = Σ log p(x_i | x_{<i})
With masking to prevent learning on feedback tokens.

Key Features:
1. Historical sequence learning with feedback annotation
2. Hindsight relabeling for counterfactual learning
3. Trajectory-based credit assignment
4. Feedback-conditioned generation
5. Importance weighting for experience replay
6. Integration with anomaly detection improvement

This module enables Mercury Agent to learn from past detection
sequences and feedback to improve future performance.
"""

import hashlib
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio

# Learning parameters
MAX_SEQUENCE_LENGTH = 100
MAX_HISTORY_SIZE = 10000
DEFAULT_LEARNING_RATE = 0.01
FEEDBACK_WEIGHT = 0.3


class FeedbackQuality(Enum):
    """Quality levels for feedback."""

    EXCELLENT = "excellent"  # Strong positive signal
    GOOD = "good"  # Positive signal
    NEUTRAL = "neutral"  # No clear signal
    BAD = "bad"  # Negative signal
    TERRIBLE = "terrible"  # Strong negative signal


class SequenceType(Enum):
    """Types of historical sequences."""

    DETECTION = "detection"  # Anomaly detection sequence
    DECISION = "decision"  # Decision-making sequence
    REASONING = "reasoning"  # Reasoning chain
    ACTION = "action"  # Action sequence


class RelabelingStrategy(Enum):
    """Strategies for hindsight relabeling."""

    OPTIMAL = "optimal"  # Relabel with optimal action
    COUNTERFACTUAL = "counterfactual"  # What if different choice?
    MARGINAL = "marginal"  # Marginal improvement
    RANDOM = "random"  # Random alternative


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class SequenceStep:
    """A single step in a historical sequence.

    Represents one timestep with input, output, and feedback.

    Attributes:
        step_id: Unique identifier
        input_state: Input at this step
        output_action: Output/action taken
        feedback: Feedback received (if any)
        reward: Numerical reward signal
        confidence: Confidence in the output
        timestamp: When this step occurred
        metadata: Additional metadata
    """

    step_id: str
    input_state: dict[str, Any]
    output_action: str
    feedback: str | None = None
    reward: float = 0.0
    confidence: float = 0.8
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.step_id,
            "input": self.input_state,
            "output": self.output_action,
            "feedback": self.feedback,
            "reward": self.reward,
            "confidence": self.confidence,
        }


@dataclass
class HistoricalSequence:
    """A complete historical sequence with feedback.

    Represents a trajectory of steps from start to end.

    Attributes:
        sequence_id: Unique identifier
        sequence_type: Type of sequence
        steps: Ordered list of steps
        total_reward: Cumulative reward
        feedback_quality: Quality of overall feedback
        outcome: Final outcome
        created_at: Creation timestamp
        metadata: Additional metadata
    """

    sequence_id: str
    sequence_type: SequenceType
    steps: list[SequenceStep]
    total_reward: float = 0.0
    feedback_quality: FeedbackQuality = FeedbackQuality.NEUTRAL
    outcome: str = ""
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return sequence length."""
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.sequence_id,
            "type": self.sequence_type.value,
            "length": len(self.steps),
            "total_reward": self.total_reward,
            "feedback_quality": self.feedback_quality.value,
            "outcome": self.outcome,
        }

    def get_feedback_annotated(self) -> str:
        """Get sequence with feedback annotations."""
        lines = [f"Sequence: {self.sequence_id} ({self.outcome})"]
        for i, step in enumerate(self.steps):
            lines.append(f"  Step {i + 1}: {step.output_action}")
            if step.feedback:
                lines.append(f"    Feedback: {step.feedback}")
        return "\n".join(lines)


@dataclass
class HindsightRelabeling:
    """Result of hindsight relabeling.

    Contains alternative action and expected improvement.

    Attributes:
        original_action: Original action taken
        relabeled_action: Alternative action
        strategy: Relabeling strategy used
        expected_improvement: Expected reward improvement
        confidence: Confidence in relabeling
        reasoning: Reasoning for relabeling
    """

    original_action: str
    relabeled_action: str
    strategy: RelabelingStrategy
    expected_improvement: float
    confidence: float
    reasoning: str


@dataclass
class LearningSignal:
    """A learning signal derived from hindsight.

    Used to update model behavior.

    Attributes:
        signal_id: Unique identifier
        sequence_id: Source sequence
        step_index: Step in sequence
        positive_example: What should have been done
        negative_example: What was actually done (if bad)
        weight: Importance weight
        feedback_context: Feedback that generated this signal
    """

    signal_id: str
    sequence_id: str
    step_index: int
    positive_example: dict[str, Any]
    negative_example: dict[str, Any] | None
    weight: float
    feedback_context: str


@dataclass
class PolicyUpdate:
    """Recommended policy update from CoH learning.

    Attributes:
        update_id: Unique identifier
        context_pattern: Pattern of contexts this applies to
        old_behavior: Previous behavior pattern
        new_behavior: Recommended new behavior
        confidence: Confidence in update
        evidence_count: Number of supporting examples
        expected_improvement: Expected performance gain
    """

    update_id: str
    context_pattern: dict[str, Any]
    old_behavior: str
    new_behavior: str
    confidence: float
    evidence_count: int
    expected_improvement: float


# =============================================================================
# Credit Assignment
# =============================================================================


class CreditAssignment:
    """
    Temporal credit assignment for sequential decisions.

    Assigns credit to individual steps based on final outcome.
    """

    def __init__(
        self,
        discount_factor: float = 0.95,
        use_advantage: bool = True,
    ):
        """Initialize credit assignment.

        Args:
            discount_factor: Discount for future rewards (gamma)
            use_advantage: Use advantage-based credit
        """
        self.discount_factor = discount_factor
        self.use_advantage = use_advantage

    def assign_credit(
        self,
        sequence: HistoricalSequence | list[dict[str, Any]],
        gamma: float | None = None,
    ) -> list[float]:
        """Assign credit to each step in sequence.

        Args:
            sequence: Historical sequence or list of step dicts
            gamma: Optional discount factor override

        Returns:
            List of credit values for each step
        """
        discount = gamma if gamma is not None else self.discount_factor

        # Handle list of dicts (test API)
        if isinstance(sequence, list):
            n_steps = len(sequence)
            if n_steps == 0:
                return []

            credits = [0.0] * n_steps

            # Forward credit assignment: later decisions with positive outcomes get higher credit
            # This reflects that later decisions are closer to the outcome
            for t in range(n_steps):
                step_dict = sequence[t]
                outcome = step_dict.get("outcome", {})
                reward = outcome.get("reward", 0.0)
                # Weight by position: later steps get higher weight
                position_weight = (t + 1) / n_steps
                credits[t] = reward * position_weight

            # Apply discount for temporal consistency
            for t in range(n_steps - 2, -1, -1):
                credits[t] += discount * credits[t + 1] * 0.1  # Small propagation

            # Normalize credits
            if self.use_advantage and len(credits) > 1:
                mean_credit = np.mean(credits)
                std_credit = np.std(credits) + 1e-8
                credits = [(c - mean_credit) / std_credit for c in credits]

            return credits

        # Original API with HistoricalSequence
        n_steps = len(sequence.steps)
        if n_steps == 0:
            return []

        credits = [0.0] * n_steps

        # Monte Carlo return estimation
        running_return = 0.0
        for t in range(n_steps - 1, -1, -1):
            step = sequence.steps[t]
            running_return = step.reward + discount * running_return
            credits[t] = running_return

        # Normalize credits
        if self.use_advantage and len(credits) > 1:
            mean_credit = np.mean(credits)
            std_credit = np.std(credits) + 1e-8
            credits = [(c - mean_credit) / std_credit for c in credits]

        return credits

    def get_key_steps(
        self, sequence: HistoricalSequence, top_k: int = 3
    ) -> list[tuple[int, float]]:
        """Get most important steps by credit.

        Args:
            sequence: Historical sequence
            top_k: Number of key steps to return

        Returns:
            List of (step_index, credit) tuples
        """
        credits = self.assign_credit(sequence)
        indexed_credits = [(i, c) for i, c in enumerate(credits)]
        indexed_credits.sort(key=lambda x: abs(x[1]), reverse=True)
        return indexed_credits[:top_k]


# =============================================================================
# Hindsight Relabeler
# =============================================================================


class HindsightRelabeler:
    """
    Relabels historical sequences with hindsight knowledge.

    Enables counterfactual learning by considering what
    should have been done given the outcome.
    """

    def __init__(
        self,
        anomaly_threshold: float = 0.5,
        improvement_threshold: float = 0.1,
    ):
        """Initialize relabeler.

        Args:
            anomaly_threshold: Threshold for anomaly classification
            improvement_threshold: Minimum improvement for relabeling
        """
        self.anomaly_threshold = anomaly_threshold
        self.improvement_threshold = improvement_threshold

        # Action alternatives
        self._action_alternatives = {
            "anomaly_detected": ["normal_classified", "flag_for_review"],
            "normal_classified": ["anomaly_detected", "flag_for_review"],
            "flag_for_review": ["anomaly_detected", "normal_classified"],
            "escalate": ["monitor", "ignore"],
            "monitor": ["escalate", "ignore"],
            "ignore": ["monitor", "escalate"],
        }

    def relabel(
        self,
        sequence: HistoricalSequence | list[dict[str, Any]],
        strategy_or_achieved_goal: RelabelingStrategy | str = RelabelingStrategy.OPTIMAL,
    ) -> list[HindsightRelabeling] | list[dict[str, Any]]:
        """Relabel sequence with hindsight.

        Args:
            sequence: Sequence to relabel (HistoricalSequence or list of step dicts)
            strategy_or_achieved_goal: Relabeling strategy or achieved goal string

        Returns:
            List of relabeling suggestions or relabeled trajectory
        """
        # Handle list of dicts (test API: trajectory with achieved_goal)
        if isinstance(sequence, list):
            achieved_goal = (
                strategy_or_achieved_goal
                if isinstance(strategy_or_achieved_goal, str)
                else "achieved"
            )
            relabeled_trajectory = []
            for step_dict in sequence:
                relabeled_step = step_dict.copy()
                relabeled_step["goal"] = achieved_goal
                relabeled_trajectory.append(relabeled_step)
            return relabeled_trajectory

        # Original API with HistoricalSequence
        strategy = (
            strategy_or_achieved_goal
            if isinstance(strategy_or_achieved_goal, RelabelingStrategy)
            else RelabelingStrategy.OPTIMAL
        )

        relabelings: list[HindsightRelabeling] = []
        credits = CreditAssignment().assign_credit(sequence)

        for i, step in enumerate(sequence.steps):
            credit = credits[i] if i < len(credits) else 0.0

            # Only relabel if step had negative impact
            if credit < -self.improvement_threshold:
                relabeling = self._relabel_step(step, credit, strategy, sequence)
                if relabeling:
                    relabelings.append(relabeling)

        return relabelings

    def _relabel_step(
        self,
        step: SequenceStep,
        credit: float,
        strategy: RelabelingStrategy,
        sequence: HistoricalSequence,
    ) -> HindsightRelabeling | None:
        """Relabel a single step."""
        alternatives = self._action_alternatives.get(step.output_action, [])
        if not alternatives:
            return None

        if strategy == RelabelingStrategy.OPTIMAL:
            # Choose best alternative based on outcome
            best_alt = self._find_optimal_alternative(step, sequence)
        elif strategy == RelabelingStrategy.COUNTERFACTUAL:
            # What if we did the opposite?
            best_alt = alternatives[0] if alternatives else step.output_action
        elif strategy == RelabelingStrategy.MARGINAL:
            # Small improvement
            best_alt = "flag_for_review"
        else:  # RANDOM
            best_alt = np.random.choice(alternatives)

        expected_improvement = abs(credit) * 0.5  # Conservative estimate

        return HindsightRelabeling(
            original_action=step.output_action,
            relabeled_action=best_alt,
            strategy=strategy,
            expected_improvement=expected_improvement,
            confidence=0.7,
            reasoning=f"Step had negative credit ({credit:.3f}), suggesting {best_alt} might be better",
        )

    def _find_optimal_alternative(self, step: SequenceStep, sequence: HistoricalSequence) -> str:
        """Find optimal alternative action based on outcome."""
        outcome = sequence.outcome.lower()

        # If outcome was false positive, prefer normal
        if "false_positive" in outcome:
            return "normal_classified"
        # If outcome was false negative, prefer anomaly
        if "false_negative" in outcome:
            return "anomaly_detected"
        # Otherwise flag for review
        return "flag_for_review"


# =============================================================================
# Feedback Processor
# =============================================================================


class FeedbackProcessor:
    """
    Processes feedback to generate learning signals.

    Converts human feedback and outcome signals into
    structured learning signals for policy improvement.
    """

    def __init__(self, weight_by_quality: bool = True):
        """Initialize feedback processor.

        Args:
            weight_by_quality: Weight signals by feedback quality
        """
        self.weight_by_quality = weight_by_quality
        self._signal_counter = 0

        # Quality to weight mapping
        self._quality_weights = {
            FeedbackQuality.EXCELLENT: 1.0,
            FeedbackQuality.GOOD: 0.8,
            FeedbackQuality.NEUTRAL: 0.5,
            FeedbackQuality.BAD: 0.8,  # Bad feedback is also informative
            FeedbackQuality.TERRIBLE: 1.0,  # Strong signal
        }

    def process(
        self,
        predictions: list[float],
        ground_truth: list[int | bool],
    ) -> dict[str, Any]:
        """Process predictions and ground truth to generate feedback.

        Args:
            predictions: List of predicted scores
            ground_truth: List of ground truth labels

        Returns:
            Dict with linguistic_feedback and improvement_signals
        """
        linguistic_feedback = []
        improvement_signals = []

        for i, (pred, truth) in enumerate(zip(predictions, ground_truth)):
            truth_bool = bool(truth)
            pred_class = pred > 0.5

            if pred_class == truth_bool:
                if truth_bool:
                    linguistic_feedback.append(
                        f"Step {i}: Correct positive detection (score={pred:.2f})"
                    )
                else:
                    linguistic_feedback.append(
                        f"Step {i}: Correct negative classification (score={pred:.2f})"
                    )
            elif pred_class and not truth_bool:
                linguistic_feedback.append(
                    f"Step {i}: False positive - lower threshold recommended (score={pred:.2f})"
                )
                improvement_signals.append(
                    {
                        "step": i,
                        "type": "false_positive",
                        "suggestion": "increase_threshold",
                        "score": pred,
                    }
                )
            else:
                linguistic_feedback.append(
                    f"Step {i}: False negative - raise sensitivity (score={pred:.2f})"
                )
                improvement_signals.append(
                    {
                        "step": i,
                        "type": "false_negative",
                        "suggestion": "decrease_threshold",
                        "score": pred,
                    }
                )

        return {
            "linguistic_feedback": linguistic_feedback,
            "improvement_signals": improvement_signals,
            "accuracy": (
                sum(1 for p, t in zip(predictions, ground_truth) if (p > 0.5) == bool(t))
                / len(predictions)
                if predictions
                else 0.0
            ),
        }

    def process_sequence(self, sequence: HistoricalSequence) -> list[LearningSignal]:
        """Process sequence to generate learning signals.

        Args:
            sequence: Sequence with feedback

        Returns:
            List of learning signals
        """
        signals: list[LearningSignal] = []
        credits = CreditAssignment().assign_credit(sequence)

        for i, step in enumerate(sequence.steps):
            if step.feedback or step.reward != 0:
                signal = self._create_signal(step, i, sequence, credits)
                signals.append(signal)

        return signals

    def _create_signal(
        self,
        step: SequenceStep,
        index: int,
        sequence: HistoricalSequence,
        credits: list[float],
    ) -> LearningSignal:
        """Create learning signal from step."""
        self._signal_counter += 1
        signal_id = f"signal_{self._signal_counter:06d}"

        credit = credits[index] if index < len(credits) else 0.0
        is_positive = credit > 0 or step.reward > 0

        # Calculate weight
        weight = 1.0
        if self.weight_by_quality:
            weight = self._quality_weights.get(sequence.feedback_quality, 0.5)
        weight *= abs(credit) + 0.1  # Ensure minimum weight

        if is_positive:
            # This was a good step - reinforce it
            positive_example = {
                "input": step.input_state,
                "output": step.output_action,
                "context": step.metadata,
            }
            negative_example = None
        else:
            # This was a bad step - learn from it
            positive_example = {
                "input": step.input_state,
                "output": "better_alternative",  # Placeholder
                "context": step.metadata,
            }
            negative_example = {
                "input": step.input_state,
                "output": step.output_action,
                "context": step.metadata,
            }

        return LearningSignal(
            signal_id=signal_id,
            sequence_id=sequence.sequence_id,
            step_index=index,
            positive_example=positive_example,
            negative_example=negative_example,
            weight=weight,
            feedback_context=step.feedback or str(step.reward),
        )

    def aggregate_signals(self, signals: list[LearningSignal]) -> dict[str, list[LearningSignal]]:
        """Aggregate signals by pattern.

        Args:
            signals: List of learning signals

        Returns:
            Signals grouped by context pattern
        """
        aggregated: dict[str, list[LearningSignal]] = defaultdict(list)

        for signal in signals:
            # Create pattern key from input state
            pattern_key = self._create_pattern_key(signal.positive_example.get("input", {}))
            aggregated[pattern_key].append(signal)

        return dict(aggregated)

    def _create_pattern_key(self, input_state: dict[str, Any]) -> str:
        """Create pattern key from input state."""
        key_features = sorted(input_state.keys())[:5]
        key_str = "_".join(key_features)
        return hashlib.sha3_256(key_str.encode()).hexdigest()[:8]


# =============================================================================
# Chain of Hindsight Engine
# =============================================================================


class ChainOfHindsightEngine:
    """
    Main Chain of Hindsight learning engine.

    Implements learning from historical sequences with feedback
    to improve decision-making over time.

    Key capabilities:
    1. Store and retrieve historical sequences
    2. Process feedback into learning signals
    3. Relabel sequences with hindsight
    4. Generate policy updates
    5. Track learning progress
    """

    def __init__(
        self,
        max_history_size: int = MAX_HISTORY_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        feedback_weight: float = FEEDBACK_WEIGHT,
        enable_relabeling: bool = True,
    ):
        """Initialize CoH engine.

        Args:
            max_history_size: Maximum sequences to store
            learning_rate: Learning rate for updates
            feedback_weight: Weight for feedback signals
            enable_relabeling: Enable hindsight relabeling
        """
        self.max_history_size = max_history_size
        self.learning_rate = learning_rate
        self.feedback_weight = feedback_weight
        self.enable_relabeling = enable_relabeling

        # Components
        self.credit_assignment = CreditAssignment()
        self.hindsight_relabeler = HindsightRelabeler()
        self.feedback_processor = FeedbackProcessor()

        # History storage
        self._sequences: deque[HistoricalSequence] = deque(maxlen=max_history_size)
        self._sequence_index: dict[str, int] = {}  # sequence_id -> position

        # Learning signals
        self._accumulated_signals: list[LearningSignal] = []
        self._policy_updates: list[PolicyUpdate] = []

        # Counters
        self._sequence_counter = 0
        self._update_counter = 0

        # Statistics
        self._stats = {
            "sequences_recorded": 0,
            "signals_generated": 0,
            "relabelings_performed": 0,
            "policy_updates": 0,
            "avg_sequence_length": 0.0,
            "positive_feedback_ratio": 0.0,
            "total_sequences": 0,
        }

        # Active sequences for incremental API
        self._active_sequences: dict[str, dict[str, Any]] = {}

        logger.info(
            f"ChainOfHindsightEngine initialized (max_history={max_history_size}, "
            f"lr={learning_rate})"
        )

    def start_sequence(self, task_name: str) -> str:
        """Start a new sequence for incremental recording.

        Args:
            task_name: Name/description of the task

        Returns:
            Sequence ID for subsequent operations
        """
        self._sequence_counter += 1
        sequence_id = f"seq_{self._sequence_counter:06d}"

        self._active_sequences[sequence_id] = {
            "task_name": task_name,
            "steps": [],
            "start_time": time.time(),
        }

        return sequence_id

    def record_step(
        self,
        sequence_id: str,
        decision: dict[str, Any],
        outcome: dict[str, Any],
        features: list[float] | None = None,
    ) -> None:
        """Record a step in an active sequence.

        Args:
            sequence_id: ID from start_sequence
            decision: Decision made at this step
            outcome: Outcome of the decision
            features: Optional feature vector
        """
        if sequence_id not in self._active_sequences:
            raise ValueError(f"Unknown sequence ID: {sequence_id}")

        step_idx = len(self._active_sequences[sequence_id]["steps"])
        step_id = f"{sequence_id}_step_{step_idx}"

        # Determine reward from outcome
        correct = outcome.get("correct", outcome.get("success"))
        if correct is True:
            reward = 1.0
            feedback = "Correct decision"
        elif correct is False:
            reward = -1.0
            feedback = "Incorrect decision"
        else:
            reward = 0.0
            feedback = None

        step = SequenceStep(
            step_id=step_id,
            input_state={"features": features or [], **decision},
            output_action=str(decision.get("threshold", decision.get("action", "unknown"))),
            feedback=feedback,
            reward=reward,
            confidence=decision.get("confidence", 0.5),
        )

        self._active_sequences[sequence_id]["steps"].append(step)

    def end_sequence(
        self,
        sequence_id: str,
        final_outcome: dict[str, Any],
    ) -> HistoricalSequence:
        """End an active sequence and finalize it.

        Args:
            sequence_id: ID from start_sequence
            final_outcome: Final outcome of the sequence

        Returns:
            Completed HistoricalSequence
        """
        if sequence_id not in self._active_sequences:
            raise ValueError(f"Unknown sequence ID: {sequence_id}")

        active = self._active_sequences.pop(sequence_id)
        steps = active["steps"]

        # Determine feedback quality from final outcome
        success = final_outcome.get("success", False)
        if success:
            quality = FeedbackQuality.GOOD
        else:
            quality = FeedbackQuality.BAD

        outcome_str = "success" if success else "failure"

        # Create and record the sequence
        sequence = self.record_sequence(
            steps=steps,
            sequence_type=SequenceType.DECISION,
            outcome=outcome_str,
            feedback_quality=quality,
        )

        self._stats["total_sequences"] += 1

        return sequence

    def record_sequence(
        self,
        steps: list[SequenceStep],
        sequence_type: SequenceType,
        outcome: str,
        feedback_quality: FeedbackQuality = FeedbackQuality.NEUTRAL,
    ) -> HistoricalSequence:
        """Record a new historical sequence.

        Args:
            steps: Steps in the sequence
            sequence_type: Type of sequence
            outcome: Final outcome
            feedback_quality: Quality of feedback

        Returns:
            Created HistoricalSequence
        """
        self._sequence_counter += 1
        sequence_id = f"seq_{self._sequence_counter:06d}"

        # Calculate total reward
        total_reward = sum(step.reward for step in steps)

        sequence = HistoricalSequence(
            sequence_id=sequence_id,
            sequence_type=sequence_type,
            steps=steps,
            total_reward=total_reward,
            feedback_quality=feedback_quality,
            outcome=outcome,
        )

        # Store sequence
        self._sequences.append(sequence)
        self._sequence_index[sequence_id] = len(self._sequences) - 1

        # Process for learning
        signals = self.feedback_processor.process_sequence(sequence)
        self._accumulated_signals.extend(signals)

        # Apply hindsight relabeling
        if self.enable_relabeling:
            relabelings = self.hindsight_relabeler.relabel(sequence)
            self._stats["relabelings_performed"] += len(relabelings)

        # Update statistics
        self._update_stats(sequence, signals)

        return sequence

    def record_detection_sequence(
        self,
        detections: list[dict[str, Any]],
        ground_truths: list[bool | None],
        outcome: str,
    ) -> HistoricalSequence:
        """Record an anomaly detection sequence.

        Convenience method for anomaly detection use case.

        Args:
            detections: List of detection results
            ground_truths: List of ground truth labels
            outcome: Final outcome description

        Returns:
            Created HistoricalSequence
        """
        steps = []
        for i, (det, truth) in enumerate(zip(detections, ground_truths)):
            step_id = f"det_step_{i}"

            # Determine reward based on ground truth
            if truth is not None:
                predicted = det.get("anomaly_score", 0.5) > 0.5
                if predicted == truth:
                    reward = 1.0
                    feedback = "Correct detection"
                else:
                    reward = -1.0
                    feedback = "Incorrect detection"
            else:
                reward = 0.0
                feedback = None

            step = SequenceStep(
                step_id=step_id,
                input_state={
                    "features": det.get("features", {}),
                    "anomaly_score": det.get("anomaly_score", 0.5),
                },
                output_action=(
                    "anomaly_detected"
                    if det.get("anomaly_score", 0.5) > 0.5
                    else "normal_classified"
                ),
                feedback=feedback,
                reward=reward,
                confidence=abs(det.get("anomaly_score", 0.5) - 0.5) * 2,
            )
            steps.append(step)

        # Determine feedback quality from outcome
        if "true_positive" in outcome.lower() or "true_negative" in outcome.lower():
            quality = FeedbackQuality.GOOD
        elif "false" in outcome.lower():
            quality = FeedbackQuality.BAD
        else:
            quality = FeedbackQuality.NEUTRAL

        return self.record_sequence(steps, SequenceType.DETECTION, outcome, quality)

    def learn_from_history(
        self,
        min_signals: int = 10,
    ) -> list[PolicyUpdate]:
        """Generate policy updates from accumulated signals.

        Args:
            min_signals: Minimum signals required for learning

        Returns:
            List of policy updates
        """
        if len(self._accumulated_signals) < min_signals:
            return []

        # Aggregate signals by pattern
        aggregated = self.feedback_processor.aggregate_signals(self._accumulated_signals)

        updates: list[PolicyUpdate] = []

        for pattern_key, signals in aggregated.items():
            if len(signals) < 3:  # Need multiple examples
                continue

            update = self._generate_policy_update(pattern_key, signals)
            if update:
                updates.append(update)

        self._policy_updates.extend(updates)
        self._stats["policy_updates"] += len(updates)

        # Clear processed signals
        self._accumulated_signals = []

        return updates

    def get_recommendation(
        self,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Get recommendation based on learned patterns.

        Args:
            context: Current decision context

        Returns:
            Recommendation or None
        """
        # Find relevant policy updates
        relevant_updates = [
            u for u in self._policy_updates if self._context_matches(context, u.context_pattern)
        ]

        if not relevant_updates:
            return None

        # Return highest confidence update
        best_update = max(relevant_updates, key=lambda u: u.confidence * u.evidence_count)

        return {
            "recommended_action": best_update.new_behavior,
            "avoid_action": best_update.old_behavior,
            "confidence": best_update.confidence,
            "expected_improvement": best_update.expected_improvement,
            "evidence_count": best_update.evidence_count,
        }

    def get_similar_sequences(
        self,
        context: dict[str, Any],
        top_k: int = 5,
    ) -> list[HistoricalSequence]:
        """Get similar historical sequences.

        Args:
            context: Current context
            top_k: Number of sequences to return

        Returns:
            List of similar sequences
        """
        scored: list[tuple[HistoricalSequence, float]] = []

        for seq in self._sequences:
            if seq.steps:
                first_input = seq.steps[0].input_state
                similarity = self._compute_similarity(context, first_input)
                scored.append((seq, similarity))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [seq for seq, _ in scored[:top_k]]

    def get_learning_insights(self) -> dict[str, Any]:
        """Get insights from learning history.

        Returns:
            Dictionary of learning insights
        """
        if not self._sequences:
            return {"status": "no_data"}

        # Analyze sequences
        total_reward = sum(s.total_reward for s in self._sequences)
        avg_reward = total_reward / len(self._sequences)

        # Quality distribution
        quality_dist: defaultdict[str, int] = defaultdict(int)
        for seq in self._sequences:
            quality_dist[seq.feedback_quality.value] += 1

        # Type distribution
        type_dist: defaultdict[str, int] = defaultdict(int)
        for seq in self._sequences:
            type_dist[seq.sequence_type.value] += 1

        # Key learning signals
        key_patterns = self._identify_key_patterns()

        return {
            "total_sequences": len(self._sequences),
            "total_reward": total_reward,
            "avg_reward": avg_reward,
            "quality_distribution": dict(quality_dist),
            "type_distribution": dict(type_dist),
            "policy_updates_generated": len(self._policy_updates),
            "key_patterns": key_patterns,
            "accumulated_signals": len(self._accumulated_signals),
        }

    def export_for_training(self) -> list[dict[str, Any]]:
        """Export sequences in training format.

        Returns:
            List of sequences in CoH training format
        """
        training_data = []

        for seq in self._sequences:
            # Format: sequence with feedback annotations
            example = {
                "sequence_id": seq.sequence_id,
                "type": seq.sequence_type.value,
                "steps": [step.to_dict() for step in seq.steps],
                "outcome": seq.outcome,
                "feedback_quality": seq.feedback_quality.value,
                "total_reward": seq.total_reward,
                "annotated_text": seq.get_feedback_annotated(),
            }
            training_data.append(example)

        return training_data

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _generate_policy_update(
        self,
        pattern_key: str,
        signals: list[LearningSignal],
    ) -> PolicyUpdate | None:
        """Generate policy update from signals."""
        if not signals:
            return None

        self._update_counter += 1
        update_id = f"update_{self._update_counter:06d}"

        # Analyze signals
        positive_actions: list[str] = []
        negative_actions: list[str] = []
        total_weight = 0.0

        for signal in signals:
            weight = signal.weight
            total_weight += weight

            positive_actions.append(signal.positive_example.get("output", ""))
            if signal.negative_example:
                negative_actions.append(signal.negative_example.get("output", ""))

        # Find most common patterns
        if not positive_actions:
            return None

        from collections import Counter

        pos_counter = Counter(positive_actions)
        neg_counter = Counter(negative_actions)

        most_common_positive = pos_counter.most_common(1)[0][0] if pos_counter else ""
        most_common_negative = neg_counter.most_common(1)[0][0] if neg_counter else ""

        # Calculate confidence
        confidence = min(0.95, 0.5 + len(signals) * 0.05)

        # Calculate expected improvement
        avg_weight = total_weight / len(signals)
        expected_improvement = avg_weight * self.learning_rate

        # Extract context pattern from first signal
        context_pattern = signals[0].positive_example.get("input", {})

        return PolicyUpdate(
            update_id=update_id,
            context_pattern=context_pattern,
            old_behavior=most_common_negative,
            new_behavior=most_common_positive,
            confidence=confidence,
            evidence_count=len(signals),
            expected_improvement=expected_improvement,
        )

    def _context_matches(
        self,
        context: dict[str, Any],
        pattern: dict[str, Any],
    ) -> bool:
        """Check if context matches pattern."""
        if not pattern:
            return True

        # Check key overlap
        common_keys = set(context.keys()) & set(pattern.keys())
        if not common_keys:
            return False

        # Check value similarity for common keys
        matches = 0
        for key in common_keys:
            if context[key] == pattern[key]:
                matches += 1
            elif isinstance(context[key], (int, float)) and isinstance(pattern[key], (int, float)):
                if abs(context[key] - pattern[key]) < 0.2:
                    matches += 1

        return matches / len(common_keys) > 0.5 if common_keys else False

    def _compute_similarity(
        self,
        context1: dict[str, Any],
        context2: dict[str, Any],
    ) -> float:
        """Compute similarity between contexts."""
        if not context1 or not context2:
            return 0.0

        keys1 = set(context1.keys())
        keys2 = set(context2.keys())
        common = keys1 & keys2

        if not common:
            return 0.0

        key_sim = len(common) / len(keys1 | keys2)

        value_matches: float = 0
        for key in common:
            v1, v2 = context1[key], context2[key]
            if v1 == v2:
                value_matches += 1
            elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                max_val = max(abs(v1), abs(v2), 1)
                value_matches += 1 - abs(v1 - v2) / max_val

        value_sim = value_matches / len(common)

        return 0.4 * key_sim + 0.6 * value_sim

    def _identify_key_patterns(self) -> list[dict[str, Any]]:
        """Identify key learning patterns."""
        patterns: list[dict[str, Any]] = []

        for update in self._policy_updates:
            if update.confidence > 0.7 and update.evidence_count > 5:
                patterns.append(
                    {
                        "pattern": update.context_pattern,
                        "change": f"{update.old_behavior} -> {update.new_behavior}",
                        "confidence": update.confidence,
                        "evidence": update.evidence_count,
                    }
                )

        patterns.sort(key=lambda x: x["confidence"] * x["evidence"], reverse=True)
        return patterns[:10]

    def _update_stats(self, sequence: HistoricalSequence, signals: list[LearningSignal]) -> None:
        """Update engine statistics."""
        self._stats["sequences_recorded"] += 1
        self._stats["signals_generated"] += len(signals)

        # Update averages
        n = self._stats["sequences_recorded"]
        self._stats["avg_sequence_length"] = (
            self._stats["avg_sequence_length"] * (n - 1) + len(sequence)
        ) / n

        # Update positive feedback ratio
        positive = (
            1
            if sequence.feedback_quality in [FeedbackQuality.GOOD, FeedbackQuality.EXCELLENT]
            else 0
        )
        self._stats["positive_feedback_ratio"] = (
            self._stats["positive_feedback_ratio"] * (n - 1) + positive
        ) / n

    def get_statistics(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            **self._stats,
            "history_size": len(self._sequences),
            "max_history_size": self.max_history_size,
            "learning_rate": self.learning_rate,
            "relabeling_enabled": self.enable_relabeling,
        }


# =============================================================================
# Anomaly Detection Integration
# =============================================================================


class AnomalyChainOfHindsight:
    """
    Chain of Hindsight specialized for anomaly detection.

    Integrates CoH learning with Mercury Agent's anomaly
    detection pipeline for continuous improvement.
    """

    def __init__(
        self,
        coh_engine: ChainOfHindsightEngine | None = None,
        batch_size: int = 100,
    ):
        """Initialize anomaly CoH.

        Args:
            coh_engine: Base CoH engine
            batch_size: Batch size for recording sequences
        """
        self.engine = coh_engine or ChainOfHindsightEngine()
        self.batch_size = batch_size

        self._current_batch: list[dict[str, Any]] = []
        self._current_truths: list[bool | None] = []

    def add_detection(
        self,
        detection: dict[str, Any],
        ground_truth: bool | None = None,
    ) -> None:
        """Add a detection to current batch.

        Args:
            detection: Detection result
            ground_truth: Ground truth label (if known)
        """
        self._current_batch.append(detection)
        self._current_truths.append(ground_truth)

        if len(self._current_batch) >= self.batch_size:
            self._flush_batch()

    def _flush_batch(self) -> None:
        """Process current batch."""
        if not self._current_batch:
            return

        # Determine outcome
        correct = sum(
            1
            for det, truth in zip(self._current_batch, self._current_truths)
            if truth is not None and (det.get("anomaly_score", 0.5) > 0.5) == truth
        )
        total = sum(1 for t in self._current_truths if t is not None)

        if total > 0:
            accuracy = correct / total
            if accuracy > 0.8:
                outcome = "high_accuracy_batch"
            elif accuracy > 0.5:
                outcome = "moderate_accuracy_batch"
            else:
                outcome = "low_accuracy_batch"
        else:
            outcome = "unlabeled_batch"

        # Record sequence
        self.engine.record_detection_sequence(self._current_batch, self._current_truths, outcome)

        # Clear batch
        self._current_batch = []
        self._current_truths = []

    def get_threshold_adjustment(self) -> dict[str, Any]:
        """Get recommended threshold adjustment.

        Returns:
            Threshold adjustment recommendation
        """
        # Analyze recent sequences
        recent_sequences = list(self.engine._sequences)[-50:]
        if not recent_sequences:
            return {"recommendation": "maintain", "confidence": 0.5}

        # Count false positives and negatives
        fp_count = 0
        fn_count = 0
        for seq in recent_sequences:
            for step in seq.steps:
                if step.feedback == "Incorrect detection":
                    if step.output_action == "anomaly_detected":
                        fp_count += 1
                    else:
                        fn_count += 1

        # Generate recommendation
        if fn_count > fp_count * 2:
            return {
                "recommendation": "lower_threshold",
                "reasoning": f"High false negative rate ({fn_count} FN vs {fp_count} FP)",
                "suggested_delta": -0.05,
                "confidence": 0.8,
            }
        elif fp_count > fn_count * 2:
            return {
                "recommendation": "raise_threshold",
                "reasoning": f"High false positive rate ({fp_count} FP vs {fn_count} FN)",
                "suggested_delta": 0.05,
                "confidence": 0.8,
            }
        else:
            return {
                "recommendation": "maintain",
                "reasoning": f"Balanced error rate ({fp_count} FP, {fn_count} FN)",
                "confidence": 0.9,
            }

    def learn_from_history(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Learn from a history of detection results.

        Args:
            history: List of detection history entries with timestamp, detection, label

        Returns:
            Dict with threshold_recommendations or pattern_insights
        """
        if not history:
            return {"pattern_insights": [], "threshold_recommendations": []}

        # Analyze patterns in history
        scores = []
        labels = []
        for entry in history:
            detection = entry.get("detection", {})
            score = detection.get("score", 0.5)
            label = entry.get("label")
            scores.append(score)
            if label is not None:
                labels.append((score, label))

        # Generate insights
        pattern_insights = []
        threshold_recommendations = []

        if labels:
            # Analyze false positives and negatives
            fp_scores = [s for s, label in labels if s > 0.5 and not label]
            fn_scores = [s for s, label in labels if s <= 0.5 and label]

            if fp_scores:
                avg_fp = np.mean(fp_scores)
                pattern_insights.append(f"False positives cluster around score {avg_fp:.2f}")
                threshold_recommendations.append(
                    {
                        "type": "increase_threshold",
                        "reason": f"False positives at avg score {avg_fp:.2f}",
                        "suggested_value": min(0.9, avg_fp + 0.1),
                    }
                )

            if fn_scores:
                avg_fn = np.mean(fn_scores)
                pattern_insights.append(f"False negatives cluster around score {avg_fn:.2f}")
                threshold_recommendations.append(
                    {
                        "type": "decrease_threshold",
                        "reason": f"False negatives at avg score {avg_fn:.2f}",
                        "suggested_value": max(0.1, avg_fn - 0.1),
                    }
                )

        # Temporal pattern analysis
        if len(scores) >= 3:
            score_trend = np.polyfit(range(len(scores)), scores, 1)[0]
            if score_trend > 0.01:
                pattern_insights.append("Anomaly scores trending upward over time")
            elif score_trend < -0.01:
                pattern_insights.append("Anomaly scores trending downward over time")

        return {
            "pattern_insights": pattern_insights,
            "threshold_recommendations": threshold_recommendations,
            "history_length": len(history),
            "labeled_samples": len(labels),
        }

    def force_learn(self) -> list[PolicyUpdate]:
        """Force learning from current history.

        Returns:
            Generated policy updates
        """
        self._flush_batch()
        return self.engine.learn_from_history(min_signals=5)
