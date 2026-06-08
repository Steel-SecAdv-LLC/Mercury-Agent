# Copyright (C) 2025 Steel Security Advisors LLC
"""Agentic AI Autonomy Module.

Inspired by Bain 2025 report on agentic AI transformation:
"At full potential, agents will run complete processes and workflows."

Implements autonomous agent framework for anomaly detection
that can operate with minimal human oversight.

Research source: Bain & Company Technology Report 2025
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class AgentState(Enum):
    """Agent operational states."""

    IDLE = 1
    OBSERVING = 2
    ANALYZING = 3
    ACTING = 4
    LEARNING = 5


@dataclass
class AgentAction:
    """Represents an action taken by the agent."""

    action_type: str
    parameters: dict[str, Any]
    confidence: float
    rationale: str
    outcome: float | None = None  # Reward/outcome after action execution
    state_hash: int | None = None  # Hash of state when action was taken
    # State features captured at policy-selection time.  Carried through to
    # ``_learn_from_action`` so the Q-table key the epsilon-greedy policy read
    # from is the exact key the TD update writes to (no off-by-one drift from
    # ``action_history`` mutating between selection and learning).
    state_features: tuple[float, ...] | None = None


@dataclass
class Experience:
    """Experience tuple for reinforcement learning replay buffer."""

    state_features: tuple[float, ...]  # Hashable state representation
    action_type: str
    reward: float
    next_state_features: tuple[float, ...] | None
    done: bool


@dataclass
class LearningConfig:
    """Configuration for reinforcement learning."""

    learning_rate: float = 0.01
    discount_factor: float = 0.95  # Gamma for future rewards
    exploration_rate: float = 0.1  # Epsilon for exploration
    exploration_decay: float = 0.995
    min_exploration_rate: float = 0.01
    batch_size: int = 32
    memory_size: int = 10000
    reward_scale: float = 1.0


@dataclass
class PolicyMetrics:
    """Metrics tracking policy performance over time."""

    total_rewards: float = 0.0
    episode_count: int = 0
    successful_actions: int = 0
    failed_actions: int = 0
    average_confidence: float = 0.0
    convergence_history: list[float] = field(default_factory=list)


class AgenticAutonomy:
    """Autonomous agent framework for anomaly detection.

    Agents can operate complete processes and workflows with
    minimal human oversight, inspired by Bain's agentic AI vision.

    Implements reinforcement learning for continuous improvement:
    - Q-learning style value updates for action selection
    - Experience replay for stable learning
    - Adaptive exploration-exploitation balance
    """

    # Action types and their base indices for Q-table
    ACTION_TYPES: list[str] = ["flag_anomaly", "escalate", "suppress", "investigate", "log"]

    def __init__(
        self,
        autonomy_level: float = 0.8,
        learning_config: LearningConfig | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize agentic autonomy system.

        Args:
            autonomy_level: Level of autonomy (0-1), higher = more autonomous
            learning_config: Configuration for reinforcement learning
            seed: Optional seed for the per-instance numpy ``Generator``
                that drives experience-replay batch sampling and
                exploration / random action selection.  Pass an
                explicit seed for reproducible RL audits; the legacy
                global ``np.random`` state is never used.
        """
        self.autonomy_level = autonomy_level
        self.state = AgentState.IDLE
        self.action_history: list[AgentAction] = []
        self.decision_threshold = 1.0 - autonomy_level

        # Reinforcement learning components
        self.learning_config = learning_config or LearningConfig()
        self.experience_buffer: deque[Experience] = deque(maxlen=self.learning_config.memory_size)
        self.exploration_rate = self.learning_config.exploration_rate
        self.policy_metrics = PolicyMetrics()
        self._rng: np.random.Generator = np.random.default_rng(seed)

        # Q-table: maps (state_bucket, action_type) -> Q-value
        # State buckets are discretized state representations
        self._q_table: dict[tuple[int, str], float] = {}

        # Track workflow patterns for meta-learning
        self._workflow_value_estimates: dict[str, float] = {}
        self._workflow_execution_counts: dict[str, int] = {}

    def autonomous_detect(
        self, data: np.ndarray[Any, Any], context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Autonomously detect anomalies with minimal human oversight.

        Args:
            data: Input data to analyze
            context: Optional context information

        Returns:
            Detection results with actions taken
        """
        self.state = AgentState.OBSERVING

        observations = self._observe_patterns(data)

        self.state = AgentState.ANALYZING

        anomaly_score = self._analyze_anomalies(observations)

        if anomaly_score > self.decision_threshold:
            self.state = AgentState.ACTING
            # RL policy in the loop: derive the observation state, let the
            # epsilon-greedy Q-policy choose the action type (explore vs.
            # exploit the learned Q-table), then materialise that action.
            # Replaces the prior hardcoded ``flag_anomaly`` so the Q-table the
            # learner builds actually steers behaviour.
            state_features = self._observation_state_features(anomaly_score)
            action_type = self.select_action_with_policy(state_features)
            action = self._decide_action(anomaly_score, observations, action_type)
            action.state_features = state_features
            self.action_history.append(action)

            self.state = AgentState.LEARNING
            self._learn_from_action(action)
        else:
            action = None

        self.state = AgentState.IDLE

        return {
            "anomaly_detected": bool(anomaly_score > self.decision_threshold),
            "anomaly_score": float(anomaly_score),
            "action_taken": action,
            "action_type": action.action_type if action is not None else None,
            "autonomous": True,
            "human_oversight_needed": bool(anomaly_score < self.decision_threshold),
        }

    def _observation_state_features(self, anomaly_score: float) -> tuple[float, ...]:
        """Build Q-learning state features from the observation context.

        Computed *before* the action is chosen so the epsilon-greedy policy
        and the TD update key the Q-table on the same observation-derived
        state (the action type is the decision variable, not part of the
        state).  Layout matches :meth:`_extract_state_features` so the
        replay buffer and the live update share a coordinate system.
        """
        features = [
            float(anomaly_score),
            1.0 if anomaly_score > 0.8 else 0.0,
            1.0 if anomaly_score > 0.95 else 0.0,
            self.autonomy_level,
            len(self.action_history) / 100.0,
        ]
        return tuple(float(f) for f in features)

    def _observe_patterns(self, data: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Observe patterns in data."""
        return {"mean": np.mean(data), "std": np.std(data), "trend": self._detect_trend(data)}

    def _analyze_anomalies(self, observations: dict[str, Any]) -> float:
        """Analyze observations for anomalies."""
        score = abs(observations["mean"]) / (observations["std"] + 1e-8)
        return float(min(score / 10.0, 1.0))

    def _decide_action(
        self,
        anomaly_score: float,
        observations: dict[str, Any],
        action_type: str = "flag_anomaly",
    ) -> AgentAction:
        """Materialise the policy-selected action.

        ``action_type`` comes from :meth:`select_action_with_policy` (the
        epsilon-greedy Q-policy).  Severity is derived from the anomaly score;
        the rationale is action-type-specific so the audit trail records
        *why* the policy chose that action, not a single template string.
        """
        severity = (
            "critical" if anomaly_score > 0.95 else ("high" if anomaly_score > 0.8 else "medium")
        )
        rationales = {
            "flag_anomaly": f"Flagging anomaly (score {anomaly_score:.3f})",
            "escalate": f"Escalating {severity}-severity anomaly (score {anomaly_score:.3f})",
            "suppress": f"Suppressing low-confidence signal (score {anomaly_score:.3f})",
            "investigate": f"Investigating uncertain signal (score {anomaly_score:.3f})",
            "log": f"Logging observation (score {anomaly_score:.3f})",
        }
        return AgentAction(
            action_type=action_type,
            parameters={"severity": severity},
            confidence=anomaly_score,
            rationale=rationales.get(action_type, rationales["flag_anomaly"]),
        )

    def _learn_from_action(self, action: AgentAction) -> None:
        """Learn from action outcomes using Q-learning with experience replay.

        Implements temporal difference learning:
        Q(s, a) <- Q(s, a) + α * (r + γ * max_a' Q(s', a') - Q(s, a))

        Args:
            action: The action that was taken with its outcome
        """
        # Compute reward from action outcome
        reward = self._compute_action_reward(action)
        action.outcome = reward

        # Prefer the state features the policy actually selected on (set in
        # ``autonomous_detect``) so the TD update writes the exact Q-key the
        # epsilon-greedy read; fall back to deriving them from the action for
        # direct callers of ``_learn_from_action``.
        state_features = (
            action.state_features
            if action.state_features is not None
            else self._extract_state_features(action)
        )
        action.state_hash = hash(state_features)

        # Store experience
        experience = Experience(
            state_features=state_features,
            action_type=action.action_type,
            reward=reward,
            next_state_features=None,  # Terminal for single-step actions
            done=True,
        )
        self.experience_buffer.append(experience)

        # Update Q-value for this state-action pair
        state_bucket = self._discretize_state(state_features)
        key = (state_bucket, action.action_type)

        current_q = self._q_table.get(key, 0.0)
        # TD update (terminal state, so no next state value)
        td_target = reward * self.learning_config.reward_scale
        td_error = td_target - current_q
        new_q = current_q + self.learning_config.learning_rate * td_error
        self._q_table[key] = new_q

        # Update policy metrics
        self.policy_metrics.total_rewards += reward
        if reward > 0:
            self.policy_metrics.successful_actions += 1
        else:
            self.policy_metrics.failed_actions += 1

        # Decay exploration rate
        self.exploration_rate = max(
            self.learning_config.min_exploration_rate,
            self.exploration_rate * self.learning_config.exploration_decay,
        )

        # Perform experience replay if buffer is large enough
        if len(self.experience_buffer) >= self.learning_config.batch_size:
            self._experience_replay()

    def _compute_action_reward(self, action: AgentAction) -> float:
        """Compute reward signal for an action.

        Reward structure:
        - High confidence correct actions: +1.0
        - Medium confidence actions: confidence score
        - Escalations when needed: +0.5
        - False positives (low confidence flags): -0.5

        Args:
            action: The action taken

        Returns:
            Reward value in [-1, 1] range
        """
        base_reward = 0.0

        if action.action_type == "flag_anomaly":
            # Reward based on confidence - high confidence flags are better
            if action.confidence > 0.8:
                base_reward = 1.0
            elif action.confidence > 0.5:
                base_reward = action.confidence
            else:
                # Low confidence flags are penalized (potential false positives)
                base_reward = -0.5

        elif action.action_type == "escalate":
            # Escalation is appropriate for high-severity cases
            severity = action.parameters.get("severity", "medium")
            if severity == "high":
                base_reward = 0.8
            elif severity == "critical":
                base_reward = 1.0
            else:
                base_reward = 0.3

        elif action.action_type == "suppress":
            # Suppression is good for low-confidence anomalies
            if action.confidence < 0.3:
                base_reward = 0.5
            else:
                base_reward = -0.3  # Suppressing real anomalies is bad

        elif action.action_type == "investigate":
            # Investigation is neutral but encouraged for uncertain cases
            base_reward = 0.2

        elif action.action_type == "log":
            # Logging is minimal but acceptable
            base_reward = 0.1

        return float(np.clip(base_reward, -1.0, 1.0))

    def _extract_state_features(self, action: AgentAction) -> tuple[float, ...]:
        """Extract state features from action context for Q-learning.

        Args:
            action: The action with context

        Returns:
            Tuple of state features (hashable)
        """
        features = [
            action.confidence,
            1.0 if action.parameters.get("severity") == "high" else 0.0,
            1.0 if action.parameters.get("severity") == "critical" else 0.0,
            self.autonomy_level,
            len(self.action_history) / 100.0,  # Normalized action count
        ]
        return tuple(float(f) for f in features)

    def _discretize_state(self, state_features: tuple[float, ...]) -> int:
        """Discretize continuous state features into bucket index.

        Uses binning to create discrete state space for Q-table.

        Args:
            state_features: Continuous state features

        Returns:
            Integer bucket index
        """
        # Discretize each feature into 10 bins
        bins = 10
        discretized = []
        for f in state_features:
            bin_idx = int(np.clip(f * bins, 0, bins - 1))
            discretized.append(bin_idx)

        # Create unique hash from discretized values
        bucket = 0
        for i, val in enumerate(discretized):
            bucket += val * (bins**i)

        return bucket

    def _experience_replay(self) -> None:
        """Perform experience replay for stable learning.

        Samples random batch from experience buffer and performs Q-learning updates.
        """
        if len(self.experience_buffer) < self.learning_config.batch_size:
            return

        # Sample random batch
        indices = self._rng.choice(
            len(self.experience_buffer),
            size=self.learning_config.batch_size,
            replace=False,
        )

        batch_td_errors = []

        for idx in indices:
            exp = self.experience_buffer[idx]
            state_bucket = self._discretize_state(exp.state_features)
            key = (state_bucket, exp.action_type)

            current_q = self._q_table.get(key, 0.0)

            if exp.done or exp.next_state_features is None:
                # Terminal state
                td_target = exp.reward * self.learning_config.reward_scale
            else:
                # Non-terminal: include discounted future value
                next_bucket = self._discretize_state(exp.next_state_features)
                max_next_q = max(
                    self._q_table.get((next_bucket, a), 0.0) for a in self.ACTION_TYPES
                )
                td_target = (
                    exp.reward * self.learning_config.reward_scale
                    + self.learning_config.discount_factor * max_next_q
                )

            td_error = td_target - current_q
            batch_td_errors.append(abs(td_error))

            new_q = current_q + self.learning_config.learning_rate * td_error
            self._q_table[key] = new_q

        # Track convergence via average TD error
        avg_td_error = float(np.mean(batch_td_errors))
        self.policy_metrics.convergence_history.append(avg_td_error)

    def get_q_value(self, state_features: tuple[float, ...], action_type: str) -> float:
        """Get Q-value for a state-action pair.

        Args:
            state_features: State features tuple
            action_type: Type of action

        Returns:
            Q-value (0.0 if not seen before)
        """
        state_bucket = self._discretize_state(state_features)
        return self._q_table.get((state_bucket, action_type), 0.0)

    def select_action_with_policy(
        self, state_features: tuple[float, ...], available_actions: list[str] | None = None
    ) -> str:
        """Select action using epsilon-greedy policy.

        Args:
            state_features: Current state features
            available_actions: List of available action types (defaults to all)

        Returns:
            Selected action type
        """
        if available_actions is None:
            available_actions = self.ACTION_TYPES

        # Epsilon-greedy exploration
        if self._rng.random() < self.exploration_rate:
            return str(self._rng.choice(available_actions))

        # Greedy selection based on Q-values
        q_values = [
            (action, self.get_q_value(state_features, action)) for action in available_actions
        ]
        best_action = max(q_values, key=lambda x: x[1])[0]
        return best_action

    def _detect_trend(self, data: np.ndarray[Any, Any]) -> str:
        """Detect trend in data."""
        flat_data = data.flatten()
        if len(flat_data) < 2:
            return "stable"
        diff = np.diff(flat_data)
        if np.mean(diff) > 0:
            return "increasing"
        elif np.mean(diff) < 0:
            return "decreasing"
        return "stable"

    def execute_workflow(
        self, workflow_definition: dict[str, Any], input_data: np.ndarray[Any, Any]
    ) -> dict[str, Any]:
        """Execute complete workflow autonomously.

        Bain 2025: "At full potential, agents will run complete processes and workflows."
        Implements end-to-end workflow execution with minimal human oversight.

        Args:
            workflow_definition: Dict defining workflow steps and conditions
            input_data: Input data for workflow

        Returns:
            Workflow execution results with outcomes and actions
        """
        workflow_id = workflow_definition.get("id", "workflow_001")
        steps = workflow_definition.get("steps", [])

        self.state = AgentState.OBSERVING

        workflow_results = {
            "workflow_id": workflow_id,
            "steps_executed": [],
            "outputs": {},
            "autonomous_decisions": [],
            "human_oversight_required": False,
        }

        current_data = input_data

        # Build a step-id → index map so ``decision_point`` steps can
        # branch to a named target.  Steps without an explicit ``id``
        # field receive the canonical ``step_{idx}`` synthetic id used
        # everywhere else in this method.
        id_to_idx: dict[str, int] = {}
        for i, step in enumerate(steps):
            sid = step.get("id", f"step_{i}")
            id_to_idx[sid] = i

        # Index-based iteration so ``decision_point`` can jump.  Cycle
        # detection: every jump (vs. linear advance) counts against
        # ``max_jumps``; once exceeded the workflow halts with an
        # explicit ``branching_cycle_detected`` status so an infinite
        # loop in operator-authored workflows fails closed rather than
        # hanging the agent.  The bound scales with ``len(steps)`` so
        # legitimately complex workflows still complete; ``4×`` mirrors
        # the AWS Step Functions practical visit limit for a state.
        max_jumps = max(len(steps) * 4, 64)
        jumps_taken = 0
        step_idx = 0

        while 0 <= step_idx < len(steps):
            step = steps[step_idx]
            step_id = step.get("id", f"step_{step_idx}")
            step_type = step.get("type", "unknown")

            self.state = AgentState.ANALYZING

            # ``next_step_idx`` overrides the default linear advance
            # when set by a ``decision_point``.  Branch targets that
            # cannot be resolved are recorded as ``branching_errors``
            # so the operator sees the gap; resolution failures fall
            # through to linear advance rather than halting silently.
            next_step_idx: int | None = None

            if step_type == "anomaly_detection":
                result = self.autonomous_detect(current_data)
                workflow_results["steps_executed"].append(
                    {
                        "step_id": step_id,
                        "type": step_type,
                        "result": result,
                    }
                )

                if result["anomaly_detected"] and step.get("escalate_on_anomaly", False):
                    workflow_results["human_oversight_required"] = True
                    workflow_results["escalation_reason"] = f"Anomaly detected in {step_id}"
                    break

            elif step_type == "data_transformation":
                transformation = step.get("transformation", "normalize")
                transformed_data = self._apply_transformation(current_data, transformation)
                current_data = transformed_data
                workflow_results["steps_executed"].append(
                    {
                        "step_id": step_id,
                        "type": step_type,
                        "transformation": transformation,
                    }
                )

            elif step_type == "decision_point":
                condition = step.get("condition", {})
                decision = self._evaluate_condition(current_data, condition)
                workflow_results["autonomous_decisions"].append(
                    {
                        "step_id": step_id,
                        "decision": decision,
                        "rationale": f"Condition {condition} evaluated to {decision}",
                    }
                )

                target_key: str | None = None
                if decision and step.get("on_true"):
                    target_key = step["on_true"]
                elif not decision and step.get("on_false"):
                    target_key = step["on_false"]

                if target_key is not None:
                    if target_key in id_to_idx:
                        next_step_idx = id_to_idx[target_key]
                    else:
                        workflow_results.setdefault("branching_errors", []).append(
                            {
                                "step_id": step_id,
                                "target": target_key,
                                "reason": "unknown step id",
                            }
                        )

            elif step_type == "action":
                self.state = AgentState.ACTING
                action_type = step.get("action", "log")
                action_result = self._execute_action(action_type, current_data)
                workflow_results["steps_executed"].append(
                    {
                        "step_id": step_id,
                        "type": step_type,
                        "action": action_type,
                        "result": action_result,
                    }
                )

            if next_step_idx is not None:
                jumps_taken += 1
                if jumps_taken > max_jumps:
                    workflow_results.setdefault("branching_errors", []).append(
                        {
                            "step_id": step_id,
                            "reason": "max_jumps exceeded; possible cycle",
                            "max_jumps": max_jumps,
                        }
                    )
                    workflow_results["status"] = "branching_cycle_detected"
                    break
                step_idx = next_step_idx
            else:
                step_idx += 1

        self.state = AgentState.LEARNING
        self._learn_from_workflow(workflow_results)

        self.state = AgentState.IDLE

        # Preserve an already-set terminal status (e.g.
        # ``branching_cycle_detected``).  Only synthesize the
        # ``completed``/``escalated`` status when no terminal state
        # has been raised by the executor loop itself.
        if "status" not in workflow_results:
            workflow_results["status"] = (
                "completed" if not workflow_results["human_oversight_required"] else "escalated"
            )
        workflow_results["total_steps"] = len(steps)
        workflow_results["completed_steps"] = len(workflow_results["steps_executed"])

        return workflow_results

    def _apply_transformation(
        self, data: np.ndarray[Any, Any], transformation: str
    ) -> np.ndarray[Any, Any]:
        """Apply data transformation."""
        if transformation == "normalize":
            return np.asarray((data - np.mean(data)) / (np.std(data) + 1e-8))  # type: ignore[no-any-return, unused-ignore]
        elif transformation == "scale":
            return np.asarray((data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8))  # type: ignore[no-any-return, unused-ignore]
        else:
            return data

    def _evaluate_condition(self, data: np.ndarray[Any, Any], condition: dict[str, Any]) -> bool:
        """Evaluate decision condition."""
        metric = condition.get("metric", "mean")
        operator = condition.get("operator", ">")
        threshold = condition.get("threshold", 0.5)

        if metric == "mean":
            value = np.mean(data)
        elif metric == "max":
            value = np.max(data)
        elif metric == "std":
            value = np.std(data)
        else:
            value = 0.0

        if operator == ">":
            return bool(value > threshold)
        elif operator == "<":
            return bool(value < threshold)
        elif operator == "==":
            return bool(abs(value - threshold) < 1e-6)
        else:
            return False

    def _execute_action(self, action_type: str, data: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Execute workflow action."""
        if action_type == "log":
            return {"logged": True, "data_summary": f"mean={np.mean(data):.3f}"}
        elif action_type == "alert":
            return {"alert_sent": True, "severity": "medium"}
        elif action_type == "store":
            return {"stored": True, "timestamp": "now"}
        else:
            return {"action": action_type, "status": "unknown"}

    def _learn_from_workflow(self, workflow_results: dict[str, Any]) -> None:
        """Learn from workflow execution outcomes using policy gradient-style updates.

        Implements workflow-level reinforcement learning:
        - Track workflow success/failure patterns
        - Update value estimates for workflow configurations
        - Adjust decision thresholds based on outcomes
        - Learn temporal patterns from step sequences

        Args:
            workflow_results: Complete workflow execution results
        """
        workflow_id = workflow_results.get("workflow_id", "unknown")
        status = workflow_results.get("status", "unknown")
        steps_executed = workflow_results.get("steps_executed", [])
        autonomous_decisions = workflow_results.get("autonomous_decisions", [])

        # Compute workflow reward
        workflow_reward = self._compute_workflow_reward(workflow_results)

        # Update workflow value estimate (exponential moving average)
        alpha = self.learning_config.learning_rate
        current_value = self._workflow_value_estimates.get(workflow_id, 0.0)
        self._workflow_value_estimates[workflow_id] = (
            1 - alpha
        ) * current_value + alpha * workflow_reward

        # Track execution count
        self._workflow_execution_counts[workflow_id] = (
            self._workflow_execution_counts.get(workflow_id, 0) + 1
        )

        # Learn from individual steps as a sequence
        prev_state_features = None
        for step in steps_executed:
            step_type = step.get("type", "unknown")
            _ = step.get("result", {})  # Result used in _extract_step_features

            # Extract features for this step
            current_state_features = self._extract_step_features(step, workflow_results)

            # Create experience for multi-step learning
            if prev_state_features is not None:
                # Compute step reward (discounted portion of workflow reward)
                step_idx = steps_executed.index(step)
                discount = self.learning_config.discount_factor ** (
                    len(steps_executed) - step_idx - 1
                )
                step_reward = workflow_reward * discount * 0.5  # Partial credit

                experience = Experience(
                    state_features=prev_state_features,
                    action_type=step_type,
                    reward=step_reward,
                    next_state_features=current_state_features,
                    done=False,
                )
                self.experience_buffer.append(experience)

            prev_state_features = current_state_features

        # Final step gets remaining reward
        if prev_state_features is not None and steps_executed:
            final_step = steps_executed[-1]
            experience = Experience(
                state_features=prev_state_features,
                action_type=final_step.get("type", "unknown"),
                reward=workflow_reward * 0.5,  # Terminal reward
                next_state_features=None,
                done=True,
            )
            self.experience_buffer.append(experience)

        # Learn from autonomous decisions (policy improvement)
        for decision in autonomous_decisions:
            decision_value = 1.0 if decision.get("decision", False) else 0.0
            decision_state = (
                decision_value,
                workflow_reward,
                float(len(steps_executed)) / 10.0,
                self.autonomy_level,
                1.0 if status == "completed" else 0.0,
            )

            state_bucket = self._discretize_state(decision_state)
            action = "decide_true" if decision.get("decision") else "decide_false"
            key = (state_bucket, action)

            current_q = self._q_table.get(key, 0.0)
            td_target = workflow_reward * self.learning_config.reward_scale
            new_q = current_q + self.learning_config.learning_rate * (td_target - current_q)
            self._q_table[key] = new_q

        # Update policy metrics
        self.policy_metrics.episode_count += 1
        self.policy_metrics.total_rewards += workflow_reward

        # Perform batch learning from accumulated experiences
        if len(self.experience_buffer) >= self.learning_config.batch_size:
            self._experience_replay()

    def _compute_workflow_reward(self, workflow_results: dict[str, Any]) -> float:
        """Compute reward for entire workflow execution.

        Reward structure:
        - Successful completion: +1.0
        - Completed with escalation: +0.5 (human oversight was needed)
        - Partial completion: proportional to steps completed
        - Failure: -0.5

        Args:
            workflow_results: Workflow execution results

        Returns:
            Workflow reward in [-1, 1] range
        """
        status = workflow_results.get("status", "unknown")
        total_steps = workflow_results.get("total_steps", 1)
        completed_steps = workflow_results.get("completed_steps", 0)
        human_oversight_required = workflow_results.get("human_oversight_required", False)

        if status == "completed":
            if human_oversight_required:
                # Completed but needed escalation - partial success
                base_reward = 0.5
            else:
                # Fully autonomous completion
                base_reward = 1.0
        elif status == "escalated":
            # Appropriately escalated - this is correct behavior
            base_reward = 0.6
        # Unknown or failed status
        elif total_steps > 0:
            completion_ratio = completed_steps / total_steps
            base_reward = completion_ratio * 0.5 - 0.25
        else:
            base_reward = -0.5

        # Bonus for efficient execution (fewer decisions needed)
        autonomous_decisions = workflow_results.get("autonomous_decisions", [])
        efficiency_bonus = max(0, 0.1 - len(autonomous_decisions) * 0.02)

        return float(np.clip(base_reward + efficiency_bonus, -1.0, 1.0))

    def _extract_step_features(
        self, step: dict[str, Any], workflow_results: dict[str, Any]
    ) -> tuple[float, ...]:
        """Extract features from a workflow step for learning.

        Args:
            step: Step execution data
            workflow_results: Overall workflow context

        Returns:
            Tuple of state features
        """
        step_type = step.get("type", "unknown")
        step_result = step.get("result", {})

        # Encode step type
        step_type_encoding = {
            "anomaly_detection": 0.0,
            "data_transformation": 0.25,
            "decision_point": 0.5,
            "action": 0.75,
            "unknown": 1.0,
        }

        features = [
            step_type_encoding.get(step_type, 1.0),
            float(step_result.get("anomaly_score", 0.5)) if isinstance(step_result, dict) else 0.5,
            (
                float(step_result.get("anomaly_detected", False))
                if isinstance(step_result, dict)
                else 0.0
            ),
            self.autonomy_level,
            float(workflow_results.get("completed_steps", 0))
            / max(workflow_results.get("total_steps", 1), 1),
        ]

        return tuple(features)

    def get_autonomy_metrics(self) -> dict[str, Any]:
        """Get metrics on autonomous operation including learning statistics.

        Returns:
            Metrics showing autonomy level, decision count, intervention rate,
            and reinforcement learning statistics
        """
        total_actions = len(self.action_history)

        # Compute success rate
        total_evaluated = (
            self.policy_metrics.successful_actions + self.policy_metrics.failed_actions
        )
        success_rate = (
            self.policy_metrics.successful_actions / total_evaluated if total_evaluated > 0 else 0.0
        )

        # Compute average reward
        avg_reward = self.policy_metrics.total_rewards / max(total_evaluated, 1)

        # Compute convergence status from recent TD errors
        is_converged = False
        if len(self.policy_metrics.convergence_history) >= 10:
            recent_errors = self.policy_metrics.convergence_history[-10:]
            is_converged = bool(np.mean(recent_errors) < 0.1)  # type: ignore[assignment, unused-ignore]

        return {
            "autonomy_level": self.autonomy_level,
            "total_autonomous_actions": total_actions,
            "current_state": self.state.name,
            "decision_threshold": self.decision_threshold,
            "actions_without_intervention": total_actions,
            "bain_vision_alignment": "Agents running complete processes with minimal oversight",
            # Reinforcement learning metrics
            "rl_metrics": {
                "exploration_rate": self.exploration_rate,
                "experience_buffer_size": len(self.experience_buffer),
                "q_table_size": len(self._q_table),
                "total_rewards": self.policy_metrics.total_rewards,
                "average_reward": avg_reward,
                "success_rate": success_rate,
                "episode_count": self.policy_metrics.episode_count,
                "is_converged": is_converged,
                "workflow_patterns_learned": len(self._workflow_value_estimates),
            },
        }
