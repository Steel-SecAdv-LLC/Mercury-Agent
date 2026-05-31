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
Hierarchical Planning Agent for Mercury Agent.

Implements hierarchical reinforcement learning and planning inspired by:
- "Hierarchical Reinforcement Learning with Options" (Sutton et al., 1999)
- "MAXQ Value Function Decomposition" (Dietterich, 2000)
- "HAM: Hierarchies of Abstract Machines" (Parr & Russell, 1998)
- "Feudal Reinforcement Learning" (Dayan & Hinton, 1993)

Hierarchical planning decomposes complex tasks into:
1. High-level goals (strategic layer)
2. Mid-level subgoals (tactical layer)
3. Low-level actions (operational layer)

This architecture improves:
- Sample efficiency (reuse high-level skills)
- Interpretability (explicit goal hierarchy)
- Transfer learning (compose existing skills)
- Long-horizon planning (temporal abstraction)

This module enables Mercury Agent to handle complex, multi-step
anomaly detection and response tasks.
"""

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

# Planning parameters
MAX_PLAN_DEPTH = 5
MAX_SUBGOALS = 10
MAX_OPTIONS = 20
DEFAULT_DISCOUNT = 0.99
PLANNING_HORIZON = 50


class GoalStatus(Enum):
    """Status of a goal in the hierarchy."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


class PlannerType(Enum):
    """Types of hierarchical planners."""

    OPTIONS = "options"  # Options framework
    MAXQ = "maxq"  # MAXQ decomposition
    FEUDAL = "feudal"  # Feudal networks
    HAM = "ham"  # Hierarchies of Abstract Machines


class AbstractionLevel(Enum):
    """Levels of temporal abstraction."""

    STRATEGIC = "strategic"  # Long-term goals (minutes to hours)
    TACTICAL = "tactical"  # Medium-term subgoals (seconds to minutes)
    OPERATIONAL = "operational"  # Immediate actions (milliseconds)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class Goal:
    """A goal in the hierarchical plan.

    Represents a desired state or outcome to achieve.

    Attributes:
        goal_id: Unique identifier
        description: Human-readable description
        level: Abstraction level
        parent_id: Parent goal (if any)
        child_ids: Child subgoals
        preconditions: Required conditions to start
        postconditions: Expected conditions after completion
        priority: Goal priority (higher = more important)
        deadline: Optional deadline
        status: Current status
        progress: Completion progress (0-1)
        reward: Reward for achieving goal
        created_at: Creation timestamp
        metadata: Additional metadata
    """

    goal_id: str
    description: str
    level: AbstractionLevel
    parent_id: str | None = None
    child_ids: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    priority: float = 0.5
    deadline: float | None = None
    status: GoalStatus = GoalStatus.PENDING
    progress: float = 0.0
    reward: float = 1.0
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.goal_id,
            "description": self.description,
            "level": self.level.value,
            "status": self.status.value,
            "progress": self.progress,
            "priority": self.priority,
            "children": len(self.child_ids),
        }


@dataclass
class Option:
    """
    A temporally extended action (option).

    Options are multi-step policies that run until termination.

    Attributes:
        option_id: Unique identifier
        name: Option name
        initiation_set: States where option can start
        policy: Action selection policy
        termination_condition: When option terminates
        expected_duration: Expected number of steps
        expected_reward: Expected cumulative reward
        skill_level: Reusability/skill level
        metadata: Additional metadata
    """

    option_id: str
    name: str
    initiation_set: dict[str, Any]
    policy: dict[str, str]  # state_key -> action
    termination_condition: dict[str, Any]
    expected_duration: float = 10.0
    expected_reward: float = 0.0
    skill_level: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def can_initiate(self, state: dict[str, Any]) -> bool:
        """Check if option can start in given state."""
        for key, expected in self.initiation_set.items():
            if key not in state:
                return False
            if isinstance(expected, (list, tuple)):
                if state[key] not in expected:
                    return False
            elif state[key] != expected:
                return False
        return True

    def should_terminate(self, state: dict[str, Any]) -> bool:
        """Check if option should terminate."""
        for key, expected in self.termination_condition.items():
            if key in state and state[key] == expected:
                return True
        return False

    def get_action(self, state: dict[str, Any]) -> str:
        """Get action for current state."""
        # Simple lookup policy
        state_key = str(sorted(state.items()))[:50]
        return self.policy.get(state_key, "default_action")


@dataclass
class Subgoal:
    """
    A subgoal in the tactical layer.

    Bridges strategic goals and operational actions.

    Attributes:
        subgoal_id: Unique identifier
        goal_id: Parent goal
        description: Subgoal description
        target_state: Target state to achieve
        action_sequence: Planned action sequence
        estimated_steps: Estimated steps to complete
        priority: Subgoal priority
        status: Current status
        level: Abstraction level of the subgoal
    """

    subgoal_id: str
    goal_id: str
    description: str
    target_state: dict[str, Any]
    action_sequence: list[str] = field(default_factory=list)
    estimated_steps: int = 10
    priority: float = 0.5
    status: GoalStatus = GoalStatus.PENDING
    level: AbstractionLevel = AbstractionLevel.TACTICAL


@dataclass
class PlanNode:
    """
    A node in the hierarchical plan tree.

    Attributes:
        node_id: Unique identifier
        goal: Associated goal
        subgoals: Child subgoals
        options: Available options at this node
        depth: Depth in plan tree
        value: Estimated value
        visited: Visit count
        parent: Parent node
    """

    node_id: str
    goal: Goal
    subgoals: list[Subgoal] = field(default_factory=list)
    options: list[Option] = field(default_factory=list)
    depth: int = 0
    value: float = 0.0
    visited: int = 0
    parent: PlanNode | None = None

    def __lt__(self, other: PlanNode) -> bool:
        """Comparison for priority queue."""
        return self.value > other.value  # Higher value = higher priority


@dataclass
class HierarchicalPlan:
    """
    A complete hierarchical plan.

    Attributes:
        plan_id: Unique identifier
        root_goal: Top-level goal
        goal_hierarchy: All goals by ID
        subgoals: All subgoals
        options_used: Options included in plan
        estimated_reward: Total estimated reward
        estimated_duration: Total estimated duration
        created_at: Creation timestamp
    """

    plan_id: str
    root_goal: Goal
    goal_hierarchy: dict[str, Goal]
    subgoals: list[Subgoal]
    options_used: list[Option]
    estimated_reward: float
    estimated_duration: float
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plan_id": self.plan_id,
            "root_goal": self.root_goal.to_dict(),
            "num_goals": len(self.goal_hierarchy),
            "num_subgoals": len(self.subgoals),
            "num_options": len(self.options_used),
            "estimated_reward": self.estimated_reward,
            "estimated_duration": self.estimated_duration,
        }


@dataclass
class PlanExecutionState:
    """
    State of plan execution.

    Attributes:
        plan: The plan being executed
        current_goal: Currently active goal
        current_subgoal: Currently active subgoal
        current_option: Currently executing option
        steps_taken: Total steps taken
        reward_accumulated: Total reward so far
        goal_stack: Stack of active goals
    """

    plan: HierarchicalPlan
    current_goal: Goal | None = None
    current_subgoal: Subgoal | None = None
    current_option: Option | None = None
    steps_taken: int = 0
    reward_accumulated: float = 0.0
    goal_stack: list[str] = field(default_factory=list)


# =============================================================================
# Hierarchical Value Function
# =============================================================================


class HierarchicalValueFunction:
    """
    Value function decomposition for hierarchical planning.

    Implements MAXQ-style value decomposition:
    V(s, g) = V(s, g1) + C(s, g1, g) + V(s, g2) + C(s, g2, g) + ...

    Where C is the completion function (reward for completing subgoal).
    """

    def __init__(
        self,
        discount: float = DEFAULT_DISCOUNT,
        num_levels: int = 3,
    ):
        """
        Initialize value function.

        Args:
            discount: Discount factor (gamma)
            num_levels: Number of hierarchy levels
        """
        self.discount = discount
        self.num_levels = num_levels

        # Value tables
        self._goal_values: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._completion_values: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._option_values: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Visit counts for exploration
        self._visit_counts: dict[str, int] = defaultdict(int)

    def get_value(
        self,
        state: dict[str, Any],
        goal: Goal,
    ) -> float:
        """
        Get value estimate for state-goal pair.

        Args:
            state: Current state
            goal: Goal to achieve

        Returns:
            Value estimate
        """
        state_key = self._state_to_key(state)
        return self._goal_values[goal.goal_id][state_key]

    def get_completion_value(
        self,
        state: dict[str, Any],
        subgoal: Subgoal,
        parent_goal: Goal,
    ) -> float:
        """
        Get completion value for subgoal.

        Args:
            state: Current state
            subgoal: Subgoal being completed
            parent_goal: Parent goal

        Returns:
            Completion value
        """
        state_key = self._state_to_key(state)
        key = f"{subgoal.subgoal_id}_{parent_goal.goal_id}"
        return self._completion_values[key][state_key]

    def update_value(
        self,
        state: dict[str, Any],
        goal: Goal,
        reward: float,
        next_state: dict[str, Any],
        done: bool,
    ) -> None:
        """
        Update value estimate with TD learning.

        Args:
            state: Current state
            goal: Goal being pursued
            reward: Reward received
            next_state: Resulting state
            done: Whether goal is complete
        """
        state_key = self._state_to_key(state)
        next_key = self._state_to_key(next_state)

        # TD update
        current_value = self._goal_values[goal.goal_id][state_key]
        next_value = 0.0 if done else self._goal_values[goal.goal_id][next_key]

        td_target = reward + self.discount * next_value
        td_error = td_target - current_value

        # Learning rate based on visit count
        self._visit_counts[state_key] += 1
        alpha = 1.0 / (1 + np.sqrt(self._visit_counts[state_key]))

        self._goal_values[goal.goal_id][state_key] = current_value + alpha * td_error

    def compute_value(
        self,
        state: dict[str, Any],
        option: str,
    ) -> float:
        """
        Compute value for state-option pair (simplified API).

        Args:
            state: Current state dict
            option: Option name/identifier

        Returns:
            Value estimate as float
        """
        state_key = self._state_to_key(state)

        # Check if we have a stored value
        if option in self._option_values and state_key in self._option_values[option]:
            return self._option_values[option][state_key]

        # Compute heuristic value based on state features
        base_value = 0.5

        # Adjust based on state features
        if "threat_level" in state:
            base_value += state["threat_level"] * 0.3
        if "system_health" in state:
            base_value += (1 - state["system_health"]) * 0.2

        self._option_values[option][state_key] = base_value
        return base_value

    def _state_to_key(self, state: dict[str, Any]) -> str:
        """Convert state to hashable key."""
        items = sorted((k, str(v)[:20]) for k, v in state.items())
        return str(items)[:100]


# =============================================================================
# Goal Decomposer
# =============================================================================


class GoalDecomposer:
    """
    Decomposes high-level goals into subgoals.

    Uses domain knowledge and learned patterns to create meaningful goal hierarchies.
    """

    def __init__(self, max_subgoals: int = MAX_SUBGOALS):
        """
        Initialize decomposer.

        Args:
            max_subgoals: Maximum subgoals per goal
        """
        self.max_subgoals = max_subgoals
        self._decomposition_counter = 0

        # Domain-specific decomposition templates
        self._templates = {
            "detect_anomaly": [
                "collect_data",
                "extract_features",
                "compute_scores",
                "classify_anomaly",
            ],
            "respond_to_threat": [
                "assess_severity",
                "determine_response",
                "execute_response",
                "verify_mitigation",
            ],
            "monitor_system": [
                "initialize_monitoring",
                "collect_metrics",
                "analyze_patterns",
                "report_status",
            ],
            "investigate_incident": [
                "gather_evidence",
                "correlate_events",
                "identify_root_cause",
                "document_findings",
            ],
        }

    def decompose(
        self,
        goal: Goal | dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> list[Subgoal] | list[dict[str, Any]]:
        """
        Decompose a goal into subgoals.

        Args:
            goal: Goal to decompose (Goal object or dict)
            context: Optional context for decomposition

        Returns:
            List of subgoals (Subgoal objects or dicts depending on input)
        """
        # Handle dict input (test API)
        if isinstance(goal, dict):
            description = str(goal.get("type", goal.get("description", "unknown_goal")))
            goal_id = f"goal_dict_{self._decomposition_counter:06d}"
            priority = 0.5

            template_key = self._find_template(description)
            if template_key:
                template_subgoals = self._templates[template_key]
            else:
                # Generate generic subgoals based on goal type
                template_subgoals = [
                    f"prepare_{description}",
                    f"execute_{description}",
                    f"verify_{description}",
                ]

            # Return list of dicts for dict input
            result = []
            for i, subgoal_desc in enumerate(template_subgoals[: self.max_subgoals]):
                self._decomposition_counter += 1
                result.append(
                    {
                        "type": subgoal_desc,
                        "level": i,
                        "parent": goal_id,
                        "priority": priority * (1 - i * 0.1),
                        "constraints": goal.get("constraints", {}),
                    }
                )
            return result

        # Original Goal object handling
        subgoals: list[Subgoal] = []

        template_key = self._find_template(goal.description)
        if template_key:
            template_subgoals = self._templates[template_key]
        else:
            template_subgoals = self._generate_generic_subgoals(goal)

        # Create subgoals from template
        for i, subgoal_desc in enumerate(template_subgoals[: self.max_subgoals]):
            self._decomposition_counter += 1
            subgoal_id = f"subgoal_{self._decomposition_counter:06d}"

            subgoal = Subgoal(
                subgoal_id=subgoal_id,
                goal_id=goal.goal_id,
                description=subgoal_desc,
                target_state=self._infer_target_state(subgoal_desc, context),
                estimated_steps=10 + i * 5,  # Later subgoals take longer
                priority=goal.priority * (1 - i * 0.1),  # Decreasing priority
            )
            subgoals.append(subgoal)
            goal.child_ids.append(subgoal_id)

        return subgoals

    def _find_template(self, description: str) -> str | None:
        """Find matching decomposition template."""
        description_lower = description.lower()
        for template_key in self._templates:
            template_words = template_key.replace("_", " ")
            if template_words in description_lower:
                return template_key
        return None

    def _generate_generic_subgoals(self, goal: Goal) -> list[str]:
        """Generate generic subgoals for unknown goal types."""
        return [
            f"initialize_{goal.goal_id}",
            f"process_{goal.goal_id}",
            f"complete_{goal.goal_id}",
        ]

    def _infer_target_state(
        self,
        description: str,
        context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Infer target state from description."""
        target = {"subgoal_complete": True}

        # Add context-specific targets
        if context:
            if "anomaly_threshold" in context:
                target["threshold_met"] = True

        return target


# =============================================================================
# Option Library
# =============================================================================


class OptionLibrary:
    """
    Library of reusable options (skills).

    Manages temporally extended actions that can be composed to achieve complex goals.
    """

    def __init__(self, max_options: int = MAX_OPTIONS):
        """
        Initialize option library.

        Args:
            max_options: Maximum options to store
        """
        self.max_options = max_options
        self._options: dict[str, Option] = {}
        self._option_usage: dict[str, int] = defaultdict(int)
        self._option_counter = 0

        # Initialize built-in options
        self._initialize_builtin_options()

    def _initialize_builtin_options(self) -> None:
        """Initialize built-in anomaly detection options."""
        builtin_options = [
            Option(
                option_id="opt_statistical_detection",
                name="Statistical Anomaly Detection",
                initiation_set={"has_numerical_data": True},
                policy={"default": "compute_z_score"},
                termination_condition={"score_computed": True},
                expected_duration=5.0,
                expected_reward=0.5,
                skill_level=0.8,
            ),
            Option(
                option_id="opt_temporal_analysis",
                name="Temporal Pattern Analysis",
                initiation_set={"has_time_series": True},
                policy={"default": "analyze_trends"},
                termination_condition={"trends_analyzed": True},
                expected_duration=10.0,
                expected_reward=0.6,
                skill_level=0.7,
            ),
            Option(
                option_id="opt_alert_generation",
                name="Alert Generation",
                initiation_set={"anomaly_detected": True},
                policy={"default": "generate_alert"},
                termination_condition={"alert_sent": True},
                expected_duration=2.0,
                expected_reward=0.3,
                skill_level=0.9,
            ),
            Option(
                option_id="opt_data_collection",
                name="Data Collection",
                initiation_set={"data_source_available": True},
                policy={"default": "collect_data"},
                termination_condition={"data_collected": True},
                expected_duration=15.0,
                expected_reward=0.4,
                skill_level=0.6,
            ),
            Option(
                option_id="opt_feature_extraction",
                name="Feature Extraction",
                initiation_set={"raw_data_available": True},
                policy={"default": "extract_features"},
                termination_condition={"features_ready": True},
                expected_duration=8.0,
                expected_reward=0.5,
                skill_level=0.75,
            ),
        ]

        for option in builtin_options:
            self._options[option.option_id] = option

    def get_applicable_options(
        self,
        state: dict[str, Any],
        goal: Goal | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get options applicable in current state.

        Args:
            state: Current state
            goal: Optional goal context

        Returns:
            List of applicable options as dicts
        """
        applicable = []
        for option in self._options.values():
            if option.can_initiate(state):
                applicable.append(option)

        # Sort by expected reward and skill level
        applicable.sort(
            key=lambda o: o.expected_reward * o.skill_level,
            reverse=True,
        )

        # Return as dicts for API compatibility
        return [
            {
                "option_id": o.option_id,
                "name": o.name,
                "initiation_set": o.initiation_set,
                "policy": o.policy,
                "termination_condition": o.termination_condition,
                "expected_duration": o.expected_duration,
                "expected_reward": o.expected_reward,
                "skill_level": o.skill_level,
            }
            for o in applicable
        ]

    def add_option(
        self,
        option: Option | None = None,
        *,
        name: str | None = None,
        initiation_set: dict[str, Any] | None = None,
        policy: dict[str, Any] | None = None,
        termination_condition: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a new option to the library.

        Args:
            option: Option to add (original API)
            name: Option name (keyword API)
            initiation_set: Conditions for initiating option (keyword API)
            policy: Policy dict (keyword API)
            termination_condition: Termination conditions (keyword API)
        """
        if len(self._options) >= self.max_options:
            self._evict_least_used()

        # Handle keyword argument API (test API)
        if option is None and name is not None:
            self._option_counter += 1
            option_id = f"opt_custom_{self._option_counter:06d}"
            option = Option(
                option_id=option_id,
                name=name,
                initiation_set=initiation_set or {},
                policy=policy or {},
                termination_condition=termination_condition or {},
                expected_duration=10.0,
                expected_reward=0.5,
                skill_level=0.5,
            )

        if option is not None:
            self._options[option.option_id] = option

    def record_usage(
        self,
        option_id: str,
        reward: float,
        success: bool,
    ) -> None:
        """
        Record option usage for learning.

        Args:
            option_id: Option that was used
            reward: Reward received
            success: Whether option succeeded
        """
        self._option_usage[option_id] += 1

        if option_id in self._options:
            option = self._options[option_id]
            alpha = 0.1
            option.expected_reward = (1 - alpha) * option.expected_reward + alpha * reward

    def _evict_least_used(self) -> None:
        """Evict least used option."""
        if not self._options:
            return

        # Find least used option (excluding built-ins)
        non_builtin = [
            (oid, self._option_usage.get(oid, 0))
            for oid in self._options
            if not oid.startswith("opt_")
        ]

        if non_builtin:
            least_used = min(non_builtin, key=lambda x: x[1])[0]
            del self._options[least_used]

    def get_statistics(self) -> dict[str, Any]:
        """Get library statistics."""
        return {
            "total_options": len(self._options),
            "builtin_options": sum(1 for o in self._options if o.startswith("opt_")),
            "total_usages": sum(self._option_usage.values()),
        }


# =============================================================================
# Hierarchical Planner
# =============================================================================


class HierarchicalPlanner:
    """
    Main hierarchical planning engine.

    Combines goal decomposition, option selection, and
    value estimation for efficient planning.

    Key capabilities:
    1. Goal hierarchy construction
    2. Option-based action selection
    3. Value function learning
    4. Plan execution monitoring
    """

    def __init__(
        self,
        planner_type: PlannerType = PlannerType.OPTIONS,
        max_depth: int = MAX_PLAN_DEPTH,
        planning_horizon: int = PLANNING_HORIZON,
        discount: float = DEFAULT_DISCOUNT,
    ):
        """
        Initialize hierarchical planner.

        Args:
            planner_type: Type of hierarchical planner
            max_depth: Maximum plan depth
            planning_horizon: Planning horizon steps
            discount: Discount factor
        """
        self.planner_type = planner_type
        self.max_depth = max_depth
        self.planning_horizon = planning_horizon
        self.discount = discount

        # Components
        self.goal_decomposer = GoalDecomposer()
        self.option_library = OptionLibrary()
        self.value_function = HierarchicalValueFunction(discount=discount)

        # Counters
        self._goal_counter = 0
        self._plan_counter = 0

        # Statistics
        self._stats = {
            "plans_created": 0,
            "total_plans": 0,
            "goals_completed": 0,
            "goals_failed": 0,
            "avg_plan_depth": 0.0,
            "avg_reward": 0.0,
        }

        logger.info(
            f"HierarchicalPlanner initialized (type={planner_type.value}, "
            f"max_depth={max_depth})"
        )

    def create_goal(
        self,
        description: str,
        level: AbstractionLevel = AbstractionLevel.STRATEGIC,
        parent_id: str | None = None,
        priority: float = 0.5,
        preconditions: list[str] | None = None,
        postconditions: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Goal:
        """
        Create a new goal.

        Args:
            description: Goal description
            level: Abstraction level
            parent_id: Parent goal ID
            priority: Goal priority
            preconditions: Required conditions
            postconditions: Expected outcomes
            metadata: Additional metadata

        Returns:
            Created Goal object
        """
        self._goal_counter += 1
        goal_id = f"goal_{self._goal_counter:06d}"

        goal = Goal(
            goal_id=goal_id,
            description=description,
            level=level,
            parent_id=parent_id,
            priority=priority,
            preconditions=preconditions or [],
            postconditions=postconditions or [],
            metadata=metadata or {},
        )

        return goal

    def plan(
        self,
        root_goal: Goal,
        current_state: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> HierarchicalPlan:
        """
        Create a hierarchical plan for a goal.

        Args:
            root_goal: Top-level goal to achieve
            current_state: Current system state
            context: Additional planning context

        Returns:
            Complete HierarchicalPlan
        """
        self._plan_counter += 1
        plan_id = f"plan_{self._plan_counter:06d}"

        goal_hierarchy = self._build_goal_hierarchy(root_goal, context)

        all_subgoals: list[Subgoal] = []
        for goal in goal_hierarchy.values():
            if goal.level in [AbstractionLevel.STRATEGIC, AbstractionLevel.TACTICAL]:
                subgoals = self.goal_decomposer.decompose(goal, context)
                all_subgoals.extend(s for s in subgoals if isinstance(s, Subgoal))

        # Select options for subgoals
        options_used: list[Option] = []
        for subgoal in all_subgoals:
            applicable = self.option_library.get_applicable_options(current_state)
            if applicable:
                first_option = applicable[0]
                if isinstance(first_option, Option):
                    options_used.append(first_option)

        estimated_reward = self._estimate_plan_reward(
            goal_hierarchy, all_subgoals, options_used, current_state
        )

        estimated_duration = sum(o.expected_duration for o in options_used)

        plan = HierarchicalPlan(
            plan_id=plan_id,
            root_goal=root_goal,
            goal_hierarchy=goal_hierarchy,
            subgoals=all_subgoals,
            options_used=options_used,
            estimated_reward=estimated_reward,
            estimated_duration=estimated_duration,
        )

        self._stats["plans_created"] += 1
        self._stats["total_plans"] += 1
        self._update_stats(plan)

        return plan

    def create_plan(
        self,
        goal: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Create a plan from goal and state dicts (simplified API).

        Args:
            goal: Goal specification dict with objective, priority, deadline
            state: Current state dict

        Returns:
            Plan dict with actions/steps and estimated_success/confidence
        """
        # Create Goal object from dict
        objective = goal.get("objective", goal.get("description", "achieve_goal"))
        priority = goal.get("priority", "normal")
        priority_value = (
            {"low": 0.3, "normal": 0.5, "high": 0.7, "critical": 0.9}.get(priority, 0.5)
            if isinstance(priority, str)
            else float(priority)
        )

        root_goal = self.create_goal(
            description=objective,
            level=AbstractionLevel.STRATEGIC,
            priority=priority_value,
            metadata=goal,
        )

        # Create hierarchical plan
        hierarchical_plan = self.plan(root_goal, state, goal)

        # Convert to simplified dict format
        actions = []
        for option in hierarchical_plan.options_used:
            actions.append(
                {
                    "name": option.name,
                    "policy": option.policy,
                    "expected_duration": option.expected_duration,
                }
            )

        steps = []
        for subgoal in hierarchical_plan.subgoals:
            steps.append(
                {
                    "description": subgoal.description,
                    "level": (
                        subgoal.level.value
                        if hasattr(subgoal.level, "value")
                        else str(subgoal.level)
                    ),
                    "target_state": subgoal.target_state,
                }
            )

        return {
            "plan_id": hierarchical_plan.plan_id,
            "actions": actions,
            "steps": steps,
            "estimated_success": min(1.0, max(0.0, hierarchical_plan.estimated_reward)),
            "confidence": min(1.0, max(0.0, hierarchical_plan.estimated_reward * 0.8 + 0.2)),
            "estimated_duration": hierarchical_plan.estimated_duration,
        }

    def select_action(
        self,
        state: dict[str, Any],
        execution_state: PlanExecutionState,
    ) -> tuple[str, Option | None]:
        """
        Select next action based on current plan.

        Args:
            state: Current state
            execution_state: Current execution state

        Returns:
            Tuple of (action, option_used)
        """
        # Check if current option should terminate
        if execution_state.current_option:
            if execution_state.current_option.should_terminate(state):
                execution_state.current_option = None

        # Select new option if needed
        if execution_state.current_option is None:
            applicable = self.option_library.get_applicable_options(
                state, execution_state.current_goal
            )
            if applicable:
                first_applicable = applicable[0]
                if isinstance(first_applicable, Option):
                    execution_state.current_option = first_applicable

        # Get action from option
        if execution_state.current_option:
            action = execution_state.current_option.get_action(state)
            return action, execution_state.current_option

        # Fallback action
        return "default_action", None

    def update_on_feedback(
        self,
        state: dict[str, Any],
        action: str,
        reward: float,
        next_state: dict[str, Any],
        execution_state: PlanExecutionState,
    ) -> None:
        """
        Update planner based on feedback.

        Args:
            state: State before action
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            execution_state: Current execution state
        """
        # Update value function
        if execution_state.current_goal:
            goal_complete = self._check_goal_complete(execution_state.current_goal, next_state)
            self.value_function.update_value(
                state,
                execution_state.current_goal,
                reward,
                next_state,
                goal_complete,
            )

        # Update option statistics
        if execution_state.current_option:
            success = reward > 0
            self.option_library.record_usage(
                execution_state.current_option.option_id,
                reward,
                success,
            )

        execution_state.steps_taken += 1
        execution_state.reward_accumulated += reward

    def replan(
        self,
        execution_state: PlanExecutionState,
        current_state: dict[str, Any],
        reason: str = "goal_failed",
    ) -> HierarchicalPlan:
        """
        Replan due to failure or changed conditions.

        Args:
            execution_state: Current execution state
            current_state: Current system state
            reason: Reason for replanning

        Returns:
            New plan
        """
        logger.info(f"Replanning due to: {reason}")

        # Mark current goal as failed if applicable
        if execution_state.current_goal:
            if reason == "goal_failed":
                execution_state.current_goal.status = GoalStatus.FAILED
                self._stats["goals_failed"] += 1

        # Create new plan from root
        return self.plan(
            execution_state.plan.root_goal,
            current_state,
        )

    def _build_goal_hierarchy(
        self,
        root_goal: Goal,
        context: dict[str, Any] | None,
    ) -> dict[str, Goal]:
        """Build complete goal hierarchy."""
        hierarchy: dict[str, Goal] = {root_goal.goal_id: root_goal}

        # BFS to create child goals
        queue: deque[Goal] = deque([root_goal])
        depth = 0

        while queue and depth < self.max_depth:
            level_size = len(queue)
            for _ in range(level_size):
                goal = queue.popleft()

                # Create child goals at next level
                if goal.level == AbstractionLevel.STRATEGIC:
                    child_level = AbstractionLevel.TACTICAL
                elif goal.level == AbstractionLevel.TACTICAL:
                    child_level = AbstractionLevel.OPERATIONAL
                else:
                    continue

                child_goals = self._generate_child_goals(goal, child_level, context)
                for child in child_goals:
                    hierarchy[child.goal_id] = child
                    goal.child_ids.append(child.goal_id)
                    if child_level != AbstractionLevel.OPERATIONAL:
                        queue.append(child)

            depth += 1

        return hierarchy

    def _generate_child_goals(
        self,
        parent: Goal,
        level: AbstractionLevel,
        context: dict[str, Any] | None,
    ) -> list[Goal]:
        """Generate child goals for a parent."""
        children: list[Goal] = []

        # Generate 2-3 child goals based on parent
        num_children = 2 if level == AbstractionLevel.OPERATIONAL else 3

        for i in range(num_children):
            child_desc = f"{parent.description} - part {i + 1}"
            child = self.create_goal(
                description=child_desc,
                level=level,
                parent_id=parent.goal_id,
                priority=parent.priority * (1 - i * 0.1),
            )
            children.append(child)

        return children

    def _estimate_plan_reward(
        self,
        hierarchy: dict[str, Goal],
        subgoals: list[Subgoal],
        options: list[Option],
        state: dict[str, Any],
    ) -> float:
        """Estimate total plan reward."""
        # Sum of goal rewards weighted by completion probability
        goal_reward = sum(g.reward * g.priority for g in hierarchy.values())

        # Sum of option expected rewards
        option_reward = sum(o.expected_reward for o in options)

        # Value function estimate
        root_goal = next(iter(hierarchy.values()))
        value_estimate = self.value_function.get_value(state, root_goal)

        return goal_reward * 0.3 + option_reward * 0.3 + value_estimate * 0.4

    def _check_goal_complete(self, goal: Goal, state: dict[str, Any]) -> bool:
        """Check if goal is complete."""
        # Check postconditions
        for condition in goal.postconditions:
            if condition not in state or not state[condition]:
                return False
        return True

    def _update_stats(self, plan: HierarchicalPlan) -> None:
        """Update planner statistics."""
        n = self._stats["plans_created"]

        depth = len(plan.goal_hierarchy)
        self._stats["avg_plan_depth"] = (self._stats["avg_plan_depth"] * (n - 1) + depth) / n

        self._stats["avg_reward"] = (
            self._stats["avg_reward"] * (n - 1) + plan.estimated_reward
        ) / n

    def get_statistics(self) -> dict[str, Any]:
        """Get planner statistics."""
        return {
            **self._stats,
            "planner_type": self.planner_type.value,
            "max_depth": self.max_depth,
            "option_library": self.option_library.get_statistics(),
        }


# =============================================================================
# Anomaly Detection Integration
# =============================================================================


class AnomalyHierarchicalPlanner:
    """
    Hierarchical planner specialized for anomaly detection.

    Provides domain-specific planning for Mercury Agent's anomaly detection and response tasks.
    """

    def __init__(
        self,
        planner: HierarchicalPlanner | None = None,
    ):
        """
        Initialize anomaly hierarchical planner.

        Args:
            planner: Base hierarchical planner
        """
        self.planner = planner or HierarchicalPlanner()

        # Domain-specific goal templates
        self._goal_templates = {
            "full_detection": {
                "description": "Complete anomaly detection cycle",
                "subgoals": ["data_collection", "preprocessing", "detection", "reporting"],
            },
            "quick_scan": {
                "description": "Quick anomaly scan",
                "subgoals": ["fast_detection", "basic_reporting"],
            },
            "deep_investigation": {
                "description": "Deep investigation of detected anomaly",
                "subgoals": ["evidence_gathering", "correlation", "root_cause", "documentation"],
            },
            "response_action": {
                "description": "Respond to confirmed anomaly",
                "subgoals": ["assess_impact", "plan_response", "execute_response", "verify"],
            },
        }

    def plan_detection_cycle(
        self,
        data_source: str,
        urgency: str = "normal",
        context: dict[str, Any] | None = None,
    ) -> HierarchicalPlan:
        """
        Plan a complete detection cycle.

        Args:
            data_source: Source of data to analyze
            urgency: Urgency level (low, normal, high, critical)
            context: Additional context

        Returns:
            Hierarchical plan for detection
        """
        # Select template based on urgency
        if urgency == "critical":
            template = self._goal_templates["quick_scan"]
        elif urgency == "low":
            template = self._goal_templates["deep_investigation"]
        else:
            template = self._goal_templates["full_detection"]

        # Create root goal
        root_goal = self.planner.create_goal(
            description=str(template["description"]),
            level=AbstractionLevel.STRATEGIC,
            priority={"low": 0.3, "normal": 0.5, "high": 0.7, "critical": 0.9}.get(urgency, 0.5),
            metadata={"data_source": data_source, "urgency": urgency},
        )

        # Create plan
        plan_context = context or {}
        plan_context["data_source"] = data_source
        plan_context["urgency"] = urgency

        return self.planner.plan(root_goal, {}, plan_context)

    def plan_response(
        self,
        anomaly_or_type: str | dict[str, Any],
        severity: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> HierarchicalPlan | dict[str, Any]:
        """
        Plan response to detected anomaly.

        Args:
            anomaly_or_type: Type of anomaly detected (str) or anomaly dict
            severity: Severity score (0-1), optional if anomaly dict provided
            context: Additional context

        Returns:
            Hierarchical plan for response (HierarchicalPlan or dict)
        """
        # Handle dict input (test API)
        if isinstance(anomaly_or_type, dict):
            anomaly = anomaly_or_type
            anomaly_type = anomaly.get("type", "unknown")
            severity = anomaly.get("severity", 0.5)
            affected_systems = anomaly.get("affected_systems", [])

            # Create response plan as dict
            strategic_goals = [
                f"assess_{anomaly_type}_impact",
                f"contain_{anomaly_type}_threat",
                "remediate_affected_systems",
            ]

            tactical_actions = []
            for system in affected_systems:
                tactical_actions.append(
                    {
                        "target": system,
                        "action": "isolate" if severity > 0.7 else "monitor",
                        "priority": severity,
                    }
                )

            return {
                "strategic_goals": strategic_goals,
                "tactical_actions": tactical_actions,
                "severity": severity,
                "estimated_duration": len(affected_systems) * 10.0,
            }

        # Original string API
        anomaly_type = anomaly_or_type
        if severity is None:
            severity = 0.5

        root_goal = self.planner.create_goal(
            description=f"Respond to {anomaly_type} anomaly",
            level=AbstractionLevel.STRATEGIC,
            priority=severity,
            metadata={"anomaly_type": anomaly_type, "severity": severity},
        )

        plan_context = context or {}
        plan_context["anomaly_type"] = anomaly_type
        plan_context["severity"] = severity

        return self.planner.plan(root_goal, {}, plan_context)

    def get_current_action(
        self,
        system_state: dict[str, Any],
        execution_state: PlanExecutionState,
    ) -> dict[str, Any]:
        """
        Get current recommended action.

        Args:
            system_state: Current system state
            execution_state: Plan execution state

        Returns:
            Action recommendation
        """
        action, option = self.planner.select_action(system_state, execution_state)

        return {
            "action": action,
            "option_name": option.name if option else None,
            "expected_duration": option.expected_duration if option else 1.0,
            "current_goal": (
                execution_state.current_goal.description if execution_state.current_goal else None
            ),
            "progress": execution_state.reward_accumulated,
        }
