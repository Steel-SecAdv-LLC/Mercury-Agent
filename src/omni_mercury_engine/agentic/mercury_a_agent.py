# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mercury A. Autonomous Agent Framework.

Implements an autonomous agent with planning, execution, reflection layers,
and comprehensive memory systems. Designed for multi-agent orchestration
with ethical constraints and domain-specific task planning.

Key Components:
- MercuryAgent: Main orchestrator with planning, execution, reasoning, learning
- MercuryPlanner: Goal decomposition and task orchestration
- MercuryReasoner: Chain-of-thought reasoning with correlation graph building
- AgentMemory: Short-term, long-term, episodic, semantic memory systems

References:
    - ReAct: Yao et al. (2022) "ReAct: Synergizing Reasoning and Acting"
    - Memory systems: Tulving (1972) episodic/semantic distinction
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.cognitive.ethical_bounding import (
    MINIMUM_BENEVOLENCE_FLOOR,
    BenevolenceScorer,
    sanitize_domain,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from omni_mercury_engine.agentic.bayesian_calibrator import BayesianConfidenceCalibrator
    from omni_mercury_engine.agentic.subagents.base import SubAgentResult, SubAgentTask
    from omni_mercury_engine.agentic.subagents.fleet import FleetResult, SubAgentFleet


class AgentMode(Enum):
    """Operational modes for Mercury Agent."""

    DORMANT = "dormant"
    REASONING = "reasoning"
    EXECUTING = "executing"
    LEARNING = "learning"
    REFLECTING = "reflecting"
    VOICE = "voice"


class TaskPriority(Enum):
    """Task priority levels."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class DomainType(Enum):
    """Domain types for specialized planning."""

    GENERAL = "general"
    MEDICAL = "medical"
    SECURITY = "security"
    ENERGY = "energy"
    INFRASTRUCTURE = "infrastructure"
    HUMANITARIAN = "humanitarian"
    SCIENTIFIC = "scientific"
    FINANCIAL = "financial"


@dataclass
class MemoryEntry:
    """Entry in agent memory system."""

    entry_id: str
    content: Any
    timestamp: float
    memory_type: str
    importance: float = 0.5
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """Represents a task for the agent to execute."""

    task_id: str
    description: str
    domain: DomainType
    priority: TaskPriority
    dependencies: list[str] = field(default_factory=list)
    status: str = "pending"
    result: Any | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningStep:
    """A step in the reasoning chain."""

    step_id: str
    thought: str
    action: str | None = None
    observation: str | None = None
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)


@dataclass
class PlanResult:
    """Result of planning operation."""

    plan_id: str
    tasks: list[Task]
    estimated_duration: float
    confidence: float
    domain: DomainType
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentMemory:
    """Comprehensive memory system for Mercury Agent.

    Implements four memory types:
    - Short-term: Recent observations and context (limited capacity)
    - Long-term: Persistent knowledge and learned patterns
    - Episodic: Specific experiences and events
    - Semantic: General knowledge and facts
    """

    def __init__(
        self,
        short_term_capacity: int = 100,
        long_term_capacity: int = 10000,
    ):
        """Initialize the instance."""
        self.short_term_capacity = short_term_capacity
        self.long_term_capacity = long_term_capacity

        self.short_term: deque[MemoryEntry] = deque[Any](maxlen=short_term_capacity)
        self.long_term: dict[str, MemoryEntry] = {}
        self.episodic: dict[str, MemoryEntry] = {}
        self.semantic: dict[str, MemoryEntry] = {}

        self.logger = logging.getLogger(__name__)

    def store_short_term(
        self, content: Any, importance: float = 0.5, metadata: dict[str, Any] | None = None
    ) -> str:
        """Store content in short-term memory."""
        entry_id = f"st_{uuid.uuid4().hex[:8]}"
        entry = MemoryEntry(
            entry_id=entry_id,
            content=content,
            timestamp=time.time(),
            memory_type="short_term",
            importance=importance,
            metadata=metadata or {},
        )
        self.short_term.append(entry)

        if importance > 0.7:
            self._consolidate_to_long_term(entry)

        return entry_id

    def store_long_term(
        self, content: Any, importance: float = 0.5, metadata: dict[str, Any] | None = None
    ) -> str:
        """Store content in long-term memory."""
        entry_id = f"lt_{uuid.uuid4().hex[:8]}"
        entry = MemoryEntry(
            entry_id=entry_id,
            content=content,
            timestamp=time.time(),
            memory_type="long_term",
            importance=importance,
            metadata=metadata or {},
        )

        if len(self.long_term) >= self.long_term_capacity:
            self._prune_long_term()

        self.long_term[entry_id] = entry
        return entry_id

    def store_episodic(
        self,
        event: str,
        context: dict[str, Any],
        outcome: str | None = None,
        importance: float = 0.5,
    ) -> str:
        """Store an episodic memory (specific experience)."""
        entry_id = f"ep_{uuid.uuid4().hex[:8]}"
        content = {
            "event": event,
            "context": context,
            "outcome": outcome,
        }
        entry = MemoryEntry(
            entry_id=entry_id,
            content=content,
            timestamp=time.time(),
            memory_type="episodic",
            importance=importance,
        )
        self.episodic[entry_id] = entry
        return entry_id

    def store_semantic(self, fact: str, category: str, confidence: float = 0.8) -> str:
        """Store a semantic memory (general knowledge)."""
        entry_id = f"sm_{uuid.uuid4().hex[:8]}"
        content = {
            "fact": fact,
            "category": category,
            "confidence": confidence,
        }
        entry = MemoryEntry(
            entry_id=entry_id,
            content=content,
            timestamp=time.time(),
            memory_type="semantic",
            importance=confidence,
        )
        self.semantic[entry_id] = entry
        return entry_id

    def retrieve_recent(self, n: int = 10) -> list[MemoryEntry]:
        """Retrieve n most recent short-term memories."""
        return list(self.short_term)[-n:]

    def retrieve_by_importance(
        self, threshold: float = 0.7, memory_type: str = "all"
    ) -> list[MemoryEntry]:
        """Retrieve memories above importance threshold."""
        results = []

        if memory_type in ("all", "short_term"):
            results.extend([m for m in self.short_term if m.importance >= threshold])

        if memory_type in ("all", "long_term"):
            results.extend([m for m in self.long_term.values() if m.importance >= threshold])

        if memory_type in ("all", "episodic"):
            results.extend([m for m in self.episodic.values() if m.importance >= threshold])

        if memory_type in ("all", "semantic"):
            results.extend([m for m in self.semantic.values() if m.importance >= threshold])

        return sorted(results, key=lambda x: x.importance, reverse=True)

    def search_semantic(self, query: str) -> list[MemoryEntry]:
        """Search semantic memories by keyword."""
        results = []
        query_lower = query.lower()

        for entry in self.semantic.values():
            content = entry.content
            if isinstance(content, dict):
                fact = content.get("fact", "").lower()
                category = content.get("category", "").lower()
                if query_lower in fact or query_lower in category:
                    entry.access_count += 1
                    results.append(entry)

        return results

    def _consolidate_to_long_term(self, entry: MemoryEntry) -> None:
        """Consolidate important short-term memory to long-term."""
        lt_entry = MemoryEntry(
            entry_id=f"lt_{entry.entry_id}",
            content=entry.content,
            timestamp=entry.timestamp,
            memory_type="long_term",
            importance=entry.importance,
            metadata=entry.metadata,
        )
        self.long_term[lt_entry.entry_id] = lt_entry

    def _prune_long_term(self) -> None:
        """Prune least important long-term memories."""
        if not self.long_term:
            return

        sorted_entries = sorted(
            self.long_term.items(),
            key=lambda x: (x[1].importance, x[1].access_count),
        )

        prune_count = len(self.long_term) // 10
        for entry_id, _ in sorted_entries[:prune_count]:
            del self.long_term[entry_id]

    def get_statistics(self) -> dict[str, Any]:
        """Get memory system statistics."""
        return {
            "short_term_count": len(self.short_term),
            "short_term_capacity": self.short_term_capacity,
            "long_term_count": len(self.long_term),
            "long_term_capacity": self.long_term_capacity,
            "episodic_count": len(self.episodic),
            "semantic_count": len(self.semantic),
            "total_memories": (
                len(self.short_term) + len(self.long_term) + len(self.episodic) + len(self.semantic)
            ),
        }


class MercuryReasoner:
    """Chain-of-thought reasoning engine with correlation graph building.

    Implements ReAct-style reasoning: Thought → Action → Observation loop.
    """

    def __init__(self, max_steps: int = 15) -> None:
        """Initialize the instance."""
        self.max_steps = max_steps
        self.reasoning_chain: list[ReasoningStep] = []
        self.correlation_graph: dict[str, list[str]] = {}
        self.logger = logging.getLogger(__name__)

    def reason(
        self,
        query: str,
        context: dict[str, Any],
        tools: dict[str, Callable[..., Any]] | None = None,
    ) -> dict[str, Any]:
        """Perform chain-of-thought reasoning on a query.

        Args:
            query: The query to reason about
            context: Context information
            tools: Available tools for action execution

        Returns:
            Reasoning result with conclusion and trace
        """
        self.reasoning_chain = []
        tools = tools or {}

        for step_num in range(self.max_steps):
            thought = self._generate_thought(query, context, step_num)

            action, action_input = self._decide_action(thought, tools)

            if action == "conclude":
                return self._conclude(thought)

            observation = self._execute_action(action, action_input, tools)

            step = ReasoningStep(
                step_id=f"step_{step_num}",
                thought=thought,
                action=action,
                observation=observation,
                confidence=self._estimate_confidence(step_num),
            )
            self.reasoning_chain.append(step)

            self._update_correlation_graph(thought, observation)

            context["previous_observations"] = context.get("previous_observations", [])
            context["previous_observations"].append(observation)

        return self._conclude("Max reasoning steps reached")

    def _generate_thought(self, query: str, context: dict[str, Any], step_num: int) -> str:
        """Generate a thought based on query and context."""
        if step_num == 0:
            return f"Analyzing query: {query}"

        prev_obs = context.get("previous_observations", [])
        if prev_obs:
            return f"Based on observation '{prev_obs[-1]}', considering next step"

        return f"Step {step_num}: Continuing analysis of {query}"

    def _decide_action(
        self, thought: str, tools: dict[str, Callable[..., Any]]
    ) -> tuple[str, str | None]:
        """Decide what action to take based on thought."""
        if "conclude" in thought.lower() or "final" in thought.lower():
            return "conclude", None

        if tools:
            tool_name = next(iter(tools.keys()))
            return tool_name, thought

        return "observe", thought

    def _execute_action(
        self,
        action: str,
        action_input: str | None,
        tools: dict[str, Callable[..., Any]],
    ) -> str:
        """Execute an action and return observation."""
        if action in tools:
            try:
                result = tools[action](action_input)
                return str(result)
            except Exception as e:
                return f"Error executing {action}: {e}"

        return f"Observed: {action_input}"

    def _estimate_confidence(self, step_num: int) -> float:
        """Estimate confidence based on reasoning progress."""
        base_confidence = 0.9
        decay = 0.05 * step_num
        return max(0.3, base_confidence - decay)

    def _update_correlation_graph(self, thought: str, observation: str) -> None:
        """Update correlation graph with new thought-observation pair."""
        thought_key = thought[:50]
        if thought_key not in self.correlation_graph:
            self.correlation_graph[thought_key] = []
        self.correlation_graph[thought_key].append(observation[:100])

    def _conclude(self, final_thought: str) -> dict[str, Any]:
        """Generate conclusion from reasoning chain."""
        avg_confidence = (
            np.mean([s.confidence for s in self.reasoning_chain]) if self.reasoning_chain else 0.5
        )

        return {
            "conclusion": final_thought,
            "confidence": float(avg_confidence),
            "reasoning_steps": len(self.reasoning_chain),
            "reasoning_trace": [
                {
                    "step_id": s.step_id,
                    "thought": s.thought,
                    "action": s.action,
                    "observation": s.observation,
                }
                for s in self.reasoning_chain
            ],
            "correlation_graph_size": len(self.correlation_graph),
        }

    def get_reasoning_trace(self) -> list[ReasoningStep]:
        """Get the full reasoning trace."""
        return self.reasoning_chain.copy()


class MercuryPlanner:
    """Goal decomposition and task orchestration with domain-specific planning.

    Supports specialized planning for medical, security, energy, infrastructure, humanitarian,
    scientific, and financial domains.

    Now includes Bayesian confidence calibration that replaces the fixed 0.76 heuristic with a
    learned, continuously improving confidence model.
    """

    def __init__(self, calibrator: BayesianConfidenceCalibrator | None = None) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)
        self.domain_strategies = self._initialize_domain_strategies()
        self.calibrator = calibrator  # Bayesian confidence calibrator

    def _initialize_domain_strategies(self) -> dict[DomainType, dict[str, Any]]:
        """Initialize domain-specific planning strategies."""
        return {
            DomainType.MEDICAL: {
                "priority_boost": 1.5,
                "safety_checks": True,
                "requires_verification": True,
                "max_parallel_tasks": 3,
            },
            DomainType.SECURITY: {
                "priority_boost": 1.3,
                "safety_checks": True,
                "requires_verification": True,
                "max_parallel_tasks": 5,
            },
            DomainType.HUMANITARIAN: {
                "priority_boost": 1.4,
                "safety_checks": True,
                "requires_verification": False,
                "max_parallel_tasks": 10,
            },
            DomainType.INFRASTRUCTURE: {
                "priority_boost": 1.2,
                "safety_checks": True,
                "requires_verification": True,
                "max_parallel_tasks": 4,
            },
            DomainType.GENERAL: {
                "priority_boost": 1.0,
                "safety_checks": False,
                "requires_verification": False,
                "max_parallel_tasks": 8,
            },
            DomainType.SCIENTIFIC: {
                "priority_boost": 1.1,
                "safety_checks": False,
                "requires_verification": True,
                "max_parallel_tasks": 6,
            },
            DomainType.ENERGY: {
                "priority_boost": 1.2,
                "safety_checks": True,
                "requires_verification": True,
                "max_parallel_tasks": 4,
            },
            DomainType.FINANCIAL: {
                "priority_boost": 1.1,
                "safety_checks": True,
                "requires_verification": True,
                "max_parallel_tasks": 5,
            },
        }

    def plan(
        self,
        goal: str,
        domain: DomainType = DomainType.GENERAL,
        context: dict[str, Any] | None = None,
    ) -> PlanResult:
        """Create a plan to achieve a goal.

        Args:
            goal: The goal to achieve
            domain: Domain type for specialized planning
            context: Optional context information

        Returns:
            PlanResult with decomposed tasks
        """
        context = context or {}
        strategy = self.domain_strategies.get(domain, self.domain_strategies[DomainType.GENERAL])

        tasks = self._decompose_goal(goal, domain, strategy)

        self._establish_dependencies(tasks)

        estimated_duration = self._estimate_duration(tasks, strategy)

        # Use Bayesian calibrator if available, otherwise fall back to heuristic
        if self.calibrator is not None:
            confidence = self.calibrator.get_confidence(domain.value, goal)
            legacy_confidence = self._estimate_plan_confidence(tasks, domain)
        else:
            confidence = self._estimate_plan_confidence(tasks, domain)
            legacy_confidence = confidence

        return PlanResult(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            tasks=tasks,
            estimated_duration=estimated_duration,
            confidence=confidence,
            domain=domain,
            metadata={
                "strategy": strategy,
                "context": context,
                "goal": goal,
                "legacy_confidence_heuristic": legacy_confidence,
            },
        )

    def _decompose_goal(
        self, goal: str, domain: DomainType, strategy: dict[str, Any]
    ) -> list[Task]:
        """Decompose a goal into subtasks."""
        tasks = []
        goal_lower = goal.lower()

        if "analyze" in goal_lower or "detect" in goal_lower:
            tasks.extend(self._create_analysis_tasks(goal, domain, strategy))
        elif "monitor" in goal_lower or "track" in goal_lower:
            tasks.extend(self._create_monitoring_tasks(goal, domain, strategy))
        elif "respond" in goal_lower or "action" in goal_lower:
            tasks.extend(self._create_response_tasks(goal, domain, strategy))
        else:
            tasks.extend(self._create_generic_tasks(goal, domain, strategy))

        return tasks

    def _create_analysis_tasks(
        self, goal: str, domain: DomainType, strategy: dict[str, Any]
    ) -> list[Task]:
        """Create tasks for analysis goals."""
        priority = TaskPriority.HIGH if strategy["priority_boost"] > 1.2 else TaskPriority.MEDIUM

        tasks = [
            Task(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                description=f"Gather data for: {goal}",
                domain=domain,
                priority=priority,
            ),
            Task(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                description="Preprocess and validate data",
                domain=domain,
                priority=priority,
            ),
            Task(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                description="Run analysis algorithms",
                domain=domain,
                priority=priority,
            ),
            Task(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                description="Generate analysis report",
                domain=domain,
                priority=TaskPriority.MEDIUM,
            ),
        ]

        if strategy.get("requires_verification"):
            tasks.append(
                Task(
                    task_id=f"task_{uuid.uuid4().hex[:8]}",
                    description="Verify analysis results",
                    domain=domain,
                    priority=TaskPriority.HIGH,
                )
            )

        return tasks

    def _create_monitoring_tasks(
        self, goal: str, domain: DomainType, strategy: dict[str, Any]
    ) -> list[Task]:
        """Create tasks for monitoring goals."""
        return [
            Task(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                description=f"Initialize monitoring for: {goal}",
                domain=domain,
                priority=TaskPriority.HIGH,
            ),
            Task(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                description="Configure alert thresholds",
                domain=domain,
                priority=TaskPriority.MEDIUM,
            ),
            Task(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                description="Start continuous monitoring loop",
                domain=domain,
                priority=TaskPriority.HIGH,
            ),
        ]

    def _create_response_tasks(
        self, goal: str, domain: DomainType, strategy: dict[str, Any]
    ) -> list[Task]:
        """Create tasks for response goals."""
        return [
            Task(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                description=f"Assess situation for: {goal}",
                domain=domain,
                priority=TaskPriority.CRITICAL,
            ),
            Task(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                description="Determine response options",
                domain=domain,
                priority=TaskPriority.HIGH,
            ),
            Task(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                description="Execute response actions",
                domain=domain,
                priority=TaskPriority.CRITICAL,
            ),
            Task(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                description="Monitor response effectiveness",
                domain=domain,
                priority=TaskPriority.HIGH,
            ),
        ]

    def _create_generic_tasks(
        self, goal: str, domain: DomainType, strategy: dict[str, Any]
    ) -> list[Task]:
        """Create generic tasks for unclassified goals."""
        return [
            Task(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                description=f"Initialize: {goal}",
                domain=domain,
                priority=TaskPriority.MEDIUM,
            ),
            Task(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                description="Execute main operation",
                domain=domain,
                priority=TaskPriority.MEDIUM,
            ),
            Task(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                description="Finalize and report",
                domain=domain,
                priority=TaskPriority.LOW,
            ),
        ]

    def _establish_dependencies(self, tasks: list[Task]) -> None:
        """Establish dependencies between tasks."""
        for i in range(1, len(tasks)):
            tasks[i].dependencies.append(tasks[i - 1].task_id)

    def _estimate_duration(self, tasks: list[Task], strategy: dict[str, Any]) -> float:
        """Estimate total duration for plan execution."""
        base_duration_per_task = 60.0
        total = sum(base_duration_per_task * task.priority.value for task in tasks)

        max_parallel = strategy.get("max_parallel_tasks", 1)
        if max_parallel > 1:
            total = total / min(max_parallel, len(tasks))

        return total

    def _estimate_plan_confidence(self, tasks: list[Task], domain: DomainType) -> float:
        """Estimate confidence in plan success."""
        base_confidence = 0.85

        task_penalty = 0.02 * len(tasks)

        domain_bonus = {
            DomainType.GENERAL: 0.0,
            DomainType.MEDICAL: -0.05,
            DomainType.SECURITY: -0.03,
            DomainType.HUMANITARIAN: 0.02,
        }.get(domain, 0.0)

        return max(0.5, min(1.0, base_confidence - task_penalty + domain_bonus))


class MercuryAgent:
    """Mercury A.

    Autonomous Agent - Main Orchestrator.
        Combines planning, execution, reasoning, and learning capabilities
        with comprehensive memory systems and ethical constraints.

        Features:
        - Multi-mode operation (dormant, reasoning, executing, learning, reflecting)
        - Domain-specific task planning
        - Chain-of-thought reasoning
        - Four-tier memory system
        - Ethical constraint enforcement
        - Bayesian confidence calibration (replaces fixed 0.76 heuristic)
    """

    def __init__(
        self,
        name: str = "Mercury",
        autonomy_level: float = 0.8,
        ethical_threshold: float = 0.93,
        enable_calibration: bool = True,
    ):
        """Initialize Mercury Agent.

        Args:
            name: Agent name
            autonomy_level: Level of autonomous operation (0-1)
            ethical_threshold: Minimum ethical score for operations
            enable_calibration: Enable Bayesian confidence calibration
        """
        from omni_mercury_engine.agentic.bayesian_calibrator import (
            BayesianConfidenceCalibrator,
        )

        self.name = name
        self.autonomy_level = autonomy_level
        self.ethical_threshold = ethical_threshold

        self.mode = AgentMode.DORMANT
        self.memory = AgentMemory()

        # Initialize Bayesian confidence calibrator
        self.confidence_calibrator: BayesianConfidenceCalibrator | None = (
            BayesianConfidenceCalibrator() if enable_calibration else None
        )

        # Pass calibrator to planner for confidence estimation
        self.planner = MercuryPlanner(calibrator=self.confidence_calibrator)
        self.reasoner = MercuryReasoner()

        self.current_plan: PlanResult | None = None
        self.execution_history: list[dict[str, Any]] = []
        self.tools: dict[str, Callable[..., Any]] = {}

        # Internal subagent fleet (lazily enabled). The main agent delegates
        # tasks to specialized full-capability subagents through this; it is
        # internal-only and never exposed on the public package surface.
        self.fleet: SubAgentFleet | None = None

        # General-purpose capability layer (web research + document generation),
        # lazily enabled via enable_assistant(). Native, no new deps, no LLM.
        self.assistant: Any | None = None

        # Fail-closed ethical gate on the execution path (mirrors the OODA
        # reference's pre-act ethical check).  Constructed eagerly; the floor
        # matches the engine/orchestrator boundary contract.
        self._benevolence_scorer = BenevolenceScorer(
            benevolence_threshold=MINIMUM_BENEVOLENCE_FLOOR
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Mercury Agent '{name}' initialized (calibration={'enabled' if enable_calibration else 'disabled'})"
        )

    def register_tool(self, name: str, tool: Callable[..., Any]) -> None:
        """Register a tool for agent use."""
        self.tools[name] = tool
        self.logger.debug(f"Registered tool: {name}")

    # ------------------------------------------------------------------
    # Subagent delegation
    # ------------------------------------------------------------------

    def enable_fleet(
        self,
        *,
        engine: Any | None = None,
        seed: int | None = None,
    ) -> SubAgentFleet:
        """Enable (idempotently) the internal subagent fleet for delegation.

        The fleet is constructed with the package-private access sentinel —
        users cannot address subagents directly; only the main agent (or the
        engine) calls on them. Repeated calls return the existing fleet.

        Args:
            engine: Optional :class:`OmniMercuryEngine` the detection
                specialization uses for Mercury's own real detection.
            seed: Base seed for deterministic subagent construction.

        Returns:
            The enabled :class:`SubAgentFleet`.
        """
        if self.fleet is None:
            from omni_mercury_engine.agentic.subagents.base import _INTERNAL
            from omni_mercury_engine.agentic.subagents.fleet import SubAgentFleet

            self.fleet = SubAgentFleet(access=_INTERNAL, seed=seed, engine=engine)
            self.logger.info("Subagent fleet enabled for %s", self.name)
        return self.fleet

    def delegate(
        self,
        task: SubAgentTask,
        specialty: str | None = None,
    ) -> SubAgentResult:
        """Delegate one task to the most competent subagent (lazy-enabling the fleet).

        Args:
            task: The task to delegate.
            specialty: Force a specialty; defaults to capability-based routing.

        Returns:
            The committed (dual-gated) subagent result.
        """
        return self.enable_fleet().dispatch(task, specialty)

    def delegate_masses(
        self,
        task: SubAgentTask,
        replicas: int,
        specialty: str | None = None,
    ) -> FleetResult:
        """Delegate one task to ``replicas`` subagents concurrently ("in the masses").

        Args:
            task: The task to fan out.
            replicas: Number of concurrent full-capability subagents.
            specialty: Force a specialty; defaults to capability-based routing.

        Returns:
            The fleet result with per-replica outcomes and an honest aggregate.
        """
        return self.enable_fleet().scale_dispatch(task, replicas, specialty)

    # ------------------------------------------------------------------
    # General-purpose capabilities (web research + document generation)
    # ------------------------------------------------------------------

    def enable_assistant(self, *, researcher: Any | None = None) -> Any:
        """Enable (idempotently) the general-purpose assistant capability layer.

        Gives Mercury general usefulness beyond anomaly detection: open-web
        research, source synthesis, and document generation -- native (stdlib +
        numpy, no new dependencies, no language model) and governed by this
        agent's own fail-closed benevolence gate. Returns the
        :class:`GeneralAssistant`.
        """
        if getattr(self, "assistant", None) is None:
            from omni_mercury_engine.agentic.capabilities import GeneralAssistant

            self.assistant = GeneralAssistant(
                researcher=researcher, benevolence_scorer=self._benevolence_scorer
            )
            self.logger.info("General assistant capability enabled for %s", self.name)
        return self.assistant

    def research(self, query: str, *, max_sources: int = 5, fmt: str = "markdown") -> Any:
        """Research a question on the open web and return a cited report.

        Fail-closed and honest: an ethics-refused query, an unreachable network,
        or zero readable sources each yield a report flagged accordingly rather
        than a fabricated answer. Returns a
        :class:`~omni_mercury_engine.agentic.capabilities.assistant.ResearchReport`.
        """
        return self.enable_assistant().research_report(query, max_sources=max_sources, fmt=fmt)

    def answer(self, question: str, *, max_sources: int = 3) -> str:
        """Answer a question with sentences extracted (verbatim) from web sources."""
        result: str = self.enable_assistant().answer(question, max_sources=max_sources)
        return result

    def write_document(
        self,
        title: str,
        sections: Any,
        *,
        fmt: str = "markdown",
        metadata: dict[str, str] | None = None,
        sources: list[str] | None = None,
    ) -> Any:
        """Generate a Markdown/HTML/text document, gated by the benevolence check.

        Returns the rendered ``Document`` or ``None`` if the content is refused.
        """
        return self.enable_assistant().write_document(
            title, sections, fmt=fmt, metadata=metadata, sources=sources
        )

    def analyze(
        self,
        data: Any,
        domain: DomainType = DomainType.GENERAL,
        goal: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Analyze data with autonomous planning and reasoning.

        Args:
            data: Data to analyze
            domain: Domain type for specialized handling
            goal: Optional specific goal
            context: Optional context information

        Returns:
            Analysis results with reasoning trace
        """
        self.mode = AgentMode.REASONING
        context = context or {}
        context["data"] = data

        goal = goal or f"Analyze {domain.value} data"
        self.current_plan = self.planner.plan(goal, domain, context)

        self.memory.store_short_term(
            {"action": "plan_created", "goal": goal, "domain": domain.value},
            importance=0.6,
        )

        reasoning_result = self.reasoner.reason(goal, context, self.tools)

        self.mode = AgentMode.EXECUTING
        execution_results = self._execute_plan(self.current_plan, context)

        self.mode = AgentMode.LEARNING
        self._learn_from_execution(execution_results)

        self.mode = AgentMode.DORMANT

        result = {
            "agent": self.name,
            "goal": goal,
            "domain": domain.value,
            "plan_confidence": self.current_plan.confidence,
            "reasoning": reasoning_result,
            "execution": execution_results,
            "memory_stats": self.memory.get_statistics(),
        }

        self.execution_history.append(result)
        return result

    def explain_reasoning(self) -> dict[str, Any]:
        """Explain the agent's reasoning process.

        Returns:
            Explanation of reasoning with trace
        """
        trace = self.reasoner.get_reasoning_trace()

        return {
            "agent": self.name,
            "reasoning_steps": len(trace),
            "trace": [
                {
                    "step": s.step_id,
                    "thought": s.thought,
                    "action": s.action,
                    "observation": s.observation,
                    "confidence": s.confidence,
                }
                for s in trace
            ],
            "correlation_graph_size": len(self.reasoner.correlation_graph),
        }

    def get_state(self) -> dict[str, Any]:
        """Get current agent state.

        Returns:
            Current state information
        """
        state = {
            "name": self.name,
            "mode": self.mode.value,
            "autonomy_level": self.autonomy_level,
            "ethical_threshold": self.ethical_threshold,
            "registered_tools": list(self.tools.keys()),
            "current_plan": (
                {
                    "plan_id": self.current_plan.plan_id,
                    "task_count": len(self.current_plan.tasks),
                    "confidence": self.current_plan.confidence,
                }
                if self.current_plan
                else None
            ),
            "memory": self.memory.get_statistics(),
            "execution_history_count": len(self.execution_history),
        }

        # Include calibrator statistics if available
        if self.confidence_calibrator is not None:
            state["confidence_calibrator"] = self.confidence_calibrator.get_summary()

        return state

    def _execute_plan(self, plan: PlanResult, context: dict[str, Any]) -> dict[str, Any]:
        """Execute a plan's tasks.

        ``success_rate`` is measured over *executed* tasks (completed +
        failed); honestly-skipped tasks (no tool bound) do not inflate it, so
        a plan of pure reasoning tasks no longer reports a fabricated 1.0.
        """
        tasks_completed: int = 0
        tasks_failed: int = 0
        tasks_skipped: int = 0
        task_results: list[dict[str, Any]] = []

        for task in plan.tasks:
            if not self._check_dependencies(task, task_results):
                continue

            task_result = self._execute_task(task, context)
            task_results.append(task_result)

            status = task_result["status"]
            if status == "completed":
                tasks_completed += 1
            elif status == "skipped":
                tasks_skipped += 1
            else:
                tasks_failed += 1

        executed = tasks_completed + tasks_failed
        success_rate = tasks_completed / executed if executed else 0.0

        return {
            "plan_id": plan.plan_id,
            "tasks_completed": tasks_completed,
            "tasks_failed": tasks_failed,
            "tasks_skipped": tasks_skipped,
            "task_results": task_results,
            "success_rate": success_rate,
        }

    def _enforce_task_ethics(self, task: Task) -> None:
        """Fail-closed ethical gate on a task before any tool side-effect.

        Mirrors the OODA reference (`cognitive/autonomous_agent.py`), which
        scores the decision and refuses to act below its ethical threshold.
        Here the *task itself* is scored (its description rides into the
        action text alongside the agent's defensive-purpose keywords), so a
        harmful goal that propagated into a task description fails closed —
        the violation propagates as :class:`EthicalConstraintViolationError`
        and halts the plan rather than being recorded as a benign result.
        Domain hints are collapsed by ``sanitize_domain`` first.
        """
        safe_domain = sanitize_domain(getattr(task.domain, "value", str(task.domain)))
        action = (
            f"agentic_task:{safe_domain}:{task.description} audit verify protect "
            "research evidence fair oversight monitor data care help support"
        )
        context = {
            "purpose": "autonomous anomaly-analysis task execution",
            "safety": "protect verify monitor evidence",
            "domain": safe_domain,
        }
        # ``enforce`` raises EthicalConstraintViolationError on violation; it is
        # intentionally NOT wrapped in the execution try/except below.
        self._benevolence_scorer.enforce(action, context)

    def _execute_task(self, task: Task, context: dict[str, Any]) -> dict[str, Any]:
        """Execute a single task by dispatching to a bound tool.

        Real execution with genuine success/failure outcomes (replacing the
        prior no-op that always reported ``completed``):

        * The fail-closed ethical gate runs first and is not swallowed.
        * A task that binds a registered tool via ``task.metadata['tool']``
          is executed for real with ``task.metadata.get('tool_args', {})``;
          a raising tool yields ``status="failed"`` with the error captured.
        * An unregistered tool yields ``status="failed"``.
        * A task with no bound tool yields an honest ``status="skipped"`` —
          never a fabricated ``completed``.

        ``context['data']`` is injected as ``data=`` for tools that accept it,
        so detection-style tools can run on the analysed batch.
        """
        task.status = "executing"

        # Fail-closed ethical gate (propagates; not inside the try/except).
        self._enforce_task_ethics(task)

        tool_name = task.metadata.get("tool")
        if tool_name is None:
            task.status = "skipped"
            task.completed_at = time.time()
            return {
                "task_id": task.task_id,
                "description": task.description,
                "status": "skipped",
                "reason": "no tool bound (task.metadata['tool'] unset)",
            }
        if tool_name not in self.tools:
            task.status = "failed"
            task.completed_at = time.time()
            return {
                "task_id": task.task_id,
                "description": task.description,
                "status": "failed",
                "tool": tool_name,
                "error": f"tool '{tool_name}' is not registered",
            }

        tool = self.tools[tool_name]
        tool_args = dict(task.metadata.get("tool_args", {}))
        # Offer the analysed batch to tools that declare a ``data`` parameter.
        if "data" not in tool_args and "data" in context:
            try:
                import inspect

                if "data" in inspect.signature(tool).parameters:
                    tool_args["data"] = context["data"]
            except (TypeError, ValueError):
                pass

        try:
            output = tool(**tool_args)
            task.result = output
            task.status = "completed"
            task.completed_at = time.time()
            return {
                "task_id": task.task_id,
                "description": task.description,
                "status": "completed",
                "tool": tool_name,
                "output": output,
            }
        except Exception as e:  # genuine tool execution failure
            task.status = "failed"
            task.completed_at = time.time()
            return {
                "task_id": task.task_id,
                "description": task.description,
                "status": "failed",
                "tool": tool_name,
                "error": str(e),
            }

    def _check_dependencies(self, task: Task, completed_results: list[dict[str, Any]]) -> bool:
        """Check if task dependencies are satisfied."""
        completed_ids = {r["task_id"] for r in completed_results if r["status"] == "completed"}
        return all(dep in completed_ids for dep in task.dependencies)

    def _learn_from_execution(self, execution_results: dict[str, Any]) -> None:
        """Learn from execution results and update confidence calibrator."""
        success_rate = execution_results.get("success_rate", 0.0)

        # Update Bayesian confidence calibrator with execution outcome
        if self.confidence_calibrator is not None and self.current_plan is not None:
            goal = self.current_plan.metadata.get("goal", "")
            domain = self.current_plan.domain.value
            success = success_rate >= 0.99  # Consider 99%+ as success
            self.confidence_calibrator.update(domain, goal, success, time.time())

        # Store episodic memory with domain/goal for future retrieval
        self.memory.store_episodic(
            event="plan_execution",
            context={
                "plan_id": execution_results.get("plan_id"),
                "domain": self.current_plan.domain.value if self.current_plan else "unknown",
                "goal": self.current_plan.metadata.get("goal", "") if self.current_plan else "",
            },
            outcome=f"success_rate={success_rate:.2f}",
            importance=0.7 if success_rate > 0.8 else 0.5,
        )

        if success_rate > 0.9:
            self.memory.store_semantic(
                fact="High success rate achieved with current strategy",
                category="learning",
                confidence=success_rate,
            )


def create_mercury_agent(
    name: str = "Mercury",
    autonomy_level: float = 0.8,
    ethical_threshold: float = 0.93,
) -> MercuryAgent:
    """Factory function to create a Mercury Agent.

    Args:
        name: Agent name
        autonomy_level: Level of autonomous operation (0-1)
        ethical_threshold: Minimum ethical score for operations

    Returns:
        Configured MercuryAgent instance
    """
    return MercuryAgent(
        name=name,
        autonomy_level=autonomy_level,
        ethical_threshold=ethical_threshold,
    )
