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
Autonomous Agent - OODA Loop, User Synchronization, and Self-Maintenance

Implements Phase 5 of the neuro-symbolic evolution:
- Closed-loop OODA agent (Observe, Orient, Decide, Act, Reflect)
- Bidirectional user synchronization interface
- Self-maintenance diagnostics and repair

Research Sources:
- OODA Loop (Boyd, 1987)
- Autonomous Agents (Russell & Norvig, 2020)
- Self-Healing Systems (Kephart & Chess, 2003)
- Human-AI Collaboration (Amershi et al., 2019)

Integration:
    This module provides the autonomous agent capabilities that
    integrate with the neuro-symbolic fusion engine for intelligent
    decision-making with human oversight.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent operational states."""

    IDLE = "idle"
    OBSERVING = "observing"
    ORIENTING = "orienting"
    DECIDING = "deciding"
    ACTING = "acting"
    REFLECTING = "reflecting"
    PAUSED = "paused"
    AWAITING_APPROVAL = "awaiting_approval"
    ERROR = "error"


class ActionRisk(Enum):
    """Risk levels for actions."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ApprovalStatus(Enum):
    """Status of approval requests."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


@dataclass
class Observation:
    """Data observed by the agent."""

    observation_id: str
    source: str
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.8
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Orientation:
    """Agent's understanding of the situation."""

    orientation_id: str
    patterns: list[dict[str, Any]]
    predictions: list[dict[str, Any]]
    threats: list[dict[str, Any]]
    opportunities: list[dict[str, Any]]
    confidence: float = 0.8
    timestamp: float = field(default_factory=time.time)


@dataclass
class Decision:
    """Decision made by the agent."""

    decision_id: str
    action: str
    risk_level: ActionRisk
    ethical_score: float
    confidence: float
    reasoning: str
    requires_approval: bool
    alternatives: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ActionResult:
    """Result of an executed action."""

    result_id: str
    action: str
    success: bool
    outcome: dict[str, Any]
    side_effects: list[str]
    timestamp: float = field(default_factory=time.time)


@dataclass
class Reflection:
    """Agent's reflection on outcomes."""

    reflection_id: str
    action: str
    outcome_assessment: str
    lessons_learned: list[str]
    rule_updates: list[dict[str, Any]]
    memory_updates: list[dict[str, Any]]
    confidence_adjustment: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class ApprovalRequest:
    """Request for user approval."""

    request_id: str
    decision: Decision
    context: dict[str, Any]
    urgency: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    user_response: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class DiagnosticResult:
    """Result of self-diagnostic."""

    diagnostic_id: str
    component: str
    status: str
    issues: list[str]
    recommendations: list[str]
    timestamp: float = field(default_factory=time.time)


class UserSyncInterface:
    """
    Bidirectional interface for user synchronization.

    Enables real-time user input augmentation and approval workflows.
    """

    def __init__(self, approval_timeout: float = 300.0) -> None:
        """
        Initialize user sync interface.

        Args:
            approval_timeout: Timeout for approval requests in seconds
        """
        self.approval_timeout = approval_timeout
        self.pending_approvals: dict[str, ApprovalRequest] = {}
        self.user_preferences: dict[str, Any] = {}
        self.user_inputs: list[dict[str, Any]] = []
        self._approval_counter = 0
        self._lock = threading.Lock()

        self._callbacks: dict[str, list[Callable]] = {
            "on_approval_request": [],
            "on_user_input": [],
            "on_preference_change": [],
        }

    def request_approval(
        self,
        decision: Decision,
        context: dict[str, Any],
        urgency: str = "normal",
    ) -> ApprovalRequest:
        """
        Request user approval for a decision.

        Args:
            decision: Decision requiring approval
            context: Context for the decision
            urgency: Urgency level (low, normal, high, critical)

        Returns:
            ApprovalRequest object
        """
        with self._lock:
            self._approval_counter += 1
            request_id = f"approval_{self._approval_counter:06d}"

            request = ApprovalRequest(
                request_id=request_id,
                decision=decision,
                context=context,
                urgency=urgency,
            )

            self.pending_approvals[request_id] = request

            for callback in self._callbacks["on_approval_request"]:
                try:
                    callback(request)
                except Exception as e:
                    logger.error(f"Approval callback error: {e}")

            return request

    def provide_approval(
        self,
        request_id: str,
        approved: bool,
        response: str | None = None,
    ) -> bool:
        """
        Provide approval response.

        Args:
            request_id: Request identifier
            approved: Whether approved
            response: Optional response message

        Returns:
            True if request was found and updated
        """
        with self._lock:
            if request_id not in self.pending_approvals:
                return False

            request = self.pending_approvals[request_id]
            request.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
            request.user_response = response

            return True

    def check_approval_status(self, request_id: str) -> ApprovalStatus:
        """Check status of an approval request."""
        with self._lock:
            if request_id not in self.pending_approvals:
                return ApprovalStatus.TIMEOUT

            request = self.pending_approvals[request_id]

            if request.status == ApprovalStatus.PENDING:
                elapsed = time.time() - request.timestamp
                if elapsed > self.approval_timeout:
                    request.status = ApprovalStatus.TIMEOUT

            return request.status

    def add_user_input(self, input_data: dict[str, Any]) -> None:
        """
        Add user input to augment agent memory.

        Args:
            input_data: User input data
        """
        input_entry = {
            "data": input_data,
            "timestamp": time.time(),
            "processed": False,
        }
        self.user_inputs.append(input_entry)

        for callback in self._callbacks["on_user_input"]:
            try:
                callback(input_entry)
            except Exception as e:
                logger.error(f"User input callback error: {e}")

    def get_pending_inputs(self) -> list[dict[str, Any]]:
        """Get unprocessed user inputs."""
        pending = [i for i in self.user_inputs if not i["processed"]]
        return pending

    def mark_input_processed(self, index: int) -> None:
        """Mark a user input as processed."""
        if 0 <= index < len(self.user_inputs):
            self.user_inputs[index]["processed"] = True

    def set_preference(self, key: str, value: Any) -> None:
        """Set a user preference."""
        old_value = self.user_preferences.get(key)
        self.user_preferences[key] = value

        for callback in self._callbacks["on_preference_change"]:
            try:
                callback(key, old_value, value)
            except Exception as e:
                logger.error(f"Preference callback error: {e}")

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference."""
        return self.user_preferences.get(key, default)

    def register_callback(self, event: str, callback: Callable[..., Any]) -> None:
        """Register a callback for an event."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def get_statistics(self) -> dict[str, Any]:
        """Get interface statistics."""
        return {
            "pending_approvals": len(
                [r for r in self.pending_approvals.values() if r.status == ApprovalStatus.PENDING]
            ),
            "total_approvals": len(self.pending_approvals),
            "user_inputs": len(self.user_inputs),
            "preferences_set": len(self.user_preferences),
        }


class SelfMaintenance:
    """
    Self-maintenance and diagnostic routines.

    Implements self-healing capabilities including memory pruning,
    rule repair, and confidence-triggered reflection.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.90,
        memory_limit: int = 10000,
        redundancy_threshold: float = 0.95,
    ):
        """
        Initialize self-maintenance.

        Args:
            confidence_threshold: Threshold below which reflection is triggered
            memory_limit: Maximum memory entries before pruning
            redundancy_threshold: Similarity threshold for redundant memories
        """
        self.confidence_threshold = confidence_threshold
        self.memory_limit = memory_limit
        self.redundancy_threshold = redundancy_threshold

        self._diagnostic_counter = 0
        self.diagnostic_history: list[DiagnosticResult] = []
        self.maintenance_log: list[dict[str, Any]] = []

    def run_diagnostics(
        self,
        components: dict[str, Any],
    ) -> list[DiagnosticResult]:
        """
        Run diagnostics on agent components.

        Args:
            components: Dictionary of component name to component object

        Returns:
            List of diagnostic results
        """
        results = []

        for name, component in components.items():
            self._diagnostic_counter += 1
            diagnostic_id = f"diag_{self._diagnostic_counter:06d}"

            issues = []
            recommendations = []
            status = "healthy"

            if hasattr(component, "get_statistics"):
                try:
                    stats = component.get_statistics()
                    issues, recommendations = self._analyze_stats(name, stats)
                    if issues:
                        status = "degraded" if len(issues) < 3 else "critical"
                except Exception as e:
                    issues.append(f"Failed to get statistics: {e}")
                    status = "error"

            result = DiagnosticResult(
                diagnostic_id=diagnostic_id,
                component=name,
                status=status,
                issues=issues,
                recommendations=recommendations,
            )

            results.append(result)
            self.diagnostic_history.append(result)

        return results

    def _analyze_stats(
        self,
        component_name: str,
        stats: dict[str, Any],
    ) -> tuple[list[str], list[str]]:
        """Analyze component statistics for issues."""
        issues = []
        recommendations = []

        if "error_count" in stats and stats["error_count"] > 10:
            issues.append(f"High error count: {stats['error_count']}")
            recommendations.append("Review error logs and address root causes")

        if "memory_usage" in stats and stats["memory_usage"] > 0.9:
            issues.append(f"High memory usage: {stats['memory_usage']:.0%}")
            recommendations.append("Consider pruning old memories")

        if "confidence" in stats and stats["confidence"] < self.confidence_threshold:
            issues.append(f"Low confidence: {stats['confidence']:.2%}")
            recommendations.append("Trigger reflection cycle")

        return issues, recommendations

    def prune_memories(
        self,
        memories: list[dict[str, Any]],
        importance_key: str = "importance",
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Prune redundant and low-importance memories.

        Args:
            memories: List of memory entries
            importance_key: Key for importance score

        Returns:
            Tuple of (pruned_memories, count_removed)
        """
        if len(memories) <= self.memory_limit:
            return memories, 0

        sorted_memories = sorted(
            memories,
            key=lambda m: m.get(importance_key, 0.5),
            reverse=True,
        )

        pruned = sorted_memories[: self.memory_limit]
        removed = len(memories) - len(pruned)

        self._log_maintenance(
            "memory_prune",
            {
                "original_count": len(memories),
                "pruned_count": len(pruned),
                "removed": removed,
            },
        )

        return pruned, removed

    def detect_redundant_memories(
        self,
        memories: list[dict[str, Any]],
        content_key: str = "content",
    ) -> list[tuple[int, int, float]]:
        """
        Detect redundant memory pairs.

        Args:
            memories: List of memory entries
            content_key: Key for memory content

        Returns:
            List of (index1, index2, similarity) tuples
        """
        redundant_pairs = []

        for i in range(len(memories)):
            for j in range(i + 1, min(i + 100, len(memories))):
                similarity = self._compute_similarity(
                    memories[i].get(content_key, {}),
                    memories[j].get(content_key, {}),
                )
                if similarity >= self.redundancy_threshold:
                    redundant_pairs.append((i, j, similarity))

        return redundant_pairs

    def _compute_similarity(
        self,
        content1: dict[str, Any],
        content2: dict[str, Any],
    ) -> float:
        """Compute similarity between two memory contents."""
        if not content1 or not content2:
            return 0.0

        keys1 = set(content1.keys())
        keys2 = set(content2.keys())

        if not keys1 or not keys2:
            return 0.0

        key_overlap = len(keys1 & keys2) / len(keys1 | keys2)

        value_matches = 0
        common_keys = keys1 & keys2
        for key in common_keys:
            if content1[key] == content2[key]:
                value_matches += 1

        value_similarity = value_matches / len(common_keys) if common_keys else 0

        return 0.5 * key_overlap + 0.5 * value_similarity

    def repair_rule_inconsistencies(
        self,
        rules: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """
        Repair inconsistencies in rules.

        Args:
            rules: List of rule definitions

        Returns:
            Tuple of (repaired_rules, repair_log)
        """
        repaired = []
        repair_log = []

        for rule in rules:
            if "confidence" not in rule:
                rule["confidence"] = 0.5
                repair_log.append(f"Added default confidence to rule {rule.get('id', 'unknown')}")

            if "enabled" not in rule:
                rule["enabled"] = True
                repair_log.append(f"Enabled rule {rule.get('id', 'unknown')}")

            if rule.get("confidence", 0) < 0.1:
                repair_log.append(f"Disabled low-confidence rule {rule.get('id', 'unknown')}")
                rule["enabled"] = False

            repaired.append(rule)

        if repair_log:
            self._log_maintenance(
                "rule_repair",
                {
                    "repairs": len(repair_log),
                    "details": repair_log,
                },
            )

        return repaired, repair_log

    def should_trigger_reflection(self, confidence: float) -> bool:
        """Check if reflection should be triggered based on confidence."""
        return confidence < self.confidence_threshold

    def generate_maintenance_task(
        self,
        issue: str,
        priority: str = "normal",
    ) -> dict[str, Any]:
        """Generate a maintenance task."""
        task = {
            "task_id": f"maint_{int(time.time())}",
            "issue": issue,
            "priority": priority,
            "status": "pending",
            "created_at": time.time(),
        }

        self._log_maintenance("task_generated", task)
        return task

    def _log_maintenance(self, action: str, details: dict[str, Any]) -> None:
        """Log a maintenance action."""
        self.maintenance_log.append(
            {
                "action": action,
                "details": details,
                "timestamp": time.time(),
            }
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get maintenance statistics."""
        return {
            "diagnostics_run": self._diagnostic_counter,
            "maintenance_actions": len(self.maintenance_log),
            "confidence_threshold": self.confidence_threshold,
            "memory_limit": self.memory_limit,
        }


class OODAAgent:
    """
    Autonomous agent implementing the OODA loop.

    Observe -> Orient -> Decide -> Act -> Reflect

    Integrates with neuro-symbolic fusion for intelligent
    decision-making with human oversight.
    """

    def __init__(
        self,
        risk_threshold: ActionRisk = ActionRisk.MEDIUM,
        ethical_threshold: float = 0.99,
        confidence_threshold: float = 0.90,
        approval_timeout: float = 300.0,
    ):
        """
        Initialize OODA agent.

        Args:
            risk_threshold: Maximum risk level for autonomous action
            ethical_threshold: Minimum ethical score for action
            confidence_threshold: Minimum confidence for autonomous action
            approval_timeout: Timeout for approval requests
        """
        self.risk_threshold = risk_threshold
        self.ethical_threshold = ethical_threshold
        self.confidence_threshold = confidence_threshold

        self.state = AgentState.IDLE
        self.user_sync = UserSyncInterface(approval_timeout=approval_timeout)
        self.maintenance = SelfMaintenance(confidence_threshold=confidence_threshold)

        self._observation_counter = 0
        self._orientation_counter = 0
        self._decision_counter = 0
        self._action_counter = 0
        self._reflection_counter = 0

        self.observations: list[Observation] = []
        self.orientations: list[Orientation] = []
        self.decisions: list[Decision] = []
        self.actions: list[ActionResult] = []
        self.reflections: list[Reflection] = []

        self._kill_switch = False
        self._paused = False

        self.audit_log: list[dict[str, Any]] = []

        logger.info("OODAAgent initialized")

    def observe(self, data: dict[str, Any], source: str = "internal") -> Observation:
        """
        Observe phase: Ingest and process data.

        Args:
            data: Data to observe
            source: Source of the data

        Returns:
            Observation object
        """
        if self._kill_switch:
            raise RuntimeError("Agent kill switch activated")

        self._check_paused()
        self.state = AgentState.OBSERVING

        self._observation_counter += 1
        observation_id = f"obs_{self._observation_counter:06d}"

        user_inputs = self.user_sync.get_pending_inputs()
        for i, user_input in enumerate(user_inputs):
            data["user_input"] = user_input["data"]
            self.user_sync.mark_input_processed(i)

        observation = Observation(
            observation_id=observation_id,
            source=source,
            data=data,
            confidence=self._assess_data_quality(data),
        )

        self.observations.append(observation)
        self._audit("observe", {"observation_id": observation_id, "source": source})

        return observation

    def orient(
        self,
        observation: Observation,
        analyzer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> Orientation:
        """
        Orient phase: Analyze patterns and make predictions.

        Args:
            observation: Observation to analyze
            analyzer: Optional analysis function

        Returns:
            Orientation object
        """
        if self._kill_switch:
            raise RuntimeError("Agent kill switch activated")

        self._check_paused()
        self.state = AgentState.ORIENTING

        self._orientation_counter += 1
        orientation_id = f"orient_{self._orientation_counter:06d}"

        patterns = []
        predictions = []
        threats = []
        opportunities = []

        if analyzer:
            try:
                analysis = analyzer(observation.data)
                patterns = analysis.get("patterns", [])
                predictions = analysis.get("predictions", [])
                threats = analysis.get("threats", [])
                opportunities = analysis.get("opportunities", [])
            except Exception as e:
                logger.error(f"Analysis error: {e}")

        orientation = Orientation(
            orientation_id=orientation_id,
            patterns=patterns,
            predictions=predictions,
            threats=threats,
            opportunities=opportunities,
            confidence=observation.confidence,
        )

        self.orientations.append(orientation)
        self._audit(
            "orient",
            {
                "orientation_id": orientation_id,
                "patterns_found": len(patterns),
                "threats_found": len(threats),
            },
        )

        return orientation

    def decide(
        self,
        orientation: Orientation,
        ethical_scorer: Callable[[str, dict[str, Any]], float] | None = None,
    ) -> Decision:
        """
        Decide phase: Determine action with ethical scoring.

        Args:
            orientation: Orientation to base decision on
            ethical_scorer: Optional ethical scoring function

        Returns:
            Decision object
        """
        if self._kill_switch:
            raise RuntimeError("Agent kill switch activated")

        self._check_paused()
        self.state = AgentState.DECIDING

        self._decision_counter += 1
        decision_id = f"dec_{self._decision_counter:06d}"

        action, risk_level = self._determine_action(orientation)

        context = {
            "patterns": orientation.patterns,
            "threats": orientation.threats,
            "opportunities": orientation.opportunities,
        }

        if ethical_scorer:
            ethical_score = ethical_scorer(action, context)
        else:
            ethical_score = self._default_ethical_score(action, context)

        requires_approval = self._requires_approval(
            risk_level, ethical_score, orientation.confidence
        )

        reasoning = self._generate_reasoning(orientation, action, ethical_score)

        decision = Decision(
            decision_id=decision_id,
            action=action,
            risk_level=risk_level,
            ethical_score=ethical_score,
            confidence=orientation.confidence,
            reasoning=reasoning,
            requires_approval=requires_approval,
            alternatives=self._generate_alternatives(orientation),
        )

        self.decisions.append(decision)
        self._audit(
            "decide",
            {
                "decision_id": decision_id,
                "action": action,
                "risk_level": risk_level.value,
                "ethical_score": ethical_score,
                "requires_approval": requires_approval,
            },
        )

        return decision

    def act(
        self,
        decision: Decision,
        executor: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> ActionResult | None:
        """
        Act phase: Execute decision if approved.

        Args:
            decision: Decision to execute
            executor: Optional execution function

        Returns:
            ActionResult if executed, None if blocked
        """
        if self._kill_switch:
            raise RuntimeError("Agent kill switch activated")

        self._check_paused()

        if decision.ethical_score < self.ethical_threshold:
            self._audit(
                "act_blocked",
                {
                    "decision_id": decision.decision_id,
                    "reason": "ethical_score_below_threshold",
                    "score": decision.ethical_score,
                },
            )
            return None

        if decision.requires_approval:
            self.state = AgentState.AWAITING_APPROVAL

            request = self.user_sync.request_approval(
                decision=decision,
                context={"reasoning": decision.reasoning},
                urgency="high" if decision.risk_level == ActionRisk.HIGH else "normal",
            )

            status = self._wait_for_approval(request.request_id)

            if status != ApprovalStatus.APPROVED:
                self._audit(
                    "act_blocked",
                    {
                        "decision_id": decision.decision_id,
                        "reason": f"approval_{status.value}",
                    },
                )
                return None

        self.state = AgentState.ACTING

        self._action_counter += 1
        result_id = f"result_{self._action_counter:06d}"

        outcome = {}
        success = True
        side_effects = []

        if executor:
            try:
                outcome = executor(decision.action, {"decision": decision})
                success = outcome.get("success", True)
                side_effects = outcome.get("side_effects", [])
            except Exception as e:
                logger.error(f"Execution error: {e}")
                success = False
                outcome = {"error": str(e)}

        result = ActionResult(
            result_id=result_id,
            action=decision.action,
            success=success,
            outcome=outcome,
            side_effects=side_effects,
        )

        self.actions.append(result)
        self._audit(
            "act",
            {
                "result_id": result_id,
                "action": decision.action,
                "success": success,
            },
        )

        return result

    def reflect(
        self,
        decision: Decision,
        result: ActionResult | None,
    ) -> Reflection:
        """
        Reflect phase: Learn from outcomes and update rules/memories.

        Args:
            decision: Decision that was made
            result: Result of the action (None if blocked)

        Returns:
            Reflection object
        """
        if self._kill_switch:
            raise RuntimeError("Agent kill switch activated")

        self.state = AgentState.REFLECTING

        self._reflection_counter += 1
        reflection_id = f"reflect_{self._reflection_counter:06d}"

        if result is None:
            outcome_assessment = "Action was blocked - reviewing decision criteria"
            confidence_adjustment = -0.05
        elif result.success:
            outcome_assessment = "Action succeeded - reinforcing decision patterns"
            confidence_adjustment = 0.02
        else:
            outcome_assessment = "Action failed - analyzing for improvements"
            confidence_adjustment = -0.03

        lessons = self._extract_lessons(decision, result)
        rule_updates = self._generate_rule_updates(decision, result)
        memory_updates = self._generate_memory_updates(decision, result)

        reflection = Reflection(
            reflection_id=reflection_id,
            action=decision.action,
            outcome_assessment=outcome_assessment,
            lessons_learned=lessons,
            rule_updates=rule_updates,
            memory_updates=memory_updates,
            confidence_adjustment=confidence_adjustment,
        )

        self.reflections.append(reflection)
        self._audit(
            "reflect",
            {
                "reflection_id": reflection_id,
                "lessons_count": len(lessons),
                "confidence_adjustment": confidence_adjustment,
            },
        )

        if self.maintenance.should_trigger_reflection(decision.confidence + confidence_adjustment):
            self._audit(
                "maintenance_triggered",
                {
                    "reason": "low_confidence",
                    "confidence": decision.confidence + confidence_adjustment,
                },
            )

        self.state = AgentState.IDLE
        return reflection

    def run_cycle(
        self,
        data: dict[str, Any],
        source: str = "internal",
        analyzer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        ethical_scorer: Callable[[str, dict[str, Any]], float] | None = None,
        executor: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Run a complete OODA cycle.

        Args:
            data: Input data
            source: Data source
            analyzer: Analysis function
            ethical_scorer: Ethical scoring function
            executor: Action execution function

        Returns:
            Cycle results
        """
        observation = self.observe(data, source)
        orientation = self.orient(observation, analyzer)
        decision = self.decide(orientation, ethical_scorer)
        result = self.act(decision, executor)
        reflection = self.reflect(decision, result)

        return {
            "observation": observation,
            "orientation": orientation,
            "decision": decision,
            "result": result,
            "reflection": reflection,
        }

    def activate_kill_switch(self) -> None:
        """Activate the kill switch to stop all operations."""
        self._kill_switch = True
        self.state = AgentState.ERROR
        self._audit("kill_switch_activated", {})
        logger.warning("Agent kill switch activated")

    def deactivate_kill_switch(self) -> None:
        """Deactivate the kill switch."""
        self._kill_switch = False
        self.state = AgentState.IDLE
        self._audit("kill_switch_deactivated", {})
        logger.info("Agent kill switch deactivated")

    def pause(self) -> None:
        """Pause agent operations."""
        self._paused = True
        self.state = AgentState.PAUSED
        self._audit("paused", {})

    def resume(self) -> None:
        """Resume agent operations."""
        self._paused = False
        self.state = AgentState.IDLE
        self._audit("resumed", {})

    def _check_paused(self) -> None:
        """Check if agent is paused and wait if so."""
        while self._paused and not self._kill_switch:
            time.sleep(0.1)

    def _assess_data_quality(self, data: dict[str, Any]) -> float:
        """Assess quality/confidence of input data."""
        if not data:
            return 0.3

        score = 0.5

        if "timestamp" in data:
            score += 0.1
        if "source" in data:
            score += 0.1
        if "confidence" in data:
            score = data["confidence"]
        if len(data) > 5:
            score += 0.1

        return min(1.0, score)

    def _determine_action(
        self,
        orientation: Orientation,
    ) -> tuple[str, ActionRisk]:
        """Determine appropriate action based on orientation."""
        if orientation.threats:
            return "mitigate_threat", ActionRisk.HIGH

        if orientation.opportunities:
            return "capitalize_opportunity", ActionRisk.MEDIUM

        if orientation.patterns:
            return "monitor_patterns", ActionRisk.LOW

        return "continue_observation", ActionRisk.LOW

    def _default_ethical_score(
        self,
        action: str,
        context: dict[str, Any],
    ) -> float:
        """Default ethical scoring function."""
        base_score = 0.95

        if "threat" in action.lower():
            base_score -= 0.05
        if "harm" in str(context).lower():
            base_score -= 0.1
        if "humanitarian" in str(context).lower():
            base_score += 0.04

        return max(0.0, min(1.0, base_score))

    def _requires_approval(
        self,
        risk_level: ActionRisk,
        ethical_score: float,
        confidence: float,
    ) -> bool:
        """Determine if action requires user approval."""
        risk_order = [ActionRisk.LOW, ActionRisk.MEDIUM, ActionRisk.HIGH, ActionRisk.CRITICAL]
        threshold_order = [ActionRisk.LOW, ActionRisk.MEDIUM, ActionRisk.HIGH, ActionRisk.CRITICAL]

        if risk_order.index(risk_level) > threshold_order.index(self.risk_threshold):
            return True

        if ethical_score < self.ethical_threshold:
            return True

        return confidence < self.confidence_threshold

    def _generate_reasoning(
        self,
        orientation: Orientation,
        action: str,
        ethical_score: float,
    ) -> str:
        """Generate reasoning for decision."""
        parts = [f"Action: {action}"]

        if orientation.patterns:
            parts.append(f"Based on {len(orientation.patterns)} detected patterns")
        if orientation.threats:
            parts.append(f"Responding to {len(orientation.threats)} identified threats")
        if orientation.opportunities:
            parts.append(f"Capitalizing on {len(orientation.opportunities)} opportunities")

        parts.append(f"Ethical score: {ethical_score:.2%}")
        parts.append(f"Confidence: {orientation.confidence:.2%}")

        return ". ".join(parts)

    def _generate_alternatives(self, orientation: Orientation) -> list[str]:
        """Generate alternative actions."""
        alternatives = ["continue_observation"]

        if orientation.patterns:
            alternatives.append("deep_analysis")
        if orientation.threats:
            alternatives.append("escalate_to_user")
        if orientation.opportunities:
            alternatives.append("defer_action")

        return alternatives

    def _extract_lessons(
        self,
        decision: Decision,
        result: ActionResult | None,
    ) -> list[str]:
        """Extract lessons from decision outcome."""
        lessons = []

        if result is None:
            lessons.append("Action was blocked - review approval criteria")
        elif result.success:
            lessons.append(
                f"Action '{decision.action}' succeeded with confidence {decision.confidence:.2%}"
            )
        else:
            lessons.append(f"Action '{decision.action}' failed - investigate causes")

        if decision.ethical_score < 0.95:
            lessons.append("Consider improving ethical alignment")

        return lessons

    def _generate_rule_updates(
        self,
        decision: Decision,
        result: ActionResult | None,
    ) -> list[dict[str, Any]]:
        """Generate rule updates based on outcome."""
        updates = []

        if result and result.success:
            updates.append(
                {
                    "type": "reinforce",
                    "action": decision.action,
                    "confidence_boost": 0.01,
                }
            )
        elif result and not result.success:
            updates.append(
                {
                    "type": "weaken",
                    "action": decision.action,
                    "confidence_reduction": 0.02,
                }
            )

        return updates

    def _generate_memory_updates(
        self,
        decision: Decision,
        result: ActionResult | None,
    ) -> list[dict[str, Any]]:
        """Generate memory updates based on outcome."""
        return [
            {
                "type": "episodic",
                "content": {
                    "action": decision.action,
                    "success": result.success if result else False,
                    "ethical_score": decision.ethical_score,
                },
                "importance": 0.7 if result and result.success else 0.5,
            }
        ]

    def _wait_for_approval(
        self,
        request_id: str,
        check_interval: float = 0.5,
    ) -> ApprovalStatus:
        """Wait for approval with periodic checks."""
        while True:
            status = self.user_sync.check_approval_status(request_id)
            if status != ApprovalStatus.PENDING:
                return status
            time.sleep(check_interval)

    def _audit(self, action: str, details: dict[str, Any]) -> None:
        """Add entry to audit log."""
        self.audit_log.append(
            {
                "action": action,
                "details": details,
                "state": self.state.value,
                "timestamp": time.time(),
            }
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get agent statistics."""
        return {
            "state": self.state.value,
            "observations": len(self.observations),
            "orientations": len(self.orientations),
            "decisions": len(self.decisions),
            "actions": len(self.actions),
            "reflections": len(self.reflections),
            "kill_switch": self._kill_switch,
            "paused": self._paused,
            "user_sync": self.user_sync.get_statistics(),
            "maintenance": self.maintenance.get_statistics(),
        }

    def get_audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent audit log entries."""
        return self.audit_log[-limit:]
