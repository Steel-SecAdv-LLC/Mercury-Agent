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
Symbolic Logic Layer - Logic Graphs and Explainable Decisions

Implements the symbolic layer of the neuro-symbolic architecture:
- Logic graphs using NetworkX for rule representation
- Explainable decision generation with audit trails
- Anomaly classification via symbolic thresholds
- Integration with neural layer outputs for hybrid scoring

Research Sources:
- Logic Tensor Networks (Serafini & Garcez, 2016)
- PyReason: Temporal First-Order Logic (AAAI 2023)
- Knowledge Graphs for AI (Hogan et al., 2021)
- Explainable AI (XAI) principles

Integration:
    This module receives neural features from NeuralMemoryLayer
    and produces explainable decisions for the hybrid fusion layer.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None
    logging.warning("NetworkX not available, using fallback graph implementation")

logger = logging.getLogger(__name__)


class RuleType(Enum):
    """Types of symbolic rules."""

    THRESHOLD = "threshold"
    IMPLICATION = "implication"
    CONJUNCTION = "conjunction"
    DISJUNCTION = "disjunction"
    NEGATION = "negation"
    TEMPORAL = "temporal"
    ETHICAL = "ethical"


class DecisionType(Enum):
    """Types of decisions."""

    ANOMALY = "anomaly"
    NORMAL = "normal"
    UNCERTAIN = "uncertain"
    ESCALATE = "escalate"
    BLOCK = "block"
    APPROVE = "approve"


class ExplanationType(Enum):
    """Types of explanations."""

    RULE_BASED = "rule_based"
    THRESHOLD_BASED = "threshold_based"
    PATTERN_BASED = "pattern_based"
    ETHICAL_BASED = "ethical_based"
    HYBRID = "hybrid"


@dataclass
class SymbolicRule:
    """A symbolic rule in the logic graph."""

    rule_id: str
    rule_type: RuleType
    premise: str
    conclusion: str
    confidence: float = 0.9
    priority: int = 1
    category: str = "general"
    explanation_template: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.explanation_template:
            self.explanation_template = f"Rule {self.rule_id}: {self.premise} => {self.conclusion}"


@dataclass
class ThresholdRule:
    """A threshold-based rule for numeric comparisons."""

    rule_id: str
    variable: str
    operator: str
    threshold: float
    conclusion: str
    confidence: float = 0.9
    explanation_template: str = ""

    def evaluate(self, value: float) -> bool:
        """Evaluate the threshold rule."""
        if self.operator == ">":
            return value > self.threshold
        elif self.operator == ">=":
            return value >= self.threshold
        elif self.operator == "<":
            return value < self.threshold
        elif self.operator == "<=":
            return value <= self.threshold
        elif self.operator == "==":
            return abs(value - self.threshold) < 1e-6
        elif self.operator == "!=":
            return abs(value - self.threshold) >= 1e-6
        return False


@dataclass
class ExplainableDecision:
    """An explainable decision with full audit trail."""

    decision_id: str
    decision_type: DecisionType
    confidence: float
    explanation: str
    explanation_type: ExplanationType
    rules_fired: list[str]
    neural_contribution: float = 0.0
    symbolic_contribution: float = 0.0
    timestamp: float = field(default_factory=time.time)
    audit_trail: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LogicGraphNode:
    """A node in the logic graph."""

    node_id: str
    node_type: str
    label: str
    attributes: dict[str, Any] = field(default_factory=dict)
    activation: float = 0.0


@dataclass
class LogicGraphEdge:
    """An edge in the logic graph."""

    source_id: str
    target_id: str
    edge_type: str
    weight: float = 1.0
    rule_id: str | None = None


class FallbackGraph:
    """Fallback graph implementation when NetworkX is not available."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str, dict[str, Any]]] = []

    def add_node(self, node_id: str, **attrs: Any) -> None:
        self.nodes[node_id] = attrs

    def add_edge(self, source: str, target: str, **attrs: Any) -> None:
        self.edges.append((source, target, attrs))

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    def predecessors(self, node_id: str) -> list[str]:
        return [s for s, t, _ in self.edges if t == node_id]

    def successors(self, node_id: str) -> list[str]:
        return [t for s, t, _ in self.edges if s == node_id]

    def number_of_nodes(self) -> int:
        return len(self.nodes)

    def number_of_edges(self) -> int:
        return len(self.edges)


class LogicGraph:
    """
    Logic Graph for symbolic reasoning.

    Uses NetworkX (or fallback) to represent rules as a directed graph
    where nodes are propositions and edges are implications.
    """

    graph: nx.DiGraph[str] | FallbackGraph

    def __init__(self) -> None:
        """Initialize the logic graph."""
        if NETWORKX_AVAILABLE and nx is not None:
            self.graph = nx.DiGraph()
        else:
            self.graph = FallbackGraph()

        self.rules: dict[str, SymbolicRule] = {}
        self.threshold_rules: dict[str, ThresholdRule] = {}
        self._rule_counter = 0

    def add_rule(self, rule: SymbolicRule) -> str:
        """
        Add a symbolic rule to the graph.

        Args:
            rule: The symbolic rule to add

        Returns:
            Rule ID
        """
        self.rules[rule.rule_id] = rule

        if not self.graph.has_node(rule.premise):
            self.graph.add_node(
                rule.premise,
                node_type="proposition",
                activation=0.0,
            )

        if not self.graph.has_node(rule.conclusion):
            self.graph.add_node(
                rule.conclusion,
                node_type="proposition",
                activation=0.0,
            )

        self.graph.add_edge(
            rule.premise,
            rule.conclusion,
            rule_id=rule.rule_id,
            rule_type=rule.rule_type.value,
            confidence=rule.confidence,
            weight=rule.confidence * rule.priority,
        )

        logger.debug(f"Added rule: {rule.rule_id} ({rule.premise} => {rule.conclusion})")
        return rule.rule_id

    def add_threshold_rule(self, rule: ThresholdRule) -> str:
        """
        Add a threshold rule.

        Args:
            rule: The threshold rule to add

        Returns:
            Rule ID
        """
        self.threshold_rules[rule.rule_id] = rule
        return rule.rule_id

    def create_rule(
        self,
        premise: str,
        conclusion: str,
        rule_type: RuleType = RuleType.IMPLICATION,
        confidence: float = 0.9,
        priority: int = 1,
        category: str = "general",
    ) -> str:
        """
        Create and add a new rule.

        Args:
            premise: Rule premise
            conclusion: Rule conclusion
            rule_type: Type of rule
            confidence: Rule confidence
            priority: Rule priority
            category: Rule category

        Returns:
            Rule ID
        """
        self._rule_counter += 1
        rule_id = f"rule_{self._rule_counter:04d}"

        rule = SymbolicRule(
            rule_id=rule_id,
            rule_type=rule_type,
            premise=premise,
            conclusion=conclusion,
            confidence=confidence,
            priority=priority,
            category=category,
        )

        return self.add_rule(rule)

    def forward_chain(
        self,
        facts: set[str],
        max_iterations: int = 100,
    ) -> tuple[set[str], list[str]]:
        """
        Perform forward chaining inference.

        Args:
            facts: Initial set of facts
            max_iterations: Maximum inference iterations

        Returns:
            Tuple of (derived facts, rules fired)
        """
        derived = set(facts)
        rules_fired: list[str] = []

        for _ in range(max_iterations):
            new_facts = set()

            for rule_id, rule in self.rules.items():
                if rule.premise in derived and rule.conclusion not in derived:
                    new_facts.add(rule.conclusion)
                    rules_fired.append(rule_id)

            if not new_facts:
                break

            derived.update(new_facts)

        return derived, rules_fired

    def backward_chain(
        self,
        goal: str,
        facts: set[str],
        visited: set[str] | None = None,
    ) -> tuple[bool, list[str], list[str]]:
        """
        Perform backward chaining to prove a goal.

        Args:
            goal: Goal to prove
            facts: Known facts
            visited: Already visited goals (for cycle detection)

        Returns:
            Tuple of (success, proof path, rules used)
        """
        if visited is None:
            visited = set()

        if goal in facts:
            return True, [goal], []

        if goal in visited:
            return False, [], []

        visited.add(goal)

        if not self.graph.has_node(goal):
            return False, [], []

        if NETWORKX_AVAILABLE:
            predecessors = list(self.graph.predecessors(goal))
        else:
            predecessors = self.graph.predecessors(goal)

        for pred in predecessors:
            success, path, rules = self.backward_chain(pred, facts, visited)
            if success:
                for rule_id, rule in self.rules.items():
                    if rule.premise == pred and rule.conclusion == goal:
                        return True, path + [goal], rules + [rule_id]

        return False, [], []

    def evaluate_thresholds(
        self,
        values: dict[str, float],
    ) -> tuple[set[str], list[str]]:
        """
        Evaluate all threshold rules against values.

        Args:
            values: Dictionary of variable values

        Returns:
            Tuple of (derived conclusions, rules fired)
        """
        conclusions = set()
        rules_fired = []

        for rule_id, rule in self.threshold_rules.items():
            if rule.variable in values:
                if rule.evaluate(values[rule.variable]):
                    conclusions.add(rule.conclusion)
                    rules_fired.append(rule_id)

        return conclusions, rules_fired

    def get_explanation_chain(
        self,
        conclusion: str,
        rules_fired: list[str],
    ) -> list[str]:
        """
        Generate explanation chain for a conclusion.

        Args:
            conclusion: The derived conclusion
            rules_fired: Rules that were fired

        Returns:
            List of explanation strings
        """
        explanations = []

        for rule_id in rules_fired:
            rule: SymbolicRule | ThresholdRule
            if rule_id in self.rules:
                rule = self.rules[rule_id]
                if rule.conclusion == conclusion or rule.premise in [
                    self.rules[r].premise for r in rules_fired if r in self.rules
                ]:
                    explanations.append(rule.explanation_template)
            elif rule_id in self.threshold_rules:
                rule = self.threshold_rules[rule_id]
                if rule.conclusion == conclusion:
                    explanations.append(
                        rule.explanation_template
                        or f"{rule.variable} {rule.operator} {rule.threshold}"
                    )

        return explanations

    def get_statistics(self) -> dict[str, Any]:
        """Get graph statistics."""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_rules": len(self.rules),
            "num_threshold_rules": len(self.threshold_rules),
            "rule_types": {
                rt.value: sum(1 for r in self.rules.values() if r.rule_type == rt)
                for rt in RuleType
            },
        }


class SymbolicReasoner:
    """
    Symbolic Reasoner for explainable decision making.

    Combines logic graph inference with threshold evaluation
    to produce explainable decisions with full audit trails.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        require_explanation: bool = True,
    ):
        """
        Initialize symbolic reasoner.

        Args:
            confidence_threshold: Minimum confidence for decisions
            require_explanation: Whether to require explanations
        """
        self.confidence_threshold = confidence_threshold
        self.require_explanation = require_explanation
        self.logic_graph = LogicGraph()
        self._decision_counter = 0

        self._initialize_default_rules()

    def _initialize_default_rules(self) -> None:
        """Initialize default symbolic rules."""
        self.logic_graph.create_rule(
            premise="high_anomaly_score",
            conclusion="is_anomaly",
            rule_type=RuleType.THRESHOLD,
            confidence=0.9,
            category="anomaly_detection",
        )

        self.logic_graph.create_rule(
            premise="pattern_deviation",
            conclusion="requires_investigation",
            rule_type=RuleType.IMPLICATION,
            confidence=0.85,
            category="pattern_analysis",
        )

        self.logic_graph.create_rule(
            premise="escalation_detected",
            conclusion="high_priority",
            rule_type=RuleType.IMPLICATION,
            confidence=0.95,
            category="priority",
        )

        self.logic_graph.create_rule(
            premise="ethical_violation",
            conclusion="action_blocked",
            rule_type=RuleType.ETHICAL,
            confidence=1.0,
            priority=10,
            category="ethics",
        )

        self.logic_graph.create_rule(
            premise="is_anomaly",
            conclusion="requires_investigation",
            rule_type=RuleType.IMPLICATION,
            confidence=0.9,
            category="workflow",
        )

        self.logic_graph.create_rule(
            premise="high_priority",
            conclusion="immediate_action",
            rule_type=RuleType.IMPLICATION,
            confidence=0.9,
            category="workflow",
        )

        self.logic_graph.add_threshold_rule(
            ThresholdRule(
                rule_id="thresh_anomaly_high",
                variable="anomaly_score",
                operator=">",
                threshold=0.7,
                conclusion="high_anomaly_score",
                confidence=0.9,
                explanation_template="Anomaly score exceeds 0.7 threshold",
            )
        )

        self.logic_graph.add_threshold_rule(
            ThresholdRule(
                rule_id="thresh_deviation",
                variable="deviation_score",
                operator=">",
                threshold=2.0,
                conclusion="pattern_deviation",
                confidence=0.85,
                explanation_template="Pattern deviation exceeds 2 standard deviations",
            )
        )

        self.logic_graph.add_threshold_rule(
            ThresholdRule(
                rule_id="thresh_escalation",
                variable="escalation_rate",
                operator=">",
                threshold=0.1,
                conclusion="escalation_detected",
                confidence=0.9,
                explanation_template="Escalation rate exceeds 0.1 threshold",
            )
        )

        self.logic_graph.add_threshold_rule(
            ThresholdRule(
                rule_id="thresh_benevolence",
                variable="benevolence_score",
                operator="<",
                threshold=0.99,
                conclusion="ethical_review_required",
                confidence=1.0,
                explanation_template="Benevolence score below 0.99 requires review",
            )
        )

    def add_rule(self, rule: SymbolicRule) -> str:
        """Add a custom rule."""
        return self.logic_graph.add_rule(rule)

    def add_threshold_rule(self, rule: ThresholdRule) -> str:
        """Add a custom threshold rule."""
        return self.logic_graph.add_threshold_rule(rule)

    def reason(
        self,
        facts: set[str],
        values: dict[str, float],
        neural_score: float = 0.0,
    ) -> ExplainableDecision:
        """
        Perform symbolic reasoning and produce explainable decision.

        Args:
            facts: Known facts
            values: Numeric values for threshold evaluation
            neural_score: Neural network contribution

        Returns:
            ExplainableDecision with full audit trail
        """
        self._decision_counter += 1
        decision_id = f"decision_{self._decision_counter:06d}"

        audit_trail: list[dict[str, Any]] = []

        audit_trail.append(
            {
                "step": "input",
                "facts": list(facts),
                "values": values,
                "neural_score": neural_score,
                "timestamp": time.time(),
            }
        )

        threshold_conclusions, threshold_rules = self.logic_graph.evaluate_thresholds(values)
        all_facts = facts.union(threshold_conclusions)

        audit_trail.append(
            {
                "step": "threshold_evaluation",
                "conclusions": list(threshold_conclusions),
                "rules_fired": threshold_rules,
                "timestamp": time.time(),
            }
        )

        derived_facts, inference_rules = self.logic_graph.forward_chain(all_facts)

        audit_trail.append(
            {
                "step": "forward_chaining",
                "derived_facts": list(derived_facts - all_facts),
                "rules_fired": inference_rules,
                "timestamp": time.time(),
            }
        )

        all_rules_fired = threshold_rules + inference_rules

        decision_type, symbolic_confidence = self._determine_decision(
            derived_facts, all_rules_fired
        )

        combined_confidence = self._combine_confidence(
            symbolic_confidence, neural_score, len(all_rules_fired)
        )

        explanation = self._generate_explanation(
            decision_type, derived_facts, all_rules_fired, neural_score
        )

        explanation_type = self._determine_explanation_type(
            threshold_rules, inference_rules, neural_score
        )

        audit_trail.append(
            {
                "step": "decision",
                "decision_type": decision_type.value,
                "confidence": combined_confidence,
                "timestamp": time.time(),
            }
        )

        return ExplainableDecision(
            decision_id=decision_id,
            decision_type=decision_type,
            confidence=combined_confidence,
            explanation=explanation,
            explanation_type=explanation_type,
            rules_fired=all_rules_fired,
            neural_contribution=neural_score,
            symbolic_contribution=symbolic_confidence,
            audit_trail=audit_trail,
            metadata={
                "derived_facts": list(derived_facts),
                "threshold_conclusions": list(threshold_conclusions),
            },
        )

    def _determine_decision(
        self,
        derived_facts: set[str],
        rules_fired: list[str],
    ) -> tuple[DecisionType, float]:
        """Determine decision type from derived facts."""
        if "action_blocked" in derived_facts:
            return DecisionType.BLOCK, 1.0

        if "immediate_action" in derived_facts:
            return DecisionType.ESCALATE, 0.95

        if "is_anomaly" in derived_facts:
            confidence = 0.9 if len(rules_fired) > 2 else 0.8
            return DecisionType.ANOMALY, confidence

        if "requires_investigation" in derived_facts:
            return DecisionType.UNCERTAIN, 0.7

        if rules_fired:
            return DecisionType.UNCERTAIN, 0.6

        return DecisionType.NORMAL, 0.8

    def _combine_confidence(
        self,
        symbolic_confidence: float,
        neural_score: float,
        num_rules: int,
    ) -> float:
        """Combine symbolic and neural confidence."""
        if neural_score > 0:
            symbolic_weight = 0.6
            neural_weight = 0.4
            combined = symbolic_weight * symbolic_confidence + neural_weight * neural_score
        else:
            combined = symbolic_confidence

        rule_boost = min(0.1, num_rules * 0.02)
        combined = min(1.0, combined + rule_boost)

        return combined

    def _generate_explanation(
        self,
        decision_type: DecisionType,
        derived_facts: set[str],
        rules_fired: list[str],
        neural_score: float,
    ) -> str:
        """Generate human-readable explanation."""
        parts = [f"Decision: {decision_type.value.upper()}"]

        if rules_fired:
            rule_explanations = []
            for rule_id in rules_fired[:5]:
                rule: SymbolicRule | ThresholdRule
                if rule_id in self.logic_graph.rules:
                    rule = self.logic_graph.rules[rule_id]
                    rule_explanations.append(f"  - {rule.explanation_template}")
                elif rule_id in self.logic_graph.threshold_rules:
                    rule = self.logic_graph.threshold_rules[rule_id]
                    rule_explanations.append(
                        f"  - {rule.explanation_template or f'{rule.variable} {rule.operator} {rule.threshold}'}"
                    )

            if rule_explanations:
                parts.append("Reasoning:")
                parts.extend(rule_explanations)

        key_facts = [
            f for f in derived_facts if f in ["is_anomaly", "high_priority", "action_blocked"]
        ]
        if key_facts:
            parts.append(f"Key findings: {', '.join(key_facts)}")

        if neural_score > 0:
            parts.append(f"Neural contribution: {neural_score:.2%}")

        return "\n".join(parts)

    def _determine_explanation_type(
        self,
        threshold_rules: list[str],
        inference_rules: list[str],
        neural_score: float,
    ) -> ExplanationType:
        """Determine the type of explanation."""
        has_threshold = len(threshold_rules) > 0
        has_inference = len(inference_rules) > 0
        has_neural = neural_score > 0

        if has_threshold and has_inference and has_neural:
            return ExplanationType.HYBRID
        elif has_threshold and not has_inference:
            return ExplanationType.THRESHOLD_BASED
        elif has_inference and not has_threshold:
            return ExplanationType.RULE_BASED
        elif has_neural:
            return ExplanationType.PATTERN_BASED
        else:
            return ExplanationType.RULE_BASED

    def prove_goal(
        self,
        goal: str,
        facts: set[str],
        values: dict[str, float],
    ) -> tuple[bool, list[str], str]:
        """
        Attempt to prove a goal using backward chaining.

        Args:
            goal: Goal to prove
            facts: Known facts
            values: Numeric values

        Returns:
            Tuple of (success, proof path, explanation)
        """
        threshold_conclusions, _ = self.logic_graph.evaluate_thresholds(values)
        all_facts = facts.union(threshold_conclusions)

        success, path, rules = self.logic_graph.backward_chain(goal, all_facts)

        if success:
            explanation = f"Goal '{goal}' proven via: {' -> '.join(path)}"
        else:
            explanation = f"Goal '{goal}' could not be proven from available facts"

        return success, path, explanation

    def get_statistics(self) -> dict[str, Any]:
        """Get reasoner statistics."""
        return {
            "decisions_made": self._decision_counter,
            "confidence_threshold": self.confidence_threshold,
            "graph_stats": self.logic_graph.get_statistics(),
        }


class SymbolicLogicLayer:
    """
    Symbolic Logic Layer - Main interface for symbolic reasoning.

    Integrates logic graphs, threshold rules, and explainable decisions
    into a unified interface for the neuro-symbolic architecture.

    This is the symbolic component that receives neural features
    and produces explainable decisions.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        benevolence_threshold: float = 0.99,
    ):
        """
        Initialize Symbolic Logic Layer.

        Args:
            confidence_threshold: Minimum confidence for decisions
            benevolence_threshold: Minimum benevolence score for actions
        """
        self.confidence_threshold = confidence_threshold
        self.benevolence_threshold = benevolence_threshold

        self.reasoner = SymbolicReasoner(
            confidence_threshold=confidence_threshold,
            require_explanation=True,
        )

        self.decisions: list[ExplainableDecision] = []
        self._initialize_ethical_rules()

        logger.info(
            f"SymbolicLogicLayer initialized (conf={confidence_threshold}, "
            f"benevolence={benevolence_threshold})"
        )

    def _initialize_ethical_rules(self) -> None:
        """Initialize ethical constraint rules."""
        self.reasoner.add_rule(
            SymbolicRule(
                rule_id="ethical_harm_prevention",
                rule_type=RuleType.ETHICAL,
                premise="potential_harm",
                conclusion="action_blocked",
                confidence=1.0,
                priority=100,
                category="ethics",
                explanation_template="Action blocked: Potential harm detected",
            )
        )

        self.reasoner.add_rule(
            SymbolicRule(
                rule_id="ethical_consent",
                rule_type=RuleType.ETHICAL,
                premise="requires_consent AND NOT consent_given",
                conclusion="action_blocked",
                confidence=1.0,
                priority=100,
                category="ethics",
                explanation_template="Action blocked: Consent required but not given",
            )
        )

        self.reasoner.add_rule(
            SymbolicRule(
                rule_id="ethical_privacy",
                rule_type=RuleType.ETHICAL,
                premise="privacy_violation",
                conclusion="action_blocked",
                confidence=1.0,
                priority=100,
                category="ethics",
                explanation_template="Action blocked: Privacy violation detected",
            )
        )

        self.reasoner.add_rule(
            SymbolicRule(
                rule_id="humanitarian_priority",
                rule_type=RuleType.IMPLICATION,
                premise="humanitarian_context",
                conclusion="high_priority",
                confidence=0.95,
                priority=5,
                category="humanitarian",
                explanation_template="Humanitarian context detected: Elevated priority",
            )
        )

        self.reasoner.add_threshold_rule(
            ThresholdRule(
                rule_id="thresh_benevolence_block",
                variable="benevolence_score",
                operator="<",
                threshold=self.benevolence_threshold,
                conclusion="ethical_review_required",
                confidence=1.0,
                explanation_template=f"Benevolence score below {self.benevolence_threshold} threshold",
            )
        )

        self.reasoner.add_threshold_rule(
            ThresholdRule(
                rule_id="thresh_gini_equity",
                variable="gini_coefficient",
                operator=">",
                threshold=0.4,
                conclusion="equity_concern",
                confidence=0.9,
                explanation_template="Gini coefficient indicates potential equity concern",
            )
        )

    def process_neural_output(
        self,
        neural_features: dict[str, float],
        context_facts: set[str] | None = None,
    ) -> ExplainableDecision:
        """
        Process neural layer output and produce explainable decision.

        Args:
            neural_features: Features from neural layer
            context_facts: Additional context facts

        Returns:
            ExplainableDecision with full audit trail
        """
        facts = context_facts or set()

        neural_score = neural_features.get("anomaly_score", 0.0)

        decision = self.reasoner.reason(
            facts=facts,
            values=neural_features,
            neural_score=neural_score,
        )

        self.decisions.append(decision)

        return decision

    def evaluate_action(
        self,
        action: str,
        context: dict[str, Any],
        benevolence_score: float,
    ) -> tuple[bool, ExplainableDecision]:
        """
        Evaluate whether an action should be allowed.

        Args:
            action: Action to evaluate
            context: Action context
            benevolence_score: Computed benevolence score

        Returns:
            Tuple of (allowed, decision)
        """
        facts = set()

        if context.get("requires_consent") and not context.get("consent_given"):
            facts.add("requires_consent")

        if context.get("potential_harm"):
            facts.add("potential_harm")

        if context.get("privacy_sensitive"):
            facts.add("privacy_violation")

        if context.get("humanitarian"):
            facts.add("humanitarian_context")

        values = {
            "benevolence_score": benevolence_score,
            "gini_coefficient": context.get("gini_coefficient", 0.0),
            "harm_score": context.get("harm_score", 0.0),
        }

        decision = self.reasoner.reason(facts=facts, values=values)

        allowed = decision.decision_type not in [DecisionType.BLOCK]
        allowed = allowed and benevolence_score >= self.benevolence_threshold

        self.decisions.append(decision)

        return allowed, decision

    def add_custom_rule(
        self,
        premise: str,
        conclusion: str,
        rule_type: RuleType = RuleType.IMPLICATION,
        confidence: float = 0.9,
        category: str = "custom",
    ) -> str:
        """Add a custom symbolic rule."""
        return self.reasoner.logic_graph.create_rule(
            premise=premise,
            conclusion=conclusion,
            rule_type=rule_type,
            confidence=confidence,
            category=category,
        )

    def get_decision_history(
        self,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get recent decision history."""
        recent = self.decisions[-limit:]
        return [
            {
                "decision_id": d.decision_id,
                "decision_type": d.decision_type.value,
                "confidence": d.confidence,
                "explanation": d.explanation,
                "rules_fired": d.rules_fired,
                "timestamp": d.timestamp,
            }
            for d in recent
        ]

    def get_symbolic_features(self) -> np.ndarray[Any, Any]:
        """
        Get aggregated symbolic features for fusion layer.

        Returns:
            Feature vector summarizing symbolic layer state
        """
        if not self.decisions:
            return np.zeros(20)

        recent_decisions = self.decisions[-10:]

        decision_type_counts = np.zeros(len(DecisionType))
        for d in recent_decisions:
            decision_type_counts[list(DecisionType).index(d.decision_type)] += 1
        decision_type_counts /= len(recent_decisions)

        avg_confidence = np.mean([d.confidence for d in recent_decisions])
        avg_neural = np.mean([d.neural_contribution for d in recent_decisions])
        avg_symbolic = np.mean([d.symbolic_contribution for d in recent_decisions])
        avg_rules = np.mean([len(d.rules_fired) for d in recent_decisions])

        stats = self.reasoner.get_statistics()
        graph_stats = stats["graph_stats"]

        features = np.concatenate(
            [
                decision_type_counts,
                [avg_confidence],
                [avg_neural],
                [avg_symbolic],
                [avg_rules / 10],
                [graph_stats["num_rules"] / 100],
                [graph_stats["num_threshold_rules"] / 20],
                [len(self.decisions) / 1000],
            ]
        )

        return features

    def get_statistics(self) -> dict[str, Any]:
        """Get layer statistics."""
        return {
            "total_decisions": len(self.decisions),
            "confidence_threshold": self.confidence_threshold,
            "benevolence_threshold": self.benevolence_threshold,
            "reasoner_stats": self.reasoner.get_statistics(),
            "decision_type_distribution": {
                dt.value: sum(1 for d in self.decisions if d.decision_type == dt)
                for dt in DecisionType
            },
        }
