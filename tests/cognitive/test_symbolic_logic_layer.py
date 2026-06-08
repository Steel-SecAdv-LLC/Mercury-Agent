# Copyright (C) 2025 Steel Security Advisors LLC
"""Tests for Symbolic Logic Layer - Logic Graphs and Explainable Decisions."""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.cognitive.symbolic_logic_layer import (
    DecisionType,
    ExplainableDecision,
    ExplanationType,
    LogicGraph,
    RuleType,
    SymbolicLogicLayer,
    SymbolicReasoner,
    SymbolicRule,
    ThresholdRule,
)


class TestThresholdRule:
    """Tests for ThresholdRule."""

    def test_evaluate_greater_than(self) -> None:
        """Test greater than evaluation."""
        rule = ThresholdRule(
            rule_id="test",
            variable="score",
            operator=">",
            threshold=0.5,
            conclusion="high",
        )
        assert rule.evaluate(0.6) is True
        assert rule.evaluate(0.5) is False
        assert rule.evaluate(0.4) is False

    def test_evaluate_greater_equal(self) -> None:
        """Test greater than or equal evaluation."""
        rule = ThresholdRule(
            rule_id="test",
            variable="score",
            operator=">=",
            threshold=0.5,
            conclusion="high",
        )
        assert rule.evaluate(0.6) is True
        assert rule.evaluate(0.5) is True
        assert rule.evaluate(0.4) is False

    def test_evaluate_less_than(self) -> None:
        """Test less than evaluation."""
        rule = ThresholdRule(
            rule_id="test",
            variable="score",
            operator="<",
            threshold=0.5,
            conclusion="low",
        )
        assert rule.evaluate(0.4) is True
        assert rule.evaluate(0.5) is False
        assert rule.evaluate(0.6) is False

    def test_evaluate_less_equal(self) -> None:
        """Test less than or equal evaluation."""
        rule = ThresholdRule(
            rule_id="test",
            variable="score",
            operator="<=",
            threshold=0.5,
            conclusion="low",
        )
        assert rule.evaluate(0.4) is True
        assert rule.evaluate(0.5) is True
        assert rule.evaluate(0.6) is False

    def test_evaluate_equal(self) -> None:
        """Test equality evaluation."""
        rule = ThresholdRule(
            rule_id="test",
            variable="score",
            operator="==",
            threshold=0.5,
            conclusion="exact",
        )
        assert rule.evaluate(0.5) is True
        assert rule.evaluate(0.500000001) is True
        assert rule.evaluate(0.6) is False

    def test_evaluate_not_equal(self) -> None:
        """Test not equal evaluation."""
        rule = ThresholdRule(
            rule_id="test",
            variable="score",
            operator="!=",
            threshold=0.5,
            conclusion="different",
        )
        assert rule.evaluate(0.6) is True
        assert rule.evaluate(0.5) is False


class TestSymbolicRule:
    """Tests for SymbolicRule."""

    def test_init_default_template(self) -> None:
        """Test default explanation template generation."""
        rule = SymbolicRule(
            rule_id="test_rule",
            rule_type=RuleType.IMPLICATION,
            premise="A",
            conclusion="B",
            confidence=0.9,
        )
        assert "test_rule" in rule.explanation_template
        assert "A" in rule.explanation_template
        assert "B" in rule.explanation_template

    def test_init_custom_template(self) -> None:
        """Test custom explanation template."""
        rule = SymbolicRule(
            rule_id="test_rule",
            rule_type=RuleType.THRESHOLD,
            premise="high_score",
            conclusion="anomaly",
            confidence=0.95,
            explanation_template="Custom explanation",
        )
        assert rule.explanation_template == "Custom explanation"


class TestLogicGraph:
    """Tests for LogicGraph."""

    def test_init(self) -> None:
        """Test graph initialization."""
        graph = LogicGraph()
        assert graph.graph is not None
        assert len(graph.rules) == 0

    def test_add_rule(self) -> None:
        """Test adding rules to graph."""
        graph = LogicGraph()
        rule = SymbolicRule(
            rule_id="rule1",
            rule_type=RuleType.IMPLICATION,
            premise="A",
            conclusion="B",
            confidence=0.9,
        )
        rule_id = graph.add_rule(rule)
        assert rule_id == "rule1"
        assert "rule1" in graph.rules

    def test_create_rule(self) -> None:
        """Test creating rules via helper method."""
        graph = LogicGraph()
        rule_id = graph.create_rule(
            premise="X",
            conclusion="Y",
            rule_type=RuleType.IMPLICATION,
            confidence=0.85,
        )
        assert rule_id.startswith("rule_")
        assert len(graph.rules) == 1

    def test_add_threshold_rule(self) -> None:
        """Test adding threshold rules."""
        graph = LogicGraph()
        rule = ThresholdRule(
            rule_id="thresh1",
            variable="score",
            operator=">",
            threshold=0.5,
            conclusion="high",
        )
        rule_id = graph.add_threshold_rule(rule)
        assert rule_id == "thresh1"
        assert "thresh1" in graph.threshold_rules

    def test_forward_chain_simple(self) -> None:
        """Test simple forward chaining."""
        graph = LogicGraph()
        graph.create_rule("A", "B")
        graph.create_rule("B", "C")

        derived, rules = graph.forward_chain({"A"})
        assert "B" in derived
        assert "C" in derived
        assert len(rules) == 2

    def test_forward_chain_no_match(self) -> None:
        """Test forward chaining with no matching rules."""
        graph = LogicGraph()
        graph.create_rule("A", "B")

        derived, rules = graph.forward_chain({"X"})
        assert derived == {"X"}
        assert len(rules) == 0

    def test_backward_chain_success(self) -> None:
        """Test successful backward chaining."""
        graph = LogicGraph()
        graph.create_rule("A", "B")
        graph.create_rule("B", "C")

        success, path, rules = graph.backward_chain("C", {"A"})
        assert success is True
        assert "C" in path

    def test_backward_chain_failure(self) -> None:
        """Test failed backward chaining."""
        graph = LogicGraph()
        graph.create_rule("A", "B")

        success, path, rules = graph.backward_chain("C", {"A"})
        assert success is False

    def test_evaluate_thresholds(self) -> None:
        """Test threshold evaluation."""
        graph = LogicGraph()
        graph.add_threshold_rule(
            ThresholdRule(
                rule_id="t1",
                variable="score",
                operator=">",
                threshold=0.7,
                conclusion="high",
            )
        )
        graph.add_threshold_rule(
            ThresholdRule(
                rule_id="t2",
                variable="count",
                operator=">=",
                threshold=10,
                conclusion="many",
            )
        )

        conclusions, rules = graph.evaluate_thresholds({"score": 0.8, "count": 15})
        assert "high" in conclusions
        assert "many" in conclusions
        assert len(rules) == 2

    def test_get_statistics(self) -> None:
        """Test statistics retrieval."""
        graph = LogicGraph()
        graph.create_rule("A", "B")
        graph.add_threshold_rule(ThresholdRule("t1", "x", ">", 0.5, "high"))

        stats = graph.get_statistics()
        assert stats["num_rules"] == 1
        assert stats["num_threshold_rules"] == 1


class TestSymbolicReasoner:
    """Tests for SymbolicReasoner."""

    def test_init(self) -> None:
        """Test reasoner initialization."""
        reasoner = SymbolicReasoner(confidence_threshold=0.8)
        assert reasoner.confidence_threshold == 0.8
        assert len(reasoner.logic_graph.rules) > 0

    def test_reason_normal(self) -> None:
        """Test reasoning with normal input."""
        reasoner = SymbolicReasoner()
        decision = reasoner.reason(
            facts=set(),
            values={"anomaly_score": 0.3, "deviation_score": 0.5},
            neural_score=0.2,
        )
        assert decision.decision_type == DecisionType.NORMAL
        assert decision.confidence > 0

    def test_reason_anomaly(self) -> None:
        """Test reasoning with anomalous input."""
        reasoner = SymbolicReasoner()
        decision = reasoner.reason(
            facts=set(),
            values={"anomaly_score": 0.9, "deviation_score": 3.0},
            neural_score=0.8,
        )
        assert decision.decision_type in [
            DecisionType.ANOMALY,
            DecisionType.ESCALATE,
            DecisionType.UNCERTAIN,
        ]
        assert decision.confidence > 0.5

    def test_reason_blocked(self) -> None:
        """Test reasoning with ethical violation."""
        reasoner = SymbolicReasoner()
        decision = reasoner.reason(
            facts={"ethical_violation"},
            values={},
            neural_score=0.0,
        )
        assert decision.decision_type == DecisionType.BLOCK
        assert decision.confidence == 1.0

    def test_reason_audit_trail(self) -> None:
        """Test that reasoning produces audit trail."""
        reasoner = SymbolicReasoner()
        decision = reasoner.reason(
            facts={"test_fact"},
            values={"anomaly_score": 0.5},
            neural_score=0.3,
        )
        assert len(decision.audit_trail) > 0
        assert decision.audit_trail[0]["step"] == "input"

    def test_prove_goal_success(self) -> None:
        """Test successful goal proving."""
        reasoner = SymbolicReasoner()
        reasoner.logic_graph.create_rule("fact_a", "goal_x")

        success, path, explanation = reasoner.prove_goal(
            goal="goal_x",
            facts={"fact_a"},
            values={},
        )
        assert success is True
        assert "proven" in explanation.lower()

    def test_prove_goal_failure(self) -> None:
        """Test failed goal proving."""
        reasoner = SymbolicReasoner()
        success, path, explanation = reasoner.prove_goal(
            goal="impossible_goal",
            facts={"unrelated_fact"},
            values={},
        )
        assert success is False
        assert "could not be proven" in explanation.lower()

    def test_get_statistics(self) -> None:
        """Test statistics retrieval."""
        reasoner = SymbolicReasoner()
        reasoner.reason(facts=set(), values={}, neural_score=0.0)

        stats = reasoner.get_statistics()
        assert stats["decisions_made"] == 1
        assert "graph_stats" in stats


class TestExplainableDecision:
    """Tests for ExplainableDecision dataclass."""

    def test_create_decision(self) -> None:
        """Test creating an explainable decision."""
        decision = ExplainableDecision(
            decision_id="test_001",
            decision_type=DecisionType.ANOMALY,
            confidence=0.85,
            explanation="Test explanation",
            explanation_type=ExplanationType.RULE_BASED,
            rules_fired=["rule1", "rule2"],
            neural_contribution=0.3,
            symbolic_contribution=0.7,
        )
        assert decision.decision_id == "test_001"
        assert decision.decision_type == DecisionType.ANOMALY
        assert len(decision.rules_fired) == 2


class TestSymbolicLogicLayer:
    """Tests for SymbolicLogicLayer main interface."""

    def test_init(self) -> None:
        """Test layer initialization."""
        layer = SymbolicLogicLayer(
            confidence_threshold=0.75,
            benevolence_threshold=0.98,
        )
        assert layer.confidence_threshold == 0.75
        assert layer.benevolence_threshold == 0.98

    def test_process_neural_output_normal(self) -> None:
        """Test processing normal neural output."""
        layer = SymbolicLogicLayer()
        decision = layer.process_neural_output(
            neural_features={"anomaly_score": 0.2, "deviation_score": 0.5},
            context_facts=set(),
        )
        assert decision.decision_type == DecisionType.NORMAL
        assert len(layer.decisions) == 1

    def test_process_neural_output_anomaly(self) -> None:
        """Test processing anomalous neural output."""
        layer = SymbolicLogicLayer()
        decision = layer.process_neural_output(
            neural_features={"anomaly_score": 0.85, "deviation_score": 2.5},
            context_facts=set(),
        )
        assert decision.decision_type in [DecisionType.ANOMALY, DecisionType.UNCERTAIN]

    def test_evaluate_action_allowed(self) -> None:
        """Test evaluating allowed action."""
        layer = SymbolicLogicLayer(benevolence_threshold=0.9)
        allowed, decision = layer.evaluate_action(
            action="test_action",
            context={"humanitarian": True},
            benevolence_score=0.95,
        )
        assert allowed is True

    def test_evaluate_action_blocked_benevolence(self) -> None:
        """Test action blocked due to low benevolence."""
        layer = SymbolicLogicLayer(benevolence_threshold=0.99)
        allowed, decision = layer.evaluate_action(
            action="test_action",
            context={},
            benevolence_score=0.8,
        )
        assert allowed is False

    def test_evaluate_action_blocked_harm(self) -> None:
        """Test action blocked due to potential harm."""
        layer = SymbolicLogicLayer()
        allowed, decision = layer.evaluate_action(
            action="harmful_action",
            context={"potential_harm": True},
            benevolence_score=0.99,
        )
        assert allowed is False
        assert decision.decision_type == DecisionType.BLOCK

    def test_add_custom_rule(self) -> None:
        """Test adding custom rules."""
        layer = SymbolicLogicLayer()
        initial_rules = len(layer.reasoner.logic_graph.rules)
        rule_id = layer.add_custom_rule(
            premise="custom_premise",
            conclusion="custom_conclusion",
            rule_type=RuleType.IMPLICATION,
            confidence=0.9,
        )
        assert rule_id is not None
        assert len(layer.reasoner.logic_graph.rules) == initial_rules + 1

    def test_get_decision_history(self) -> None:
        """Test decision history retrieval."""
        layer = SymbolicLogicLayer()
        layer.process_neural_output({"anomaly_score": 0.3}, set())
        layer.process_neural_output({"anomaly_score": 0.5}, set())

        history = layer.get_decision_history(limit=10)
        assert len(history) == 2
        assert "decision_id" in history[0]
        assert "decision_type" in history[0]

    def test_get_symbolic_features(self) -> None:
        """Test symbolic feature extraction."""
        layer = SymbolicLogicLayer()
        layer.process_neural_output({"anomaly_score": 0.4}, set())
        layer.process_neural_output({"anomaly_score": 0.6}, set())

        features = layer.get_symbolic_features()
        assert features.shape[0] > 0
        assert not np.all(features == 0)

    def test_get_symbolic_features_empty(self) -> None:
        """Test symbolic features with no decisions."""
        layer = SymbolicLogicLayer()
        features = layer.get_symbolic_features()
        assert features.shape == (20,)
        assert np.all(features == 0)

    def test_get_statistics(self) -> None:
        """Test statistics retrieval."""
        layer = SymbolicLogicLayer()
        layer.process_neural_output({"anomaly_score": 0.5}, set())

        stats = layer.get_statistics()
        assert stats["total_decisions"] == 1
        assert "reasoner_stats" in stats
        assert "decision_type_distribution" in stats


class TestDecisionTypes:
    """Tests for decision type enums."""

    def test_decision_types(self) -> None:
        """Test all decision types exist."""
        assert DecisionType.ANOMALY.value == "anomaly"
        assert DecisionType.NORMAL.value == "normal"
        assert DecisionType.UNCERTAIN.value == "uncertain"
        assert DecisionType.ESCALATE.value == "escalate"
        assert DecisionType.BLOCK.value == "block"
        assert DecisionType.APPROVE.value == "approve"

    def test_rule_types(self) -> None:
        """Test all rule types exist."""
        assert RuleType.THRESHOLD.value == "threshold"
        assert RuleType.IMPLICATION.value == "implication"
        assert RuleType.ETHICAL.value == "ethical"

    def test_explanation_types(self) -> None:
        """Test all explanation types exist."""
        assert ExplanationType.RULE_BASED.value == "rule_based"
        assert ExplanationType.THRESHOLD_BASED.value == "threshold_based"
        assert ExplanationType.HYBRID.value == "hybrid"
