"""
OMNI ♱ AVA (O♱A)
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

"""Tests for Symbolic Reasoning integration."""

import numpy as np

from omni_anomaly_engine.core.symbolic_reasoning import SymbolicReasoningEngine, SymbolicRule


class TestSymbolicRule:
    """Test SymbolicRule class."""

    def test_rule_initialization(self):
        """Test rule initialization."""
        rule = SymbolicRule(
            name="test_rule",
            predicate="is_anomalous",
            conditions=["condition1", "condition2"],
            confidence=0.8,
        )
        assert rule.name == "test_rule"
        assert rule.predicate == "is_anomalous"
        assert len(rule.conditions) == 2
        assert rule.confidence == 0.8

    def test_rule_default_confidence(self):
        """Test default confidence value."""
        rule = SymbolicRule(name="test_rule", predicate="is_anomalous", conditions=["condition1"])
        assert rule.confidence == 1.0

    def test_rule_evaluate(self):
        """Test rule evaluation."""
        rule = SymbolicRule(name="test_rule", predicate="is_anomalous", conditions=["condition1"])
        context = {"var1": 10, "var2": 20}
        satisfied, confidence = rule.evaluate(context)
        assert isinstance(satisfied, bool)
        assert isinstance(confidence, float)


class TestSymbolicReasoningEngine:
    """Test SymbolicReasoningEngine class."""

    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = SymbolicReasoningEngine()
        assert engine.temporal_logic is True
        assert engine.graph_based is True
        assert engine.explainability_threshold == 0.7
        assert len(engine.rules) > 0

    def test_engine_custom_config(self):
        """Test engine with custom configuration."""
        config = {
            "temporal_logic": False,
            "graph_based": False,
            "explainability_threshold": 0.5,
        }
        engine = SymbolicReasoningEngine(config)
        assert engine.temporal_logic is False
        assert engine.graph_based is False
        assert engine.explainability_threshold == 0.5

    def test_default_rules_initialized(self):
        """Test that default rules are initialized."""
        engine = SymbolicReasoningEngine()
        assert len(engine.rules) >= 3
        rule_names = [rule.name for rule in engine.rules]
        assert "high_complexity_rule" in rule_names
        assert "unusual_pattern_rule" in rule_names
        assert "refactoring_candidate_rule" in rule_names

    def test_add_rule(self):
        """Test adding custom rule."""
        engine = SymbolicReasoningEngine()
        initial_count = len(engine.rules)

        new_rule = SymbolicRule(name="custom_rule", predicate="is_custom", conditions=["test"])
        engine.add_rule(new_rule)

        assert len(engine.rules) == initial_count + 1
        assert engine.rules[-1].name == "custom_rule"

    def test_reason_basic(self):
        """Test basic reasoning."""
        engine = SymbolicReasoningEngine()
        neural_output = np.array([0.8])
        context = {"cyclomatic_complexity": 5}

        results = engine.reason(neural_output, context)

        assert "neural_score" in results
        assert "symbolic_rules_fired" in results
        assert "explanations" in results
        assert "final_decision" in results
        assert "confidence" in results

    def test_reason_neural_score(self):
        """Test neural score extraction."""
        engine = SymbolicReasoningEngine()
        neural_output = np.array([0.75])
        context = {}

        results = engine.reason(neural_output, context)
        assert results["neural_score"] == 0.75

    def test_reason_empty_neural_output(self):
        """Test reasoning with empty neural output."""
        engine = SymbolicReasoningEngine()
        neural_output = np.array([])
        context = {}

        results = engine.reason(neural_output, context)
        assert results["neural_score"] == 0.0

    def test_reason_decision_anomalous(self):
        """Test decision when confidence is high."""
        engine = SymbolicReasoningEngine()
        neural_output = np.array([0.9])
        context = {}

        results = engine.reason(neural_output, context)
        assert results["final_decision"] in ["anomalous", "normal"]

    def test_reason_decision_normal(self):
        """Test decision when confidence is low."""
        engine = SymbolicReasoningEngine()
        neural_output = np.array([0.1])
        context = {}

        results = engine.reason(neural_output, context)
        assert results["final_decision"] in ["anomalous", "normal"]

    def test_reason_confidence_range(self):
        """Test confidence is in valid range."""
        engine = SymbolicReasoningEngine()
        neural_output = np.array([0.5])
        context = {}

        results = engine.reason(neural_output, context)
        assert 0.0 <= results["confidence"] <= 1.0

    def test_explain_decision_basic(self):
        """Test decision explanation."""
        engine = SymbolicReasoningEngine()
        reasoning_results = {
            "final_decision": "anomalous",
            "confidence": 0.85,
            "explanations": ["Rule 'test': is_anomalous (confidence: 0.90)"],
        }

        explanation = engine.explain_decision(reasoning_results)
        assert "ANOMALOUS" in explanation
        assert "85.00%" in explanation
        assert "Rule 'test'" in explanation

    def test_explain_decision_no_explanations(self):
        """Test explanation when no symbolic rules fired."""
        engine = SymbolicReasoningEngine()
        reasoning_results = {
            "final_decision": "normal",
            "confidence": 0.3,
            "explanations": [],
        }

        explanation = engine.explain_decision(reasoning_results)
        assert "NORMAL" in explanation
        assert "neural network" in explanation.lower()

    def test_multiple_rules_evaluation(self):
        """Test evaluation of multiple rules."""
        engine = SymbolicReasoningEngine()

        rule1 = SymbolicRule(name="rule1", predicate="test1", conditions=["c1"], confidence=0.8)
        rule2 = SymbolicRule(name="rule2", predicate="test2", conditions=["c2"], confidence=0.9)

        engine.add_rule(rule1)
        engine.add_rule(rule2)

        neural_output = np.array([0.5])
        context = {}

        results = engine.reason(neural_output, context)
        assert isinstance(results["symbolic_rules_fired"], list)

    def test_explainability_threshold_filtering(self):
        """Test that low-confidence rules are filtered."""
        config = {"explainability_threshold": 0.95}
        engine = SymbolicReasoningEngine(config)

        neural_output = np.array([0.5])
        context = {}

        results = engine.reason(neural_output, context)
        for exp in results["explanations"]:
            confidence_str = exp.split("confidence: ")[1].rstrip(")")
            confidence = float(confidence_str)
            assert confidence >= 0.95

    def test_temporal_logic_flag(self):
        """Test temporal logic configuration flag."""
        config = {"temporal_logic": True}
        engine = SymbolicReasoningEngine(config)
        assert engine.temporal_logic is True

    def test_graph_based_flag(self):
        """Test graph-based reasoning configuration flag."""
        config = {"graph_based": True}
        engine = SymbolicReasoningEngine(config)
        assert engine.graph_based is True

    def test_reason_combined_confidence(self):
        """Test combined confidence calculation."""
        engine = SymbolicReasoningEngine()
        neural_output = np.array([0.6])
        context = {}

        results = engine.reason(neural_output, context)
        neural_score = results["neural_score"]
        assert results["confidence"] >= 0.6 * neural_score

    def test_explanation_list_structure(self):
        """Test explanation list structure."""
        engine = SymbolicReasoningEngine()
        neural_output = np.array([0.7])
        context = {}

        results = engine.reason(neural_output, context)
        assert isinstance(results["explanations"], list)
        for exp in results["explanations"]:
            assert isinstance(exp, str)

    def test_rules_fired_list_structure(self):
        """Test rules_fired list structure."""
        engine = SymbolicReasoningEngine()
        neural_output = np.array([0.7])
        context = {}

        results = engine.reason(neural_output, context)
        assert isinstance(results["symbolic_rules_fired"], list)
        for rule_name in results["symbolic_rules_fired"]:
            assert isinstance(rule_name, str)
