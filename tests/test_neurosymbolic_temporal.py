# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for enhanced neurosymbolic engine components."""

from __future__ import annotations

import numpy as np
import pytest

# Import components
from omni_mercury_engine.models.neurosymbolic_temporal import (
    CausalReasoningModule,
    FuzzyOperators,
    FuzzySemantics,
    KnowledgeGraphBridge,
    MetaCognitionLayer,
    ProbabilisticLogicLayer,
    ReasoningState,
    TemporalGraphReasoner,
    TemporalNeurosymbolicEngine,
)

# Check if PyTorch is available
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestFuzzyOperators:
    """Tests for fuzzy logic operators."""

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
    def test_and_product(self) -> None:
        """Test product t-norm."""
        x = torch.tensor([0.8, 0.6])
        y = torch.tensor([0.9, 0.5])

        result = FuzzyOperators.and_product(x, y)

        assert torch.allclose(result, torch.tensor([0.72, 0.30]))

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
    def test_and_godel(self) -> None:
        """Test Gödel t-norm (min)."""
        x = torch.tensor([0.8, 0.6])
        y = torch.tensor([0.9, 0.5])

        result = FuzzyOperators.and_godel(x, y)

        assert torch.allclose(result, torch.tensor([0.8, 0.5]))

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
    def test_or_product(self) -> None:
        """Test product t-conorm."""
        x = torch.tensor([0.8])
        y = torch.tensor([0.5])

        result = FuzzyOperators.or_product(x, y)

        expected = 0.8 + 0.5 - 0.8 * 0.5  # = 0.9
        assert torch.allclose(result, torch.tensor([expected]))

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
    def test_not_standard(self) -> None:
        """Test standard negation."""
        x = torch.tensor([0.3, 0.7, 1.0])

        result = FuzzyOperators.not_standard(x)

        assert torch.allclose(result, torch.tensor([0.7, 0.3, 0.0]))

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
    def test_implies_product(self) -> None:
        """Test Reichenbach implication."""
        x = torch.tensor([0.8])
        y = torch.tensor([0.9])

        result = FuzzyOperators.implies_product(x, y)

        expected = 1 - 0.8 + 0.8 * 0.9  # = 0.92
        assert torch.allclose(result, torch.tensor([expected]))


class TestTemporalGraphReasoner:
    """Tests for temporal graph reasoning."""

    def test_add_node_and_edge(self) -> None:
        """Test adding nodes and edges."""
        reasoner = TemporalGraphReasoner()

        reasoner.add_node("patient_1", "patient", {"age": 45})
        reasoner.add_node("fever_symptom", "symptom", {"severity": 0.8})
        edge = reasoner.add_edge("patient_1", "fever_symptom", "has_symptom")

        assert "patient_1" in reasoner.nodes
        assert len(reasoner.edges) == 1
        assert edge.relation == "has_symptom"

    def test_add_temporal_rule(self) -> None:
        """Test adding temporal rules."""
        reasoner = TemporalGraphReasoner()

        rule = reasoner.add_rule(
            name="sepsis_detection",
            premise="fever AND tachycardia",
            conclusion="sepsis_risk",
            confidence=0.85,
            time_window=6,
        )

        assert len(reasoner.rules) == 1
        assert rule.time_window == 6

    def test_basic_reasoning(self) -> None:
        """Test basic reasoning query."""
        reasoner = TemporalGraphReasoner()

        # Add nodes representing facts
        reasoner.add_node("fever", "symptom", truth_value=0.9)
        reasoner.add_node("tachycardia", "symptom", truth_value=0.8)

        # Add rule
        reasoner.add_rule(
            name="infection_indicator",
            premise="symptom(fever) AND symptom(tachycardia)",
            conclusion="infection_risk",
            confidence=0.85,
        )

        result = reasoner.reason(query="infection_risk")

        assert "derived_facts" in result
        assert "explanation" in result

    def test_open_world_assumption(self) -> None:
        """Test open world reasoning (unknown != false)."""
        reasoner = TemporalGraphReasoner(open_world=True)

        result = reasoner.reason(query="unknown_fact")

        # With open world, unknown should be 0.5 (uncertain), not 0
        assert result["confidence"] == 0.5

    def test_closed_world_assumption(self) -> None:
        """Test closed world reasoning (unknown = false)."""
        reasoner = TemporalGraphReasoner(open_world=False)

        result = reasoner.reason(query="unknown_fact")

        # With closed world, unknown should be 0
        assert result["confidence"] == 0.0


class TestKnowledgeGraphBridge:
    """Tests for knowledge graph integration."""

    def test_core_knowledge_initialized(self) -> None:
        """Test that core knowledge is initialized."""
        kg = KnowledgeGraphBridge()

        # Check some core knowledge exists
        assert len(kg.knowledge_base) > 0

    def test_add_and_query_knowledge(self) -> None:
        """Test adding and querying knowledge."""
        kg = KnowledgeGraphBridge()

        kg.add_knowledge("anomaly", "IsA", "deviation", 0.9, "test")
        results = kg.query("anomaly", "IsA")

        assert len(results) > 0
        assert any(r.object == "deviation" for r in results)

    def test_inference(self) -> None:
        """Test knowledge inference."""
        kg = KnowledgeGraphBridge()

        # Direct relation
        kg.add_knowledge("sepsis", "IsA", "medical_emergency", 0.95)
        confidence = kg.infer("sepsis", "IsA", "medical_emergency")

        assert confidence == 0.95

    def test_enhance_reasoning(self) -> None:
        """Test reasoning enhancement with commonsense."""
        kg = KnowledgeGraphBridge()

        context = {
            "sepsis": True,
            "elevated_heart_rate": True,
        }

        inferences = kg.enhance_reasoning(context)

        assert isinstance(inferences, dict)


class TestMetaCognitionLayer:
    """Tests for meta-cognition layer."""

    def test_assess_state(self) -> None:
        """Test reasoning state assessment."""
        meta = MetaCognitionLayer()

        predictions = np.array([0.9, 0.1, 0.8, 0.2])
        state = meta.assess_state(predictions)

        assert isinstance(state, ReasoningState)
        assert 0 <= state.confidence <= 1
        assert 0 <= state.uncertainty <= 1

    def test_should_continue_reasoning(self) -> None:
        """Test reasoning continuation decision."""
        meta = MetaCognitionLayer(
            confidence_threshold=0.8,
            max_reasoning_depth=5,
        )

        # Low confidence - should continue
        state_low = ReasoningState(
            confidence=0.5,
            uncertainty=0.5,
            reasoning_depth=2,
            rules_fired=0,
            time_elapsed_ms=0,
            errors_encountered=0,
        )
        assert meta.should_continue_reasoning(state_low)

        # High confidence - should stop
        state_high = ReasoningState(
            confidence=0.9,
            uncertainty=0.1,
            reasoning_depth=2,
            rules_fired=0,
            time_elapsed_ms=0,
            errors_encountered=0,
        )
        assert not meta.should_continue_reasoning(state_high)

        # Max depth reached - should stop
        state_deep = ReasoningState(
            confidence=0.5,
            uncertainty=0.5,
            reasoning_depth=5,
            rules_fired=0,
            time_elapsed_ms=0,
            errors_encountered=0,
        )
        assert not meta.should_continue_reasoning(state_deep)

    def test_select_reasoning_strategy(self) -> None:
        """Test strategy selection based on state."""
        meta = MetaCognitionLayer()

        # High uncertainty -> deep symbolic
        high_uncertainty = ReasoningState(
            confidence=0.3,
            uncertainty=0.7,
            reasoning_depth=0,
            rules_fired=0,
            time_elapsed_ms=0,
            errors_encountered=0,
        )
        assert meta.select_reasoning_strategy(high_uncertainty) == "deep_symbolic"

        # High confidence -> neural only
        high_confidence = ReasoningState(
            confidence=0.85,
            uncertainty=0.15,
            reasoning_depth=0,
            rules_fired=0,
            time_elapsed_ms=0,
            errors_encountered=0,
        )
        assert meta.select_reasoning_strategy(high_confidence) == "neural_only"

    def test_uncertainty_quantification(self) -> None:
        """Test uncertainty quantification."""
        meta = MetaCognitionLayer()

        predictions = np.array([0.5, 0.5, 0.5])  # Maximum uncertainty
        uncertainty = meta.quantify_uncertainty(predictions)

        assert "aleatoric" in uncertainty
        assert "epistemic" in uncertainty
        assert "total" in uncertainty
        assert uncertainty["aleatoric"] == 0.25  # 0.5 * 0.5

    def test_calibration_with_ground_truth(self) -> None:
        """Test confidence calibration."""
        meta = MetaCognitionLayer()

        # Simulate some predictions with ground truth
        for i in range(20):
            preds = np.random.rand(10)
            gt = (np.random.rand(10) > 0.5).astype(int)
            meta.assess_state(preds, gt)

        # Should have calibration data
        assert len(meta.calibration_data) > 0


class TestCausalReasoningModule:
    """Tests for causal reasoning."""

    def test_add_causal_edge(self) -> None:
        """Test adding causal edges."""
        causal = CausalReasoningModule()

        causal.add_causal_edge("infection", "inflammation", 0.9)
        causal.add_causal_edge("inflammation", "fever", 0.85)

        assert "infection" in causal.causal_graph
        assert len(causal.causal_graph["infection"]) == 1

    def test_observe_variables(self) -> None:
        """Test observing variable values."""
        causal = CausalReasoningModule()

        causal.observe("temperature", 38.5)
        causal.observe("heart_rate", 95)

        assert causal.variable_values["temperature"] == 38.5
        assert causal.variable_values["heart_rate"] == 95

    def test_intervention(self) -> None:
        """Test do-calculus intervention."""
        causal = CausalReasoningModule()

        causal.add_causal_edge("treatment", "recovery", 0.8)
        causal.observe("treatment", 0)
        causal.observe("recovery", 0)

        # Intervene: do(treatment = 1)
        result = causal.intervene("treatment", 1)

        assert result["treatment"] == 1
        assert result["recovery"] == 0.8  # Should be affected

    def test_counterfactual(self) -> None:
        """Test counterfactual reasoning."""
        causal = CausalReasoningModule()

        causal.add_causal_edge("smoking", "lung_disease", 0.7)

        observation: dict[str, float] = {"smoking": 1.0, "lung_disease": 0.7}
        intervention: dict[str, float] = {"smoking": 0.0}

        result = causal.counterfactual(observation, intervention)

        # If hadn't smoked, lung disease risk should be 0
        assert result["lung_disease"] == 0

    def test_find_root_causes(self) -> None:
        """Test root cause analysis."""
        causal = CausalReasoningModule()

        # Chain: A -> B -> C
        causal.add_causal_edge("root_cause", "intermediate", 0.9)
        causal.add_causal_edge("intermediate", "effect", 0.8)

        causes = causal.find_root_causes("effect")

        assert len(causes) >= 1
        # Should find intermediate first, then root_cause
        cause_names = [c[0] for c in causes]
        assert "intermediate" in cause_names


class TestProbabilisticLogicLayer:
    """Tests for probabilistic logic."""

    def test_set_and_get_probability(self) -> None:
        """Test setting and getting probability bounds."""
        prob = ProbabilisticLogicLayer()

        prob.set_probability_bounds("anomaly", 0.7, 0.9)
        low, high = prob.get_probability("anomaly")

        assert low == 0.7
        assert high == 0.9

    def test_and_probability(self) -> None:
        """Test conjunction probability with Fréchet bounds."""
        prob = ProbabilisticLogicLayer()

        prob.set_probability_bounds("A", 0.8, 0.9)
        prob.set_probability_bounds("B", 0.7, 0.8)

        low, high = prob.and_probability("A", "B")

        # Fréchet bounds: max(0, P(A) + P(B) - 1) <= P(A∧B) <= min(P(A), P(B))
        assert 0 <= low <= high <= 1

    def test_or_probability(self) -> None:
        """Test disjunction probability."""
        prob = ProbabilisticLogicLayer()

        prob.set_probability_bounds("A", 0.3, 0.4)
        prob.set_probability_bounds("B", 0.5, 0.6)

        low, high = prob.or_probability("A", "B")

        assert low >= 0.5  # At least max(P(A), P(B))
        assert high <= 1


class TestEnhancedNeurosymbolicEngine:
    """Tests for unified enhanced engine."""

    def test_initialization(self) -> None:
        """Test engine initialization."""
        engine = TemporalNeurosymbolicEngine(
            input_dim=32,
            hidden_dim=128,
            use_knowledge_graph=True,
            use_meta_cognition=True,
            use_causal=True,
        )

        stats = engine.get_statistics()

        assert stats["input_dim"] == 32
        assert stats["hidden_dim"] == 128
        assert stats["knowledge_graph_size"] > 0
        assert stats["temporal_rules"] > 0

    def test_predict(self) -> None:
        """Test prediction with full neuro-symbolic stack."""
        engine = TemporalNeurosymbolicEngine(input_dim=16)

        features = np.random.randn(5, 16).astype(np.float32)
        context = {"elevated_heart_rate": True}

        result = engine.predict(features, context)

        assert "anomaly_scores" in result
        assert "neural_output" in result
        assert "temporal_reasoning" in result
        assert "explanation" in result

        if result["anomaly_scores"] is not None:
            assert len(result["anomaly_scores"]) == 5

    def test_extract_features(self) -> None:
        """Test feature extraction for detector integration."""
        engine = TemporalNeurosymbolicEngine(input_dim=16)

        data = np.random.randn(10, 16).astype(np.float32)
        features = engine.extract_features(data)

        assert features.shape[0] == 10
        assert features.dtype == np.float32

    def test_explanation_generation(self) -> None:
        """Test explanation generation."""
        engine = TemporalNeurosymbolicEngine(input_dim=16)

        features = np.random.randn(3, 16).astype(np.float32)
        result = engine.predict(features)

        explanation = result["explanation"]

        assert isinstance(explanation, str)
        assert "OMNI" in explanation or "Neural" in explanation

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
    def test_ltn_grounding(self) -> None:
        """Test Logic Tensor Network predicate grounding."""
        from omni_mercury_engine.models.neurosymbolic_temporal import TemporalLogicTensorNetwork

        ltn = TemporalLogicTensorNetwork(
            input_dim=32,
            hidden_dim=64,
            num_predicates=8,
            semantics=FuzzySemantics.PRODUCT,
        )

        x = torch.randn(4, 32)
        output = ltn(x)

        assert "predicate_values" in output
        assert "satisfaction" in output
        assert output["predicate_values"].shape == (4, 8)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch required")
    def test_formula_evaluation(self) -> None:
        """Test logical formula evaluation."""
        from omni_mercury_engine.models.neurosymbolic_temporal import TemporalLogicTensorNetwork

        ltn = TemporalLogicTensorNetwork(
            input_dim=32,
            hidden_dim=64,
            num_predicates=4,
        )

        x = torch.randn(2, 32)
        pred_values = ltn.ground_predicates(x)

        # Evaluate formula: P0 AND P1
        result = ltn.evaluate_formula(pred_values, "P0 AND P1")

        assert result.shape == (2, 1)
        assert torch.all((result >= 0) & (result <= 1))
