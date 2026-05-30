"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests for the differentiable symbolic-constraint LTN layer
(``omni_mercury_engine.ml.symbolic_constraint``).

These verify the three properties that make the layer a *genuine*
neuro-symbolic component rather than theater:

1. Logical semantics -- agreement between detector consensus and the fusion
   output is rewarded; contradiction is penalised.
2. Differentiability -- the constraint loss backpropagates into both the
   fusion output (shared weights) and the layer's own learnable parameters.
3. Robustness -- degenerate inputs (no detectors, boundary probabilities)
   stay finite and ``backward``-safe.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from omni_mercury_engine.ml.symbolic_constraint import (
    Rule,
    RuleGraph,
    SymbolicConstraintModule,
    consensus_rule_graph,
)


class TestRuleGraph:
    """Declarative rule-graph structure."""

    def test_default_graph_has_two_complementary_rules(self) -> None:
        graph = consensus_rule_graph()
        assert len(graph) == 2
        names = {r.name for r in graph.rules}
        assert names == {"R1_evidence", "R2_precision"}

    def test_predicates_are_derived_from_rules(self) -> None:
        graph = consensus_rule_graph()
        assert {"Consensus", "Anomalous", "NotConsensus", "NotAnomalous"} <= graph.predicates

    def test_empty_graph_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one rule"):
            RuleGraph(name="empty", rules=())


class TestConstruction:
    """Constructor validation and parameter registration."""

    def test_learnable_parameters_registered(self) -> None:
        module = SymbolicConstraintModule(num_detectors=5)
        names = {name for name, _ in module.named_parameters()}
        assert "rule_weights" in names
        assert "detector_logits" in names
        assert "sharpness_logit" in names

    def test_no_detector_logits_when_reliability_disabled(self) -> None:
        module = SymbolicConstraintModule(num_detectors=5, learn_detector_reliability=False)
        assert module.detector_logits is None

    def test_invalid_num_detectors_rejected(self) -> None:
        with pytest.raises(ValueError, match="num_detectors"):
            SymbolicConstraintModule(num_detectors=-1)

    def test_invalid_aggregator_rejected(self) -> None:
        with pytest.raises(ValueError, match="p_aggregator"):
            SymbolicConstraintModule(num_detectors=3, p_aggregator=0.5)


class TestSemantics:
    """The fuzzy logic must reward agreement and penalise contradiction."""

    @staticmethod
    def _module() -> SymbolicConstraintModule:
        torch.manual_seed(0)
        # Disable learned reliability so the test exercises pure rule logic
        # with a flat (mean) consensus.
        return SymbolicConstraintModule(num_detectors=4, learn_detector_reliability=False)

    def test_output_keys_and_bounds(self) -> None:
        module = self._module()
        scores = torch.rand(16, 4)
        prob = torch.rand(16, 1)
        out = module(prob, scores)
        assert set(out) >= {"satisfaction", "loss", "rule_satisfaction", "rule_weights"}
        sat = float(out["satisfaction"].detach())
        assert 0.0 <= sat <= 1.0
        assert abs(float(out["loss"].detach()) - (1.0 - sat)) < 1e-6

    def test_agreement_satisfies_more_than_contradiction(self) -> None:
        module = self._module()
        n = 64
        high = torch.full((n, 4), 0.95)
        low = torch.full((n, 4), 0.05)
        anomalous = torch.full((n, 1), 0.95)
        normal = torch.full((n, 1), 0.05)

        # Agreement: high scores -> high prob, low scores -> low prob.
        agree_scores = torch.cat([high, low], dim=0)
        agree_prob = torch.cat([anomalous, normal], dim=0)
        # Contradiction: high scores -> low prob, low scores -> high prob.
        contra_scores = torch.cat([high, low], dim=0)
        contra_prob = torch.cat([normal, anomalous], dim=0)

        agree_sat = float(module(agree_prob, agree_scores)["satisfaction"].detach())
        contra_sat = float(module(contra_prob, contra_scores)["satisfaction"].detach())
        assert agree_sat > contra_sat + 0.2, (agree_sat, contra_sat)
        # Agreement should be near-perfectly satisfied.
        assert agree_sat > 0.9

    def test_precision_rule_fires_on_false_positive(self) -> None:
        # Detectors jointly see nothing (low) but fusion fires (high) -> the
        # R2 precision rule should be violated (low satisfaction).
        module = self._module()
        scores = torch.full((32, 4), 0.02)
        prob = torch.full((32, 1), 0.98)
        out = module(prob, scores)
        per_rule = out["rule_satisfaction"]
        # Rule order matches consensus_rule_graph(): index 1 == R2_precision.
        assert float(per_rule[1]) < 0.2


class TestDifferentiability:
    """The constraint must train both the network and itself."""

    def test_gradient_flows_to_anomaly_prob(self) -> None:
        module = SymbolicConstraintModule(num_detectors=3)
        scores = torch.rand(8, 3)
        prob = torch.rand(8, 1, requires_grad=True)
        loss = module.constraint_loss(prob, scores)
        loss.backward()
        assert prob.grad is not None
        assert torch.isfinite(prob.grad).all()
        assert prob.grad.abs().sum() > 0

    def test_gradient_flows_to_module_parameters(self) -> None:
        module = SymbolicConstraintModule(num_detectors=3)
        scores = torch.rand(8, 3)
        prob = torch.rand(8, 1)
        module.constraint_loss(prob, scores).backward()
        assert module.rule_weights.grad is not None
        assert module.detector_logits is not None
        assert module.detector_logits.grad is not None
        assert torch.isfinite(module.rule_weights.grad).all()


class TestRobustness:
    """Degenerate inputs stay finite and backward-safe."""

    def test_zero_detectors_trivially_satisfied(self) -> None:
        module = SymbolicConstraintModule(num_detectors=0)
        prob = torch.rand(8, 1, requires_grad=True)
        scores = torch.zeros(8, 0)
        out = module(prob, scores)
        assert abs(float(out["satisfaction"]) - 1.0) < 1e-6
        out["loss"].backward()  # must not raise

    def test_boundary_probabilities_finite(self) -> None:
        module = SymbolicConstraintModule(num_detectors=2)
        prob = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
        scores = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
        out = module(prob, scores)
        assert torch.isfinite(out["satisfaction"])
        assert torch.isfinite(out["loss"])

    def test_accepts_one_dimensional_probability(self) -> None:
        module = SymbolicConstraintModule(num_detectors=2)
        prob = torch.rand(10)  # (B,) rather than (B, 1)
        scores = torch.rand(10, 2)
        out = module(prob, scores)
        assert torch.isfinite(out["satisfaction"])


class TestExplain:
    """Explainability output is JSON-serialisable and complete."""

    def test_explain_reports_each_rule(self) -> None:
        module = SymbolicConstraintModule(num_detectors=3)
        out = module.explain(torch.rand(8, 1), torch.rand(8, 3))
        assert out["graph"] == "detector_consensus"
        assert set(out["rules"]) == {"R1_evidence", "R2_precision"}
        for rule in out["rules"].values():
            assert "satisfaction" in rule
            assert "confidence" in rule
        assert len(out["detector_weights"]) == 3

    def test_custom_rule_graph_round_trips(self) -> None:
        graph = RuleGraph(
            name="custom",
            rules=(Rule("only", "Consensus", "Anomalous", "demo"),),
        )
        module = SymbolicConstraintModule(num_detectors=2, rule_graph=graph)
        out = module.explain(torch.rand(4, 1), torch.rand(4, 2))
        assert out["graph"] == "custom"
        assert set(out["rules"]) == {"only"}
