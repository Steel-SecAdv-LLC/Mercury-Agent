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
    ScarcityWeightSchedule,
    SymbolicConstraintModule,
    consensus_rule_graph,
    consensus_salience_rule_graph,
    resolve_rule_graph,
    resolve_symbolic_weight,
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


class TestScarcityWeightSchedule:
    """Label-scarcity schedule: full strength when scarce, neural when abundant."""

    def test_monotone_decay_in_positive_count(self) -> None:
        s = ScarcityWeightSchedule()
        weights = [s.weight_for(n) for n in (0, 5, 10, 25, 50)]
        assert all(a >= b for a, b in zip(weights, weights[1:]))

    def test_max_at_zero_positives(self) -> None:
        s = ScarcityWeightSchedule(lam_max=0.1)
        assert s.weight_for(0) == pytest.approx(0.1)

    def test_snaps_to_zero_when_abundant(self) -> None:
        # Far past the decay scale the weight floors to exactly 0 (neural path).
        s = ScarcityWeightSchedule(lam_max=0.1, n0=25.0, floor=1e-3)
        assert s.weight_for(10_000) == 0.0

    def test_negative_count_treated_as_zero(self) -> None:
        s = ScarcityWeightSchedule()
        assert s.weight_for(-5) == s.weight_for(0)

    def test_rejects_bad_params(self) -> None:
        with pytest.raises(ValueError):
            ScarcityWeightSchedule(n0=0.0)
        with pytest.raises(ValueError):
            ScarcityWeightSchedule(lam_max=-0.1)


class TestResolveSymbolicWeight:
    """resolve_symbolic_weight maps every spec onto a concrete lambda."""

    def test_float_passthrough(self) -> None:
        assert resolve_symbolic_weight(0.1, 999) == pytest.approx(0.1)
        assert resolve_symbolic_weight(0.0, 1) == 0.0

    def test_adaptive_aliases_use_schedule(self) -> None:
        for alias in ("adaptive", "scarcity", "auto", "ADAPTIVE"):
            assert resolve_symbolic_weight(alias, 5) == pytest.approx(
                ScarcityWeightSchedule().weight_for(5)
            )

    def test_adaptive_resolves_low_for_abundant_high_for_scarce(self) -> None:
        scarce = resolve_symbolic_weight("adaptive", 3)
        abundant = resolve_symbolic_weight("adaptive", 10_000)
        assert scarce > abundant
        assert abundant == 0.0

    def test_explicit_schedule_instance(self) -> None:
        sched = ScarcityWeightSchedule(lam_max=0.2, n0=10.0)
        assert resolve_symbolic_weight(sched, 0) == pytest.approx(0.2)

    def test_unknown_string_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_symbolic_weight("nonsense", 5)

    def test_negative_float_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_symbolic_weight(-0.5, 5)


class TestImplicationSemantics:
    """The revived crisp implication operators are correct and differentiable."""

    def test_lukasiewicz_known_values(self) -> None:
        from omni_mercury_engine.models.neurosymbolic_enhanced import FuzzyOperators

        x = torch.tensor([1.0, 0.0, 0.5, 0.8])
        y = torch.tensor([0.0, 0.3, 0.5, 0.2])
        got = FuzzyOperators.implies_lukasiewicz(x, y)
        # min(1, 1 - x + y)
        expected = torch.tensor([0.0, 1.0, 1.0, 0.4])
        assert torch.allclose(got, expected, atol=1e-6)

    def test_lukasiewicz_gradient_non_saturating(self) -> None:
        # Where the implication is < 1 the slope in x is a constant -1 (bounded),
        # unlike the product residuum whose slope vanishes as x -> 0.
        from omni_mercury_engine.models.neurosymbolic_enhanced import FuzzyOperators

        x = torch.tensor([0.9], requires_grad=True)
        y = torch.tensor([0.1])
        FuzzyOperators.implies_lukasiewicz(x, y).backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert abs(float(x.grad) + 1.0) < 1e-6

    def test_constraint_supports_each_semantics(self) -> None:
        scores = torch.rand(16, 4)
        prob = torch.rand(16, 1, requires_grad=True)
        for sem in ("product", "reichenbach", "lukasiewicz", "godel"):
            module = SymbolicConstraintModule(num_detectors=4, semantics=sem)
            assert module.semantics == sem.lower() or (
                sem == "reichenbach" and module.semantics == "reichenbach"
            )
            out = module(prob, scores)
            assert torch.isfinite(out["satisfaction"])
            assert 0.0 <= float(out["satisfaction"].detach()) <= 1.0

    def test_lukasiewicz_constraint_backpropagates(self) -> None:
        module = SymbolicConstraintModule(num_detectors=3, semantics="lukasiewicz")
        prob = torch.rand(8, 1, requires_grad=True)
        module.constraint_loss(prob, torch.rand(8, 3)).backward()
        assert prob.grad is not None and torch.isfinite(prob.grad).all()

    def test_explain_reports_semantics(self) -> None:
        module = SymbolicConstraintModule(num_detectors=3, semantics="godel")
        out = module.explain(torch.rand(8, 1), torch.rand(8, 3))
        assert out["semantics"] == "godel"

    def test_invalid_semantics_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown semantics"):
            SymbolicConstraintModule(num_detectors=3, semantics="bogus")


class TestSalienceRuleGraph:
    """The richer consensus+salience graph (revived ThresholdRule idea)."""

    def test_graph_has_three_rules_and_salient_predicate(self) -> None:
        g = consensus_salience_rule_graph()
        assert len(g) == 3
        assert {r.name for r in g.rules} == {"R1_evidence", "R2_precision", "R3_salience"}
        assert "Salient" in g.predicates

    def test_resolve_rule_graph(self) -> None:
        assert resolve_rule_graph("consensus").name == "detector_consensus"
        assert resolve_rule_graph("consensus_salience").name == "detector_consensus_salience"
        with pytest.raises(ValueError, match="unknown rule graph"):
            resolve_rule_graph("bogus")

    def test_consensus_graph_has_no_salience_params(self) -> None:
        # Default graph must stay parameter-identical to before (no salience).
        module = SymbolicConstraintModule(num_detectors=4)
        assert module._has_salience is False
        assert module.salience_threshold is None
        names = {n for n, _ in module.named_parameters()}
        assert "salience_threshold" not in names

    def test_salience_graph_registers_threshold_params(self) -> None:
        module = SymbolicConstraintModule(
            num_detectors=4, rule_graph=consensus_salience_rule_graph()
        )
        assert module._has_salience is True
        names = {n for n, _ in module.named_parameters()}
        assert "salience_threshold" in names
        assert "salience_sharpness" in names

    def test_salience_grounds_and_backpropagates(self) -> None:
        module = SymbolicConstraintModule(
            num_detectors=3, rule_graph=consensus_salience_rule_graph()
        )
        prob = torch.rand(8, 1, requires_grad=True)
        out = module(prob, torch.rand(8, 3))
        assert out["rule_satisfaction"].shape[0] == 3
        assert torch.isfinite(out["satisfaction"])
        out["loss"].backward()
        assert prob.grad is not None and torch.isfinite(prob.grad).all()
        assert module.salience_threshold is not None
        assert module.salience_threshold.grad is not None

    def test_salient_predicate_fires_on_single_strong_detector(self) -> None:
        # One detector very high, the rest silent: the soft-existential Salient
        # predicate should be high even though the (averaged) Consensus is not.
        module = SymbolicConstraintModule(
            num_detectors=4,
            rule_graph=consensus_salience_rule_graph(),
            learn_detector_reliability=False,
        )
        scores = torch.tensor([[0.99, 0.02, 0.02, 0.02]])
        salient = module._salience(scores)
        assert float(salient) > 0.6
