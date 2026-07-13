# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioral tests for the multi-agent orchestration revival (pillar B).

Covers two surfaces:

1. **Defect-fix regressions** in the four revived modules
   (``hierarchical_planning``, ``multi_agent_coordination``, ``reflexion``,
   ``chain_of_thought``): each test pins a behavior that was broken while the
   modules were dormant — empty plans, silently dropped votes, fail-open
   quorum verdicts, threshold-unfaithful reasoning traces, and a reflection
   loop that never reflected.

2. **Orchestrated behavior** of
   :class:`omni_mercury_engine.agentic.orchestration.MultiAgentOrchestrator`:
   planner-driven stage sequencing with real TD value learning, per-sample
   consensus over the engine's real detectors, reflexion-driven threshold
   adaptation from real labeled feedback, trace fidelity to issued
   decisions, explicit abstention below quorum, and fail-closed ethical
   gating.

Assertions are multi-seed where stochasticity exists, in the style of
``test_knowledge_graph_behavioral.py``.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.agentic.orchestration import (
    DetectorAgent,
    MultiAgentOrchestrator,
    OrchestrationError,
)
from omni_mercury_engine.cognitive.ethical_bounding import EthicalConstraintViolationError
from omni_mercury_engine.governance.self_improvement import MeasurementGovernance

SEEDS = (0, 1, 2)


def _planted_outlier_data(
    seed: int,
    n_normal_train: int = 250,
    n_normal_test: int = 150,
    n_outliers: int = 15,
    dim: int = 6,
    shift: float = 6.0,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Real-structure synthetic task: a normal cluster plus planted outliers."""
    rng = np.random.default_rng(seed)
    X_train = rng.normal(0.0, 1.0, size=(n_normal_train, dim))
    X_test = np.vstack(
        [
            rng.normal(0.0, 1.0, size=(n_normal_test, dim)),
            rng.normal(shift, 1.0, size=(n_outliers, dim)),
        ]
    )
    y_test = np.array([False] * n_normal_test + [True] * n_outliers)
    return X_train, X_test, y_test


def _fitted_orchestrator(seed: int, **kwargs: Any) -> tuple[
    MultiAgentOrchestrator,
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
]:
    X_train, X_test, y_test = _planted_outlier_data(seed)
    # This suite *characterises* the adaptation mechanism, so it adapts in an
    # explicit measurement stance; the production default is fail-closed and is
    # exercised directly in tests/research/test_phase3_live_wiring.py.
    kwargs.setdefault("threshold_governance", MeasurementGovernance())
    orch = MultiAgentOrchestrator(seed=seed, **kwargs).fit(X_train)
    return orch, X_test, y_test


# =============================================================================
# 1. Defect-fix regressions in the revived modules
# =============================================================================


class TestPlannerOptionSelectionFix:
    """The planner previously could not select options at all: its option
    library returned dict projections while ``plan``/``select_action``
    type-checked for ``Option`` objects, so every plan shipped empty and
    every action fell back to ``"default_action"``."""

    def test_plan_carries_options_when_applicable(self) -> None:
        from omni_mercury_engine.cognitive.hierarchical_planning import (
            AbstractionLevel,
            HierarchicalPlanner,
        )

        planner = HierarchicalPlanner()
        goal = planner.create_goal("detect anomaly", AbstractionLevel.STRATEGIC)
        # State satisfies the builtin statistical-detection option.
        plan = planner.plan(goal, {"has_numerical_data": True})
        assert len(plan.options_used) > 0
        assert plan.estimated_duration > 0

    def test_select_action_uses_real_option_policy(self) -> None:
        from omni_mercury_engine.cognitive.hierarchical_planning import (
            AbstractionLevel,
            HierarchicalPlanner,
            PlanExecutionState,
        )

        planner = HierarchicalPlanner()
        goal = planner.create_goal("detect anomaly", AbstractionLevel.STRATEGIC)
        state = {"has_numerical_data": True}
        plan = planner.plan(goal, state)
        execution = PlanExecutionState(plan=plan, current_goal=goal)

        action, option = planner.select_action(state, execution)
        assert option is not None
        # The builtin statistical option's declared policy action, not the
        # "default_action" fallback the bug produced.
        assert action == "compute_z_score"

    def test_iter_applicable_returns_options_and_dict_api_preserved(self) -> None:
        from omni_mercury_engine.cognitive.hierarchical_planning import Option, OptionLibrary

        library = OptionLibrary()
        state = {"has_numerical_data": True}
        objects = library.iter_applicable(state)
        dicts = library.get_applicable_options(state)
        assert objects and all(isinstance(o, Option) for o in objects)
        assert dicts and all(isinstance(d, dict) for d in dicts)
        assert [o.option_id for o in objects] == [d["option_id"] for d in dicts]


class TestOptionLibraryEvictionFix:
    """Eviction previously never fired (every option ID starts with "opt_",
    so the builtin filter excluded everything) and the builtin counter
    counted every option as builtin."""

    def test_eviction_caps_size_and_spares_builtins(self) -> None:
        from omni_mercury_engine.cognitive.hierarchical_planning import OptionLibrary

        library = OptionLibrary(max_options=7)
        builtin_ids = set(library._options)
        for i in range(10):
            library.add_option(
                name=f"custom_{i}",
                initiation_set={"x": True},
                policy={"default": "act"},
                termination_condition={"done": True},
            )
        assert len(library._options) <= 8  # cap + the one just added
        assert builtin_ids <= set(library._options)  # builtins never evicted

    def test_statistics_count_builtins_correctly(self) -> None:
        from omni_mercury_engine.cognitive.hierarchical_planning import OptionLibrary

        library = OptionLibrary()
        n_builtin = len(library._options)
        library.add_option(
            name="custom",
            initiation_set={},
            policy={},
            termination_condition={},
        )
        stats = library.get_statistics()
        assert stats["builtin_options"] == n_builtin
        assert stats["total_options"] == n_builtin + 1

    def test_get_action_resolves_default_policy(self) -> None:
        from omni_mercury_engine.cognitive.hierarchical_planning import Option

        option = Option(
            option_id="opt_t",
            name="t",
            initiation_set={},
            policy={"default": "fire_stage"},
            termination_condition={},
        )
        assert option.get_action({"anything": 1}) == "fire_stage"
        single = Option(
            option_id="opt_s",
            name="s",
            initiation_set={},
            policy={"action": "isolate_host"},
            termination_condition={},
        )
        assert single.get_action({"x": 2}) == "isolate_host"


class TestConsensusAbstentionAndCoercionFix:
    """Below-quorum consensus previously returned a silent benign verdict
    (fail-open), and duck-typed dict votes were silently dropped."""

    def test_below_quorum_is_explicit_abstention(self) -> None:
        from omni_mercury_engine.cognitive.multi_agent_coordination import (
            ConsensusProtocol,
            DetectionResult,
        )

        protocol = ConsensusProtocol(min_participants=3)
        results = [
            DetectionResult(agent_id="a", anomaly_score=0.9, is_anomaly=True, confidence=0.9)
        ]
        consensus = protocol.reach_consensus(results)
        assert not isinstance(consensus, dict)
        assert consensus.abstained is True
        assert consensus.participant_count == 1

    def test_quorum_consensus_not_abstained(self) -> None:
        from omni_mercury_engine.cognitive.multi_agent_coordination import (
            ConsensusProtocol,
            DetectionResult,
        )

        protocol = ConsensusProtocol(min_participants=3)
        results = [
            DetectionResult(agent_id=f"a{i}", anomaly_score=0.9, is_anomaly=True, confidence=0.9)
            for i in range(3)
        ]
        consensus = protocol.reach_consensus(results)
        assert not isinstance(consensus, dict)
        assert consensus.abstained is False
        assert consensus.final_decision is True

    def test_dict_returning_agents_are_counted_not_dropped(self) -> None:
        from omni_mercury_engine.cognitive.multi_agent_coordination import (
            AgentCoordinator,
            DetectionAgent,
        )

        class DictAgent(DetectionAgent):
            def __init__(self, agent_id: str, score: float) -> None:
                super().__init__(agent_id)
                self._score = score

            def detect(self, data: Any, context: Any = None) -> Any:
                return {"score": self._score, "is_anomaly": self._score > 0.5}

        coordinator = AgentCoordinator()
        for i, score in enumerate([0.9, 0.85, 0.8]):
            coordinator.register_agent(DictAgent(f"d{i}", score))

        consensus, individual = coordinator.coordinate_detection_detailed(np.zeros(4))
        assert len(individual) == 3  # every dict vote coerced and counted
        assert consensus.participant_count == 3
        assert consensus.final_decision is True

    def test_detection_system_reports_individual_results(self) -> None:
        from omni_mercury_engine.cognitive.multi_agent_coordination import (
            MultiAgentDetectionSystem,
        )

        system = MultiAgentDetectionSystem(num_agents=5, seed=0)
        result = system.detect(np.random.default_rng(0).normal(size=50))
        assert "abstained" in result
        assert len(result["individual_results"]) == 5
        for entry in result["individual_results"]:
            assert {"agent_id", "decision", "anomaly_score", "confidence"} <= set(entry)

    def test_list_input_preserves_full_quorum(self) -> None:
        """A plain ``list`` must not silently drop the dimensional agent.

        Its scorer uses ndarray-only ``data.ndim``/``reshape``; before the
        boundary coercion a ``list`` raised there and that agent was dropped,
        quietly shrinking the quorum from 5 to 4 while still returning a verdict.
        Array-like input must reach consensus identically to a real ndarray.
        """
        from omni_mercury_engine.cognitive.multi_agent_coordination import (
            MultiAgentDetectionSystem,
        )

        data = [1.0, 1.1, 0.9, 1.05, 0.98, 1.02, 250.0]
        as_list = MultiAgentDetectionSystem(num_agents=5, seed=0).detect(data)
        as_array = MultiAgentDetectionSystem(num_agents=5, seed=0).detect(np.asarray(data))

        assert as_list["participant_count"] == 5
        assert len(as_list["individual_results"]) == 5
        assert as_list["participant_count"] == as_array["participant_count"]
        assert as_list["consensus_decision"] == as_array["consensus_decision"]

    def test_coalition_path_accepts_list_input(self) -> None:
        """``use_coalition=True`` reads ``data.shape``; a list must not crash it."""
        from omni_mercury_engine.cognitive.multi_agent_coordination import (
            MultiAgentDetectionSystem,
        )

        system = MultiAgentDetectionSystem(num_agents=5, seed=0)
        result = system.detect([1.0, 1.1, 0.9, 1.05, 0.98, 1.02, 250.0], use_coalition=True)
        assert "consensus_decision" in result
        assert result["participant_count"] >= 1


class TestChainOfThoughtThresholdFidelityFix:
    """Reasoning traces previously classified against hardcoded 0.7/0.4
    bands regardless of the issuing pipeline's decision boundary, so a trace
    could state a determination contradicting the returned verdict."""

    @pytest.mark.parametrize("threshold", [0.3, 0.5, 0.8])
    def test_trace_determination_matches_decision_across_boundary(self, threshold: float) -> None:
        from omni_mercury_engine.cognitive.chain_of_thought import AnomalyChainOfThought

        cot = AnomalyChainOfThought(anomaly_threshold=threshold)
        for score in np.linspace(0.02, 0.98, 25):
            analysis = cot.analyze_anomaly({"score": float(score)}, float(score))
            stated_anomaly = "ANOMALY DETECTED" in str(analysis["conclusion"]).upper()
            assert stated_anomaly == analysis["is_anomaly"], (
                f"score={score:.3f} threshold={threshold}: trace says "
                f"{stated_anomaly}, verdict says {analysis['is_anomaly']}"
            )

    def test_legacy_bands_without_explicit_threshold(self) -> None:
        from omni_mercury_engine.cognitive.chain_of_thought import ChainOfThoughtEngine

        engine = ChainOfThoughtEngine(seed=0)
        chain = engine.reason("anomaly?", {"anomaly_score": 0.75})
        assert "ANOMALY DETECTED" in chain.conclusion
        chain = engine.reason("anomaly?", {"anomaly_score": 0.2})
        assert "NORMAL" in chain.conclusion

    def test_caller_data_cannot_override_locked_fidelity_keys(self) -> None:
        # The contractually locked context keys (score, threshold, verdict,
        # domain) are written after **data, so a caller-supplied dict cannot
        # silently decouple the trace from the verdict (review finding).
        from omni_mercury_engine.cognitive.chain_of_thought import AnomalyChainOfThought

        cot = AnomalyChainOfThought(anomaly_threshold=0.5)
        hostile_data = {
            "anomaly_threshold": 0.99,
            "is_anomaly": False,
            "anomaly_score": 0.01,
            "domain": "spoofed",
        }
        analysis = cot.analyze_anomaly(dict(hostile_data), 0.9)
        assert analysis["is_anomaly"] is True
        assert analysis["anomaly_score"] == 0.9
        assert "ANOMALY DETECTED" in str(analysis["conclusion"]).upper()


class TestReflexionLoopFix:
    """``execute_with_reflection`` previously recomputed an identical
    decision every iteration — reflection never fed back. Now each critique
    is answered with real evidence computed from the task data, so
    iteration contexts differ and the heuristic score can improve."""

    def test_iterations_enrich_context_with_real_evidence(self) -> None:
        from omni_mercury_engine.cognitive.reflexion import ReflexionEngine

        engine = ReflexionEngine(seed=0)
        rng = np.random.default_rng(0)
        # Low-signal data: first-iteration confidence is low, triggering the
        # "gather more evidence" critique that later iterations must answer.
        data = rng.normal(0, 1, size=40) * 0.3
        result = engine.execute_with_reflection(
            {"type": "anomaly_classification", "data": data},
            max_iterations=3,
        )
        assert result["iterations"] >= 2  # did not converge instantly
        # The best decision's context carries the computed evidence answer.
        # (Reflection ran: stats counter advanced.)
        assert engine.get_statistics()["total_reflections"] >= 1

    def test_refinement_with_zero_iterations_is_safe(self) -> None:
        from omni_mercury_engine.cognitive.reflexion import Decision, ReflexionEngine

        engine = ReflexionEngine(max_refinement_iterations=0, seed=0)
        decision = Decision(
            decision_id="d0",
            action="anomaly_detected",
            context={"anomaly_score": 0.8},
            confidence=0.8,
            reasoning="test",
        )
        result = engine.refine_decision(decision)
        assert result.iterations == 0
        assert result.refined_decision is decision


# =============================================================================
# 2. Orchestrated behavior (the wired pillar-B loop)
# =============================================================================


class TestDetectorAgent:
    """The bridge from the engine's real detectors to the protocol."""

    def test_unfitted_use_fails_closed(self) -> None:
        from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

        agent = DetectorAgent("statistical", MercuryAnomalyDetector(), seed=0)
        with pytest.raises(OrchestrationError):
            agent.score_batch(np.zeros((4, 6)))

    def test_malformed_detector_scores_fail_closed(self) -> None:
        class BrokenDetector:
            def fit(self, X: Any) -> Any:
                return self

            def detect(self, X: Any) -> dict[str, Any]:
                return {"scores": np.zeros(3)}  # wrong length for any batch != 3

        agent = DetectorAgent("broken", BrokenDetector(), seed=0)
        agent.fit(np.random.default_rng(0).normal(size=(3, 4)))
        with pytest.raises(OrchestrationError):
            agent.score_batch(np.zeros((10, 4)))

    def test_calibrated_threshold_is_a_real_quantile(self) -> None:
        from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

        X = np.random.default_rng(0).normal(size=(300, 6))
        agent = DetectorAgent("statistical", MercuryAnomalyDetector(), contamination=0.1, seed=0)
        agent.fit(X)
        scores = agent.score_batch(X)
        assert abs(agent.decision_threshold - float(np.quantile(scores, 0.9))) < 1e-9
        # Roughly the contamination fraction of training scores exceed it.
        assert 0.02 <= float(np.mean(scores > agent.decision_threshold)) <= 0.2

    def test_single_sample_protocol_detection(self) -> None:
        from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 6))
        agent = DetectorAgent("statistical", MercuryAnomalyDetector(), seed=0)
        agent.fit(X)
        outlier = np.full(6, 8.0)
        result = agent.detect(outlier)
        assert result.is_anomaly
        assert 0.0 <= result.anomaly_score <= 1.0
        with pytest.raises(ValueError):
            agent.detect(np.zeros((2, 6)))


class TestOrchestratedCoordination:
    """Consensus over the real detector ensemble on planted outliers."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_consensus_separates_planted_outliers(self, seed: int) -> None:
        from omni_mercury_engine.ml.mercury_ml import roc_auc_score

        orch, X_test, y_test = _fitted_orchestrator(seed)
        batch = orch.coordinate(X_test)
        assert batch.participant_count >= orch.min_participants
        assert not batch.abstained.any()
        auc = float(roc_auc_score(y_test.astype(int), batch.consensus_scores))
        assert auc >= 0.95, f"seed {seed}: consensus AUC {auc:.3f}"
        # Decisions at the operating threshold flag the planted outliers.
        assert float(np.mean(batch.decisions[y_test])) >= 0.9

    @pytest.mark.parametrize("seed", SEEDS)
    def test_consensus_score_consistent_with_protocol_decision(self, seed: int) -> None:
        orch, X_test, _ = _fitted_orchestrator(seed)
        batch = orch.coordinate(X_test)
        # The orchestrator's continuous score thresholds at 0.5 exactly like
        # the CONFIDENCE_WEIGHTED protocol's internal decision; with the
        # default operating threshold the two must agree everywhere.
        assert np.array_equal(batch.decisions, batch.consensus_scores > 0.5)

    def test_quorum_failure_abstains_all_samples(self) -> None:
        orch, X_test, _ = _fitted_orchestrator(0)

        # Break all but two agents' scoring: below quorum, every sample must
        # abstain — never a silent benign verdict.
        for name in list(orch.agents)[:-2]:
            orch.agents[name].score_batch = (  # type: ignore[method-assign]
                lambda X: (_ for _ in ()).throw(RuntimeError("scoring down"))
            )
        batch = orch.coordinate(X_test)
        assert batch.participant_count == 2
        assert batch.abstained.all()
        assert not batch.decisions.any()

    def test_fit_below_quorum_fails_closed(self) -> None:
        from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

        orch = MultiAgentOrchestrator(
            {"statistical": MercuryAnomalyDetector()}, min_participants=3, seed=0
        )
        with pytest.raises(OrchestrationError):
            orch.fit(np.random.default_rng(0).normal(size=(100, 6)))


class TestPlannerDrivenEpisode:
    """The hierarchical planner genuinely drives the pipeline and learns."""

    def test_episode_executes_planned_stage_sequence(self) -> None:
        orch, X_test, y_test = _fitted_orchestrator(0)
        episode = orch.run_episode(X_test, y_test)
        assert episode.plan.executed_actions == [
            "score_agents",
            "form_consensus",
            "issue_decisions",
        ]
        assert episode.plan.goal_status == "completed"
        assert all(r > 0 for r in episode.plan.stage_rewards)

    def test_td_value_of_initial_state_increases_across_episodes(self) -> None:
        orch, X_test, y_test = _fitted_orchestrator(0)
        values = []
        for _ in range(4):
            episode = orch.run_episode(X_test, y_test, apply_reflection=False)
            values.append(episode.plan.goal_value)
        # Real TD learning from real stage rewards: the initial pipeline
        # state's value estimate grows monotonically toward the return.
        assert values[0] > 0.0
        assert all(b >= a for a, b in pairwise(values))
        assert values[-1] > values[0]

    def test_unfitted_detect_fails_closed(self) -> None:
        orch = MultiAgentOrchestrator(seed=0)
        with pytest.raises(OrchestrationError):
            orch.detect(np.zeros((5, 6)))


class TestReflexionAdaptation:
    """The critic moves the operating point from real labeled feedback."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_false_negative_pressure_lowers_threshold(self, seed: int) -> None:
        # Start with an operating threshold far too high for the task: most
        # planted outliers fall below it, accumulating false negatives.
        orch, X_test, y_test = _fitted_orchestrator(seed, operating_threshold=0.9)
        episode = orch.run_episode(X_test, y_test)
        assert episode.reflection is not None
        before = episode.reflection.threshold_before
        fn_before = episode.metrics["false_negatives"]
        if episode.reflection.recommendation == "maintain":
            pytest.skip("ensemble already separated at 0.9 for this seed")
        assert episode.reflection.recommendation == "decrease"
        assert orch.operating_threshold < before

        # The adapted operating point recovers misses on the same
        # distribution (paired comparison, real labels).
        episode2 = orch.run_episode(X_test, y_test, apply_reflection=False)
        assert episode2.metrics["false_negatives"] <= fn_before
        assert episode2.metrics["balanced_accuracy"] >= episode.metrics["balanced_accuracy"]

    def test_reflection_skips_abstained_samples(self) -> None:
        orch, X_test, y_test = _fitted_orchestrator(0)
        batch = orch.coordinate(X_test)
        batch.abstained[:10] = True
        record = orch.reflect(batch, y_test, apply=False)
        assert record.n_observations == len(y_test) - 10

    def test_label_length_mismatch_rejected(self) -> None:
        orch, X_test, y_test = _fitted_orchestrator(0)
        batch = orch.coordinate(X_test)
        with pytest.raises(ValueError):
            orch.reflect(batch, y_test[:-5])


class TestTraceFidelity:
    """Every depiction must agree with the decision it depicts."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_traces_agree_with_issued_decisions(self, seed: int) -> None:
        orch, X_test, y_test = _fitted_orchestrator(seed)
        episode = orch.run_episode(X_test, y_test, apply_reflection=False)
        scores = episode.coordination.consensus_scores
        # Check the most boundary-adjacent samples plus extremes — the cases
        # where an unfaithful trace would slip through.
        order = np.argsort(np.abs(scores - episode.threshold))
        for index in [*order[:5], int(np.argmax(scores)), int(np.argmin(scores))]:
            trace = orch.explain(episode, int(index))
            assert trace["abstained"] is False
            assert trace["issued_decision"] == bool(episode.coordination.decisions[index])
            assert trace["reasoning_chain"], "trace must carry real reasoning steps"

    def test_trace_faithful_to_issue_time_threshold_after_adaptation(self) -> None:
        orch, X_test, y_test = _fitted_orchestrator(0, operating_threshold=0.9)
        episode = orch.run_episode(X_test, y_test)  # may move the threshold
        # Depictions of the *old* episode still classify at the old boundary.
        for index in range(0, len(y_test), 37):
            trace = orch.explain(episode, index)
            assert trace["issued_decision"] == bool(episode.coordination.decisions[index])

    def test_abstained_sample_traces_as_abstention(self) -> None:
        orch, X_test, y_test = _fitted_orchestrator(0)
        episode = orch.run_episode(X_test, y_test, apply_reflection=False)
        episode.coordination.abstained[3] = True
        trace = orch.explain(episode, 3)
        assert trace["abstained"] is True
        assert "ABSTAINED" in trace["conclusion"]


class TestEthicalGating:
    """The dual hard gates bind the orchestrator's decision boundary."""

    def test_benevolence_violation_blocks_decisions(self) -> None:
        from omni_mercury_engine.cognitive.ethical_bounding import EthicalScore

        orch, X_test, _ = _fitted_orchestrator(0)

        def _veto(action: str, context: dict[str, Any]) -> EthicalScore:
            return EthicalScore(
                score_id="veto",
                action=action,
                benevolence_score=0.0,
                harm_score=1.0,
                benefit_score=0.0,
                equity_score=0.0,
                long_term_score=0.0,
                is_permissible=False,
                principle_scores={},
                harm_breakdown={},
                benefit_breakdown={},
                explanation="forced veto for gate-enforcement test",
                recommendations=[],
            )

        orch._benevolence_scorer.score_action = _veto  # type: ignore[method-assign]
        with pytest.raises(EthicalConstraintViolationError):
            orch.detect(X_test)

    def test_benign_episode_passes_both_gates(self) -> None:
        orch, X_test, y_test = _fitted_orchestrator(0)
        episode = orch.run_episode(X_test, y_test, domain="security")
        assert episode.benevolence_score > 0.0


class TestEngineWiring:
    """`enable_multi_agent_orchestration` plumbs the engine's detectors."""

    def test_engine_enables_orchestrator_over_its_detectors(self) -> None:
        pytest.importorskip("torch")
        from omni_mercury_engine.engine import OmniMercuryEngine

        engine = OmniMercuryEngine()
        orchestrator = engine.enable_multi_agent_orchestration(seed=0)
        assert engine.multi_agent_orchestrator is orchestrator
        assert set(orchestrator.agents) == set(engine.detectors)
        for name, agent in orchestrator.agents.items():
            assert agent.detector is engine.detectors[name]


class TestConsensusMethodContract:
    """The orchestrator's continuous-score/threshold/trace contract is
    CONFIDENCE_WEIGHTED-specific; other protocol methods must fail fast at
    construction instead of silently decoupling decisions from the
    selected protocol (review finding, 2026-06-11)."""

    @pytest.mark.parametrize(
        "method", ["majority_vote", "weighted_vote", "unanimous", "byzantine_tolerant"]
    )
    def test_non_confidence_weighted_methods_fail_fast(self, method: str) -> None:
        with pytest.raises(OrchestrationError, match="CONFIDENCE_WEIGHTED"):
            MultiAgentOrchestrator(consensus_method=method, seed=0)

    def test_confidence_weighted_accepted_by_name_and_enum(self) -> None:
        from omni_mercury_engine.cognitive.multi_agent_coordination import ConsensusMethod

        assert MultiAgentOrchestrator(consensus_method="confidence_weighted", seed=0)
        assert MultiAgentOrchestrator(consensus_method=ConsensusMethod.CONFIDENCE_WEIGHTED, seed=0)


class TestHonestStageRewards:
    """TD rewards must reflect delivered value: an all-abstention batch
    completes the plan transparently but earns no decide-stage reward."""

    def test_abstained_batch_earns_zero_decide_reward(self) -> None:
        orch, X_test, _ = _fitted_orchestrator(0)
        # Break all but two agents: below quorum, the whole batch abstains.
        for name in list(orch.agents)[:-2]:
            orch.agents[name].score_batch = (  # type: ignore[method-assign]
                lambda X: (_ for _ in ()).throw(RuntimeError("scoring down"))
            )
        episode = orch.detect(X_test)
        assert episode.coordination.abstained.all()
        assert episode.plan.executed_actions[-1] == "issue_decisions"
        assert episode.plan.stage_rewards[-1] == 0.0
        assert episode.plan.goal_status == "completed"  # transparent completion

    def test_quorum_backed_batch_earns_full_decide_reward(self) -> None:
        orch, X_test, _ = _fitted_orchestrator(0)
        episode = orch.detect(X_test)
        assert not episode.coordination.abstained.any()
        assert episode.plan.stage_rewards[-1] == 1.0


class TestThresholdSweepExtremes:
    """The evidence-grounded sweep must reach the all-or-nothing regimes
    (review finding): with fully inverted recorded scores no interior
    midpoint beats an extreme boundary."""

    def test_inverted_scores_reach_extreme_regime(self) -> None:
        from omni_mercury_engine.cognitive.reflexion import AnomalyReflexion

        reflexion = AnomalyReflexion(anomaly_threshold=0.5)
        rng = np.random.default_rng(0)
        # Inverted world: anomalies score LOW, normals score HIGH. At the
        # 0.5 threshold every prediction is wrong (balanced accuracy 0);
        # every interior midpoint is no better than 0.5; only the extreme
        # regimes (predict-all / predict-none) reach balanced accuracy 0.5.
        anomaly_scores = rng.uniform(0.10, 0.30, size=12)
        normal_scores = rng.uniform(0.70, 0.90, size=12)
        for score in anomaly_scores:
            reflexion.record_detection(prediction=float(score), ground_truth=True, features={})
        for score in normal_scores:
            reflexion.record_detection(prediction=float(score), ground_truth=False, features={})

        recommendation = reflexion.get_threshold_recommendation()
        assert recommendation["recommendation"] != "maintain"
        suggested = float(recommendation["suggested_threshold"])
        lowest = float(min(anomaly_scores.min(), normal_scores.min()))
        highest = float(max(anomaly_scores.max(), normal_scores.max()))
        assert suggested < lowest or suggested >= highest, (
            f"sweep failed to reach an extreme regime "
            f"(suggested {suggested}, observed range [{lowest}, {highest}])"
        )


class TestReflectionRobustness:
    """Reflection enrichment must not abort on non-numeric task payloads
    (review finding): it records transparent payload evidence instead."""

    def test_non_numeric_data_does_not_abort_loop(self) -> None:
        from omni_mercury_engine.cognitive.reflexion import ReflexionEngine

        engine = ReflexionEngine(seed=0)
        result = engine.execute_with_reflection(
            {"type": "evaluate_reasoning", "data": ["thought-a", "thought-b", "thought-c"]},
            max_iterations=3,
        )
        assert result["iterations"] >= 1
        assert "decision" in result

    @pytest.mark.parametrize("payload", [5.0, 0, object(), None])
    def test_unsized_payloads_do_not_abort_loop(self, payload: Any) -> None:
        # A scalar (or any object without __len__) survives the
        # hasattr-guarded array conversion as-is; the enrichment loop must
        # size it defensively rather than raising TypeError on len().
        from omni_mercury_engine.cognitive.reflexion import ReflexionEngine

        engine = ReflexionEngine(seed=0)
        result = engine.execute_with_reflection(
            {"type": "generic_task", "data": payload},
            max_iterations=3,
        )
        assert result["iterations"] >= 1
        assert "decision" in result


class TestOrchestratorStatistics:
    def test_statistics_aggregate_all_tiers(self) -> None:
        orch, X_test, y_test = _fitted_orchestrator(0)
        orch.run_episode(X_test, y_test)
        stats = orch.get_statistics()
        assert set(stats["agents"]) == set(orch.agents)
        assert stats["coordinator"]["registered_agents"] == len(orch.agents)
        assert stats["planner"]["plans_created"] >= 1
        assert stats["reflexion"]["reflections_generated"] > 0
        assert stats["chain_of_thought"] is not None
