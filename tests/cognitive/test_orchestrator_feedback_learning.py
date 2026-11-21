# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``CognitiveOrchestrator.learn_from_feedback`` reaches what ``analyze`` created.

Feedback used to be addressed to ``analysis_<id>``, a pattern no plasticity
connection has: ``analyze`` creates ``anomaly_type_<n> -> domain_<x>``. So no
weight ever moved, and the indicator half of the method was a bare ``pass``.
The same pipeline reported "Added observation to <domain> domain" for every
domain, although the knowledge graph only held five ``domain_*`` nodes and
silently dropped the edge for any other, the default ``general`` among them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from omni_mercury_engine.cognitive.ethical_bounding import EthicalConstraintViolationError
from omni_mercury_engine.cognitive.indicator_system import IndicatorType
from omni_mercury_engine.cognitive.knowledge_graph import EdgeType
from omni_mercury_engine.cognitive.orchestrator import CognitiveOrchestrator
from omni_mercury_engine.cognitive.plasticity_engine import PlasticityEngine

if TYPE_CHECKING:
    from pathlib import Path

_ANOMALY: dict[str, Any] = {"is_anomaly": True, "anomaly_prob": 0.9, "severity": 0.8}
_CONNECTION = ("anomaly_type_8", "domain_cyber")


@pytest.fixture
def orchestrator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CognitiveOrchestrator:
    # Keep the durable gate audit out of the repo tree.
    monkeypatch.setenv("MERCURY_GATE_AUDIT_LOG", str(tmp_path / "gate_audit.jsonl"))
    return CognitiveOrchestrator(enable_causal=False, enable_ipb=False)


def _analysis_id(result: Any) -> int:
    """The id ``analyze`` put on its result (never None for a completed analysis)."""
    assert isinstance(result.analysis_id, int)
    return result.analysis_id


def _weight(orch: CognitiveOrchestrator, key: tuple[str, str] = _CONNECTION) -> float:
    assert orch.plasticity is not None
    return float(orch.plasticity._connections[key].weight)


class TestFeedbackReachesThePlasticityConnection:
    def test_negative_feedback_weakens_the_connection_the_analysis_made(
        self, orchestrator: CognitiveOrchestrator
    ) -> None:
        orchestrator.analyze(_ANOMALY, context={"domain": "cyber"})
        before = _weight(orchestrator)

        orchestrator.learn_from_feedback(orchestrator._analysis_count, was_correct=False)

        assert _weight(orchestrator) < before

    def test_positive_feedback_strengthens_the_connection_the_analysis_made(
        self, orchestrator: CognitiveOrchestrator
    ) -> None:
        orchestrator.analyze(_ANOMALY, context={"domain": "cyber"})
        before = _weight(orchestrator)

        orchestrator.learn_from_feedback(orchestrator._analysis_count, was_correct=True)

        assert _weight(orchestrator) > before

    def test_the_result_names_the_analysis_and_the_report_says_what_moved(
        self, orchestrator: CognitiveOrchestrator
    ) -> None:
        result = orchestrator.analyze(_ANOMALY, context={"domain": "cyber"})
        before = _weight(orchestrator)

        report = orchestrator.learn_from_feedback(_analysis_id(result), was_correct=True)

        assert result.analysis_id == 1
        assert report["known"] is True
        assert report["applied"] is True
        assert report["plasticity_weights_changed"] == 1
        (connection,) = report["plasticity_connections"]
        assert (connection["source"], connection["target"]) == _CONNECTION
        assert connection["weight_before"] == pytest.approx(before)
        assert connection["weight_after"] == pytest.approx(_weight(orchestrator))
        assert connection["weight_after"] > connection["weight_before"]
        assert connection["eligibility"] > 0.1

    def test_the_default_domain_is_addressed_too(self, orchestrator: CognitiveOrchestrator) -> None:
        """No domain in the context: the connection is to ``domain_general``."""
        result = orchestrator.analyze(_ANOMALY)
        key = ("anomaly_type_8", "domain_general")
        before = _weight(orchestrator, key)

        orchestrator.learn_from_feedback(_analysis_id(result), was_correct=False)

        assert _weight(orchestrator, key) < before

    @pytest.mark.parametrize(
        ("was_correct", "delay_s"),
        [(True, 10.0), (False, 10.0), (True, 6 * 3600.0), (False, 6 * 3600.0)],
        ids=["right-10s", "wrong-10s", "right-6h", "wrong-6h"],
    )
    def test_feedback_after_the_eligibility_window_still_moves_the_weight(
        self, orchestrator: CognitiveOrchestrator, was_correct: bool, delay_s: float
    ) -> None:
        """Human feedback arrives seconds to hours late, and it has to land.

        The engine's eligibility trace decays in wall-clock time and is below
        the 0.1 credit bar about four seconds after a strong detection, so
        feedback routed through the trace moved no weight once a person had
        read the alert. It is addressed to the connection the analysis made
        instead, and the report says so.
        """
        result = orchestrator.analyze(_ANOMALY, context={"domain": "cyber"})
        assert orchestrator.plasticity is not None
        connection = orchestrator.plasticity._connections[_CONNECTION]
        connection.eligibility.last_update -= delay_s
        before = _weight(orchestrator)

        report = orchestrator.learn_from_feedback(_analysis_id(result), was_correct=was_correct)

        assert report["applied"] is True
        assert report["plasticity_connections"][0]["eligibility"] < 0.1
        assert report["plasticity_weights_changed"] == 1
        if was_correct:
            assert _weight(orchestrator) > before
        else:
            assert _weight(orchestrator) < before
        assert report["plasticity_connections"][0]["weight_after"] == pytest.approx(
            _weight(orchestrator)
        )

    def test_late_feedback_is_still_applied_only_once(
        self, orchestrator: CognitiveOrchestrator
    ) -> None:
        result = orchestrator.analyze(_ANOMALY, context={"domain": "cyber"})
        assert orchestrator.plasticity is not None
        orchestrator.plasticity._connections[_CONNECTION].eligibility.last_update -= 3600.0
        orchestrator.learn_from_feedback(_analysis_id(result), was_correct=True)
        after_first = _weight(orchestrator)

        report = orchestrator.learn_from_feedback(_analysis_id(result), was_correct=True)

        assert report["applied"] is False
        assert report["reason"] == "feedback_already_applied"
        assert report["plasticity_weights_changed"] == 0
        assert _weight(orchestrator) == after_first

    def test_feedback_moves_only_the_connection_the_analysis_made(
        self, orchestrator: CognitiveOrchestrator
    ) -> None:
        """Same source pattern, another domain: that analysis was not judged."""
        judged = orchestrator.analyze(_ANOMALY, context={"domain": "cyber"})
        orchestrator.analyze(_ANOMALY, context={"domain": "medical"})
        other = ("anomaly_type_8", "domain_medical")
        other_before = _weight(orchestrator, other)
        before = _weight(orchestrator)

        orchestrator.learn_from_feedback(_analysis_id(judged), was_correct=False)

        assert _weight(orchestrator) < before
        assert _weight(orchestrator, other) == other_before

    def test_feedback_on_one_analysis_is_applied_once(
        self, orchestrator: CognitiveOrchestrator
    ) -> None:
        result = orchestrator.analyze(_ANOMALY, context={"domain": "cyber"})
        orchestrator.learn_from_feedback(_analysis_id(result), was_correct=True)
        after_first = _weight(orchestrator)

        report = orchestrator.learn_from_feedback(_analysis_id(result), was_correct=True)

        assert report["applied"] is False
        assert report["reason"] == "feedback_already_applied"
        assert _weight(orchestrator) == after_first

    def test_an_analysis_that_never_ran_changes_nothing(
        self, orchestrator: CognitiveOrchestrator
    ) -> None:
        orchestrator.analyze(_ANOMALY, context={"domain": "cyber"})
        assert orchestrator.plasticity is not None
        before = _weight(orchestrator)
        feedback_keys = set(orchestrator.plasticity._outcome_feedback)

        report = orchestrator.learn_from_feedback(987_654, was_correct=False)

        assert report["known"] is False
        assert report["applied"] is False
        assert report["reason"] == "unknown_analysis"
        assert _weight(orchestrator) == before
        # ``analysis_<id>`` used to be stored as a new key per call, for ever.
        assert set(orchestrator.plasticity._outcome_feedback) == feedback_keys

    def test_an_analysis_without_an_anomaly_has_no_connection_to_credit(
        self, orchestrator: CognitiveOrchestrator
    ) -> None:
        result = orchestrator.analyze({"is_anomaly": False, "anomaly_prob": 0.1, "severity": 0.0})

        report = orchestrator.learn_from_feedback(_analysis_id(result), was_correct=True)

        assert report["known"] is True
        assert report["plasticity_connections"] == []
        assert report["plasticity_weights_changed"] == 0


class TestAddressedReward:
    """``PlasticityEngine.reinforce``: a reward for one named connection."""

    def _engine_with_cold_connection(self, weight: float = 0.5) -> PlasticityEngine:
        engine = PlasticityEngine(learning_rate=0.1)
        engine.adapt("source", "target", strength=0.9)
        connection = engine._connections[("source", "target")]
        connection.weight = weight
        connection.eligibility.last_update -= 24 * 3600.0  # the trace is long gone
        assert connection.eligibility.get_trace() < 0.1
        return engine

    def test_the_trace_based_path_ignores_a_cold_connection(self) -> None:
        """What ``feedback`` does, and keeps doing: no trace, no credit."""
        engine = self._engine_with_cold_connection()
        engine.feedback("source", success=True, magnitude=1.0)
        assert engine._connections[("source", "target")].weight == 0.5

    def test_reward_moves_a_cold_connection_by_the_learning_rate(self) -> None:
        engine = self._engine_with_cold_connection()
        delta = engine.reinforce("source", "target", success=True)
        assert delta == pytest.approx(0.1)
        assert engine._connections[("source", "target")].weight == pytest.approx(0.6)

    def test_penalty_is_half_the_reward_as_in_the_trace_based_path(self) -> None:
        engine = self._engine_with_cold_connection()
        delta = engine.reinforce("source", "target", success=False)
        assert delta == pytest.approx(-0.05)

    @pytest.mark.parametrize("magnitude", [5.0, 1e9, float("inf")])
    def test_one_reward_never_moves_a_weight_by_more_than_the_learning_rate(
        self, magnitude: float
    ) -> None:
        engine = self._engine_with_cold_connection()
        delta = engine.reinforce("source", "target", success=True, magnitude=magnitude)
        assert delta is not None
        assert abs(delta) <= engine._global_learning_rate + 1e-12

    @pytest.mark.parametrize("magnitude", [-1.0, float("nan"), float("-inf")])
    def test_a_meaningless_magnitude_moves_nothing(self, magnitude: float) -> None:
        engine = self._engine_with_cold_connection()
        assert engine.reinforce("source", "target", success=True, magnitude=magnitude) == 0.0
        assert engine._connections[("source", "target")].weight == 0.5

    def test_the_weight_stays_inside_the_engine_bounds(self) -> None:
        high = self._engine_with_cold_connection(weight=0.98)
        low = self._engine_with_cold_connection(weight=0.02)
        for _ in range(20):
            high.reinforce("source", "target", success=True)
            low.reinforce("source", "target", success=False)
        assert high._connections[("source", "target")].weight == 1.0
        assert low._connections[("source", "target")].weight == 0.0

    def test_an_unknown_connection_is_reported_and_not_created(self) -> None:
        engine = self._engine_with_cold_connection()
        feedback_keys = set(engine._outcome_feedback)
        assert engine.reinforce("source", "nowhere", success=True) is None
        assert ("source", "nowhere") not in engine._connections
        assert set(engine._outcome_feedback) == feedback_keys

    def test_only_the_named_connection_moves(self) -> None:
        engine = self._engine_with_cold_connection()
        engine.adapt("source", "elsewhere", strength=0.9)
        elsewhere = engine._connections[("source", "elsewhere")].weight
        engine.reinforce("source", "target", success=True)
        assert engine._connections[("source", "elsewhere")].weight == elsewhere

    def test_the_outcome_feeds_metaplasticity_like_any_other_feedback(self) -> None:
        engine = self._engine_with_cold_connection()
        for _ in range(5):
            engine.reinforce("source", "target", success=False)
        assert engine._outcome_feedback["source"] == [(False, 1.0)] * 5
        assert engine._recent_error_rate == 1.0


class TestFeedbackValidatesTheIndicatorsThatFired:
    def _indicator(self, orch: CognitiveOrchestrator) -> Any:
        assert orch.indicators is not None
        return orch.indicators.develop_indicator(
            pattern={"type": "anomaly"},
            name="any anomaly",
            description="fires on every anomaly observation",
            indicator_type=IndicatorType.ANOMALY,
            domain="cyber",
            threshold=0.5,
        )

    def test_a_correct_detection_counts_as_a_true_positive(
        self, orchestrator: CognitiveOrchestrator
    ) -> None:
        indicator = self._indicator(orchestrator)
        result = orchestrator.analyze(_ANOMALY, context={"domain": "cyber"})
        assert result.triggered_indicators == ["any anomaly"]

        report = orchestrator.learn_from_feedback(_analysis_id(result), was_correct=True)

        assert indicator.true_positive_count == 1
        assert indicator.false_positive_count == 0
        assert report["indicators_validated"] == [indicator.indicator_id]

    def test_a_wrong_detection_counts_as_a_false_positive(
        self, orchestrator: CognitiveOrchestrator
    ) -> None:
        indicator = self._indicator(orchestrator)
        result = orchestrator.analyze(_ANOMALY, context={"domain": "cyber"})

        orchestrator.learn_from_feedback(_analysis_id(result), was_correct=False)

        assert indicator.true_positive_count == 0
        assert indicator.false_positive_count == 1


class TestFeedbackMemoryIsBounded:
    def test_the_oldest_analyses_are_forgotten_first(
        self, orchestrator: CognitiveOrchestrator, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(CognitiveOrchestrator, "FEEDBACK_MEMORY_LIMIT", 3)
        ids = [
            _analysis_id(orchestrator.analyze(_ANOMALY, context={"domain": "cyber"}))
            for _ in range(5)
        ]

        assert len(orchestrator._feedback_records) == 3
        assert orchestrator.learn_from_feedback(ids[0], was_correct=True)["known"] is False
        assert orchestrator.learn_from_feedback(ids[-1], was_correct=True)["known"] is True

    def test_a_refused_request_leaves_no_feedback_record(
        self, orchestrator: CognitiveOrchestrator
    ) -> None:
        with pytest.raises(EthicalConstraintViolationError):
            orchestrator.analyze(
                _ANOMALY,
                context={
                    "domain": "cyber",
                    "note": (
                        "end-to-end procedure to synthesize sarin nerve agent and "
                        "weaponize it for mass casualties"
                    ),
                },
            )

        assert len(orchestrator._feedback_records) == 0


class TestObservationLinkIsReportedTruthfully:
    def _linked(self, orch: CognitiveOrchestrator, domain_node: str) -> bool:
        edges = orch.knowledge_graph._edges.get(f"obs_{orch._analysis_count}", [])
        return any(
            edge.target_id == domain_node and edge.edge_type == EdgeType.INSTANCE_OF
            for edge in edges
        )

    def test_a_seeded_domain_is_linked_and_said_to_be(
        self, orchestrator: CognitiveOrchestrator
    ) -> None:
        result = orchestrator.analyze(_ANOMALY, context={"domain": "cyber"})

        assert self._linked(orchestrator, "domain_cyber")
        assert result.knowledge_updates == ["Added observation to cyber domain"]

    def test_the_default_general_domain_is_really_linked(
        self, orchestrator: CognitiveOrchestrator
    ) -> None:
        """``general`` is a known domain the graph was never seeded with."""
        result = orchestrator.analyze(_ANOMALY, context={"domain": "general"})

        assert orchestrator.knowledge_graph.get_node("domain_general") is not None
        assert self._linked(orchestrator, "domain_general")
        assert result.knowledge_updates == ["Added observation to general domain"]

    def test_an_unknown_domain_is_not_claimed_as_added(
        self, orchestrator: CognitiveOrchestrator
    ) -> None:
        result = orchestrator.analyze(_ANOMALY, context={"domain": "no_such_domain"})

        assert orchestrator.knowledge_graph.get_node("domain_no_such_domain") is None
        assert not self._linked(orchestrator, "domain_no_such_domain")
        assert not any(
            update.startswith("Added observation") for update in result.knowledge_updates
        )
        assert any("not linked" in update for update in result.knowledge_updates)
