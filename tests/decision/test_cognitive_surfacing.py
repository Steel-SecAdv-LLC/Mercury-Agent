# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The orchestrator's self-critique reaches the served decision record, bounded.

Before this, ``cognitive_analysis`` sat at the detection result's top level
only: the record's signals carried none of it, no production surface enabled
the orchestrator, and its knowledge graph grew one observation node per
anomalous request without limit.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.decision.decider import DecisionAbstentionResponder
from omni_mercury_engine.decision.evidence import Evidence
from omni_mercury_engine.decision.record import DecisionRecord


def _result(cognitive: Any) -> dict[str, Any]:
    return {
        "anomaly_prob": 0.97,
        "is_anomaly": True,
        "threshold_used": 0.5,
        "conformal": {"prediction_set": [1], "set_size": 1, "coverage": 0.9},
        "cognitive_analysis": cognitive,
    }


class TestCognitiveSummaryOnTheRecord:
    def test_critique_and_analogs_ride_the_record_bounded(self) -> None:
        analysis = {
            "is_reliable": False,
            "confidence": 0.41,
            "epistemic_uncertainty": 0.3,
            "aleatoric_uncertainty": 0.1,
            "ethical_permissible": True,
            "similar_cases": [f"case_{i}" * 60 for i in range(9)],
            "recommendations": ["isolate the host", "", "rotate keys"] + ["x"] * 9,
            "reasoning_chain": ["never" * 100],
            "knowledge_updates": ["never"],
        }
        record = DecisionAbstentionResponder().decide(_result(analysis))
        cognitive = record.signals["cognitive"]

        assert cognitive["is_reliable"] is False
        assert cognitive["confidence"] == pytest.approx(0.41)
        assert cognitive["epistemic_uncertainty"] == pytest.approx(0.3)
        assert cognitive["ethical_permissible"] is True
        assert len(cognitive["similar_cases"]) == 5
        assert all(len(case) <= 160 for case in cognitive["similar_cases"])
        # Empty strings are dropped; the list is truncated at five.
        assert cognitive["recommendations"][:2] == ["isolate the host", "rotate keys"]
        assert len(cognitive["recommendations"]) == 5
        # The raw analysis never rides the record.
        flattened = json.dumps(record.to_dict())
        assert "reasoning_chain" not in flattened
        assert "knowledge_updates" not in flattened
        # And it round-trips.
        again = DecisionRecord.from_dict(json.loads(flattened))
        assert again.signals["cognitive"] == cognitive

    def test_without_the_orchestrator_the_field_is_none(self) -> None:
        result = _result(None)
        del result["cognitive_analysis"]
        record = DecisionAbstentionResponder().decide(result)
        assert record.signals["cognitive"] is None
        # Old records without the key still load.
        payload = json.loads(json.dumps(record.to_dict()))
        payload["signals"].pop("cognitive")
        assert DecisionRecord.from_dict(payload).signals.get("cognitive") is None

    def test_malformed_analysis_reads_as_absent_fields(self) -> None:
        record = DecisionAbstentionResponder().decide(
            _result({"confidence": "high", "similar_cases": "not-a-list"})
        )
        cognitive = record.signals["cognitive"]
        assert cognitive["confidence"] is None
        assert cognitive["similar_cases"] == []
        assert cognitive["is_reliable"] is None

    def test_direct_evidence_defaults_to_none(self) -> None:
        evidence = Evidence(anomaly_prob=0.5, is_anomaly=False, threshold=0.5, severity=0.0)
        assert evidence.cognitive is None
        assert evidence.to_dict()["cognitive"] is None


class TestServedFlagshipEnablesTheOrchestrator:
    def _fresh_engine(self, monkeypatch: pytest.MonkeyPatch, value: str | None) -> Any:
        pytest.importorskip("torch")
        from omni_mercury_engine.api.routes import detection as det

        if value is None:
            monkeypatch.delenv("MERCURY_FLAGSHIP_COGNITIVE", raising=False)
        else:
            monkeypatch.setenv("MERCURY_FLAGSHIP_COGNITIVE", value)
        monkeypatch.setattr(det, "_flagship_engine", None)
        with det._flagship_lock:
            return det._served_flagship_engine()

    def test_on_by_default_and_surfaced_end_to_end(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = self._fresh_engine(monkeypatch, None)
        assert engine.cognitive_orchestrator is not None

        result = engine.detect_with_fusion(np.zeros((1, 16), dtype=np.float32))
        assert "cognitive_analysis" in result
        cognitive = result["decision"]["signals"]["cognitive"]
        assert cognitive is not None
        assert isinstance(cognitive["is_reliable"], bool)
        assert cognitive["confidence"] is not None

    def test_opt_out_disables_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = self._fresh_engine(monkeypatch, "off")
        assert engine.cognitive_orchestrator is None


class TestBoundedObservationMemory:
    def test_the_knowledge_graph_does_not_grow_past_the_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One observation node per anomaly, oldest evicted past the limit."""
        from omni_mercury_engine.cognitive.orchestrator import CognitiveOrchestrator

        monkeypatch.setattr(CognitiveOrchestrator, "OBSERVATION_MEMORY_LIMIT", 6)
        orchestrator = CognitiveOrchestrator(
            enable_causal=False, enable_ipb=False, enable_indicators=False
        )
        rng = np.random.default_rng(3)
        baseline = len(orchestrator.knowledge_graph._nodes)

        counts = []
        for i in range(20):
            orchestrator.analyze(
                detection_result={
                    "is_anomaly": True,
                    "anomaly_prob": 0.95,
                    "severity": 0.8,
                },
                raw_data=rng.normal(size=(8, 4)),
                context={"domain": "security"},
            )
            counts.append(len(orchestrator.knowledge_graph._nodes))

        # Grows while filling, then holds: once the limit is reached no
        # further analysis grows the graph, and the oldest observation is out.
        assert counts[-1] == counts[8]
        assert counts[-1] - baseline <= 6 + 2  # observations + the domain node
        assert orchestrator.knowledge_graph.get_node("obs_1") is None
        assert orchestrator.knowledge_graph.get_node("obs_20") is not None

    def test_remove_node_purges_edges_both_ways(self) -> None:
        from omni_mercury_engine.cognitive.knowledge_graph import (
            EdgeType,
            KnowledgeGraph,
            NodeType,
        )

        graph = KnowledgeGraph(enable_embeddings=False)
        for name in ("a", "b", "c"):
            graph.add_node(name, NodeType.CONCEPT, name)
        graph.add_edge("a", "b", EdgeType.CAUSES)
        graph.add_edge("c", "a", EdgeType.CAUSES)

        assert graph.remove_node("a") is True
        assert graph.remove_node("a") is False
        assert graph.get_node("a") is None
        assert graph.get_neighbors("c", direction="outgoing") == []
        assert graph.get_neighbors("b", direction="incoming") == []
        assert graph._edges.get("c", []) == [] and graph._reverse_edges.get("b", []) == []
