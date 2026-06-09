"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""The closed loop end to end: decide -> deter -> verify, plus the audit ledger.

Pure-Python tier (no torch): the loop consumes :class:`ConfidenceSignal` inputs,
so the orchestration, ledger, and feedback seam are exercisable everywhere.
"""

import json

from omni_mercury_engine.core.types import ThreatLevel
from omni_mercury_engine.decision.confidence import ConfidenceSignal
from omni_mercury_engine.decision.loop import AuditLedger, DecisionResponseLoop
from omni_mercury_engine.decision.response import Authorization, deny_all_gate, permit_all_gate
from omni_mercury_engine.decision.types import LoopResult, ResponseStatus, Verdict


def _loop(**kwargs: object) -> DecisionResponseLoop:
    return DecisionResponseLoop(ethical_gate=permit_all_gate, **kwargs)  # type: ignore[arg-type]


class TestLoopStep:
    def test_grounded_positive_high_escalates_and_defers(self) -> None:
        loop = _loop()
        result = loop.step(
            ConfidenceSignal(0.99, prediction_set=(1,), coverage=0.9),
            domain="network_security",
            severity=ThreatLevel.CRITICAL,
        )
        assert result.decision.verdict is Verdict.POSITIVE
        assert result.response.status is ResponseStatus.DEFERRED

    def test_grounded_positive_with_authorization_applies(self) -> None:
        loop = _loop()
        result = loop.step(
            ConfidenceSignal(0.99, prediction_set=(1,), coverage=0.9),
            severity=ThreatLevel.CRITICAL,
            authorization=Authorization(authority="ops", reason="drill"),
        )
        assert result.response.status is ResponseStatus.APPLIED

    def test_abstain_path_gathers_evidence(self) -> None:
        loop = _loop()
        result = loop.step(ConfidenceSignal(0.5, prediction_set=(0, 1), coverage=0.9))
        assert result.decision.abstained
        assert result.response.status is ResponseStatus.APPLIED  # gather-evidence is safe
        assert result.three_state.value == "unavailable"

    def test_negative_derives_none_severity(self) -> None:
        loop = _loop()
        result = loop.step(ConfidenceSignal(0.02, prediction_set=(0,), coverage=0.9))
        assert result.decision.verdict is Verdict.NEGATIVE
        assert result.provenance["severity"] == ThreatLevel.NONE.name

    def test_deny_gate_blocks_response(self) -> None:
        loop = DecisionResponseLoop(ethical_gate=deny_all_gate)
        result = loop.step(
            ConfidenceSignal(0.95, prediction_set=(1,), coverage=0.9),
            severity=ThreatLevel.LOW,
        )
        assert result.response.status is ResponseStatus.BLOCKED


class TestEngineResultStep:
    def test_step_from_engine_result_end_to_end(self) -> None:
        loop = _loop()
        result = loop.step_from_engine_result(
            {
                "anomaly_prob": 0.96,
                "severity": 0.2,
                "conformal": {
                    "prediction_set": [1],
                    "set_size": 1,
                    "abstain": False,
                    "coverage": 0.9,
                },
            },
            domain="energy",
        )
        assert result.decision.verdict is Verdict.POSITIVE
        # severity 0.2 -> LOW -> notify, reversible, auto-applied under permit gate.
        assert result.response.status is ResponseStatus.APPLIED

    def test_severity_taken_from_result_field(self) -> None:
        loop = _loop()
        result = loop.step_from_engine_result({"anomaly_prob": 0.95, "severity": 0.95}, domain="x")
        assert result.provenance["severity"] == ThreatLevel.CRITICAL.name


class TestAuditLedger:
    def test_records_every_pass(self) -> None:
        loop = _loop()
        loop.step(ConfidenceSignal(0.95, prediction_set=(1,), coverage=0.9))
        loop.step(ConfidenceSignal(0.5, prediction_set=(0, 1), coverage=0.9))
        assert len(loop.ledger) == 2

    def test_summary_counts_and_abstention_rate(self) -> None:
        loop = _loop()
        loop.step(ConfidenceSignal(0.95, prediction_set=(1,), coverage=0.9))
        loop.step(ConfidenceSignal(0.5, prediction_set=(0, 1), coverage=0.9))
        loop.step(ConfidenceSignal(0.02, prediction_set=(0,), coverage=0.9))
        summary = loop.ledger.summary()
        assert summary["total"] == 3
        assert summary["abstention_rate"] == 1 / 3
        assert summary["by_verdict"]["abstain"] == 1

    def test_to_list_is_json_serializable(self) -> None:
        loop = _loop()
        loop.step(ConfidenceSignal(0.95, prediction_set=(1,), coverage=0.9))
        json.dumps(loop.ledger.to_list())

    def test_shared_ledger_across_loops(self) -> None:
        ledger = AuditLedger()
        first = DecisionResponseLoop(ethical_gate=permit_all_gate, ledger=ledger)
        second = DecisionResponseLoop(ethical_gate=permit_all_gate, ledger=ledger)
        first.step(ConfidenceSignal(0.95, prediction_set=(1,), coverage=0.9))
        second.step(ConfidenceSignal(0.02, prediction_set=(0,), coverage=0.9))
        assert len(ledger) == 2

    def test_empty_ledger_summary(self) -> None:
        assert AuditLedger().summary()["abstention_rate"] == 0.0


class TestFeedbackSeam:
    def test_feedback_sink_receives_each_result(self) -> None:
        captured: list[LoopResult] = []
        loop = DecisionResponseLoop(ethical_gate=permit_all_gate, feedback=captured.append)
        loop.step(ConfidenceSignal(0.95, prediction_set=(1,), coverage=0.9))
        loop.step(ConfidenceSignal(0.5, prediction_set=(0, 1), coverage=0.9))
        assert len(captured) == 2
        assert captured[1].decision.abstained
