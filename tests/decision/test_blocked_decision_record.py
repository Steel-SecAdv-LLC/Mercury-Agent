# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every ethically blocked decision carries a full, reconstructable record.

Before this, the structured 403 payload held only the refusal report: the
gates raise before any DecisionRecord exists, so the decider's fail-closed
ethical Stage 1 was unreachable from serving and a blocked decision left no
record.  These tests pin the closure: the refusal payload now carries a
complete UNDECIDABLE/HOLD record whose ``signals.ethics_report`` is the same
structured report, JSON-safe end to end.
"""

from __future__ import annotations

import json

import pytest

from omni_mercury_engine.decision.decider import DecisionAbstentionResponder
from omni_mercury_engine.decision.evidence import Evidence
from omni_mercury_engine.decision.states import Disposition
from omni_mercury_engine.verifiers.three_state import ThreeState

ETHICS = {
    "check": "harm_uplift",
    "score": 0.12,
    "threshold": 0.5,
    "ruleset_version": 9,
    "details": {"matched": "an operational-uplift pattern"},
}


class TestEvidenceFromEthicalBlock:
    def test_builds_fail_closed_hold(self) -> None:
        record = DecisionAbstentionResponder().decide_from_evidence(
            Evidence.from_ethical_block(ETHICS)
        )
        assert record.state is ThreeState.UNDECIDABLE
        assert record.disposition is Disposition.HOLD
        assert record.abstained is True
        assert record.anomaly_prob is None

    def test_record_carries_the_full_report(self) -> None:
        record = DecisionAbstentionResponder().decide_from_evidence(
            Evidence.from_ethical_block(ETHICS)
        )
        signals = record.signals
        assert signals["ethical_gate_passed"] is False
        assert signals["ethics_report"] == ETHICS
        assert signals["ethical_score"] == pytest.approx(0.12)
        assert signals["ethical_threshold"] == pytest.approx(0.5)

    def test_record_is_json_safe(self) -> None:
        record = DecisionAbstentionResponder().decide_from_evidence(
            Evidence.from_ethical_block(ETHICS)
        )
        json.dumps(record.to_dict())

    def test_report_absent_on_ordinary_evidence(self) -> None:
        record = DecisionAbstentionResponder().decide_from_evidence(
            Evidence(anomaly_prob=0.9, is_anomaly=True, threshold=0.5, severity=0.5)
        )
        assert record.signals["ethics_report"] is None


class TestStructured403CarriesTheRecord:
    def test_refusal_detail_carries_decision(self) -> None:
        pytest.importorskip("fastapi", reason="the API routes need the [api] extra")
        from omni_mercury_engine.api.routes.detection import _ethical_refusal_detail

        class _Refusal(Exception):
            check = "sigma_immutable"
            score = 0.2
            threshold = 0.6
            details = {"vector": "sub-threshold"}

        detail = _ethical_refusal_detail(_Refusal(), "Detection blocked.")
        assert detail["ethics"]["check"] == "sigma_immutable"
        decision = detail["decision"]
        assert decision["state"] == "undecidable"
        assert decision["disposition"] == "hold"
        assert decision["abstained"] is True
        assert decision["signals"]["ethics_report"] == detail["ethics"]
        json.dumps(detail)
