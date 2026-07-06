# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The bridge carries a decision into the existing autonomy + alerting channels."""

from __future__ import annotations

from typing import Any

import pytest

from omni_mercury_engine.decision import (
    DecisionAbstentionResponder,
    DecisionRecord,
    Disposition,
    to_agent_action,
    to_cap_alert,
)


def _decide(**over: Any) -> DecisionRecord:
    base: dict[str, Any] = {"anomaly_prob": 0.5, "is_anomaly": False, "threshold_used": 0.5}
    base.update(over)
    return DecisionAbstentionResponder().decide(base, domain=over.pop("domain", "security"))


class TestAgentActionBridge:
    def test_grounded_anomaly_maps_to_flag(self) -> None:
        rec = _decide(
            anomaly_prob=0.92,
            is_anomaly=True,
            severity=0.5,
            conformal={"prediction_set": [1], "set_size": 1, "abstain": False, "coverage": 0.9},
        )
        action = to_agent_action(rec)
        assert action.action_type == "flag_anomaly"
        assert action.confidence == pytest.approx(0.9)  # the coverage-grounded confidence
        assert action.parameters["decision"]["state"] == "grounded"
        assert action.rationale  # carries the human-readable explanation

    def test_abstention_maps_to_escalate_with_raw_prob_confidence(self) -> None:
        rec = _decide(
            anomaly_prob=0.55,
            conformal={"prediction_set": [0, 1], "set_size": 2, "abstain": True, "coverage": 0.9},
        )
        action = to_agent_action(rec)
        assert action.action_type == "escalate"
        # Abstained -> no grounded confidence, fall back to the raw probability.
        assert action.confidence == pytest.approx(0.55)

    def test_clear_maps_to_log(self) -> None:
        rec = _decide(
            anomaly_prob=0.02,
            conformal={"prediction_set": [0], "set_size": 1, "abstain": False, "coverage": 0.9},
        )
        assert to_agent_action(rec).action_type == "log"

    def test_every_disposition_has_an_action_type(self) -> None:
        from omni_mercury_engine.decision.bridge import _DISPOSITION_TO_ACTION_TYPE

        assert set(_DISPOSITION_TO_ACTION_TYPE) == set(Disposition)


class TestCapAlertBridge:
    def test_notifying_decision_emits_cap_xml(self) -> None:
        rec = _decide(
            anomaly_prob=0.95,
            is_anomaly=True,
            severity=0.9,
            conformal={"prediction_set": [1], "set_size": 1, "abstain": False, "coverage": 0.9},
        )
        xml = to_cap_alert(rec, area_description="Sector 7")
        assert xml is not None
        assert "<cap:alert" in xml or "<alert" in xml
        # Decision provenance is threaded into the alert parameters.
        assert "grounded" in xml

    def test_monitor_decision_emits_no_alert(self) -> None:
        rec = _decide(
            anomaly_prob=0.02,
            conformal={"prediction_set": [0], "set_size": 1, "abstain": False, "coverage": 0.9},
        )
        assert rec.response.notify is False
        assert to_cap_alert(rec) is None

    def test_rca_causes_attached_as_alert_attribution(self) -> None:
        rec = _decide(
            anomaly_prob=0.95,
            is_anomaly=True,
            severity=0.9,
            conformal={"prediction_set": [1], "set_size": 1, "abstain": False, "coverage": 0.9},
        )
        xml = to_cap_alert(rec, area_description="Sector 7", rca_causes=[(3, 0.42), (1, 0.31)])
        assert xml is not None
        # Ranked root causes ride along as attribution for on-call triage.
        assert "RootCauses" in xml
        assert "node3" in xml
