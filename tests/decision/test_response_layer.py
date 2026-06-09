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

"""The response ('deter') layer: proportionate, reversible-by-default, fail-closed.

Pure-Python tier (no torch): the planner and actuator are exercised against
synthetic decisions so the entire safety contract is verifiable everywhere.
"""

from typing import TYPE_CHECKING

import pytest

from omni_mercury_engine.core.types import ThreatLevel
from omni_mercury_engine.decision.response import (
    Authorization,
    ResponseActuator,
    ResponsePlanner,
    ResponseVetoError,
    deny_all_gate,
    permit_all_gate,
    threat_level_from_score,
)
from omni_mercury_engine.decision.types import (
    Decision,
    ResponseAction,
    ResponseStatus,
    ResponseTier,
    Verdict,
    verdict_to_three_state,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def _decision(verdict: Verdict, confidence: float = 0.9, novelty: bool = False) -> Decision:
    return Decision(
        verdict=verdict,
        state=verdict_to_three_state(verdict),
        confidence=confidence,
        margin=0.5,
        reason="synthetic",
        policy="test",
        novelty=novelty,
    )


class TestPlannerProportionality:
    """``(verdict, severity)`` selects a single proportionate action."""

    def test_grounded_negative_monitors(self) -> None:
        action = ResponsePlanner().plan(_decision(Verdict.NEGATIVE), severity=ThreatLevel.NONE)
        assert action.tier is ResponseTier.MONITOR
        assert action.reversible and not action.requires_human_authorization

    @pytest.mark.parametrize(
        ("severity", "tier", "auth"),
        [
            (ThreatLevel.LOW, ResponseTier.NOTIFY, False),
            (ThreatLevel.MODERATE, ResponseTier.NOTIFY, False),
            (ThreatLevel.SUBSTANTIAL, ResponseTier.SOFT_CONTAIN, False),
            (ThreatLevel.HIGH, ResponseTier.SOFT_CONTAIN, False),
            (ThreatLevel.SEVERE, ResponseTier.ESCALATE, True),
            (ThreatLevel.CRITICAL, ResponseTier.ESCALATE, True),
        ],
    )
    def test_positive_ladder(self, severity: ThreatLevel, tier: ResponseTier, auth: bool) -> None:
        action = ResponsePlanner().plan(_decision(Verdict.POSITIVE), severity=severity)
        assert action.tier is tier
        assert action.requires_human_authorization is auth

    def test_soft_contain_is_reversible(self) -> None:
        action = ResponsePlanner().plan(_decision(Verdict.POSITIVE), severity=ThreatLevel.HIGH)
        assert action.reversible

    @pytest.mark.parametrize(
        "severity",
        [ThreatLevel.NONE, ThreatLevel.LOW, ThreatLevel.HIGH, ThreatLevel.CRITICAL],
    )
    def test_abstention_never_yields_a_deterrent(self, severity: ThreatLevel) -> None:
        action = ResponsePlanner().plan(_decision(Verdict.ABSTAIN), severity=severity)
        assert action.tier in (ResponseTier.GATHER_EVIDENCE, ResponseTier.NOTIFY)
        assert action.tier not in (ResponseTier.SOFT_CONTAIN, ResponseTier.ESCALATE)
        assert action.reversible


class TestActuatorContract:
    """The actuator owns gating, authorization, and reversibility decisions."""

    def test_none_tier_is_noop(self) -> None:
        action = ResponseAction(
            name="x",
            tier=ResponseTier.NONE,
            severity=ThreatLevel.NONE,
            reversible=True,
            requires_human_authorization=False,
            rationale="",
        )
        outcome = ResponseActuator(permit_all_gate).actuate(action, _decision(Verdict.NEGATIVE))
        assert outcome.status is ResponseStatus.NOOP

    def test_gate_veto_blocks(self) -> None:
        planner = ResponsePlanner()
        action = planner.plan(_decision(Verdict.POSITIVE), severity=ThreatLevel.HIGH)
        outcome = ResponseActuator(deny_all_gate).actuate(action, _decision(Verdict.POSITIVE))
        assert outcome.status is ResponseStatus.BLOCKED
        assert outcome.ethical_gate_passed is False

    def test_custom_raising_gate_blocks(self) -> None:
        def gate(*, action: ResponseAction, decision: Decision, domain: str | None) -> None:
            raise ResponseVetoError("policy says no")

        planner = ResponsePlanner()
        action = planner.plan(_decision(Verdict.POSITIVE), severity=ThreatLevel.MODERATE)
        outcome = ResponseActuator(gate).actuate(action, _decision(Verdict.POSITIVE))
        assert outcome.status is ResponseStatus.BLOCKED

    def test_escalate_defers_without_authorization(self) -> None:
        action = ResponsePlanner().plan(_decision(Verdict.POSITIVE), severity=ThreatLevel.CRITICAL)
        outcome = ResponseActuator(permit_all_gate).actuate(action, _decision(Verdict.POSITIVE))
        assert outcome.status is ResponseStatus.DEFERRED
        assert outcome.ethical_gate_passed is True

    def test_escalate_applies_with_authorization(self) -> None:
        action = ResponsePlanner().plan(_decision(Verdict.POSITIVE), severity=ThreatLevel.CRITICAL)
        outcome = ResponseActuator(permit_all_gate).actuate(
            action,
            _decision(Verdict.POSITIVE),
            authorization=Authorization(authority="operator:7", reason="confirmed incident"),
        )
        assert outcome.status is ResponseStatus.APPLIED
        assert outcome.provenance["authorization"]["authority"] == "operator:7"

    def test_reversible_low_severity_applies(self) -> None:
        action = ResponsePlanner().plan(_decision(Verdict.POSITIVE), severity=ThreatLevel.LOW)
        outcome = ResponseActuator(permit_all_gate).actuate(action, _decision(Verdict.POSITIVE))
        assert outcome.status is ResponseStatus.APPLIED

    def test_abstention_deter_guard_defers(self) -> None:
        # Defence in depth: even if a deterrent action is handed in for an
        # abstention, the actuator refuses to actuate it.
        deterrent = ResponseAction(
            name="soft_contain.quarantine_review",
            tier=ResponseTier.SOFT_CONTAIN,
            severity=ThreatLevel.HIGH,
            reversible=True,
            requires_human_authorization=False,
            rationale="(should never run on an abstention)",
        )
        outcome = ResponseActuator(permit_all_gate).actuate(deterrent, _decision(Verdict.ABSTAIN))
        assert outcome.status is ResponseStatus.DEFERRED
        assert outcome.provenance["guard"] == "abstention_no_deter"

    def test_custom_handler_is_invoked(self) -> None:
        seen: dict[str, object] = {}

        def handler(action: ResponseAction, context: Mapping[str, object]) -> dict[str, object]:
            seen["name"] = action.name
            return {"ran": True}

        actuator = ResponseActuator(permit_all_gate)
        action = ResponsePlanner().plan(_decision(Verdict.POSITIVE), severity=ThreatLevel.LOW)
        actuator.register_handler(action.name, handler)
        outcome = actuator.actuate(action, _decision(Verdict.POSITIVE))
        assert outcome.status is ResponseStatus.APPLIED
        assert seen["name"] == action.name
        assert outcome.provenance["effect"] == {"ran": True}


class TestThreatLevelMapping:
    @pytest.mark.parametrize(
        ("score", "level"),
        [
            (0.0, ThreatLevel.NONE),
            (0.05, ThreatLevel.NONE),
            (0.2, ThreatLevel.LOW),
            (0.3, ThreatLevel.MODERATE),
            (0.5, ThreatLevel.SUBSTANTIAL),
            (0.65, ThreatLevel.HIGH),
            (0.8, ThreatLevel.SEVERE),
            (0.95, ThreatLevel.CRITICAL),
        ],
    )
    def test_score_to_level(self, score: float, level: ThreatLevel) -> None:
        assert threat_level_from_score(score) is level

    def test_clamps_out_of_range(self) -> None:
        assert threat_level_from_score(2.0) is ThreatLevel.CRITICAL
        assert threat_level_from_score(-1.0) is ThreatLevel.NONE
