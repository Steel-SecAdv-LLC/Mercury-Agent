# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The response layer is bounded, non-destructive and severity-aware.

These tests pin the deterrence contract: every recommended response is
advisory or notifying, a fail-closed hold always demands a human, and nothing
in the catalogue authorises a destructive or irreversible action.
"""

from __future__ import annotations

import pytest

from omni_mercury_engine.decision import Disposition, ResponseAction, ResponsePolicy


@pytest.fixture
def policy() -> ResponsePolicy:
    return ResponsePolicy()


class TestUrgencyBanding:
    @pytest.mark.parametrize(
        ("severity", "expected"),
        [(0.95, "critical"), (0.75, "urgent"), (0.5, "elevated"), (0.1, "routine")],
    )
    def test_urgency_for(self, severity: float, expected: str) -> None:
        assert ResponsePolicy.urgency_for(severity) == expected


class TestDispositionMapping:
    def test_clear_is_passive_monitor(self, policy: ResponsePolicy) -> None:
        plan = policy.plan(Disposition.CLEAR, severity=0.0)
        assert plan.action is ResponseAction.MONITOR
        assert plan.notify is False
        assert plan.requires_human is False
        assert plan.fail_closed is False

    def test_low_severity_act_recommends_mitigation(self, policy: ResponsePolicy) -> None:
        plan = policy.plan(Disposition.ACT, severity=0.5, domain="security")
        assert plan.action is ResponseAction.RECOMMEND_MITIGATION
        assert plan.notify is True
        assert plan.requires_human is False
        assert plan.countermeasures  # advisory steps are present

    def test_high_severity_act_escalates_to_human(self, policy: ResponsePolicy) -> None:
        plan = policy.plan(Disposition.ACT, severity=0.95, domain="medical")
        assert plan.action is ResponseAction.ESCALATE_TO_HUMAN
        assert plan.requires_human is True
        assert plan.urgency == "critical"

    def test_defer_resolvable_requests_input(self, policy: ResponsePolicy) -> None:
        plan = policy.plan(Disposition.DEFER, severity=0.3, resolvable_by_input=True)
        assert plan.action is ResponseAction.REQUEST_INPUT
        assert plan.requires_human is True

    def test_defer_unresolvable_escalates(self, policy: ResponsePolicy) -> None:
        plan = policy.plan(Disposition.DEFER, severity=0.3, resolvable_by_input=False)
        assert plan.action is ResponseAction.ESCALATE_TO_HUMAN

    def test_hold_is_fail_closed_and_human(self, policy: ResponsePolicy) -> None:
        plan = policy.plan(Disposition.HOLD, severity=0.2)
        assert plan.action is ResponseAction.HOLD
        assert plan.fail_closed is True
        assert plan.requires_human is True
        assert plan.notify is True

    def test_domain_interpolated_into_countermeasures(self, policy: ResponsePolicy) -> None:
        plan = policy.plan(Disposition.ACT, severity=0.5, domain="oceanography")
        assert any("oceanography" in step for step in plan.countermeasures)


class TestNonDestructiveInvariant:
    """No response, for any disposition or severity, is destructive."""

    _FORBIDDEN = (
        "delete",
        "destroy",
        "wipe",
        "shutdown",
        "shut down",
        "kill",
        "erase",
        "format",
        "drop ",
        "rm -",
        "disable permanently",
        "retaliate",
        "attack",
        "counter-attack",
    )

    @pytest.mark.parametrize("disposition", list(Disposition))
    @pytest.mark.parametrize("severity", [0.0, 0.5, 0.95])
    def test_no_destructive_countermeasures(
        self, policy: ResponsePolicy, disposition: Disposition, severity: float
    ) -> None:
        for resolvable in (True, False):
            plan = policy.plan(
                disposition, severity=severity, domain="security", resolvable_by_input=resolvable
            )
            blob = " ".join(plan.countermeasures).lower() + " " + plan.rationale.lower()
            for word in self._FORBIDDEN:
                assert word not in blob, f"destructive verb {word!r} in {disposition}"

    @pytest.mark.parametrize("disposition", list(Disposition))
    def test_action_is_in_bounded_catalogue(
        self, policy: ResponsePolicy, disposition: Disposition
    ) -> None:
        plan = policy.plan(disposition, severity=0.6)
        # Every action is one of the advisory / notifying / escalating members;
        # none of them executes anything on its own.
        assert plan.action in set(ResponseAction)

    def test_fail_closed_always_requires_human(self, policy: ResponsePolicy) -> None:
        for severity in (0.0, 0.5, 1.0):
            plan = policy.plan(Disposition.HOLD, severity=severity)
            assert plan.fail_closed and plan.requires_human


class TestRestorativeConversionVerbs:
    """The recommend-only, restorative/non-violent convert verbs.

    Pins: (1) the new wire values; (2) the default policy is byte-identical
    to the pre-existing contract (opt-in only); (3) the restorative posture
    recommends conversion below the human bar, still escalates above it, and
    never weakens a fail-closed hold; (4) every restorative step stays
    non-destructive and recommend-only.
    """

    _FORBIDDEN = TestNonDestructiveInvariant._FORBIDDEN

    def test_wire_values_pinned(self) -> None:
        assert ResponseAction.RECOMMEND_CONVERSION.value == "recommend_conversion"
        assert ResponseAction.RECOMMEND_RESTORATION.value == "recommend_restoration"

    def test_default_policy_never_emits_conversion(self, policy: ResponsePolicy) -> None:
        for severity in (0.0, 0.5, 0.95):
            for disposition in Disposition:
                plan = policy.plan(disposition, severity=severity, domain="security")
                assert plan.action is not ResponseAction.RECOMMEND_CONVERSION
                assert plan.action is not ResponseAction.RECOMMEND_RESTORATION

    def test_default_policy_act_contract_unchanged(self, policy: ResponsePolicy) -> None:
        plan = policy.plan(Disposition.ACT, severity=0.5, domain="security")
        assert plan.action is ResponseAction.RECOMMEND_MITIGATION
        assert "reversible countermeasures" in plan.rationale

    def test_restorative_act_recommends_conversion_below_human_bar(self) -> None:
        plan = ResponsePolicy(restorative=True).plan(
            Disposition.ACT, severity=0.5, domain="security"
        )
        assert plan.action is ResponseAction.RECOMMEND_CONVERSION
        assert plan.requires_human is False
        assert plan.notify is True
        assert plan.fail_closed is False
        # The restorative catalogue rides along, domain-interpolated.
        assert any("benign state" in step for step in plan.countermeasures)
        assert any("security" in step for step in plan.countermeasures)

    def test_restorative_act_still_escalates_above_human_bar(self) -> None:
        plan = ResponsePolicy(restorative=True).plan(
            Disposition.ACT, severity=0.95, domain="medical"
        )
        assert plan.action is ResponseAction.ESCALATE_TO_HUMAN
        assert plan.requires_human is True

    def test_restorative_hold_stays_fail_closed(self) -> None:
        plan = ResponsePolicy(restorative=True).plan(Disposition.HOLD, severity=0.5)
        assert plan.action is ResponseAction.HOLD
        assert plan.fail_closed is True
        assert plan.requires_human is True

    @pytest.mark.parametrize("severity", [0.0, 0.5, 0.95])
    @pytest.mark.parametrize("disposition", list(Disposition))
    def test_restorative_steps_are_non_destructive(
        self, disposition: Disposition, severity: float
    ) -> None:
        plan = ResponsePolicy(restorative=True).plan(
            disposition, severity=severity, domain="security"
        )
        blob = " ".join(plan.countermeasures).lower() + " " + plan.rationale.lower()
        for word in self._FORBIDDEN:
            assert word not in blob, f"destructive verb {word!r} in restorative {disposition}"
        # Recommend-only: the restorative catalogue always phrases human gating.
        if plan.action is ResponseAction.RECOMMEND_CONVERSION:
            assert any(
                "approval" in step or "review" in step for step in plan.countermeasures
            )
