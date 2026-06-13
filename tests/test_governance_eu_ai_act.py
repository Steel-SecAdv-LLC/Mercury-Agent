# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the EU AI Act risk tier gate -- a tag, never a registered scalar."""

from __future__ import annotations

from omni_mercury_engine.governance import eu_ai_act
from omni_mercury_engine.governance.contract import GOVERNANCE_FAMILY_VET, SignalClass
from omni_mercury_engine.governance.eu_ai_act import EuAiActTag, EuAiActTier


def test_highest_applicable_tier_wins() -> None:
    """Prohibited > high > limited > minimal; the most severe declared flag decides."""
    assert (
        eu_ai_act.eu_ai_act_tier({"prohibited_practice": True, "annex_iii_high_risk": True}).tier
        is EuAiActTier.UNACCEPTABLE
    )
    assert eu_ai_act.eu_ai_act_tier({"annex_iii_high_risk": True}).tier is EuAiActTier.HIGH
    assert eu_ai_act.eu_ai_act_tier({"transparency_obligation": True}).tier is EuAiActTier.LIMITED
    assert eu_ai_act.eu_ai_act_tier({}).tier is EuAiActTier.MINIMAL


def test_tag_never_registers() -> None:
    """An EU AI Act result is a tag (registers=False), not a GovernanceScalar."""
    tag = eu_ai_act.eu_ai_act_tier({"annex_iii_high_risk": True})
    assert isinstance(tag, EuAiActTag)
    assert tag.registers is False
    assert tag.family == "eu_ai_act"


def test_module_exposes_no_scalar_builder() -> None:
    """By design there is no ``*_scalar`` function -- the tier is never a scalar."""
    assert not [n for n in dir(eu_ai_act) if n.endswith("_scalar")]


def test_vet_marks_eu_ai_act_tag_only() -> None:
    """The vet records EU AI Act as TAG_ONLY (a gate/tag, never a scalar)."""
    assert GOVERNANCE_FAMILY_VET["eu_ai_act"].classification is SignalClass.TAG_ONLY
