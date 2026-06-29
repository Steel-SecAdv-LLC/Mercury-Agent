# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Structural tests for the Mercury subagent pantheon roster.

Pins the org-chart invariants of the 33-member fleet: unique pantheon ids, valid
depth labels, real subsystem bindings (every subsystem resolves to a live
``omni_mercury_engine`` module/package), Omni-Code anchors drawn from the Seven,
exactly seven code-bearers (one per Code, anchor-consistent), the autonomy-from-
anchor binding, and the hard constraint that **no "memorial code" terminology
exists anywhere in the subsystem** (Omni-Codes only).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import omni_mercury_engine.agentic.subagents as subagents_pkg
from omni_mercury_engine.agentic.subagents.base import anchor_autonomy, resolve_anchor
from omni_mercury_engine.agentic.subagents.roster import (
    ALL_ENTRIES,
    GENERALIST_FLOOR,
    ROSTER,
    code_bearers,
    entry_by_id,
    validate_roster,
)
from omni_mercury_engine.utils.constants import OmniCodes


def test_roster_validates() -> None:
    # Single call asserting every structural invariant (unique ids, real
    # subsystem resolution, valid anchors, 7 code-bearers, depth/impl coherence).
    validate_roster()


def test_exactly_33_public_members_plus_internal_floor() -> None:
    assert len(ROSTER) == 33
    assert GENERALIST_FLOOR.internal is True
    assert GENERALIST_FLOOR not in ROSTER
    assert all(not e.internal for e in ROSTER)


def test_all_ids_unique_and_roman_numeral_form() -> None:
    ids = [e.id for e in ALL_ENTRIES]
    assert len(ids) == len(set(ids))
    # Every public member is "<Name>_<RomanNumeral>".
    roman = set("IVXLC")
    for e in ROSTER:
        stem, _, numeral = e.id.rpartition("_")
        assert stem and numeral, f"{e.id} not in Name_Numeral form"
        assert set(numeral) <= roman, f"{e.id} numeral {numeral!r} not Roman"


def test_every_anchor_is_one_of_the_seven_omni_codes() -> None:
    valid = set(OmniCodes.get_all())
    assert len(valid) == 7
    for e in ALL_ENTRIES:
        assert e.anchor in valid, f"{e.id}: anchor {e.anchor!r} not an Omni-Code"


def test_seven_code_bearers_one_per_code_anchor_consistent() -> None:
    bearers = code_bearers()  # anchor -> bearer id
    assert set(bearers) == set(OmniCodes.get_all())
    assert len(bearers) == 7
    # Each bearer's own anchor must equal the Code it bears.
    for anchor, bearer_id in bearers.items():
        assert entry_by_id(bearer_id).anchor == anchor


def test_deep_members_have_impl_coordinators_do_not() -> None:
    deep = [e for e in ALL_ENTRIES if e.depth == "deep"]
    coord = [e for e in ALL_ENTRIES if e.depth == "coordinator"]
    assert deep and coord
    assert all(e.impl_path for e in deep)
    assert all(e.impl_path is None for e in coord)


def test_autonomy_is_anchored_and_capped() -> None:
    # The anchor's stability monotonically sets the autonomy ceiling, capped 0.95.
    codes = sorted(OmniCodes.get_all().values(), key=lambda c: c.stability)
    autonomies = [anchor_autonomy(c) for c in codes]
    assert all(0.0 < a <= 0.95 for a in autonomies)
    assert autonomies == sorted(autonomies)  # monotonic non-decreasing in stability
    # Highest-stability anchor reaches the cap; lowest is strictly below it.
    assert autonomies[-1] == pytest.approx(0.95)
    assert autonomies[0] < autonomies[-1]


def test_resolve_anchor_rejects_unknown() -> None:
    with pytest.raises(KeyError):
        resolve_anchor("OMNI_NONEXISTENT")


def test_no_memorial_terminology_anywhere_in_subsystem() -> None:
    # Hard requirement: Omni-Codes only — the term "memorial" must not appear in
    # any source file of the subagent subsystem.
    root = Path(subagents_pkg.__file__).parent
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if "memorial" in path.read_text(encoding="utf-8").lower()
    ]
    assert not offenders, f"'memorial' terminology found in: {offenders}"
