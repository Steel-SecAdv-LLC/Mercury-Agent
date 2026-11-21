# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What asking the classifier about a text a part at a time does is measured, and said as measured.

``research/meaning_level_windows/measure.py`` puts requests inside other text
and counts what the gate still blocks; the counts are in ``evidence.json``
beside it. The code and the document quote them. These tests fail if what is
quoted and what was counted part company, or if the gate's settings are no
longer the ones the counts are of.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from omni_mercury_engine.cognitive import ethical_bounding
from research.meaning_level_windows import measure

_REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE: dict[str, Any] = json.loads(measure.EVIDENCE.read_text(encoding="utf-8"))
DOC = " ".join(
    (_REPO_ROOT / "docs" / "MEANING_LEVEL_CLASSIFIER.md").read_text(encoding="utf-8").split()
)


def _row(section: str, filler: str, name: str, label: str) -> list[int]:
    cell = EVIDENCE[section][filler][name][label]
    return [cell["blocked"][str(words)] for words in EVIDENCE["padding"]]


def _table(numbers: list[int]) -> str:
    return " | ".join(str(number) for number in numbers) + " |"


def test_the_evidence_is_of_the_gate_as_it_ships() -> None:
    assert EVIDENCE["windows"] == [list(pair) for pair in ethical_bounding.MEANING_WINDOWS]
    assert EVIDENCE["units"] == ethical_bounding.MEANING_UNITS
    assert EVIDENCE["asked"] is ethical_bounding.MEANING_WINDOWS_ASKED is True
    assert EVIDENCE["padding"] == list(measure.PADDING)
    assert set(measure.SECTIONS) <= set(EVIDENCE)


def test_read_as_one_text_most_requests_are_lost_and_a_part_at_a_time_few() -> None:
    whole = _row("as_a_whole", "neutral", "train_offensive", "offensive")
    parts = _row("blocked", "neutral", "train_offensive", "offensive")

    assert whole[0] == parts[0] == 400  # alone, every one is refused either way
    assert all(part >= one for part, one in zip(parts, whole, strict=True))
    assert whole[-1] < 200 and parts[-1] > 380
    assert f"read as one text | {_table(whole)}" in DOC
    assert f"**a part at a time** | {_table(parts)}" in DOC


def test_no_harmless_row_is_refused_for_it() -> None:
    for section in ("as_a_whole", "blocked"):
        assert not any(_row(section, "neutral", "train_benign", "benign"))
        assert not any(_row(section, "neutral", "held_out", "benign"))
    assert EVIDENCE["blocked"]["neutral"]["train_benign"]["benign"]["texts"] == 400
    assert "0 of 400 training rows" in DOC and "0 of the 30 held-out" in DOC


def test_the_held_out_slice_is_reported_as_it_came_out() -> None:
    whole = _row("as_a_whole", "neutral", "held_out", "offensive")
    parts = _row("blocked", "neutral", "held_out", "offensive")

    assert EVIDENCE["blocked"]["neutral"]["held_out"]["offensive"]["texts"] == 133
    assert whole[0] == parts[0]  # one request alone is read as it was
    assert f"read as one text | {_table(whole)}" in DOC
    assert f"**a part at a time** | {_table(parts)}" in DOC


def test_the_code_gives_those_counts_as_its_reason() -> None:
    reason = " ".join((inspect.getdoc(ethical_bounding._meaning_windows) or "").split())
    at = EVIDENCE["padding"].index(96)
    whole = _row("as_a_whole", "neutral", "train_offensive", "offensive")[at]
    parts = _row("blocked", "neutral", "train_offensive", "offensive")[at]

    assert f"{whole} of 400 requests were still found" in reason
    assert f"read this way, {parts}," in reason
    assert "research/meaning_level_windows" in reason


def test_what_the_lexical_axes_refuse_among_other_requests_is_theirs_and_is_stated() -> None:
    """Harmless rows among how-to requests about other things: refused with no classifier too."""
    shipped = _row("blocked", "procedural", "train_benign", "benign")
    lexical = _row("lexical_only", "procedural", "train_benign", "benign")
    professional = _row("blocked", "procedural", "held_out", "benign")

    assert shipped == lexical
    assert professional == _row("lexical_only", "procedural", "held_out", "benign")
    at = [EVIDENCE["padding"].index(words) for words in (24, 96, 384)]
    assert (
        "{}, {} and {} of the 400 harmless training rows".format(*(shipped[i] for i in at)) in DOC
    )
    assert (
        f"{professional[at[1]]} and {professional[at[2]]} of the 30 held-out professional rows"
        in DOC
    )


def test_the_articles_are_refused_no_more_often_for_it() -> None:
    for name, groups in EVIDENCE["articles"].items():
        general, hazard = groups["general"], groups["hazard"]
        assert general["with_windows"] == general["without_windows"] == [], name
        assert set(hazard["without_windows"]) <= set(hazard["with_windows"]), name
        for group, label in ((general, "on general subjects"), (hazard, "about hazards")):
            row = (
                f"{name} set, {group['articles']} {label} | "
                f"{len(group['without_windows'])} | {len(group['with_windows'])} |"
            )
            assert row in DOC, row
    assert set(EVIDENCE["articles"]) == {"first", "second"}
