# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What the gate does with a long text is measured, and the code says what was measured.

``research/harm_gate_long_text/measure.py`` read 67 encyclopedia articles
through the gate three ways; the verdicts are in ``evidence.json`` beside it.
The gate pairs a hazard with an intent inside a passage and not across a whole
text, and the reason it gives is a count from that file. These tests fail if
the count and the reason part company.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from omni_mercury_engine.cognitive import decision_gate
from research.harm_gate_long_text.measure import TITLES, count, refused, report

_REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE: dict[str, Any] = json.loads(
    (_REPO_ROOT / "research" / "harm_gate_long_text" / "evidence.json").read_text(encoding="utf-8")
)


def test_every_article_read_is_one_the_harness_names() -> None:
    read = {(article["group"], article["title"]) for article in EVIDENCE["articles"]}

    assert read <= {(group, title) for group, titles in TITLES.items() for title in titles}
    assert len(read) == len(EVIDENCE["articles"]) == 67
    # Every one is longer than the gate used to read.
    assert (
        min(article["chars"] for article in EVIDENCE["articles"]) > decision_gate.MAX_SUBJECT_CHARS
    )


def test_reading_everything_refuses_no_general_article() -> None:
    """Reading more must not mean refusing more, on text that asks for nothing."""
    assert count(EVIDENCE, "general") == 37
    assert refused(EVIDENCE, "general", "first") == []
    assert refused(EVIDENCE, "general", "passages") == []


def test_pairing_at_any_distance_is_what_refuses_them() -> None:
    whole = refused(EVIDENCE, "general", "whole")

    assert sorted(whole) == ["Concrete", "Epidemiology", "French Revolution", "Photosynthesis"]


def test_the_gate_gives_that_count_as_its_reason() -> None:
    reason = " ".join((inspect.getdoc(decision_gate._assess) or "").split())
    general = count(EVIDENCE, "general")
    more = len(refused(EVIDENCE, "general", "whole")) - len(
        refused(EVIDENCE, "general", "passages")
    )

    assert f"{len(EVIDENCE['articles'])} encyclopedia articles" in reason
    assert f"refused {more} of {general} on general subjects" in reason
    assert "research/harm_gate_long_text" in reason


def test_what_reading_everything_costs_on_hazard_articles_is_stated_not_hidden() -> None:
    """An article on an agent says how it is made and used; the gate reads that now."""
    first = refused(EVIDENCE, "hazard", "first")
    passages = refused(EVIDENCE, "hazard", "passages")
    changelog = " ".join((_REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8").split())

    assert count(EVIDENCE, "hazard") == 30
    assert (len(first), len(passages)) == (1, 10)
    assert set(first) <= set(passages)
    assert f"{len(passages)} of {count(EVIDENCE, 'hazard')}" in changelog


def test_the_table_prints_from_the_evidence() -> None:
    table = report(EVIDENCE)

    assert "general: 37 articles" in table and "hazard: 30 articles" in table
