# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Word-boundary keyword matching for caption-vote VLM detectors.

The caption/keyword heuristics in :mod:`blip_vlm` and :mod:`lavad` used plain
substring tests, so ``"usual"`` matched inside ``"unusual"``, ``"normal"``
inside ``"abnormal"``, ``"hit"`` inside ``"white"``, and the description word
``"or"`` inside ``"floor"`` / ``"door"`` (audited defects B1/B4).  This module
is their shared cure: case-insensitive whole-word matching, and a small
stop-word filter for the free-text anomaly description so function words
never count as anomaly evidence.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

#: Function words in an anomaly description ("fire or smoke") that carry no
#: evidence of their own and must never match caption text.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "or",
        "and",
        "of",
        "in",
        "on",
        "at",
        "is",
        "are",
        "was",
        "were",
        "be",
        "being",
        "been",
        "to",
        "for",
        "with",
        "near",
        "by",
        "from",
        "into",
        "over",
        "under",
        "up",
        "down",
        "out",
        "off",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "as",
        "any",
        "all",
        "some",
        "no",
        "not",
    }
)


def _word_pattern(word: str) -> re.Pattern[str]:
    """Compile a case-insensitive whole-word pattern for ``word``."""
    return re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)


def find_word_matches(text: str, words: Iterable[str]) -> list[str]:
    r"""Return the ``words`` that occur as whole words in ``text``.

    Matching is case-insensitive and word-boundary anchored (``\b``), so a
    keyword never matches inside a longer word.  Each matching word appears
    once, in the order given.

    Args:
        text: The caption / response text to search.
        words: Candidate keywords.

    Returns:
        The lowercased keywords found as whole words in ``text``.
    """
    matches: list[str] = []
    seen: set[str] = set()
    for word in words:
        lowered = word.lower().strip()
        if not lowered or lowered in seen:
            continue
        seen.add(lowered)
        if _word_pattern(lowered).search(text):
            matches.append(lowered)
    return matches


def count_word_matches(text: str, words: Iterable[str]) -> int:
    """Count how many distinct ``words`` occur as whole words in ``text``."""
    return len(find_word_matches(text, words))


def content_words(description: str, exclude: Iterable[str] = ()) -> list[str]:
    """Extract the evidence-bearing words of a free-text anomaly description.

    Lowercases the description, splits on non-word characters, and drops
    stop words, one/two-letter fragments and anything in ``exclude`` (used
    to deduplicate against an explicit keyword list so one caption word is
    not double-counted).

    Args:
        description: Free-text anomaly description (e.g. ``"fire or smoke"``).
        exclude: Words already counted elsewhere.

    Returns:
        Deduplicated lowercase content words, in order of first appearance.
    """
    excluded = {word.lower() for word in exclude} | STOPWORDS
    terms: list[str] = []
    for raw in re.split(r"[^\w]+", description.lower()):
        if len(raw) <= 2 or raw in excluded or raw in terms:
            continue
        terms.append(raw)
    return terms
