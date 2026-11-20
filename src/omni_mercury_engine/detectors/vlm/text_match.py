# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Word-boundary keyword matching for caption-vote VLM detectors.

The caption/keyword heuristics in :mod:`blip_vlm` and :mod:`lavad` used plain
substring tests, so ``"usual"`` matched inside ``"unusual"``, ``"normal"``
inside ``"abnormal"``, ``"hit"`` inside ``"white"``, and the description word
``"or"`` inside ``"floor"`` / ``"door"`` (audited defects B1/B4).  This module
is their shared cure: case-insensitive matching anchored at the START of a
word, and a small stop-word filter for the free-text anomaly description so
function words never count as anomaly evidence.

The left word boundary is what fixed those four cases.  A right boundary
placed directly after the keyword went further than that and dropped every
inflected form -- ``"flood"`` no longer matched ``"flooded"``, ``"crash"``
not ``"crashed"``, ``"fire"`` not ``"fires"`` -- so captions that plainly
describe the anomaly raised no alarm.  A keyword therefore matches itself
and its REGULAR inflections (plural, possessive, ``-ed`` / ``-d``, ``-ing``,
with consonant doubling, e-dropping and ``y -> ies``), and nothing else:
never the inside of a longer word, never a derivation (``"fire"`` does not
match ``"fireman"``).  Irregular forms (``"fell"``, ``"fled"``,
``"knives"``) are not derived; list them as keywords.
"""

from __future__ import annotations

import re
from functools import lru_cache
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


_VOWELS = frozenset("aeiou")
#: The possessive ``'s`` (straight or typographic apostrophe), accepted after
#: any form.  (A plural owner's bare apostrophe needs nothing: the word
#: already ends at it.)
_POSSESSIVE = "(?:['\u2019]s)?"


def _is_consonant(letter: str) -> bool:
    return letter.isalpha() and letter not in _VOWELS


def _inflected_forms(word: str) -> list[str]:
    """Regex alternatives for ``word`` and its regular inflections.

    English inflection only, by its spelling rules:

    * sibilant endings (``s``, ``x``, ``z``, ``ch``, ``sh``) and consonant +
      ``o`` take ``-es``: crash -> crashes;
    * consonant + ``y`` turns to ``-ies`` / ``-ied``: emergency ->
      emergencies;
    * a final ``e`` takes ``-s`` / ``-d`` and is dropped before ``-ing``:
      damage -> damages, damaged, damaging;
    * anything else takes ``-s`` / ``-ed`` / ``-ing``, doubling a final
      consonant after a single vowel: flood -> flooded, hit -> hitting;
    * every form may carry a possessive ``'s``.

    Words shorter than three letters, or not ending in a letter, are matched
    exactly.

    Args:
        word: The lowercase keyword.

    Returns:
        Regex source strings, each matching one family of forms.
    """
    base = re.escape(word)
    if len(word) < 3 or not word[-1].isalpha():
        return [base]

    last, before = word[-1], word[-2]
    if word.endswith(("s", "x", "z", "ch", "sh")) or (last == "o" and _is_consonant(before)):
        forms = [base + "(?:es|ed|ing)?"]
    elif last == "y" and _is_consonant(before):
        forms = [base + "(?:ing)?", re.escape(word[:-1]) + "(?:ies|ied)"]
    elif last == "e":
        # "fleeing" keeps its e; "firing" drops it.
        forms = [base + "(?:s|d|ing)?", re.escape(word[:-1]) + "ing"]
    else:
        forms = [base + "(?:s|ed|ing)?"]
        if (
            _is_consonant(last)
            and last not in "wxy"
            and before in _VOWELS
            and _is_consonant(word[-3])
        ):
            forms.append(base + re.escape(last) + "(?:ed|ing)")
    return forms


@lru_cache(maxsize=2048)
def _word_pattern(word: str) -> re.Pattern[str]:
    r"""Compile a case-insensitive pattern for ``word`` and its inflections.

    Anchored with ``\b`` on the left (the keyword starts a word) and on the
    right after the inflectional suffix (nothing but an inflection follows).
    """
    forms = "|".join(_inflected_forms(word))
    return re.compile(rf"\b(?:{forms}){_POSSESSIVE}\b", re.IGNORECASE)


def same_word(first: str, second: str) -> bool:
    """Whether two words are the same word up to regular inflection.

    Args:
        first: One word.
        second: Another word.

    Returns:
        ``True`` if either is the other, or a regular inflection of it.
    """
    a, b = first.lower().strip(), second.lower().strip()
    if not a or not b:
        return False
    return bool(_word_pattern(a).fullmatch(b) or _word_pattern(b).fullmatch(a))


def find_word_matches(text: str, words: Iterable[str]) -> list[str]:
    r"""Return the ``words`` that occur in ``text``, in any regular inflection.

    Matching is case-insensitive and anchored at word boundaries (``\b``): a
    keyword matches where it starts a word and is followed by nothing but a
    regular inflectional ending (see :func:`_inflected_forms`), so it never
    matches inside a longer word.  Each matching word appears once, in the
    order given.

    Args:
        text: The caption / response text to search.
        words: Candidate keywords.

    Returns:
        The lowercased keywords found in ``text``.
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
    """Count how many distinct ``words`` occur in ``text`` (see :func:`find_word_matches`)."""
    return len(find_word_matches(text, words))


def content_words(description: str, exclude: Iterable[str] = ()) -> list[str]:
    """Extract the evidence-bearing words of a free-text anomaly description.

    Lowercases the description, splits on non-word characters, and drops
    stop words, one/two-letter fragments and anything in ``exclude`` (used
    to deduplicate against an explicit keyword list so one caption word is
    not double-counted).  Deduplication is up to regular inflection
    (:func:`same_word`): with the keyword ``"fire"``, the description word
    ``"fires"`` would match the very caption words the keyword already
    counts.

    Args:
        description: Free-text anomaly description (e.g. ``"fire or smoke"``).
        exclude: Words already counted elsewhere.

    Returns:
        Deduplicated lowercase content words, in order of first appearance.
    """
    counted = [word.lower() for word in exclude]
    terms: list[str] = []
    for raw in re.split(r"[^\w]+", description.lower()):
        if len(raw) <= 2 or raw in STOPWORDS:
            continue
        if any(same_word(raw, other) for other in (*counted, *terms)):
            continue
        terms.append(raw)
    return terms
