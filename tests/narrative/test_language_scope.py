# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Language-scope honesty guard: the README states what the language layer does, and no more.

Until the language layer shipped, this guard pinned the README's
"operates in English only; multilingual natural language is a future epic"
statement (ROADMAP row 18). The layer (``narrative/languages.py``) now
understands questions in seven languages and answers the social turns, the
fallback menu and the follow-up prompts in the asker's language, while the
grounded answers (a detection narrative, a knowledge profile) stay in
English with a one-line note. That is a partial capability, and the guard
now pins both halves of it against the code rather than against prose:

* the README names exactly the languages ``SUPPORTED`` holds, so adding or
  dropping a language without saying so fails;
* it states the boundary (no machine translation of narratives or profiles)
  and that the rest of the epic is still open;
* it keeps the distinction that "multi-language" in the Implementation
  Languages block means the implementation stack;
* ROADMAP row 18 is marked partly closed, not open and not closed, and names
  what remains.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from omni_mercury_engine.narrative.languages import SUPPORTED

_REPO_ROOT = Path(__file__).resolve().parents[2]
_README = _REPO_ROOT / "README.md"
_ROADMAP = _REPO_ROOT / "docs" / "ROADMAP.md"

#: The English name of each supported code, as the README writes it.
_ENGLISH_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
}

_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}


@pytest.fixture(scope="module")
def readme_text() -> str:
    return _README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def language_block(readme_text: str) -> str:
    start = readme_text.index("**Natural language is a separate axis")
    end = readme_text.index("</details>", start)
    return " ".join(readme_text[start:end].split())


class TestLanguageScopeHonesty:
    def test_implementation_language_distinction_present(self, readme_text: str) -> None:
        assert "Implementation Languages" in readme_text
        assert re.search(r"not\*{0,2}\s+to multilingual natural", readme_text), (
            "README must keep the 'multi-language is implementation, not natural "
            "language' distinction"
        )

    def test_the_readme_names_exactly_the_supported_languages(self, language_block: str) -> None:
        assert set(_ENGLISH_NAMES) == set(
            SUPPORTED
        ), "a language was added to or dropped from SUPPORTED; name it here and in the README"
        count = _WORDS[len(SUPPORTED)]
        assert f"{count} languages" in language_block
        named = re.search(r"languages \(([^)]*)\)", language_block)
        assert named, "the README must list the supported languages in parentheses"
        listed = {name.strip() for name in re.split(r",", named.group(1))}
        assert listed == set(_ENGLISH_NAMES.values())

    def test_the_grounded_answers_stay_english_and_that_is_stated(
        self, language_block: str
    ) -> None:
        assert re.search(r"grounded answers are built in \*{0,2}English", language_block)
        assert "never machine translated" in language_block
        assert "one-line note" in language_block

    def test_an_unsupported_language_is_refused_as_stated(self, language_block: str) -> None:
        assert "422" in language_block and "`auto`" in language_block

    def test_the_rest_of_the_epic_is_still_open(self, language_block: str) -> None:
        assert "future epic" in language_block
        assert "row 18" in language_block

    def test_roadmap_row_18_is_partly_closed_and_names_what_remains(self) -> None:
        rows = [
            ln for ln in _ROADMAP.read_text(encoding="utf-8").splitlines() if ln.startswith("| 18")
        ]
        assert len(rows) == 1, "ROADMAP must retain exactly one row 18"
        row = rows[0]
        assert "Multilingual natural-language interface" in row
        assert "PARTLY CLOSED" in row
        assert "**Open:**" in row and "never machine translated" in row
        for code in SUPPORTED:
            assert code in row
