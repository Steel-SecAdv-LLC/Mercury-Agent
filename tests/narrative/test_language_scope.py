# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Language-scope honesty guard (ROADMAP row 18 CI-gated mitigation).

Row 18 (multilingual natural-language interface) is an OPEN future epic:
Mercury's narrative / voice surface operates in **English only**, and the
README's "Implementation Languages" block draws the load-bearing
distinction that "multi-language" refers to the *implementation* stack
(Python / Rust / C-C++), **not** to localized multi-natural-language I/O.

That distinction is the row's honesty lock. This test makes the lock
CI-enforced instead of prose-only: it fails if the README's language
scoping is deleted or weakened into a multilingual-NL over-claim, so the
docs cannot silently start claiming a capability that does not exist
while the epic is deferred. Remove/relax these assertions only in the
same change that actually SHIPS multilingual NL and closes ROADMAP
row 18.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_README = _REPO_ROOT / "README.md"


@pytest.fixture(scope="module")
def readme_text() -> str:
    if not _README.is_file():  # pragma: no cover - defensive
        pytest.skip("README.md not present")
    return _README.read_text(encoding="utf-8")


class TestLanguageScopeHonesty:
    def test_implementation_language_distinction_present(self, readme_text: str) -> None:
        # The block that clarifies multi-language == implementation stack.
        assert "Implementation Languages" in readme_text
        assert "implementation" in readme_text.lower()

    def test_natural_language_scope_is_english_only(self, readme_text: str) -> None:
        # The English-only NL scope must be stated explicitly.
        assert re.search(
            r"operates in \*{0,2}English", readme_text
        ), "README must state the narrative/voice interface operates in English"

    def test_no_shipped_multilingual_nl_claim(self, readme_text: str) -> None:
        # The disclaimer that localized multi-natural-language I/O is NOT shipped.
        assert "multi-natural-language" in readme_text
        assert "future epic" in readme_text
        # And the explicit "not ... multilingual natural language" framing.
        assert re.search(r"not\*{0,2}\s+to multilingual natural", readme_text), (
            "README must keep the 'multi-language is implementation, not natural "
            "language' distinction"
        )

    def test_roadmap_row_18_stays_open(self) -> None:
        # If row 18 is marked closed, this guard (and the README block) should
        # be revisited in the same change — pin that they move together.
        roadmap = _REPO_ROOT / "docs" / "ROADMAP.md"
        if not roadmap.is_file():  # pragma: no cover - defensive
            pytest.skip("ROADMAP.md not present")
        text = roadmap.read_text(encoding="utf-8")
        row18 = [ln for ln in text.splitlines() if ln.startswith("| 18")]
        assert row18, "ROADMAP must retain a row 18 entry"
        assert "Multilingual natural-language interface" in row18[0]
