# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Repository-wide guard: two retired phrasings must never re-enter Mercury's text.

Mercury is a civilization-first, people-first system, and its code system is the
Omni Codes. Two framings that displaced those were injected into earlier
revisions and have been replaced everywhere:

* a mission phrasing that put one humanitarian deployment slice ahead of the
  mission (replaced by *civilization-first*), and
* an alternative name for the code system (replaced by *Omni Codes*).

This test scans every git-tracked text file — source, tests, docs, configs,
workflows, data — so the replacement cannot silently regress. The patterns are
assembled from fragments so that this guard itself never spells the retired
phrasings (the same convention the Trivy secret allowlist follows for the GCP
marker). Ordinary uses of the individual words are not affected: a red-team
candidate that *survives* a gate, or a union-find root that survives a merge,
does not match.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

_SEPARATOR = r"[\s_\-‐-―]*"
RETIRED_PATTERNS: dict[str, re.Pattern[str]] = {
    "retired mission framing (use 'civilization-first')": re.compile(
        "surviv" + "or" + _SEPARATOR + "first", re.IGNORECASE
    ),
    "retired code-system name (use 'Omni Codes')": re.compile(
        "memor" + "ial" + _SEPARATOR + "codes?", re.IGNORECASE
    ),
}

#: Binary or generated artefacts with no prose; skipped by suffix.
_BINARY_SUFFIXES = frozenset(
    {".pt", ".pth", ".npz", ".npy", ".pkl", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".db"}
)
_MAX_BYTES = 8 * 1024 * 1024


def _tracked_text_files() -> list[Path]:
    try:
        listing = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
            timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        pytest.skip("not a git checkout: the tracked-file set is undefined here")
    paths = [REPO_ROOT / p for p in listing.decode("utf-8").split("\0") if p]
    return [
        p
        for p in paths
        if p.is_file()
        and p.suffix.lower() not in _BINARY_SUFFIXES
        and p.stat().st_size <= _MAX_BYTES
    ]


def test_scan_is_non_vacuous() -> None:
    files = _tracked_text_files()
    names = {p.relative_to(REPO_ROOT).as_posix() for p in files}
    # The scan must actually cover code, tests, docs, and security data files.
    assert len(files) > 1000
    assert "src/omni_mercury_engine/core/ethical_config.py" in names
    assert "src/omni_mercury_engine/security/sigma_immutable_baseline.json" in names
    assert "README.md" in names


@pytest.mark.parametrize("label", sorted(RETIRED_PATTERNS))
def test_retired_phrasing_absent_everywhere(label: str) -> None:
    pattern = RETIRED_PATTERNS[label]
    offenders: list[str] = []
    for path in _tracked_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{lineno}")
    assert not offenders, f"{label} found at: {offenders}"


@pytest.mark.parametrize(
    ("text", "should_match"),
    [
        ("civilization-first mission", False),
        ("the red-team candidates that survive the gate", False),
        ("Omni Codes", False),
        ("surviv" + "or-first", True),
        ("Surviv" + "or_First_principle", True),
        ("surviv" + "or first", True),
        ("Memor" + "ial Codes", True),
        ("memor" + "ial_code", True),
    ],
)
def test_patterns_catch_every_separator_and_case(text: str, should_match: bool) -> None:
    matched = any(p.search(text) for p in RETIRED_PATTERNS.values())
    assert matched is should_match
