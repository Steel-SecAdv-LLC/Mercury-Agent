# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Coverage-floor doc drift guard: every doc quoting the CI floors must match ci.yml.

``.github/workflows/ci.yml`` (``COVERAGE_THRESHOLD_CORE`` / ``COVERAGE_THRESHOLD_FULL``)
is the single authoritative source for the blocking coverage gates.  PR #339
graduated the floors 25→30 / 50→55 but left nine doc locations quoting the
superseded values as current; this gate pins each of those locations to the
live ci.yml numbers so the next graduation cannot drift silently.  Offline;
no heavy imports.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
_CI = _REPO / ".github" / "workflows" / "ci.yml"


def _floors() -> tuple[int, int]:
    """Read (CORE, FULL) from the ci.yml env block — the authoritative source."""
    text = _CI.read_text(encoding="utf-8")
    core = re.search(r"^\s*COVERAGE_THRESHOLD_CORE:\s*(\d+)\s*$", text, re.M)
    full = re.search(r"^\s*COVERAGE_THRESHOLD_FULL:\s*(\d+)\s*$", text, re.M)
    assert core and full, "ci.yml no longer declares the coverage threshold env vars"
    return int(core.group(1)), int(full.group(1))


# Each entry: (repo-relative path, template that must appear verbatim once the
# {core}/{full} placeholders are filled with the live ci.yml values).
_DOC_SITES: list[tuple[str, str]] = [
    ("README.md", "(CORE ≥ {core} %, FULL ≥ {full} %) on every PR"),
    ("README.md", "`COVERAGE_THRESHOLD_CORE = {core} %` on the curated core lane"),
    ("README.md", "`COVERAGE_THRESHOLD_FULL = {full} %` on the ML lane"),
    ("README.md", "CI floors: CORE ≥ {core} %, FULL ≥ {full} % (measured)"),
    ("README.md", "≥ {core} % combined stmt+branch coverage on the curated core lane"),
    ("README.md", "Full suite under `tests/`, ≥ {full} % coverage"),
    ("ARCHITECTURE.md", "CORE ≥ {core} % on the curated core lane and FULL ≥ {full} %"),
    ("ARCHITECTURE.md", "`COVERAGE_THRESHOLD_CORE = {core}` on the curated core lane"),
    ("ARCHITECTURE.md", "`COVERAGE_THRESHOLD_FULL = {full}` on the full suite"),
    ("ARCHITECTURE.md", "CORE ≥ {core} % / FULL ≥ {full} %; the aspirational target"),
    ("CONTRIBUTING.md", "`COVERAGE_THRESHOLD_CORE = {core} %` on the curated core-tests lane"),
    ("CONTRIBUTING.md", "`COVERAGE_THRESHOLD_FULL = {full} %` on the full ml-tests lane"),
    ("CONTRIBUTING.md", "lane floors (CORE >= {core} %,"),
    ("CONTRIBUTING.md", "FULL >= {full} %) and trends toward"),
    ("docs/DEPLOYMENT.md", "(`COVERAGE_THRESHOLD_FULL={full}`)"),
    ("docs/DEPLOYMENT.md", "(`COVERAGE_THRESHOLD_CORE={core}`)"),
    ("docs/ROADMAP.md", "| **{full}** |"),
    ("docs/ROADMAP.md", "| **{core}** |"),
    (".coveragerc", "``COVERAGE_THRESHOLD_CORE={core}``"),
    (".coveragerc", "``COVERAGE_THRESHOLD_FULL={full}``"),
]


@pytest.mark.parametrize(("rel_path", "template"), _DOC_SITES)
def test_doc_quotes_live_floor(rel_path: str, template: str) -> None:
    core, full = _floors()
    expected = template.format(core=core, full=full)
    text = (_REPO / rel_path).read_text(encoding="utf-8")
    assert expected in text, (
        f"{rel_path} no longer quotes the live coverage floors "
        f"(CORE={core}, FULL={full}): expected snippet {expected!r}. "
        "Update the doc to match .github/workflows/ci.yml — the doc follows "
        "the workflow, never the other way around."
    )


def test_floors_are_sane() -> None:
    """The floors stay ordered and inside the documented aspirational band."""
    core, full = _floors()
    assert 0 < core < full <= 85, (core, full)
