# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Releases before 3.0.0 live in the changelog archive, and the gates know it.

``CHANGELOG.md`` had grown to 13,104 lines. The eleven releases before 3.0.0
were moved, unchanged, to ``docs/changelog/CHANGELOG-pre-3.0.md``. A file
under ``docs/`` is read by scanners the root changelog was exempt from, and
old release notes cite tests that have since been renamed; these tests pin the
split and the exemptions that keep history out of the current-prose gates.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_ROOT = _REPO / "CHANGELOG.md"
_ARCHIVE = _REPO / "docs" / "changelog" / "CHANGELOG-pre-3.0.md"
_HEADING = re.compile(r"^## \[(\d+\.\d+\.\d+)\] - \d{4}-\d{2}-\d{2}$", re.MULTILINE)

_ARCHIVED = [
    "2.0.0",
    "1.7.0",
    "1.6.0",
    "1.5.1",
    "1.5.0",
    "1.4.0",
    "1.2.0",
    "1.1.2",
    "1.1.1",
    "1.1.0",
    "0.1.0",
]


def _versions(path: Path) -> list[str]:
    return _HEADING.findall(path.read_text(encoding="utf-8"))


def test_the_root_changelog_starts_at_three() -> None:
    versions = _versions(_ROOT)
    assert versions, "CHANGELOG.md has no release heading"
    assert all(int(version.split(".")[0]) >= 3 for version in versions), versions
    assert "3.0.0" in versions


def test_the_archive_holds_every_earlier_release_in_order() -> None:
    assert _versions(_ARCHIVE) == _ARCHIVED


def test_no_release_is_in_both_files_or_in_neither() -> None:
    root, archive = set(_versions(_ROOT)), set(_versions(_ARCHIVE))
    assert not root & archive
    assert root | archive >= {"3.0.0", *_ARCHIVED}


def test_each_file_points_at_the_other() -> None:
    root = _ROOT.read_text(encoding="utf-8")
    assert "## Earlier releases" in root
    for version in _ARCHIVED:
        assert f"| [{version}](docs/changelog/CHANGELOG-pre-3.0.md#" in root, version
    archive = _ARCHIVE.read_text(encoding="utf-8")
    assert "[`CHANGELOG.md`](../../CHANGELOG.md)" in archive


def test_every_compare_link_sits_with_its_release() -> None:
    link = re.compile(r"^\[(\d+\.\d+\.\d+)\]: https://github\.com/", re.MULTILINE)
    assert set(link.findall(_ROOT.read_text(encoding="utf-8"))) == set(_versions(_ROOT))
    assert link.findall(_ARCHIVE.read_text(encoding="utf-8")) == _ARCHIVED


def test_the_archive_is_exempt_from_the_current_prose_gates() -> None:
    """History is preserved, not rewritten: the scanners skip it by name."""
    spec = importlib.util.spec_from_file_location("doc_lint", _REPO / "scripts" / "doc_lint.py")
    assert spec and spec.loader
    doc_lint = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = doc_lint
    spec.loader.exec_module(doc_lint)
    excluded = {prefix for prefix, _reason in doc_lint.EXCLUDED}
    assert {"CHANGELOG.md", "docs/changelog/"} <= excluded

    from omni_mercury_engine.tools import workflow_version_drift_gate as drift

    assert {"CHANGELOG.md", _ARCHIVE.name} <= drift._DOC_SKIP_FILES


def test_the_algorithm_name_gate_still_reads_the_archived_releases() -> None:
    """The split must not shrink what that gate scans."""
    from omni_mercury_engine.tools import algorithm_name_drift_gate as gate

    assert _ARCHIVE in gate._default_docs(_REPO)
    assert _ROOT in gate._default_docs(_REPO)
