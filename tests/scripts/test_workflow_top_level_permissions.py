# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""No workflow grants a write scope at the top level; the gate enforces it.

OpenSSF Scorecard's Token-Permissions check filed a High alert on 2026-11-21
for every workflow whose document-level ``permissions:`` block granted a
write scope (``security-events: write`` in ci/security/docker,
``issues: write`` in gate-watchdog, ``contents``/``pull-requests: write`` in
dependabot-auto-merge). A top-level grant is the default for every job in
the file, so it leaks into jobs that never use it. The scopes were moved to
the one job that uses each, and ``check_workflow_hardening.py`` now fails on
any top-level write, so the class cannot return. These tests hold the rule
itself (it sees write grants however they are spelled, and only at the
document level) and the tree (no workflow violates it today).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "check_workflow_hardening.py"
_WORKFLOW_DIR = _REPO_ROOT / ".github" / "workflows"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("hw_toplevel", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker() -> ModuleType:
    return _load()


class TestTheRule:
    @pytest.mark.parametrize(
        "block",
        [
            "permissions:\n  contents: read\n  security-events: write\n",
            "permissions:\n  issues: 'write'\n",
            'permissions:\n  pull-requests: "write"\n',
            "permissions:\n  contents: write  # with a comment\n",
            "permissions:\n  actions: write-all\n",
        ],
    )
    def test_a_top_level_write_is_found_however_spelled(
        self, checker: ModuleType, block: str
    ) -> None:
        text = f"name: t\non: push\n{block}jobs:\n  a:\n    runs-on: ubuntu-latest\n"
        assert checker.top_level_write_scopes(text), block

    def test_a_job_level_write_is_not_this_rules_business(self, checker: ModuleType) -> None:
        """Job-level writes are governed by WRITE_OK, not by this rule."""
        text = (
            "name: t\non: push\npermissions:\n  contents: read\n"
            "jobs:\n  a:\n    runs-on: ubuntu-latest\n"
            "    permissions:\n      security-events: write\n"
        )
        assert checker.top_level_write_scopes(text) == []

    def test_read_scopes_and_comments_pass(self, checker: ModuleType) -> None:
        text = (
            "name: t\non: push\n"
            "permissions:\n  # write scopes live on jobs\n  contents: read\n  actions: read\n"
            "jobs:\n  a:\n    runs-on: ubuntu-latest\n"
        )
        assert checker.top_level_write_scopes(text) == []

    def test_the_gate_reports_the_scope_and_the_file(
        self, checker: ModuleType, tmp_path: Path
    ) -> None:
        """check_workflow fails on the prior state of ci.yml (and its kin)."""
        prior = (
            "name: t\non: push\n"
            "permissions:\n  contents: read\n  security-events: write\n"
            "concurrency:\n  group: t-${{ github.ref }}\n  cancel-in-progress: true\n"
            "jobs:\n  a:\n    runs-on: ubuntu-latest\n    steps: []\n"
        )
        path = tmp_path / "prior.yml"
        path.write_text(prior)
        errors = checker.check_workflow(path)
        assert any("security-events: write" in e and "one job" in e for e in errors), errors

    def test_write_ok_does_not_exempt_the_top_level(
        self, checker: ModuleType, tmp_path: Path
    ) -> None:
        """An allow-listed workflow still may not grant writes at the top."""
        name = sorted(checker.WRITE_OK)[0]
        path = tmp_path / name
        path.write_text(
            "name: t\non: push\n"
            "permissions:\n  contents: write\n"
            "concurrency:\n  group: t-${{ github.ref }}\n  cancel-in-progress: true\n"
            "jobs:\n  a:\n    runs-on: ubuntu-latest\n    steps: []\n"
        )
        errors = checker.check_workflow(path)
        assert any("top-level permissions grant 'contents: write'" in e for e in errors), errors


class TestTheTree:
    def test_no_committed_workflow_grants_a_top_level_write(self, checker: ModuleType) -> None:
        offenders = {
            path.name: scopes
            for path in sorted(_WORKFLOW_DIR.glob("*.yml"))
            if (scopes := checker.top_level_write_scopes(path.read_text(encoding="utf-8")))
        }
        assert offenders == {}

    def test_every_sarif_upload_sits_in_a_job_that_may_upload(self) -> None:
        """A job that uploads SARIF declares security-events: write itself.

        The scopes were moved out of the top level; this holds that none of
        the uploads was orphaned in the move (an upload without the scope
        fails at run time with 403, long after merge).
        """
        for path in sorted(_WORKFLOW_DIR.glob("*.yml")):
            text = path.read_text(encoding="utf-8")
            if "codeql-action/upload-sarif" not in text:
                continue
            lines = text.splitlines()
            upload_lines = [i for i, line in enumerate(lines) if "upload-sarif" in line]
            for at in upload_lines:
                indent_of = {
                    i: len(line) - len(line.lstrip())
                    for i, line in enumerate(lines)
                    if line.strip()
                }
                # Walk up to the job header (indent 2, "name:" style key).
                job_start = max(
                    i
                    for i in indent_of
                    if i <= at and indent_of[i] == 2 and lines[i].rstrip().endswith(":")
                )
                job_end_candidates = [i for i in indent_of if i > job_start and indent_of[i] == 2]
                job_end = min(job_end_candidates) if job_end_candidates else len(lines)
                job_text = "\n".join(lines[job_start:job_end])
                assert "security-events: write" in job_text, (
                    f"{path.name}: the job at line {job_start + 1} uploads SARIF without "
                    "declaring security-events: write at job level"
                )
