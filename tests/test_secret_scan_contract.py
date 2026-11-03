# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The pull-request secret scan covers everything a squash merge lands.

PR #388 passed Secret Detection and its squash commit then failed it on main:
gitleaks-action builds a pull request's scan range from the first page of the
PR's commit list (30 commits), and the two test keys were added in a later
commit. ``security.yml`` now also scans a synthetic squash-equivalent commit
on every pull request, which is the exact unit main scans after the merge.

These checks keep that step in place and keep ``.gitleaks.toml`` honest.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
SECURITY = REPO / ".github" / "workflows" / "security.yml"
GITLEAKS = REPO / ".gitleaks.toml"


def _secret_scan_steps() -> list[dict[str, Any]]:
    workflow = yaml.safe_load(SECURITY.read_text(encoding="utf-8"))
    steps: list[dict[str, Any]] = workflow["jobs"]["secret-scan"]["steps"]
    return steps


def _squash_step() -> dict[str, Any]:
    steps = [s for s in _secret_scan_steps() if "squash-equivalent" in str(s.get("name", ""))]
    assert len(steps) == 1, "security.yml secret-scan lost its squash-equivalent scan"
    return steps[0]


def test_security_workflow_runs_on_pull_requests_to_main() -> None:
    workflow = yaml.safe_load(SECURITY.read_text(encoding="utf-8"))
    # PyYAML reads the bare ``on`` key as boolean True.
    triggers = workflow.get("on", workflow.get(True))
    assert "main" in triggers["pull_request"]["branches"]


def test_squash_equivalent_scan_runs_on_every_pull_request() -> None:
    step = _squash_step()
    assert step["if"] == "github.event_name == 'pull_request'"
    run = step["run"]
    # The merged tree on the base parent: one commit, the PR's whole net change.
    assert "commit-tree 'HEAD^{tree}' -p 'HEAD^1'" in run
    assert '--log-opts="-1 ${squash}"' in run
    assert "--exit-code 1" in run and "--config .gitleaks.toml" in run
    # It refuses to run on anything but the two-parent merge ref.
    assert 'wc -w)" -eq 3' in run


def test_squash_scan_binary_is_pinned_and_verified() -> None:
    env = _squash_step()["env"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", env["GITLEAKS_VERSION"])
    assert re.fullmatch(r"[0-9a-f]{64}", env["GITLEAKS_SHA256"])
    assert "sha256sum -c -" in _squash_step()["run"]


def test_secret_scan_checkout_has_full_history() -> None:
    checkouts = [s for s in _secret_scan_steps() if "actions/checkout" in str(s.get("uses", ""))]
    assert checkouts
    for step in checkouts:
        assert (step.get("with") or {}).get("fetch-depth") == 0


def test_gitleaks_config_extends_the_default_rules() -> None:
    config = tomllib.loads(GITLEAKS.read_text(encoding="utf-8"))
    # Without this the custom file replaces every built-in rule with none.
    assert config["extend"]["useDefault"] is True


def test_every_allowlisted_secret_is_an_exact_literal_still_in_use() -> None:
    """An allowlist entry is one known string, and it is still in the tree.

    A pattern (``.*``, a character class) would silence real secrets of a
    similar shape; an entry whose string left the tree is dead weight that a
    future, real key could match. History-only entries say so in the comment
    above them.
    """
    text = GITLEAKS.read_text(encoding="utf-8")
    config = tomllib.loads(text)
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True, text=True
    ).stdout.split("\0")
    corpus = []
    for name in tracked:
        path = REPO / name
        if not name or name in {".gitleaks.toml"} or not path.is_file():
            continue
        if path.stat().st_size > 5_000_000:
            continue
        try:
            corpus.append(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue
    tree = "\n".join(corpus)
    for regex in config["allowlist"]["regexes"]:
        # Only escapes of literal dots are allowed; no quantifiers or classes.
        assert not re.search(r"(?<!\\)[*+?\[\]{}()|^$]", regex), regex
        literal = regex.replace("\\.", ".")
        preceding = text[: text.index(regex)].rsplit("\n", 6)[-6:]
        history_only = any("History only" in line for line in preceding)
        assert history_only or literal in tree, f"allowlisted string no longer in the tree: {regex}"


def test_the_config_cannot_widen_what_both_scans_skip() -> None:
    """Both scans read the pull request's own ``.gitleaks.toml``.

    So the file is held to a shape that cannot quietly widen what the scans
    skip: only the known top-level tables (no ``[[rules]]`` override that
    could replace or weaken a built-in rule), allowlisted paths that each name
    one existing file (no pattern such as ``.*``), and allowlisted commits as
    full SHAs.
    """
    config = tomllib.loads(GITLEAKS.read_text(encoding="utf-8"))
    assert set(config) <= {"title", "extend", "allowlist"}, sorted(config)
    assert set(config["extend"]) == {"useDefault"}
    allowlist = config["allowlist"]
    assert set(allowlist) <= {"description", "commits", "paths", "regexes"}
    for pattern in allowlist.get("paths", []):
        assert not re.search(r"(?<!\\)[*+?\[\]{}()|^$]", pattern), pattern
        literal = pattern.replace("\\.", ".")
        assert (REPO / literal).is_file(), f"allowlisted path is not a file in the tree: {pattern}"
    for sha in allowlist.get("commits", []):
        assert re.fullmatch(r"[0-9a-f]{40}", sha), sha


def test_no_fingerprint_ignore_file() -> None:
    """A ``.gitleaksignore`` fingerprint names a commit SHA.

    The squash-equivalent scan builds a new commit on every run, so a
    fingerprint written for it could never match: a finding would be ignored
    on main's push scan and still fail every pull request, or the reverse.
    Exceptions belong in ``.gitleaks.toml`` as exact literals.
    """
    assert not (REPO / ".gitleaksignore").exists()
