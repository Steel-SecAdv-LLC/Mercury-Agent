# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``uv.lock`` is the one resolution every uv-managed install path consumes.

The Dockerfile builder, the Code Quality, Type Checking, Core Tests and ML
Tests lanes install with uv; the lockfile must exist, state the project's
Python floor, and be installed with ``--locked`` everywhere so a stale lock is
refused rather than silently re-resolved. The Workflow Hardening lane runs
``uv lock --check`` so the drift is named on the pull request.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / "uv.lock"
CI = REPO / ".github" / "workflows" / "ci.yml"
DOCKERFILE = REPO / "Dockerfile"


def test_lockfile_exists_and_states_the_python_floor() -> None:
    lock = LOCK.read_text(encoding="utf-8")
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    floor = re.search(r'^requires-python = "([^"]+)"', pyproject, re.M)
    assert floor is not None
    assert f'requires-python = "{floor.group(1)}"' in lock
    assert 'name = "mercury-agent-global"' in lock
    # The lock resolves every extra, the git-pinned [pqc] extra included.
    assert "AMA-Cryptography.git" in lock


def test_every_uv_sync_is_locked() -> None:
    for path in (CI, DOCKERFILE):
        text = path.read_text(encoding="utf-8")
        syncs = re.findall(r"uv sync[^\n]*", text)
        assert syncs, path
        for line in syncs:
            assert "--locked" in line, (path, line)


def test_ci_lanes_and_dockerfile_pin_the_same_uv_version() -> None:
    ci = CI.read_text(encoding="utf-8")
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    ci_versions = set(re.findall(r'^\s+version: "(\d+\.\d+\.\d+)"\s*$', ci, re.M))
    docker_version = re.search(r'"uv==(\d+\.\d+\.\d+)"', dockerfile)
    assert docker_version is not None
    assert ci_versions == {docker_version.group(1)}, (ci_versions, docker_version.group(1))


def test_hardening_lane_checks_the_lock_and_builder_copies_it() -> None:
    ci = CI.read_text(encoding="utf-8")
    assert "run: uv lock --check" in ci
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert re.search(r"^COPY pyproject.toml uv.lock README.md /app/$", dockerfile, re.M)
    # uv is build tooling: removed with pip before the venv leaves the builder.
    assert "/opt/venv/bin/python -m pip uninstall -y uv" in dockerfile
