# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every CI job that installs torch frees runner disk first.

The default PyPI torch wheel for Linux brings the CUDA libraries: about 3.5 GB
of wheels and several GB more once installed. PR #388 added torch to the
``[api]`` extra. ``ci.yml``'s Security Scan job was given a disk-free step and
``--no-cache-dir`` in the same change; ``security.yml``'s Dependency Security
Scan, which runs the same install, was not, and on main it failed with
"Disk quota exceeded" while installing torch. The pull request had passed on a
runner with more headroom, so nothing caught the gap before merge.

This module turns the rule into a check that runs on every pull request:

* the extras that resolve torch are derived from ``pyproject.toml`` (so a
  future extra that gains torch is covered the day it does);
* any workflow step that installs one of them (``pip install .[...]``,
  ``uv sync --extra ...`` / ``--all-extras``) or installs torch from the
  default index is a heavy install;
* a job with a heavy install must run ``./.github/actions/free-runner-disk``
  before it;
* a heavy ``pip install`` must not write pip's cache (``--no-cache-dir``),
  unless the job deliberately keeps a pip cache between runs, in which case
  its disk step must be the deep one.
"""

from __future__ import annotations

import re
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO / ".github" / "workflows"
ACTION = REPO / ".github" / "actions" / "free-runner-disk" / "action.yml"
FREE_DISK_USES = "./.github/actions/free-runner-disk"
CPU_INDEX = "download.pytorch.org/whl/cpu"
PROJECT = "mercury-agent-global"


@lru_cache(maxsize=1)
def torch_extras() -> frozenset[str]:
    """Extras of this project whose resolved closure contains torch."""
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    extras: dict[str, list[str]] = pyproject["project"]["optional-dependencies"]

    def closure(name: str, seen: frozenset[str]) -> set[str]:
        if name in seen:
            return set()
        names: set[str] = set()
        for requirement in extras.get(name, []):
            nested = re.match(rf"{re.escape(PROJECT)}\[([^\]]+)\]", requirement)
            if nested:
                for extra in nested.group(1).split(","):
                    names |= closure(extra.strip(), seen | {name})
            else:
                names.add(re.split(r"[<>=!~;\[ ]", requirement, maxsplit=1)[0].lower())
        return names

    return frozenset(name for name in extras if "torch" in closure(name, frozenset()))


def _logical_lines(script: str) -> list[str]:
    """Shell lines with backslash continuations joined."""
    return re.sub(r"\\\n\s*", " ", script).splitlines()


def is_heavy_install(line: str) -> bool:
    """Whether one shell line installs a torch-bearing environment."""
    heavy = torch_extras()
    pip = re.search(r"\bpip3?\s+install\b(.*)", line)
    if pip:
        args = pip.group(1)
        for group in re.findall(r"\.\[([^\]]+)\]", args):
            if {extra.strip() for extra in group.split(",")} & heavy:
                return True
        if CPU_INDEX in args:
            return False
        return re.search(r"(?<![\w-])torch(?![\w-])", args) is not None
    sync = re.search(r"\buv\s+sync\b(.*)", line)
    if sync:
        args = sync.group(1)
        if "--all-extras" in args:
            return True
        return bool(set(re.findall(r"--extra[ =]([\w-]+)", args)) & heavy)
    return False


def _jobs() -> list[tuple[str, str, dict[str, Any]]]:
    jobs = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for name, job in (workflow.get("jobs") or {}).items():
            jobs.append((path.name, name, job))
    return jobs


def _heavy_jobs() -> list[tuple[str, str, dict[str, Any]]]:
    found = []
    for workflow, name, job in _jobs():
        steps = job.get("steps") or []
        if any(
            is_heavy_install(line)
            for step in steps
            for line in _logical_lines(str(step.get("run") or ""))
        ):
            found.append((workflow, name, job))
    return found


def _keeps_pip_cache(steps: list[dict[str, Any]]) -> bool:
    for step in steps:
        uses = str(step.get("uses") or "")
        with_ = step.get("with") or {}
        if "actions/setup-python" in uses and str(with_.get("cache", "")) == "pip":
            return True
        if "actions/cache" in uses and "pip" in str(with_.get("path", "")).lower():
            return True
    return False


HEAVY_JOBS = _heavy_jobs()


def test_torch_extras_are_derived_from_pyproject() -> None:
    extras = torch_extras()
    # [ml] declares torch directly; [api] since PR #388; [all] composes both.
    assert {"ml", "api", "all"} <= extras
    # The core install and the light extras stay out of the rule.
    assert not {"dev", "compliance", "performance"} & extras


@pytest.mark.parametrize(
    ("line", "heavy"),
    [
        # The exact security.yml line that ran out of disk on main.
        ('/tmp/mercury-audit-env/bin/pip install -e ".[api]" --upgrade "click>=8.2"', True),
        ('pip install -e ".[ml,api,dev]"', True),
        ("pip install numpy scipy torch scikit-learn sympy", True),
        ('pip install "torch>=2.2.0"', True),
        ("uv sync --locked --extra ml --extra dev", True),
        ("uv sync --locked --all-extras", True),
        ("pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision", False),
        ('pip install -e ".[dev,compliance]"', False),
        ("pip install -e .", False),
        ("pip install pytest pytest-asyncio torchmetrics-free-name", False),
        ("pip install pytorch-lightning-docs", False),
        ("uv sync --locked --extra dev", False),
    ],
)
def test_heavy_install_classifier(line: str, heavy: bool) -> None:
    assert is_heavy_install(line) is heavy


def test_the_scan_finds_the_heavy_jobs() -> None:
    names = {(workflow, job) for workflow, job, _ in HEAVY_JOBS}
    assert ("security.yml", "dependency-scan") in names
    assert ("ci.yml", "security-scan") in names
    assert len(names) >= 10, names


@pytest.mark.parametrize(
    "job", HEAVY_JOBS, ids=[f"{workflow}::{name}" for workflow, name, _ in HEAVY_JOBS]
)
def test_heavy_job_frees_disk_before_installing(job: tuple[str, str, dict[str, Any]]) -> None:
    workflow, name, body = job
    steps: list[dict[str, Any]] = body.get("steps") or []
    free = [i for i, step in enumerate(steps) if step.get("uses") == FREE_DISK_USES]
    heavy = [
        i
        for i, step in enumerate(steps)
        if any(is_heavy_install(line) for line in _logical_lines(str(step.get("run") or "")))
    ]
    assert free, f"{workflow}::{name} installs torch without {FREE_DISK_USES}"
    assert free[0] < heavy[0], f"{workflow}::{name} frees disk after its first torch install"
    checkouts = [i for i, s in enumerate(steps) if "actions/checkout" in str(s.get("uses") or "")]
    assert (
        checkouts and checkouts[0] < free[0]
    ), f"{workflow}::{name}: a local action needs the repository checked out first"


@pytest.mark.parametrize(
    "job", HEAVY_JOBS, ids=[f"{workflow}::{name}" for workflow, name, _ in HEAVY_JOBS]
)
def test_heavy_pip_install_does_not_double_its_footprint(
    job: tuple[str, str, dict[str, Any]],
) -> None:
    workflow, name, body = job
    steps: list[dict[str, Any]] = body.get("steps") or []
    if _keeps_pip_cache(steps):
        # A deliberate cross-run cache is a speed trade; it is paid for with
        # the deep cleanup, never left to the default headroom.
        deep = [
            s
            for s in steps
            if s.get("uses") == FREE_DISK_USES
            and str((s.get("with") or {}).get("deep", "false")) == "true"
        ]
        assert deep, f"{workflow}::{name} keeps a pip cache but frees disk shallowly"
        return
    for step in steps:
        for line in _logical_lines(str(step.get("run") or "")):
            if is_heavy_install(line) and re.search(r"\bpip3?\s+install\b", line):
                assert "--no-cache-dir" in line, (
                    f"{workflow}::{name}: {line.strip()!r} writes pip's cache, which "
                    "nothing reuses and which holds a second copy of the torch stack"
                )


def test_free_disk_action_fails_loudly_below_its_floor() -> None:
    action = yaml.safe_load(ACTION.read_text(encoding="utf-8"))
    assert action["runs"]["using"] == "composite"
    script = action["runs"]["steps"][0]["run"]
    assert "/usr/local/lib/android" in script and "/opt/hostedtoolcache/CodeQL" in script
    assert "exit 1" in script and "MIN_FREE_GB" in script
    assert int(action["inputs"]["minimum-free-gb"]["default"]) >= 10


def test_no_inline_disk_cleanup_left_beside_the_action() -> None:
    # One implementation: an inline copy drifts (security.yml's did).
    for path in sorted(WORKFLOWS.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        assert "sudo rm -rf /usr/share/dotnet" not in text, path.name
