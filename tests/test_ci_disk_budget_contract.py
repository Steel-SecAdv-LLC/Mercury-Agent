# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every CI job that installs torch keeps it off /tmp and checks its disk first.

PR #388 put torch in the ``[api]`` extra. On main, Security / Dependency
Security Scan then failed with ``OSError: [Errno 122] Disk quota exceeded``
while installing it into ``/tmp/mercury-audit-env``, on a runner with 80+ GB
free on ``/``. On the ubuntu-26.04 image ``/tmp`` can be a RAM-backed tmpfs
(about 7.8 GB) with a per-user quota of about 6.3 GB (actions/runner-images
#14777; the image fix, #14814, is still rolling out), and the CUDA torch stack
is about 7 GB installed. The PR had passed on a newer image whose ``/tmp`` is
on the root disk, so nothing caught it.

The rule, checked on every pull request:

* the extras that resolve torch are derived from ``pyproject.toml``, so an
  extra that gains torch is covered the day it does;
* a workflow step that installs one of them, or installs torch from an index
  that serves the CUDA build, is a heavy install;
* a job with a heavy install runs ``./.github/actions/runner-disk-guard``
  first: it points ``TMPDIR`` at ``$RUNNER_TEMP`` and checks the free space
  there, cleaning up only when it is short;
* no heavy install targets ``/tmp`` (a venv, ``--target`` or ``--prefix``
  under it): ``TMPDIR`` cannot move a path written by name.
"""

from __future__ import annotations

import re
import shlex
import sys
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO / ".github" / "workflows"
ACTION = REPO / ".github" / "actions" / "runner-disk-guard" / "action.yml"
GUARD_USES = "./.github/actions/runner-disk-guard"
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


def _heavy_extras(group: str) -> bool:
    # An extra list built from an expression (``.[${{ matrix.extras }}]``)
    # cannot be checked statically, so it counts as heavy.
    return "${{" in group or bool({e.strip() for e in group.split(",")} & torch_extras())


def _requirements_file_is_heavy(path: str) -> bool:
    file = REPO / path
    if not file.is_file():
        return True  # generated at run time: cannot be shown to be light
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if re.match(r"(?i)torch(?![\w-])", line):
            return True
        for group in re.findall(rf"(?:^|{re.escape(PROJECT)}|\.)\[([^\]]+)\]", line):
            if _heavy_extras(group):
                return True
    return False


def is_heavy_install(line: str) -> bool:
    """Whether one shell line installs a torch-bearing environment."""
    pip = re.search(r"\bpip3?[\"']?\s+install\b(.*)", line)
    if pip:
        args = pip.group(1)
        for group in re.findall(r"(?:\.|\b[\w.-]+|\.whl)\[([^\]]+)\]", args):
            if _heavy_extras(group):
                return True
        for req_file in re.findall(r"(?:-r|--requirement)[=\s]+[\"']?([^\s\"']+)", args):
            if _requirements_file_is_heavy(req_file):
                return True
        torch_named = re.search(r"(?<![\w-])torch(?![\w-])", args) is not None
        # Only a CPU *primary* index keeps torch light: with --extra-index-url
        # pip may still take the CUDA build from PyPI.
        cpu_only = re.search(r"--index-url[=\s]+\S*" + re.escape(CPU_INDEX), args) is not None
        return torch_named and not cpu_only
    uv = re.search(r"\buv\s+(?:sync|run)\b(.*)", line)
    if uv:
        args = uv.group(1)
        if "--all-extras" in args:
            return True
        if any(
            _heavy_extras(e) for e in re.findall(r"--extra[ =](\$\{\{[^}]*\}\}|[^\s\"']+)", args)
        ):
            return True
        return re.search(r"--with[ =][\"']?torch(?![\w-])", args) is not None
    return False


_VENV_RE = re.compile(
    r"\b(?:python3?\s+-m\s+venv|uv\s+venv|virtualenv)\s+[\"']?(\S+?)[\"']?(?:\s|$)"
)


def heavy_install_writes_tmp(line: str, venvs_in_tmp: set[str]) -> bool:
    """Whether a heavy install line puts its environment under /tmp by name."""
    if re.search(r"--(?:target|prefix|root)[=\s]+[\"']?/tmp(?:/|\s|$)", line):
        return True
    words = shlex.split(line, posix=True) if line.count('"') % 2 == 0 else line.split()
    pip_word = next((w for w in words if re.search(r"(?:^|/)pip3?$", w)), "")
    return pip_word.startswith("/tmp/") or any(pip_word.startswith(v + "/") for v in venvs_in_tmp)


def _jobs() -> list[tuple[str, str, dict[str, Any]]]:
    jobs = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for name, job in (workflow.get("jobs") or {}).items():
            jobs.append((path.name, name, job))
    return jobs


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return list(job.get("steps") or [])


def _heavy_indices(job: dict[str, Any]) -> list[int]:
    return [
        i
        for i, step in enumerate(_steps(job))
        if any(is_heavy_install(line) for line in _logical_lines(str(step.get("run") or "")))
    ]


HEAVY_JOBS = [(w, n, j) for w, n, j in _jobs() if _heavy_indices(j)]
IDS = [f"{workflow}::{name}" for workflow, name, _ in HEAVY_JOBS]


def test_torch_extras_are_derived_from_pyproject() -> None:
    extras = torch_extras()
    # [ml] declares torch directly; [api] since PR #388; [all] composes both.
    assert {"ml", "api", "all"} <= extras
    assert not {"dev", "compliance", "performance"} & extras


@pytest.mark.parametrize(
    ("line", "heavy"),
    [
        # The exact security.yml line that hit the /tmp quota on main.
        ('/tmp/mercury-audit-env/bin/pip install -e ".[api]" --upgrade "click>=8.2"', True),
        ('pip install -e ".[ml,api,dev]"', True),
        ('pip install "mercury-agent-global[api]"', True),
        ('pip install "dist/mercury_agent_global-3.0.0-py3-none-any.whl[ml]"', True),
        ('pip install -e ".[${{ matrix.extras }}]"', True),
        ("pip install numpy scipy torch scikit-learn sympy", True),
        ('pip install "torch>=2.2.0"', True),
        ("pip install --extra-index-url https://download.pytorch.org/whl/cpu torch", True),
        ("uv sync --locked --extra ml --extra dev", True),
        ("uv sync --locked --all-extras", True),
        ("uv run --extra api pytest", True),
        ("uv run --with torch python -c 1", True),
        ("uv pip install -e .[ml]", True),
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


def test_a_requirements_file_is_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    heavy = tmp_path / "heavy.txt"
    heavy.write_text("numpy\ntorch==2.15.0  # pinned\n", encoding="utf-8")
    light = tmp_path / "light.txt"
    light.write_text("numpy\n# torch is not here\npytest\n", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "REPO", tmp_path)
    assert is_heavy_install("pip install -r heavy.txt")
    assert not is_heavy_install("pip install -r light.txt")
    assert is_heavy_install("pip install -r generated-at-runtime.txt")


@pytest.mark.parametrize(
    ("line", "venvs", "writes_tmp"),
    [
        ('/tmp/mercury-audit-env/bin/pip install -e ".[api]"', set(), True),
        ('"${RUNNER_TEMP}/mercury-audit-env/bin/pip" install -e ".[api]"', set(), False),
        ('pip install --target /tmp/site -e ".[ml]"', set(), True),
        ('"/tmp/env/bin/pip" install ".[ml]"', {"/tmp/env"}, True),
        ('pip install -e ".[ml]"', set(), False),
    ],
)
def test_tmp_target_detector(line: str, venvs: set[str], writes_tmp: bool) -> None:
    assert heavy_install_writes_tmp(line, venvs) is writes_tmp


def test_the_scan_finds_the_heavy_jobs() -> None:
    names = {(workflow, job) for workflow, job, _ in HEAVY_JOBS}
    assert ("security.yml", "dependency-scan") in names
    assert ("ci.yml", "security-scan") in names
    assert len(names) >= 10, names


@pytest.mark.parametrize("job", HEAVY_JOBS, ids=IDS)
def test_heavy_job_runs_the_disk_guard_before_installing(
    job: tuple[str, str, dict[str, Any]],
) -> None:
    workflow, name, body = job
    steps = _steps(body)
    guard = [i for i, step in enumerate(steps) if step.get("uses") == GUARD_USES]
    assert guard, f"{workflow}::{name} installs torch without {GUARD_USES}"
    assert guard[0] < _heavy_indices(body)[0], f"{workflow}::{name} guards after installing"
    checkouts = [i for i, s in enumerate(steps) if "actions/checkout" in str(s.get("uses") or "")]
    assert (
        checkouts and checkouts[0] < guard[0]
    ), f"{workflow}::{name}: a local action needs the repository checked out first"


@pytest.mark.parametrize("job", HEAVY_JOBS, ids=IDS)
def test_heavy_install_does_not_target_tmp(job: tuple[str, str, dict[str, Any]]) -> None:
    workflow, name, body = job
    venvs_in_tmp: set[str] = set()
    for step in _steps(body):
        for line in _logical_lines(str(step.get("run") or "")):
            for target in _VENV_RE.findall(line):
                if target.startswith("/tmp/"):
                    venvs_in_tmp.add(target.rstrip("/"))
            if is_heavy_install(line):
                assert not heavy_install_writes_tmp(line, venvs_in_tmp), (
                    f"{workflow}::{name}: {line.strip()!r} builds a torch environment under "
                    "/tmp, which can be a quota-limited tmpfs; use $RUNNER_TEMP"
                )


def test_the_pre_fix_dependency_scan_fails_both_rules() -> None:
    """The job as it was on main at 9bc3cee would not pass this contract."""
    before = {
        "steps": [
            {"uses": "actions/checkout@sha"},
            {
                "run": "python -m venv /tmp/mercury-audit-env\n"
                '/tmp/mercury-audit-env/bin/pip install -e ".[api]" \\\n  --upgrade "click>=8.2"'
            },
        ]
    }
    assert _heavy_indices(before) == [1]
    assert not any(s.get("uses") == GUARD_USES for s in _steps(before))
    line = _logical_lines(str(before["steps"][1]["run"]))[1]
    assert heavy_install_writes_tmp(line, {"/tmp/mercury-audit-env"})


def test_disk_guard_action_contract() -> None:
    action = yaml.safe_load(ACTION.read_text(encoding="utf-8"))
    assert action["runs"]["using"] == "composite"
    script = action["runs"]["steps"][0]["run"]
    # Every later step writes temporary files to the root disk, not /tmp.
    assert 'echo "TMPDIR=${RUNNER_TEMP}" >> "${GITHUB_ENV}"' in script
    # Measured in bytes (``df -BG`` rounds up), on the filesystem in use.
    assert "-B1" in script and 'free_gb "${RUNNER_TEMP}"' in script
    # Cleanup only when short of the floor; a job with room pays nothing.
    assert 'if [ "${before}" -lt "${floor}" ]; then' in script
    # A failed removal is a warning; the floor decides.
    assert '|| echo "::warning::could not remove' in script
    assert "exit 1" in script and "Runner disk below floor" in script
    assert int(action["inputs"]["minimum-free-gb"]["default"]) >= 10


def test_no_inline_disk_cleanup_left_beside_the_action() -> None:
    # One implementation: an inline copy drifts (security.yml's did).
    for path in sorted(WORKFLOWS.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        assert "sudo rm -rf /usr/share/dotnet" not in text, path.name
