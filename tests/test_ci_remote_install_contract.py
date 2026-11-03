# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CI installs tools from pinned, verified artifacts, never from a moving script.

On 2026-11-03, right after PR #394 merged, every Trivy job on main failed in
"Install Trivy binary" with curl's exit code 6. The step piped
``contrib/install.sh`` from the aquasecurity/trivy *main* branch into
``sudo sh``; upstream had changed that script to download the archive from
``get.trivy.dev``, a host the runners could not resolve. Nothing in this
repository had changed. The Lean step in ``verifiers.yml`` had the same shape
(``elan-init.sh`` from the elan *master* branch, which installs the latest
elan and the moving "stable" Lean).

The rules, checked on every pull request:

* no workflow or composite action pipes a downloaded script into a shell;
* nothing is fetched from a moving branch (``main``/``master``/``HEAD``) of a
  raw GitHub URL;
* every Trivy scan step skips the action's own installer and runs after the
  pinned ``install-trivy`` action in the same job;
* the pinned installers verify a recorded SHA-256 before installing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((REPO / ".github" / "workflows").glob("*.yml"))
ACTIONS = sorted((REPO / ".github" / "actions").glob("*/action.yml"))
INSTALL_TRIVY = "./.github/actions/install-trivy"

_PIPE_TO_SHELL = re.compile(
    r"\b(?:curl|wget)\b[^\n|]*\|\s*(?:sudo\s+(?:-\S+\s+)*)?(?:ba|z|da)?sh\b"
)
_MOVING_RAW = re.compile(r"raw\.githubusercontent\.com/[^/\s]+/[^/\s]+/(?:main|master|HEAD)/")


def _joined(script: str) -> str:
    return re.sub(r"\\\n\s*", " ", script)


def _steps(path: Path) -> list[tuple[str, dict[str, Any]]]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "runs" in doc:  # composite action
        return [("action", s) for s in doc["runs"].get("steps", [])]
    out = []
    for name, job in (doc.get("jobs") or {}).items():
        out += [(name, s) for s in job.get("steps") or []]
    return out


ALL_RUN_STEPS = [
    (path.relative_to(REPO).as_posix(), job, step)
    for path in [*WORKFLOWS, *ACTIONS]
    for job, step in _steps(path)
    if step.get("run")
]


@pytest.mark.parametrize(
    ("script", "flagged"),
    [
        # The exact step that broke main on 2026-11-03.
        (
            "curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/"
            "contrib/install.sh \\\n  | sudo sh -s -- -b /usr/local/bin v0.70.0",
            True,
        ),
        ("curl -fsSL https://x.example/install | bash", True),
        ("wget -qO- https://x.example/i.sh | sudo -E sh", True),
        ('curl -fsSL -o "${work}/a.tar.gz" "${url}"', False),
        ("curl -s https://api.example/status | jq .state", False),
    ],
)
def test_pipe_detector(script: str, flagged: bool) -> None:
    assert bool(_PIPE_TO_SHELL.search(_joined(script))) is flagged


@pytest.mark.parametrize(
    ("where", "job", "step"), ALL_RUN_STEPS, ids=[f"{w}::{j}" for w, j, _ in ALL_RUN_STEPS]
)
def test_no_remote_script_is_piped_into_a_shell(where: str, job: str, step: dict[str, Any]) -> None:
    script = _joined(str(step["run"]))
    match = _PIPE_TO_SHELL.search(script)
    assert (
        match is None
    ), f"{where}::{job} pipes a downloaded script into a shell: {match.group(0)!r}"


def test_nothing_is_fetched_from_a_moving_branch() -> None:
    for path in [*WORKFLOWS, *ACTIONS]:
        text = path.read_text(encoding="utf-8")
        found = _MOVING_RAW.findall(text)
        assert not found, f"{path.relative_to(REPO)} fetches from a moving branch: {found}"


def _trivy_jobs() -> list[tuple[str, str, list[dict[str, Any]]]]:
    jobs = []
    for path in WORKFLOWS:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for name, job in (doc.get("jobs") or {}).items():
            steps = job.get("steps") or []
            if any("aquasecurity/trivy-action" in str(s.get("uses", "")) for s in steps):
                jobs.append((path.name, name, steps))
    return jobs


TRIVY_JOBS = _trivy_jobs()


def test_the_trivy_jobs_are_found() -> None:
    names = {(w, j) for w, j, _ in TRIVY_JOBS}
    assert {("ci.yml", "docker-build"), ("security.yml", "container-scan")} <= names
    assert len(names) >= 3, names


@pytest.mark.parametrize("job", TRIVY_JOBS, ids=[f"{w}::{j}" for w, j, _ in TRIVY_JOBS])
def test_every_trivy_scan_uses_the_pinned_binary(
    job: tuple[str, str, list[dict[str, Any]]],
) -> None:
    workflow, name, steps = job
    installs = [i for i, s in enumerate(steps) if s.get("uses") == INSTALL_TRIVY]
    scans = [
        i for i, s in enumerate(steps) if "aquasecurity/trivy-action" in str(s.get("uses", ""))
    ]
    assert installs, f"{workflow}::{name} scans with Trivy without {INSTALL_TRIVY}"
    assert installs[0] < scans[0], f"{workflow}::{name} installs Trivy after scanning"
    for i in scans:
        assert (steps[i].get("with") or {}).get(
            "skip-setup-trivy"
        ) is True, f"{workflow}::{name}: a trivy-action step would run its own installer"


def test_install_trivy_verifies_a_recorded_digest() -> None:
    action = yaml.safe_load((REPO / INSTALL_TRIVY / "action.yml").read_text(encoding="utf-8"))
    assert re.fullmatch(r"[0-9a-f]{64}", action["inputs"]["sha256"]["default"])
    assert re.fullmatch(r"\d+\.\d+\.\d+", action["inputs"]["version"]["default"])
    script = action["runs"]["steps"][0]["run"]
    assert "github.com/aquasecurity/trivy/releases/download/" in script
    assert "sha256sum -c -" in script
    # The verify comes before anything is extracted or installed.
    assert script.index("sha256sum -c -") < script.index("tar -xzf")
    assert "install.sh" not in script


def test_lean_is_installed_from_a_pinned_verified_elan() -> None:
    doc = yaml.safe_load(
        (REPO / ".github" / "workflows" / "verifiers.yml").read_text(encoding="utf-8")
    )
    step = next(
        s
        for s in doc["jobs"]["verifiers"]["steps"]
        if s.get("name", "").startswith("Install Lean 4")
    )
    env = step["env"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", env["ELAN_VERSION"])
    assert re.fullmatch(r"[0-9a-f]{64}", env["ELAN_SHA256"])
    assert re.fullmatch(r"leanprover/lean4:v\d+\.\d+\.\d+", env["LEAN_TOOLCHAIN"])
    run = step["run"]
    assert "sha256sum -c -" in run and run.index("sha256sum -c -") < run.index("elan-init")
    assert "stable" not in run and "elan-init.sh" not in run
