# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for ``scripts/mercury_session_setup.sh`` (environment bootstrap).

The script provisions disposable session containers: AMA Cryptography
(mandatory, fail-closed PQC backend) plus the Mercury ML/test/lint stack.
These tests pin the three contracts a bootstrap regression would silently
break:

1. **Disposable-environment guard** — without the hosted-session marker
   (``CLAUDE_CODE_REMOTE=true``) or an explicit ``MERCURY_SETUP_FORCE=1``,
   the script must refuse to modify the environment and exit 0 *before*
   any install step runs.
2. **Tooling completeness** — the install line must pull the ``[dev]``
   extra (alongside ``[ml]``), because the repository's own pytest
   configuration requires it: ``pyproject.toml`` sets ``asyncio_mode``
   (pytest-asyncio) and the CI invocation uses ``-n`` / ``--timeout`` /
   ``--cov`` (pytest-xdist / -timeout / -cov).  A bare ``pytest`` install
   was observed to fail the CI-configuration suite outright
   (``unrecognized arguments: -n 4 --timeout=300 --cov=...``), which is
   exactly the "tests run out of the box" promise the script makes.
3. **Version lockstep** — the script's ``AMA_REF`` must match the
   ``pyproject.toml`` ``[pqc]`` extra's git tag, and its pydocstyle pin
   must match the CI code-quality lane, so the bootstrap cannot drift
   from the surfaces CI actually validates.
"""

from __future__ import annotations

import os
import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "mercury_session_setup.sh"


def _script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_script_exists_and_bash_syntax_clean() -> None:
    assert SCRIPT.is_file()
    proc = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_guard_refuses_unmarked_environment_before_any_install() -> None:
    """Run the real script with both opt-in markers absent.

    It must exit 0 with the refusal message and must not reach the
    install body (no AMA build, no pip install output).
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"CLAUDE_CODE_REMOTE", "MERCURY_SETUP_FORCE"}
    }
    proc = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "leaving this environment untouched" in proc.stdout
    assert "Installing" not in proc.stdout
    assert "Building AMA" not in proc.stdout


def test_guard_recognises_both_opt_in_markers() -> None:
    text = _script_text()
    assert "CLAUDE_CODE_REMOTE" in text
    assert "MERCURY_SETUP_FORCE" in text


def test_install_line_uses_dev_extra_for_ci_capable_tooling() -> None:
    """The pip install must carry [ml,dev] — the [dev] extra is the
    single source of truth for the pytest plugin set and the
    black/mypy/types-requests pins that CI enforces."""
    text = _script_text()
    assert "[ml,dev]" in text, (
        "bootstrap must install the [dev] extra: bare pytest cannot run "
        "the suite in CI configuration (-n/--timeout/--cov, asyncio_mode)"
    )


def test_ama_ref_lockstep_with_pyproject_pqc_pin() -> None:
    script_ref = re.search(r'^AMA_REF="([^"]+)"', _script_text(), re.MULTILINE)
    assert script_ref is not None, "AMA_REF assignment not found in script"

    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    pqc_ref = re.search(r"ama-cryptography @ git\+https://[^\s\"]+@([\w.+-]+)", pyproject)
    assert pqc_ref is not None, "[pqc] extra git tag not found in pyproject.toml"
    assert script_ref.group(1) == pqc_ref.group(1), (
        f"script builds AMA {script_ref.group(1)} but pyproject [pqc] pins "
        f"{pqc_ref.group(1)} — bump them in lockstep"
    )


def test_lint_tooling_comes_from_the_ci_lint_group_like_the_code_quality_lane() -> None:
    """The script and CI install pydocstyle/flake8 from one place.

    Both export the pyproject ``ci-lint`` dependency group from uv.lock with
    hashes, so the pins cannot diverge (scripts/check_pinned_tool_versions.py
    holds the group's pins equal to the lock and the other surfaces).
    """
    assert re.search(
        r"uv export\b[^\n]*\\\n[^\n]*--group ci-lint", _script_text()
    ), "script must install the ci-lint group from uv.lock"
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "--only-group ci-lint" in ci, "code-quality lane must install the ci-lint group"
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    group = re.search(r"^ci-lint = \[(?P<body>[^\]]*)\]", pyproject, re.MULTILINE)
    assert group is not None and re.search(r'"pydocstyle==[\d.]+"', group.group("body"))


# AMA v5.0.0's setup.py preflight floors (PYSEC-2025-49 + GHSA-cx63-2mw6-8hw5,
# GHSA-8rrh-rw8j-w5fx, and the cmake shim its _check_cmake_version reads).
_AMA_FLOORS = {"setuptools": (84, 0, 0), "wheel": (0, 48, 0), "cmake": (4, 4, 4)}
_AMA_BUILD_FILE = "requirements/ci/ama-build.txt"


def _version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text)[:3])


def test_ama_build_floors_match_the_other_build_paths() -> None:
    """AMA's setup.py preflight rejects older build tools, so the session
    bootstrap, ``build_ama_native.sh`` (Dockerfile) and the CI composite
    action must all install the same floors. The bootstrap once fell behind
    (setuptools 83 / wheel 0.47 / cmake 4.4.0 against AMA v5.0.0's
    84 / 0.48 / 4.4.4), so a fresh container failed at its first AMA build.
    All three now install one hash-locked file, which must meet the floors."""
    paths = {
        "session setup": _script_text(),
        "scripts/build_ama_native.sh": (REPO / "scripts" / "build_ama_native.sh").read_text(
            encoding="utf-8"
        ),
        "build-ama-cryptography action": (
            REPO / ".github" / "actions" / "build-ama-cryptography" / "action.yml"
        ).read_text(encoding="utf-8"),
    }
    for label, text in paths.items():
        assert "requirements/ci/ama-build.txt" in text, label
        assert re.search(r"install --require-hashes\b", text), label
    pins = dict(
        re.findall(
            r"^(setuptools|wheel|cmake)==([0-9][0-9.]*)",
            (REPO / _AMA_BUILD_FILE).read_text(encoding="utf-8"),
            re.MULTILINE,
        )
    )
    assert set(pins) == set(_AMA_FLOORS), pins
    for name, floor in _AMA_FLOORS.items():
        assert _version(pins[name]) >= floor, (name, pins[name])
