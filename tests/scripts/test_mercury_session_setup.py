# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
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


def test_pydocstyle_pin_matches_ci_code_quality_lane() -> None:
    script_pin = re.search(r"pydocstyle==([\d.]+)", _script_text())
    assert script_pin is not None, "script must pin pydocstyle (CI does)"

    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    ci_pin = re.search(r"pydocstyle==([\d.]+)", ci)
    assert ci_pin is not None
    assert script_pin.group(1) == ci_pin.group(1), (
        f"script pins pydocstyle=={script_pin.group(1)} but ci.yml pins "
        f"=={ci_pin.group(1)} — bump them in lockstep"
    )
