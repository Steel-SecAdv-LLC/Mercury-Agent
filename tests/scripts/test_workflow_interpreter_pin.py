# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every job that runs ``pip install`` must pin its interpreter first.

`scripts/check_workflow_hardening.py` gains a check that a job running a real
``pip install`` has an ``actions/setup-python`` step before it. Without that
step the job installs into whatever Python the runner image ships — the same
borrowed-from-the-image exposure the CVE-2026-6357 pip floor closes, one level
down.

This check exists because the gap actually shipped: PR #365's ``manifests``
job pip-installed with no ``setup-python``, matched no other job in ci.yml, and
no gate caught it until a human noticed. These tests pin the rule and every
exemption so it stays mechanical.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "check_workflow_hardening.py"
_WORKFLOW_DIR = _REPO_ROOT / ".github" / "workflows"

_HEADER = (
    "name: t\n"
    "on: push\n"
    "permissions:\n"
    "  contents: read\n"
    "concurrency:\n"
    "  group: t-${{ github.ref }}\n"
    "  cancel-in-progress: true\n"
    "jobs:\n"
)
_SHA = "a" * 40


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("hw_interp", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(tmp_path: Path, body: str) -> list[str]:
    module = _load()
    wf = tmp_path / "wf.yml"
    wf.write_text(_HEADER + body, encoding="utf-8")
    # The module is loaded dynamically, so its attributes are ``Any`` to the
    # type checker; narrow the result through an annotated local so the strict
    # ``tests/scripts/`` lane sees a concrete ``list[str]``.
    errors: list[str] = module._check_pip_uses_pinned_interpreter(
        wf, wf.read_text(encoding="utf-8")
    )
    return errors


class TestFlagsUnpinnedInstalls:
    def test_pip_install_without_setup_python_is_flagged(self, tmp_path: Path) -> None:
        body = (
            "  build:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            f"      - uses: actions/checkout@{_SHA}\n"
            "      - name: install\n"
            "        run: pip install requests\n"
        )
        errors = _run(tmp_path, body)
        assert errors, "a pip install with no setup-python must be flagged"
        assert "setup-python" in errors[0]
        assert "build" in errors[0]

    def test_python_m_pip_form_is_flagged(self, tmp_path: Path) -> None:
        body = (
            "  build:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            '        run: python -m pip install --upgrade "pip>=26.1"\n'
        )
        assert _run(tmp_path, body), "the floor line itself runs pip and needs an interpreter"

    def test_setup_python_after_the_install_does_not_count(self, tmp_path: Path) -> None:
        """Ordering matters: setup must precede the install it protects."""
        body = (
            "  build:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - name: install\n"
            "        run: pip install requests\n"
            f"      - uses: actions/setup-python@{_SHA}\n"
        )
        assert _run(tmp_path, body), "setup-python after the install came too late"

    def test_one_error_per_job_not_per_line(self, tmp_path: Path) -> None:
        body = (
            "  build:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - name: install several\n"
            "        run: |\n"
            "          pip install a\n"
            "          pip install b\n"
            "          pip install c\n"
        )
        assert len(_run(tmp_path, body)) == 1


class TestAcceptsPinnedAndExemptJobs:
    def test_setup_python_before_install_passes(self, tmp_path: Path) -> None:
        body = (
            "  build:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            f"      - uses: actions/setup-python@{_SHA}\n"
            "      - name: install\n"
            '        run: python -m pip install --upgrade "pip>=26.1"\n'
        )
        assert _run(tmp_path, body) == []

    def test_container_job_is_exempt(self, tmp_path: Path) -> None:
        """A container image is itself the pinned interpreter."""
        body = (
            "  build:\n"
            "    runs-on: ubuntu-latest\n"
            "    container: python:3.12\n"
            "    steps:\n"
            "      - name: install\n"
            "        run: pip install requests\n"
        )
        assert _run(tmp_path, body) == []

    def test_job_with_no_real_pip_install_is_not_flagged(self, tmp_path: Path) -> None:
        body = (
            "  build:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            f"      - uses: actions/checkout@{_SHA}\n"
            "      - name: build docs\n"
            "        run: make html\n"
        )
        assert _run(tmp_path, body) == []

    def test_echo_documentation_line_is_not_an_install(self, tmp_path: Path) -> None:
        """``echo "pip install ..." >> file`` writes text; it installs nothing.

        This is exactly release.yml's ``release`` job, which must stay clean.
        """
        body = (
            "  build:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            f"      - uses: actions/checkout@{_SHA}\n"
            "      - name: write changelog\n"
            '        run: echo "pip install mercury-agent==$VERSION" >> NOTES.md\n'
        )
        assert _run(tmp_path, body) == []

    def test_pip_install_inside_a_heredoc_body_is_not_a_command(self, tmp_path: Path) -> None:
        body = (
            "  build:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            f"      - uses: actions/checkout@{_SHA}\n"
            "      - name: emit script\n"
            "        run: |\n"
            "          cat > install.sh <<'EOF'\n"
            "          pip install requests\n"
            "          EOF\n"
        )
        assert _run(tmp_path, body) == []

    def test_setup_python_fork_or_mirror_counts(self, tmp_path: Path) -> None:
        body = (
            "  build:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            f"      - uses: some-mirror/setup-python@{_SHA}\n"
            "      - run: pip install requests\n"
        )
        assert _run(tmp_path, body) == []


class TestRealRepositoryPasses:
    def test_every_shipped_workflow_pins_its_interpreter(self) -> None:
        """The gate must be green on the repository as it stands."""
        module = _load()
        offenders: list[str] = []
        for wf in sorted(_WORKFLOW_DIR.glob("*.yml")) + sorted(_WORKFLOW_DIR.glob("*.yaml")):
            found: list[str] = module._check_pip_uses_pinned_interpreter(
                wf, wf.read_text(encoding="utf-8")
            )
            offenders.extend(found)
        assert offenders == [], offenders

    def test_the_manifests_job_specifically_is_pinned(self) -> None:
        """Regression anchor for the exact job whose gap motivated this check."""
        module = _load()
        ci = _WORKFLOW_DIR / "ci.yml"
        errors: list[str] = module._check_pip_uses_pinned_interpreter(
            ci, ci.read_text(encoding="utf-8")
        )
        assert not any("manifests" in e for e in errors)
