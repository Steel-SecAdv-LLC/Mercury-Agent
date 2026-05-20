"""Regression guard for CVE-2026-6357 (pip arbitrary code execution).

CVE-2026-6357 lets a malicious wheel hijack the install process on pip
versions earlier than 26.1.  We mitigate it project-wide by:

1. Pinning ``pip>=26.1`` in every CI workflow, every Dockerfile, and
   every devcontainer / dev-tooling script that installs from PyPI.
2. Failing the ``Workflow Hardening`` CI job if any workflow YAML adds
   a ``pip install`` step that is not preceded by ``pip install
   --upgrade "pip>=26.1"`` earlier in the same job.

This test file is the *gate on the gate*: it directly exercises the
hardening checker (``scripts/check_workflow_hardening.py``) and the
real workflow / Dockerfile inventory so a future drift — a new
workflow that forgets the floor, a Dockerfile that re-introduces an
unpinned ``pip install``, or a regression in the checker itself — is
caught by ``pytest`` long before it can land in a release branch.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HARDENING_SCRIPT = REPO_ROOT / "scripts" / "check_workflow_hardening.py"
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
DOCKERFILE = REPO_ROOT / "Dockerfile"

# Independent regexes that mirror the production guard's contract
# without sharing source.  The production guard's regex is allowed to
# evolve (e.g. to match new install front-ends); these stay anchored
# to the textual contract of CVE-2026-6357 so a regression that
# silently weakens the production regex still fails this suite.
LOCAL_PIP_INSTALL_RE = re.compile(
    r"(?<![\w-])(?:python\s+-m\s+)?pip\s+install\b"
)
LOCAL_PIP_UPGRADE_RE = re.compile(
    r"(?:python\s+-m\s+)?pip\s+install\b[^\n]*?--upgrade\b[^\n]*['\"]pip>=26(?:\.\d+)*['\"]"
)


def _load_hardening_module() -> object:
    """Load ``scripts/check_workflow_hardening.py`` as a regular module."""
    spec = importlib.util.spec_from_file_location(
        "check_workflow_hardening", HARDENING_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


class TestWorkflowInventory:
    """Every workflow + Dockerfile that runs ``pip install`` floors pip first."""

    def test_every_workflow_with_pip_install_has_pip_floor(self) -> None:
        offenders: list[str] = []
        for workflow in sorted(WORKFLOW_DIR.glob("*.yml")):
            text = workflow.read_text(encoding="utf-8")
            installs = [
                line
                for line in text.splitlines()
                if LOCAL_PIP_INSTALL_RE.match(line.strip())
                and not LOCAL_PIP_UPGRADE_RE.search(line)
            ]
            if not installs:
                continue
            if not LOCAL_PIP_UPGRADE_RE.search(text):
                offenders.append(workflow.name)
        assert not offenders, (
            "Workflows that run ``pip install`` MUST also run "
            f"``pip install --upgrade 'pip>=26.1'`` (CVE-2026-6357). "
            f"Offenders: {offenders}"
        )

    def test_dockerfile_pins_pip(self) -> None:
        text = DOCKERFILE.read_text(encoding="utf-8")
        # Both build stages (builder + runtime) must pin pip.
        upgrade_matches = LOCAL_PIP_UPGRADE_RE.findall(text)
        assert len(upgrade_matches) >= 2, (
            "Dockerfile must pin ``pip>=26.1`` in both the builder and "
            f"runtime stages (CVE-2026-6357).  Found {len(upgrade_matches)} "
            "matching upgrade lines."
        )


class TestHardeningChecker:
    """The production guard catches a freshly introduced regression."""

    def test_guard_passes_on_real_repo(self) -> None:
        module = _load_hardening_module()
        # Returns 0 on success, 1 on failure.  We run main() directly
        # rather than spawning a subprocess so the assertion fails
        # with a real Python traceback if the script crashes.
        exit_code = module.main()  # type: ignore[attr-defined]
        assert exit_code == 0, (
            "scripts/check_workflow_hardening.py failed on the real repo "
            "— CVE-2026-6357 regression-guard or a sibling hardening check "
            "is now firing."
        )

    def test_guard_detects_injected_unpinned_install(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A synthetic workflow without the floor must trigger the guard."""
        module = _load_hardening_module()
        # Fabricate a minimal workflow with the required top-level keys
        # plus an *unpinned* ``pip install`` — the regression-guard must
        # flag it without depending on the other hardening checks.
        fake_workflow = tmp_path / "drift.yml"
        fake_workflow.write_text(
            "name: drift\n"
            "on: push\n"
            "permissions:\n"
            "  contents: read\n"
            "concurrency:\n"
            "  group: drift-${{ github.ref }}\n"
            "  cancel-in-progress: true\n"
            "jobs:\n"
            "  drift:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - name: install thing\n"
            "        run: pip install requests\n",
            encoding="utf-8",
        )
        errors = module._check_pip_cve_2026_6357(  # type: ignore[attr-defined]
            fake_workflow, fake_workflow.read_text(encoding="utf-8")
        )
        assert errors, "Guard failed to detect an unpinned ``pip install``"
        assert any("CVE-2026-6357" in e for e in errors)

    def test_guard_accepts_floored_install(self, tmp_path: Path) -> None:
        """A workflow that floors pip first must NOT trigger the guard."""
        module = _load_hardening_module()
        fake_workflow = tmp_path / "good.yml"
        fake_workflow.write_text(
            "name: good\n"
            "on: push\n"
            "permissions:\n"
            "  contents: read\n"
            "concurrency:\n"
            "  group: good-${{ github.ref }}\n"
            "  cancel-in-progress: true\n"
            "jobs:\n"
            "  good:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - name: floor pip\n"
            '        run: python -m pip install --upgrade "pip>=26.1"\n'
            "      - name: install thing\n"
            "        run: pip install requests\n",
            encoding="utf-8",
        )
        errors = module._check_pip_cve_2026_6357(  # type: ignore[attr-defined]
            fake_workflow, fake_workflow.read_text(encoding="utf-8")
        )
        assert not errors, f"Guard misfired on a properly-floored workflow: {errors}"
