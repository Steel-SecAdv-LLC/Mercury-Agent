# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Every job that editable-installs the engine must provision the AMA backend.

``omni_mercury_engine`` enforces its PQC gate unconditionally at import time, so
a job that ``pip install -e .`` s the engine without building AMA fails at its
first engine import. ``competitive-benchmark.yml`` shipped exactly that gap and
its weekly scheduled run was red from 2026-08-09 for eight consecutive weeks
without any gate noticing. ``scripts/check_workflow_hardening.py`` now requires
the ``./.github/actions/build-ama-cryptography`` composite in every such job;
these tests pin the rule, its exemptions, and the real repository state.
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
_AMA_STEP = (
    "      - uses: ./.github/actions/build-ama-cryptography\n"
    "        with:\n"
    "          ama-ref: v4.0.0\n"
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("hw_ama", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check(path: Path) -> list[str]:
    module = _load()
    errors: list[str] = module._check_engine_install_provisions_ama(
        path, path.read_text(encoding="utf-8")
    )
    return errors


def _run(tmp_path: Path, body: str) -> list[str]:
    wf = tmp_path / "wf.yml"
    wf.write_text(_HEADER + body, encoding="utf-8")
    return _check(wf)


def _job(name: str, install: str, *, ama: bool) -> str:
    return (
        f"  {name}:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        f"      - uses: actions/checkout@{_SHA}\n"
        f"      - uses: actions/setup-python@{_SHA}\n"
        "      - name: install\n"
        "        run: |\n"
        '          python -m pip install --upgrade "pip>=26.1"\n'
        f"          {install}\n"
        + (_AMA_STEP if ama else "")
        + "      - run: python -c 'import omni_mercury_engine'\n"
    )


class TestFlagsUnprovisionedEngineInstalls:
    def test_editable_extras_install_without_ama_is_flagged(self, tmp_path: Path) -> None:
        errors = _run(tmp_path, _job("bench", 'pip install -e ".[ml,benchmark]"', ama=False))
        assert len(errors) == 1
        assert "bench" in errors[0]
        assert "build-ama-cryptography" in errors[0]

    def test_bare_dot_and_python_m_pip_forms_are_flagged(self, tmp_path: Path) -> None:
        body = _job("a", "pip install -e .", ama=False) + _job(
            "b", "python -m pip install --no-deps --editable=.", ama=False
        )
        errors = _run(tmp_path, body)
        assert len(errors) == 2
        assert "``a``" in errors[0] and "``b``" in errors[1]

    def test_inline_run_form_is_flagged(self, tmp_path: Path) -> None:
        body = (
            "  inline:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            f"      - uses: actions/setup-python@{_SHA}\n"
            "      - run: pip install -e '.[api]'\n"
        )
        assert len(_run(tmp_path, body)) == 1

    def test_ama_in_a_different_job_does_not_count(self, tmp_path: Path) -> None:
        body = _job("provisioned", "pip install -e .", ama=True) + _job(
            "bare", "pip install -e .", ama=False
        )
        errors = _run(tmp_path, body)
        assert len(errors) == 1
        assert "``bare``" in errors[0]


class TestAcceptsProvisionedAndOutOfScopeJobs:
    def test_install_followed_by_ama_composite_passes(self, tmp_path: Path) -> None:
        assert _run(tmp_path, _job("ok", 'pip install -e ".[ml,dev]"', ama=True)) == []

    def test_non_editable_and_third_party_installs_are_out_of_scope(self, tmp_path: Path) -> None:
        body = _job("tools", 'pip install "pyyaml>=6.0" ./dist/other.whl', ama=False)
        assert _run(tmp_path, body) == []

    def test_path_addressed_pip_into_an_isolated_env_is_out_of_scope(self, tmp_path: Path) -> None:
        # security.yml's dependency scan installs the engine into a throwaway
        # audit venv only to enumerate its dependency set; the job's own
        # interpreter never imports it.
        body = _job("audit", '/tmp/audit-env/bin/pip install -e ".[api]"', ama=False)
        assert _run(tmp_path, body) == []

    def test_echo_comment_and_heredoc_mentions_are_not_installs(self, tmp_path: Path) -> None:
        body = (
            "  docs:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            f"      - uses: actions/setup-python@{_SHA}\n"
            "      - run: |\n"
            "          # pip install -e .\n"
            '          echo "pip install -e ." >> "$GITHUB_STEP_SUMMARY"\n'
            "          cat > howto.txt <<'EOF'\n"
            "          pip install -e .\n"
            "          EOF\n"
        )
        assert _run(tmp_path, body) == []


class TestRealRepositoryPasses:
    def test_every_shipped_workflow_provisions_ama_where_it_installs_the_engine(self) -> None:
        workflows = sorted(_WORKFLOW_DIR.glob("*.yml"))
        assert workflows, "no workflows found; the check would be vacuous"
        errors = [e for wf in workflows for e in _check(wf)]
        assert errors == [], "\n".join(errors)

    def test_the_competitive_benchmark_lane_specifically_provisions_ama(self) -> None:
        text = (_WORKFLOW_DIR / "competitive-benchmark.yml").read_text(encoding="utf-8")
        assert "uses: ./.github/actions/build-ama-cryptography" in text
