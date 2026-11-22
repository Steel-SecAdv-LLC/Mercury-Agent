# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every install in the repository is pinned to a verified artifact.

OpenSSF Scorecard's Pinned-Dependencies check had filed one Medium alert per
unpinned install across the workflows, the composite actions, the shell
scripts and the Dockerfile (``pip install "pkg>=x"``, ``pip install -r``
without hashes, ``pip install .``, ``go install tool@vX``). They now install
from uv.lock or ``requirements/ci`` with ``--require-hashes``, and
``scripts/check_pinned_installs.py`` fails on any install that does not.

These tests hold the rule (what counts as pinned, including the spellings an
earlier line-regex scan missed: a quoted venv pip, a continuation line, a
heredoc) and the tree (nothing violates it today).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "check_pinned_installs.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pinned_installs", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate() -> ModuleType:
    return _load()


class TestTheRule:
    @pytest.mark.parametrize(
        "line",
        [
            # The audit-env spelling a line regex missed: a quoted venv pip.
            '"${RUNNER_TEMP}/mercury-audit-env/bin/pip" install --upgrade "pip>=26.1"',
            'pip install --upgrade "setuptools>=84.0.0" "wheel>=0.48.0"',
            "python -m pip install -r requirements.txt",
            "python3.12 -m pip install pytest",
            '"${venv_py}" -m pip install --upgrade setuptools',
            "uv pip install sympy",
            "sudo pip install foo",
            "if pip install foo; then echo ok; fi",
            "timeout 600 pip install foo",
            "FOO=1 pip install foo",
            "pip install .",
            "pip install -e .",
            'pip install --no-deps -e ".[ml]"',
            "pip install dist/mercury-1.0-py3-none-any.whl",
            "go install github.com/rhysd/actionlint/cmd/actionlint@v1.7.12",
            "npm install",
            "curl -sSL https://example.invalid/install.sh | bash",
            # --require-hashes present but after an option whose value a
            # scanner reading in order would take for a package name.
            'pip install --find-links "$dir" --require-hashes -r req.txt',
            # Hash mode inside the checkout without --no-config: uv applies
            # pyproject's constraint ranges and refuses the install.
            "uv pip install --require-hashes -r f.txt",
        ],
    )
    def test_an_unpinned_install_is_found(self, gate: ModuleType, line: str) -> None:
        assert gate.line_findings(line), line

    @pytest.mark.parametrize(
        "line",
        [
            'python -m pip install --require-hashes -r "${GITHUB_WORKSPACE}/requirements/ci/pip.txt"',
            'uv pip install --require-hashes --no-config --python "$(command -v python)" -r f.txt',
            '"${audit_py}" -m pip install --require-hashes --no-cache-dir -r audit.txt',
            "python -m pip install --no-deps --no-build-isolation -e .",
            "pip install --no-deps dist/mercury-1.0-py3-none-any.whl",
            "npm ci --prefix tests/api/browser --ignore-scripts",
            "go install github.com/rhysd/actionlint/cmd/actionlint@"
            "914e7df21a07ef503a81201c76d2b11c789d3fca",
            'echo "pip install foo"',
            "uv export --frozen --only-group ci-lint -o req.txt",
        ],
    )
    def test_a_pinned_install_passes(self, gate: ModuleType, line: str) -> None:
        assert gate.line_findings(line) == [], line

    def test_a_continuation_line_is_part_of_its_command(self, gate: ModuleType) -> None:
        text = 'python -m pip install \\\n  "foo>=1"\n'
        findings = gate.check_shell("x.sh", 10, text)
        assert [f.line for f in findings] == [10]

    def test_a_heredoc_body_is_data(self, gate: ModuleType) -> None:
        text = "python - <<'PY'\nprint('pip install foo')\nPY\npip install bar\n"
        findings = gate.check_shell("x.sh", 1, text)
        assert [f.command for f in findings] == ["pip install bar"]

    def test_dockerfile_run_instructions_are_read(self, gate: ModuleType, tmp_path: Path) -> None:
        (tmp_path / "Dockerfile").write_text(
            "FROM scratch\nRUN pip install --require-hashes -r /r.txt\n"
            'RUN pip install --no-cache-dir \\\n    "uv==0.11.32"\n'
        )
        findings = [f for f in gate.check_tree(tmp_path) if f.path == "Dockerfile"]
        assert [f.line for f in findings] == [3]

    def test_workflow_run_blocks_are_read_with_their_lines(
        self, gate: ModuleType, tmp_path: Path
    ) -> None:
        workflows = tmp_path / ".github" / "workflows"
        workflows.mkdir(parents=True)
        (workflows / "w.yml").write_text(
            "on: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - run: pip install --require-hashes -r r.txt\n"
            "      - run: |\n          echo hi\n          pip install pytest\n"
        )
        findings = gate.check_tree(tmp_path)
        assert [(f.path, f.line) for f in findings] == [(".github/workflows/w.yml", 9)]

    def test_requirements_files_must_be_hashed(self, gate: ModuleType, tmp_path: Path) -> None:
        req = tmp_path / "requirements" / "ci"
        req.mkdir(parents=True)
        (req / "tool.in").write_text("tool\n")
        (req / "tool.txt").write_text(
            "good==1.0 \\\n    --hash=sha256:" + "a" * 64 + "\nbad>=2.0\nunhashed==3.0\n"
        )
        (req / "orphan.in").write_text("x\n")
        reasons = sorted(f.reason for f in gate.check_requirements(tmp_path))
        assert reasons == [
            "needs both .in and compiled .txt",
            "requirement not ==-pinned with --hash",
            "requirement not ==-pinned with --hash",
        ]

    def test_pip_file_below_the_cve_floor_fails(self, gate: ModuleType, tmp_path: Path) -> None:
        req = tmp_path / "requirements" / "ci"
        req.mkdir(parents=True)
        (req / "pip.in").write_text("pip\n")
        (req / "pip.txt").write_text("pip==26.0 \\\n    --hash=sha256:" + "b" * 64 + "\n")
        reasons = [f.reason for f in gate.check_requirements(tmp_path)]
        assert reasons == ["pip below 26.1 (CVE-2026-6357)"]


class TestTheTree:
    def test_every_install_in_the_repository_is_pinned(self, gate: ModuleType) -> None:
        findings = [str(f) for f in gate.check_tree(_REPO_ROOT)]
        assert findings == []

    def test_the_scan_actually_sees_the_installs(self, gate: ModuleType) -> None:
        """A parser that silently read nothing would pass the tree test vacuously."""
        installs = 0
        for _path, _line, text in gate.shell_sources(_REPO_ROOT):
            for _offset, line in gate.logical_lines(text):
                for command in gate.simple_commands(line):
                    if gate.pip_install_arguments(command) is not None:
                        installs += 1
        assert installs >= 150, installs

    def test_the_gate_runs_in_ci_and_locally(self) -> None:
        ci = (_REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        local = (_REPO_ROOT / "scripts" / "run_ci_gates.sh").read_text(encoding="utf-8")
        assert "python scripts/check_pinned_installs.py" in ci
        assert "python scripts/check_pinned_installs.py" in local
