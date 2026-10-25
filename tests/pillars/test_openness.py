# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pillar: openness — the source is licensed, and every file says so.

"Open source" is a claim about the licence a recipient actually receives, so
the test is mechanical: every shipped ``.py`` carries the SPDX identifier and
the copyright line, and the same checker CI runs is the one that decides.

This module reuses ``scripts/normalize_headers.py`` rather than re-implementing
the scan, so a header rule can never mean one thing here and another in CI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "omni_mercury_engine"

SPDX_LINE = "# SPDX-License-Identifier: AGPL-3.0-or-later"
COPYRIGHT_LINE = "# Copyright (C) 2025 Steel Security Advisors LLC"


def _src_python_files() -> list[Path]:
    return sorted(SRC_ROOT.rglob("*.py"))


class TestEverySourceFileIsLicensed:
    def test_the_source_tree_is_not_empty(self) -> None:
        """Guard the guard: an empty glob would make every assertion vacuous."""
        assert len(_src_python_files()) > 500

    def test_every_src_python_file_declares_spdx(self) -> None:
        missing = [
            str(path.relative_to(REPO_ROOT))
            for path in _src_python_files()
            if SPDX_LINE not in path.read_text(encoding="utf-8")
        ]
        assert not missing, f"{len(missing)} src file(s) without SPDX: {missing[:10]}"

    def test_every_src_python_file_declares_copyright(self) -> None:
        missing = [
            str(path.relative_to(REPO_ROOT))
            for path in _src_python_files()
            if COPYRIGHT_LINE not in path.read_text(encoding="utf-8")
        ]
        assert not missing, f"{len(missing)} src file(s) without copyright: {missing[:10]}"

    def test_the_header_appears_in_the_first_lines(self) -> None:
        """A licence buried below the code is not a licence a reader will find."""
        offenders = []
        for path in _src_python_files():
            head = path.read_text(encoding="utf-8").splitlines()[:5]
            if not any(line.strip() == SPDX_LINE for line in head):
                offenders.append(str(path.relative_to(REPO_ROOT)))
        assert not offenders, offenders[:10]


class TestTheProjectShipsItsLicence:
    def test_agpl_licence_file_is_present(self) -> None:
        licence = REPO_ROOT / "LICENSE"
        assert licence.is_file()
        body = licence.read_text(encoding="utf-8")
        assert "GNU AFFERO GENERAL PUBLIC LICENSE" in body.upper()
        assert "13. Remote Network Interaction" in body

    def test_pyproject_declares_the_same_licence(self) -> None:
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "AGPL-3.0-or-later" in pyproject

    def test_licence_text_is_verbatim(self) -> None:
        """LICENSE is the FSF's AGPL-3.0 text, byte for byte: grants live elsewhere."""
        import hashlib

        digest = hashlib.sha256((REPO_ROOT / "LICENSE").read_bytes()).hexdigest()
        assert digest == "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0"

    def test_section_7_permission_is_scoped_and_keeps_section_13(self) -> None:
        """The NVIDIA grant is a section 7 permission and relaxes nothing else."""
        grant = (REPO_ROOT / "LICENSE-EXCEPTION").read_text(encoding="utf-8")
        assert "section 7" in grant
        assert "NVIDIA CUDA Libraries" in grant
        assert "covers those libraries and nothing else" in grant
        assert "section 13 still requires" in grant
        assert "may\nremove this additional permission" in grant

    def test_permission_ships_wherever_the_licence_does(self) -> None:
        """Packaging includes LICENSE-EXCEPTION next to LICENSE.

        Without an explicit ``license-files``, setuptools bundles files matching
        its default ``LICEN[CS]E*`` pattern; if ``license-files`` is ever set,
        it must still cover both files.
        """
        import fnmatch
        import tomllib

        project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))["project"]
        patterns = project.get("license-files", ["LICEN[CS]E*"])
        for name in ("LICENSE", "LICENSE-EXCEPTION"):
            assert any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns), name


class TestCiRunsTheSameChecker:
    """The gate in CI and the gate here must be one script, not two rules."""

    def test_ci_invokes_normalize_headers_in_check_mode(self) -> None:
        workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        assert "scripts/normalize_headers.py --check" in workflow

    def test_pre_commit_invokes_normalize_headers_in_check_mode(self) -> None:
        config = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
        assert "scripts/normalize_headers.py --check" in config

    @pytest.mark.slow
    def test_the_checker_passes_on_the_current_tree(self) -> None:
        """Run the real gate. If it fails here it fails in CI, and vice versa."""
        completed = subprocess.run(
            [sys.executable, "scripts/normalize_headers.py", "--check"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-4000:]
