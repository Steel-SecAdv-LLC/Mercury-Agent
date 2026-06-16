# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for :mod:`scripts.check_pinned_tool_versions`.

The script is the pinned dev-tool version parity gate: ``black`` / ``mypy`` /
``types-requests`` / ``pydocstyle`` are pinned *exactly*, and the gate fails
closed when a tool's pins diverge across the surfaces that pin it.  Each tool
appears in a different subset of the four surfaces (``pyproject.toml``,
``ci.yml``, ``format.yml``, ``.pre-commit-config.yaml``) -- ``black`` in all
four, the others in fewer -- so the gate only cross-checks where a tool is
actually pinned and enforces a >=2-surface floor.  These tests exercise the
live repository (so the gate stays honest as the pins evolve) plus synthetic
fixtures covering each documented failure mode:

1. the live repo's four surfaces agree;
2. a clean synthetic fixture passes;
3. a single divergent surface trips the gate -- the "Dependabot only edits
   pyproject.toml" bug class that motivated the gate;
4. the original pre-existing drift (pre-commit ahead of every requirement
   surface) is caught;
5. ``v``-prefixed pre-commit revs compare equal to bare requirement pins, so
   ``mypy`` does not false-positive;
6. eroding a tool to a single surface trips the min-files floor;
7. prose mentions without ``==`` are never captured as pins.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import check_pinned_tool_versions as cptv


def _write_repo(
    root: Path,
    *,
    black: tuple[str, str, str, str] = ("26.5.1", "26.5.1", "26.5.1", "26.5.1"),
    mypy: str = "2.1.0",
    types_requests: str = "2.33.0.20260518",
    pydocstyle: str = "6.3.0",
    pydocstyle_in_precommit: bool = True,
) -> None:
    """Write the four parity surfaces under ``root``.

    ``black`` is a per-surface 4-tuple (pyproject, ci.yml, format.yml,
    pre-commit) so a test can diverge exactly one surface; the other tools are
    written consistently unless a test opts out (e.g. dropping pydocstyle from
    pre-commit to exercise the min-files floor).
    """
    b_py, b_ci, b_fmt, b_pc = black
    (root / ".github" / "workflows").mkdir(parents=True, exist_ok=True)

    (root / "pyproject.toml").write_text(
        "[project.optional-dependencies]\n"
        "dev = [\n"
        f'    "black=={b_py}",\n'
        f'    "mypy=={mypy}",\n'
        f'    "types-requests=={types_requests}",\n'
        "]\n",
        encoding="utf-8",
    )
    (root / ".github" / "workflows" / "ci.yml").write_text(
        "      - name: quality\n"
        f'        run: pip install "black=={b_ci}" mypy=={mypy} '
        f"pydocstyle=={pydocstyle} types-requests=={types_requests}\n",
        encoding="utf-8",
    )
    (root / ".github" / "workflows" / "format.yml").write_text(
        f'      - run: pip install "black=={b_fmt}"\n',
        encoding="utf-8",
    )

    precommit = (
        "repos:\n"
        "  - repo: https://github.com/psf/black\n"
        f"    rev: {b_pc}\n"
        "    hooks:\n"
        "      - id: black\n"
        "  - repo: https://github.com/pre-commit/mirrors-mypy\n"
        f"    rev: v{mypy}\n"
        "    hooks:\n"
        "      - id: mypy\n"
    )
    if pydocstyle_in_precommit:
        precommit += (
            "  - repo: https://github.com/pycqa/pydocstyle\n"
            f"    rev: {pydocstyle}\n"
            "    hooks:\n"
            "      - id: pydocstyle\n"
        )
    (root / ".pre-commit-config.yaml").write_text(precommit, encoding="utf-8")


def test_live_repo_parity_holds() -> None:
    """The shipped pyproject/ci/format/pre-commit pins must agree.

    Live regression: if anyone bumps one surface (or Dependabot edits
    pyproject.toml only) without the others, this fails immediately.
    """
    assert cptv.main([]) == 0


def test_clean_fixture_passes(tmp_path: Path) -> None:
    """All four surfaces consistent -> parity holds."""
    _write_repo(tmp_path)
    assert cptv.main(["--root", str(tmp_path)]) == 0


def test_single_divergent_surface_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """One surface (ci.yml) on a stale black trips the gate and is pinpointed."""
    _write_repo(tmp_path, black=("26.5.1", "26.3.1", "26.5.1", "26.5.1"))
    assert cptv.main(["--root", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "black" in err
    assert "26.3.1" in err and "26.5.1" in err


def test_precommit_ahead_of_requirements_fails(tmp_path: Path) -> None:
    """Reproduce the original main-branch drift: pre-commit ahead of the rest."""
    _write_repo(tmp_path, black=("26.3.1", "26.3.1", "26.3.1", "26.5.1"))
    assert cptv.main(["--root", str(tmp_path)]) == 1


def test_v_prefixed_precommit_rev_is_normalized(tmp_path: Path) -> None:
    """``rev: v2.1.0`` in pre-commit equals ``mypy==2.1.0`` -> no false drift."""
    _write_repo(tmp_path)
    occurrences = cptv.collect_occurrences(tmp_path)
    mypy_precommit = [
        o for o in occurrences if o.tool == "mypy" and o.location.startswith(".pre-commit")
    ]
    assert mypy_precommit, "expected a pre-commit mypy rev to be collected"
    assert all(o.version == "2.1.0" for o in mypy_precommit)
    assert all("mypy" not in v for v in cptv.find_violations(occurrences))


def test_single_surface_pin_fails_floor(tmp_path: Path) -> None:
    """A tool pinned in only one surface trips the min-files floor."""
    _write_repo(tmp_path, pydocstyle_in_precommit=False)  # pydocstyle now only in ci.yml
    violations = cptv.find_violations(cptv.collect_occurrences(tmp_path))
    assert any("pydocstyle" in v and "surface" in v for v in violations), violations


def test_prose_mentions_without_operator_are_not_pins(tmp_path: Path) -> None:
    """A changelog-style ``mypy 2.1`` mention must never be captured as a pin."""
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "# Concretely, mypy 1.19 and mypy 2.1 disagree; black 26.3 changed nothing.\n"
        '    "black==26.5.1",\n',
        encoding="utf-8",
    )
    occurrences = cptv._scan_requirements(pyproject, "pyproject.toml")
    assert {o.version for o in occurrences if o.tool == "black"} == {"26.5.1"}
    assert not [o for o in occurrences if o.tool == "mypy"]
