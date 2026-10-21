# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Guardrail: PyOD stays a benchmark baseline, never an operational dependency.

PyOD (and the scikit-learn estimators underneath it) exists in this repository
for one purpose: head-to-head comparison against Mercury's own detectors. The
authorised surface is:

* the top-level ``benchmarks/`` directory (competitor baselines), and
* ``src/omni_mercury_engine/comparison/`` — the lazy, comparison-only
  integration lane documented in ``pyproject.toml`` (``[benchmark]`` extra),
  plus the tests that exercise exactly that lane.

Until now this boundary was documentation only — ``tests/test_no_sklearn_in_src.py``
bans sklearn *by name*, but PyOD wraps sklearn estimators, so a PyOD import in
production code would pull sklearn into an operational path without ever
spelling "sklearn". These guards make the PyOD boundary executable, the same
way the sklearn guard works (AST-level usage detection: static imports plus the
``importorskip`` / ``import_module`` / ``__import__`` / ``find_spec`` dynamic
forms; mentions in comments, docstrings and string labels never trip it).

A second guard pins the architectural half of the isolation: no production
module may import ``omni_mercury_engine.comparison`` at all, so the comparison
lane cannot silently assume detection, API or engine responsibilities.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

# Reuse the sklearn guard's scanning machinery — one detector, two policies.
# A second hand-rolled AST walker would drift from the original (e.g. miss the
# keyword-argument dynamic-import spellings it already handles).
from tests.test_no_sklearn_in_src import (
    _DYNAMIC_IMPORT_CALLS,
    _REPO,
    _called_name,
    _dynamic_import_modname,
    _scanned_python_files,
)

if TYPE_CHECKING:
    from pathlib import Path

#: Path prefixes (relative to the repo root, POSIX separators) where PyOD is
#: authorised. ``benchmarks/`` is already pruned from ``_scanned_python_files``;
#: it is listed here for the non-vacuity check below.
_PYOD_ALLOWED_PREFIXES = (
    "benchmarks/",
    "src/omni_mercury_engine/comparison/",
    # Tests of the comparison lane itself (they importorskip pyod by design).
    "tests/benchmarks/",
    "tests/test_pyod_comparison.py",
    # This guard names the module in test fixtures.
    "tests/test_pyod_isolation_guard.py",
)


def _is_pyod_name(name: str) -> bool:
    return name == "pyod" or name.startswith("pyod.")


def _pyod_usage(path: Path) -> str | None:
    """Return a short reason iff the module actually pulls in PyOD.

    Mirrors ``_sklearn_usage``: static ``import`` / ``from … import`` plus the
    dynamic/probe call forms, positional and keyword spellings alike. Pure
    mentions in comments, docstrings or string labels return ``None``.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_pyod_name(alias.name):
                    return f"import {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level == 0 and _is_pyod_name(mod):
                return f"from {mod} import ..."
        elif isinstance(node, ast.Call):
            name = _called_name(node.func)
            if name in _DYNAMIC_IMPORT_CALLS:
                modname = _dynamic_import_modname(node)
                if modname is not None and _is_pyod_name(modname):
                    return f"{name}({modname!r})"
    return None


def _is_allowed(rel_posix: str) -> bool:
    return any(
        rel_posix == prefix or rel_posix.startswith(prefix) for prefix in _PYOD_ALLOWED_PREFIXES
    )


def test_pyod_only_in_comparison_and_benchmarks() -> None:
    """PyOD usage outside its authorised comparison surface is an offence."""
    offenders = {
        rel: reason
        for p in _scanned_python_files()
        if not _is_allowed(rel := p.relative_to(_REPO).as_posix())
        and (reason := _pyod_usage(p)) is not None
    }
    assert not offenders, (
        "PyOD is a benchmark baseline, authorised only under benchmarks/ and "
        "src/omni_mercury_engine/comparison/ (plus that lane's tests). Use "
        "Mercury's own detectors in operational code, or move the head-to-head "
        "comparison into the comparison lane. Offending modules:\n  "
        + "\n  ".join(f"{f}: {r}" for f, r in sorted(offenders.items()))
    )


def test_comparison_lane_not_imported_by_production_code() -> None:
    """No src/ module outside the lane may import ``omni_mercury_engine.comparison``.

    This is the architectural half of benchmark isolation: even a PyOD-free
    import of the comparison lane from detection, API or engine code would put
    benchmark machinery on a production path (and one lazy call away from
    loading PyOD + sklearn). Benchmarks and tests remain free to import it —
    that is what the lane is for.
    """
    lane_prefix = "src/omni_mercury_engine/comparison/"
    offenders: dict[str, str] = {}
    for p in _scanned_python_files():
        rel = p.relative_to(_REPO).as_posix()
        if not rel.startswith("src/") or rel.startswith(lane_prefix):
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("omni_mercury_engine.comparison"):
                        offenders[rel] = f"import {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                mod = node.module or ""
                if mod.startswith("omni_mercury_engine.comparison"):
                    offenders[rel] = f"from {mod} import ..."
                elif mod == "omni_mercury_engine":
                    names = ", ".join(alias.name for alias in node.names)
                    if "comparison" in {alias.name for alias in node.names}:
                        offenders[rel] = f"from omni_mercury_engine import {names}"
    assert not offenders, (
        "production code must not import the comparison lane; it exists solely "
        "for benchmark-side head-to-head runs. Offending modules:\n  "
        + "\n  ".join(f"{f}: {r}" for f, r in sorted(offenders.items()))
    )


def test_comparison_lane_really_uses_pyod() -> None:
    """Non-vacuity: the allow-list covers code that genuinely names PyOD.

    If the lane stopped using PyOD (or moved), the exemption would be dead and
    this guard could pass without testing anything — fail loudly instead so
    the allow-list tracks reality.
    """
    lane = _REPO / "src" / "omni_mercury_engine" / "comparison"
    users = [p for p in lane.rglob("*.py") if _pyod_usage(p) is not None]
    assert users, (
        "expected PyOD usage under src/omni_mercury_engine/comparison/; found "
        "none — either the lane moved (update this guard) or the detector broke."
    )


def test_pyod_detector_catches_dynamic_forms(tmp_path: Path) -> None:
    """Pin the detector on the dynamic spellings, mirroring the sklearn guard."""
    offender = tmp_path / "probe.py"
    offender.write_text("import importlib\nimportlib.import_module(name='pyod.models.iforest')\n")
    assert _pyod_usage(offender) == "import_module('pyod.models.iforest')"

    benign = tmp_path / "benign.py"
    benign.write_text(
        '"""Mentions pyod only in prose."""\nimport importlib\n'
        "importlib.import_module(name='numpy')\n"
    )
    assert _pyod_usage(benign) is None
