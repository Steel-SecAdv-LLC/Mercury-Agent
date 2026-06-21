# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Guardrail: scikit-learn must never appear outside ``benchmarks/`` — repo-wide.

Mercury builds its own ML in ``omni_mercury_engine.ml.mercury_ml`` (own
LogisticRegression / StandardScaler / roc_auc_score / Mann-Whitney). scikit-learn
is a **competitor**, not a dependency: it is authorised *solely* under the
top-level ``benchmarks/`` directory as a head-to-head baseline for Mercury to
beat — exactly like the other competitor detectors. It must appear nowhere else
in the repository (``src/``, ``tests/``, ``research/``, ``scripts/``,
``tools/`` …).

This guard fails loudly if any module outside ``benchmarks/`` pulls sklearn in,
catching **both**:

* static imports — ``import sklearn`` / ``from sklearn import …`` (AST), and
* dynamic / probe forms — ``pytest.importorskip("sklearn")``,
  ``importlib.import_module("sklearn")``, ``__import__("sklearn")`` and
  ``find_spec("sklearn")`` (the last is the "fall back to sklearn" scaffolding).

It is deliberately *usage*-aware, not substring-aware: comments and docstrings
that merely mention sklearn (e.g. "no sklearn dependency"), string literals such
as the ``ModelFramework.SKLEARN`` API label, and regex patterns that *scan* for
sklearn are all legitimate and never trip the guard. Only code that actually
names sklearn as a module to load or probe is an offence.
"""

from __future__ import annotations

import ast
from pathlib import Path

# Repo root = two levels up from this file: tests/<this file> -> tests -> repo.
_REPO = Path(__file__).resolve().parents[1]

# scikit-learn is a competitor baseline, permitted ONLY under this top-level dir
# (matched as a path *prefix*, so ``tests/benchmarks/`` is NOT exempt).
_ALLOWED_ROOT = "benchmarks/"

# Directories we never descend into: VCS, caches, build output, virtualenvs and
# vendored trees. Keeps the scan to first-party source and bounds it.
_SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "build",
        "dist",
        ".eggs",
        "site-packages",
        ".egg-info",
    }
)

# Dynamic-import / probe callables that take the module name as a string literal.
_DYNAMIC_IMPORT_CALLS = frozenset({"importorskip", "import_module", "__import__", "find_spec"})


def _is_sklearn_name(name: str) -> bool:
    return name == "sklearn" or name.startswith("sklearn.")


def _scanned_python_files() -> list[Path]:
    """Every first-party ``*.py`` outside ``benchmarks/`` and the skip-dirs."""
    files: list[Path] = []
    for p in _REPO.rglob("*.py"):
        rel = p.relative_to(_REPO)
        if any(part in _SKIP_DIR_NAMES for part in rel.parts):
            continue
        if rel.as_posix().startswith(_ALLOWED_ROOT):
            continue
        files.append(p)
    return files


def _called_name(func: ast.expr) -> str | None:
    """The bare callable name for ``foo(...)`` or ``a.b.foo(...)``."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _sklearn_usage(path: Path) -> str | None:
    """Return a short reason iff the module *actually pulls in* sklearn.

    Detects static imports and the dynamic/probe forms. Pure mentions in
    comments, docstrings, string labels or regexes are not usage and return
    ``None``.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_sklearn_name(alias.name):
                    return f"import {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if _is_sklearn_name(mod):
                return f"from {mod} import ..."
        elif isinstance(node, ast.Call):
            if _called_name(node.func) in _DYNAMIC_IMPORT_CALLS and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if _is_sklearn_name(arg.value):
                        name = _called_name(node.func)
                        return f"{name}({arg.value!r})"
    return None


def test_no_sklearn_outside_benchmarks() -> None:
    """No module outside ``benchmarks/`` may import or probe scikit-learn."""
    offenders = {
        str(p.relative_to(_REPO)): reason
        for p in _scanned_python_files()
        if (reason := _sklearn_usage(p)) is not None
    }
    assert not offenders, (
        "scikit-learn is a competitor baseline, permitted ONLY under benchmarks/. "
        "Use omni_mercury_engine.ml.mercury_ml instead, or move the head-to-head "
        "comparison into benchmarks/. Offending modules:\n  "
        + "\n  ".join(f"{f}: {r}" for f, r in sorted(offenders.items()))
    )


def test_guard_scans_repo_wide_not_vacuously() -> None:
    """The scan must genuinely span the repo (not the old src+research subset)."""
    files = _scanned_python_files()
    assert len(files) > 500, (
        f"repo-wide scan unexpectedly small ({len(files)} files) — the skip-list "
        "is probably too broad and the guard could pass vacuously."
    )
    roots = {p.relative_to(_REPO).parts[0] for p in files}
    assert {"src", "tests"} <= roots, f"expected src+tests in the scan, saw {sorted(roots)}"


def test_dynamic_import_detection_is_live() -> None:
    """Pin the detector so a future AST refactor can't silently stop catching
    the dynamic ``importorskip``/``find_spec`` forms (a pure import-node scan
    would miss them, which is how the original gap let sklearn into tests)."""
    src = (
        "import pytest\n"
        "from importlib.util import find_spec\n"
        "pytest.importorskip('sklearn')\n"
        "find_spec('sklearn')\n"
        "import numpy  # not sklearn\n"
    )
    tree = ast.parse(src)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _called_name(node.func) in _DYNAMIC_IMPORT_CALLS:
            arg = node.args[0] if node.args else None
            if isinstance(arg, ast.Constant) and _is_sklearn_name(str(arg.value)):
                hits.append(_called_name(node.func))
    assert sorted(hits) == ["find_spec", "importorskip"], hits


def test_benchmarks_are_exempt_and_really_use_sklearn() -> None:
    """Non-vacuity for the exemption: benchmarks/ is where sklearn legitimately
    lives, and it is genuinely excluded from the scanned set (so the allow-rule
    is exercised, not dead)."""
    bench = _REPO / "benchmarks"
    bench_importers = (
        [p for p in bench.rglob("*.py") if _sklearn_usage(p) is not None] if bench.exists() else []
    )
    assert bench_importers, (
        "expected scikit-learn baselines under benchmarks/; found none — either "
        "the baselines moved (update this guard) or the detector is broken."
    )
    scanned = set(_scanned_python_files())
    leaked = sorted(str(p.relative_to(_REPO)) for p in bench_importers if p in scanned)
    assert not leaked, f"benchmarks/ sklearn users leaked into the scanned set: {leaked}"
