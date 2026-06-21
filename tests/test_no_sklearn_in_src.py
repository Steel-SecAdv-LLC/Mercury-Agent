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
import os
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

# Keyword names under which those callables accept the module name as their first
# parameter: ``import_module(name=…)`` / ``find_spec(name=…)`` / ``__import__(name=…)``
# and ``pytest.importorskip(modname=…)``. Scanning only the first *positional*
# arg would miss these keyword spellings — an easy bypass of the guard.
_DYNAMIC_IMPORT_NAME_KWARGS = frozenset({"name", "modname"})


def _is_sklearn_name(name: str) -> bool:
    return name == "sklearn" or name.startswith("sklearn.")


def _scanned_python_files() -> list[Path]:
    """Every first-party ``*.py`` outside ``benchmarks/`` and the skip-dirs.

    Uses a *pruned* ``os.walk`` — skip-dirs are removed from ``dirnames``
    in-place so heavy trees (``.git``, ``.venv``, ``node_modules`` …) are never
    descended into. ``rglob('*.py')`` would still walk those subtrees in full
    and only filter afterwards, which is needlessly slow on CI checkouts with a
    large ``.git``. ``benchmarks/`` is pruned at the repo root only (a nested
    ``tests/benchmarks/`` is still scanned), matching the prefix exemption.
    """
    allowed_root = _ALLOWED_ROOT.rstrip("/")
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(_REPO):
        rel_root = Path(dirpath).relative_to(_REPO)
        at_repo_root = rel_root == Path()
        # Prune in-place so os.walk does not descend skipped trees; the
        # top-level benchmarks/ root is exempt (competitor baselines live there).
        dirnames[:] = [
            d
            for d in dirnames
            if d not in _SKIP_DIR_NAMES and not (at_repo_root and d == allowed_root)
        ]
        for name in filenames:
            if name.endswith(".py"):
                files.append(Path(dirpath) / name)
    return files


def _called_name(func: ast.expr) -> str | None:
    """The bare callable name for ``foo(...)`` or ``a.b.foo(...)``."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _dynamic_import_modname(node: ast.Call) -> str | None:
    """The module-name string literal a dynamic-import call resolves, or ``None``.

    Considers the first *positional* argument **and** the first-parameter
    *keyword* spellings (``import_module(name=…)``, ``importorskip(modname=…)``);
    a positional-only scan would miss the keyword forms, which is an easy bypass.
    Non-literal or absent module-name arguments return ``None``.
    """
    arg: ast.expr | None = node.args[0] if node.args else None
    if arg is None:
        for kw in node.keywords:
            if kw.arg in _DYNAMIC_IMPORT_NAME_KWARGS:
                arg = kw.value
                break
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
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
            name = _called_name(node.func)
            if name in _DYNAMIC_IMPORT_CALLS:
                modname = _dynamic_import_modname(node)
                if modname is not None and _is_sklearn_name(modname):
                    return f"{name}({modname!r})"
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
    the dynamic ``importorskip``/``find_spec``/``import_module`` forms — including
    the keyword spellings (``modname=``/``name=``) a pure positional scan would
    miss (the original gap that let sklearn into tests). Exercises the real
    ``_dynamic_import_modname`` helper rather than re-implementing it."""
    src = (
        "import importlib\n"
        "import pytest\n"
        "from importlib.util import find_spec\n"
        "pytest.importorskip('sklearn')\n"  # positional
        "find_spec('sklearn')\n"  # positional
        "pytest.importorskip(modname='sklearn')\n"  # keyword bypass
        "importlib.import_module(name='sklearn')\n"  # keyword bypass
        "import numpy  # not sklearn\n"
    )
    tree = ast.parse(src)
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _called_name(node.func)
        if name is None or name not in _DYNAMIC_IMPORT_CALLS:
            continue
        modname = _dynamic_import_modname(node)
        if modname is not None and _is_sklearn_name(modname):
            hits.append(name)
    assert sorted(hits) == ["find_spec", "import_module", "importorskip", "importorskip"], hits


def test_sklearn_usage_detects_keyword_module_arg(tmp_path: Path) -> None:
    """End-to-end: the file-level detector flags the keyword-arg dynamic forms,
    closing the ``import_module(name=…)`` / ``importorskip(modname=…)`` bypass —
    while a benign keyword call to an unrelated module stays clean."""
    offender = tmp_path / "probe.py"
    offender.write_text("import importlib\nimportlib.import_module(name='sklearn.svm')\n")
    assert _sklearn_usage(offender) == "import_module('sklearn.svm')"

    benign = tmp_path / "benign.py"
    benign.write_text("import importlib\nimportlib.import_module(name='numpy')\n")
    assert _sklearn_usage(benign) is None


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
