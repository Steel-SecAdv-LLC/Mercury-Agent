"""Guardrail: scikit-learn must never be imported in shipped code or harnesses.

Mercury builds its own ML in ``omni_mercury_engine.ml.mercury_ml``; scikit-learn
is *only* a conceptual benchmark baseline (allowed solely under ``benchmarks/``
and behind a ``pytest.importorskip`` cross-check). This test fails loudly if any
AI/session re-introduces ``import sklearn`` / ``from sklearn`` into ``src/`` or
``research/`` — stopping the well-worn fallback pattern at the gate.
"""

from __future__ import annotations

import ast
from pathlib import Path

# Repo root = three levels up from this file (tests/ -> repo).
_REPO = Path(__file__).resolve().parents[1]
_SCANNED_DIRS = ("src/omni_mercury_engine", "research")


def _python_files() -> list[Path]:
    files: list[Path] = []
    for rel in _SCANNED_DIRS:
        root = _REPO / rel
        if root.exists():
            files.extend(p for p in root.rglob("*.py"))
    return files


def _imports_sklearn(path: Path) -> bool:
    """True iff the module has a real ``import sklearn`` / ``from sklearn`` (AST)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name == "sklearn" or a.name.startswith("sklearn.") for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "sklearn" or mod.startswith("sklearn."):
                return True
    return False


def test_no_sklearn_import_in_src_or_research() -> None:
    offenders = [str(p.relative_to(_REPO)) for p in _python_files() if _imports_sklearn(p)]
    assert not offenders, (
        "scikit-learn must not be imported in shipped code or research harnesses "
        "(use omni_mercury_engine.ml.mercury_ml). Offending files:\n  "
        + "\n  ".join(sorted(offenders))
    )


def test_guardrail_actually_scans_files() -> None:
    # Sanity: the scan finds a non-trivial number of modules (so a silent
    # empty-scan can never make the guardrail vacuously pass).
    assert len(_python_files()) > 50
