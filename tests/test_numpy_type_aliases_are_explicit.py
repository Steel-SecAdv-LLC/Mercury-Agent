# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Guardrail: a numpy array type alias must be annotated ``: TypeAlias``.

``Array = np.ndarray[Any, Any]`` is an *implicit* alias (PEP 613 calls the
annotated form explicit).  Whether MyPy accepts an implicit one depends on how
the right-hand side evaluates at runtime, which is a numpy-version-dependent
detail: under numpy 2.4 the tree's 14 implicit aliases typed clean, and under
numpy 2.5 -- which ``pyproject.toml`` admits, since it pins ``numpy>=2.4.0``
with no ceiling -- MyPy rejected every one of them as "Variable ... is not
valid as a type" and cascaded into **520 errors across 19 files**, against 14
once the aliases were annotated.  A blocking type gate that passes or fails on
a transitive dependency's patch release is not a gate.

``X: TypeAlias = np.ndarray[...]`` is unambiguous to every checker and every
numpy, so it is the form this repository uses.  This guard is AST-based and
usage-aware: it reads module-level assignments and only objects when the
right-hand side is a subscript of ``np.ndarray``, ``np.dtype``, ``NDArray`` or
``npt.NDArray`` and the target carries no annotation.  Strings, comments and
non-numpy aliases never trip it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCOPES = ("src", "tests", "scripts", "tools", "benchmarks", "examples", "research")

#: Subscript bases that build a numpy array type.
_ARRAY_BASES = frozenset({"np.ndarray", "np.dtype", "numpy.ndarray", "numpy.dtype", "NDArray"})


def _dotted(node: ast.expr) -> str:
    """``np.ndarray`` for an Attribute chain, ``NDArray`` for a bare Name, else ""."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted(node.value)
        return f"{prefix}.{node.attr}" if prefix else ""
    return ""


def _is_numpy_array_type(node: ast.expr) -> bool:
    return isinstance(node, ast.Subscript) and _dotted(node.value) in _ARRAY_BASES


def _python_files() -> list[Path]:
    files: list[Path] = []
    for scope in _SCOPES:
        root = _REPO / scope
        if root.is_dir():
            files.extend(
                p
                for p in root.rglob("*.py")
                if "__pycache__" not in p.parts and ".venv" not in p.parts
            )
    assert files, "no Python files found -- the scope list is wrong"
    return sorted(files)


def _implicit_aliases(path: Path) -> list[tuple[int, str]]:
    """Module-level ``NAME = np.ndarray[...]`` assignments with no annotation."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - not our files
        return []
    found = []
    for node in tree.body:  # module level only: a local alias is not a public type
        if not isinstance(node, ast.Assign) or not _is_numpy_array_type(node.value):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                found.append((node.lineno, target.id))
    return found


def test_every_numpy_array_alias_is_annotated_as_a_type_alias() -> None:
    """No module-level numpy array alias is left for the checker to guess at."""
    offences = [
        f"{path.relative_to(_REPO)}:{line}: {name} = np.ndarray[...] "
        f"-> write `{name}: TypeAlias = ...`"
        for path in _python_files()
        for line, name in _implicit_aliases(path)
    ]
    assert not offences, "implicit numpy type aliases found:\n" + "\n".join(offences)


@pytest.mark.parametrize(
    "module",
    [
        "omni_mercury_engine.vision.imaging",
        "omni_mercury_engine.biometric.rf_vital_signs",
        "omni_mercury_engine.ml.rl_envs",
    ],
)
def test_annotating_the_alias_did_not_change_what_it_binds(module: str) -> None:
    """``X: TypeAlias = Y`` binds exactly what ``X = Y`` bound -- annotation only."""
    import importlib

    mod = importlib.import_module(module)
    aliases = [n for n in vars(mod) if n.endswith("Array")]

    assert aliases, f"{module} declares no array alias; update this test's module list"
    for name in aliases:
        assert vars(mod)[name] is not None
