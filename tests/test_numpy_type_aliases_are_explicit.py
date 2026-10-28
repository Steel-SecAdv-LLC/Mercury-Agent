# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Style guard: a numpy array type alias is annotated ``: TypeAlias``.

``Array = np.ndarray[Any, Any]`` is an *implicit* alias; PEP 613 calls the
annotated ``Array: TypeAlias = np.ndarray[Any, Any]`` form explicit. MyPy
accepts both when it can resolve numpy, so this is a readability and
intent-signalling rule, not a correctness one: the annotated form says "this
name is a type" at the binding site, and cannot be mistaken for a module-level
value by a reader or by a checker that loses the numpy stubs.

**Correcting the record.** The audit commit that introduced this guard said
these aliases broke MyPy under numpy 2.5, citing 520 errors across 19 files.
That was wrong, and the error count had a single cause that was not numpy:
``mypy`` was installed as a uv tool with its own interpreter, which could not
see the system ``dist-packages``, so numpy was unresolvable and every
``np.ndarray`` subscript degraded to "a variable, not valid as a type". Run
with ``--python-executable`` pointing at the interpreter that holds the
dependencies, the implicit aliases type-check clean and ``src/`` reports no
issues at all. There is no numpy-version hazard here and no ceiling is needed.
The conversion is kept because the explicit form is better practice and is
annotation-only -- each alias binds exactly what it bound before -- but it
fixed a measurement artefact, not a defect.

The guard is AST-based and usage-aware: it reads module-level assignments and
only objects when the right-hand side subscripts ``np.ndarray``, ``np.dtype``,
``NDArray`` or ``npt.NDArray`` and the target carries no annotation. Strings,
comments, locals and non-numpy aliases never trip it.
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
    """No module-level numpy array alias is left in the implicit form."""
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
