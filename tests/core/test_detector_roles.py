# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every public ``*Detector`` class is a manifest entry or has a verified role.

``core/detector_roles.py`` says what each class outside the manifest is; this
re-derives each claim from the source, so the table cannot drift from the code
and a new ``*Detector`` class cannot go unclassified.
"""

from __future__ import annotations

import ast
import importlib
import re
from functools import cache
from pathlib import Path

import pytest

from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST
from omni_mercury_engine.core.detector_roles import DETECTOR_ROLES

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
ROLES = sorted(DETECTOR_ROLES.items())


def _module_file(module: str) -> Path:
    path = SRC.joinpath(*module.split("."))
    return path / "__init__.py" if path.is_dir() else path.with_suffix(".py")


@cache
def _tree(module: str) -> ast.Module:
    return ast.parse(_module_file(module).read_text())


@cache
def _all_classes() -> dict[str, tuple[str, ast.ClassDef]]:
    """``module:Class`` -> (module, node) for every top-level class in ``src``."""
    found: dict[str, tuple[str, ast.ClassDef]] = {}
    for path in (SRC / "omni_mercury_engine").rglob("*.py"):
        rel = path.relative_to(SRC).with_suffix("")
        module = ".".join(rel.parts[:-1] if rel.name == "__init__" else rel.parts)
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ClassDef):
                found[f"{module}:{node.name}"] = (module, node)
    return found


def _split(ref: str) -> tuple[str, str]:
    module, _, name = ref.partition(":")
    return module, name


def _calls(node: ast.AST, name: str) -> bool:
    """Whether anything under ``node`` calls ``name(...)`` (or ``x.name(...)``)."""
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            called = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if called == name:
                return True
    return False


def _resolve(module: str, name: str) -> str:
    """``module:name`` as the class's defining module spells it.

    A manifest entry may name a class by the package that re-exports it; the
    source scan knows only where it is defined.
    """
    try:
        owner = importlib.import_module(module)
        return f"{getattr(owner, name).__module__}:{name}"
    except (ImportError, AttributeError):
        defined = [ref for ref in _all_classes() if ref.endswith(f":{name}")]
        assert len(defined) == 1, f"cannot resolve {module}:{name}"
        return defined[0]


def _manifest_classes() -> set[str]:
    refs = set()
    for entry in DETECTOR_MANIFEST:
        refs.add(_resolve(entry.module_path, entry.class_name))
        if entry.wraps:
            refs.add(_resolve(*_split(entry.wraps)))
    return refs


def _public_detector_classes() -> set[str]:
    return {
        ref
        for ref, (_, node) in _all_classes().items()
        if node.name.endswith("Detector") and not node.name.startswith("_")
    }


def test_every_detector_class_is_a_manifest_entry_or_has_a_role() -> None:
    """The rule that stops an unclassified detector landing unnoticed."""
    unaccounted = _public_detector_classes() - _manifest_classes() - set(DETECTOR_ROLES)
    assert not unaccounted, (
        "add each to DETECTOR_MANIFEST (a seam or a library-only entry) or to "
        f"core/detector_roles.py with a verified role: {sorted(unaccounted)}"
    )


def test_no_class_is_both_a_manifest_entry_and_in_the_role_table() -> None:
    both = _manifest_classes() & set(DETECTOR_ROLES)
    assert not both, f"a manifest class needs no role: {sorted(both)}"


@pytest.mark.parametrize(("ref", "role"), ROLES, ids=[ref.split(":")[-1] for ref, _ in ROLES])
def test_the_role_claim_is_true_of_the_source(ref: str, role: object) -> None:
    from omni_mercury_engine.core.detector_roles import DetectorRole

    assert isinstance(role, DetectorRole)
    classes = _all_classes()
    assert ref in classes, f"{ref} no longer exists: remove its role"
    module, node = classes[ref]

    if role.kind == "abstract":
        bases = {ast.unparse(b).split(".")[-1] for b in node.bases}
        has_abstract = any(
            isinstance(m, ast.FunctionDef)
            and any("abstractmethod" in ast.unparse(d) for d in m.decorator_list)
            for m in node.body
        )
        assert bases & {"ABC", "Protocol"} or has_abstract, f"{ref} is concrete"
        assert not role.parts

    elif role.kind == "component":
        assert role.parts, f"{ref}: name the class(es) that construct it"
        for part in role.parts:
            assert part in classes, f"{part} (a constructor of {ref}) does not exist"
            assert _calls(classes[part][1], node.name), f"{part} never constructs {node.name}"

    elif role.kind == "variant":
        (base,) = role.parts
        assert base in classes, f"{base} does not exist"
        bases = {ast.unparse(b).split(".")[-1] for b in node.bases}
        assert _split(base)[1] in bases, f"{node.name} does not subclass {_split(base)[1]}"

    elif role.kind == "deprecated":
        text = _module_file(module).read_text()
        assert (
            "DeprecationWarning" in text and "warnings.warn(" in text
        ), f"{module} is not deprecated"

    elif role.kind == "library":
        assert not role.parts
        assert any(
            isinstance(m, ast.FunctionDef) and not m.name.startswith("_") for m in node.body
        ), f"{ref} has no public API of its own"
        tests = [
            p
            for p in (REPO / "tests").rglob("*.py")
            if re.search(rf"\b{node.name}\b", p.read_text())
        ]
        assert tests, f"{ref} is a library detector with no test referencing it"


_NEEDS_PROVENANCE = [(ref, role) for ref, role in ROLES if role.kind in ("library", "variant")]


@pytest.mark.parametrize(
    "ref",
    [ref for ref, _ in _NEEDS_PROVENANCE],
    ids=[r.split(":")[-1] for r, _ in _NEEDS_PROVENANCE],
)
def test_a_library_or_variant_class_declares_where_its_alarm_rule_comes_from(ref: str) -> None:
    """A class a caller can use directly must say what backs its decision rule.

    (A variant inherits its base class's declaration; ``getattr`` follows that.)
    """
    from omni_mercury_engine.detectors.provenance import DecisionSource

    module, name = _split(ref)
    cls = getattr(importlib.import_module(module), name)
    source = getattr(cls, "decision_source", None)
    assert source is not None, f"{ref} declares no decision_source"
    records = source if isinstance(source, tuple) else (source,)
    assert records, f"{ref}: decision_source is an empty tuple"
    assert all(isinstance(r, DecisionSource) for r in records)
