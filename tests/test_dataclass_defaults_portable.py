# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""No dataclass field defaults to a mappingproxy instance.

Before Python 3.12 a ``types.MappingProxyType`` is unhashable, so
``dataclasses`` rejects it as a mutable default with ``ValueError`` at class
creation, and importing the module fails. Python 3.12+ accepts it, so the
defect is invisible on newer interpreters; this static check catches it on
any version. Use ``field(default_factory=lambda: PROXY)`` instead.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "omni_mercury_engine"


def _is_proxy_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    return name == "MappingProxyType"


def _is_dataclass(cls: ast.ClassDef) -> bool:
    for deco in cls.decorator_list:
        target = deco.func if isinstance(deco, ast.Call) else deco
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
        if name == "dataclass":
            return True
    return False


def _offenders(path: Path, root: Path = SRC.parent) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    proxies = {
        target.id
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and node.value is not None
        and _is_proxy_call(node.value)
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }
    found = []
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef) or not _is_dataclass(cls):
            continue
        for stmt in cls.body:
            if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                value = stmt.value
                if _is_proxy_call(value) or (isinstance(value, ast.Name) and value.id in proxies):
                    found.append(f"{path.relative_to(root)}:{stmt.lineno} {cls.name}")
    return found


def test_no_dataclass_field_defaults_to_a_mappingproxy() -> None:
    offenders = [o for path in sorted(SRC.rglob("*.py")) for o in _offenders(path)]
    assert offenders == [], offenders


def test_the_check_catches_the_pattern(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text(
        "from dataclasses import dataclass\n"
        "from types import MappingProxyType\n"
        "TABLE = MappingProxyType({'a': 1})\n"
        "@dataclass(frozen=True)\n"
        "class C:\n"
        "    table: object = TABLE\n"
        "    inline: object = MappingProxyType({})\n"
    )
    assert _offenders(bad, root=tmp_path) == ["bad.py:6 C", "bad.py:7 C"]
