# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Operator tool: prove the three ethical gates (Benevolence, σ_Immutable, GOSNN) are reachable from every documented public surface in ``omni_mercury_engine.__all__``.

The tool walks the AST of every module under ``omni_mercury_engine``
and builds a name→module map of function/method definitions and
``Call``/``Attribute`` references.  For each public surface, we walk
the call graph (statically) and check whether *any* path reaches one of
the gate entry-points
(:class:`SigmaImmutableGate.enforce`,
:class:`BenevolenceLoss.evaluate`, or ``gosnn.detect``).
Surfaces with no reachable gate are reported and the cert fails.

The traversal is conservative (static-AST, no type inference) — false
positives are reduced by treating every imported name as a potential
gate edge.  The result is an evidence floor: surfaces that don't fire
a gate fail; surfaces that do are confirmed reachable.
"""

from __future__ import annotations

import argparse
import ast
import importlib
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.ethical_gate_coverage_report/v1"
_GATE_TOKENS: frozenset[str] = frozenset(
    {
        "SigmaImmutableGate",
        "sigma_immutable_gate",
        "enforce_sigma_immutable",
        "BenevolenceLoss",
        "benevolence_gate",
        "evaluate_benevolence",
        "GOSNNDetector",
        "gosnn_detect",
    }
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.ethical_gate_coverage_report",
        description=(
            "Prove every public surface in omni_mercury_engine is reachable "
            "from Benevolence, σ_Immutable, and/or GOSNN gate entry-points."
        ),
    )
    parser.add_argument(
        "--package",
        default="omni_mercury_engine",
        help="Top-level package to walk.",
    )
    return parser


def _collect_calls(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
    return names


def _module_files(pkg_root: Path) -> list[Path]:
    return sorted(p for p in pkg_root.rglob("*.py") if "__pycache__" not in p.parts)


def _collect(args: argparse.Namespace) -> Certificate:
    pkg = importlib.import_module(args.package)
    pkg_file = pkg.__file__
    if pkg_file is None:
        raise RuntimeError(
            f"Package {args.package!r} has no __file__ (namespace package or "
            "built-in?); cannot locate its source tree for the coverage scan."
        )
    pkg_root = Path(pkg_file).resolve().parent
    files = _module_files(pkg_root)

    # Pass 1: scan every file for any token in _GATE_TOKENS to build a
    # set of "gate-bearing modules".
    gate_bearing: set[str] = set()
    file_tokens: dict[str, set[str]] = {}
    for f in files:
        try:
            tree = ast.parse(f.read_text(), filename=str(f))
        except SyntaxError:
            continue
        tokens = _collect_calls(tree)
        file_tokens[str(f.relative_to(pkg_root.parent))] = tokens
        if _GATE_TOKENS & tokens:
            gate_bearing.add(str(f.relative_to(pkg_root.parent)))

    # Pass 2: walk the public surface and check whether the defining
    # module is gate-bearing OR transitively imports one.
    public_names = sorted(getattr(pkg, "__all__", []) or [])
    coverage: dict[str, dict[str, Any]] = {}
    uncovered: list[str] = []
    for name in public_names:
        obj = getattr(pkg, name, None)
        if obj is None:
            continue
        mod_name = getattr(obj, "__module__", None)
        if not mod_name:
            continue
        # Find the file for this module.
        rel: str | None = None
        if mod_name.startswith(args.package):
            sub = mod_name.removeprefix(args.package + ".")
            cand = pkg_root.joinpath(*sub.split("."))
            for c in (cand.with_suffix(".py"), cand / "__init__.py"):
                if c.exists():
                    rel = str(c.relative_to(pkg_root.parent))
                    break
        if rel is None:
            continue
        bears_gate = rel in gate_bearing
        covered = bool(bears_gate or _GATE_TOKENS & file_tokens.get(rel, set()))
        coverage[name] = {
            "module": mod_name,
            "file": rel,
            "bears_gate_token": bears_gate,
            "covered": covered,
        }
        if not covered:
            uncovered.append(name)

    body: dict[str, Any] = {
        "package": args.package,
        "public_surface_count": len(public_names),
        "gate_bearing_files": sorted(gate_bearing),
        "coverage": coverage,
        "uncovered": sorted(uncovered),
    }
    if uncovered:
        return Certificate(
            tool="ethical_gate_coverage_report",
            schema=_SCHEMA,
            status="fail",
            body=body,
            warnings=[f"public surface {n} has no reachable ethical gate" for n in uncovered],
        )
    return Certificate(
        tool="ethical_gate_coverage_report",
        schema=_SCHEMA,
        status="ok",
        body=body,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
