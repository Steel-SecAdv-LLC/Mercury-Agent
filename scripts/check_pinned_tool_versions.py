#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pinned developer-tool version parity gate.

Several quality tools are pinned *exactly* (not range-floored) so the
formatter / type-checker / docstring ruleset behave byte-identically for a
developer running them locally and for CI running them on a pull request:

* ``black``          -- byte-identical auto-formatting
* ``mypy``           -- identical ``# type: ignore`` set
* ``types-requests`` -- bumped in lockstep with ``mypy``
* ``pydocstyle``     -- identical google-convention docstring codes
* ``ruff``           -- identical lint ruleset (an unpinned CI install picked
  up 0.16.0 the day it released and turned 237 newly-stabilised PLR0917
  findings into a repo-wide Code Quality red -- the incident that widened
  this gate's registry)
* ``flake8``         -- identical pycodestyle/pyflakes rule surface (it
  floated in ``ci.yml`` while pre-commit pinned 7.0.0 -- the same drift
  class as the ruff incident, closed by the same registry widening)

Those exact pins live across four independent surfaces -- though each tool
appears only in the subset that actually installs or runs it, not in all four:

* ``pyproject.toml``                 (``[project.optional-dependencies]`` ml/dev)
* ``.github/workflows/ci.yml``       (Code Quality + Type Checking jobs)
* ``.github/workflows/format.yml``   (Auto-Format job)
* ``.pre-commit-config.yaml``        (hook ``rev:`` pins)

Concretely: ``black`` is pinned in all four; ``mypy`` in three (not
``format.yml``); ``types-requests`` only where ``mypy`` is installed
(``pyproject.toml`` + ``ci.yml`` -- in pre-commit it is an *un*pinned
``additional_dependencies`` entry, not a ``rev`` pin); ``pydocstyle`` in
``ci.yml`` + ``.pre-commit-config.yaml`` (it has no ``pyproject`` dependency
pin, only a ``[tool.pydocstyle]`` config block); ``ruff`` in
``pyproject.toml`` + ``ci.yml`` + ``.pre-commit-config.yaml``; ``flake8`` in
the same three.  The gate therefore compares each tool's pins *wherever they
are pinned* and requires every tracked tool to stay pinned in at least two
surfaces (see ``TRACKED_TOOLS``) so a lone, drift-prone pin cannot slip
through.

Dependabot's ``pip`` ecosystem only rewrites ``pyproject.toml``; it cannot
reach the inline ``pip install black==X`` strings in the workflow ``run:``
blocks, and nothing else kept them in lockstep.  The result was a silent
divergence (pre-commit on one black, CI on another) -- exactly the drift the
"PINNED exact for byte-identical formatting local<->CI" comments were meant to
prevent.

This gate makes the lockstep structural: it extracts every *exact* pin of each
tracked tool from all four surfaces and fails closed if any tool's pins
disagree, or if a tool's pin thinned out to a single surface where it could
drift unnoticed (so a vacuous "no pins, no drift" pass is impossible).

Pure stdlib and import-free of :mod:`omni_mercury_engine` so it runs in
pre-commit and the ``workflow-hardening`` CI job without the package (and its
mandatory PQC backend) installed.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Tools pinned *exactly* for local<->CI parity, mapped to the minimum number of
# distinct files each must still be pinned in.  The floor is what turns a
# single-surface pin (which can drift unnoticed) into a failure rather than a
# silently-accepted state, and stops a "delete every pin" edit from passing
# vacuously.  Current reality: black=4 surfaces, mypy=3, types-requests=2,
# pydocstyle=2, ruff=3, flake8=3 -- the floor of 2 leaves intentional slack
# while still catching any collapse to a lone, drift-prone surface.
TRACKED_TOOLS: dict[str, int] = {
    "black": 2,
    "mypy": 2,
    "types-requests": 2,
    "pydocstyle": 2,
    "ruff": 2,
    "flake8": 2,
}

# Requirement-style pin, e.g. ``black==26.5.1`` / ``"mypy==2.1.0"`` /
# ``types-requests==2.33.0.20260518``.  Quoting is irrelevant because we anchor
# on the package name and the ``==`` operator.  The leading negative lookbehind
# stops ``some-black==`` matching ``black``; a prose mention without ``==``
# (``mypy 1.19 and mypy 2.1``) is never matched.
_REQ_PIN_RE = re.compile(
    r"(?<![\w.-])(?P<name>black|mypy|types-requests|pydocstyle|ruff|flake8)"
    r"==(?P<ver>[0-9][0-9A-Za-z._-]*)"
)

# pre-commit ``rev:`` pins are mapped to a tool by the hook repo URL substring.
_PRECOMMIT_REPO_TOOL: dict[str, str] = {
    "psf/black": "black",
    "mirrors-mypy": "mypy",
    "pydocstyle": "pydocstyle",
    "ruff-pre-commit": "ruff",
    "pycqa/flake8": "flake8",
}
_PRECOMMIT_REPO_RE = re.compile(r"^\s*-\s*repo:\s*(?P<url>\S+)")
_PRECOMMIT_REV_RE = re.compile(r"^\s*rev:\s*['\"]?(?P<ver>\S+?)['\"]?\s*$")

# Files scanned for requirement-style ``name==version`` pins, relative to root.
_REQ_SURFACES: tuple[str, ...] = (
    "pyproject.toml",
    ".github/workflows/ci.yml",
    ".github/workflows/format.yml",
)
_PRECOMMIT_SURFACE = ".pre-commit-config.yaml"


class Occurrence(NamedTuple):
    """A single exact pin found in a surface file."""

    tool: str
    version: str
    location: str  # ``relative/path:lineno``


def _normalize(version: str) -> str:
    """Strip a single leading ``v``/``V`` so ``v2.1.0`` compares equal to ``2.1.0``."""
    return version[1:] if version[:1] in {"v", "V"} else version


def _scan_requirements(path: Path, rel: str) -> list[Occurrence]:
    """Collect ``name==version`` pins for tracked tools from a pyproject/workflow file."""
    if not path.exists():
        return []
    out: list[Occurrence] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for m in _REQ_PIN_RE.finditer(line):
            out.append(Occurrence(m.group("name"), _normalize(m.group("ver")), f"{rel}:{lineno}"))
    return out


def _scan_precommit(path: Path, rel: str) -> list[Occurrence]:
    """Collect hook ``rev:`` pins for tracked tools from ``.pre-commit-config.yaml``."""
    if not path.exists():
        return []
    out: list[Occurrence] = []
    current_tool: str | None = None
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        repo_match = _PRECOMMIT_REPO_RE.match(line)
        if repo_match:
            url = repo_match.group("url").lower()
            current_tool = next(
                (tool for needle, tool in _PRECOMMIT_REPO_TOOL.items() if needle in url),
                None,
            )
            continue
        if current_tool is not None:
            rev_match = _PRECOMMIT_REV_RE.match(line)
            if rev_match:
                out.append(
                    Occurrence(current_tool, _normalize(rev_match.group("ver")), f"{rel}:{lineno}")
                )
                current_tool = None
    return out


def collect_occurrences(root: Path) -> list[Occurrence]:
    """Gather every exact tracked-tool pin across all parity surfaces under ``root``."""
    occurrences: list[Occurrence] = []
    for rel in _REQ_SURFACES:
        occurrences.extend(_scan_requirements(root / rel, rel))
    occurrences.extend(_scan_precommit(root / _PRECOMMIT_SURFACE, _PRECOMMIT_SURFACE))
    return occurrences


def find_violations(occurrences: list[Occurrence]) -> list[str]:
    """Return human-readable violation strings; empty list means parity holds."""
    errors: list[str] = []
    for tool, min_files in TRACKED_TOOLS.items():
        tool_occ = sorted((o for o in occurrences if o.tool == tool), key=lambda o: o.location)
        if not tool_occ:
            errors.append(
                f"{tool}: no exact pin found in any tracked surface "
                f"(expected it pinned in >= {min_files} files)"
            )
            continue
        versions = {o.version for o in tool_occ}
        files = {o.location.rsplit(":", 1)[0] for o in tool_occ}
        if len(versions) > 1:
            detail = "; ".join(f"{o.location} -> {o.version}" for o in tool_occ)
            errors.append(f"{tool}: pin drift {sorted(versions)} across surfaces [{detail}]")
        if len(files) < min_files:
            errors.append(
                f"{tool}: pinned in only {len(files)} surface(s) {sorted(files)}; expected "
                f">= {min_files} so a single-surface pin cannot drift unnoticed"
            )
    return errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_pinned_tool_versions.py",
        description=(
            "Assert that exactly-pinned dev tools (black, mypy, types-requests, "
            "pydocstyle, ruff, flake8) carry identical versions across pyproject.toml, "
            "the CI/format workflows, and .pre-commit-config.yaml."
        ),
    )
    parser.add_argument(
        "--root",
        default=str(_REPO_ROOT),
        help="Repository root to scan (default: detected from this script's location).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point.  Returns 0 when parity holds, 1 on any drift/erosion."""
    args = _build_parser().parse_args(argv)
    root = Path(args.root).resolve()

    occurrences = collect_occurrences(root)
    errors = find_violations(occurrences)

    if errors:
        print("Pinned tool-version parity check FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print(
            "\nThese tools are pinned exactly for byte-identical behaviour local<->CI.\n"
            "Bump every surface together: pyproject.toml, .github/workflows/ci.yml,\n"
            ".github/workflows/format.yml, and .pre-commit-config.yaml.",
            file=sys.stderr,
        )
        return 1

    summary = ", ".join(
        f"{tool}=={next(o.version for o in occurrences if o.tool == tool)}"
        for tool in TRACKED_TOOLS
    )
    print(f"Pinned tool-version parity check passed ({summary}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
