#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Forbid raw ``torch.load`` outside the sanctioned safe wrapper.

``torch.load`` deserializes a Python pickle; without ``weights_only=True`` a
malicious checkpoint executes arbitrary code at load time (RCE). Mercury
routes every checkpoint load through
:func:`omni_mercury_engine.security.safe_torch.safe_torch_load`, which
hard-pins ``weights_only=True`` and validates the source. That security
property is only meaningful if it cannot be bypassed, so this gate fails the
build when any ``src/`` module calls ``torch.load(`` directly.

Detection uses :mod:`tokenize`, so ``torch.load`` appearing inside a
docstring or comment (e.g. explaining the convention) is *not* flagged —
only real call expressions are.

An explicit, reviewable allow-list (:data:`ALLOWLIST`) names the files
permitted to call ``torch.load`` directly. It contains exactly one entry:
the wrapper itself. Adding to it requires a code review and a stated reason,
which is the whole point — a new unsafe load can never land silently.

Exit codes
----------
* ``0`` — no raw ``torch.load`` calls outside the allow-list.
* ``1`` — at least one raw ``torch.load`` call found (locations listed).
* ``2`` — a scanned file could not be tokenized (syntax error).
"""

from __future__ import annotations

import argparse
import sys
import tokenize
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Only these files may call ``torch.load`` directly. The wrapper is the one
#: sanctioned home; every other call site must use ``safe_torch_load``.
ALLOWLIST: frozenset[str] = frozenset(
    {
        "src/omni_mercury_engine/security/safe_torch.py",
    }
)

DEFAULT_ROOT = Path("src")

EXCLUDED_DIR_PARTS = frozenset(
    {
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "__pycache__",
        "build",
        "dist",
    }
)


def _iter_python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        if EXCLUDED_DIR_PARTS & set(path.parts):
            continue
        files.append(path)
    return files


def find_torch_load_calls(path: Path) -> list[int]:
    """Return the 1-based line numbers of real ``torch.load(`` calls.

    Only NAME/OP token sequences ``torch . load (`` count; occurrences in
    strings (docstrings) and comments are ignored because :mod:`tokenize`
    classifies them as STRING/COMMENT tokens, never NAME.
    """
    with path.open("rb") as fh:
        tokens = list(tokenize.tokenize(fh.readline))

    # Keep only significant tokens (drop whitespace/newline/indent noise) so a
    # line break between ``torch.load`` and ``(`` does not hide the call.
    significant = [
        tok
        for tok in tokens
        if tok.type
        not in (
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.COMMENT,
            tokenize.ENCODING,
            tokenize.ENDMARKER,
        )
    ]

    hits: list[int] = []
    for i in range(len(significant) - 3):
        a, b, c, d = significant[i : i + 4]
        if (
            a.type == tokenize.NAME
            and a.string == "torch"
            and b.type == tokenize.OP
            and b.string == "."
            and c.type == tokenize.NAME
            and c.string == "load"
            and d.type == tokenize.OP
            and d.string == "("
        ):
            hits.append(a.start[0])
    return hits


def scan(root: Path) -> tuple[list[tuple[str, int]], list[str]]:
    """Scan ``root`` for raw ``torch.load`` calls outside the allow-list.

    Returns ``(violations, tokenize_errors)`` where each violation is a
    ``(repo_relative_path, lineno)`` tuple.
    """
    violations: list[tuple[str, int]] = []
    errors: list[str] = []
    for path in _iter_python_files(root):
        try:
            rel = path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            # Scanned root lives outside the repo (e.g. an ad-hoc --root in a
            # test); fall back to the raw path. Allow-list matching, which is
            # keyed on repo-relative paths, then simply never applies.
            rel = path.as_posix()
        try:
            hits = find_torch_load_calls(path)
        except (tokenize.TokenError, SyntaxError, UnicodeDecodeError) as exc:
            errors.append(f"{rel}: could not tokenize ({exc})")
            continue
        if not hits:
            continue
        if rel in ALLOWLIST:
            continue
        violations.extend((rel, lineno) for lineno in hits)
    return violations, errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="directory to scan (default: src)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = args.root if args.root.is_absolute() else REPO_ROOT / args.root
    if not root.exists():
        print(f"ERROR: scan root does not exist: {root}", file=sys.stderr)
        return 2

    violations, errors = scan(root)

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 2

    if violations:
        print(
            "FAIL: raw torch.load() found outside the safe wrapper. Route these "
            "through omni_mercury_engine.security.safe_torch.safe_torch_load "
            "(weights_only is enforced there):",
            file=sys.stderr,
        )
        for rel, lineno in violations:
            print(f"  {rel}:{lineno}", file=sys.stderr)
        return 1

    n_scanned = len(_iter_python_files(root))
    print(
        f"OK: no raw torch.load() outside the safe wrapper "
        f"({n_scanned} files scanned under {root.relative_to(REPO_ROOT)}; "
        f"{len(ALLOWLIST)} allow-listed)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
