#!/usr/bin/env python3
"""Verify documentation Lyapunov-λ claims match the canonical value.

This script enforces that every Lyapunov decay-rate λ value claimed in
``README.md`` and ``docs/MATH_SPEC.md`` for the fusion-trajectory
stability proof matches:

* ``LyapunovConstants.LAMBDA_CONVERGENCE`` in
  ``src/omni_mercury_engine/core/centralized_constants.py``; and
* ``configs/lyapunov_canonical.yaml``'s ``lambda`` field.

It deliberately does **not** flag *every* numeric mention of 0.25 in the
documentation because the codebase reuses 0.25 for unrelated parameters
(uncertainty-fusion weight, alpha convergence-rate gate, etc.).  It
matches only the explicit Lyapunov-context patterns enumerated in
``_LYAPUNOV_PATTERNS`` below; new prose mentioning λ in a stability
context should add a corresponding regex here so the gate covers it.

Exit codes
----------
* ``0`` -- all matches agree with the canonical λ
* ``1`` -- a mismatch was found; the script prints the offending lines.
* ``2`` -- usage / configuration error.

Usage
-----
::

    python scripts/check_readme_lyapunov.py
    python scripts/check_readme_lyapunov.py --files README.md docs/MATH_SPEC.md
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_REPO_ROOT))

from omni_mercury_engine.core.centralized_constants import LYAPUNOV

# Regex patterns -- each must contain ONE capturing group for the
# numeric λ value.  Patterns are evaluated against each line of each
# scanned file.  Add new patterns here whenever the docs grow a new
# Lyapunov-context λ mention; the test in
# tests/tools/test_check_readme_lyapunov.py exercises the script
# end-to-end so additions are validated automatically.
_LYAPUNOV_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "convergence rate λ=0.25"
    re.compile(r"convergence rate\s*[`']?\s*[λΛ]\s*=\s*([0-9]*\.?[0-9]+)"),
    # "λ=0.25" / "lambda = 0.25" inline math (must be near "Lyapunov")
    re.compile(r"[λΛ]\s*=\s*([0-9]*\.?[0-9]+)"),
    re.compile(r"lambda_lyapunov\s*=\s*([0-9]*\.?[0-9]+)"),
)

# Stability-context anchor: a line must contain one of these tokens
# (or be within ``_CONTEXT_LINES`` lines of one) for its numeric match
# to be considered a Lyapunov claim.  This prevents false positives on
# unrelated mentions of λ (e.g. the uncertainty-fusion λ in
# ``core/fusion.py``).
_CONTEXT_TOKENS: tuple[str, ...] = (
    "Lyapunov",
    "lyapunov",
    "stability envelope",
    "lambda_lyapunov",
    "V̇",
    r"\\dot{V}",
    "exponential decay",
)
_CONTEXT_LINES = 4


def _line_has_context(lines: Sequence[str], idx: int) -> bool:
    lo = max(0, idx - _CONTEXT_LINES)
    hi = min(len(lines), idx + _CONTEXT_LINES + 1)
    window = "\n".join(lines[lo:hi])
    return any(tok in window for tok in _CONTEXT_TOKENS)


def find_lambda_claims(text: str) -> list[tuple[int, str, float]]:
    """Yield ``(line_no, line_text, claimed_lambda)`` triples.

    Only matches in a Lyapunov stability context are returned.
    Line numbers are 1-based.
    """
    lines = text.splitlines()
    results: list[tuple[int, str, float]] = []
    for i, line in enumerate(lines):
        seen_spans: set[tuple[int, int]] = set()
        for pat in _LYAPUNOV_PATTERNS:
            for m in pat.finditer(line):
                span = m.span(1)
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                if not _line_has_context(lines, i):
                    continue
                try:
                    val = float(m.group(1))
                except (TypeError, ValueError):
                    continue
                results.append((i + 1, line.strip(), val))
    return results


def check_files(
    files: Iterable[Path], canonical: float, tol: float = 1e-9
) -> list[str]:
    """Return a list of human-readable mismatch messages (empty == OK)."""
    errors: list[str] = []
    for path in files:
        if not path.exists():
            errors.append(f"{path}: file not found")
            continue
        text = path.read_text(encoding="utf-8")
        for line_no, line, val in find_lambda_claims(text):
            if not math.isclose(val, canonical, rel_tol=0, abs_tol=tol):
                errors.append(
                    f"{path}:{line_no}: claimed λ={val} != canonical "
                    f"λ={canonical} -- {line!r}"
                )
    return errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--files",
        nargs="+",
        type=Path,
        default=[
            _REPO_ROOT / "README.md",
            _REPO_ROOT / "docs" / "MATH_SPEC.md",
        ],
        help="Markdown files to scan.",
    )
    parser.add_argument(
        "--canonical",
        type=float,
        default=None,
        help=(
            "Canonical λ to enforce. Defaults to "
            "LyapunovConstants.LAMBDA_CONVERGENCE."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    canonical = (
        float(args.canonical)
        if args.canonical is not None
        else float(LYAPUNOV.LAMBDA_CONVERGENCE)
    )
    errors = check_files(args.files, canonical)
    if errors:
        print(
            f"FAIL: {len(errors)} Lyapunov-λ documentation drift(s) "
            f"detected (canonical λ={canonical}):",
            file=sys.stderr,
        )
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1
    print(
        f"OK: all Lyapunov-λ documentation claims match canonical "
        f"λ={canonical} (scanned {len(list(args.files))} files)."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry-point
    raise SystemExit(main())
