#!/usr/bin/env python3
"""Verify documentation Lyapunov-λ claims match the canonical value.

This script enforces that every Lyapunov decay-rate λ value claimed in
``README.md`` and ``docs/MATH_SPEC.md`` for the fusion-trajectory
stability proof matches:

* ``LyapunovConstants.LAMBDA_CONVERGENCE`` in
  ``src/omni_mercury_engine/core/centralized_constants.py``; and
* ``configs/lyapunov_canonical.yaml``'s ``lambda`` field.

The gate matches three forms found in shipped documentation:

1. Greek inline math (``λ=0.25`` / ``λ = 0.25``).
2. LaTeX inline math (``$\\lambda = 0.25$`` and the bare ``\\lambda = 0.25``
   used inside Markdown code fences).
3. English prose (``lambda = 0.25``) -- but **only** when the surrounding
   window contains an explicit Lyapunov-stability anchor (see
   ``_LYAPUNOV_TOKENS``).  This prevents the unrelated double-helix
   evolution rate ``LAMBDA_DECAY = 0.18`` (a *different* λ, documented
   in ``core/double_helix_engine.py``) from being flagged.

Exit codes
----------
* ``0`` -- all matches agree with the canonical λ.
* ``1`` -- a mismatch was found OR a doc was supposed to contain a
  Lyapunov claim but none was detected (``--require-hits``).
* ``2`` -- usage / configuration error.

Usage
-----
::

    python scripts/check_readme_lyapunov.py
    python scripts/check_readme_lyapunov.py --files README.md docs/MATH_SPEC.md
    python scripts/check_readme_lyapunov.py --require-hits README.md
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Canonical-source resolution policy (in priority order):
#
#   1. ``configs/lyapunov_canonical.yaml`` -- the executable certificate.
#      Parsed with a stdlib regex on the single ``lambda:`` line so the
#      gate has **zero third-party dependencies** and can run on a bare
#      checkout (the ISO Hardening workflow's docs-lambda-drift job
#      installs neither numpy nor pyyaml -- the comment in the workflow
#      file explicitly promises ``stdlib-only``).
#   2. ``src/omni_mercury_engine/core/centralized_constants.py`` --
#      the Python constant, parsed via the same regex strategy.  Acts as
#      a second-line defence if the YAML is somehow malformed.
#   3. ``LyapunovConstants.LAMBDA_CONVERGENCE`` via package import --
#      only attempted if both regex paths fail.  Allowed to fail
#      silently; the caller can still pass ``--canonical`` explicitly.
#
# Equality between (1) and (2) is asserted by
# ``tests/tools/test_lyapunov_reconciliation.py``, so any one of them
# is sufficient for the drift gate; we honour the priority above so the
# YAML certificate stays the authoritative source.

_LAMBDA_YAML_PATTERN = re.compile(
    r"^\s*lambda\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:#.*)?$",
    re.MULTILINE,
)
_LAMBDA_CONST_PATTERN = re.compile(
    r"LAMBDA_CONVERGENCE\s*:\s*float\s*=\s*([0-9]+(?:\.[0-9]+)?)",
)


def _read_canonical_from_yaml() -> float | None:
    """Read ``configs/lyapunov_canonical.yaml``'s ``lambda:`` field.

    Implemented with a stdlib regex rather than ``yaml.safe_load`` so
    the gate stays dependency-free.  The canonical YAML has been
    structurally stable since v1.7.0 and is covered by
    ``tests/tools/test_lyapunov_validator.py`` (which DOES use yaml.safe_load),
    so a structural change to the file will fail that test before it
    silently breaks this regex.
    """
    canonical = _REPO_ROOT / "configs" / "lyapunov_canonical.yaml"
    try:
        text = canonical.read_text(encoding="utf-8")
    except OSError:
        return None
    m = _LAMBDA_YAML_PATTERN.search(text)
    if m is None:
        return None
    try:
        return float(m.group(1))
    except (TypeError, ValueError):  # pragma: no cover - regex guarantees float
        return None


def _read_canonical_from_constants() -> float | None:
    """Read ``LAMBDA_CONVERGENCE`` from ``centralized_constants.py``.

    Same stdlib-only strategy as ``_read_canonical_from_yaml``.  This
    function is the fallback the gate uses when the YAML can't be read.
    """
    src = _REPO_ROOT / "src" / "omni_mercury_engine" / "core" / "centralized_constants.py"
    try:
        text = src.read_text(encoding="utf-8")
    except OSError:
        return None
    m = _LAMBDA_CONST_PATTERN.search(text)
    if m is None:
        return None
    try:
        return float(m.group(1))
    except (TypeError, ValueError):  # pragma: no cover - regex guarantees float
        return None


def _read_canonical_from_package_import() -> float | None:
    """Last-resort: import the package and read the constant in-process."""
    for _candidate in (_REPO_ROOT, _REPO_ROOT / "src"):
        _candidate_str = str(_candidate)
        if _candidate_str not in sys.path:
            sys.path.insert(0, _candidate_str)
    try:
        from omni_mercury_engine.core.centralized_constants import LYAPUNOV

        return float(LYAPUNOV.LAMBDA_CONVERGENCE)
    except Exception:
        return None


def _resolve_default_canonical() -> float | None:
    """Resolve the canonical λ from the highest-priority available source."""
    for source in (
        _read_canonical_from_yaml,
        _read_canonical_from_constants,
        _read_canonical_from_package_import,
    ):
        value = source()
        if value is not None:
            return value
    return None


_DEFAULT_CANONICAL: float | None = _resolve_default_canonical()


# ---------------------------------------------------------------------------
# Pattern catalogue.
#
# Each entry is a single regex with exactly ONE capturing group for the
# numeric λ value.  Ordering does not affect correctness because overlapping
# spans are deduplicated by :func:`find_lambda_claims`; the entries are
# grouped here for readability.
#
# To grow the gate: add a new pattern below and either expand
# ``_LYAPUNOV_TOKENS`` (so the new prose is recognised as Lyapunov context)
# or ensure the existing tokens are already nearby.
# ---------------------------------------------------------------------------
_NUM = r"([0-9]+(?:\.[0-9]+)?)"

_LYAPUNOV_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 1. Greek inline math: "λ=0.25", "Λ = 0.25"
    re.compile(r"[λΛ]\s*=\s*" + _NUM),
    # 2. LaTeX inline math: "\lambda = 0.25" (with or without escape doubling).
    #    The leading "\\" matches a literal backslash in the source text.
    re.compile(r"\\[lL]ambda\s*=\s*" + _NUM),
    # 3. English-word form: "lambda = 0.25".  Word-boundary anchors prevent
    #    matching "LAMBDA_CONVERGENCE = 0.25" (constant assignment) which is
    #    enforced by a separate test (test_centralized_constants_*).
    re.compile(r"(?<![\w._])lambda\s*=\s*" + _NUM),
    # 4. Explicit symbolic key: "lambda_lyapunov = 0.25" used by the
    #    ``LyapunovAnomalyLoss`` constructor in code examples.
    re.compile(r"lambda_lyapunov\s*=\s*" + _NUM),
)

# Stability-context tokens.  A numeric match counts as a Lyapunov claim only
# when one of these appears within ``_CONTEXT_LINES`` lines of the match.
# ``LAMBDA_DECAY`` is intentionally NOT in this list -- the double-helix
# evolution rate is a separate constant in
# ``core/double_helix_engine.py`` and must not be conflated with the
# fusion-trajectory Lyapunov convergence rate.
_LYAPUNOV_TOKENS: tuple[str, ...] = (
    "Lyapunov",
    "lyapunov",
    "LYAPUNOV",
    "LAMBDA_CONVERGENCE",
    "stability envelope",
    "lambda_lyapunov",
    "V̇",
    r"\dot{V}",
    "exponential decay",
    "convergence rate",  # so the README "convergence rate λ=0.25" anchors itself
)

# Per-match exclusions are evaluated against the immediate left context
# of the captured numeric span: a number that is preceded by one of these
# tokens is the *historical* value, not the current canonical claim, and
# the gate must not enforce equality on it.
#
# ``elevated from`` -- e.g. "λ=0.25 (elevated from 0.18 ...)": 0.25 is the
#   active claim, 0.18 is the historical reference and must be excluded
#   only when it appears *after* "elevated from", not when it appears
#   *before*.
# ``LAMBDA_DECAY`` -- the unrelated double-helix evolutionary adaptation
#   rate (currently 0.18); a separate constant in
#   ``core/double_helix_engine.py``.
_LEFT_EXCLUSION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"elevated\s+from[^\d]*$", re.IGNORECASE),
    re.compile(r"LAMBDA_DECAY[^\d]*$", re.IGNORECASE),
    re.compile(r"\bdouble[- ]helix[^\d]*$", re.IGNORECASE),
)

_CONTEXT_LINES = 4


def _line_has_context(lines: Sequence[str], idx: int) -> bool:
    lo = max(0, idx - _CONTEXT_LINES)
    hi = min(len(lines), idx + _CONTEXT_LINES + 1)
    window = "\n".join(lines[lo:hi])
    return any(tok in window for tok in _LYAPUNOV_TOKENS)


def _match_is_excluded(line: str, match: re.Match[str]) -> bool:
    """Return True if a Lyapunov-shaped match is in an exclusion context.

    The check is *directional*: only the line content immediately to the
    LEFT of the captured numeric span is examined, so a leading "elevated
    from " phrase excludes the historical value that follows it without
    also excluding the current value that the phrase referred back to.
    """
    span_start, _ = match.span(1)
    left = line[:span_start]
    return any(p.search(left) for p in _LEFT_EXCLUSION_PATTERNS)


def find_lambda_claims(text: str) -> list[tuple[int, str, float]]:
    """Yield ``(line_no, line_text, claimed_lambda)`` triples.

    Only matches in a Lyapunov stability context (and not in any of the
    exclusion contexts) are returned.  Line numbers are 1-based.
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
                if _match_is_excluded(line, m):
                    continue
                try:
                    val = float(m.group(1))
                except (TypeError, ValueError):
                    continue
                results.append((i + 1, line.strip(), val))
    return results


def check_files(
    files: Iterable[Path],
    canonical: float,
    tol: float = 1e-9,
    require_hits: Iterable[Path] | None = None,
) -> list[str]:
    """Return a list of human-readable error messages (empty == OK).

    ``require_hits`` (optional) is the set of files in which at least one
    Lyapunov-context match MUST be detected; this guards against the
    failure mode where the regex silently drifts away from the actual prose
    form used in the docs (the test would otherwise pass vacuously).
    """
    errors: list[str] = []
    required = {Path(p).resolve() for p in (require_hits or ())}
    for path in files:
        if not path.exists():
            errors.append(f"{path}: file not found")
            continue
        text = path.read_text(encoding="utf-8")
        hits = find_lambda_claims(text)
        for line_no, line, val in hits:
            if not math.isclose(val, canonical, rel_tol=0, abs_tol=tol):
                errors.append(
                    f"{path}:{line_no}: claimed λ={val} != canonical " f"λ={canonical} -- {line!r}"
                )
        if Path(path).resolve() in required and not hits:
            errors.append(
                f"{path}: no Lyapunov-context λ claim detected, but this "
                "file is on the --require-hits list (gate would be "
                "vacuous; add a pattern to _LYAPUNOV_PATTERNS or remove "
                "this file from --require-hits)."
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
            "LyapunovConstants.LAMBDA_CONVERGENCE; required when the "
            "constants module cannot be imported."
        ),
    )
    parser.add_argument(
        "--require-hits",
        nargs="*",
        type=Path,
        default=[_REPO_ROOT / "README.md", _REPO_ROOT / "docs" / "MATH_SPEC.md"],
        help=(
            "Files that MUST yield at least one Lyapunov-context λ match. "
            "Empty list disables the vacuous-gate guard. Defaults to the "
            "shipped README and MATH_SPEC."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.canonical is not None:
        canonical = float(args.canonical)
    elif _DEFAULT_CANONICAL is not None:
        canonical = _DEFAULT_CANONICAL
    else:
        print(
            "ERROR: cannot import LyapunovConstants.LAMBDA_CONVERGENCE "
            "and --canonical was not provided.",
            file=sys.stderr,
        )
        return 2
    errors = check_files(args.files, canonical, require_hits=args.require_hits)
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
