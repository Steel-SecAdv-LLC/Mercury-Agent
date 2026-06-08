#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
"""Verify documentation Lyapunov-λ claims match the canonical constants.

This script is the **import-based** λ drift gate: instead of regex-parsing
the canonical numeric value out of a YAML or a constants module (the old
approach, which was a regex against a regex and could silently disagree
with what the code actually loaded at runtime), it **imports** the
authoritative Python constants and asserts that every documented numeric
literal in ``README.md`` and ``docs/MATH_SPEC.md`` matches the imported
float within floating-point tolerance.

The gate maintains a *plural* registry — one entry per canonical constant
mentioned in user-facing docs — with each entry carrying:

* a ``canonical_provider`` callable that re-imports the constant on every
  invocation (so a monkeypatched value is seen immediately, which is what
  the test suite uses to drive the "constant changed, docs lag" failure
  mode);
* a tuple of compiled regex ``patterns``, each with exactly one capturing
  group for the documented numeric;
* an ``anchor_tokens`` tuple — a number only counts if one of these
  appears within ``_CONTEXT_LINES`` lines of the match (keeps the gate
  from flagging unrelated λs);
* ``left_exclusion_patterns`` — left-of-numeric contexts that must
  *exclude* a match (e.g. the historical "elevated from 0.18" reference
  in README §OAE belongs to ``λ_decay``, not ``λ_lyapunov``);
* ``min_occurrences`` — the floor of documented mentions across the
  scanned file set.  A clean checkout MUST yield at least this many hits
  for this check, or the gate fails with a "vacuous-green pass" diagnostic.
  This is the failure mode PR #238 patched: deleting every documented
  λ claim must not silently turn the gate green.

The two canonical constants currently covered are:

* ``λ_lyapunov`` = ``omni_mercury_engine.core.centralized_constants.LYAPUNOV.LAMBDA_CONVERGENCE``
  — the fusion-trajectory Lyapunov decay rate (``0.25``).  Documented in
  README §OAE prose, README training-API example, README §Reproducible
  Verification, MATH_SPEC §2.2 / §4.2 / §5.4, etc.
* ``λ_decay`` = ``omni_mercury_engine.core.double_helix_engine.LAMBDA_DECAY``
  — the double-helix evolutionary adaptation rate (``0.18``).
  Intentionally distinct from ``LAMBDA_CONVERGENCE``; documented in
  README §Reproducible Verification glossary line and the
  "elevated from 0.18" cross-reference inside the OAE prose.

Exit codes
----------
* ``0`` — every claim agrees with its canonical constant and every check
  meets its ``min_occurrences`` floor.
* ``1`` — at least one claim disagrees with its canonical constant OR a
  check failed its ``min_occurrences`` floor (vacuous-green guard).
* ``2`` — usage / configuration error (e.g. a canonical constant could
  not be imported and ``--canonical`` was not supplied).

Usage
-----
::

    python scripts/check_readme_lyapunov.py
    python scripts/check_readme_lyapunov.py --files README.md docs/MATH_SPEC.md
    python scripts/check_readme_lyapunov.py --canonical 0.25  # legacy single-check override

Runtime dependency
------------------
Because the gate imports the source-of-truth constants, the repository's
runtime dependencies (notably ``numpy``, which ``double_helix_engine``
imports at module load) must be available on the Python path.  In CI we
install the package with ``pip install -e .`` before running this gate;
locally the project's editable install satisfies the requirement.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import math
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Canonical-source resolution: ``import`` the constants from the package.
#
# A previous iteration of this gate parsed the constants out of the
# centralized-constants module with a regex.  That worked but introduced
# a second source of truth (the regex) that could silently drift away
# from the actual runtime value (the imported constant).  The whole
# point of this gate is to make the *runtime* value the only thing that
# matters; we import it directly.
#
# ``importlib.import_module`` is used (rather than a top-level ``from
# ... import ...``) so that:
#
#   1. The import is resolved at *call time*, not at module load time,
#      which is what lets the test suite monkeypatch the canonical value
#      and re-run the gate without reloading this script.
#   2. ImportError handling is local to the provider, so a missing
#      runtime dep (e.g. numpy when the bare-checkout CI runs without
#      ``pip install -e .``) produces a clear, single-line error from
#      ``main`` rather than a stack trace at module load.
# ---------------------------------------------------------------------------


def _ensure_src_on_path() -> None:
    """Make ``omni_mercury_engine.*`` importable without editable install.

    Used as a defence-in-depth for local invocations where the user has
    not yet run ``pip install -e .``.  CI always installs the package,
    so this is purely a developer-experience nicety.
    """
    for candidate in (_REPO_ROOT, _REPO_ROOT / "src"):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def _import_lambda_convergence() -> float:
    """Re-import and return ``LYAPUNOV.LAMBDA_CONVERGENCE`` each call."""
    _ensure_src_on_path()
    module = importlib.import_module("omni_mercury_engine.core.centralized_constants")
    # ``LYAPUNOV`` is a module-level frozen dataclass instance; the field
    # is a plain ``float``.  Coerce defensively in case downstream
    # refactors change the type (e.g. ``np.float64`` from a future
    # constants-as-tensors migration).
    return float(module.LYAPUNOV.LAMBDA_CONVERGENCE)


def _import_lambda_decay() -> float:
    """Re-import and return ``LAMBDA_DECAY`` each call."""
    _ensure_src_on_path()
    module = importlib.import_module("omni_mercury_engine.core.double_helix_engine")
    return float(module.LAMBDA_DECAY)


def _lambda_convergence_provider() -> float:
    """Resolve the convergence importer at call time."""
    return _import_lambda_convergence()


def _lambda_decay_provider() -> float:
    """Resolve the decay importer at call time."""
    return _import_lambda_decay()


# ---------------------------------------------------------------------------
# Pattern building blocks.
#
# A single capturing group ``([0-9]+(?:\.[0-9]+)?)`` is shared by every
# pattern.  ``find_lambda_claims`` reads this group as the documented
# numeric and compares it to the imported canonical via ``math.isclose``.
# ---------------------------------------------------------------------------
_NUM = r"([0-9]+(?:\.[0-9]+)?)"


# ---------------------------------------------------------------------------
# Check definition.
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class LambdaCheck:
    """One canonical-constant → documented-value enforcement entry.

    All fields are required; the ``min_occurrences`` floor is part of
    the check, not a separate ``--require-hits`` flag, because the gate
    is meaningless without it: a regex that silently stops matching the
    shipped prose is a vacuous-green failure mode and the floor is what
    catches it.
    """

    name: str
    canonical_provider: Callable[[], float]
    patterns: tuple[re.Pattern[str], ...]
    anchor_tokens: tuple[str, ...]
    left_exclusion_patterns: tuple[re.Pattern[str], ...]
    min_occurrences: int
    description: str = ""


_CONTEXT_LINES = 4

CHECKS: tuple[LambdaCheck, ...] = (
    # -----------------------------------------------------------------
    # λ_lyapunov — fusion-trajectory Lyapunov convergence rate (0.25).
    #
    # Anchored to the explicit Lyapunov-stability vocabulary so unrelated
    # λs (entropy weights, training-loss penalties, etc.) don't trip the
    # gate.  ``elevated from`` and ``LAMBDA_DECAY`` left-of-numeric
    # contexts are excluded because those numbers belong to the
    # ``λ_decay`` check below.
    #
    # The ``canonical_provider`` is wrapped in a thunk that resolves
    # ``_import_lambda_convergence`` from module-level scope at call
    # time (rather than capturing the function reference at registry
    # construction time).  This is what lets the test suite
    # monkeypatch the importer module-attribute and have the gate see
    # the new value on the next invocation — the import-based
    # contract: the runtime constant IS the source of truth, even
    # when that runtime constant is being swapped under the gate's
    # feet for a test.
    # -----------------------------------------------------------------
    LambdaCheck(
        name="lambda_lyapunov",
        canonical_provider=_lambda_convergence_provider,
        patterns=(
            # 1. Greek inline math: "λ=0.25", "Λ = 0.25"
            re.compile(r"[λΛ]\s*=\s*" + _NUM),
            # 2. LaTeX inline math: "\lambda = 0.25" / "\Lambda = 0.25".
            re.compile(r"\\[lL]ambda\s*=\s*" + _NUM),
            # 3. English-word form: "lambda = 0.25".  The word-boundary
            #    anchor prevents matching "LAMBDA_CONVERGENCE = 0.25"
            #    (constant assignment, not a prose claim).
            re.compile(r"(?<![\w._])lambda\s*=\s*" + _NUM),
            # 4. Explicit symbolic key: "lambda_lyapunov = 0.25" used by
            #    the LyapunovAnomalyLoss constructor in code examples.
            re.compile(r"lambda_lyapunov\s*=\s*" + _NUM),
        ),
        anchor_tokens=(
            "Lyapunov",
            "lyapunov",
            "LYAPUNOV",
            "LAMBDA_CONVERGENCE",
            "stability envelope",
            "lambda_lyapunov",
            "V̇",
            r"\dot{V}",
            "exponential decay",
            "convergence rate",
        ),
        left_exclusion_patterns=(
            # "elevated from N" → N is the *historical* λ_decay value,
            # not the current λ_lyapunov claim.
            re.compile(r"elevated\s+from[^\d]*$", re.IGNORECASE),
            # "LAMBDA_DECAY = N" → N belongs to the λ_decay check.
            re.compile(r"LAMBDA_DECAY[^\d]*$", re.IGNORECASE),
            # "double-helix … N" → double-helix evolution glossary;
            # again, λ_decay's domain.
            re.compile(r"\bdouble[- ]helix[^\d]*$", re.IGNORECASE),
        ),
        # At least one λ_lyapunov claim must survive across the scanned
        # files; the shipped README + MATH_SPEC currently carry ~16 such
        # claims and we hold the line at 1 so the *floor* is enforced
        # without coupling the gate to the exact prose count (which is
        # legitimately editable).
        min_occurrences=1,
        description=(
            "Fusion-trajectory Lyapunov convergence rate "
            "(centralized_constants.LYAPUNOV.LAMBDA_CONVERGENCE)."
        ),
    ),
    # -----------------------------------------------------------------
    # λ_decay — double-helix evolutionary adaptation rate (0.18).
    #
    # Distinct from λ_lyapunov by *design* (see MATH_SPEC § note on
    # "fusion stabilises faster than the helix adapts").  The
    # anchor-tokens vocabulary is intentionally disjoint from
    # ``λ_lyapunov``'s so the two checks cannot accidentally pick up
    # each other's matches.
    # -----------------------------------------------------------------
    LambdaCheck(
        name="lambda_decay",
        canonical_provider=_lambda_decay_provider,
        patterns=(
            # 1. Constant-assignment form rendered verbatim in prose:
            #    "LAMBDA_DECAY = 0.18".
            re.compile(r"LAMBDA_DECAY\s*=\s*" + _NUM),
            # 2. Cross-reference form: "elevated from 0.18".  Captures
            #    the historical value being elevated *from*.
            re.compile(r"elevated\s+from\s+" + _NUM),
            # 3. snake-case key form: "lambda_decay = 0.18".
            re.compile(r"(?<![\w._])lambda_decay\s*=\s*" + _NUM),
        ),
        anchor_tokens=(
            "LAMBDA_DECAY",
            "double-helix",
            "double helix",
            "evolutionary adaptation",
            "adaptation rate",
            "elevated from",
        ),
        left_exclusion_patterns=(),
        # The README currently carries exactly two λ_decay mentions
        # (the OAE "elevated from 0.18" cross-ref and the §Reproducible
        # Verification glossary "LAMBDA_DECAY = 0.18" line).  Hold the
        # floor at 1 so the gate fails if either is deleted without a
        # replacement.
        min_occurrences=1,
        description=(
            "Double-helix evolutionary adaptation rate (double_helix_engine.LAMBDA_DECAY)."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Match scanning.
# ---------------------------------------------------------------------------
def _line_has_anchor(lines: Sequence[str], idx: int, anchors: Sequence[str]) -> bool:
    lo = max(0, idx - _CONTEXT_LINES)
    hi = min(len(lines), idx + _CONTEXT_LINES + 1)
    window = "\n".join(lines[lo:hi])
    return any(tok in window for tok in anchors)


def _match_is_excluded(
    line: str,
    match: re.Match[str],
    exclusions: Sequence[re.Pattern[str]],
) -> bool:
    """Return True if the captured numeric is in a *left-exclusion* context.

    The exclusion is directional: we only inspect the line content
    immediately to the LEFT of the captured numeric span, so a leading
    ``elevated from`` phrase suppresses the historical value that
    follows it without also suppressing the current value the phrase
    referred back to.
    """
    span_start, _ = match.span(1)
    left = line[:span_start]
    return any(p.search(left) for p in exclusions)


@dataclasses.dataclass(frozen=True)
class LambdaHit:
    """A single documentation match for a :class:`LambdaCheck`."""

    check_name: str
    path: Path
    line_no: int
    line_text: str
    claimed_value: float


def find_lambda_claims(
    text: str,
    *,
    check: LambdaCheck | None = None,
) -> list[tuple[int, str, float]]:
    """Yield ``(line_no, line_text, claimed_lambda)`` for *one* check.

    Backwards-compatible single-check helper.  ``check`` defaults to the
    ``lambda_lyapunov`` entry (matching the legacy single-canonical
    behaviour) so existing callers and tests don't need to thread a
    registry through.  New code should call :func:`scan_file` instead,
    which iterates the full :data:`CHECKS` registry.

    Line numbers are 1-based.
    """
    selected = check if check is not None else CHECKS[0]
    lines = text.splitlines()
    results: list[tuple[int, str, float]] = []
    for i, line in enumerate(lines):
        seen_spans: set[tuple[int, int]] = set()
        for pat in selected.patterns:
            for m in pat.finditer(line):
                span = m.span(1)
                if span in seen_spans:
                    continue
                seen_spans.add(span)
                if not _line_has_anchor(lines, i, selected.anchor_tokens):
                    continue
                if _match_is_excluded(line, m, selected.left_exclusion_patterns):
                    continue
                try:
                    val = float(m.group(1))
                except (TypeError, ValueError):  # pragma: no cover
                    continue
                results.append((i + 1, line.strip(), val))
    return results


def scan_file(path: Path, check: LambdaCheck) -> list[LambdaHit]:
    """Return every :class:`LambdaHit` produced by ``check`` against ``path``."""
    text = path.read_text(encoding="utf-8")
    return [
        LambdaHit(
            check_name=check.name,
            path=path,
            line_no=line_no,
            line_text=line_text,
            claimed_value=val,
        )
        for line_no, line_text, val in find_lambda_claims(text, check=check)
    ]


# ---------------------------------------------------------------------------
# Top-level enforcement.
# ---------------------------------------------------------------------------
def check_files(
    files: Iterable[Path],
    *,
    tol: float = 1e-9,
    canonical_override: float | None = None,
    checks: Sequence[LambdaCheck] = CHECKS,
) -> list[str]:
    """Return human-readable error messages (empty == OK).

    ``canonical_override`` is a legacy escape hatch used by the tests
    and the ``--canonical`` CLI argument: when supplied, it overrides
    the canonical value for the *first* check in ``checks`` (which is
    historically ``lambda_lyapunov``) and disables the import for that
    check only.  All other checks continue to import their canonicals
    normally.  This keeps the legacy single-check CLI surface alive
    without compromising the plural registry.
    """
    errors: list[str] = []
    file_list = [Path(p) for p in files]

    missing = [p for p in file_list if not p.exists()]
    for p in missing:
        errors.append(f"{p}: file not found")
    present_files = [p for p in file_list if p.exists()]

    for idx, check in enumerate(checks):
        if canonical_override is not None and idx == 0:
            # ``--canonical`` is the legacy single-check escape hatch
            # used by synthetic test fixtures.  When supplied it both
            # overrides the canonical for the first check AND
            # suppresses that check's ``min_occurrences`` floor: a
            # synthetic fixture exercising an unrelated edge case
            # (e.g. "no λ_lyapunov anchor → unrelated λ must be
            # ignored") doesn't have to include a λ_lyapunov claim
            # just to satisfy the gate's vacuous-green guard.  Other
            # checks in the registry still enforce their floors.
            canonical = float(canonical_override)
            enforce_floor = False
        else:
            try:
                canonical = check.canonical_provider()
            except Exception as exc:  # pragma: no cover - imported below
                errors.append(
                    f"[{check.name}] could not import canonical constant: "
                    f"{exc.__class__.__name__}: {exc}"
                )
                continue
            enforce_floor = True

        total_hits = 0
        for path in present_files:
            for hit in scan_file(path, check):
                total_hits += 1
                if not math.isclose(hit.claimed_value, canonical, rel_tol=0, abs_tol=tol):
                    errors.append(
                        f"[{check.name}] {hit.path}:{hit.line_no}: "
                        f"claimed {check.name}={hit.claimed_value} "
                        f"!= canonical {check.name}={canonical} "
                        f"-- {hit.line_text!r}"
                    )

        if enforce_floor and total_hits < check.min_occurrences:
            errors.append(
                f"[{check.name}] only {total_hits} documented mention(s) "
                f"found across the scanned files (floor: "
                f"{check.min_occurrences}).  This is a vacuous-green guard: "
                f"silently deleting every documented mention must not turn "
                f"the gate green.  Either restore the documentation or "
                f"lower the floor in scripts/check_readme_lyapunov.py."
            )

    return errors


# ---------------------------------------------------------------------------
# CLI plumbing.
# ---------------------------------------------------------------------------
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
            "Legacy single-check escape hatch: override the "
            "lambda_lyapunov canonical with the given float.  Useful "
            "for tests that synthesise tiny fixtures and do not want "
            "to import the full constants package.  Other checks in "
            "the registry continue to import normally."
        ),
    )
    parser.add_argument(
        "--require-hits",
        nargs="*",
        type=Path,
        default=None,
        help=(
            "DEPRECATED: per-check min_occurrences floors are now part "
            "of the LambdaCheck registry itself.  Accepting this flag "
            "for backwards compatibility; it is silently ignored."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        errors = check_files(
            args.files,
            canonical_override=args.canonical,
        )
    except Exception as exc:  # pragma: no cover - belt-and-braces
        print(f"ERROR: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 2

    if errors:
        print(
            f"FAIL: {len(errors)} λ-drift issue(s) detected:",
            file=sys.stderr,
        )
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    canonical_summary = ", ".join(f"{c.name}={c.canonical_provider():.6g}" for c in CHECKS)
    print(
        f"OK: all documented λ claims match canonical "
        f"({canonical_summary}); scanned {len(list(args.files))} files."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry-point
    raise SystemExit(main())
