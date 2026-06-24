#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fail the build if golden-ratio (φ) numerology re-enters a scoring path.

Mercury's integrity contract is **anti-theater**: a reported anomaly / risk /
confidence score may never be multiplied by an *unlearned constant*.  A family
of "truth-decipher" detectors historically violated this by scaling model
confidences through ``create_omni_*_scalars`` dictionaries whose every entry
was ``<coef> * φ`` (φ = 1.618…) and by thresholding those inflated scores
against ``k * golden_ratio``.  Workstream A1 excised that numerology; this gate
**locks the excision in** so it cannot silently return.

What this gate forbids (token-level, so comments / docstrings / string
literals that merely *mention* the pattern are never flagged)
-------------------------------------------------------------------
1. ``create_omni_<name>_scalars`` — the φ-scalar generator factory.  These
   functions existed only to manufacture φ-inflated multipliers; there is no
   legitimate use, so any occurrence anywhere under the engine fails.
2. ``omni_<name>_scalars[ ... ]`` — indexing such a scalar dict (the site
   where the φ multiplier was actually applied to a score).
3. ``* self.golden_ratio`` / ``self.golden_ratio *`` — φ used as a runtime
   multiplier via the instance attribute.  This idiom was *always* the
   numerology pattern, so it is forbidden on every path.
4. ``* self.phi`` / ``self.phi *`` — φ used as a runtime multiplier via a
   ``phi`` attribute.  Forbidden **except** on the explicitly allow-listed
   legitimate-mathematics paths below.
5. ``<score> * phi`` / ``phi * <score>`` and the ``golden_ratio`` equivalent
   where ``<score>`` is a bare local whose name reads as a reported quantity
   (``*score*``, ``*risk*``, ``*anomaly*``, ``*confidence*``, ``*threat*``,
   ``*threshold*``, ``*severity*``, ``*prob*``).  This catches a regression
   that reintroduces the multiply with a local φ instead of the attribute.

What this gate deliberately does NOT forbid
-------------------------------------------
* **Architectural** φ — sizing network layers, e.g. ``int(input_dim * phi)``,
  ``nn.Linear(d, int(d * phi))``.  φ as a width heuristic is not a *score*
  multiplication; ``tests/test_abms_disciplines.py::test_golden_ratio_architecture``
  validates this on purpose.  Rule 5 only fires when the multiplicand reads as
  a reported score, so architectural sizing never trips it.
* **Legitimate φ mathematics** on the allow-listed paths
  (:data:`PHI_MATH_ALLOWLIST`): ``harmonics/`` and ``core/three_r/`` (the
  prompt-sanctioned φ homes) and ``core/fusion.py`` (where ``self.phi`` is a
  *GA-optimized* term coefficient, ``self.config.get("phi", ga_optimized[10])``,
  in a named fusion ODE — a learned constant, not numerology).

Vacuous-green guard
-------------------
Like ``scripts/check_readme_lyapunov.py``, this gate refuses to pass vacuously:
if it scans fewer than :data:`MIN_FILES_SCANNED` source files (a broken glob,
a moved package) it fails with exit code 2 rather than reporting a clean run.

Exit codes
----------
* ``0`` — no numerology found on any scoring path.
* ``1`` — at least one forbidden idiom found (printed with ``file:line``).
* ``2`` — usage / configuration error (package tree not found, too few files
  scanned, or a source file failed to tokenize).
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import re
import sys
import token as token_mod
import tokenize
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ROOT = _REPO_ROOT / "src" / "omni_mercury_engine"

# A clean checkout must contain at least this many engine source files; fewer
# means the scan target moved and the "green" result would be meaningless.
MIN_FILES_SCANNED = 200

# Path *substrings* (POSIX-style) where a ``self.phi`` multiplier is legitimate
# mathematics rather than scoring numerology.  Kept deliberately small and
# documented; each entry is a load-bearing exemption, not a convenience.
PHI_MATH_ALLOWLIST: tuple[str, ...] = (
    "harmonics/",  # φ is the subject matter (harmonic/φ-resonance analysis).
    "core/three_r/",  # prompt-sanctioned φ home (three-R mechanism).
    "core/fusion.py",  # self.phi is a GA-optimized ODE term coefficient.
)

# Bare locals whose name marks them as a reported quantity.  A ``* phi`` on one
# of these is the numerology regression; a ``* phi`` on a layer dimension is not.
_SCORE_NAME_RE = re.compile(r"(?i)(score|risk|anomaly|confidence|threat|threshold|severity|prob)")


@dataclasses.dataclass(frozen=True)
class Violation:
    """One forbidden-idiom hit."""

    rule: str
    path: Path
    line_no: int
    snippet: str


def _allowlisted(path: Path, root: Path) -> bool:
    rel = path.resolve().as_posix()
    return any(frag in rel for frag in PHI_MATH_ALLOWLIST)


def _code_tokens(source: str) -> list[tokenize.TokenInfo]:
    """Return only *code* tokens (NAME / OP / NUMBER).

    Comments and string/docstring contents are dropped, so prose that merely
    references a forbidden idiom (including this module's own fix-comments)
    can never produce a false positive.
    """
    wanted = {token_mod.NAME, token_mod.OP, token_mod.NUMBER}
    toks: list[tokenize.TokenInfo] = []
    reader = io.StringIO(source).readline
    for tok in tokenize.generate_tokens(reader):
        if tok.type in wanted:
            toks.append(tok)
    return toks


_GENERATOR_RE = re.compile(r"^create_omni_\w+_scalars$")
_SCALAR_NAME_RE = re.compile(r"^omni_\w+_scalars$")


def scan_source(source: str, path: Path, *, phi_allowlisted: bool) -> list[Violation]:
    """Scan one file's source text for forbidden scoring-path numerology."""
    toks = _code_tokens(source)
    violations: list[Violation] = []

    def add(rule: str, line_no: int, snippet: str) -> None:
        violations.append(Violation(rule, path, line_no, snippet.strip()))

    for i, tok in enumerate(toks):
        # Rule 1: create_omni_*_scalars factory (NAME token).
        if tok.type == token_mod.NAME and _GENERATOR_RE.match(tok.string):
            add("omni_scalar_generator", tok.start[0], tok.line)
            continue

        # Rule 2: omni_*_scalars[ ... ] indexing (NAME followed by '[').
        if (
            tok.type == token_mod.NAME
            and _SCALAR_NAME_RE.match(tok.string)
            and i + 1 < len(toks)
            and toks[i + 1].type == token_mod.OP
            and toks[i + 1].string == "["
        ):
            add("omni_scalar_index", tok.start[0], tok.line)
            continue

        # Rules 3-5 fire on a multiplication operator; inspect both operands.
        if tok.type == token_mod.OP and tok.string == "*":
            left = _operand_before(toks, i)
            right = _operand_after(toks, i)
            hit = _classify_multiplication(left, right, phi_allowlisted)
            if hit is not None:
                add(hit, tok.start[0], tok.line)

    return violations


@dataclasses.dataclass(frozen=True)
class _Operand:
    """A multiplication operand reduced to what the gate needs to know."""

    is_self_golden_ratio: bool  # ``self.golden_ratio``
    is_self_phi: bool  # ``self.phi``
    bare_name: str | None  # the plain NAME adjacent to ``*`` (e.g. ``phi``,
    # ``risk_score``), else None for non-NAME / attribute operands.

    @property
    def is_bare_phi(self) -> bool:
        return self.bare_name in {"phi", "golden_ratio"}

    @property
    def is_score_name(self) -> bool:
        return (
            self.bare_name is not None
            and not self.is_bare_phi
            and _SCORE_NAME_RE.search(self.bare_name) is not None
        )


def _operand_before(toks: Sequence[tokenize.TokenInfo], star_idx: int) -> _Operand:
    """Reduce the operand immediately to the LEFT of ``toks[star_idx]``."""
    j = star_idx - 1
    if j < 0 or toks[j].type != token_mod.NAME:
        return _Operand(False, False, None)
    cur = toks[j]
    # ``self.phi`` / ``self.golden_ratio`` — attribute, not a bare local.
    if (
        cur.string in {"phi", "golden_ratio"}
        and j >= 2
        and toks[j - 1].type == token_mod.OP
        and toks[j - 1].string == "."
        and toks[j - 2].type == token_mod.NAME
        and toks[j - 2].string == "self"
    ):
        return _Operand(cur.string == "golden_ratio", cur.string == "phi", None)
    return _Operand(False, False, cur.string)


def _operand_after(toks: Sequence[tokenize.TokenInfo], star_idx: int) -> _Operand:
    """Reduce the operand immediately to the RIGHT of ``toks[star_idx]``."""
    j = star_idx + 1
    if j >= len(toks) or toks[j].type != token_mod.NAME:
        return _Operand(False, False, None)
    cur = toks[j]
    # ``self.phi`` / ``self.golden_ratio`` — attribute, not a bare local.
    if (
        cur.string == "self"
        and j + 2 < len(toks)
        and toks[j + 1].type == token_mod.OP
        and toks[j + 1].string == "."
        and toks[j + 2].type == token_mod.NAME
        and toks[j + 2].string in {"phi", "golden_ratio"}
    ):
        attr = toks[j + 2].string
        return _Operand(attr == "golden_ratio", attr == "phi", None)
    # A plain NAME directly followed by ``.`` is an attribute access whose base
    # name we still record (``obj.risk_score`` reads as a score), but ``self``
    # is the receiver of the attribute case handled above, never an operand.
    return _Operand(False, False, cur.string)


def _classify_multiplication(left: _Operand, right: _Operand, phi_allowlisted: bool) -> str | None:
    """Return the violated rule name for ``left * right``, or None if clean."""
    # Rule 3: ``* self.golden_ratio`` — forbidden everywhere.
    if left.is_self_golden_ratio or right.is_self_golden_ratio:
        return "golden_ratio_multiplier"
    # Rule 4: ``* self.phi`` — forbidden unless on a legitimate-math path.
    if (left.is_self_phi or right.is_self_phi) and not phi_allowlisted:
        return "phi_attr_multiplier"
    # Rule 5: ``<score> * phi`` / ``phi * <score>`` — a bare φ local multiplying
    # a bare local whose name reads as a reported score.  Architectural sizing
    # (``int(dim * phi)``) never matches because ``dim`` is not a score name.
    if (left.is_bare_phi and right.is_score_name) or (right.is_bare_phi and left.is_score_name):
        return "phi_local_score_multiplier"
    return None


def iter_source_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def check_tree(root: Path, *, min_files: int = MIN_FILES_SCANNED) -> tuple[list[Violation], int]:
    """Scan ``root`` for forbidden idioms.

    Returns ``(violations, files_scanned)``.  ``files_scanned`` lets the caller
    enforce the vacuous-green guard.
    """
    violations: list[Violation] = []
    files = iter_source_files(root)
    for path in files:
        source = path.read_text(encoding="utf-8")
        try:
            file_violations = scan_source(source, path, phi_allowlisted=_allowlisted(path, root))
        except tokenize.TokenError as exc:  # pragma: no cover - malformed source
            raise RuntimeError(f"{path}: failed to tokenize: {exc}") from exc
        violations.extend(file_violations)
    return violations, len(files)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=_DEFAULT_ROOT,
        help="Engine package root to scan (default: src/omni_mercury_engine).",
    )
    parser.add_argument(
        "--min-files",
        type=int,
        default=MIN_FILES_SCANNED,
        help="Vacuous-green floor: fail if fewer than this many files scanned.",
    )
    args = parser.parse_args(argv)

    if not args.root.exists():
        print(f"ERROR: scan root does not exist: {args.root}", file=sys.stderr)
        return 2

    try:
        violations, files_scanned = check_tree(args.root, min_files=args.min_files)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if files_scanned < args.min_files:
        print(
            f"ERROR: only {files_scanned} source file(s) scanned under "
            f"{args.root} (floor: {args.min_files}).  This is a vacuous-green "
            f"guard: a moved or empty scan target must not pass silently.",
            file=sys.stderr,
        )
        return 2

    if violations:
        print(
            f"FAIL: {len(violations)} golden-ratio numerology hit(s) on scoring "
            f"path(s) across {files_scanned} scanned files:",
            file=sys.stderr,
        )
        for v in violations:
            rel = v.path.relative_to(_REPO_ROOT) if _REPO_ROOT in v.path.parents else v.path
            print(f"  [{v.rule}] {rel}:{v.line_no}: {v.snippet}", file=sys.stderr)
        print(
            "\nA reported score must never be multiplied by an unlearned "
            "constant.  Use the raw calibrated confidence and derive thresholds "
            "from the calibration path, not φ.  See WORKSTREAM A1 / "
            "scripts/check_no_numerology_in_scoring.py.",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: no golden-ratio numerology on any scoring path "
        f"({files_scanned} files scanned under {args.root})."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry-point
    raise SystemExit(main())
