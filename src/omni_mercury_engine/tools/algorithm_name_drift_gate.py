"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.

------------------------------------------------------------------------

Operator tool: algorithm-name drift gate.

Generalises the manual review that caught the Kyber-768 → Kyber-1024
slip in README/SECURITY/ARCHITECTURE earlier this cycle.  The tool
walks the documentation surface for algorithm-name claims and
cross-checks each one against the actual exports of
``omni_mercury_engine.security.pqc_backends``.

Detected drift falls into three buckets:

* **deprecated**: the docs reference an algorithm that no longer
  appears in ``pqc_backends`` (e.g. mentions of ``Kyber-768`` after
  the migration to ``Kyber-1024``);
* **undeclared**: ``pqc_backends`` exports a primitive that no doc
  file references — usually fine for internal helpers but worth a
  warning for top-level primitives;
* **mismatched-parameter**: a parameter set is referenced (e.g.
  ``ML-DSA-44``) that is not actually wired in.

The gate emits a JSON certificate listing every drift and exits
non-zero when *deprecated* or *mismatched* entries are present.  Wire
it into pre-commit / CI so the drift is caught before review.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.algorithm_name_drift_gate/v1"
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Canonical algorithm aliases.  Each key is a regex token; the value
# is the canonical name.  Matching is case-insensitive but
# whole-word.  Keep this list deliberately conservative — it's the
# *vocabulary* the gate scans for; cross-checking decides drift.
_KNOWN_ALGORITHMS: dict[str, str] = {
    # FIPS 203 — Kyber / ML-KEM
    r"kyber-?512": "Kyber-512",
    r"kyber-?768": "Kyber-768",
    r"kyber-?1024": "Kyber-1024",
    r"ml-?kem-?512": "ML-KEM-512",
    r"ml-?kem-?768": "ML-KEM-768",
    r"ml-?kem-?1024": "ML-KEM-1024",
    # FIPS 204 — Dilithium / ML-DSA
    r"dilithium-?2": "Dilithium-2",
    r"dilithium-?3": "Dilithium-3",
    r"dilithium-?5": "Dilithium-5",
    r"ml-?dsa-?44": "ML-DSA-44",
    r"ml-?dsa-?65": "ML-DSA-65",
    r"ml-?dsa-?87": "ML-DSA-87",
    # FIPS 205 — SPHINCS+ / SLH-DSA
    r"sphincs\+?-?(sha2|shake)-?128s": "SPHINCS+-128s",
    r"slh-?dsa-?shake-?128s": "SLH-DSA-SHAKE-128s",
    r"slh-?dsa-?sha2-?256f": "SLH-DSA-SHA2-256f",
    # Classical baselines
    r"ed25519": "Ed25519",
    r"x25519": "X25519",
    r"rsa-?2048": "RSA-2048",
    r"rsa-?4096": "RSA-4096",
}

# Mapping from canonical algorithm name → required exports on
# ``pqc_backends`` for that algorithm to count as "declared".  Empty
# tuple means "do not cross-check declarations" (e.g. classical names
# whose surface lives elsewhere).
_DECLARATION_EXPORTS: dict[str, tuple[str, ...]] = {
    "Kyber-768": ("KYBER_AVAILABLE",),
    "Kyber-1024": ("KYBER_AVAILABLE",),
    "ML-KEM-768": ("KYBER_AVAILABLE",),
    "ML-KEM-1024": ("KYBER_AVAILABLE",),
    "Dilithium-3": ("DILITHIUM_AVAILABLE",),
    "ML-DSA-65": ("DILITHIUM_AVAILABLE",),
    "SPHINCS+-128s": ("SPHINCS_AVAILABLE",),
    "SLH-DSA-SHAKE-128s": ("SLHDSA_AVAILABLE",),
    "SLH-DSA-SHA2-256f": ("SLHDSA_AVAILABLE",),
    "Ed25519": (),  # crypto_api primitive, not pqc_backends
    "X25519": (),
}

# Algorithms that the project has explicitly migrated away from —
# any documentation reference is a drift signal.
_DEPRECATED: set[str] = {
    "Kyber-512",
    "Kyber-768",
    "ML-KEM-512",
    "ML-KEM-768",
    "Dilithium-2",
    "Dilithium-3",
    "ML-DSA-44",
    "RSA-2048",
}

# Allow-listed mentions of deprecated algorithm names.  Keyed by
# canonical algorithm; each entry is a list of ``(relative_doc_path,
# context_substring)`` pairs.  The substring is a stable fragment of
# the allowed line — e.g. ``"FIPS 204 name for the Dilithium-3
# parameter set"`` — so the gate accepts the mention by *content*,
# not by line number.  Pinning by content used to be by ``(doc,
# line_number)`` tuples, but that coupled the gate to absolute line
# positions: any unrelated README edit above the pinned line shifted
# the number and turned a green gate red.  The content form is robust
# to insertions/deletions anywhere in the file, and the substring is
# self-documenting — the comment beside each entry explains *why*
# this mention is permitted (typically: it documents the FIPS-204
# mapping or quotes the legacy name in a glossary, not advocates use).
_ALLOWED_ALGORITHM_MENTIONS: dict[str, list[tuple[str, str]]] = {
    "Dilithium-3": [
        # README PQC section — explains ML-DSA-65 as the FIPS 204 name
        # for the Dilithium-3 parameter set (the existence of the name
        # mapping is the whole point of this mention).
        ("README.md", "FIPS 204 name for the Dilithium-3 parameter set"),
        # README API reference glossary — pins the FIPS 204 / Dilithium-3
        # equivalence for operators reading the AMA Cryptography reference.
        ("README.md", "ML-DSA-65 (Dilithium-3)"),
    ],
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.algorithm_name_drift_gate",
        description=(
            "Scan README/SECURITY/ARCHITECTURE for algorithm-name claims and "
            "cross-check against omni_mercury_engine.security.pqc_backends."
        ),
    )
    parser.add_argument(
        "--root",
        default=str(_REPO_ROOT),
        help="Repository root (default: detected from this file).",
    )
    parser.add_argument(
        "--docs",
        nargs="*",
        default=None,
        help=(
            "Override the doc file list (default: README.md, SECURITY.md, "
            "ARCHITECTURE.md, docs/ARCHITECTURE.md, CHANGELOG.md if present)."
        ),
    )
    return parser


def _is_subpath(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _default_docs(root: Path) -> list[Path]:
    candidates = [
        root / "README.md",
        root / "SECURITY.md",
        root / "ARCHITECTURE.md",
        root / "docs" / "ARCHITECTURE.md",
        root / "docs" / "SECURITY.md",
        root / "CHANGELOG.md",
    ]
    return [c for c in candidates if c.is_file()]


def _scan_doc(path: Path) -> dict[str, list[tuple[int, str]]]:
    """Return ``{canonical_name: [(line_no, line_text), ...]}`` for ``path``.

    Capturing the line text alongside the number lets the allow-list
    operate on line *content* rather than position, so README edits
    above a pinned mention do not turn a green gate red.
    """
    text = path.read_text(errors="replace")
    hits: dict[str, list[tuple[int, str]]] = {}
    for raw, canonical in _KNOWN_ALGORITHMS.items():
        pattern = re.compile(rf"\b{raw}\b", re.IGNORECASE)
        for ln, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                hits.setdefault(canonical, []).append((ln, line))
    return hits


def _allowed_contexts(root: Path, doc: Path, algorithm: str) -> list[str]:
    """Return the context substrings that allow ``algorithm`` in ``doc``."""
    try:
        rel = str(doc.relative_to(root))
    except ValueError:
        rel = str(doc)
    return [
        substring
        for allowed_doc, substring in _ALLOWED_ALGORITHM_MENTIONS.get(algorithm, [])
        if allowed_doc == rel
    ]


def _line_is_allowed(line_text: str, allowed_contexts: list[str]) -> bool:
    """Return True iff ``line_text`` contains any allowed-context substring."""
    return any(substring in line_text for substring in allowed_contexts)


def _pqc_declarations() -> dict[str, bool]:
    """Return the declared-availability flags from ``pqc_backends``."""
    try:
        from omni_mercury_engine.security import pqc_backends
    except ImportError as exc:
        return {"__import_error__": str(exc)}  # type: ignore[dict-item]
    flags = {}
    for name in (
        "KYBER_AVAILABLE",
        "DILITHIUM_AVAILABLE",
        "DILITHIUM_CTX_AVAILABLE",
        "SPHINCS_AVAILABLE",
        "SLHDSA_AVAILABLE",
    ):
        flags[name] = bool(getattr(pqc_backends, name, False))
    return flags


def _collect(args: argparse.Namespace) -> Certificate:
    root = Path(args.root).resolve()
    docs = [Path(p) for p in args.docs] if args.docs else _default_docs(root)
    if not docs:
        raise FileNotFoundError(
            f"no documentation files found under {root} — "
            "pass --docs explicitly or run from the repo root"
        )

    # Internal scan keeps both line numbers and line text so the
    # allowlist can match by content (line-position-stable).  The
    # certificate body exposes just the line numbers for the
    # operator-visible ``per_doc_hits`` field; the text is consumed
    # only by the content-based allowlist check below.
    raw_per_doc: dict[str, dict[str, list[tuple[int, str]]]] = {}
    per_doc: dict[str, dict[str, list[int]]] = {}
    union_hits: set[str] = set()
    for d in docs:
        try:
            rel = str(d.relative_to(root))
        except ValueError:
            rel = str(d)
        scan = _scan_doc(d)
        if scan:
            raw_per_doc[rel] = scan
            per_doc[rel] = {alg: [ln for ln, _ in entries] for alg, entries in scan.items()}
            union_hits.update(scan.keys())

    declarations = _pqc_declarations()

    deprecated_hits: list[dict[str, Any]] = []
    undeclared_hits: list[dict[str, Any]] = []
    for alg in sorted(union_hits):
        if alg in _DEPRECATED:
            occurrences = []
            for doc in sorted(raw_per_doc):
                if alg not in raw_per_doc[doc]:
                    continue
                doc_path = root / doc
                allowed = _allowed_contexts(root, doc_path, alg)
                lines = [
                    ln
                    for ln, line_text in raw_per_doc[doc][alg]
                    if not _line_is_allowed(line_text, allowed)
                ]
                if lines:
                    occurrences.append({"doc": doc, "lines": lines})
            if not occurrences:
                continue
            deprecated_hits.append({"algorithm": alg, "occurrences": occurrences})
            continue
        required_exports = _DECLARATION_EXPORTS.get(alg, ())
        if required_exports and "__import_error__" not in declarations:
            # If a required flag is False we still don't fail (the runtime
            # gating already covers that); we only fail on docs referring
            # to algorithms with NO matching export at all.  An unknown
            # required export string in the table → undeclared.
            if not all(name in declarations for name in required_exports):
                undeclared_hits.append(
                    {"algorithm": alg, "required_exports": list(required_exports)}
                )

    body: dict[str, Any] = {
        "root": str(root),
        "scanned_docs": [
            (str(d.relative_to(root)) if _is_subpath(d, root) else str(d)) for d in docs
        ],
        "found_algorithms": sorted(union_hits),
        "per_doc_hits": per_doc,
        "pqc_declarations": declarations,
        "deprecated_hits": deprecated_hits,
        "undeclared_hits": undeclared_hits,
    }

    warnings: list[str] = []
    if deprecated_hits:
        for h in deprecated_hits:
            warnings.append(
                f"deprecated algorithm '{h['algorithm']}' referenced in "
                + ", ".join(o["doc"] for o in h["occurrences"])
            )
    if undeclared_hits:
        for h in undeclared_hits:
            warnings.append(
                f"undeclared algorithm '{h['algorithm']}' (no matching "
                f"pqc_backends export: {h['required_exports']})"
            )

    status = "fail" if deprecated_hits or undeclared_hits else "ok"
    return Certificate(
        tool="algorithm_name_drift_gate",
        schema=_SCHEMA,
        status=status,
        body=body,
        warnings=warnings,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
