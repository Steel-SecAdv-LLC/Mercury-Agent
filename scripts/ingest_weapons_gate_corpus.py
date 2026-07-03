#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rolling ingestion + versioning for the weapons-gate evaluation corpus.

The seed corpus is generated from templates in ``benchmarks/weapons_gate_corpus.py``
and dumped to ``benchmarks/weapons_gate_corpus.jsonl`` (362 cases). That JSONL is
the **authoritative, versioned** corpus this tool maintains: new labeled examples
are ingested into it over time (a *rolling* corpus), each version pinned by a
content hash in ``benchmarks/weapons_gate_corpus_manifest.json``.

Every operation is deterministic and dependency-free (stdlib only), so it runs in
any lane without building the PQC backend:

* ``--add FILE``     merge new labeled rows from an external JSONL (validated,
                     deduped, kept disjoint from the held-out adversarial set).
* ``--validate FILE``dry-run: validate an external JSONL without writing.
* ``--manifest``     (re)write the manifest from the current corpus.
* ``--check``        verify the corpus matches its manifest hash and the class
                     balance invariants (CI integrity gate; exit 1 on drift).
* ``--rebuild-seed`` regenerate the JSONL from the templates (the initial
                     362-case migration; discards ingested rows -- use with care).

Row schema (one JSON object per line, ``sort_keys`` canonical form):
``{"text": str, "label": "offensive"|"benign", "expected": "block"|"allow",
   "split": "train"|"val"|"test", "tags": [str, ...]}``. ``label`` and ``expected``
must agree (offensive->block, benign->allow); ``split`` is auto-assigned from the
stable text hash when omitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
_BENCHMARKS = _REPO / "benchmarks"
sys.path.insert(0, str(_BENCHMARKS))

from weapons_gate_corpus import _split_for, rows_as_dicts

CORPUS_PATH = _BENCHMARKS / "weapons_gate_corpus.jsonl"
ADVERSARIAL_PATH = _BENCHMARKS / "weapons_gate_adversarial.jsonl"
MANIFEST_PATH = _BENCHMARKS / "weapons_gate_corpus_manifest.json"

_VALID_LABELS = {"offensive", "benign"}
_VALID_EXPECTED = {"block", "allow"}
_VALID_SPLITS = {"train", "val", "test"}
_LABEL_TO_EXPECTED = {"offensive": "block", "benign": "allow"}
#: Class-balance floors enforced by ``--check`` (mirror tests/ethical/test_weapons_gate_eval).
_MIN_PER_CLASS = 120


class CorpusValidationError(ValueError):
    """An ingested row (or the corpus) failed schema/consistency validation."""


# --------------------------------------------------------------------------- #
# IO + canonical serialization (byte-identical to the template dump).
# --------------------------------------------------------------------------- #
def _canonical_line(row: dict[str, Any]) -> str:
    """One corpus row as a canonical JSON line (sorted keys, unicode preserved)."""
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def dump_rows(rows: list[dict[str, Any]]) -> str:
    """Serialize rows to the canonical JSONL text: sorted by text, trailing newline."""
    ordered = sorted(rows, key=lambda r: r["text"])
    return "\n".join(_canonical_line(r) for r in ordered) + "\n"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts (empty list when absent)."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CorpusValidationError(f"{path.name}:{i}: invalid JSON ({exc})") from exc
        if not isinstance(obj, dict):
            raise CorpusValidationError(f"{path.name}:{i}: expected a JSON object")
        rows.append(obj)
    return rows


def corpus_sha256(rows: list[dict[str, Any]]) -> str:
    """Content hash of the corpus: sha256 over the canonical serialization."""
    return hashlib.sha256(dump_rows(rows).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Validation.
# --------------------------------------------------------------------------- #
def validate_row(row: dict[str, Any], *, where: str) -> dict[str, Any]:
    """Return a normalized, schema-valid copy of ``row`` or raise CorpusValidationError."""
    text = str(row.get("text", "")).strip()
    if not text:
        raise CorpusValidationError(f"{where}: empty 'text'")
    label = str(row.get("label", "")).strip()
    if label not in _VALID_LABELS:
        raise CorpusValidationError(f"{where}: label {label!r} not in {sorted(_VALID_LABELS)}")
    expected = str(row.get("expected", _LABEL_TO_EXPECTED[label])).strip()
    if expected not in _VALID_EXPECTED:
        raise CorpusValidationError(
            f"{where}: expected {expected!r} not in {sorted(_VALID_EXPECTED)}"
        )
    if expected != _LABEL_TO_EXPECTED[label]:
        raise CorpusValidationError(
            f"{where}: label/expected disagree ({label!r} implies "
            f"{_LABEL_TO_EXPECTED[label]!r}, got {expected!r})"
        )
    split = str(row.get("split", "")).strip() or _split_for(text)
    if split not in _VALID_SPLITS:
        raise CorpusValidationError(f"{where}: split {split!r} not in {sorted(_VALID_SPLITS)}")
    raw_tags = row.get("tags", [])
    if not isinstance(raw_tags, (list, tuple)):
        raise CorpusValidationError(f"{where}: tags must be a list")
    tags = [str(t) for t in raw_tags]
    return {"text": text, "label": label, "expected": expected, "split": split, "tags": tags}


def adversarial_texts() -> set[str]:
    """Stripped texts of the held-out adversarial set (never mixed into training)."""
    return {str(r.get("text", "")).strip() for r in load_jsonl(ADVERSARIAL_PATH)}


# --------------------------------------------------------------------------- #
# Manifest.
# --------------------------------------------------------------------------- #
def build_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the versioning manifest for a corpus (counts + content hash + version)."""
    sha = corpus_sha256(rows)
    per_split: dict[str, int] = {}
    per_label: dict[str, int] = {}
    per_expected: dict[str, int] = {}
    for r in rows:
        per_split[r["split"]] = per_split.get(r["split"], 0) + 1
        per_label[r["label"]] = per_label.get(r["label"], 0) + 1
        per_expected[r["expected"]] = per_expected.get(r["expected"], 0) + 1
    return {
        "corpus": CORPUS_PATH.name,
        "corpus_version": f"{len(rows)}-{sha[:12]}",
        "n_cases": len(rows),
        "sha256": sha,
        "per_split": dict(sorted(per_split.items())),
        "per_label": dict(sorted(per_label.items())),
        "per_expected": dict(sorted(per_expected.items())),
        "schema": ["expected", "label", "split", "tags", "text"],
        "adversarial_corpus": ADVERSARIAL_PATH.name,
        "adversarial_n": len(load_jsonl(ADVERSARIAL_PATH)),
    }


def write_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Write and return the manifest for ``rows``."""
    manifest = build_manifest(rows)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------- #
# Operations.
# --------------------------------------------------------------------------- #
def _current_corpus() -> list[dict[str, Any]]:
    rows = load_jsonl(CORPUS_PATH)
    return [validate_row(r, where=f"{CORPUS_PATH.name}[{i}]") for i, r in enumerate(rows)]


def ingest(add_path: Path) -> tuple[list[dict[str, Any]], int, int]:
    """Merge validated rows from ``add_path`` into the corpus. Returns (corpus, added, skipped).

    Rows are validated, deduped against the existing corpus by stripped text, and
    refused if they collide with the held-out adversarial set (so an ingest can
    never contaminate the never-trained adversarial eval).
    """
    corpus = _current_corpus()
    by_text = {r["text"]: r for r in corpus}
    adversarial = adversarial_texts()
    incoming = load_jsonl(add_path)
    added = 0
    skipped = 0
    for i, raw in enumerate(incoming):
        row = validate_row(raw, where=f"{add_path.name}[{i}]")
        if row["text"] in adversarial:
            raise CorpusValidationError(
                f"{add_path.name}[{i}]: text collides with the held-out adversarial "
                f"corpus; refusing (would contaminate the never-trained eval set): "
                f"{row['text'][:60]!r}"
            )
        if row["text"] in by_text:
            skipped += 1
            continue
        by_text[row["text"]] = row
        added += 1
    merged = sorted(by_text.values(), key=lambda r: r["text"])
    return merged, added, skipped


def check() -> list[str]:
    """Return a list of integrity violations (empty == corpus is consistent)."""
    problems: list[str] = []
    try:
        corpus = _current_corpus()
    except CorpusValidationError as exc:
        return [str(exc)]
    if not MANIFEST_PATH.is_file():
        problems.append(f"missing manifest {MANIFEST_PATH.name}; run --manifest")
        return problems
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = build_manifest(corpus)
    if manifest.get("sha256") != expected["sha256"]:
        problems.append(
            f"corpus sha256 {expected['sha256'][:12]} != manifest "
            f"{str(manifest.get('sha256'))[:12]}; corpus changed without re-manifest"
        )
    if manifest.get("n_cases") != expected["n_cases"]:
        problems.append(
            f"n_cases {expected['n_cases']} != manifest {manifest.get('n_cases')}"
        )
    per_label = expected["per_label"]
    for cls in ("offensive", "benign"):
        if per_label.get(cls, 0) < _MIN_PER_CLASS:
            problems.append(
                f"class balance: {cls}={per_label.get(cls, 0)} < floor {_MIN_PER_CLASS}"
            )
    # Disjointness from the held-out adversarial set.
    overlap = {r["text"] for r in corpus} & adversarial_texts()
    if overlap:
        problems.append(f"{len(overlap)} corpus row(s) overlap the adversarial set")
    return problems


def _write_corpus(rows: list[dict[str, Any]]) -> None:
    CORPUS_PATH.write_text(dump_rows(rows), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--add", metavar="FILE", help="merge new labeled rows from a JSONL file")
    group.add_argument("--validate", metavar="FILE", help="validate a JSONL file (dry run)")
    group.add_argument("--check", action="store_true", help="verify corpus vs manifest + balance")
    group.add_argument("--manifest", action="store_true", help="(re)write the manifest")
    group.add_argument(
        "--rebuild-seed", action="store_true", help="regenerate the JSONL from templates"
    )
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except CorpusValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace) -> int:
    """Run the selected operation (raises CorpusValidationError on bad input)."""
    if args.validate:
        rows = load_jsonl(Path(args.validate))
        for i, raw in enumerate(rows):
            validate_row(raw, where=f"{Path(args.validate).name}[{i}]")
        print(f"OK: {len(rows)} row(s) valid in {args.validate}")
        return 0

    if args.check:
        problems = check()
        if problems:
            print("CORPUS CHECK FAILED:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 1
        manifest = build_manifest(_current_corpus())
        print(f"OK: corpus {manifest['corpus_version']} ({manifest['n_cases']} cases) consistent")
        return 0

    if args.rebuild_seed:
        seed = [validate_row(r, where="seed") for r in rows_as_dicts()]
        _write_corpus(seed)
        manifest = write_manifest(seed)
        print(f"rebuilt seed corpus: {manifest['n_cases']} cases -> {manifest['corpus_version']}")
        return 0

    if args.add:
        merged, added, skipped = ingest(Path(args.add))
        _write_corpus(merged)
        manifest = write_manifest(merged)
        print(
            f"ingested {added} new row(s), skipped {skipped} duplicate(s); "
            f"corpus now {manifest['n_cases']} cases -> {manifest['corpus_version']}"
        )
        return 0

    # Default: (re)write the manifest from the current corpus.
    manifest = write_manifest(_current_corpus())
    print(f"manifest written: {manifest['corpus_version']} ({manifest['n_cases']} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
