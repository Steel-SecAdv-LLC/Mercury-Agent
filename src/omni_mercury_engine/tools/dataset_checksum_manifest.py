# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Operator tool: SHA-256 dataset checksum manifest.

Mercury currently flags synthetic-fallback datasets via the
``MERCURY_ALLOW_SYNTHETIC`` env-var and a runtime warning, but the
on-disk integrity of *downloaded* datasets is not anchored — a
silently-corrupted CSV or HDF5 file would pass the loader's structural
checks and contaminate downstream benchmark results.

This tool walks a dataset cache directory, computes the SHA-256 of
every file (chunked to stay constant-memory), and emits a manifest::

    {
      "schema": "mercury.dataset_checksum_manifest/v1",
      "root": "/path/to/cache",
      "entries": [
        {"path": "relative/file.csv", "size": 123456, "sha256": "abc...def"},
        ...
      ]
    }

Two modes:

* **emit** (default) — produce the manifest;
* **verify** — re-walk the same root and compare against a saved
  manifest, listing additions/removals/checksum mismatches.

The verify mode is what operators wire into the loader hot-path:
``DatasetLoader.load(verify_manifest=...)`` can call this tool's
``verify_manifest()`` function (importable separately) and fail-closed
on any discrepancy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.dataset_checksum_manifest/v1"
_CHUNK = 1 << 20  # 1 MiB — large enough to amortise system-call overhead.


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.dataset_checksum_manifest",
        description=(
            "Emit or verify a SHA-256 manifest for a Mercury dataset cache "
            "directory.  Closes the on-disk integrity gap that "
            "MERCURY_ALLOW_SYNTHETIC flagging only partially addresses."
        ),
    )
    parser.add_argument("root", help="Dataset cache directory to walk.")
    parser.add_argument(
        "--verify",
        default=None,
        help=(
            "Path to an existing manifest JSON to verify against. "
            "If omitted, a fresh manifest is emitted."
        ),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help=(
            "Glob fragment to skip (relative to root). Repeatable.  Useful "
            "for transient lock-files or extraction temp directories."
        ),
    )
    return parser


def _sha256_file(path: Path) -> tuple[str, int]:
    """Return ``(hex_digest, size_bytes)`` for ``path`` (constant memory)."""
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def build_manifest(root: Path, excludes: list[str] | None = None) -> dict[str, Any]:
    """Walk ``root`` and return a manifest dict.

    Sorted by relative path so the output is byte-stable across hosts
    and filesystems — important for signing and for diff-based review.
    """
    if not root.exists():
        raise FileNotFoundError(f"dataset cache root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"dataset cache root is not a directory: {root}")
    excludes = excludes or []

    entries: list[dict[str, Any]] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if any(frag in rel for frag in excludes):
            continue
        digest, size = _sha256_file(p)
        entries.append({"path": rel, "size": size, "sha256": digest})

    # Aggregate root digest: SHA-256 over the canonical JSON of entries.
    # This single hex string is what operators sign / pin in release
    # manifests; comparing it is cheaper than diffing entries by hand.
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    root_digest = hashlib.sha256(canonical).hexdigest()

    return {
        "schema": _SCHEMA,
        "root": str(root),
        "file_count": len(entries),
        "total_bytes": sum(e["size"] for e in entries),
        "root_sha256": root_digest,
        "entries": entries,
    }


def verify_manifest(
    root: Path,
    manifest: dict[str, Any],
    excludes: list[str] | None = None,
) -> dict[str, Any]:
    """Compare ``root`` against a saved ``manifest`` and report drift.

    The returned dict has ``status`` ∈ {``"ok"``, ``"drift"``} and
    lists ``missing`` (in manifest, not on disk), ``unexpected``
    (on disk, not in manifest), and ``mismatched`` (path present in
    both with different SHA-256).
    """
    current = build_manifest(root, excludes)
    saved_entries = {e["path"]: e for e in manifest.get("entries", [])}
    current_entries = {e["path"]: e for e in current["entries"]}

    missing = sorted(set(saved_entries) - set(current_entries))
    unexpected = sorted(set(current_entries) - set(saved_entries))
    mismatched = sorted(
        p
        for p in set(saved_entries) & set(current_entries)
        if saved_entries[p]["sha256"] != current_entries[p]["sha256"]
    )

    return {
        "status": "ok" if not (missing or unexpected or mismatched) else "drift",
        "missing": missing,
        "unexpected": unexpected,
        "mismatched": mismatched,
        "current_root_sha256": current["root_sha256"],
        "saved_root_sha256": manifest.get("root_sha256"),
    }


def _collect(args: argparse.Namespace) -> Certificate:
    root = Path(args.root).resolve()

    if args.verify:
        saved = json.loads(Path(args.verify).read_text())
        result = verify_manifest(root, saved, args.exclude)
        status = "ok" if result["status"] == "ok" else "fail"
        warnings: list[str] = []
        if result["missing"]:
            warnings.append(f"{len(result['missing'])} files missing on disk")
        if result["unexpected"]:
            warnings.append(f"{len(result['unexpected'])} files not in manifest")
        if result["mismatched"]:
            warnings.append(f"{len(result['mismatched'])} files with SHA-256 mismatch")
        body: dict[str, Any] = {
            "mode": "verify",
            "root": str(root),
            "saved_manifest": args.verify,
            **result,
        }
        return Certificate(
            tool="dataset_checksum_manifest",
            schema=_SCHEMA,
            status=status,
            body=body,
            warnings=warnings,
        )

    manifest = build_manifest(root, args.exclude)
    return Certificate(
        tool="dataset_checksum_manifest",
        schema=_SCHEMA,
        status="ok",
        body={"mode": "emit", **manifest},
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
