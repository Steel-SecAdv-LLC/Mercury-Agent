# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Operator tool: verify the hash-chained audit log produced by :mod:`audit_log_signer`."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.audit_log_verifier/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.audit_log_verifier",
        description="Verify the hash-chained audit log produced by audit_log_signer.",
    )
    parser.add_argument("--log", required=True)
    parser.add_argument(
        "--key-hex",
        default=os.environ.get("MERCURY_AUDIT_HMAC_KEY"),
        help="HMAC key (64 hex chars).  Defaults to $MERCURY_AUDIT_HMAC_KEY.",
    )
    return parser


def _hmac(key: bytes, data: bytes) -> str:
    return hmac.new(key, data, hashlib.sha256).hexdigest()


def _collect(args: argparse.Namespace) -> Certificate:
    log_path = Path(args.log)
    if not log_path.exists():
        return Certificate(
            tool="audit_log_verifier",
            schema=_SCHEMA,
            status="fail",
            body={"log": str(log_path), "error": "audit log not found"},
        )
    if not args.key_hex or len(args.key_hex) != 64:
        return Certificate(
            tool="audit_log_verifier",
            schema=_SCHEMA,
            status="fail",
            body={"log": str(log_path), "error": "key not supplied or wrong length"},
        )
    key = bytes.fromhex(args.key_hex)

    issues: list[str] = []
    entries: list[dict[str, Any]] = []
    prev = "0" * 64
    for ln, raw in enumerate(log_path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError as exc:
            issues.append(f"line {ln}: unparseable JSON: {exc}")
            continue
        sig = entry.pop("hmac", None)
        if entry.get("prev_hmac") != prev:
            issues.append(f"line {ln}: prev_hmac {entry.get('prev_hmac')!r} != expected {prev!r}")
        expected = _hmac(key, json.dumps(entry, sort_keys=True).encode("utf-8"))
        if sig != expected:
            issues.append(f"line {ln}: hmac mismatch (entry forged or key wrong)")
        prev = sig or prev
        entries.append({**entry, "hmac": sig})

    body: dict[str, Any] = {
        "log": str(log_path),
        "entry_count": len(entries),
        "issues": issues,
        "head_hmac": prev,
    }
    return Certificate(
        tool="audit_log_verifier",
        schema=_SCHEMA,
        status="fail" if issues else "ok",
        body=body,
        warnings=issues,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
