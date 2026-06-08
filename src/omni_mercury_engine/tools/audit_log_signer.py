# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, atomic_write_text, run_tool

_SCHEMA = "mercury.tools.audit_log_signer/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.audit_log_signer",
        description="Append a signed entry to the rolling audit log (hash-chained).",
    )
    parser.add_argument("--log", required=True, help="JSONL audit log path.")
    parser.add_argument("--actor", required=True, help="Subject acting (user/component).")
    parser.add_argument("--action", required=True, help="Action performed.")
    parser.add_argument(
        "--detail",
        action="append",
        default=[],
        help="Repeatable key=value detail pair.",
    )
    parser.add_argument(
        "--key-hex",
        default=os.environ.get("MERCURY_AUDIT_HMAC_KEY"),
        help=(
            "Hex-encoded HMAC key (32 bytes / 64 hex chars).  Defaults to "
            "$MERCURY_AUDIT_HMAC_KEY.  Generated ephemerally when omitted; "
            "production deployments MUST pin a key (env var) so a verifier "
            "can re-derive the chain."
        ),
    )
    return parser


def _hmac(key: bytes, data: bytes) -> str:
    return hmac.new(key, data, hashlib.sha256).hexdigest()


def _parse_detail(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        out[k] = v
    return out


def _read_last_hmac(path: Path) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return "0" * 64
    last = ""
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            last = line
    if not last:
        return "0" * 64
    try:
        prior = json.loads(last)
    except json.JSONDecodeError:
        return "0" * 64
    if isinstance(prior, dict):
        hmac_val = prior.get("hmac")
        if isinstance(hmac_val, str):
            return hmac_val
    return "0" * 64


def _collect(args: argparse.Namespace) -> Certificate:
    log_path = Path(args.log)
    key_hex = args.key_hex or secrets.token_hex(32)
    if len(key_hex) != 64:
        return Certificate(
            tool="audit_log_signer",
            schema=_SCHEMA,
            status="fail",
            body={"error": f"--key-hex must be 64 hex chars; got {len(key_hex)}"},
        )
    key = bytes.fromhex(key_hex)
    prev = _read_last_hmac(log_path)
    entry: dict[str, Any] = {
        "ts": _dt.datetime.now(_dt.UTC).isoformat(),
        "actor": args.actor,
        "action": args.action,
        "detail": _parse_detail(args.detail),
        "prev_hmac": prev,
    }
    entry["hmac"] = _hmac(key, json.dumps(entry, sort_keys=True).encode("utf-8"))

    # Append-only write: read existing content + append the new line + atomic replace.
    existing = log_path.read_text() if log_path.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    atomic_write_text(log_path, existing + json.dumps(entry, sort_keys=True) + "\n")

    body: dict[str, Any] = {
        "log": str(log_path),
        "entry": entry,
        "key_pinned_via_env": bool(args.key_hex),
    }
    warnings: list[str] = []
    if not args.key_hex:
        warnings.append(
            "ephemeral HMAC key generated; pin $MERCURY_AUDIT_HMAC_KEY for verifier reuse"
        )
    return Certificate(
        tool="audit_log_signer",
        schema=_SCHEMA,
        status="warn" if warnings else "ok",
        body=body,
        warnings=warnings,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
