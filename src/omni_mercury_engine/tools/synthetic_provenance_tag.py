# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, atomic_write_text, run_tool

_SCHEMA = "mercury.tools.synthetic_provenance_tag/v1"
_TAG_SCHEMA = "mercury.synthetic_provenance/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.synthetic_provenance_tag",
        description="Emit or verify a synthetic-data provenance tag sidecar.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emit", action="store_true")
    group.add_argument("--verify", action="store_true")
    parser.add_argument("--data", required=True, help="Path to the synthetic data file.")
    parser.add_argument(
        "--tag", default=None, help="Sidecar path (default: <data>.provenance.json)."
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--upstream-attempted", default=None)
    parser.add_argument("--fallback-reason", default=None)
    return parser


def _sidecar_path(data: Path) -> Path:
    return data.with_suffix(data.suffix + ".provenance.json")


def _collect(args: argparse.Namespace) -> Certificate:
    data = Path(args.data)
    if not data.is_file():
        return Certificate(
            tool="synthetic_provenance_tag",
            schema=_SCHEMA,
            status="fail",
            body={"error": f"data file not found: {data}"},
        )
    sidecar = Path(args.tag) if args.tag else _sidecar_path(data)
    data_digest = hashlib.sha256(data.read_bytes()).hexdigest()

    if args.emit:
        tag = {
            "schema": _TAG_SCHEMA,
            "data_path": str(data),
            "data_sha256": data_digest,
            "synthesised_at": _dt.datetime.now(_dt.UTC).isoformat(),
            "seed": args.seed,
            "rows": args.rows,
            "method": args.method,
            "upstream_attempted": args.upstream_attempted,
            "fallback_reason": args.fallback_reason,
        }
        atomic_write_text(sidecar, json.dumps(tag, indent=2, sort_keys=True) + "\n")
        return Certificate(
            tool="synthetic_provenance_tag",
            schema=_SCHEMA,
            status="ok",
            body={"mode": "emit", "tag_path": str(sidecar), "tag": tag},
        )

    # verify
    if not sidecar.exists():
        return Certificate(
            tool="synthetic_provenance_tag",
            schema=_SCHEMA,
            status="fail",
            body={"error": f"sidecar tag not found: {sidecar}", "data_path": str(data)},
        )
    tag = json.loads(sidecar.read_text())
    issues: list[str] = []
    if tag.get("schema") != _TAG_SCHEMA:
        issues.append(f"unexpected schema {tag.get('schema')!r}")
    saved_digest = tag.get("data_sha256")
    if saved_digest and saved_digest != data_digest:
        issues.append(f"data digest drifted: saved {saved_digest}, current {data_digest}")
    body: dict[str, Any] = {
        "mode": "verify",
        "tag_path": str(sidecar),
        "data_sha256": data_digest,
        "tag": tag,
        "issues": issues,
    }
    return Certificate(
        tool="synthetic_provenance_tag",
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
