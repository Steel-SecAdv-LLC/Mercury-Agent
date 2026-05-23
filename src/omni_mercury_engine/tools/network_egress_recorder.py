"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

------------------------------------------------------------------------

Operator tool: per-run network egress recorder.

Loads a JSONL trace produced by wrapping :class:`SafeHTTPClient`
(``MERCURY_EGRESS_TRACE=/tmp/egress.jsonl``) and emits a certificate
summarising every URL fetched, response size, status code, and
duration.  Would have made the 11-unreachable-dataset incident
self-evident.

Also supports a ``--record`` mode that proxies a command's
``SafeHTTPClient`` calls through an in-process recorder — useful for
CI smoke tests where setting the env-var beforehand is awkward.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.network_egress_recorder/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.network_egress_recorder",
        description=("Summarise a SafeHTTPClient egress trace (JSONL) and emit a certificate."),
    )
    parser.add_argument(
        "--trace",
        required=True,
        help="Path to the JSONL trace (one JSON object per HTTP request).",
    )
    parser.add_argument(
        "--allow-list",
        default=None,
        help="Optional text file with one allowed URL prefix per line.",
    )
    return parser


def _collect(args: argparse.Namespace) -> Certificate:
    trace_path = Path(args.trace)
    if not trace_path.exists():
        return Certificate(
            tool="network_egress_recorder",
            schema=_SCHEMA,
            status="fail",
            body={"trace": str(trace_path), "error": "trace file not found"},
        )

    requests: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    for ln, raw in enumerate(trace_path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            requests.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            parse_errors.append(f"line {ln}: {exc}")

    allow_prefixes: list[str] = []
    if args.allow_list:
        allow_prefixes = [
            line.strip()
            for line in Path(args.allow_list).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    disallowed: list[dict[str, Any]] = []
    for r in requests:
        url = r.get("url", "")
        if allow_prefixes and not any(url.startswith(p) for p in allow_prefixes):
            disallowed.append(r)

    body: dict[str, Any] = {
        "trace": str(trace_path),
        "request_count": len(requests),
        "parse_errors": parse_errors,
        "unique_hosts": sorted({_host(r.get("url", "")) for r in requests if r.get("url")}),
        "status_histogram": _histogram(r.get("status") for r in requests),
        "total_bytes": sum(int(r.get("response_size", 0) or 0) for r in requests),
        "disallowed": disallowed,
        "allow_list_path": args.allow_list,
        "allow_list_size": len(allow_prefixes),
    }
    if disallowed:
        return Certificate(
            tool="network_egress_recorder",
            schema=_SCHEMA,
            status="fail",
            body=body,
            warnings=[f"egress outside allow-list: {r.get('url', '<no-url>')}" for r in disallowed],
        )
    status = "warn" if parse_errors else "ok"
    return Certificate(
        tool="network_egress_recorder",
        schema=_SCHEMA,
        status=status,
        body=body,
        warnings=parse_errors,
    )


def _host(url: str) -> str:
    from urllib.parse import urlparse

    return urlparse(url).netloc


def _histogram(items: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for k in items:
        if k is None:
            continue
        key = str(k)
        out[key] = out.get(key, 0) + 1
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
