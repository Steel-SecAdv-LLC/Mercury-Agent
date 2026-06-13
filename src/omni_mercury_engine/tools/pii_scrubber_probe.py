# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Operator tool: handwritten PII regex/heuristic gate.

Scans the supplied text (file or stdin) and reports matches for:

* email addresses,
* US Social Security Numbers (XXX-XX-XXXX with valid area-prefix),
* US phone numbers (E.164, dash/dot/space-separated, parens area code),
* ICD-10 diagnostic codes (e.g. ``A00.0``, ``Z99.89``),
* high-precision latitude/longitude pairs (>= 5 decimal places).

Fails on any leak.  The probe is intended to be wired into every
loader's smoke-test path so a silent regression in upstream data
sanitation surfaces immediately.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.pii_scrubber_probe/v1"

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    # US SSN: AAA-GG-SSSS with AAA != 000, 666, 9XX; GG != 00; SSSS != 0000.
    (
        "ssn",
        re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
    ),
    (
        "phone_us",
        re.compile(r"(?:\+?1[-.\s]?)?(?:\(?[2-9]\d{2}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"),
    ),
    ("icd10", re.compile(r"\b[A-TV-Z]\d{2}(?:\.[0-9A-Z]{1,4})?\b")),
    # Lat,lon with >= 5 decimals — high enough precision to identify a building.
    (
        "geo_high_precision",
        re.compile(r"\b-?\d{1,3}\.\d{5,}\s*,\s*-?\d{1,3}\.\d{5,}\b"),
    ),
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.pii_scrubber_probe",
        description="Scan a text file (or stdin) for PII leaks and fail-closed on any match.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", default=None, help="Text file to scan.")
    group.add_argument("--stdin", action="store_true", help="Read from stdin.")
    parser.add_argument(
        "--max-snippets",
        type=int,
        default=5,
        help="Per-pattern maximum snippets to embed in the cert body.",
    )
    return parser


def scan_text(text: str, max_snippets: int = 5) -> dict[str, Any]:
    """Run every pattern and return ``{pattern: {count, snippets}}``."""
    out: dict[str, Any] = {}
    for name, pat in _PATTERNS:
        matches = pat.findall(text)
        if not matches:
            continue
        snippets = matches[:max_snippets]
        # Redact: keep the first and last char, replace the middle with *.
        redacted = []
        for s in snippets:
            if isinstance(s, tuple):
                s = "/".join(s)
            if len(s) <= 2:
                redacted.append(s)
            else:
                redacted.append(s[0] + "*" * (len(s) - 2) + s[-1])
        out[name] = {"count": len(matches), "redacted_snippets": redacted}
    return out


def _collect(args: argparse.Namespace) -> Certificate:
    if args.file:
        text = Path(args.file).read_text(errors="replace")
        source = args.file
    else:
        text = sys.stdin.read()
        source = "<stdin>"
    findings = scan_text(text, max_snippets=int(args.max_snippets))
    body: dict[str, Any] = {
        "source": source,
        "bytes_scanned": len(text.encode("utf-8")),
        "findings": findings,
        "leak_count": sum(int(v["count"]) for v in findings.values()),
    }
    if findings:
        return Certificate(
            tool="pii_scrubber_probe",
            schema=_SCHEMA,
            status="fail",
            body=body,
            warnings=[f"{k}: {v['count']} match(es)" for k, v in findings.items()],
        )
    return Certificate(
        tool="pii_scrubber_probe",
        schema=_SCHEMA,
        status="ok",
        body=body,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
