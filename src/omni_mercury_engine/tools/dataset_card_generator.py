# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Operator tool: Google-style dataset card generator.

Composes a Markdown dataset card from sibling tool evidence:

* upstream licence + SPDX expression from
  :mod:`dataset_license_auditor`,
* output schema from :mod:`loader_schema_pinner`,
* PII status from :mod:`pii_scrubber_probe`,
* checksum digest from :mod:`dataset_checksum_manifest`.

The card itself becomes the auditor's single source of truth for the
dataset; the per-tool certs are the signed evidence behind it.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, atomic_write_text, run_tool

_SCHEMA = "mercury.tools.dataset_card_generator/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.dataset_card_generator",
        description="Generate a Google-style dataset card from sibling tool evidence.",
    )
    parser.add_argument("--name", required=True, help="Dataset display name.")
    parser.add_argument("--license-cert", default=None)
    parser.add_argument("--schema-cert", default=None)
    parser.add_argument("--pii-cert", default=None)
    parser.add_argument("--checksum-cert", default=None)
    parser.add_argument("--markdown", default=None, help="Output Markdown file.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        parsed = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _card_to_markdown(card: dict[str, Any]) -> str:
    lines = [f"# Dataset Card — {card['name']}", ""]
    lines += [f"_Generated {card['generated_at']}_", ""]
    if card.get("license"):
        lines += ["## Licence", "", "```json", json.dumps(card["license"], indent=2), "```", ""]
    if card.get("schema"):
        lines += ["## Schema", "", "```json", json.dumps(card["schema"], indent=2), "```", ""]
    if card.get("pii"):
        lines += ["## PII", "", "```json", json.dumps(card["pii"], indent=2), "```", ""]
    if card.get("checksum"):
        lines += ["## Checksum", "", "```json", json.dumps(card["checksum"], indent=2), "```", ""]
    return "\n".join(lines) + "\n"


def _collect(args: argparse.Namespace) -> Certificate:
    card: dict[str, Any] = {
        "name": args.name,
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "license": _load(args.license_cert),
        "schema": _load(args.schema_cert),
        "pii": _load(args.pii_cert),
        "checksum": _load(args.checksum_cert),
    }
    md = _card_to_markdown(card)
    if args.markdown and not args.dry_run:
        atomic_write_text(Path(args.markdown), md)
    return Certificate(
        tool="dataset_card_generator",
        schema=_SCHEMA,
        status="ok",
        body={"card": card, "markdown_path": args.markdown, "dry_run": bool(args.dry_run)},
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
