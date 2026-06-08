# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool
from omni_mercury_engine.tools.secret_scan_baseline import _scan_file

_SCHEMA = "mercury.tools.config_secret_redactor/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.config_secret_redactor",
        description="Refuse to commit configs that contain secret-shaped values.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["configs"],
        help="Config files or directories to scan (default: configs/).",
    )
    parser.add_argument("--entropy-min", type=float, default=4.5)
    parser.add_argument("--entropy-run-min", type=int, default=24)
    return parser


def _gather(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out.extend(
                sorted(
                    q
                    for q in p.rglob("*")
                    if q.is_file()
                    and q.suffix.lower() in {".yaml", ".yml", ".toml", ".json", ".ini", ".env"}
                )
            )
        elif p.is_file():
            out.append(p)
    return out


def _collect(args: argparse.Namespace) -> Certificate:
    files = _gather(args.paths)
    findings: list[dict[str, Any]] = []
    for f in files:
        for finding in _scan_file(
            f,
            entropy_min=float(args.entropy_min),
            entropy_run_min=int(args.entropy_run_min),
        ):
            findings.append({"path": str(f), **finding})

    body: dict[str, Any] = {
        "paths_scanned": [str(p) for p in files],
        "findings": findings,
        "finding_count": len(findings),
    }
    return Certificate(
        tool="config_secret_redactor",
        schema=_SCHEMA,
        status="fail" if findings else "ok",
        body=body,
        warnings=[f"{f['path']}:{f['line']} ({f['kind']})" for f in findings],
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
