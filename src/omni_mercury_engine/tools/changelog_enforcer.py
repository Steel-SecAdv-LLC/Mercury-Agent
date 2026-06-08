# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.changelog_enforcer/v1"
_UNRELEASED_HEADERS = ("## [Unreleased]", "## Unreleased")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.changelog_enforcer",
        description="Verify every public-surface change has a CHANGELOG entry.",
    )
    parser.add_argument("--changelog", default="CHANGELOG.md")
    parser.add_argument("--base-ref", default="HEAD~1")
    parser.add_argument(
        "--watch",
        action="append",
        default=[
            "src/omni_mercury_engine/__init__.py",
            "src/omni_mercury_engine/__init__.pyi",
            "src/omni_mercury_engine/tools/__init__.py",
        ],
        help="Public-surface file (repeatable).  Modifications require a CHANGELOG bullet.",
    )
    return parser


def _git_diff_names(base: str) -> list[str]:
    try:
        out = subprocess.check_output(
            ["git", "diff", "--name-only", base, "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode()
        return [line.strip() for line in out.splitlines() if line.strip()]
    except subprocess.CalledProcessError:
        return []


def _git_diff_added_lines(base: str, path: str) -> list[str]:
    try:
        out = subprocess.check_output(
            ["git", "diff", "-U0", base, "HEAD", "--", path],
            stderr=subprocess.DEVNULL,
        ).decode()
    except subprocess.CalledProcessError:
        return []
    return [
        line[1:].rstrip()
        for line in out.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]


def _unreleased_added_bullets(base: str, changelog: str) -> list[str]:
    """Return bullets added under the ``[Unreleased]`` heading since ``base``."""
    added = _git_diff_added_lines(base, changelog)
    return [line for line in added if line.lstrip().startswith(("-", "*"))]


def _collect(args: argparse.Namespace) -> Certificate:
    changed = _git_diff_names(args.base_ref)
    watch_hit = [f for f in changed if f in args.watch]
    bullets = _unreleased_added_bullets(args.base_ref, args.changelog)
    changelog_present = Path(args.changelog).exists()
    body: dict[str, Any] = {
        "base_ref": args.base_ref,
        "changed_files": changed,
        "watched_files_changed": watch_hit,
        "new_unreleased_bullets": bullets,
        "changelog_path": args.changelog,
        "changelog_present": changelog_present,
    }
    failures: list[str] = []
    if not changelog_present:
        failures.append(f"{args.changelog} missing")
    if watch_hit and not bullets:
        failures.append(
            f"public surface changed ({watch_hit}) but no new bullets under {_UNRELEASED_HEADERS[0]}"
        )
    return Certificate(
        tool="changelog_enforcer",
        schema=_SCHEMA,
        status="fail" if failures else "ok",
        body=body,
        warnings=failures,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
