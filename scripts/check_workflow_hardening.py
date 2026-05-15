#!/usr/bin/env python3
"""Repository-specific GitHub Actions hardening checks."""

from __future__ import annotations

import re
import sys
from pathlib import Path


WORKFLOW_DIR = Path(".github/workflows")
WRITE_OK = {
    "benchmark.yml",
    "dependabot-auto-merge.yml",
    "release.yml",
}
SHA_REF_RE = re.compile(r"^[0-9a-f]{40}$")
USES_RE = re.compile(r"^\s*uses:\s*([^@\s]+)@([^#\s]+)", re.MULTILINE)


def has_top_level_key(text: str, key: str) -> bool:
    return re.search(rf"^{re.escape(key)}\s*:", text, re.MULTILINE) is not None


def check_workflow(path: Path) -> list[str]:
    errors: list[str] = []
    warnings: list[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")

    if "pull_request_target:" in text:
        errors.append(f"{path}: pull_request_target is not allowed without a security review")

    if not has_top_level_key(text, "permissions"):
        errors.append(f"{path}: add top-level least-privilege permissions")
    elif path.name not in WRITE_OK and re.search(r"^  contents:\s*write\b", text, re.MULTILINE):
        errors.append(f"{path}: contents: write requires explicit allow-listing")

    if not has_top_level_key(text, "concurrency"):
        errors.append(f"{path}: add concurrency cancellation for PR velocity")

    for match in USES_RE.finditer(text):
        action, ref = match.groups()
        if action.startswith("./") or action.startswith("docker://"):
            continue
        if not SHA_REF_RE.match(ref):
            warnings.append(f"{path}: {action}@{ref} is tag-pinned, not SHA-pinned")

    for warning in warnings:
        print(f"::warning title=Workflow supply-chain hardening::{warning}")
    return errors


def main() -> int:
    errors: list[str] = []
    for path in sorted(WORKFLOW_DIR.glob("*.yml")):
        errors.extend(check_workflow(path))

    if errors:
        print("Workflow hardening check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("Workflow hardening check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
