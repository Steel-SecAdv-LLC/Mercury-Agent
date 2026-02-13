#!/usr/bin/env python3
"""
Mercury Agent - Documentation audit for internal/aspirational content.

Scans documentation files for content that should be removed or updated:
- Internal meeting notes and planning docs
- Aspirational claims without backing data
- Version references that don't match v1.4.0
- Business/organizational documentation

Copyright (C) 2025 Steel Security Advisors LLC
License: GPL-3.0+
"""

from __future__ import annotations

import re
from pathlib import Path

PATTERNS_TO_FLAG = [
    (r"TODO.*timeline", "TODO with timeline reference"),
    (r"internal.*meeting", "Internal meeting reference"),
    (r"business.*strategy", "Business strategy reference"),
    (r"revolutionary", "Aspirational adjective"),
    (r"groundbreaking", "Aspirational adjective"),
    (r"v1\.5\.0", "Old version reference (should be v1.4.0)"),
    (r"v1\.5\.1", "Invalid version reference"),
]

PATHS_TO_CHECK = [
    "README.md",
    "ARCHITECTURE.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "SECURITY.md",
    "DEPRECATION.md",
    "docs/BENCHMARKS.md",
    "docs/LIVE_DATA_VALIDATION.md",
    "docs/ROADMAP.md",
]


def audit() -> list[tuple[str, int, str, str]]:
    """Find content that may need review.

    Returns:
        List of (filepath, line_number, matched_text, reason) tuples.
    """
    issues: list[tuple[str, int, str, str]] = []

    for path_str in PATHS_TO_CHECK:
        filepath = Path(path_str)
        if not filepath.exists():
            continue
        if not filepath.is_file():
            continue

        with open(filepath) as f:
            for i, line in enumerate(f, 1):
                for pattern, reason in PATTERNS_TO_FLAG:
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append((str(filepath), i, line.strip()[:80], reason))

    return issues


def main() -> None:
    """Run audit and print results."""
    issues = audit()

    if issues:
        print(f"Content to review ({len(issues)} items):")
        for filepath, line_no, text, reason in issues:
            print(f"  {filepath}:{line_no}: [{reason}] {text}")
    else:
        print("All documentation is clean")


if __name__ == "__main__":
    main()
