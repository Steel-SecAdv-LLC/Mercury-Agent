#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Validate that source-control suppressions stay explicit and reviewable."""

from __future__ import annotations

import argparse
import io
import re
import sys
import tokenize
from pathlib import Path

DEFAULT_PATHS = (
    Path("src"),
    Path("tests"),
    Path("scripts"),
    Path("benchmarks"),
    Path(".github/workflows"),
    Path("pyproject.toml"),
    Path(".safety-policy.yml"),
    Path(".safety-policy-v2.yml"),
)

TYPE_IGNORE_RE = re.compile(r"^\s*#\s*type:\s*ignore\b(?!\[)")
NOSEC_RE = re.compile(r"(?:^|\s)#\s*nosec")
EXPLICIT_NOSEC_RE = re.compile(r"(?:^|\s)#\s*nosec(?:\s*:\s*|\s+)B\d{3}\b")
SEMGREP_RE = re.compile(r"#\s*semgrep:\s*ignore(?!\s+[A-Za-z0-9_.:/-]+)")
# ``.md`` is intentionally excluded: documentation regularly includes
# the literal pragma strings inside fenced code blocks (worked examples
# of `# nosec B603` / `# type: ignore[attr-defined]` etc.), and the
# regex matchers below cannot reliably distinguish a fenced example
# from a real suppression in markdown.  Suppression hygiene is enforced
# on the source/config surfaces that actually emit the pragmas.
TEXT_SUFFIXES = {".py", ".yml", ".yaml", ".toml"}
# Directories anywhere in a scanned tree that should be skipped because
# they hold vendored or generated content that does not belong to the
# project's suppression posture (e.g. a developer's local ``.venv``
# under ``src/``).  Match by *any* path part starting with the prefix
# so descendants are also excluded.
EXCLUDED_DIR_PREFIXES = (
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "build",
    "dist",
    ".eggs",
)


def iter_files(paths: tuple[Path, ...]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            files.append(path)
            continue
        files.extend(
            child
            for child in path.rglob("*")
            if child.is_file()
            and child.suffix in TEXT_SUFFIXES
            and not any(part.startswith(EXCLUDED_DIR_PREFIXES) for part in child.parts)
        )
    return sorted(set(files))


def iter_comment_lines(path: Path, text: str) -> list[tuple[int, str]]:
    if path.suffix != ".py":
        return [(lineno, line) for lineno, line in enumerate(text.splitlines(), start=1)]

    comments: list[tuple[int, str]] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT:
                comments.append((token.start[0], token.string))
    except tokenize.TokenError:
        comments.extend((lineno, line) for lineno, line in enumerate(text.splitlines(), start=1))
    return comments


def check_text_file(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for lineno, line in iter_comment_lines(path, text):
        if TYPE_IGNORE_RE.search(line):
            errors.append(
                f"{path}:{lineno}: use a specific mypy code, e.g. # type: ignore[attr-defined]"
            )
        if NOSEC_RE.search(line) and not EXPLICIT_NOSEC_RE.search(line):
            errors.append(f"{path}:{lineno}: use an explicit Bandit code, e.g. # nosec B603")
        if SEMGREP_RE.search(line):
            errors.append(f"{path}:{lineno}: semgrep ignores must name the ignored rule")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=list(DEFAULT_PATHS))
    args = parser.parse_args()

    errors: list[str] = []
    selected_paths = tuple(args.paths)
    for path in iter_files(selected_paths):
        errors.extend(check_text_file(path))

    if errors:
        print("Suppression hygiene check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("Suppression hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
