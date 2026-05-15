#!/usr/bin/env python3
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
    Path(".trivyignore"),
    Path(".safety-policy.yml"),
    Path(".safety-policy-v2.yml"),
)

TYPE_IGNORE_RE = re.compile(r"#\s*type:\s*ignore(?!\[)")
NOSEC_RE = re.compile(r"(?:^|\s)#\s*nosec")
EXPLICIT_NOSEC_RE = re.compile(r"(?:^|\s)#\s*nosec(?:\s*:\s*|\s+)B\d{3}\b")
SEMGREP_RE = re.compile(r"#\s*semgrep:\s*ignore(?!\s+[A-Za-z0-9_.:/-]+)")
CVE_RE = re.compile(r"^(CVE-\d{4}-\d+|GHSA-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{4})$", re.I)
TEXT_SUFFIXES = {".py", ".yml", ".yaml", ".toml", ".md"}


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
            and not any(part.startswith(".mypy_cache") for part in child.parts)
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
                f"{path}:{lineno}: use a specific mypy code, e.g. " "# type: ignore[attr-defined]"
            )
        if NOSEC_RE.search(line) and not EXPLICIT_NOSEC_RE.search(line):
            errors.append(f"{path}:{lineno}: use an explicit Bandit code, e.g. # nosec B603")
        if SEMGREP_RE.search(line):
            errors.append(f"{path}:{lineno}: semgrep ignores must name the ignored rule")
    return errors


def check_trivyignore(path: Path) -> list[str]:
    if not path.exists():
        return []
    errors: list[str] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not CVE_RE.match(stripped):
            continue
        window = "\n".join(lines[max(0, idx - 35) : idx])
        if "Justification for Acceptance" not in window:
            errors.append(f"{path}:{idx + 1}: Trivy waiver lacks acceptance rationale")
        if "Expiry:" not in window:
            errors.append(f"{path}:{idx + 1}: Trivy waiver lacks expiry date")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=list(DEFAULT_PATHS))
    args = parser.parse_args()

    errors: list[str] = []
    selected_paths = tuple(args.paths)
    for path in iter_files(selected_paths):
        errors.extend(check_text_file(path))
    errors.extend(check_trivyignore(Path(".trivyignore")))

    if errors:
        print("Suppression hygiene check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("Suppression hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
