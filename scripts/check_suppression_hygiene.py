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

TYPE_IGNORE_RE = re.compile(r"^\s*#\s*type:\s*ignore\b(?!\[)")
NOSEC_RE = re.compile(r"(?:^|\s)#\s*nosec")
EXPLICIT_NOSEC_RE = re.compile(r"(?:^|\s)#\s*nosec(?:\s*:\s*|\s+)B\d{3}\b")
SEMGREP_RE = re.compile(r"#\s*semgrep:\s*ignore(?!\s+[A-Za-z0-9_.:/-]+)")
CVE_RE = re.compile(r"^(CVE-\d{4}-\d+|GHSA-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{4})$", re.I)
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
    # ``check_trivyignore`` is scoped to the user's requested paths
    # rather than always running unconditionally.  This keeps the CLI
    # contract consistent: a developer running
    # ``python check_suppression_hygiene.py src/`` to scope the check
    # to just ``src/`` no longer accidentally also runs the Trivy
    # waiver audit.  The default invocation (no positional arguments
    # → ``DEFAULT_PATHS``) still includes ``.trivyignore`` so the CI
    # gate is unchanged.
    trivyignore = Path(".trivyignore")
    if trivyignore in selected_paths or any(
        path == trivyignore
        or (path.is_dir() and trivyignore.resolve().is_relative_to(path.resolve()))
        for path in selected_paths
        if path.exists()
    ):
        errors.extend(check_trivyignore(trivyignore))

    if errors:
        print("Suppression hygiene check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("Suppression hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
