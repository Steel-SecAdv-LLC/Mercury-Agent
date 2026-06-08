# Copyright (C) 2025 Steel Security Advisors LLC
"""Normalize Python file headers and module docstrings."""

from __future__ import annotations

import argparse
import ast
import difflib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET_DIRS = (
    "src",
    "tests",
    "scripts",
    "tools",
    "research",
    "benchmarks",
    "examples",
    "assets",
)
CANONICAL_HEADER = "# Copyright (C) 2025 Steel Security Advisors LLC"
CODING_RE = re.compile(r"^#.*coding[:=]\s*[-\w.]+")
LICENSE_MARKERS = (
    "This program is free software",
    "GNU General Public License",
    "WITHOUT ANY WARRANTY",
    "You should have received a copy",
    "SPDX-License-Identifier",
    "Copyright (C) 2025 Steel Security Advisors LLC",
    "Copyright (C) Steel Security Advisors LLC",
)
SKIP_LINE_FRAGMENTS = (
    "Copyright",
    "License",
    "GPL",
    "Mercury Agent",
    "https://www.gnu.org",
    "https://gnu.org",
    "https://github.com/Steel-SecAdv-LLC",
)
SUMMARY_ENDINGS = (".", "!", "?")


@dataclass(frozen=True)
class StringSpan:
    """Record a top-level string expression and its source span."""

    text: str
    start_line: int
    end_line: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="fail if files are not normalized")
    mode.add_argument("--apply", action="store_true", help="rewrite files in place")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="optional files or directories to check instead of the repository targets",
    )
    return parser.parse_args()


def iter_python_files(paths: tuple[Path, ...]) -> list[Path]:
    """Return Python files under the requested paths."""
    if paths:
        roots = [path if path.is_absolute() else ROOT / path for path in paths]
    else:
        roots = [ROOT / dirname for dirname in TARGET_DIRS]

    files: list[Path] = []
    for path in roots:
        if not path.exists():
            continue
        if path.is_file():
            if path.suffix == ".py":
                files.append(path)
            continue
        files.extend(sorted(child for child in path.rglob("*.py") if child.is_file()))
    return sorted(dict.fromkeys(files))


def prologue_end(lines: list[str]) -> int:
    """Return the count of shebang and encoding-cookie lines to preserve first."""
    index = 0
    if lines and lines[0].startswith("#!"):
        index = 1
    if index < len(lines) and CODING_RE.match(lines[index]):
        index += 1
    elif index == 0 and len(lines) > 1 and CODING_RE.match(lines[1]):
        index = 2
    return index


def collect_top_level_strings(path: Path, text: str) -> list[StringSpan]:
    """Collect standalone top-level string expressions from a module."""
    try:
        module = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        raise SystemExit(f"{path}: cannot parse Python source: {exc}") from exc

    spans: list[StringSpan] = []
    for node in module.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and node.end_lineno is not None
        ):
            spans.append(StringSpan(node.value.value, node.lineno, node.end_lineno))
    return spans


def is_license_text(text: str) -> bool:
    """Return whether a string expression is a duplicated license notice."""
    return any(marker in text for marker in LICENSE_MARKERS)


def _is_boilerplate_line(line: str) -> bool:
    """Return True if the line is part of license/copyright boilerplate."""
    return any(frag in line for frag in SKIP_LINE_FRAGMENTS) or line.startswith(
        ("This program", "You should", "See ", "see ")
    )


def derive_summary_from_license(text: str) -> str | None:
    """Extract a non-license summary from a legacy boilerplate string."""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Mercury Agent - "):
            candidate = line.removeprefix("Mercury Agent - ").strip()
            if not _is_boilerplate_line(candidate):
                return candidate
            continue
        if _is_boilerplate_line(line):
            continue
        return line
    return None


def default_summary(path: Path) -> str:
    """Build a deterministic fallback module summary from the path."""
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    if path.name == "__init__.py":
        package = path.parent.name.replace("_", " ")
        return f"Initialize the {package} package."
    stem = path.stem
    words = stem.removeprefix("test_").replace("_", " ").replace("-", " ")
    if rel.parts and rel.parts[0] == "tests":
        return f"Test {words}."
    if rel.parts and rel.parts[0] in {"scripts", "tools"}:
        return f"Run {words}."
    return f"Provide {words}."


def choose_summary_span(path: Path, spans: list[StringSpan]) -> tuple[str, StringSpan | None]:
    """Choose the docstring text to preserve as the module summary."""
    for span in spans:
        if not is_license_text(span.text):
            return span.text, span
    for span in spans:
        summary = derive_summary_from_license(span.text)
        if summary:
            return summary, None
    return default_summary(path), None


def punctuate(summary: str) -> str:
    """Ensure the docstring summary ends with sentence punctuation."""
    stripped = summary.strip()
    if not stripped:
        return "Provide module functionality."
    if stripped.endswith(SUMMARY_ENDINGS):
        return stripped
    return f"{stripped}."


def normalize_docstring_text(text: str, fallback: str) -> str:
    """Normalize a module docstring body while preserving detail text."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return punctuate(fallback)

    summary = punctuate(lines[0])
    details = lines[1:]
    while details and not details[0].strip():
        details.pop(0)
    while details and not details[-1].strip():
        details.pop()
    if not details:
        return summary
    return "\n".join([summary, "", *details])


def render_docstring(text: str) -> list[str]:
    """Render a normalized docstring as source lines."""
    delimiter = '"""' if '"""' not in text else "'''"
    if "\n" not in text:
        return [f"{delimiter}{text}{delimiter}\n"]
    return [f"{delimiter}{text}\n", f"{delimiter}\n"]


def header_comment_lines(lines: list[str], start: int) -> set[int]:
    """Return 1-based line numbers for existing leading header comments."""
    delete: set[int] = set()
    index = start
    while index < len(lines) and not lines[index].strip():
        delete.add(index + 1)
        index += 1

    inspected = 0
    while index < len(lines) and inspected < 12:
        stripped = lines[index].strip()
        if not stripped:
            delete.add(index + 1)
            index += 1
            inspected += 1
            continue
        if stripped.startswith("# SPDX-License-Identifier:") or stripped == CANONICAL_HEADER:
            delete.add(index + 1)
            index += 1
            inspected += 1
            continue
        if stripped.startswith("# Copyright") and "Steel Security Advisors LLC" in stripped:
            delete.add(index + 1)
            index += 1
            inspected += 1
            continue
        break
    return delete


def remove_deleted_lines(lines: list[str], delete: set[int], preserved_count: int) -> list[str]:
    """Remove marked lines and leading blanks after the preserved prologue."""
    body = [
        line if line.endswith("\n") else f"{line}\n"
        for number, line in enumerate(lines, start=1)
        if number > preserved_count and number not in delete
    ]
    while body and not body[0].strip():
        body.pop(0)
    return body


def normalized_source(path: Path, original: str) -> str:
    """Return normalized source for one Python file."""
    lines = original.splitlines(keepends=True)
    preserved_count = prologue_end(lines)
    spans = collect_top_level_strings(path, original)
    summary_text, summary_span = choose_summary_span(path, spans)
    fallback = default_summary(path)
    docstring = normalize_docstring_text(summary_text, fallback)

    delete = header_comment_lines(lines, preserved_count)
    for span in spans:
        if is_license_text(span.text) or span == summary_span:
            delete.update(range(span.start_line, span.end_line + 1))
            line_after = span.end_line + 1
            while line_after <= len(lines) and not lines[line_after - 1].strip():
                delete.add(line_after)
                line_after += 1

    prefix = [line if line.endswith("\n") else f"{line}\n" for line in lines[:preserved_count]]
    body = remove_deleted_lines(lines, delete, preserved_count)
    normalized = [*prefix, f"{CANONICAL_HEADER}\n", *render_docstring(docstring)]
    if body:
        normalized.append("\n")
        normalized.extend(body)
    return "".join(normalized)


def diff_for(path: Path, original: str, normalized: str) -> str:
    """Return a unified diff for one changed file."""
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            normalized.splitlines(keepends=True),
            fromfile=f"{rel} (current)",
            tofile=f"{rel} (normalized)",
        )
    )


def main() -> int:
    """Run the header normalizer."""
    args = parse_args()
    paths = tuple(args.paths)
    changed: list[Path] = []
    for path in iter_python_files(paths):
        original = path.read_text(encoding="utf-8")
        normalized = normalized_source(path, original)
        if normalized == original:
            continue
        changed.append(path)
        if args.apply:
            path.write_text(normalized, encoding="utf-8")
        else:
            print(diff_for(path, original, normalized))

    if args.check and changed:
        print(f"{len(changed)} Python file(s) need normalized headers.", file=sys.stderr)
        return 1
    if args.apply:
        print(f"Normalized {len(changed)} Python file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
