# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
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
COPYRIGHT_HEADER = "# Copyright (C) 2025 Steel Security Advisors LLC"
SPDX_HEADER = "# SPDX-License-Identifier: GPL-3.0-or-later"
CANONICAL_HEADERS = (COPYRIGHT_HEADER, SPDX_HEADER)
CODING_RE = re.compile(r"^#.*coding[:=]\s*[-\w.]+")
SEPARATOR_RE = re.compile(r"^\s*[-=]{6,}\s*$")
LICENSE_MARKERS = (
    "This program is free software",
    "redistribute it",
    "GNU General Public License",
    "WITHOUT ANY WARRANTY",
    "MERCHANTABILITY",
    "FITNESS FOR A PARTICULAR PURPOSE",
    "implied warranty",
    "Free Software Foundation",
    "any later version",
    "You should have received a copy",
    "SPDX-License-Identifier",
    "Copyright (C) 2025 Steel Security Advisors LLC",
    "Copyright (C) Steel Security Advisors LLC",
)
FORBIDDEN_DOCSTRING_PHRASES = (
    "any later version",
    "WITHOUT ANY WARRANTY",
    "MERCHANTABILITY",
    "FITNESS FOR A PARTICULAR PURPOSE",
    "redistribute it",
    "Free Software Foundation",
    "GNU General Public License",
    "implied warranty",
)
SKIP_LINE_FRAGMENTS = (
    "Copyright",
    "License",
    "GPL",
    "https://www.gnu.org",
    "https://gnu.org",
    "https://github.com/Steel-SecAdv-LLC",
    "Free Software Foundation",
    "any later version",
    "WITHOUT ANY WARRANTY",
    "MERCHANTABILITY",
    "FITNESS FOR A PARTICULAR PURPOSE",
    "implied warranty",
    "redistribute it",
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
            segment = ast.get_source_segment(text, node)
            raw_text = string_literal_body(segment) if segment is not None else node.value.value
            spans.append(StringSpan(raw_text, node.lineno, node.end_lineno))
    return spans


def string_literal_body(segment: str) -> str:
    """Return the raw body of a triple-quoted string source segment."""
    stripped = segment.strip()
    quote_positions = [
        (index, quote) for quote in ('"""', "'''") if (index := stripped.find(quote)) >= 0
    ]
    if not quote_positions:
        return ast.literal_eval(stripped)
    start, quote = min(quote_positions, key=lambda item: item[0])
    end = stripped.rfind(quote)
    if end <= start:
        return ast.literal_eval(stripped)
    return stripped[start + len(quote) : end]


def is_license_text(text: str) -> bool:
    """Return whether a string expression is a duplicated license notice."""
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in LICENSE_MARKERS)


def contains_forbidden_docstring_phrase(text: str) -> bool:
    """Return whether module documentation still contains GPL boilerplate."""
    lowered = text.lower()
    return any(phrase.lower() in lowered for phrase in FORBIDDEN_DOCSTRING_PHRASES)


def _is_boilerplate_line(line: str) -> bool:
    """Return True if the line is part of license/copyright boilerplate."""
    stripped = line.strip()
    if stripped in {"Mercury Agent", "see"} or stripped.startswith("Mercury Agent Copyright"):
        return True
    lowered = line.lower()
    return any(frag.lower() in lowered for frag in SKIP_LINE_FRAGMENTS) or line.startswith(
        ("This program", "You should", "See ", "see ")
    )


def trim_blank_edges(lines: list[str]) -> list[str]:
    """Trim only leading and trailing blank lines."""
    trimmed = list(lines)
    while trimmed and not trimmed[0].strip():
        trimmed.pop(0)
    while trimmed and not trimmed[-1].strip():
        trimmed.pop()
    return trimmed


def description_after_separator(text: str) -> str | None:
    """Return the genuine module documentation after a legacy separator."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for index, line in enumerate(lines):
        if not SEPARATOR_RE.match(line):
            continue
        previous = next((prior.strip() for prior in reversed(lines[:index]) if prior.strip()), "")
        if previous and not (
            _is_boilerplate_line(previous) or contains_forbidden_docstring_phrase(previous)
        ):
            continue
        description = trim_blank_edges(lines[index + 1 :])
        if not description:
            return None
        candidate = "\n".join(line.rstrip() for line in description)
        if contains_forbidden_docstring_phrase(candidate):
            return None
        return candidate
    return None


def _blocks(lines: list[str]) -> list[list[str]]:
    """Split docstring lines into blank-line-delimited blocks."""
    result: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.strip():
            current.append(line)
            continue
        if current:
            result.append(current)
            current = []
    if current:
        result.append(current)
    return result


def _is_gpl_block(block: list[str]) -> bool:
    """Return whether a paragraph is GPL boilerplate that should be dropped whole."""
    text = " ".join(line.strip() for line in block).lower()
    return any(
        marker in text
        for marker in (
            "this program is free software",
            "terms of the gnu general public license",
            "gnu general public license",
            "free software foundation",
            "without any warranty",
            "implied warranty",
            "merchantability",
            "fitness for a particular purpose",
            "you should have received a copy",
            "www.gnu.org/licenses",
            "at your option) any later version",
        )
    )


def derive_docstring_from_license(text: str) -> str | None:
    """Extract non-license documentation from a legacy boilerplate string."""
    separated = description_after_separator(text)
    if separated:
        return separated

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    kept_blocks: list[list[str]] = []
    for block in _blocks(lines):
        if _is_gpl_block(block):
            continue
        kept_lines: list[str] = []
        for raw_line in block:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("Mercury Agent - "):
                kept_line = line.removeprefix("Mercury Agent - ").strip()
            else:
                kept_line = raw_line.rstrip()
            if _is_boilerplate_line(line) or contains_forbidden_docstring_phrase(line):
                continue
            kept_lines.append(kept_line)
        if kept_lines:
            kept_blocks.append(kept_lines)
    if not kept_blocks:
        return None
    docstring = "\n\n".join("\n".join(line.rstrip() for line in block) for block in kept_blocks)
    if contains_forbidden_docstring_phrase(docstring):
        return None
    return docstring


def real_docstring_from_spans(path: Path, spans: list[StringSpan]) -> str:
    """Return the real module documentation from a file's top-level string nodes."""
    for span in spans:
        if is_license_text(span.text) or contains_forbidden_docstring_phrase(span.text):
            docstring = derive_docstring_from_license(span.text)
            if docstring:
                return docstring
            continue
        return span.text
    return default_summary(path)


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
    """Choose the module documentation to preserve."""
    for span in spans:
        if is_license_text(span.text) or contains_forbidden_docstring_phrase(span.text):
            docstring = derive_docstring_from_license(span.text)
            if docstring:
                return docstring, span
            continue
        else:
            return span.text, span
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
    lines = trim_blank_edges(lines)
    if not lines:
        return punctuate(fallback)
    first_blank = next((index for index, line in enumerate(lines) if not line.strip()), len(lines))
    summary_lines = lines[:first_blank]
    details = trim_blank_edges(lines[first_blank + 1 :]) if first_blank < len(lines) else []
    summary = " ".join(line.strip() for line in summary_lines if line.strip())
    if not summary:
        summary = fallback
    summary = punctuate(summary)
    docstring = summary if not details else "\n".join([summary, "", *details])
    if contains_forbidden_docstring_phrase(docstring):
        return punctuate(fallback)
    return docstring


def render_docstring(text: str) -> list[str]:
    """Render a normalized docstring as source lines."""
    delimiter = '"""' if '"""' not in text else "'''"
    prefix = "r" if "\\" in text else ""
    if "\n" not in text:
        return [f"{prefix}{delimiter}{text}{delimiter}\n"]
    return [f"{prefix}{delimiter}{text}\n", f"{delimiter}\n"]


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
        if stripped.startswith("# SPDX-License-Identifier:") or stripped in CANONICAL_HEADERS:
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
    normalized = [
        *prefix,
        *(f"{header}\n" for header in CANONICAL_HEADERS),
        *render_docstring(docstring),
    ]
    if body:
        normalized.append("\n")
        normalized.extend(body)
    return "".join(normalized)


def validation_errors(path: Path, source: str) -> list[str]:
    """Return semantic header/docstring validation errors."""
    lines = source.splitlines()
    start = prologue_end([f"{line}\n" for line in lines])
    expected = list(CANONICAL_HEADERS)
    actual = lines[start : start + len(expected)]
    errors: list[str] = []
    if actual != expected:
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        errors.append(f"{rel}: missing canonical copyright/SPDX header pair")

    try:
        module = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        errors.append(f"{path}: cannot parse normalized source: {exc}")
        return errors

    docstring = ast.get_docstring(module, clean=False)
    if not docstring:
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        errors.append(f"{rel}: missing module docstring")
    elif contains_forbidden_docstring_phrase(docstring):
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        errors.append(f"{rel}: module docstring contains GPL boilerplate phrase")
    return errors


def module_docstring(path: Path, source: str) -> str:
    """Return the normalized real module documentation for a source string."""
    spans = collect_top_level_strings(path, source)
    return normalize_docstring_text(real_docstring_from_spans(path, spans), default_summary(path))


def squashed(text: str) -> str:
    """Collapse whitespace for docstring-retention length comparisons."""
    return " ".join(text.split())


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
    errors: list[str] = []
    for path in iter_python_files(paths):
        original = path.read_text(encoding="utf-8")
        normalized = normalized_source(path, original)
        errors.extend(validation_errors(path, normalized if args.apply else original))
        if normalized == original:
            continue
        changed.append(path)
        if args.apply:
            path.write_text(normalized, encoding="utf-8")
        else:
            print(diff_for(path, original, normalized))

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    if args.check and changed:
        print(f"{len(changed)} Python file(s) need normalized headers.", file=sys.stderr)
        return 1
    if args.apply:
        print(f"Normalized {len(changed)} Python file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
