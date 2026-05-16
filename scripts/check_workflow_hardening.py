#!/usr/bin/env python3
"""Repository-specific GitHub Actions hardening checks."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
WRITE_OK = {
    "benchmark.yml",
    "dependabot-auto-merge.yml",
    "release.yml",
}
SHA_REF_RE = re.compile(r"^[0-9a-fA-F]{40}$")
USES_RE = re.compile(r"^\s*uses:\s*([^@\s]+)@([^#\s]+)", re.MULTILINE)
MAPPING_KEY_RE = re.compile(
    r"^(?P<indent>\s*)(?P<key>[A-Za-z_][A-Za-z0-9_-]*|\"[^\"]+\"|'[^']+'):\s*(?P<value>.*)$"
)


def top_level_indent(text: str) -> int:
    indents = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = MAPPING_KEY_RE.match(line)
        if match:
            indents.append(len(match.group("indent")))
    return min(indents, default=0)


def normalize_key(key: str) -> str:
    return key.strip("'\"")


def iter_top_level_keys(text: str) -> list[tuple[str, int, str, int]]:
    document_indent = top_level_indent(text)
    keys = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        match = MAPPING_KEY_RE.match(line)
        if match and len(match.group("indent")) == document_indent:
            keys.append(
                (
                    normalize_key(match.group("key")),
                    len(match.group("indent")),
                    match.group("value").strip(),
                    lineno,
                )
            )
    return keys


def has_top_level_key(text: str, key: str) -> bool:
    return any(name == key for name, _, _, _ in iter_top_level_keys(text))


def iter_permissions_blocks(text: str) -> list[tuple[int, str, int]]:
    """Return every ``permissions:`` mapping in the document, top-level or per-job.

    Yields ``(indent, value, lineno)`` for each occurrence.  Job-level
    permissions blocks are inspected the same way as the top-level one
    so that a job cannot quietly grant ``contents: write`` while the
    workflow file is not in ``WRITE_OK``.
    """
    blocks: list[tuple[int, str, int]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = MAPPING_KEY_RE.match(line)
        if match and normalize_key(match.group("key")) == "permissions":
            blocks.append((len(match.group("indent")), match.group("value").strip(), lineno))
    return blocks


def has_disallowed_contents_write(text: str) -> bool:
    lines = text.splitlines()
    for indent, value, lineno in iter_permissions_blocks(text):
        if value == "write-all":
            return True
        for line in lines[lineno:]:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            line_indent = len(line) - len(line.lstrip())
            if line_indent <= indent:
                break
            match = MAPPING_KEY_RE.match(line)
            if match and normalize_key(match.group("key")) == "contents":
                # Strip an inline comment, then quotes, so that
                # ``contents: "write"`` / ``contents: 'write'`` / bare
                # ``contents: write`` are all detected (YAML treats them
                # as equivalent — quoting must not be a bypass).
                contents_value = match.group("value").strip().split("#", 1)[0].strip().strip("'\"")
                if contents_value == "write":
                    return True
    return False


def has_pull_request_target(text: str) -> bool:
    """True iff ``pull_request_target`` appears as an ``on:`` event key.

    Inspecting the ``on:`` mapping (rather than a raw substring search
    over the whole file) avoids false positives from YAML comments and
    ``run:`` block scalars that merely mention the trigger by name.
    """
    lines = text.splitlines()
    for name, indent, value, lineno in iter_top_level_keys(text):
        if name != "on":
            continue
        # Inline list form: ``on: [push, pull_request_target]``
        stripped_value = value.split("#", 1)[0].strip()
        if stripped_value.startswith("["):
            inline = stripped_value.strip("[]")
            tokens = [t.strip().strip("'\"") for t in inline.split(",")]
            if "pull_request_target" in tokens:
                return True
            continue
        # Mapping form: nested keys under ``on:``
        for line in lines[lineno:]:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            line_indent = len(line) - len(line.lstrip())
            if line_indent <= indent:
                break
            child = MAPPING_KEY_RE.match(line)
            if child and normalize_key(child.group("key")) == "pull_request_target":
                return True
    return False


def check_workflow(path: Path) -> list[str]:
    errors: list[str] = []
    warnings: list[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")

    if has_pull_request_target(text):
        errors.append(f"{path}: pull_request_target is not allowed without a security review")

    if not has_top_level_key(text, "permissions"):
        errors.append(f"{path}: add top-level least-privilege permissions")
    elif path.name not in WRITE_OK and has_disallowed_contents_write(text):
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
    workflows = sorted([*WORKFLOW_DIR.glob("*.yml"), *WORKFLOW_DIR.glob("*.yaml")])
    if not workflows:
        # Fail loud: silently passing when ``WORKFLOW_DIR`` resolves to an
        # empty directory (e.g. script invoked from outside the repo root
        # with a broken path) would mean the gate is no-op.  WORKFLOW_DIR
        # is now resolved relative to the script location so this should
        # only trigger if the workflows directory was actually deleted.
        print(
            f"Workflow hardening check failed: no workflow files found in {WORKFLOW_DIR}",
            file=sys.stderr,
        )
        return 1
    for path in workflows:
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
