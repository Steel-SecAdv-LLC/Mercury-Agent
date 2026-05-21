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

# CVE-2026-6357 (pip arbitrary code execution via malicious wheel).
# Every ``pip install`` step in every workflow file MUST first floor pip
# to >=26.1 so a runner image that ships an older pip cannot install a
# poisoned wheel.  We detect any ``pip install`` invocation (or
# ``python -m pip install``) that is *not* preceded by a ``pip>=26.1``
# upgrade in the same shell block.  An exemption is provided for the
# explicit upgrade line itself (which is the cure, not the wound) and
# for example/docstring-style strings inside ``run:`` blocks (those are
# detected because they appear inside quotes/printable strings rather
# than as the leading executable token of a line).
# Matches a ``pip install`` invocation appearing anywhere on a line —
# both as the leading executable token of a ``run: |`` block line and
# as the trailing payload of a single-line ``run: pip install ...``
# step.  ``[^#\n]*`` before the pattern allows leading shell tokens
# like ``set -e &&`` while excluding everything past a ``#`` comment
# (so ``# pip install x`` in a comment is not a false positive).
PIP_INSTALL_RE = re.compile(r"(?<![\w-])(?:python\s+-m\s+)?pip\s+install\b")
PIP_UPGRADE_RE = re.compile(
    r"(?:python\s+-m\s+)?pip\s+install\b[^\n]*?--upgrade\b[^\n]*['\"]pip>=26(?:\.\d+)*['\"]"
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

    errors.extend(_check_pip_cve_2026_6357(path, text))

    for warning in warnings:
        print(f"::warning title=Workflow supply-chain hardening::{warning}")
    return errors


# Matches a ``cat <<EOF`` / ``cat <<-'EOF'`` style heredoc opener.  The
# delimiter token (capture group ``delim``) is the closing marker we
# look for at the start of a later line to know the heredoc body ended.
# Surrounding quotes around the delimiter (``<<'EOF'`` / ``<<"EOF"``)
# are stripped from ``delim`` so the closing comparison works against
# the bare token POSIX shells emit.
HEREDOC_OPEN_RE = re.compile(
    r"<<-?\s*(?P<quote>['\"])?(?P<delim>[A-Za-z_][A-Za-z0-9_]*)(?P=quote)?\s*$"
)


def _check_pip_cve_2026_6357(path: Path, text: str) -> list[str]:
    """Every ``pip install`` step must floor pip to >=26.1 (CVE-2026-6357).

    The runner's Python ``site-packages`` is shared across every step of
    a single job, so the floor only needs to be applied *once per job*
    before any other ``pip install`` step.  We walk each ``- name:``
    step boundary and require that the first ``pip install`` line in
    any step is either (a) the upgrade itself, or (b) preceded by an
    upgrade step earlier in the same job.

    Returning a non-empty list fails the workflow-hardening gate.
    """
    errors: list[str] = []
    lines = text.splitlines()

    # Track the job each line belongs to via the ``  job_id:`` indent.
    # A new job resets the ``pip_floored`` state; a new step inside a
    # job inherits the existing state.
    #
    # ``in_jobs_block`` guards the job-boundary heuristic from
    # misfiring on unrelated indent-2 keys (e.g. ``push:`` under
    # ``on:``, ``contents:`` under ``permissions:``).  Only mappings
    # that are *immediate children of the top-level ``jobs:`` key*
    # qualify as job ids; everything else leaves state untouched.
    pip_floored = False
    current_job_key: str | None = None
    job_start_line = 0
    in_jobs_block = False
    heredoc_delim: str | None = None
    for lineno, raw in enumerate(lines, start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        # If we are currently inside a heredoc body, the line is shell
        # *input data*, not an executable command — skip it entirely
        # for install-detection purposes, and look only for the
        # closing delimiter so we can leave the heredoc state.
        if heredoc_delim is not None:
            if raw.strip() == heredoc_delim:
                heredoc_delim = None
            continue
        # Top-level block boundaries: every indent-0 key resets
        # ``in_jobs_block``; only ``jobs:`` flips it on.
        match = MAPPING_KEY_RE.match(raw)
        if match and len(match.group("indent")) == 0:
            in_jobs_block = normalize_key(match.group("key")) == "jobs"
            # An indent-0 key is by definition not a job id, so clear
            # any in-flight per-job tracking.
            pip_floored = False
            current_job_key = None
            continue
        # Job boundary: indent-2 mapping key, but only when we are
        # already inside the ``jobs:`` block.
        if (
            match
            and len(match.group("indent")) == 2
            and in_jobs_block
            and normalize_key(match.group("key")) not in {"jobs"}
        ):
            pip_floored = False
            current_job_key = normalize_key(match.group("key"))
            job_start_line = lineno
            continue
        # Strip an inline comment so a ``# pip install`` substring in
        # a trailing comment is never inspected.
        scan = raw.split("#", 1)[0]
        # If this line *opens* a heredoc (``cat <<EOF`` /
        # ``cat <<-'EOF'``), remember the delimiter so the next lines
        # are treated as data, not commands.  The opener line itself
        # is not a ``pip install`` invocation by construction (it ends
        # in ``<<DELIM``), so falling through to the install check is
        # safe but unnecessary.
        heredoc_match = HEREDOC_OPEN_RE.search(scan)
        if heredoc_match:
            heredoc_delim = heredoc_match.group("delim")
            continue
        if PIP_UPGRADE_RE.search(scan):
            pip_floored = True
            continue
        # Documentation-emission lines that *write* the string
        # ``pip install ...`` into a file (typical patterns:
        # ``echo "pip install ..." >> CHANGELOG``,
        # ``printf 'pip install ...\n'``) are not actual installs
        # and must not trip the guard.  Detect them by an
        # ``echo``/``printf`` leading token only.  A bare ``>>`` /
        # ``<<`` substring is **not** a safe exemption: a real
        # invocation like ``pip install requests >> install.log``
        # genuinely installs ``requests`` and must still fail the
        # gate, and a heredoc that *contains* a ``pip install`` line
        # is already handled by the ``heredoc_delim`` state above.
        leading_token = scan.lstrip().split(" ", 1)[0].strip()
        if leading_token in {"echo", "printf"}:
            continue
        if PIP_INSTALL_RE.search(scan) and not pip_floored:
            job_hint = (
                f" (in job ``{current_job_key}`` starting at line {job_start_line})"
                if current_job_key
                else ""
            )
            errors.append(
                f"{path}:{lineno}: ``pip install`` without prior "
                f"``pip>=26.1`` upgrade earlier in the same job{job_hint} "
                "(CVE-2026-6357 regression guard)"
            )
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
