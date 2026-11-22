#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fail on any package install that is not pinned to an exact, verified artifact.

OpenSSF Scorecard's Pinned-Dependencies check filed one Medium alert per
unpinned install: ``pip install pkg>=x``, ``pip install -r file`` without
hashes, ``pip install .``, ``go install tool@vX`` and so on, across every
workflow, composite action, shell script and Dockerfile. Every such install now
draws from a hash-locked source -- ``uv.lock`` exported with ``uv export``, the
``requirements/ci/*.txt`` files compiled by ``scripts/lock_ci_requirements.sh``,
or a wheel hashed by ``scripts/ama_wheel_requirement.py`` -- and this gate
keeps it that way.

An install command passes when it is one of:

* ``pip install --require-hashes ...`` (also ``python -m pip`` and
  ``uv pip``). ``--require-hashes`` must come directly after ``install``: a
  scanner that reads arguments in order would otherwise take the value of an
  earlier option (``--find-links DIR``, ``--python PY``) for a package name.
  ``uv pip`` also needs ``--no-config``: inside the checkout uv applies
  pyproject's ``constraint-dependencies`` ranges, which hash mode rejects;
* ``pip install --no-deps -e <local path>`` -- the project's own checkout,
  with nothing resolved from an index;
* ``pip install --no-deps <file>.whl`` -- a wheel already on disk;
* ``npm ci``; ``go install <module>@<40-hex commit>``.

Anything else that installs (``npm install``, ``curl ... | sh``, a range, a
bare name, a requirements file without hashes, a directory) fails. The
``requirements/ci/*.txt`` files are checked too: every requirement is ``==``
pinned with at least one ``--hash``, every ``.in`` has its ``.txt``, and
``pip.txt`` keeps pip at or above 26.1 (CVE-2026-6357).

Exit status is 1 with one line per finding, 0 when clean.
"""

from __future__ import annotations

import re
import shlex
import sys
from itertools import pairwise
from pathlib import Path
from typing import NamedTuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_DIR = Path("requirements") / "ci"

_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_PYTHON_RE = re.compile(r"^python(?:\d+(?:\.\d+)?)?$")
_PIP_RE = re.compile(r"^pip(?:\d+(?:\.\d+)?)?$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
_SEPARATORS = frozenset({";", "&&", "||", "|", "&", "(", ")", "{", "}", "!"})
_KEYWORDS = frozenset({"if", "then", "else", "elif", "do", "while", "until", "time"})
_WRAPPERS = frozenset({"env", "sudo", "exec", "command", "nice", "nohup"})
_SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh"})
# Options whose value is the next token (pip and uv pip, the ones in use).
_VALUE_OPTIONS = frozenset(
    {
        "-c",
        "--constraint",
        "-f",
        "--find-links",
        "-i",
        "--index-url",
        "--extra-index-url",
        "--python",
        "-p",
        "--target",
        "-t",
        "--prefix",
        "--root",
        "--platform",
        "--python-version",
        "--implementation",
        "--abi",
        "--upgrade-strategy",
        "--cache-dir",
        "--log",
        "--timeout",
        "--retries",
        "--src",
        "--config-settings",
        "-C",
        "--progress-bar",
    }
)
_REQUIREMENT_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==(?P<version>[^\s;\\]+)"
    r"(?:\s*;[^-]*)?(?P<hashes>(?:\s+--hash=sha256:[0-9a-f]{64})+)\s*$"
)


class Finding(NamedTuple):
    """One unpinned install: where it is and why it fails.

    A ``NamedTuple`` rather than a ``@dataclass``: tests load this script with
    ``importlib.util.spec_from_file_location`` without registering it in
    ``sys.modules``, which ``dataclasses`` needs to resolve annotations.
    """

    path: str
    line: int
    reason: str
    command: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.reason}: {self.command}"


# --------------------------------------------------------------------------
# Where shell lives
# --------------------------------------------------------------------------


def _run_blocks(node: yaml.Node) -> list[tuple[int, str]]:
    """Every ``run:`` value in a workflow/action document, with its 1-based line."""
    found: list[tuple[int, str]] = []
    if isinstance(node, yaml.MappingNode):
        for key, value in node.value:
            if (
                isinstance(key, yaml.ScalarNode)
                and key.value == "run"
                and isinstance(value, yaml.ScalarNode)
            ):
                # A block scalar (``run: |``) starts on the line after the key.
                first = value.start_mark.line + (2 if value.style in ("|", ">") else 1)
                found.append((first, value.value))
            else:
                found.extend(_run_blocks(value))
    elif isinstance(node, yaml.SequenceNode):
        for item in node.value:
            found.extend(_run_blocks(item))
    return found


def _dockerfile_runs(text: str) -> list[tuple[int, str]]:
    """Every ``RUN`` instruction's shell, continuation lines joined."""
    found: list[tuple[int, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        match = re.match(r"\s*RUN\s+(.*)$", lines[i], re.IGNORECASE)
        if not match:
            i += 1
            continue
        start, body = i + 1, [match.group(1)]
        while body[-1].rstrip().endswith("\\") and i + 1 < len(lines):
            i += 1
            body.append(lines[i])
        found.append((start, "\n".join(body)))
        i += 1
    return found


def shell_sources(root: Path) -> list[tuple[str, int, str]]:
    """``(relative path, first line, shell text)`` for every shell the repo runs."""
    sources: list[tuple[str, int, str]] = []
    yaml_files = sorted((root / ".github" / "workflows").glob("*.y*ml")) + sorted(
        (root / ".github" / "actions").rglob("action.y*ml")
    )
    for path in yaml_files:
        rel = path.relative_to(root).as_posix()
        document = yaml.compose(path.read_text(encoding="utf-8"))
        if document is not None:
            sources.extend((rel, line, text) for line, text in _run_blocks(document))
    skip = {".git", ".venv", "venv", "node_modules", "build", "dist", ".tox"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or skip.intersection(path.relative_to(root).parts):
            continue
        rel = path.relative_to(root).as_posix()
        if path.name.startswith("Dockerfile"):
            text = path.read_text(encoding="utf-8", errors="replace")
            sources.extend((rel, line, body) for line, body in _dockerfile_runs(text))
        elif path.suffix in {".sh", ".bash"}:
            sources.append((rel, 1, path.read_text(encoding="utf-8", errors="replace")))
    return sources


# --------------------------------------------------------------------------
# Shell into simple commands
# --------------------------------------------------------------------------


def logical_lines(text: str) -> list[tuple[int, str]]:
    """Lines with continuations joined and heredoc bodies (data, not commands) dropped."""
    out: list[tuple[int, str]] = []
    lines = text.splitlines()
    i = 0
    heredoc: str | None = None
    while i < len(lines):
        raw = lines[i]
        if heredoc is not None:
            if raw.strip() == heredoc:
                heredoc = None
            i += 1
            continue
        start, parts = i, [raw]
        while parts[-1].rstrip().endswith("\\") and i + 1 < len(lines):
            parts[-1] = parts[-1].rstrip()[:-1]
            i += 1
            parts.append(lines[i])
        line = " ".join(parts)
        opener = _HEREDOC_RE.search(line)
        if opener:
            heredoc = opener.group(2)
        out.append((start, line))
        i += 1
    return out


def simple_commands(line: str) -> list[list[str]]:
    """The simple commands of one logical line, wrappers and assignments removed."""
    lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|(){}!")
    lexer.whitespace_split = True
    lexer.commenters = "#"
    try:
        tokens = list(lexer)
    except ValueError:
        tokens = line.split()
    commands: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in _SEPARATORS or set(token) <= set(";&|(){}!"):
            if current:
                commands.append(current)
            current = []
            continue
        current.append(token)
    if current:
        commands.append(current)
    cleaned: list[list[str]] = []
    for command in commands:
        while command and (
            _ASSIGNMENT_RE.match(command[0]) or command[0] in _WRAPPERS or command[0] in _KEYWORDS
        ):
            command = command[1:]
        if command[:1] == ["timeout"]:
            command = command[2:]
        if command:
            cleaned.append(command)
    return cleaned


# --------------------------------------------------------------------------
# The rules
# --------------------------------------------------------------------------


def _is_local_path(target: str) -> bool:
    if "://" in target or target.startswith("git+") or "[" in target:
        return False
    return target in {".", "./"} or target.startswith(("./", "../", "/", "${", "$"))


def pip_install_arguments(command: list[str]) -> list[str] | None:
    """The arguments after ``install`` when ``command`` runs a pip installer, else None."""
    executable = command[0].rsplit("/", 1)[-1]
    args = command[1:]
    if args[:3] == ["-m", "pip", "install"]:
        return args[3:]
    if _PIP_RE.match(executable) and args[:1] == ["install"]:
        return args[1:]
    if executable == "uv" and args[:2] == ["pip", "install"]:
        return args[2:]
    return None


def pip_verdict(args: list[str]) -> str | None:
    """Why a pip install is unpinned, or None when it is pinned."""
    if args[:1] == ["--require-hashes"]:
        return None
    if "--require-hashes" in args:
        return "--require-hashes must come directly after `install`"
    editable: list[str] = []
    positional: list[str] = []
    requirement_files = False
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in {"-e", "--editable"}:
            if i + 1 < len(args):
                editable.append(args[i + 1])
            i += 2
            continue
        if arg in {"-r", "--requirement"}:
            requirement_files = True
            i += 2
            continue
        if arg in _VALUE_OPTIONS:
            i += 2
            continue
        if not arg.startswith("-"):
            positional.append(arg)
        i += 1
    no_deps = "--no-deps" in args
    if requirement_files:
        return "requirements file installed without --require-hashes"
    if editable and not positional:
        if no_deps and all(_is_local_path(target) for target in editable):
            return None
        return "editable install needs --no-deps and a local path"
    if positional and all(p.endswith(".whl") for p in positional):
        return None if no_deps else "wheel install without --no-deps resolves its dependencies"
    if not positional:
        return None if not editable else "editable install needs --no-deps"
    return "not hash-pinned (use --require-hashes with a locked file)"


def command_findings(command: list[str]) -> str | None:
    """Why one simple command is an unpinned install, or None."""
    args = pip_install_arguments(command)
    executable = command[0].rsplit("/", 1)[-1]
    if args is not None:
        verdict = pip_verdict(args)
        if verdict is None and executable == "uv" and args[:1] == ["--require-hashes"]:
            # uv applies the project's [tool.uv] constraint-dependencies (ranges)
            # to ``uv pip install`` run inside the checkout, and hash mode rejects
            # any requirement that is not ``==``: the install fails outright.
            if "--no-config" not in args:
                return "`uv pip install --require-hashes` needs --no-config"
        return verdict
    rest = command[1:]
    if executable in {"npm", "pnpm", "yarn"} and rest[:1] and rest[0] in {"install", "i", "add"}:
        return f"`{executable} {rest[0]}` resolves ranges; use `npm ci` against the lockfile"
    if executable == "go" and rest[:1] and rest[0] in {"install", "get"}:
        modules = [arg for arg in rest[1:] if not arg.startswith("-")]
        for module in modules:
            version = module.rsplit("@", 1)[1] if "@" in module else ""
            if not _COMMIT_RE.match(version):
                return "go module not pinned to a 40-hex commit"
    return None


def line_findings(line: str) -> list[tuple[str, str]]:
    """``(reason, command text)`` for every unpinned install on one logical line."""
    found: list[tuple[str, str]] = []
    commands = simple_commands(line)
    for command in commands:
        reason = command_findings(command)
        if reason:
            found.append((reason, " ".join(command)))
    # Download piped straight into an interpreter.
    segments = [segment.strip() for segment in re.split(r"(?<!\|)\|(?!\|)", line)]
    for left, right in pairwise(segments):
        right_commands = simple_commands(right)
        left_commands = simple_commands(left)
        if not right_commands or not left_commands:
            continue
        downloads = left_commands[-1][0].rsplit("/", 1)[-1] in {"curl", "wget"}
        runs = right_commands[0][0].rsplit("/", 1)[-1]
        if downloads and (runs in _SHELLS or _PYTHON_RE.match(runs)):
            found.append(("download piped into an interpreter", f"{left} | {right}"))
    return found


def check_shell(path: str, first_line: int, text: str) -> list[Finding]:
    """Findings for one block of shell."""
    findings: list[Finding] = []
    for offset, line in logical_lines(text):
        for reason, command in line_findings(line):
            findings.append(Finding(path, first_line + offset, reason, command))
    return findings


def check_requirements(root: Path) -> list[Finding]:
    """Every ``requirements/ci`` file is fully ``==``-pinned with hashes."""
    findings: list[Finding] = []
    directory = root / REQUIREMENTS_DIR
    inputs = {p.stem for p in directory.glob("*.in")}
    outputs = {p.stem for p in directory.glob("*.txt")}
    for stem in sorted(inputs ^ outputs):
        rel = (REQUIREMENTS_DIR / stem).as_posix()
        findings.append(Finding(rel, 1, "needs both .in and compiled .txt", stem))
    for path in sorted(directory.glob("*.txt")):
        rel = path.relative_to(root).as_posix()
        for lineno, line in logical_lines(path.read_text(encoding="utf-8")):
            stripped = line.split(" #", 1)[0].strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith(("--index-url", "--extra-index-url")):
                continue
            match = _REQUIREMENT_RE.match(stripped)
            if not match:
                findings.append(
                    Finding(rel, lineno + 1, "requirement not ==-pinned with --hash", stripped)
                )
                continue
            if path.stem == "pip" and match.group("name").lower() == "pip":
                release = tuple(int(x) for x in re.findall(r"\d+", match.group("version"))[:2])
                if release < (26, 1):
                    findings.append(
                        Finding(rel, lineno + 1, "pip below 26.1 (CVE-2026-6357)", stripped)
                    )
    pip_file = directory / "pip.txt"
    if pip_file.exists() and not re.search(r"^pip==", pip_file.read_text(), re.MULTILINE):
        findings.append(Finding(pip_file.relative_to(root).as_posix(), 1, "pip not pinned", ""))
    return findings


def check_tree(root: Path = ROOT) -> list[Finding]:
    """Every finding in the repository."""
    findings: list[Finding] = []
    for path, first_line, text in shell_sources(root):
        findings.extend(check_shell(path, first_line, text))
    findings.extend(check_requirements(root))
    return findings


def main(argv: list[str] | None = None) -> int:
    """Print the findings; exit 1 when there are any."""
    root = Path(argv[0]).resolve() if argv else ROOT
    findings = check_tree(root)
    for finding in findings:
        print(finding)
    if findings:
        print(
            f"{len(findings)} unpinned install(s). Install from uv.lock "
            "(uv export ... | pip install --require-hashes) or a hash-locked "
            "requirements/ci file (bash scripts/lock_ci_requirements.sh).",
            file=sys.stderr,
        )
        return 1
    print("check_pinned_installs: every install is pinned to a verified artifact.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
