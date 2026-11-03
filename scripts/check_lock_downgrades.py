#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
r"""Refuse a ``uv.lock`` change that moves any package to an older version.

Dependabot PR #393 was titled as a minor bump of two packages (h5py
3.14.0 -> 3.16.0, nixtla 0.7.2 -> 0.9.0). Its lock also took pandas from
3.1.0 to 2.3.3 and TensorFlow from 2.21.0 to 2.20.0 on every platform,
because nixtla >= 0.8 requires ``pandas<3`` and TensorFlow 2.21 requires
``h5py<3.15``. Every check passed and auto-merge was armed: nothing compared
the versions the lock held before and after.

This gate does, per environment. A universal lock can hold several versions
of one package, each selected by ``resolution-markers`` for some Python
versions and platforms, so comparing a single "highest" version per package
misses a downgrade on the mainstream fork whenever another fork keeps the
higher version. The gate evaluates every package entry's markers on a grid of
concrete environments (each supported Python minor, plus the 3.11.5 split,
on Linux x86_64/aarch64, Windows, macOS arm64/x86_64 and Emscripten) and
reports, for each environment both locks support, any package the new lock
resolves to a lower version. A package that leaves an environment is
reported, not failed (dropping a transitive is normal).

A deliberate downgrade is declared, not silenced: a commit in the pull
request, authored by a person (not a ``[bot]``), carries a git trailer in its
final trailer block, one per package::

    Lock-Downgrade: pandas 3.1.0 -> 2.3.3: <reason>

and the gate accepts exactly that transition, nothing wider. Only real git
trailers count (``git log --format=%(trailers)``), so a line quoted inside a
Dependabot body's release notes, or in a pull request description, declares
nothing.

Usage::

    python scripts/check_lock_downgrades.py --base-ref origin/main
    python scripts/check_lock_downgrades.py --base-ref "$BASE_SHA" \\
        --commit-range "$BASE_SHA..$HEAD_SHA"
    python scripts/check_lock_downgrades.py --base-lock old.lock --head-lock uv.lock
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from packaging.markers import InvalidMarker, Marker, UndefinedEnvironmentName
from packaging.version import InvalidVersion, Version

REPO = Path(__file__).resolve().parents[1]
TRAILER = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9._-]+)\s+(?P<old>\S+)\s*->\s*(?P<new>\S+?)\s*:\s*(?P<why>\S.*)$"
)

_PYTHONS = ("3.11.0", "3.11.5", "3.12.0", "3.13.0", "3.14.0", "3.15.0")
_PLATFORMS = (
    ("linux", "x86_64", "Linux", "posix"),
    ("linux", "aarch64", "Linux", "posix"),
    ("win32", "AMD64", "Windows", "nt"),
    ("darwin", "arm64", "Darwin", "posix"),
    ("darwin", "x86_64", "Darwin", "posix"),
    ("emscripten", "wasm32", "Emscripten", "posix"),
)


def _environments() -> dict[str, dict[str, str]]:
    grid: dict[str, dict[str, str]] = {}
    for full in _PYTHONS:
        for sys_platform, machine, system, os_name in _PLATFORMS:
            grid[f"py{full}-{sys_platform}-{machine}"] = {
                "implementation_name": "cpython",
                "implementation_version": full,
                "os_name": os_name,
                "platform_machine": machine,
                "platform_python_implementation": "CPython",
                "platform_release": "",
                "platform_system": system,
                "platform_version": "",
                "python_full_version": full,
                "python_version": ".".join(full.split(".")[:2]),
                "sys_platform": sys_platform,
                "extra": "",
            }
    return grid


ENVIRONMENTS = _environments()


@dataclass(frozen=True)
class Change:
    """One package whose locked version moved in at least one environment."""

    name: str
    old: str
    new: str | None
    environments: tuple[str, ...] = field(default=(), compare=False)

    @property
    def is_downgrade(self) -> bool:
        return self.new is not None and _version(self.new) < _version(self.old)


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _version(text: str) -> Version:
    try:
        return Version(text)
    except InvalidVersion as exc:  # a lock always records PEP 440 versions
        raise SystemExit(f"unparseable version in lock: {text!r}") from exc


def _applies(markers: list[str] | None, env: dict[str, str]) -> bool:
    if not markers:
        return True
    for text in markers:
        try:
            if Marker(text).evaluate(env):
                return True
        except UndefinedEnvironmentName:
            return True
        except InvalidMarker as exc:
            raise SystemExit(f"unparseable resolution marker in lock: {text!r}") from exc
    return False


def _packages(lock_text: str) -> list[dict[str, Any]]:
    lock = tomllib.loads(lock_text)
    return [p for p in lock.get("package", []) if p.get("version") is not None]


def _supported(lock_text: str) -> set[str]:
    markers = tomllib.loads(lock_text).get("supported-markers")
    return {env_id for env_id, env in ENVIRONMENTS.items() if _applies(markers, env)}


def versions_by_environment(lock_text: str) -> dict[str, dict[str, str]]:
    """``{environment: {package: version}}`` for every environment the lock supports."""
    supported = _supported(lock_text)
    resolved: dict[str, dict[str, str]] = {env_id: {} for env_id in supported}
    for package in _packages(lock_text):
        name, version = _normalise(package["name"]), package["version"]
        markers = package.get("resolution-markers")
        for env_id in supported:
            if not _applies(markers, ENVIRONMENTS[env_id]):
                continue
            current = resolved[env_id].get(name)
            if current is None or _version(version) > _version(current):
                resolved[env_id][name] = version
    return resolved


def highest_versions(lock_text: str) -> dict[str, str]:
    """Map each package in a ``uv.lock`` to the highest version it holds."""
    best: dict[str, str] = {}
    for package in _packages(lock_text):
        name, version = _normalise(package["name"]), package["version"]
        if name not in best or _version(version) > _version(best[name]):
            best[name] = version
    return best


def lowest_versions(lock_text: str) -> dict[str, str]:
    """Map each package to the lowest version it holds.

    Where the lock splits a package across Python or platform forks,
    Dependabot reports and updates the lowest of them as "current".
    """
    low: dict[str, str] = {}
    for package in _packages(lock_text):
        name, version = _normalise(package["name"]), package["version"]
        if name not in low or _version(version) < _version(low[name]):
            low[name] = version
    return low


def compare(base_text: str, head_text: str) -> list[Change]:
    """Every (package, old, new) transition in any environment both locks support."""
    base, head = versions_by_environment(base_text), versions_by_environment(head_text)
    moves: dict[tuple[str, str, str | None], list[str]] = defaultdict(list)
    for env_id in sorted(set(base) & set(head)):
        for name, old in base[env_id].items():
            new = head[env_id].get(name)
            if new != old:
                moves[(name, old, new)].append(env_id)
    return [
        Change(name, old, new, tuple(envs))
        for (name, old, new), envs in sorted(moves.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    ]


def declared(text: str) -> set[tuple[str, str, str]]:
    """``(name, old, new)`` from ``Lock-Downgrade`` trailer values, one per line."""
    found: set[tuple[str, str, str]] = set()
    for line in text.splitlines():
        match = TRAILER.match(line)
        if match:
            found.add((_normalise(match["name"]), match["old"], match["new"]))
    return found


def undeclared_downgrades(base_text: str, head_text: str, declarations: str = "") -> list[Change]:
    """Downgrades with no matching ``Lock-Downgrade`` declaration."""
    allowed = declared(declarations)
    return [
        change
        for change in compare(base_text, head_text)
        if change.is_downgrade and (change.name, change.old, change.new) not in allowed
    ]


def _git(*args: str, cwd: Path = REPO) -> str:
    return subprocess.run(  # noqa: S603 # nosec B603 - fixed argv, read-only git
        ["git", *args],  # noqa: S607 # nosec B607 - git from PATH, as every CI gate
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def trailer_declarations(commit_range: str, cwd: Path = REPO) -> str:
    """``Lock-Downgrade`` trailer values from the non-bot commits in a range."""
    log = _git(
        "log",
        "--format=%an%x1f%(trailers:key=Lock-Downgrade,valueonly,separator=%x1e)%x1d",
        commit_range,
        cwd=cwd,
    )
    values: list[str] = []
    for record in log.split("\x1d"):
        author, _, trailers = record.strip("\n").partition("\x1f")
        if not author or author.endswith("[bot]"):
            continue
        values.extend(v.strip() for v in trailers.split("\x1e") if v.strip())
    return "\n".join(values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-ref", help="git ref whose uv.lock is the baseline")
    source.add_argument("--base-lock", type=Path, help="path to the baseline uv.lock")
    parser.add_argument("--head-lock", type=Path, default=REPO / "uv.lock")
    parser.add_argument(
        "--commit-range",
        help="commits whose Lock-Downgrade trailers count (default with --base-ref: BASE..HEAD)",
    )
    args = parser.parse_args(argv)

    if args.base_ref:
        base_text = _git("show", f"{args.base_ref}:uv.lock")
        commit_range = args.commit_range or f"{args.base_ref}..HEAD"
    else:
        base_text = args.base_lock.read_text(encoding="utf-8")
        commit_range = args.commit_range
    declarations = trailer_declarations(commit_range) if commit_range else ""
    head_text = args.head_lock.read_text(encoding="utf-8")

    dropped = sorted(_supported(base_text) - _supported(head_text))
    if dropped:
        print(f"environments no longer locked: {', '.join(dropped)}")
    allowed = declared(declarations)
    failures = 0
    for change in compare(base_text, head_text):
        where = f" [{len(change.environments)} environment(s), e.g. {change.environments[0]}]"
        if change.new is None:
            print(f"removed     {change.name} {change.old}{where}")
        elif not change.is_downgrade:
            print(f"upgraded    {change.name} {change.old} -> {change.new}{where}")
        elif (change.name, change.old, change.new) in allowed:
            print(f"declared    {change.name} {change.old} -> {change.new}{where}")
        else:
            failures += 1
            print(
                f"::error title=uv.lock downgrade::{change.name} {change.old} -> {change.new}"
                f"{where} is a downgrade with no 'Lock-Downgrade: {change.name} {change.old}"
                f" -> {change.new}: <reason>' trailer on a commit in this pull request"
            )
    print(f"{failures} undeclared downgrade(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
