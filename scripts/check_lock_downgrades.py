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

This gate does. For each package it compares the highest version the base
lock holds with the highest version the new lock holds; a lower one is a
downgrade and fails the check. A package that leaves the lock is reported,
not failed (dropping a transitive is normal).

A deliberate downgrade is declared, not silenced: a commit in the pull
request (or the pull request body) carries one trailer per package::

    Lock-Downgrade: pandas 3.1.0 -> 2.3.3: <reason>

and the gate accepts exactly that transition, nothing wider.

Usage::

    python scripts/check_lock_downgrades.py --base-ref origin/main
    python scripts/check_lock_downgrades.py --base-lock old.lock \\
        --head-lock uv.lock --declarations pr-text.txt
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from packaging.version import InvalidVersion, Version

REPO = Path(__file__).resolve().parents[1]
TRAILER = re.compile(
    r"^Lock-Downgrade:\s*(?P<name>[A-Za-z0-9._-]+)\s+(?P<old>\S+)\s*->\s*(?P<new>\S+?)\s*:\s*(?P<why>\S.*)$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Change:
    """One package whose highest locked version moved."""

    name: str
    old: str
    new: str | None

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


def highest_versions(lock_text: str) -> dict[str, str]:
    """Map each package in a ``uv.lock`` to the highest version it holds."""
    lock = tomllib.loads(lock_text)
    best: dict[str, str] = {}
    for package in lock.get("package", []):
        name = _normalise(package["name"])
        version = package.get("version")
        if version is None:  # the editable project itself
            continue
        if name not in best or _version(version) > _version(best[name]):
            best[name] = version
    return best


def lowest_versions(lock_text: str) -> dict[str, str]:
    """Map each package in a ``uv.lock`` to the lowest version it holds.

    Where the lock splits a package across Python or platform forks,
    Dependabot reports and updates the lowest of them as "current".
    """
    lock = tomllib.loads(lock_text)
    low: dict[str, str] = {}
    for package in lock.get("package", []):
        name = _normalise(package["name"])
        version = package.get("version")
        if version is None:
            continue
        if name not in low or _version(version) < _version(low[name]):
            low[name] = version
    return low


def compare(base_text: str, head_text: str) -> list[Change]:
    """Every package whose highest locked version differs or disappeared."""
    base, head = highest_versions(base_text), highest_versions(head_text)
    changes = []
    for name, old in sorted(base.items()):
        new = head.get(name)
        if new != old:
            changes.append(Change(name, old, new))
    return changes


def declared(text: str) -> set[tuple[str, str, str]]:
    """The ``Lock-Downgrade`` trailers in commit messages or a PR body."""
    return {(_normalise(m["name"]), m["old"], m["new"]) for m in TRAILER.finditer(text)}


def undeclared_downgrades(base_text: str, head_text: str, declarations: str = "") -> list[Change]:
    """Downgrades with no matching ``Lock-Downgrade`` trailer."""
    allowed = declared(declarations)
    return [
        change
        for change in compare(base_text, head_text)
        if change.is_downgrade and (change.name, change.old, change.new) not in allowed
    ]


def _git(*args: str) -> str:
    return subprocess.run(  # noqa: S603 # nosec B603 - fixed argv, read-only git
        ["git", *args],  # noqa: S607 # nosec B607 - git from PATH, as every CI gate
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-ref", help="git ref whose uv.lock is the baseline")
    source.add_argument("--base-lock", type=Path, help="path to the baseline uv.lock")
    parser.add_argument("--head-lock", type=Path, default=REPO / "uv.lock")
    parser.add_argument(
        "--declarations",
        type=Path,
        help="file holding the PR's commit messages and body (Lock-Downgrade trailers)",
    )
    args = parser.parse_args(argv)

    if args.base_ref:
        base_text = _git("show", f"{args.base_ref}:uv.lock")
        declarations = _git("log", "--format=%B", f"{args.base_ref}..HEAD")
    else:
        base_text = args.base_lock.read_text(encoding="utf-8")
        declarations = ""
    if args.declarations:
        declarations += "\n" + args.declarations.read_text(encoding="utf-8")
    head_text = args.head_lock.read_text(encoding="utf-8")

    changes = compare(base_text, head_text)
    allowed = declared(declarations)
    failures = 0
    for change in changes:
        if change.new is None:
            print(f"removed     {change.name} {change.old}")
        elif not change.is_downgrade:
            print(f"upgraded    {change.name} {change.old} -> {change.new}")
        elif (change.name, change.old, change.new) in allowed:
            print(f"declared    {change.name} {change.old} -> {change.new} (Lock-Downgrade)")
        else:
            failures += 1
            print(
                f"::error title=uv.lock downgrade::{change.name} {change.old} -> {change.new}"
                " is a downgrade with no 'Lock-Downgrade:"
                f" {change.name} {change.old} -> {change.new}: <reason>' trailer"
            )
    print(f"{len(changes)} locked package(s) changed; {failures} undeclared downgrade(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
