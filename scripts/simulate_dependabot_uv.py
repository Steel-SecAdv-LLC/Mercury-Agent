#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Replay Dependabot's uv update for every direct dependency, on the PR.

Dependabot's ``uv`` updater picks, for each direct dependency, the newest
release that installs on the project's lowest supported Python, writes
``name==<that version>`` into a copy of ``pyproject.toml`` and runs
``uv lock``. Two outcomes reached main unseen after PR #388:

* **unsolvable** -- ``shap==0.51.0`` has no solution on the Intel-macOS fork
  (shap pins ``numba<0.63`` there, and every such numba needs
  ``numpy<2.4``), so the weekly run failed with
  ``dependency_file_not_resolvable``;
The "current" version is the lowest the lock holds for a package, which is
what Dependabot reads when the lock splits a package across forks.

* **solvable only by downgrading** -- the h5py/nixtla group PR (#393) took
  pandas 3.1.0 -> 2.3.3 and TensorFlow 2.21.0 -> 2.20.0 to fit.

This script runs the same pin-and-lock for every direct dependency whose
newest release is above the locked one, and fails on either outcome. An
update that cannot be taken on purpose is declared in ``.github/dependabot.yml``
as an ``ignore`` entry with ``versions`` (and a comment saying why); the
script skips those and reports any such entry that has become takeable, so
the ignore is lifted when upstream catches up.

Network: PyPI's JSON API and ``uv lock`` (which reads PyPI). Run::

    python scripts/simulate_dependabot_uv.py            # all direct deps
    python scripts/simulate_dependabot_uv.py shap h5py  # just these
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from packaging.requirements import Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_lock_downgrades import compare, lowest_versions

REPO = Path(__file__).resolve().parents[1]
PROJECT = "mercury-agent-global"
PYPI = "https://pypi.org/pypi/{name}/json"


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass
class Ignore:
    """One ``ignore`` entry of the uv ecosystem in dependabot.yml."""

    versions: list[SpecifierSet]
    major: bool

    def covers(self, candidate: Version, locked: Version) -> bool:
        if self.major and candidate.major > locked.major:
            return True
        return any(candidate in spec for spec in self.versions)


def load_ignores(path: Path) -> dict[str, Ignore]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    ignores: dict[str, Ignore] = {}
    for update in config.get("updates", []):
        if update.get("package-ecosystem") != "uv":
            continue
        for entry in update.get("ignore", []) or []:
            name = normalise(entry["dependency-name"])
            current = ignores.setdefault(name, Ignore([], False))
            for text in entry.get("versions", []) or []:
                try:
                    current.versions.append(SpecifierSet(text.replace(" ", "")))
                except InvalidSpecifier as exc:
                    raise SystemExit(f"dependabot.yml: bad versions {text!r} for {name}") from exc
            if "version-update:semver-major" in (entry.get("update-types") or []):
                current.major = True
    return ignores


def direct_requirements(pyproject: dict[str, Any]) -> dict[str, list[str]]:
    """Every direct requirement string of the project, grouped by package."""
    strings = list(pyproject["project"].get("dependencies", []))
    for extra in pyproject["project"].get("optional-dependencies", {}).values():
        strings += extra
    found: dict[str, list[str]] = {}
    for text in strings:
        requirement = Requirement(text)
        name = normalise(requirement.name)
        if name == PROJECT or requirement.url:
            continue
        found.setdefault(name, []).append(text)
    return found


def python_floor(pyproject: dict[str, Any]) -> Version:
    spec = pyproject["project"]["requires-python"]
    match = re.search(r">=\s*(\d+\.\d+)", spec)
    if not match:
        raise SystemExit(f"cannot read the Python floor from requires-python {spec!r}")
    return Version(match.group(1))


def newest_release(name: str, floor: Version) -> Version | None:
    """Newest final, non-yanked release that installs on ``floor``."""
    data: dict[str, Any] = {}
    for attempt in range(3):
        try:
            url = PYPI.format(name=name)  # fixed https:// PyPI JSON endpoint
            with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 # nosec B310
                data = json.load(response)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2 ** (attempt + 1))
    best: Version | None = None
    for text, files in data.get("releases", {}).items():
        try:
            version = Version(text)
        except InvalidVersion:
            continue
        live = [f for f in files if not f.get("yanked")]
        if version.is_prerelease or version.is_devrelease or not live:
            continue
        requires = {f.get("requires_python") or "" for f in live}
        if not any(_admits(r, floor) for r in requires):
            continue
        if best is None or version > best:
            best = version
    return best


def _admits(requires_python: str, floor: Version) -> bool:
    if not requires_python.strip():
        return True
    try:
        return SpecifierSet(requires_python).contains(f"{floor}.0", prereleases=True) or (
            SpecifierSet(requires_python).contains(str(floor), prereleases=True)
        )
    except InvalidSpecifier:
        return True


def pin(pyproject_text: str, strings: list[str], name: str, version: Version) -> str:
    """Rewrite each requirement on ``name`` to ``==version``, as Dependabot does."""
    out = pyproject_text
    for text in strings:
        requirement = Requirement(text)
        extras = f"[{','.join(sorted(requirement.extras))}]" if requirement.extras else ""
        marker = f"; {requirement.marker}" if requirement.marker else ""
        pinned = f"{requirement.name}{extras}=={version}{marker}"
        out = out.replace(f'"{text}"', json.dumps(pinned))
    return out


@dataclass
class Outcome:
    name: str
    locked: Version
    candidate: Version | None
    status: str  # current | ignored | ok | unsolvable | downgrades | stale-ignore
    detail: str = ""


def simulate(
    name: str,
    strings: list[str],
    locked: Version,
    floor: Version,
    ignores: dict[str, Ignore],
    pyproject_text: str,
    lock_text: str,
) -> Outcome:
    candidate = newest_release(name, floor)
    if candidate is None or candidate <= locked:
        return Outcome(name, locked, candidate, "current")
    ignored = ignores.get(name)
    result = _lock_with_pin(name, strings, candidate, pyproject_text, lock_text)
    if ignored and ignored.covers(candidate, locked):
        if result.status == "ok" and any(candidate in s for s in ignored.versions):
            result.status, result.detail = "stale-ignore", "the ignored version now updates cleanly"
            return result
        return Outcome(name, locked, candidate, "ignored", result.detail)
    return result


def _lock_with_pin(
    name: str, strings: list[str], candidate: Version, pyproject_text: str, lock_text: str
) -> Outcome:
    locked = Version(lowest_versions(lock_text)[name])
    with tempfile.TemporaryDirectory(prefix=f"dependabot-{name}-") as tmp:
        work = Path(tmp)
        (work / "pyproject.toml").write_text(
            pin(pyproject_text, strings, name, candidate), encoding="utf-8"
        )
        (work / "uv.lock").write_text(lock_text, encoding="utf-8")
        shutil.copy(REPO / "README.md", work / "README.md")
        run = subprocess.run(  # noqa: S603 # nosec B603 - fixed argv, no shell
            ["uv", "lock", "--upgrade-package", name, "--no-progress"],  # noqa: S607 # nosec B607
            cwd=work,
            capture_output=True,
            text=True,
            check=False,
        )
        if run.returncode != 0:
            reason = next((ln.strip() for ln in run.stderr.splitlines() if "No solution" in ln), "")
            return Outcome(
                name, locked, candidate, "unsolvable", reason or run.stderr.strip()[-400:]
            )
        new_lock = (work / "uv.lock").read_text(encoding="utf-8")
    downgrades = [c for c in compare(lock_text, new_lock) if c.is_downgrade and c.name != name]
    if downgrades:
        detail = ", ".join(f"{c.name} {c.old} -> {c.new}" for c in downgrades)
        return Outcome(name, locked, candidate, "downgrades", detail)
    return Outcome(name, locked, candidate, "ok")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("packages", nargs="*", help="limit to these direct dependencies")
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args(argv)

    pyproject_text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    pyproject = tomllib.loads(pyproject_text)
    lock_text = (REPO / "uv.lock").read_text(encoding="utf-8")
    locked = lowest_versions(lock_text)
    floor = python_floor(pyproject)
    ignores = load_ignores(REPO / ".github" / "dependabot.yml")
    requirements = direct_requirements(pyproject)
    wanted = {normalise(p) for p in args.packages} or set(requirements)
    names = sorted(n for n in wanted if n in requirements and n in locked)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        outcomes = list(
            pool.map(
                lambda n: simulate(
                    n,
                    requirements[n],
                    Version(locked[n]),
                    floor,
                    ignores,
                    pyproject_text,
                    lock_text,
                ),
                names,
            )
        )

    failures = 0
    for o in outcomes:
        line = f"{o.status:<13} {o.name} {o.locked}" + (
            f" -> {o.candidate}" if o.candidate and o.candidate > o.locked else ""
        )
        if o.status in {"unsolvable", "downgrades", "stale-ignore"}:
            failures += 1
            print(f"::error title=Dependabot uv replay ({o.status})::{line}: {o.detail}")
        else:
            print(line + (f"  [{o.detail}]" if o.detail else ""))
    print(
        f"Python floor {floor}; {len(outcomes)} direct dependencies replayed; {failures} would fail or regress"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
