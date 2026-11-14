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
newest release is above the locked one, and fails on either outcome. It
proposes what Dependabot would: the newest release no ``ignore`` entry covers
(an ignored newest release falls back to the next one), and it also replays
each ``groups`` entry whose members would update together, since a group can
conflict where its members alone do not (#393 was a two-package group). An
update that cannot be taken on purpose is declared in ``.github/dependabot.yml``
as an ``ignore`` entry with ``versions`` (and a comment saying why); an ignore
whose release has become takeable is reported as ``stale-ignore``, so it is
lifted when upstream catches up.

Each update is replayed under two resolvers: the ``uv`` on ``PATH`` (the
version CI, the Dockerfile and ``uv.lock`` are pinned to) and the uv release
Dependabot's updater bundles (:data:`DEPENDABOT_UV_VERSION`, fetched with
``uv tool run``). They are not the same program. On 2026-11-09 the lock
resolved under the repository's 0.11.32 and under Dependabot's 0.13.7 had no
solution on an emscripten fork that 0.11.32 tolerated; the replay, which ran
only under 0.11.32, passed the day before all seven of that week's updates
failed. Before the per-package replay each resolver must relock the project
unchanged (the ``baseline`` row); a resolver that cannot is reported once,
instead of once per dependency.

The replay reads live PyPI, so a new upstream release can change its answer
without any change here. CI therefore runs it on pull requests that change
``pyproject.toml``, ``uv.lock`` or ``.github/dependabot.yml``, and on a weekly
schedule ahead of Dependabot's own run (``.github/workflows/dependabot-replay.yml``),
rather than on every pull request.

Network: PyPI's JSON API and ``uv lock`` (which reads PyPI). Run::

    python scripts/simulate_dependabot_uv.py            # all direct deps
    python scripts/simulate_dependabot_uv.py shap h5py  # just these
    python scripts/simulate_dependabot_uv.py --resolver repo   # PATH uv only
"""

from __future__ import annotations

import argparse
import fnmatch
import functools
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

#: The uv release Dependabot's updater bundles, as its own run log reports it
#: ("Package manager uv, Info : uv 0.13.7", run of 2026-11-09). Dependabot
#: moves this without notice; when one of its runs fails where this replay
#: passed, read the version from that run's log and update it here.
DEPENDABOT_UV_VERSION = "0.13.7"


@dataclass(frozen=True)
class Resolver:
    """One ``uv`` program the replay locks with."""

    name: str
    argv: tuple[str, ...]


#: The ``uv`` on ``PATH``: the version that wrote ``uv.lock``.
REPO_RESOLVER = Resolver("repo", ("uv",))

#: The uv release Dependabot runs, fetched into uv's tool cache on first use.
DEPENDABOT_RESOLVER = Resolver(
    f"dependabot-{DEPENDABOT_UV_VERSION}",
    ("uv", "tool", "run", "--quiet", "--from", f"uv=={DEPENDABOT_UV_VERSION}", "uv"),
)

RESOLVERS: dict[str, tuple[Resolver, ...]] = {
    "repo": (REPO_RESOLVER,),
    "dependabot": (DEPENDABOT_RESOLVER,),
    "both": (REPO_RESOLVER, DEPENDABOT_RESOLVER),
}


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


@functools.cache
def releases(name: str, floor: Version) -> list[Version]:
    """Final, non-yanked releases that install on ``floor``, newest first.

    Cached, so replaying under a second resolver asks PyPI nothing new and both
    resolvers are judged against the same release list.
    """
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
    found: list[Version] = []
    for text, files in data.get("releases", {}).items():
        try:
            version = Version(text)
        except InvalidVersion:
            continue
        live = [f for f in files if not f.get("yanked")]
        if version.is_prerelease or version.is_devrelease or not live:
            continue
        requires = {f.get("requires_python") or "" for f in live}
        if any(_admits(r, floor) for r in requires):
            found.append(version)
    return sorted(found, reverse=True)


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
    """What Dependabot's update of one package (or one group) would do."""

    name: str
    locked: Version
    candidate: Version | None
    status: str  # current | ignored | ok | unsolvable | downgrades | stale-ignore | error
    detail: str = ""


FAILING = frozenset({"unsolvable", "downgrades", "stale-ignore", "error"})


def update_type(locked: Version, candidate: Version) -> str:
    """Dependabot's semver class of an update (major / minor / patch)."""
    if candidate.major != locked.major:
        return "major"
    if candidate.minor != locked.minor:
        return "minor"
    return "patch"


def simulate(
    name: str,
    strings: list[str],
    locked: Version,
    floor: Version,
    ignores: dict[str, Ignore],
    pyproject_text: str,
    lock_text: str,
    resolver: Resolver = REPO_RESOLVER,
) -> Outcome:
    """Replay the update Dependabot would propose for ``name``.

    Dependabot proposes the newest release above the locked one that no
    ``ignore`` entry covers, so an ignored newest release falls back to the
    next one. An ignore that covers a release which now updates cleanly is
    reported as ``stale-ignore``, so it is lifted rather than left to hide a
    takeable update.
    """
    newer = [v for v in releases(name, floor) if v > locked]
    if not newer:
        return Outcome(name, locked, None, "current")
    rule = ignores.get(name)
    proposal = next((v for v in newer if not (rule and rule.covers(v, locked))), None)
    newest = newer[0]
    if rule and newest != proposal and any(newest in spec for spec in rule.versions):
        status, detail, _ = lock_with_pins(
            {name: (strings, newest)}, pyproject_text, lock_text, resolver=resolver
        )
        if status == "ok":
            return Outcome(
                name, locked, newest, "stale-ignore", "the ignored release now updates cleanly"
            )
    if proposal is None:
        return Outcome(name, locked, newest, "ignored", "every newer release is ignored")
    status, detail, _ = lock_with_pins(
        {name: (strings, proposal)}, pyproject_text, lock_text, resolver=resolver
    )
    if proposal != newest:
        detail = f"newest {newest} is ignored; Dependabot proposes {proposal}" + (
            f"; {detail}" if detail else ""
        )
    return Outcome(name, locked, proposal, status, detail)


def lock_with_pins(
    pins: dict[str, tuple[list[str], Version]],
    pyproject_text: str,
    lock_text: str,
    resolver: Resolver = REPO_RESOLVER,
) -> tuple[str, str, str | None]:
    """Pin every package in ``pins`` as Dependabot does and relock with ``resolver``.

    With no pins this relocks the project unchanged, which is the baseline a
    resolver must pass before any update under it means anything.

    Returns ``(status, detail, new_lock)``: ``ok``, ``unsolvable`` (the
    resolver found no solution), ``downgrades`` (another package moved to an
    older version) or ``error`` (uv failed for another reason twice, such as
    the index being unreachable; reported as itself, never as unsolvable).
    """
    text = pyproject_text
    for name, (strings, version) in pins.items():
        text = pin(text, strings, name, version)
    argv = [*resolver.argv, "lock", "--no-progress"]
    for name in pins:
        argv += ["--upgrade-package", name]
    stderr = ""
    for _ in range(2):
        with tempfile.TemporaryDirectory(prefix="dependabot-replay-") as tmp:
            work = Path(tmp)
            (work / "pyproject.toml").write_text(text, encoding="utf-8")
            (work / "uv.lock").write_text(lock_text, encoding="utf-8")
            shutil.copy(REPO / "README.md", work / "README.md")
            run = subprocess.run(  # noqa: S603 # nosec B603 - fixed argv, no shell
                argv, cwd=work, capture_output=True, text=True, check=False
            )
            if run.returncode == 0:
                new_lock = (work / "uv.lock").read_text(encoding="utf-8")
                break
            stderr = run.stderr
            if "No solution found" in stderr:
                reason = next(ln.strip() for ln in stderr.splitlines() if "No solution" in ln)
                return "unsolvable", reason, None
    else:
        return "error", stderr.strip()[-400:], None
    moved = [c for c in compare(lock_text, new_lock) if c.is_downgrade and c.name not in pins]
    if moved:
        return "downgrades", ", ".join(f"{c.name} {c.old} -> {c.new}" for c in moved), new_lock
    return "ok", "", new_lock


@dataclass
class Group:
    """A ``groups`` entry of the uv ecosystem (version updates)."""

    name: str
    patterns: list[str]
    exclude: list[str]
    update_types: list[str]

    def takes(self, package: str, kind: str) -> bool:
        if any(fnmatch.fnmatch(package, p) for p in self.exclude):
            return False
        allowed = {t.removeprefix("version-update:semver-") for t in self.update_types}
        if allowed and kind not in allowed:
            return False
        return any(fnmatch.fnmatch(package, p) for p in self.patterns)


def load_groups(path: Path) -> list[Group]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    groups: list[Group] = []
    for update in config.get("updates", []):
        if update.get("package-ecosystem") != "uv":
            continue
        for name, spec in (update.get("groups") or {}).items():
            if spec.get("applies-to", "version-updates") != "version-updates":
                continue
            groups.append(
                Group(
                    name,
                    [normalise(p) if "*" not in p else p for p in spec.get("patterns", [])],
                    list(spec.get("exclude-patterns", [])),
                    list(spec.get("update-types", [])),
                )
            )
    return groups


def group_members(outcomes: list[Outcome], groups: list[Group]) -> dict[str, list[Outcome]]:
    """Proposed updates by the first group that takes them, as Dependabot assigns."""
    members: dict[str, list[Outcome]] = {}
    for o in outcomes:
        if o.candidate is None or o.status in {"current", "ignored"}:
            continue
        kind = update_type(o.locked, o.candidate)
        for group in groups:
            if group.takes(o.name, kind):
                members.setdefault(group.name, []).append(o)
                break
    return members


def replay(
    resolver: Resolver,
    names: list[str],
    requirements: dict[str, list[str]],
    locked: dict[str, str],
    floor: Version,
    ignores: dict[str, Ignore],
    groups: list[Group],
    pyproject_text: str,
    lock_text: str,
    jobs: int,
) -> list[Outcome]:
    """Replay every update under one resolver, after its baseline relock.

    A resolver that cannot relock the unchanged project cannot solve any
    update either, so its failure is reported as the single ``baseline`` row
    and the per-package replay is skipped. The baseline also runs alone, which
    lets ``uv tool run`` fetch Dependabot's uv once before the pool starts.
    """
    status, detail, _ = lock_with_pins({}, pyproject_text, lock_text, resolver=resolver)
    baseline = Outcome("baseline", Version("0"), None, status, detail)
    if status in FAILING:
        return [baseline]

    with ThreadPoolExecutor(max_workers=jobs) as pool:
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
                    resolver,
                ),
                names,
            )
        )

    for group, members in sorted(group_members(outcomes, groups).items()):
        if len(members) < 2 or any(m.status in FAILING for m in members):
            continue  # a lone member is its own update; a failing one is already reported
        pins = {m.name: (requirements[m.name], m.candidate) for m in members if m.candidate}
        status, detail, _ = lock_with_pins(pins, pyproject_text, lock_text, resolver=resolver)
        label = "+".join(sorted(pins))
        outcomes.append(Outcome(f"group:{group}[{label}]", Version("0"), None, status, detail))
    return [baseline, *outcomes]


def report(resolver: Resolver, outcomes: list[Outcome]) -> int:
    """Print one resolver's outcomes; return how many fail or regress."""
    print(f"== resolver {resolver.name}: {' '.join(resolver.argv)} ==")
    failures = 0
    for o in outcomes:
        if o.name == "baseline" or o.name.startswith("group:"):
            line = f"{o.status:<13} {o.name}"
        else:
            line = f"{o.status:<13} {o.name} {o.locked}" + (
                f" -> {o.candidate}" if o.candidate and o.candidate > o.locked else ""
            )
        if o.status in FAILING:
            failures += 1
            title = f"Dependabot uv replay ({o.status}, {resolver.name})"
            print(f"::error title={title}::{line}: {o.detail}")
        else:
            print(line + (f"  [{o.detail}]" if o.detail else ""))
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("packages", nargs="*", help="limit to these direct dependencies")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument(
        "--resolver",
        choices=sorted(RESOLVERS),
        default="both",
        help="which uv to lock with: the one on PATH, the one Dependabot bundles "
        f"({DEPENDABOT_UV_VERSION}), or both (default)",
    )
    args = parser.parse_args(argv)

    pyproject_text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    pyproject = tomllib.loads(pyproject_text)
    lock_text = (REPO / "uv.lock").read_text(encoding="utf-8")
    locked = lowest_versions(lock_text)
    floor = python_floor(pyproject)
    ignores = load_ignores(REPO / ".github" / "dependabot.yml")
    groups = load_groups(REPO / ".github" / "dependabot.yml")
    requirements = direct_requirements(pyproject)
    wanted = {normalise(p) for p in args.packages} or set(requirements)
    names = sorted(n for n in wanted if n in requirements and n in locked)

    failures = replayed = 0
    for resolver in RESOLVERS[args.resolver]:
        outcomes = replay(
            resolver,
            names,
            requirements,
            locked,
            floor,
            ignores,
            groups,
            pyproject_text,
            lock_text,
            args.jobs,
        )
        failures += report(resolver, outcomes)
        replayed += len(outcomes) - 1  # the baseline row is not an update
    print(
        f"Python floor {floor}; {len(names)} direct dependencies under "
        f"{len(RESOLVERS[args.resolver])} resolver(s), {replayed} update(s) and group(s) "
        f"replayed; {failures} would fail or regress"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
