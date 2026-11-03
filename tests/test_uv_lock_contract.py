# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``uv.lock`` is the one resolution every uv-managed install path consumes.

The Dockerfile builder, the Code Quality, Type Checking, Core Tests and ML
Tests lanes install with uv; the lockfile must exist, state the project's
Python floor, and be installed with ``--locked`` everywhere so a stale lock is
refused rather than silently re-resolved. The Workflow Hardening lane runs
``uv lock --check`` so the drift is named on the pull request.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / "uv.lock"
CI = REPO / ".github" / "workflows" / "ci.yml"
DOCKERFILE = REPO / "Dockerfile"


def test_lockfile_exists_and_states_the_python_floor() -> None:
    lock = LOCK.read_text(encoding="utf-8")
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    floor = re.search(r'^requires-python = "([^"]+)"', pyproject, re.M)
    assert floor is not None
    assert f'requires-python = "{floor.group(1)}"' in lock
    assert 'name = "mercury-agent-global"' in lock
    # The lock resolves every extra, the git-pinned [pqc] extra included.
    assert "AMA-Cryptography.git" in lock


def test_every_uv_sync_is_locked() -> None:
    for path in (CI, DOCKERFILE):
        text = path.read_text(encoding="utf-8")
        syncs = re.findall(r"uv sync[^\n]*", text)
        assert syncs, path
        for line in syncs:
            assert "--locked" in line, (path, line)


def test_ci_lanes_and_dockerfile_pin_the_same_uv_version() -> None:
    ci = CI.read_text(encoding="utf-8")
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    ci_versions = set(re.findall(r'^\s+version: "(\d+\.\d+\.\d+)"\s*$', ci, re.M))
    docker_version = re.search(r'"uv==(\d+\.\d+\.\d+)"', dockerfile)
    assert docker_version is not None
    assert ci_versions == {docker_version.group(1)}, (ci_versions, docker_version.group(1))


def test_hardening_lane_checks_the_lock_and_builder_copies_it() -> None:
    ci = CI.read_text(encoding="utf-8")
    assert "run: uv lock --check" in ci
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert re.search(r"^COPY pyproject.toml uv.lock README.md /app/$", dockerfile, re.M)
    # The builder's sync must keep the venv's pip, setuptools, wheel and the
    # uv it installed: the floor upgrade, the AMA build and the final cleanup
    # run through them, and an exact sync may remove anything the lock did not
    # resolve.
    builder_sync = re.search(r"^RUN uv sync[^\n]*$", dockerfile, re.M)
    assert builder_sync is not None and "--inexact" in builder_sync.group(0)
    # uv is build tooling: removed with pip before the venv leaves the builder.
    assert "/opt/venv/bin/python -m pip uninstall -y uv" in dockerfile


def _wheel_fits(filename: str, python_tag: str) -> bool:
    """Whether a wheel file name installs on ``python_tag`` on Linux x86_64."""
    stem = filename[: -len(".whl")]
    py_tags, abi_tag, platform_tag = stem.split("-")[-3:]
    on_linux = platform_tag == "any" or any(
        "x86_64" in part and ("manylinux" in part or part.startswith("linux"))
        for part in platform_tag.split(".")
    )
    if not on_linux:
        return False
    if abi_tag == "abi3":
        return True
    return any(
        tag in {python_tag, "py3", "py2.py3"} or tag.startswith("py3") for tag in py_tags.split(".")
    )


def _installed_on(lock: dict[str, Any], env: dict[str, str]) -> dict[str, dict[str, Any]]:
    """The packages ``uv sync --extra all --extra dev`` installs in ``env``.

    Walks the lock's dependency graph from the project with every extra,
    following only the edges whose markers hold in ``env`` and, for a package
    the lock resolved to several versions, the version whose resolution
    markers hold there.
    """
    from packaging.markers import Marker

    def holds(marker: str | None) -> bool:
        return marker is None or Marker(marker).evaluate(env)

    by_name: dict[str, list[dict[str, Any]]] = {}
    for pkg in lock["package"]:
        by_name.setdefault(pkg["name"], []).append(pkg)

    def resolve(dep: dict[str, Any]) -> dict[str, Any] | None:
        candidates = by_name[dep["name"]]
        if "version" in dep:
            candidates = [c for c in candidates if c["version"] == dep["version"]]
        fitting = [
            c
            for c in candidates
            if not c.get("resolution-markers") or any(holds(m) for m in c["resolution-markers"])
        ]
        return fitting[0] if fitting else None

    root = next(p for p in lock["package"] if p["name"] == "mercury-agent-global")
    reached: dict[str, dict[str, Any]] = {}
    pending: list[tuple[dict[str, Any], frozenset[str]]] = [
        (root, frozenset(root.get("optional-dependencies", {})))
    ]
    seen: set[tuple[str, str, frozenset[str]]] = set()
    while pending:
        pkg, extras = pending.pop()
        key = (pkg["name"], pkg["version"], extras)
        if key in seen:
            continue
        seen.add(key)
        reached[pkg["name"]] = pkg
        edges = list(pkg.get("dependencies", []))
        for extra in extras:
            edges.extend(pkg.get("optional-dependencies", {}).get(extra, []))
        for dep in edges:
            if not holds(dep.get("marker")):
                continue
            target = resolve(dep)
            if target is not None:
                pending.append((target, frozenset(dep.get("extra", []))))
    return reached


def test_every_binary_package_has_a_wheel_for_every_supported_python_on_linux() -> None:
    """A lock that unifies one version across Pythons can pick a release with
    no wheel for the newest one, and ``uv sync`` then builds it from source on
    that lane (h5py 3.14.0 on Python 3.14 failed on the missing HDF5 library
    while 3.16 ships 3.14 wheels). Every package the lock installs on a Linux
    lane that ships binary wheels at all must ship one for that lane's Python."""
    import tomllib

    lock = tomllib.loads(LOCK.read_text(encoding="utf-8"))
    missing: list[str] = []
    for minor in (11, 12, 13, 14):
        env = {
            "python_full_version": f"3.{minor}.0",
            "python_version": f"3.{minor}",
            "sys_platform": "linux",
            "platform_system": "Linux",
            "platform_machine": "x86_64",
            "implementation_name": "cpython",
            "platform_python_implementation": "CPython",
            "os_name": "posix",
        }
        for pkg in _installed_on(lock, env).values():
            names = [w["url"].rsplit("/", 1)[-1] for w in pkg.get("wheels", [])]
            binary = any(re.search(r"-cp3\d+-", n) or "-abi3-" in n for n in names)
            if binary and not any(_wheel_fits(n, f"cp3{minor}") for n in names):
                missing.append(f"{pkg['name']}=={pkg['version']} on Python 3.{minor}")
    assert not missing, "\n".join(missing)
