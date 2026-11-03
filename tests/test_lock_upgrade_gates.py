# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The uv.lock downgrade gate and the Dependabot replay, without the network.

``scripts/check_lock_downgrades.py`` fails a pull request whose lock moves a
package to an older version (Dependabot PR #393 took pandas 3.1.0 -> 2.3.3
behind a "minor" title). ``scripts/simulate_dependabot_uv.py`` replays
Dependabot's pin-and-lock per direct dependency (the shap update that could
not be solved on main). Both run in CI's Workflow Hardening job; these tests
pin their logic and their wiring.
"""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml
from packaging.version import Version

if TYPE_CHECKING:
    from types import ModuleType

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


def _load(name: str) -> ModuleType:
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


downgrades = _load("check_lock_downgrades")
replay = _load("simulate_dependabot_uv")


def _lock(*packages: tuple[str, str]) -> str:
    body = ["version = 1", 'requires-python = ">=3.11"', ""]
    body += ["[[package]]", 'name = "mercury-agent-global"', 'source = { editable = "." }', ""]
    for name, version in packages:
        body += ["[[package]]", f'name = "{name}"', f'version = "{version}"', ""]
    return "\n".join(body)


BASE = _lock(("pandas", "3.1.0"), ("h5py", "3.14.0"), ("h5py", "3.16.0"), ("pytz", "2026.6"))


def test_a_lower_highest_version_is_a_downgrade() -> None:
    head = _lock(("pandas", "2.3.3"), ("h5py", "3.16.0"))
    found = downgrades.undeclared_downgrades(BASE, head)
    assert [(c.name, c.old, c.new) for c in found] == [("pandas", "3.1.0", "2.3.3")]


def test_dropping_a_fork_below_the_highest_is_not_a_downgrade() -> None:
    # h5py keeps 3.16.0; losing the 3.14.0 fork is not a regression.
    head = _lock(("pandas", "3.1.0"), ("h5py", "3.16.0"))
    assert downgrades.undeclared_downgrades(BASE, head) == []


def test_a_removed_package_is_reported_not_failed() -> None:
    head = _lock(("pandas", "3.1.0"), ("h5py", "3.16.0"))
    changes = downgrades.compare(BASE, head)
    assert [(c.name, c.new) for c in changes] == [("pytz", None)]
    assert not any(c.is_downgrade for c in changes)


def test_upgrades_pass() -> None:
    head = _lock(("pandas", "3.2.0"), ("h5py", "3.17.0"), ("pytz", "2026.6"))
    assert downgrades.undeclared_downgrades(BASE, head) == []


def test_a_trailer_declares_exactly_one_transition() -> None:
    head = _lock(("pandas", "2.3.3"), ("h5py", "3.16.0"), ("pytz", "2026.6"))
    exact = "Lock-Downgrade: pandas 3.1.0 -> 2.3.3: nixtla 0.9.0 needs pandas<3\n"
    assert downgrades.undeclared_downgrades(BASE, head, exact) == []
    wrong_target = "Lock-Downgrade: pandas 3.1.0 -> 2.2.0: some reason\n"
    assert downgrades.undeclared_downgrades(BASE, head, wrong_target)
    no_reason = "Lock-Downgrade: pandas 3.1.0 -> 2.3.3:\n"
    assert downgrades.undeclared_downgrades(BASE, head, no_reason)


def test_names_are_normalised_like_pip() -> None:
    base = _lock(("tf_keras", "2.21.0"))
    head = _lock(("tf-keras", "2.20.1"))
    found = downgrades.undeclared_downgrades(base, head)
    assert [(c.name, c.old, c.new) for c in found] == [("tf-keras", "2.21.0", "2.20.1")]


def test_lowest_version_is_what_dependabot_calls_current() -> None:
    assert downgrades.lowest_versions(BASE)["h5py"] == "3.14.0"
    assert downgrades.highest_versions(BASE)["h5py"] == "3.16.0"


def test_the_committed_lock_compares_clean_against_itself() -> None:
    lock = (REPO / "uv.lock").read_text(encoding="utf-8")
    assert downgrades.compare(lock, lock) == []


def test_pin_rewrites_every_requirement_like_dependabot() -> None:
    text = '[project]\ndependencies = ["shap>=0.46.0", "x[a]>=1; python_version < \'3.14\'"]\n'
    pinned = replay.pin(text, ["shap>=0.46.0"], "shap", Version("0.51.0"))
    assert '"shap==0.51.0"' in pinned
    pinned = replay.pin(text, ["x[a]>=1; python_version < '3.14'"], "x", Version("2.0"))
    assert '"x[a]==2.0; python_version < \\"3.14\\""' in pinned
    assert tomllib.loads(pinned)["project"]["dependencies"][1].startswith("x[a]==2.0")


def test_ignore_entries_cover_only_their_ranges() -> None:
    entry = replay.Ignore(versions=[replay.SpecifierSet(">=0.8.0,<0.10.0")], major=False)
    assert entry.covers(Version("0.9.0"), Version("0.7.2"))
    assert not entry.covers(Version("0.10.0"), Version("0.7.2"))
    major = replay.Ignore(versions=[], major=True)
    assert major.covers(Version("3.0"), Version("2.4"))
    assert not major.covers(Version("2.5"), Version("2.4"))


def test_dependabot_ignores_parse_and_carry_the_known_conflicts() -> None:
    ignores = replay.load_ignores(REPO / ".github" / "dependabot.yml")
    assert ignores["nixtla"].covers(Version("0.9.0"), Version("0.7.2"))
    assert ignores["h5py"].covers(Version("3.16.0"), Version("3.14.0"))
    assert ignores["torch"].major


def test_python_floor_reads_requires_python() -> None:
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert replay.python_floor(pyproject) == Version("3.11")


@pytest.mark.parametrize(
    ("requires_python", "admits"),
    [("", True), (">=3.11", True), (">=3.12", False), (">=3.10,<3.14", True)],
)
def test_requires_python_on_the_floor(requires_python: str, admits: bool) -> None:
    assert replay._admits(requires_python, Version("3.11")) is admits


def test_both_gates_run_in_workflow_hardening() -> None:
    workflow = yaml.safe_load(
        (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    hardening = next(
        job for job in workflow["jobs"].values() if job.get("name") == "Workflow Hardening"
    )
    runs = [str(step.get("run") or "") for step in hardening["steps"]]
    assert any("scripts/check_lock_downgrades.py" in run for run in runs)
    assert any("scripts/simulate_dependabot_uv.py" in run for run in runs)
    assert any("uv lock --check" in run for run in runs)
    lock_gate = next(s for s in hardening["steps"] if "check_lock_downgrades" in str(s.get("run")))
    assert lock_gate["if"] == "github.event_name == 'pull_request'"
    # The PR body reaches the shell through env, never through ${{ }} in run.
    assert "PR_BODY" in lock_gate["env"] and "${{" not in lock_gate["run"]


def test_lock_excludes_the_intel_macos_fork() -> None:
    lock = tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))
    environments = lock.get("supported-markers", [])
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    declared = pyproject["tool"]["uv"]["environments"]
    assert "sys_platform == 'darwin' and platform_machine == 'arm64'" in declared
    assert environments, "uv.lock does not record the declared environments"
    versions: dict[str, set[str]] = {}
    for package in lock["package"]:
        versions.setdefault(package["name"], set()).add(package.get("version", ""))
    # One shap per Python generation, one pandas, one nixtla.
    assert len(versions["pandas"]) == 1
    assert len(versions["nixtla"]) == 1
    # Nothing in the lock resolves for Intel macOS any more.
    assert not any("x86_64" in marker and "darwin" in marker for marker in environments)
