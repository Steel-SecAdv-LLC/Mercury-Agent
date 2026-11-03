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
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


def _lock(*packages: tuple[str, ...], supported: list[str] | None = None) -> str:
    body = ["version = 1", 'requires-python = ">=3.11"']
    if supported is not None:
        body.append("supported-markers = [" + ", ".join(f'"{m}"' for m in supported) + "]")
    body += ["", "[[package]]", 'name = "mercury-agent-global"', 'source = { editable = "." }', ""]
    for name, version, *markers in packages:
        body += ["[[package]]", f'name = "{name}"', f'version = "{version}"']
        if markers:
            body.append("resolution-markers = [" + ", ".join(f'"{m}"' for m in markers) + "]")
        body.append("")
    return "\n".join(body)


BASE = _lock(("pandas", "3.1.0"), ("h5py", "3.14.0"), ("h5py", "3.16.0"), ("pytz", "2026.6"))


def test_a_lower_version_is_a_downgrade() -> None:
    head = _lock(("pandas", "2.3.3"), ("h5py", "3.16.0"))
    found = downgrades.undeclared_downgrades(BASE, head)
    assert [(c.name, c.old, c.new) for c in found] == [("pandas", "3.1.0", "2.3.3")]


def test_a_downgrade_on_one_fork_is_caught_while_another_fork_keeps_the_version() -> None:
    """The case a highest-version comparison missed: the mainstream fork drops."""
    head = _lock(
        ("pandas", "2.3.3", "python_full_version < '3.15'"),
        ("pandas", "3.1.0", "python_full_version >= '3.15'"),
        ("h5py", "3.16.0"),
        ("pytz", "2026.6"),
    )
    found = downgrades.undeclared_downgrades(BASE, head)
    assert [(c.name, c.old, c.new) for c in found] == [("pandas", "3.1.0", "2.3.3")]
    assert all(not env.startswith("py3.15") for env in found[0].environments)
    assert len(found[0].environments) == 30  # 5 Python points x 6 platforms


def test_an_environment_dropped_from_the_lock_is_not_compared() -> None:
    base = _lock(
        ("torch", "2.2.2", "sys_platform == 'darwin' and platform_machine == 'x86_64'"),
        ("torch", "2.15.0", "sys_platform != 'darwin' or platform_machine != 'x86_64'"),
    )
    head = _lock(
        ("torch", "2.15.0"),
        supported=[
            "sys_platform != 'darwin'",
            "sys_platform == 'darwin' and platform_machine == 'arm64'",
        ],
    )
    assert downgrades.compare(base, head) == []


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


def test_a_declaration_names_exactly_one_transition() -> None:
    head = _lock(("pandas", "2.3.3"), ("h5py", "3.16.0"), ("pytz", "2026.6"))
    exact = "pandas 3.1.0 -> 2.3.3: nixtla 0.9.0 needs pandas<3"
    assert downgrades.undeclared_downgrades(BASE, head, exact) == []
    wrong_target = "pandas 3.1.0 -> 2.2.0: some reason"
    assert downgrades.undeclared_downgrades(BASE, head, wrong_target)
    no_reason = "pandas 3.1.0 -> 2.3.3:"
    assert downgrades.undeclared_downgrades(BASE, head, no_reason)


def _git(repo: Path, *args: str, author: str = "A Person") -> None:
    subprocess.run(
        ["git", "-c", f"user.name={author}", "-c", "user.email=a@example.org", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_only_real_trailers_on_non_bot_commits_declare(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "base")
    trailer = "Lock-Downgrade: pandas 3.1.0 -> 2.3.3: a reason"
    # A person's commit with the trailer in its final trailer block counts.
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", f"change\n\n{trailer}\nSigned-off-by: A")
    # The same line in the body (release notes quoted above the trailers) does not.
    _git(
        tmp_path,
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        f"bump\n\n{trailer.replace('pandas', 'numpy')}\n\nSigned-off-by: A",
    )
    # A bot's commit does not, even as a real trailer.
    _git(
        tmp_path,
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "bot\n\nLock-Downgrade: scipy 2 -> 1: x",
        author="dependabot[bot]",
    )
    values = downgrades.trailer_declarations("HEAD~3..HEAD", cwd=tmp_path)
    assert downgrades.declared(values) == {("pandas", "3.1.0", "2.3.3")}


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


def _stub(
    monkeypatch: pytest.MonkeyPatch, available: list[str], solvable: dict[str, str]
) -> list[dict[str, str]]:
    """Stub PyPI and the resolver: ``solvable`` maps 'name==version' to a status."""
    calls: list[dict[str, str]] = []
    monkeypatch.setattr(replay, "releases", lambda name, floor: [Version(v) for v in available])

    def fake_lock(pins: dict[str, Any], pyproject_text: str, lock_text: str) -> Any:
        key = {n: str(v) for n, (_, v) in pins.items()}
        calls.append(key)
        status = solvable.get("+".join(f"{n}=={v}" for n, v in sorted(key.items())), "ok")
        return status, "" if status == "ok" else "pandas 3.1.0 -> 2.3.3", None

    monkeypatch.setattr(replay, "lock_with_pins", fake_lock)
    return calls


def _run(name: str, locked: str, ignores: dict[str, Any]) -> Any:
    return replay.simulate(name, [f"{name}>=0"], Version(locked), Version("3.11"), ignores, "", "")


def test_an_ignored_newest_release_falls_back_to_the_next_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With 0.8-0.9 ignored, Dependabot proposes 0.7.4, which is what it replays."""
    calls = _stub(
        monkeypatch,
        ["0.9.0", "0.8.0", "0.7.4"],
        {"nixtla==0.9.0": "downgrades", "nixtla==0.7.4": "downgrades"},
    )
    ignores = {"nixtla": replay.Ignore([replay.SpecifierSet(">=0.8.0,<0.10.0")], False)}
    outcome = _run("nixtla", "0.7.2", ignores)
    assert (outcome.status, str(outcome.candidate)) == ("downgrades", "0.7.4")
    assert "Dependabot proposes 0.7.4" in outcome.detail
    assert {"nixtla": "0.7.4"} in calls


def test_every_newer_release_ignored_is_reported_not_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub(monkeypatch, ["0.9.0", "0.7.4"], {"nixtla==0.9.0": "downgrades"})
    ignores = {"nixtla": replay.Ignore([replay.SpecifierSet(">=0.7.3,<0.10.0")], False)}
    assert _run("nixtla", "0.7.2", ignores).status == "ignored"


def test_an_ignore_whose_release_now_updates_cleanly_is_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub(monkeypatch, ["3.16.0"], {})
    ignores = {"h5py": replay.Ignore([replay.SpecifierSet(">=3.15.0")], False)}
    outcome = _run("h5py", "3.14.0", ignores)
    assert outcome.status == "stale-ignore" and outcome.status in replay.FAILING


def test_a_semver_major_ignore_skips_majors_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, ["3.0.0", "2.5.0"], {})
    outcome = _run("torch", "2.4.0", {"torch": replay.Ignore([], True)})
    assert (outcome.status, str(outcome.candidate)) == ("ok", "2.5.0")


def test_groups_take_members_in_order_and_by_update_type() -> None:
    groups = replay.load_groups(REPO / ".github" / "dependabot.yml")
    names = [g.name for g in groups]
    assert names.index("ml-dependencies") < names.index("python-minor-and-patch")
    assert "security-updates" not in names  # applies to security updates only
    outcomes = [
        replay.Outcome("numpy", Version("2.4.6"), Version("2.5.0"), "ok"),
        replay.Outcome("anthropic", Version("2.3.0"), Version("2.4.0"), "ok"),
        replay.Outcome("openai", Version("3.42.1"), Version("3.43.0"), "ok"),
        replay.Outcome("flask", Version("2.0.0"), Version("3.0.0"), "ok"),  # major
        replay.Outcome("h5py", Version("3.14.0"), Version("3.16.0"), "ignored"),
    ]
    members = replay.group_members(outcomes, groups)
    assert [m.name for m in members["ml-dependencies"]] == ["numpy"]
    assert [m.name for m in members["python-minor-and-patch"]] == ["anthropic", "openai"]
    assert all(m.name not in {"flask", "h5py"} for ms in members.values() for m in ms)


def test_update_type_classes() -> None:
    assert replay.update_type(Version("2.4.6"), Version("3.0.0")) == "major"
    assert replay.update_type(Version("0.7.2"), Version("0.9.0")) == "minor"
    assert replay.update_type(Version("1.2.3"), Version("1.2.4")) == "patch"


def test_the_pr_replay_runs_only_when_dependabot_inputs_change() -> None:
    workflow = yaml.safe_load(
        (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )
    hardening = next(j for j in workflow["jobs"].values() if j.get("name") == "Workflow Hardening")
    step = next(s for s in hardening["steps"] if "simulate_dependabot_uv" in str(s.get("run")))
    assert step["if"] == "github.event_name == 'pull_request'"
    assert "-- pyproject.toml uv.lock .github/dependabot.yml" in step["run"]
    weekly = yaml.safe_load(
        (REPO / ".github" / "workflows" / "dependabot-replay.yml").read_text(encoding="utf-8")
    )
    triggers = weekly.get("on", weekly.get(True))
    assert triggers["schedule"] and "workflow_dispatch" in triggers
    runs = [str(s.get("run") or "") for s in weekly["jobs"]["replay"]["steps"]]
    assert any("scripts/simulate_dependabot_uv.py" in r for r in runs)


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
    run = lock_gate["run"]
    # Declarations come only from the PR's own commits: no PR body, and no
    # fallback that would read main's history.
    assert "PR_BODY" not in lock_gate["env"] and "body" not in run
    assert '--commit-range "${BASE_SHA}..${HEAD_SHA}"' in run
    assert '--shallow-exclude="${BASE_REF}"' in run and "||" not in run and "${{" not in run


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
