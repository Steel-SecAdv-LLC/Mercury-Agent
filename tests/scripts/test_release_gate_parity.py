# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The release gate tests exactly what CI proves, and the release never rewrites the version.

``.github/workflows/release.yml`` re-runs the test suite before it builds,
publishes a container image, cuts a GitHub release and offers PyPI. That gate
is only meaningful if it runs the suite CI's ``ml-tests`` lane runs: a green
``main`` is the evidence a release rests on, and a gate that tests something
else either fails a good release or passes a bad one.

It had drifted on every axis before v3.0.0, the first tag this workflow was
ever going to run on:

* it installed ``pip install -e ".[ml,dev]"`` -- unlocked, so it resolved
  package versions CI never tested, and without ``[api]``, so four security
  test modules errored at collection and pytest stopped before running a
  single test, skipping build, docker, release and publish;
* it ran sequentially with ``--timeout=60`` where CI runs ``-n 4
  --dist=loadgroup --timeout=300``;
* it skipped the real-model tiers (``MERCURY_REQUIRES_HF_MODELS``) and the
  full adversarial-ledger recomputation (``MERCURY_FULL_LEDGER_CHECK``);
* its build job ran an unanchored ``sed`` over ``pyproject.toml`` that
  rewrote mypy's ``python_version``, pylint's ``py-version`` and ruff's
  ``target-version`` to the release version inside the shipped sdist.

Two further properties are pinned because the release workflow is the most
privileged code path in the repository: it can push tags, publish packages
and mint OIDC credentials.

* No job restores a cache. ml-tests caches the uv store and HuggingFace models
  for speed; a release job that restored them would run on inputs a previous
  run wrote, and the docker job used to restore and write the GHA layer cache
  for the very image it publishes.
* Every job holds only the token scopes it uses. The workflow used to grant
  contents/packages/id-token write to all of them, so jobs that only read code
  carried a token that could publish.

These tests pin the gate to the lane, so the next drift fails CI on the pull
request that introduces it rather than on release day.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

_REPO = Path(__file__).resolve().parents[2]
_CI = _REPO / ".github" / "workflows" / "ci.yml"
_RELEASE = _REPO / ".github" / "workflows" / "release.yml"

#: Pytest options that decide what is tested or whether it passes. Reporting
#: options (``-v``, ``--cov-report=...``) are free to differ.
_DECISIVE_FLAGS = ("-n", "--dist", "--timeout", "--cov", "--cov-fail-under")


def _load(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded


def _job_steps(workflow: dict[str, Any], job: str) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = workflow["jobs"][job]["steps"]
    return steps


def _step(steps: list[dict[str, Any]], *, uses: str = "", name: str = "") -> dict[str, Any]:
    for step in steps:
        if uses and str(step.get("uses", "")).startswith(uses):
            return step
        if name and step.get("name", "").startswith(name):
            return step
    raise AssertionError(f"no step with uses={uses!r} name={name!r}")


def _ml_lane() -> list[dict[str, Any]]:
    return _job_steps(_load(_CI), "ml-tests")


def _release_gate() -> list[dict[str, Any]]:
    return _job_steps(_load(_RELEASE), "test")


def _pytest_options(run: str) -> dict[str, str]:
    """``{option: value}`` for the decisive options of the ``pytest tests/`` call."""
    joined = re.sub(r"\\\n\s*", " ", run)
    call = next(line for line in joined.splitlines() if "pytest tests/" in line)
    tokens = shlex.split(call.split("pytest", 1)[1].split(">", 1)[0])
    options: dict[str, str] = {}
    for index, token in enumerate(tokens):
        name, _, value = token.partition("=")
        if name not in _DECISIVE_FLAGS:
            continue
        if not value and index + 1 < len(tokens) and not tokens[index + 1].startswith("-"):
            value = tokens[index + 1]
        options[name] = value
    return options


def test_the_gate_installs_the_locked_closure_ci_tests() -> None:
    lane = _step(_ml_lane(), name="Install dependencies")["run"]
    gate = _step(_release_gate(), name="Install dependencies")["run"]
    sync = next(line.strip() for line in lane.splitlines() if line.strip().startswith("uv sync"))
    assert sync in gate, (
        f"the release gate does not install CI's locked closure ({sync!r}); it would "
        "test package versions and extras the ml-tests lane never proved"
    )
    assert "pip install -e" not in gate, "an unlocked pip install has crept back into the gate"


@pytest.mark.parametrize(
    "uses",
    [
        "actions/setup-python@",
        "./.github/actions/build-ama-cryptography",
    ],
)
def test_shared_setup_steps_are_identical(uses: str) -> None:
    lane = _step(_ml_lane(), uses=uses)
    gate = _step(_release_gate(), uses=uses)
    assert gate["uses"] == lane["uses"], f"{uses} pinned to a different revision than CI"
    assert gate.get("with", {}) == lane.get(
        "with", {}
    ), f"{uses} configured differently from CI: release {gate.get('with')} vs ci {lane.get('with')}"


def test_the_gate_uses_cis_uv_pin_without_its_cache() -> None:
    """Same uv revision and version as ml-tests; the cache is the one deliberate difference."""
    lane = _step(_ml_lane(), uses="astral-sh/setup-uv@")
    gate = _step(_release_gate(), uses="astral-sh/setup-uv@")
    assert gate["uses"] == lane["uses"], "setup-uv pinned to a different revision than CI"
    assert gate["with"]["version"] == lane["with"]["version"], "a different uv version than CI"
    assert gate["with"].get("enable-cache") is False, (
        "the release gate restores the uv cache; a release must not run on inputs a "
        "previous run wrote"
    )


def test_the_release_workflow_restores_no_cache() -> None:
    """No job in the release path may restore a cache another run wrote.

    A tag run can restore cache entries main wrote, so a restored cache is an
    unreviewed input to the gate, and in the docker job it was an input to the
    published image itself (``cache-from: type=gha``). Every release job builds
    from pinned, hash-verified sources instead.
    """
    offenders: list[str] = []
    for job, spec in _load(_RELEASE)["jobs"].items():
        for step in spec.get("steps", []):
            uses = str(step.get("uses", ""))
            options = step.get("with", {}) or {}
            if uses.startswith("actions/cache"):
                offenders.append(f"{job}: {uses}")
            if options.get("enable-cache") not in (None, False):
                offenders.append(f"{job}: {uses} enable-cache={options['enable-cache']}")
            for key in ("cache-from", "cache-to"):
                if options.get(key):
                    offenders.append(f"{job}: {uses} {key}={options[key]}")
    assert not offenders, "release jobs restore or write caches:\n  " + "\n  ".join(offenders)


#: Exactly the token scopes each release job uses. Anything wider is reach a
#: compromised step would inherit.
_LEAST_PRIVILEGE = {
    "validate": {"contents": "read"},
    "test": {"contents": "read"},
    "build": {"contents": "read"},
    "docker": {"contents": "read", "packages": "write"},
    "release": {"contents": "write"},
    "publish-pypi": {"contents": "read", "id-token": "write"},
    "notify": {"contents": "read"},
}


def test_every_job_holds_only_the_token_scopes_it_uses() -> None:
    workflow = _load(_RELEASE)
    default = workflow.get("permissions")
    assert default == {
        "contents": "read"
    }, f"the workflow default grants {default}; every job without its own block inherits it"
    effective = {job: spec.get("permissions", default) for job, spec in workflow["jobs"].items()}
    assert set(effective) == set(_LEAST_PRIVILEGE), (
        "release jobs changed; decide each new job's scopes here deliberately: "
        f"{sorted(set(effective) ^ set(_LEAST_PRIVILEGE))}"
    )
    widened = {job: scopes for job, scopes in effective.items() if scopes != _LEAST_PRIVILEGE[job]}
    assert not widened, f"jobs hold scopes beyond what they use: {widened}"


@pytest.mark.parametrize(
    "name",
    ["Dependency license gate", "Prefetch pinned HuggingFace models"],
)
def test_shared_validation_steps_run_the_same_command(name: str) -> None:
    lane = _step(_ml_lane(), name=name)["run"].strip()
    gate = _step(_release_gate(), name=name)["run"].strip()
    assert gate == lane, f"{name}: release runs {gate!r}, CI runs {lane!r}"


def test_the_gate_sets_every_environment_variable_the_lane_sets() -> None:
    lane = _step(_ml_lane(), name="Run full test suite")["env"]
    gate = _step(_release_gate(), name="Run tests")["env"]
    missing = {k: v for k, v in lane.items() if gate.get(k) != v}
    assert not missing, (
        f"the release gate drops or changes {missing}; each of these decides which "
        "tests must run rather than skip"
    )


def test_the_gate_runs_the_suite_with_cis_decisive_options() -> None:
    lane = _pytest_options(_step(_ml_lane(), name="Run full test suite")["run"])
    gate = _pytest_options(_step(_release_gate(), name="Run tests")["run"])
    for option in ("-n", "--dist", "--cov", "--cov-fail-under"):
        assert gate.get(option) == lane.get(
            option
        ), f"pytest {option}: release {gate.get(option)!r} vs ci {lane.get(option)!r}"
    assert int(gate["--timeout"]) >= int(lane["--timeout"]), (
        f"the release gate's per-test timeout ({gate['--timeout']}s) is tighter than "
        f"CI's ({lane['--timeout']}s), so it can fail tests CI legitimately passes"
    )


def test_the_coverage_floor_is_the_same_number_in_both_workflows() -> None:
    ci = _load(_CI)["env"]["COVERAGE_THRESHOLD_FULL"]
    release = _load(_RELEASE)["env"]["COVERAGE_THRESHOLD_FULL"]
    assert release == ci, f"release coverage floor {release} != ci {ci}"


def test_the_release_never_rewrites_pyproject() -> None:
    text = _RELEASE.read_text(encoding="utf-8")
    runs = "\n".join(
        str(step.get("run", ""))
        for job in _load(_RELEASE)["jobs"].values()
        for step in job.get("steps", [])
    )
    assert not re.search(r"sed\s+-i[^\n]*pyproject\.toml", runs), (
        "the release rewrites pyproject.toml in place; an unanchored pattern there "
        "corrupted three tool configs inside the shipped sdist"
    )
    assert "Verify the release version matches the committed version" in text


def _verify_script() -> str:
    step = _step(_job_steps(_load(_RELEASE), "build"), name="Verify the release version")
    body = str(step["run"]).split("<<'PY'\n", 1)[1].rsplit("PY", 1)[0]
    return textwrap.dedent(body)


def _run_verify(workdir: Path, release_version: str) -> subprocess.CompletedProcess[str]:
    script = workdir / "verify.py"
    script.write_text(_verify_script(), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=workdir,
        env={**os.environ, "RELEASE_VERSION": release_version},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def _fake_checkout(root: Path, project: str, fallback: str) -> Path:
    (root / "src" / "omni_mercury_engine").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "x"\nversion = "{project}"\n\n[tool.mypy]\npython_version = "3.12"\n',
        encoding="utf-8",
    )
    (root / "src" / "omni_mercury_engine" / "_version.py").write_text(
        f'_FALLBACK_VERSION = "{fallback}"\n', encoding="utf-8"
    )
    return root


def test_the_committed_version_releases(tmp_path: Path) -> None:
    pytest.importorskip("packaging")
    import tomllib

    real = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    version = real["project"]["version"]
    result = _run_verify(_fake_checkout(tmp_path, version, version), version)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("project", "fallback", "tag"),
    [
        ("3.0.0", "3.0.0", "3.0.1"),  # tag a version nobody committed
        ("3.0.1", "3.0.0", "3.0.1"),  # pyproject bumped, fallback forgotten
        ("3.0.0", "3.0.1", "3.0.1"),  # fallback bumped, pyproject forgotten
    ],
)
def test_a_tag_the_repository_did_not_commit_is_refused(
    tmp_path: Path, project: str, fallback: str, tag: str
) -> None:
    pytest.importorskip("packaging")
    result = _run_verify(_fake_checkout(tmp_path, project, fallback), tag)
    assert (
        result.returncode == 1
    ), f"tag {tag} against pyproject {project} / fallback {fallback} was not refused"
    assert "does not match the committed version" in result.stdout


def test_a_prerelease_tag_matches_its_pep440_spelling(tmp_path: Path) -> None:
    """``v3.1.0-rc1`` must release a commit whose pyproject says ``3.1.0rc1``."""
    pytest.importorskip("packaging")
    result = _run_verify(_fake_checkout(tmp_path, "3.1.0rc1", "3.1.0rc1"), "3.1.0-rc1")
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_verify_step_leaves_pyproject_untouched(tmp_path: Path) -> None:
    pytest.importorskip("packaging")
    root = _fake_checkout(tmp_path, "3.0.0", "3.0.0")
    before = (root / "pyproject.toml").read_bytes()
    _run_verify(root, "3.0.0")
    assert (root / "pyproject.toml").read_bytes() == before
