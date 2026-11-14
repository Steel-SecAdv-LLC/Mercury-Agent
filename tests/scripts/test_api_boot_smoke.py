# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``scripts/api_boot_smoke.py`` boots the real server and CI runs it at the right moments.

Every lane builds AMA before its first import of the package, so the install a
bare ``[api]`` extra produces was never run. The Core Tests job passes through
that state between ``uv sync`` and the AMA build; the smoke script is run
there with ``--expect refuse`` and again after the build with
``--expect serve``. These tests pin the wiring and run the serving half for
real on this environment.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

if TYPE_CHECKING:
    from types import ModuleType

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "api_boot_smoke.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("api_boot_smoke", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


smoke = _load()


def _core_test_steps() -> list[dict[str, Any]]:
    workflow = yaml.safe_load((_REPO / ".github" / "workflows" / "ci.yml").read_text("utf-8"))
    steps: list[dict[str, Any]] = workflow["jobs"]["core-tests"]["steps"]
    return steps


def _position(steps: list[dict[str, Any]], needle: str, key: str = "run") -> int:
    (index,) = [i for i, step in enumerate(steps) if needle in str(step.get(key) or "")]
    return index


def test_ci_checks_the_refusal_between_the_install_and_the_ama_build() -> None:
    steps = _core_test_steps()
    install = _position(steps, "uv sync --locked --extra ml --extra dev --extra api")
    refuse = _position(steps, "scripts/api_boot_smoke.py --expect refuse")
    build = _position(steps, "./.github/actions/build-ama-cryptography", key="uses")
    serve = _position(steps, "scripts/api_boot_smoke.py --expect serve")
    tests = _position(steps, "Run core tests with coverage", key="name")
    assert install < refuse < build < serve < tests


def test_the_api_extra_is_what_the_refusal_step_runs_on() -> None:
    """The lane installs ``[api]`` and not ``[pqc]`` or ``[all]``, so AMA is absent there."""
    steps = _core_test_steps()
    run = str(steps[_position(steps, "uv sync --locked")]["run"])
    assert "--extra api" in run
    assert "--extra pqc" not in run and "--extra all" not in run


@pytest.mark.parametrize(
    ("expect", "ama_present", "code"),
    [("refuse", True, 2), ("serve", False, 2)],
)
def test_a_mode_run_in_the_wrong_state_fails_its_precondition(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    expect: str,
    ama_present: bool,
    code: int,
) -> None:
    monkeypatch.setattr(
        smoke.importlib.util, "find_spec", lambda name: object() if ama_present else None
    )
    monkeypatch.setattr(smoke, "check_refuse", lambda: pytest.fail("must not run"))
    monkeypatch.setattr(smoke, "check_serve", lambda: pytest.fail("must not run"))
    assert smoke.main(["--expect", expect]) == code
    assert "precondition failed" in capsys.readouterr().out


def test_failed_expectations_are_reported_and_fail_the_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(smoke.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(smoke, "check_serve", lambda: ["routers unavailable: accounts"])
    assert smoke.main(["--expect", "serve"]) == 1
    out = capsys.readouterr().out
    assert "::error title=API boot smoke (serve)::routers unavailable: accounts" in out


def test_router_states_are_read_from_the_series_valued_one() -> None:
    text = (
        'omni_mercury_api_router_state{router="accounts",state="mounted"} 0\n'
        'omni_mercury_api_router_state{router="accounts",state="disabled"} 0\n'
        'omni_mercury_api_router_state{router="accounts",state="unavailable"} 1\n'
        'omni_mercury_api_router_state{router="batch",state="mounted"} 1\n'
    )
    assert dict(smoke._ROUTER_STATE.findall(text)) == {
        "accounts": "unavailable",
        "batch": "mounted",
    }


def test_the_server_boots_with_every_router_mounted() -> None:
    """The real program on this environment: uvicorn, /health, /metrics, OpenAPI, pages."""
    pytest.importorskip("uvicorn", reason="booting the API needs the [api] extra")
    pytest.importorskip("ama_cryptography", reason="the API does not start without AMA")
    assert smoke.check_serve() == []
