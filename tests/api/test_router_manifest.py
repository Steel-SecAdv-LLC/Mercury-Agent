# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The server records which routers it mounted and exports that on ``/metrics``.

``api/server.py`` attaches each router inside a guard that tolerates an
``ImportError``. A router dropped that way used to leave one warning line while
``/health`` reported ``healthy``: on an install without the AMA backend the
whole account API (20 paths) was missing, the browser pages that drive it were
still served, and no probe or metric said so. These tests pin the three parts
of the fix:

* every attachment is recorded as mounted, disabled or unavailable;
* the record is exported as ``omni_mercury_api_router_state`` and ``/health``
  stays the minimal probe it was;
* a server asked to serve the browser frontend without the account API refuses
  to start.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

import pytest

pytest.importorskip("fastapi", reason="the API server needs the [api] extra")

from fastapi.testclient import TestClient

from omni_mercury_engine.api import router_manifest as manifest
from omni_mercury_engine.api.router_manifest import (
    METRIC_NAME,
    RouterState,
    mounting,
    render_prometheus,
    router_manifest,
    unavailable_routers,
)

_LOG = logging.getLogger("tests.router_manifest")

#: Every surface ``api/server.py`` attaches, in attachment order.
_SERVER_ROUTERS = (
    "batch",
    "models",
    "export",
    "detection",
    "accounts",
    "hazard",
    "assessment",
    "agent_notice",
    "source",
    "catalog",
    "frontend",
    "voice",
    "metrics",
    "decoys",
)

#: Surfaces that configuration may leave off; every other one must be mounted.
_OPT_IN = frozenset({"frontend", "decoys"})


@pytest.fixture()
def private_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give a test its own empty manifest and restore the server's afterwards."""
    monkeypatch.setattr(manifest, "_RECORDS", {})


@pytest.mark.usefixtures("private_manifest")
class TestMounting:
    """The guard records exactly one state per attachment."""

    def test_a_clean_body_is_recorded_as_mounted(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger=_LOG.name), mounting("alpha", "Alpha", _LOG):
            pass
        (record,) = router_manifest()
        assert (record.name, record.label, record.state, record.detail) == (
            "alpha",
            "Alpha",
            RouterState.MOUNTED,
            None,
        )
        assert "Alpha registered" in caplog.text

    def test_an_import_error_is_recorded_and_suppressed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger=_LOG.name), mounting("beta", "Beta", _LOG):
            raise ModuleNotFoundError("No module named 'ama_cryptography'")
        (record,) = router_manifest()
        assert record.state is RouterState.UNAVAILABLE
        assert record.detail == "ModuleNotFoundError: No module named 'ama_cryptography'"
        assert unavailable_routers() == (record,)
        assert "Beta not available: No module named 'ama_cryptography'" in caplog.text

    def test_a_disabled_surface_is_not_a_fault(self) -> None:
        with mounting("gamma", "Gamma", _LOG) as mount:
            mount.disable("GAMMA_ENABLED is not true")
        (record,) = router_manifest()
        assert record.state is RouterState.DISABLED
        assert record.detail == "GAMMA_ENABLED is not true"
        assert unavailable_routers() == ()

    def test_any_other_exception_stops_the_process(self) -> None:
        with pytest.raises(ZeroDivisionError), mounting("delta", "Delta", _LOG):
            _ = 1 / 0
        assert router_manifest() == ()

    @pytest.mark.parametrize("name", ["", "Accounts", "a-b", "1st", 'x"} 1\n', "a b"])
    def test_a_name_that_is_not_a_plain_label_is_refused(self, name: str) -> None:
        with pytest.raises(ValueError, match="router name"), mounting(name, "Bad", _LOG):
            pass
        assert router_manifest() == ()

    def test_a_long_failure_reason_is_truncated(self) -> None:
        with mounting("epsilon", "Epsilon", _LOG):
            raise ImportError("x" * 5000)
        (record,) = router_manifest()
        assert record.detail is not None and len(record.detail) == 300

    def test_remounting_replaces_the_earlier_record(self) -> None:
        with mounting("zeta", "Zeta", _LOG):
            raise ImportError("first attempt")
        with mounting("zeta", "Zeta", _LOG):
            pass
        (record,) = router_manifest()
        assert record.state is RouterState.MOUNTED


@pytest.mark.usefixtures("private_manifest")
class TestExposition:
    """The manifest renders as a Prometheus state set."""

    def test_nothing_recorded_renders_nothing(self) -> None:
        assert render_prometheus() == []

    def test_each_router_has_one_series_per_state_and_one_is_set(self) -> None:
        with mounting("up", "Up", _LOG):
            pass
        with mounting("off", "Off", _LOG) as mount:
            mount.disable("off by configuration")
        with mounting("gone", "Gone", _LOG):
            raise ImportError("missing")
        lines = render_prometheus()
        assert lines[0].startswith(f"# HELP {METRIC_NAME} ")
        assert lines[1] == f"# TYPE {METRIC_NAME} gauge"
        series = lines[2:]
        assert len(series) == 3 * len(RouterState)
        for name, current in (("up", "mounted"), ("off", "disabled"), ("gone", "unavailable")):
            for state in RouterState:
                expected = 1 if state.value == current else 0
                assert (
                    f'{METRIC_NAME}{{router="{name}",state="{state.value}"}} {expected}' in series
                )

    def test_the_failure_reason_is_not_exported(self) -> None:
        # The reason can hold a filesystem path; it belongs in the log, which
        # the operator reads, not in a label any scraper stores.
        with mounting("gone", "Gone", _LOG):
            raise ImportError("/srv/private/path is unreadable")
        assert "/srv/private/path" not in "\n".join(render_prometheus())


class TestServedApplication:
    """The real application, on an install with every dependency present."""

    def test_every_surface_is_recorded_in_attachment_order(self) -> None:
        import omni_mercury_engine.api.server  # noqa: F401 - populates the manifest

        assert tuple(record.name for record in router_manifest()) == _SERVER_ROUTERS

    def test_nothing_is_unavailable_and_only_opt_in_surfaces_are_off(self) -> None:
        import omni_mercury_engine.api.server  # noqa: F401 - populates the manifest

        assert unavailable_routers() == ()
        disabled = {r.name for r in router_manifest() if r.state is RouterState.DISABLED}
        assert disabled <= _OPT_IN

    def test_metrics_exports_the_state_of_every_router(self) -> None:
        from omni_mercury_engine.api.server import app

        text = TestClient(app).get("/metrics").text
        assert f"# TYPE {METRIC_NAME} gauge" in text
        for name in _SERVER_ROUTERS:
            assert f'{METRIC_NAME}{{router="{name}",state="unavailable"}} 0' in text
        assert f'{METRIC_NAME}{{router="accounts",state="mounted"}} 1' in text

    def test_health_does_not_carry_the_manifest(self) -> None:
        from omni_mercury_engine.api.server import app

        body = TestClient(app).get("/health").json()
        assert set(body) <= {"status", "version", "uptime_seconds"}


def _import_server_with_accounts_blocked(frontend: bool) -> subprocess.CompletedProcess[str]:
    """Import the server in a fresh interpreter whose account router cannot load."""
    code = (
        "import sys\n"
        "sys.modules['omni_mercury_engine.api.routes.accounts'] = None\n"
        "import omni_mercury_engine.api.server as server\n"
        "from omni_mercury_engine.api.router_manifest import router_manifest\n"
        "from fastapi.testclient import TestClient\n"
        "states = {r.name: r.state.value for r in router_manifest()}\n"
        "print('ACCOUNTS=' + states['accounts'])\n"
        "print('FRONTEND=' + states['frontend'])\n"
        "client = TestClient(server.app)\n"
        "print('HEALTH=' + client.get('/health').json()['status'])\n"
        "print(client.get('/metrics').text)\n"
    )
    env = dict(os.environ)
    env["MERCURY_AGENT_ENV"] = "development"
    env["JWT_SECRET_KEY"] = "0123456789abcdef" * 4  # pragma: allowlist secret
    env["MERCURY_FRONTEND_ENABLED"] = "true" if frontend else "false"
    env.pop("MERCURY_DECEPTION_ENABLED", None)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=240,
        check=False,
    )


class TestDegradedBoot:
    """What the server does when a router's import fails."""

    def test_a_dropped_router_is_reported_on_metrics(self) -> None:
        result = _import_server_with_accounts_blocked(frontend=False)
        assert result.returncode == 0, result.stderr[-2000:]
        assert "ACCOUNTS=unavailable" in result.stdout
        assert "FRONTEND=disabled" in result.stdout
        # /health is a liveness probe and stays minimal; the fault is on /metrics.
        assert "HEALTH=healthy" in result.stdout
        assert f'{METRIC_NAME}{{router="accounts",state="unavailable"}} 1' in result.stdout
        assert f'{METRIC_NAME}{{router="accounts",state="mounted"}} 0' in result.stdout
        assert f'{METRIC_NAME}{{router="batch",state="mounted"}} 1' in result.stdout

    def test_the_frontend_is_not_served_over_a_missing_account_api(self) -> None:
        result = _import_server_with_accounts_blocked(frontend=True)
        assert result.returncode != 0
        assert "ACCOUNTS=" not in result.stdout
        assert "MERCURY_FRONTEND_ENABLED=true, but the account API" in result.stderr
        assert "/api/v1/auth" in result.stderr
