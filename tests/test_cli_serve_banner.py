# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``mercury-agent serve`` lists the endpoints the server will actually answer.

The banner named ``/docs``, ``/redoc``, ``/health`` and ``/api/v1`` and nothing
else, so the Mercury Console added in 3.0.0 and the Prometheus endpoint could
not be found from the terminal. It now lists ``/metrics`` always and the
browser pages only when ``MERCURY_FRONTEND_ENABLED=true`` mounts them, so it
never advertises a path that answers 404.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("fastapi", reason="the serve command needs the [api] extra")
pytest.importorskip("uvicorn", reason="the serve command needs the [api] extra")

from click.testing import CliRunner

from omni_mercury_engine.cli import _serve_endpoints, main


@pytest.fixture()
def uvicorn_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace ``uvicorn.run`` so ``serve`` returns instead of listening."""
    import uvicorn

    calls: list[dict[str, Any]] = []

    def fake_run(app: str, **kwargs: Any) -> None:
        calls.append({"app": app, **kwargs})

    monkeypatch.setattr(uvicorn, "run", fake_run)
    return calls


def test_banner_lists_metrics_and_omits_pages_that_are_not_mounted(
    monkeypatch: pytest.MonkeyPatch, uvicorn_calls: list[dict[str, Any]]
) -> None:
    monkeypatch.delenv("MERCURY_FRONTEND_ENABLED", raising=False)
    result = CliRunner().invoke(main, ["serve", "--port", "8123"])
    assert result.exit_code == 0, result.output
    assert "/metrics    - Prometheus metrics, including API router state" in result.output
    assert "/health     - Health check endpoint" in result.output
    assert "/console" not in result.output and "/login" not in result.output
    assert uvicorn_calls == [
        {
            "app": "omni_mercury_engine.api.server:app",
            "host": "127.0.0.1",
            "port": 8123,
            "workers": 1,
            "reload": False,
            "log_level": "info",
        }
    ]


def test_banner_lists_the_console_when_the_frontend_is_enabled(
    monkeypatch: pytest.MonkeyPatch, uvicorn_calls: list[dict[str, Any]]
) -> None:
    monkeypatch.setenv("MERCURY_FRONTEND_ENABLED", "true")
    result = CliRunner().invoke(main, ["serve"])
    assert result.exit_code == 0, result.output
    assert "/console    - Mercury Console (browser front end)" in result.output
    assert "/login      - Account sign-in and dashboard pages" in result.output
    assert len(uvicorn_calls) == 1


def test_every_listed_path_is_served(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each row names a route of the running application (``...`` is a prefix)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from omni_mercury_engine.api.frontend import register_frontend
    from omni_mercury_engine.api.server import app

    monkeypatch.setenv("MERCURY_FRONTEND_ENABLED", "true")
    pages = FastAPI()
    register_frontend(pages)
    api_client, page_client = TestClient(app), TestClient(pages)
    for path, _purpose in _serve_endpoints():
        if path.endswith("..."):
            prefix = path.removesuffix("...")
            assert any(route.startswith(prefix) for route in app.openapi()["paths"]), path
        elif path in {"/console", "/login"}:
            assert page_client.get(path).status_code == 200, path
        else:
            assert api_client.get(path).status_code == 200, path
