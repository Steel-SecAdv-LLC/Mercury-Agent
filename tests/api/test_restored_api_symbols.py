# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Restored ``api`` symbols removed in 7ab59916 as uncalled.

The released ``rate_limit_middleware`` raised ``HTTPException`` from an
``http`` middleware, where FastAPI's exception handlers do not reach, so an
over-limit request received a 500; the 429 test fails on it.
"""

from __future__ import annotations

import os

import pytest

from omni_mercury_engine.api import auth
from omni_mercury_engine.api.maintenance import (
    maintenance_disabled_for_tests,
    maintenance_interval_seconds,
)


def test_over_limit_request_gets_429_with_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setattr(
        auth, "_rate_limiter", auth.RequestRateLimiter(requests_per_minute=60, burst_size=1)
    )
    app = fastapi.FastAPI()
    app.middleware("http")(auth.rate_limit_middleware)

    @app.get("/probe")
    def probe() -> dict[str, bool]:
        return {"ok": True}

    client = TestClient(app, raise_server_exceptions=False)
    first = client.get("/probe")
    assert first.status_code == 200
    assert first.headers["X-RateLimit-Limit"] == "60"
    denied = client.get("/probe")
    assert denied.status_code == 429
    assert denied.json() == {"detail": "Rate limit exceeded"}
    assert int(denied.headers["Retry-After"]) >= 1
    assert denied.headers["X-RateLimit-Remaining"] == "0"


def test_auth_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        auth.AuthProvider()  # type: ignore[abstract]

    class Static(auth.AuthProvider):
        async def authenticate(self, credentials: object) -> auth.User:
            raise auth.AuthenticationError("no")

        async def validate_token(self, token: str) -> auth.User | None:
            return None

    assert isinstance(Static(), auth.AuthProvider)


@pytest.mark.parametrize("previous", [None, "120"])
def test_maintenance_disabled_for_tests_restores_the_interval(
    monkeypatch: pytest.MonkeyPatch, previous: str | None
) -> None:
    name = "MERCURY_MAINTENANCE_INTERVAL_SECONDS"
    if previous is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, previous)
    with maintenance_disabled_for_tests():
        assert maintenance_interval_seconds() == 0.0
    assert os.environ.get(name) == previous
