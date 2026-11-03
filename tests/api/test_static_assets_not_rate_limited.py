# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The API rate limiter must not break the frontend's own asset loads.

The Mercury Console page references 17 same-origin scripts and stylesheets
and three images, so one page load is 21 requests from one client, and the
default bucket is 100 requests a minute with a burst of 20. Measured against
``mercury-agent serve`` before this guard: 30 concurrent asset requests from
one address answered 10 of them 429, and the next asset request found the
bucket at zero with ``Retry-After: 60``. A browser that reloads the console
within a minute gets a page with its scripts refused.

The budget exists to protect detection compute, not file reads that the
framework serves without touching the engine. Assets under the frontend's
``/static`` mount are therefore exempt; the pages and every API route stay
limited exactly as before.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omni_mercury_engine.api import frontend
from omni_mercury_engine.api.server import RateLimitMiddleware

# The global rate limiter stays live here: these tests assert on it (see
# tests/api/conftest.py, which switches it off for every other API test).
pytestmark = pytest.mark.rate_limited


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A private app: the frontend mount plus a two-token rate limiter."""
    monkeypatch.setenv("OMNI_RATE_LIMIT_ENABLED", "true")
    monkeypatch.delenv("MERCURY_KEYSTORE_PATH", raising=False)
    app = FastAPI()

    @app.get("/api/v1/probe")
    def probe() -> dict[str, bool]:
        return {"ok": True}

    frontend.register_frontend(app)
    # Two-token burst with no meaningful refill inside the test: the third
    # limited request in a row must be refused.
    app.add_middleware(RateLimitMiddleware, requests_per_minute=1, burst_size=2)
    return TestClient(app)


def test_static_assets_are_served_past_the_burst(client: TestClient) -> None:
    """Every console asset answers 200 even when the client's bucket is empty."""
    assets = [
        "/static/console/tokens.css",
        "/static/console/port.css",
        "/static/console/api.js",
        "/static/console/dc-runtime.js",
        "/static/console/template.js",
        "/static/console/logic.js",
        "/static/console/omega-cross.png",
        "/static/mercury.css",
    ]
    statuses = [client.get(path).status_code for path in assets]
    assert statuses == [200] * len(assets), statuses
    assert "X-RateLimit-Limit" not in client.get("/static/console/port.css").headers


def test_api_routes_and_pages_stay_limited(client: TestClient) -> None:
    """The exemption is the asset mount only; the budget still applies elsewhere."""
    statuses = [client.get("/api/v1/probe").status_code for _ in range(3)]
    assert statuses[:2] == [200, 200]
    assert statuses[2] == 429
    # The page itself is one request and is still metered.
    assert client.get("/console").status_code == 429
