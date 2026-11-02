# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A 429 from the API carries the limiter's real Retry-After, not a flat 60.

``RateLimiter`` computes the seconds until the bucket holds one whole token
again, but the middleware threw that away and wrote ``Retry-After: 60`` on
every denial. The console paces its Mass Run on that header, so a burst of
45 capabilities stalled a full minute each time the burst of 20 drained even
though the next token arrived 0.6 s later.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from omni_mercury_engine.api.server import RateLimitMiddleware


def _app(rpm: int, burst: int) -> TestClient:
    app = FastAPI()

    @app.get("/api/v1/ping")
    def ping() -> dict[str, bool]:
        return {"ok": True}

    app.add_middleware(RateLimitMiddleware, requests_per_minute=rpm, burst_size=burst)
    return TestClient(app)


def test_denial_reports_the_shortfall_wait_not_sixty_seconds() -> None:
    client = _app(rpm=600, burst=3)  # ten tokens a second: the next one is 0.1 s away
    for _ in range(3):
        assert client.get("/api/v1/ping").status_code == 200
    denied = client.get("/api/v1/ping")
    assert denied.status_code == 429
    assert denied.headers["Retry-After"] == "1"  # 0.1 s rounded up to the one-second floor
    assert denied.headers["X-RateLimit-Remaining"] == "0"
    assert denied.headers["X-RateLimit-Limit"] == "600"


def test_slow_refill_reports_the_longer_wait() -> None:
    client = _app(rpm=2, burst=1)  # one token every 30 s
    assert client.get("/api/v1/ping").status_code == 200
    denied = client.get("/api/v1/ping")
    assert denied.status_code == 429
    assert 1 <= int(denied.headers["Retry-After"]) <= 30


def test_allowed_responses_carry_the_budget_headers() -> None:
    client = _app(rpm=100, burst=20)
    ok = client.get("/api/v1/ping")
    assert ok.status_code == 200
    assert ok.headers["X-RateLimit-Limit"] == "100"
    assert int(ok.headers["X-RateLimit-Remaining"]) == 19
    assert "Retry-After" not in ok.headers
