# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Browser clients can read the headers the console depends on.

A browser may only read a response header that CORS exposes. The API sets
``X-Correlation-ID`` on every response and ``Retry-After`` on every 429, and
the console shows both (a correlation id beside each result, a countdown on a
429), but neither was in ``expose_headers``, so a browser client saw neither.
This pins the exposed set.
"""

from __future__ import annotations

from fastapi.middleware.cors import CORSMiddleware

from omni_mercury_engine.api.server import app

REQUIRED = {
    "X-RateLimit-Limit",
    "X-RateLimit-Remaining",
    "X-RateLimit-Reset",
    "Retry-After",
    "X-Correlation-ID",
}


def test_cors_exposes_rate_limit_retry_after_and_correlation_headers() -> None:
    cors = [m for m in app.user_middleware if m.cls is CORSMiddleware]
    assert cors, "CORS middleware is not registered (no allowed origins configured)"
    exposed = set(cors[0].kwargs["expose_headers"])
    missing = REQUIRED - exposed
    assert not missing, f"browser clients cannot read: {sorted(missing)}"
