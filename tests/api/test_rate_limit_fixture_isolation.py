# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The ``rate_limit_off`` fixture restores the limiter even from an unbuilt stack.

Reviewed defect: patching the environment before the shared app had built its
middleware stack made ``RateLimitMiddleware`` initialise with ``enabled=False``;
the attribute patch then recorded that as the original, and teardown left the
limiter off for every later test. The helper builds the stack first.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="fastapi (optional '[api]' extra) is required")

from omni_mercury_engine.api.server import app
from tests.conftest import _rate_limit_layer, switch_rate_limiter_off

# The global rate limiter stays live here: these tests assert on it (see
# tests/api/conftest.py, which switches it off for every other API test).
pytestmark = pytest.mark.rate_limited


def _enabled() -> bool:
    layer = _rate_limit_layer()
    assert layer is not None, "no RateLimitMiddleware in the built stack"
    return bool(getattr(layer, "enabled", None))


def test_switching_off_an_unbuilt_stack_restores_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNI_RATE_LIMIT_ENABLED", "true")  # the process's own configuration
    app.middleware_stack = None  # as at the start of a session: nothing built yet

    inner = pytest.MonkeyPatch()
    try:
        switch_rate_limiter_off(inner)
        assert _enabled() is False
    finally:
        inner.undo()
    assert _enabled() is True, "teardown left the shared limiter off"


def test_switching_off_a_built_stack_restores_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNI_RATE_LIMIT_ENABLED", "true")
    app.middleware_stack = None
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        client.get("/health")
    assert _enabled() is True

    inner = pytest.MonkeyPatch()
    try:
        switch_rate_limiter_off(inner)
        assert _enabled() is False
    finally:
        inner.undo()
    assert _enabled() is True


def test_switching_off_never_runs_the_application_lifespan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building the stack must not start the app: the lifespan's maintenance sweep runs against whatever stores the process inherited."""
    import contextlib
    from collections.abc import AsyncIterator
    from typing import Any

    monkeypatch.setenv("OMNI_RATE_LIMIT_ENABLED", "true")
    app.middleware_stack = None
    entered: list[str] = []
    original = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def recording(application: Any) -> AsyncIterator[None]:
        entered.append("lifespan")
        async with original(application):
            yield

    monkeypatch.setattr(app.router, "lifespan_context", recording)
    inner = pytest.MonkeyPatch()
    try:
        switch_rate_limiter_off(inner)
        assert _enabled() is False
    finally:
        inner.undo()
    assert entered == [], "switch_rate_limiter_off entered the application lifespan"
    assert _enabled() is True
