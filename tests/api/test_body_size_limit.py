# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The app refuses request bodies over the byte cap, before parsing them.

Before this, no layer bounded a request body: FastAPI's JSON decode and
pydantic validation ran over an unbounded body before any gate (the harm
gate's caps apply only after parsing), and the one existing guard trusted
the Content-Length header, which a chunked request simply omits.  These
tests pin both halves of the middleware: the declared-length fast refusal
and the streamed count that catches chunked and lying-length bodies.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("fastapi", reason="the API server needs the [api] extra")

from omni_mercury_engine.api.server import BodySizeLimitMiddleware


async def _consuming_app(scope: Any, receive: Any, send: Any) -> None:
    """Read the whole body the way FastAPI would, then answer 200."""
    while True:
        message = await receive()
        if message["type"] == "http.request" and not message.get("more_body"):
            break
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


def _run(
    chunks: list[bytes],
    headers: list[tuple[bytes, bytes]],
    *,
    path: str = "/api/v1/x",
    limit: int = 1000,
) -> int:
    middleware = BodySizeLimitMiddleware(_consuming_app)
    middleware.default_limit = limit
    sent: list[dict[str, Any]] = []
    queue = list(chunks)

    async def receive() -> dict[str, Any]:
        body = queue.pop(0)
        return {"type": "http.request", "body": body, "more_body": bool(queue)}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {"type": "http", "path": path, "headers": headers}
    asyncio.run(middleware(scope, receive, send))
    return int(sent[0]["status"])


class TestDeclaredLength:
    def test_oversized_declared_length_is_refused_before_the_body(self) -> None:
        # The receive queue is EMPTY: a refusal that needed the body would
        # crash, so this also proves the refusal happens before any read.
        middleware = BodySizeLimitMiddleware(_consuming_app)
        middleware.default_limit = 1000
        sent: list[dict[str, Any]] = []

        async def receive() -> dict[str, Any]:  # pragma: no cover - must not run
            raise AssertionError("the body was read despite an oversized declared length")

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        scope = {
            "type": "http",
            "path": "/api/v1/x",
            "headers": [(b"content-length", b"5000")],
        }
        asyncio.run(middleware(scope, receive, send))
        assert sent[0]["status"] == 413

    def test_small_declared_length_passes(self) -> None:
        assert _run([b"x" * 100], [(b"content-length", b"100")]) == 200


class TestStreamedCounting:
    def test_chunked_oversize_is_cut_off(self) -> None:
        assert _run([b"x" * 600, b"x" * 600], []) == 413

    def test_chunked_under_cap_passes(self) -> None:
        assert _run([b"x" * 100], []) == 200

    def test_lying_content_length_is_still_cut_off(self) -> None:
        assert _run([b"x" * 2000], [(b"content-length", b"10")]) == 413


class TestScoping:
    def test_upload_surface_keeps_its_own_allowance(self) -> None:
        middleware = BodySizeLimitMiddleware(_consuming_app)
        assert middleware._limit_for("/api/v1/models/anything") == middleware.upload_limit
        assert middleware._limit_for("/api/v1/detect/flagship") == middleware.default_limit
        assert middleware.upload_limit > middleware.default_limit

    def test_non_http_scopes_pass_through(self) -> None:
        middleware = BodySizeLimitMiddleware(_consuming_app)
        passed: list[str] = []

        async def app(scope: Any, receive: Any, send: Any) -> None:
            passed.append(scope["type"])

        middleware.app = app
        asyncio.run(middleware({"type": "websocket"}, None, None))
        assert passed == ["websocket"]


class TestWiring:
    def test_the_app_carries_the_middleware(self) -> None:
        from omni_mercury_engine.api.server import app

        names = [m.cls.__name__ for m in app.user_middleware]
        assert "BodySizeLimitMiddleware" in names
