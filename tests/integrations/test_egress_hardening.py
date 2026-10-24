# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Credential-safe redirects, single-session connect, bounded alert backlog."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web
from aiohttp.test_utils import TestServer

from omni_mercury_engine.api import alert_egress
from omni_mercury_engine.api.routes.export import DetectionRecord
from omni_mercury_engine.integrations.cross_platform_hub import (
    AnomalyEvent,
    HTTPPlatformAdapter,
    PlatformConfig,
    PlatformType,
)


def _event() -> AnomalyEvent:
    return AnomalyEvent(
        event_id="e1",
        timestamp=datetime.now(UTC),
        source="mercury-agent",
        severity="high",
        score=0.9,
        is_anomaly=True,
    )


async def test_datadog_key_never_follows_a_redirect() -> None:
    """A 3xx from the sink is a failed delivery, not a hop carrying DD-API-KEY."""
    seen: list[dict[str, str]] = []

    async def other_host(request: web.Request) -> web.Response:
        seen.append(dict(request.headers))
        return web.json_response({"status": "ok"})

    target_app = web.Application()
    target_app.router.add_route("*", "/{tail:.*}", other_host)
    target = TestServer(target_app, host="127.0.0.1")
    await target.start_server()
    target_url = str(target.make_url("/stolen"))

    async def redirector(request: web.Request) -> web.Response:
        if request.path == "/api/v1/validate":
            return web.json_response({"valid": True})
        raise web.HTTPTemporaryRedirect(target_url)

    sink_app = web.Application()
    sink_app.router.add_route("*", "/{tail:.*}", redirector)
    sink = TestServer(sink_app, host="127.0.0.1")
    await sink.start_server()
    try:
        adapter = HTTPPlatformAdapter(
            PlatformConfig(
                platform_type=PlatformType.DATADOG,
                name="dd",
                endpoint=str(sink.make_url("")).rstrip("/"),
                auth_type="api_key",
                credentials={"api_key": "dd-secret"},
            )
        )
        assert await adapter.send_event(_event()) is False
        assert await adapter.send_batch([_event()]) == 0
        await adapter.disconnect()
    finally:
        await sink.close()
        await target.close()
    assert seen == []


async def test_concurrent_first_sends_open_one_session(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[str] = []

    async def handle(request: web.Request) -> web.Response:
        received.append(request.path)
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handle)
    server = TestServer(app, host="127.0.0.1")
    await server.start_server()
    created: list[Any] = []
    real_session = aiohttp.ClientSession

    def _counting_session(*args: Any, **kwargs: Any) -> Any:
        session = real_session(*args, **kwargs)
        created.append(session)
        return session

    monkeypatch.setattr(aiohttp, "ClientSession", _counting_session)
    try:
        adapter = HTTPPlatformAdapter(
            PlatformConfig(
                platform_type=PlatformType.CUSTOM,
                name="generic",
                endpoint=str(server.make_url("")).rstrip("/"),
            )
        )
        results = await asyncio.gather(*(adapter.send_event(_event()) for _ in range(8)))
        await adapter.disconnect()
    finally:
        await server.close()
    assert all(results)
    assert len(created) == 1
    assert all(session.closed for session in created)
    assert received.count("/api/events") == 8


def _record(i: int) -> DetectionRecord:
    return DetectionRecord(
        detection_id=f"d{i}",
        timestamp=datetime.now(),
        user_id="u",
        method="statistical",
        data_hash="h",
        anomaly_count=1,
        max_score=0.9,
        is_batch=False,
        sensitivity=0.5,
    )


async def test_alert_backlog_is_bounded_and_drops_are_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = asyncio.Event()

    async def _slow_dispatch(self: Any, record: DetectionRecord) -> dict[str, bool]:
        await release.wait()
        return {}

    monkeypatch.setenv(
        alert_egress.ALERT_SINKS_ENV,
        '[{"name": "x", "platform": "custom", "endpoint": "http://127.0.0.1:9"}]',
    )
    monkeypatch.setenv(alert_egress.ALERT_MAX_PENDING_ENV, "3")
    monkeypatch.setattr(alert_egress.AlertEgress, "dispatch", _slow_dispatch)
    alert_egress.reset_alert_egress()
    try:
        tasks = [alert_egress.schedule_alert(_record(i)) for i in range(10)]
        assert sum(t is not None for t in tasks) == 3
        assert len(alert_egress._pending) == 3
        egress = alert_egress.get_alert_egress()
        assert egress is not None and egress.stats["dropped_backpressure"] == 7
        release.set()
        await alert_egress.shutdown_alert_egress()
        assert not alert_egress._pending
        # Capacity frees up once in-flight deliveries finish.
        assert alert_egress.schedule_alert(_record(99)) is not None
        await alert_egress.shutdown_alert_egress()
    finally:
        alert_egress.reset_alert_egress()


def test_max_pending_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    for raw, expected in (("", 256), ("10", 10), ("0", 256), ("-4", 256), ("x", 256)):
        monkeypatch.setenv(alert_egress.ALERT_MAX_PENDING_ENV, raw)
        assert alert_egress._max_pending() == expected
