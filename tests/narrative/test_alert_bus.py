# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recorded detections fan out to live listeners, bounded to their principal.

The console's alert dock and the voice API's docstring both promised
proactive alerts; no server code produced them. The bus is the wire: the
retriever publishes every entry it logs, subscribers hear only what their
account may see, a slow subscriber loses the oldest entries rather than
blocking a detection, and the durable store exposes id cursors so a stream on
one worker sees runs recorded on another.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any, cast

from omni_mercury_engine.narrative import alert_bus
from omni_mercury_engine.narrative.detection_memory import DetectionMemoryStore
from omni_mercury_engine.narrative.retriever import KnowledgeRetriever

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest


def _entry(principal: str | None, flagged: bool = True, **extra: object) -> dict[str, object]:
    detection: dict[str, object] = {
        "detection_id": extra.pop("detection_id", "d1"),
        "method": "univariate",
        "anomaly_detected": flagged,
        "anomaly_count": 1 if flagged else 0,
        "n_samples": 10,
        "anomaly_score": 7.0,
        "threshold": 3.5,
        "severity": 0.5,
        "severity_label": "medium",
    }
    detection.update(extra)
    return {"detection": detection, "domain": "general", "principal": principal, "timestamp": 1.0}


def test_visibility_rule_matches_the_retriever() -> None:
    assert alert_bus.principal_may_see("alice", None)
    assert alert_bus.principal_may_see("alice", "alice")
    assert not alert_bus.principal_may_see("alice", "bob")
    assert alert_bus.principal_may_see(None, "anonymous")
    assert not alert_bus.principal_may_see(None, "alice")


def test_alert_payload_is_built_from_the_record() -> None:
    payload = alert_bus.to_alert(_entry("alice"))
    assert payload["initiative_type"] == "medium_anomaly"
    assert payload["severity"] == 0.5
    assert payload["domain"] == "general"
    assert "1 of 10 samples flagged" in payload["message"]
    assert payload["event"]["detection_id"] == "d1"
    assert alert_bus.to_alert(_entry("alice", flagged=False))["initiative_type"] == "clean_run"


def test_publish_reaches_only_subscribers_allowed_to_see() -> None:
    async def run() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        bus = alert_bus.AlertBus()
        t_alice, q_alice = bus.subscribe("alice")
        t_bob, q_bob = bus.subscribe("bob")
        assert bus.publish(_entry("alice")) == 1
        assert bus.publish(_entry(None)) == 2
        await asyncio.sleep(0)
        alice = [q_alice.get_nowait() for _ in range(q_alice.qsize())]
        bob = [q_bob.get_nowait() for _ in range(q_bob.qsize())]
        bus.unsubscribe(t_alice)
        bus.unsubscribe(t_bob)
        assert bus.subscriber_count == 0
        return alice, bob

    alice, bob = asyncio.run(run())
    assert len(alice) == 2 and len(bob) == 1


def test_publish_from_another_thread_lands_on_the_subscriber_loop() -> None:
    async def run() -> dict[str, Any]:
        bus = alert_bus.AlertBus()
        token, queue = bus.subscribe("alice")
        worker = threading.Thread(target=lambda: bus.publish(_entry("alice")))
        worker.start()
        worker.join()
        got = await asyncio.wait_for(queue.get(), timeout=2.0)
        bus.unsubscribe(token)
        return got

    assert asyncio.run(run())["principal"] == "alice"


def test_slow_subscriber_loses_oldest_not_the_publisher() -> None:
    async def run() -> tuple[int, int, Any]:
        bus = alert_bus.AlertBus()
        token, queue = bus.subscribe(None)
        for i in range(alert_bus.QUEUE_SIZE + 5):
            bus.publish(_entry(None, detection_id=f"d{i}"))
        await asyncio.sleep(0)
        first = queue.get_nowait()
        bus.unsubscribe(token)
        return queue.qsize() + 1, bus.dropped, first["detection"]["detection_id"]

    size, dropped, first_id = asyncio.run(run())
    assert size == alert_bus.QUEUE_SIZE
    assert dropped == 5
    assert first_id == "d5"


def test_retriever_publishes_every_logged_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = alert_bus.AlertBus()
    monkeypatch.setattr(alert_bus, "_bus", bus)
    monkeypatch.delenv("MERCURY_AGENT_MEMORY_PATH", raising=False)

    async def run() -> dict[str, Any]:
        token, queue = bus.subscribe("alice")
        retriever = KnowledgeRetriever()
        retriever.log_detection({"anomaly_detected": True}, domain="general", principal="alice")
        got = await asyncio.wait_for(queue.get(), timeout=2.0)
        bus.unsubscribe(token)
        return got

    assert asyncio.run(run())["principal"] == "alice"
    assert bus.published == 1


def test_store_cursors_see_rows_appended_after_a_point(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = DetectionMemoryStore(tmp_path / "memory.sqlite")
    assert store.latest_id() == 0
    store.append(_entry("alice", detection_id="a"))
    cursor = store.latest_id()
    assert cursor == 1
    assert store.load_after(cursor) == []
    store.append(_entry("bob", detection_id="b"))
    store.append(_entry(None, detection_id="c"))
    rows = store.load_after(cursor)
    assert [row_id for row_id, _ in rows] == [2, 3]
    assert [e["detection"]["detection_id"] for _, e in rows] == ["b", "c"]
    assert rows[0][1]["principal"] == "bob"
    store.close()


def test_a_busy_loop_gets_one_drain_callback_and_a_bounded_buffer() -> None:
    """Publishing while the subscriber loop cannot run must not pile up callbacks."""

    class StalledLoop:
        """A loop stand-in that accepts callbacks but never runs them, like a busy one."""

        def __init__(self) -> None:
            self.calls: list[tuple[Callable[..., None], tuple[object, ...]]] = []

        def is_closed(self) -> bool:
            return False

        def call_soon_threadsafe(self, fn: Callable[..., None], *args: object) -> None:
            self.calls.append((fn, args))

    async def run() -> tuple[int, int, int, int]:
        bus = alert_bus.AlertBus()
        token, queue = bus.subscribe(None)
        stalled = StalledLoop()
        # The stand-in replaces the real loop on the registered subscriber.
        bus._subscribers[token].loop = cast("asyncio.AbstractEventLoop", stalled)
        for i in range(alert_bus.QUEUE_SIZE * 3):
            bus.publish(_entry(None, detection_id=f"d{i}"))
        scheduled = len(stalled.calls)
        staged = bus.pending(token)
        dropped = bus.dropped
        # The loop finally runs its one callback: the buffer lands in the queue.
        fn, args = stalled.calls[0]
        fn(*args)
        in_queue = queue.qsize()
        first = queue.get_nowait()["detection"]["detection_id"]
        bus.unsubscribe(token)
        assert first == f"d{alert_bus.QUEUE_SIZE * 2}"
        return scheduled, staged, dropped, in_queue

    scheduled, staged, dropped, in_queue = asyncio.run(run())
    assert scheduled == 1
    assert staged == alert_bus.QUEUE_SIZE
    assert dropped == alert_bus.QUEUE_SIZE * 2
    assert in_queue == alert_bus.QUEUE_SIZE


def test_a_second_publish_after_a_drain_schedules_again() -> None:
    async def run() -> list[str]:
        bus = alert_bus.AlertBus()
        token, queue = bus.subscribe(None)
        bus.publish(_entry(None, detection_id="first"))
        await asyncio.sleep(0)
        bus.publish(_entry(None, detection_id="second"))
        await asyncio.sleep(0)
        got = [queue.get_nowait()["detection"]["detection_id"] for _ in range(queue.qsize())]
        bus.unsubscribe(token)
        return got

    assert asyncio.run(run()) == ["first", "second"]
