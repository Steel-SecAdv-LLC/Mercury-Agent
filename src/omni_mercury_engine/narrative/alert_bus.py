# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""In-process fan-out of recorded detections to live listeners.

The console's agent panel has carried a proactive-alert dock since its first
release, and the voice API's own docstring promised "proactive alerts", but
no server route ever produced one: the console's ``alertsUrl()`` returned
``null`` and the dock read "not served by this engine". This module is the
missing wire. :meth:`KnowledgeRetriever.log_detection` publishes every entry
it records here; ``GET /api/v1/voice/alerts`` subscribes and streams the
flagged ones to the caller as server-sent events, bounded to the caller's
own principal exactly as the voice's answers are.

Publishing is synchronous and thread-safe (a detection may be recorded from
a worker thread); each subscriber owns an :class:`asyncio.Queue` on its own
event loop and is fed with ``call_soon_threadsafe``. A subscriber that falls
``QUEUE_SIZE`` entries behind loses the oldest, never blocks the publisher,
and a dead loop is dropped on the next publish.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.narrative.conversation import describe_detection

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

__all__ = ["QUEUE_SIZE", "AlertBus", "get_alert_bus", "principal_may_see", "to_alert"]

#: Entries a slow subscriber may lag before the oldest are dropped.
QUEUE_SIZE = 256


def principal_may_see(requested: str | None, recorded: str | None) -> bool:
    """The retriever's visibility rule: unowned entries to all, owned to their owner."""
    if recorded is None:
        return True
    return str(recorded) == (requested or "anonymous")


def to_alert(entry: dict[str, Any]) -> dict[str, Any]:
    """The event the console's alert dock reads, built from a retriever entry.

    The console shows ``initiative_type`` as the badge, ``severity`` as the
    fraction, ``domain``, and ``message``; the full record travels under
    ``event`` so "Explain the latest detection" and the alert agree.
    """
    record = dict(entry.get("detection") or {})
    timestamp = entry.get("timestamp")
    flagged = bool(record.get("anomaly_detected", False))
    label = record.get("severity_label")
    return {
        "initiative_type": (
            f"{label}_anomaly"
            if flagged and label
            else "anomaly_detected" if flagged else "clean_run"
        ),
        "severity": record.get("severity"),
        "domain": entry.get("domain"),
        "message": describe_detection(record, timestamp),
        "timestamp": timestamp,
        "event": record,
    }


class AlertBus:
    """Fan a recorded detection out to every live subscriber that may see it."""

    def __init__(self) -> None:
        """Start with no subscribers."""
        self._lock = threading.Lock()
        self._subscribers: dict[
            int, tuple[asyncio.AbstractEventLoop, asyncio.Queue[dict[str, Any]], str | None]
        ] = {}
        self._next_id = 1
        self.published = 0
        self.dropped = 0

    def subscribe(self, principal: str | None) -> tuple[int, asyncio.Queue[dict[str, Any]]]:
        """Register the running loop's queue for ``principal``; returns ``(token, queue)``."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=QUEUE_SIZE)
        with self._lock:
            token = self._next_id
            self._next_id += 1
            self._subscribers[token] = (loop, queue, principal)
        return token, queue

    def unsubscribe(self, token: int) -> None:
        """Forget a subscriber; unknown tokens are ignored."""
        with self._lock:
            self._subscribers.pop(token, None)

    @property
    def subscriber_count(self) -> int:
        """Live subscribers."""
        with self._lock:
            return len(self._subscribers)

    def publish(self, entry: dict[str, Any]) -> int:
        """Deliver ``entry`` to every subscriber allowed to see it; returns how many."""
        with self._lock:
            targets = list(self._subscribers.items())
        delivered = 0
        for token, (loop, queue, principal) in targets:
            if not principal_may_see(principal, entry.get("principal")):
                continue
            if loop.is_closed():
                self.unsubscribe(token)
                continue
            try:
                loop.call_soon_threadsafe(self._offer, queue, entry)
                delivered += 1
            except RuntimeError:  # loop shut down between the check and the call
                self.unsubscribe(token)
        self.published += 1
        return delivered

    def _offer(self, queue: asyncio.Queue[dict[str, Any]], entry: dict[str, Any]) -> None:
        if queue.full():
            try:
                queue.get_nowait()
                self.dropped += 1
            except asyncio.QueueEmpty:  # pragma: no cover - raced with the consumer
                pass
        queue.put_nowait(entry)

    def __iter__(self) -> Iterator[int]:
        """Subscriber tokens, for diagnostics."""
        with self._lock:
            return iter(list(self._subscribers))


_bus: AlertBus | None = None
_bus_lock = threading.Lock()


def get_alert_bus() -> AlertBus:
    """The process-wide bus the retriever publishes to."""
    global _bus
    if _bus is None:
        with _bus_lock:
            if _bus is None:
                _bus = AlertBus()
    return _bus


def heartbeat() -> dict[str, Any]:
    """A keep-alive frame: what a stream sends when nothing was recorded."""
    return {"initiative_type": "heartbeat", "timestamp": time.time()}
