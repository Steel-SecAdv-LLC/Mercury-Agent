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
a worker thread); each subscriber owns a bounded staging buffer and an
:class:`asyncio.Queue` on its own event loop, fed by a single pending drain
callback scheduled with ``call_soon_threadsafe``. A subscriber that falls
``QUEUE_SIZE`` entries behind loses the oldest, never blocks the publisher and
never accumulates callbacks, and a dead loop is dropped on the next publish.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
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
    """Fan a recorded detection out to every live subscriber that may see it.

    Each subscriber owns a bounded, lock-guarded staging buffer that the
    publisher (any thread) appends to, plus the :class:`asyncio.Queue` the
    stream reads. One drain callback per subscriber is ever pending on its
    loop: the first publish after a drain schedules it, later publishes only
    append to the buffer, and the drain moves the buffer into the queue. So a
    recording thread that outpaces a busy subscriber loop can hold at most
    :data:`QUEUE_SIZE` entries in the buffer plus one pending callback, never
    an unbounded backlog of callbacks each retaining an entry.
    """

    def __init__(self) -> None:
        """Start with no subscribers."""
        self._lock = threading.Lock()
        self._subscribers: dict[int, _Subscriber] = {}
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
            self._subscribers[token] = _Subscriber(loop, queue, principal)
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

    def pending(self, token: int) -> int:
        """Entries staged for subscriber ``token`` that its loop has not drained yet."""
        with self._lock:
            sub = self._subscribers.get(token)
        return sub.pending() if sub is not None else 0

    def publish(self, entry: dict[str, Any]) -> int:
        """Deliver ``entry`` to every subscriber allowed to see it; returns how many."""
        with self._lock:
            targets = list(self._subscribers.items())
        delivered = 0
        for token, sub in targets:
            if not principal_may_see(sub.principal, entry.get("principal")):
                continue
            if sub.loop.is_closed():
                self.unsubscribe(token)
                continue
            self.dropped += sub.stage(entry)
            try:
                if sub.arm_drain():
                    sub.loop.call_soon_threadsafe(self._drain, sub)
                delivered += 1
            except RuntimeError:  # loop shut down between the check and the call
                sub.disarm_drain()
                self.unsubscribe(token)
        self.published += 1
        return delivered

    def _drain(self, sub: _Subscriber) -> None:
        """Move the staged entries into the subscriber's queue (runs on its loop)."""
        for entry in sub.take_all():
            if sub.queue.full():
                try:
                    sub.queue.get_nowait()
                    self.dropped += 1
                except asyncio.QueueEmpty:  # pragma: no cover - raced with the consumer
                    pass
            sub.queue.put_nowait(entry)

    def __iter__(self) -> Iterator[int]:
        """Subscriber tokens, for diagnostics."""
        with self._lock:
            return iter(list(self._subscribers))


class _Subscriber:
    """One subscriber: its loop, its queue, and the bounded staging buffer feeding it."""

    __slots__ = ("_buffer", "_drain_pending", "_lock", "loop", "principal", "queue")

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[dict[str, Any]],
        principal: str | None,
    ) -> None:
        self.loop = loop
        self.queue = queue
        self.principal = principal
        self._buffer: deque[dict[str, Any]] = deque(maxlen=QUEUE_SIZE)
        self._drain_pending = False
        self._lock = threading.Lock()

    def stage(self, entry: dict[str, Any]) -> int:
        """Append ``entry`` to the buffer; returns 1 if the oldest entry was pushed out."""
        with self._lock:
            dropped = 1 if len(self._buffer) == self._buffer.maxlen else 0
            self._buffer.append(entry)
        return dropped

    def arm_drain(self) -> bool:
        """Claim the single pending drain; ``True`` when the caller must schedule it."""
        with self._lock:
            if self._drain_pending:
                return False
            self._drain_pending = True
            return True

    def disarm_drain(self) -> None:
        """Release the drain claim after scheduling failed."""
        with self._lock:
            self._drain_pending = False

    def take_all(self) -> list[dict[str, Any]]:
        """Empty the buffer and release the drain claim, atomically."""
        with self._lock:
            entries = list(self._buffer)
            self._buffer.clear()
            self._drain_pending = False
        return entries

    def pending(self) -> int:
        """Entries staged and not yet drained."""
        with self._lock:
            return len(self._buffer)


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
