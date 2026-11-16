# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The loop: read each feed when it is due, judge it, remember, report."""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from omni_mercury_engine.watch.feeds import FEEDS, Feed
from omni_mercury_engine.watch.model import FeedStatus, Finding
from omni_mercury_engine.watch.store import WatchStore

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

logger = logging.getLogger(__name__)

__all__ = ["Watcher"]


class Watcher:
    """Reads feeds on their cadences and keeps the store current.

    A feed that cannot be read is recorded as down and tried again at its next
    cadence; it never stops the others. A finding is announced once, when it
    first appears or when it becomes more severe.
    """

    def __init__(
        self,
        store: WatchStore | None = None,
        feeds: Iterable[Feed] | None = None,
        *,
        on_finding: Callable[[Finding], None] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Build a watcher.

        Args:
            store: Where to remember. Defaults to the store in
                :func:`~omni_mercury_engine.watch.store.default_state_dir`.
            feeds: Feeds to read. Defaults to every feed in ``FEEDS``.
            on_finding: Called with each finding when it is first reported or
                escalates.
            clock: Returns the current UTC time; replaceable in tests.
        """
        self.store = store if store is not None else WatchStore()
        self.feeds = list(feeds) if feeds is not None else [feed() for feed in FEEDS.values()]
        self.on_finding = on_finding
        self._clock = clock if clock is not None else (lambda: datetime.now(UTC))
        self._last_read: dict[str, datetime] = {}

    def due(self, feed: Feed, now: datetime) -> bool:
        """Whether ``feed`` should be read at ``now``."""
        last = self._last_read.get(feed.name)
        return last is None or now - last >= feed.cadence

    def read_feed(self, feed: Feed) -> FeedStatus:
        """Read and judge one feed, and record what happened."""
        now = self._clock()
        self._last_read[feed.name] = now
        status = FeedStatus(feed=feed.name, title=feed.title, checked_at=now)
        try:
            judgment = feed.judge(feed.read(), self.store, now)
        except Exception as exc:  # a feed must never stop the watch
            logger.warning("watch: feed %s could not be read: %s", feed.name, exc)
            status.state = "down"
            status.error = f"{type(exc).__name__}: {str(exc)[:200]}"
            status.summary = "could not be read"
            status.open_findings = self.store.open_count(feed.name)
            self.store.save_status(status)
            return status

        announced = 0
        for finding in judgment.findings:
            if self.store.report(finding, now):
                announced += 1
                if self.on_finding is not None:
                    self.on_finding(finding)
        self.store.close_absent(
            feed.name, {finding.key for finding in judgment.findings if finding.ongoing}, now
        )
        status.latest_reading = judgment.latest_reading
        stale = (
            judgment.latest_reading is not None and now - judgment.latest_reading > feed.stale_after
        )
        status.state = "stale" if stale else "ok"
        status.readings = judgment.readings
        status.subjects = judgment.subjects
        status.summary = judgment.summary
        status.new_findings = announced
        status.open_findings = self.store.open_count(feed.name)
        self.store.save_status(status)
        return status

    def run_once(self, *, force: bool = True) -> list[FeedStatus]:
        """Read every feed that is due (every feed, when ``force``)."""
        now = self._clock()
        return [self.read_feed(feed) for feed in self.feeds if force or self.due(feed, now)]

    def run_forever(
        self, *, tick_seconds: float = 30.0, stop: threading.Event | None = None
    ) -> None:
        """Read each feed at its cadence until ``stop`` is set.

        Args:
            tick_seconds: How often to look for a feed that is due.
            stop: Set it to end the loop after the read in progress.
        """
        halt = stop if stop is not None else threading.Event()
        while not halt.is_set():
            self.run_once(force=False)
            halt.wait(tick_seconds)
