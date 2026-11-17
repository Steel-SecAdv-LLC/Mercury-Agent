# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The loop: read each feed when it is due, judge it, remember, report, check."""

from __future__ import annotations

import logging
import os
import socket
import threading
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, ClassVar

from omni_mercury_engine.watch.feeds import FEEDS, Feed
from omni_mercury_engine.watch.guidance import with_guidance
from omni_mercury_engine.watch.model import FeedStatus, Finding, Outcome
from omni_mercury_engine.watch.notify import Notifier
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

    Several watchers may share one store (the workers of ``serve --workers
    4``; ``mercury-agent watch`` beside a server). One of them holds the
    store's lease and does the watching; the rest stand by and take over
    when it stops or its lease lapses. Without that each read every source
    and each sent every message: four workers, three copies of one email.
    """

    #: How long a lease lasts unrenewed. It is renewed before every feed read
    #: and after every delivery, so it lapses only when its holder has died.
    LEASE: ClassVar[timedelta] = timedelta(minutes=15)

    def __init__(
        self,
        store: WatchStore | None = None,
        feeds: Iterable[Feed] | None = None,
        *,
        on_finding: Callable[[Finding], None] | None = None,
        clock: Callable[[], datetime] | None = None,
        notifier: Notifier | None = None,
    ) -> None:
        """Build a watcher.

        Args:
            store: Where to remember. Defaults to the store in
                :func:`~omni_mercury_engine.watch.store.default_state_dir`.
            feeds: Feeds to read. Defaults to every feed in ``FEEDS``.
            on_finding: Called with each finding when it is first reported or
                escalates.
            clock: Returns the current UTC time; replaceable in tests.
            notifier: Delivers findings to the people they concern. Defaults
                to one over ``store`` with the deployment's channels.
        """
        self.store = store if store is not None else WatchStore()
        self.feeds = list(feeds) if feeds is not None else [feed() for feed in FEEDS.values()]
        self.on_finding = on_finding
        self._clock = clock if clock is not None else (lambda: datetime.now(UTC))
        self._last_read: dict[str, datetime] = {}
        #: This watcher's name on the lease: unique to it, readable by a person.
        self.holder = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._standing_by = False
        self._looping = False
        self.notifier = (
            notifier if notifier is not None else Notifier(self.store, clock=self._clock)
        )
        for feed in self.feeds:
            feed.bind(self.store)

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
            judgment = feed.judge(feed.read(), now)
        except Exception as exc:  # a feed must never stop the watch
            logger.warning("watch: feed %s could not be read: %s", feed.name, exc)
            return self._down(status, "could not be read", exc)

        announced = 0
        findings = [with_guidance(finding) for finding in judgment.findings]
        for finding in findings:
            if self.store.report(finding, now):
                announced += 1
                self.notifier.broadcast(finding)
                if self.on_finding is not None:
                    self.on_finding(finding)
        if not judgment.partial:
            # A read that could not look everywhere ends nothing: what it does
            # not show may only be what it could not see, and a finding ended
            # for one read and found again would be told as if it were new.
            self.store.close_absent(
                feed.name, {finding.key for finding in findings if finding.ongoing}, now
            )
        try:
            # Only the watch that holds the lease delivers. One that was run
            # beside it (``watch --once`` while a server watches) has recorded
            # what it read; the holder's next pass sends it, once.
            if self.leads():
                self.notifier.sync(heartbeat=self.leads)
        except Exception as exc:  # telling people must not stop the watching
            logger.warning("watch: delivery pass failed: %s", exc)
        status.latest_reading = judgment.latest_reading
        stale = (
            judgment.latest_reading is not None and now - judgment.latest_reading > feed.stale_after
        )
        status.state = "stale" if stale else "ok"
        status.readings = judgment.readings
        status.subjects = judgment.subjects
        status.summary = judgment.summary
        status.new_findings = announced
        status.open_findings = self.store.open_count(feed.name, now)
        self.store.save_status(status)
        return status

    def leads(self) -> bool:
        """Take or renew the store's lease. Returns whether this watcher holds it."""
        return self.store.hold_lease(self.holder, self._clock(), self.LEASE)

    def _down(self, status: FeedStatus, summary: str, exc: Exception) -> FeedStatus:
        """Record ``status`` as down with why, as far as the store allows."""
        status.state = "down"
        status.error = f"{type(exc).__name__}: {str(exc)[:200]}"
        status.summary = summary
        try:
            status.open_findings = self.store.open_count(status.feed, status.checked_at)
            self.store.save_status(status)
        except Exception as store_exc:  # the store may be what failed
            logger.warning("watch: could not record %s as down: %s", status.feed, store_exc)
        return status

    def review_feed(self, feed: Feed) -> int:
        """Check the feed's past findings against later records.

        A finding is checked once a record has had time to exist, again at
        intervals while that record is not settled, and for the last time
        when the feed's review period is over. A record that cannot be
        reached leaves the finding as it was, to be tried again.

        Returns:
            How many findings were given an outcome.
        """
        now = self._clock()
        reviewed = 0
        for finding in self.store.awaiting_review(
            feed.name, now - feed.review_after, now - feed.review_every
        ):
            try:
                outcome = feed.review(finding, now)
            except Exception as exc:  # the record being down must not stop the watch
                logger.warning("watch: could not check %s: %s", finding.get("headline"), exc)
                continue
            last_chance = now - datetime.fromisoformat(finding["first_seen"]) >= feed.review_until
            if outcome is None:
                if last_chance:
                    outcome = Outcome(
                        "unconfirmed",
                        "No later record could be consulted for it.",
                        feed.source,
                        final=True,
                    )
                else:
                    continue
            elif last_chance and not outcome.final:
                outcome = Outcome(outcome.verdict, outcome.note, outcome.record, final=True)
            self.store.record_outcome(str(finding["key"]), outcome, now)
            reviewed += 1
        return reviewed

    def run_once(self, *, force: bool = True) -> list[FeedStatus]:
        """Read every feed that is due (every feed, when ``force``), then check past findings.

        Called on its own it is one pass: it delivers if no other watch holds
        the store's lease, and gives the lease back when it is done, so a
        ``watch --once`` beside a server does not leave the server standing by.
        """
        try:
            return self._pass(force=force)
        finally:
            if not self._looping:
                self.store.release_lease(self.holder)

    def _pass(self, *, force: bool) -> list[FeedStatus]:
        now = self._clock()
        statuses = []
        for feed in self.feeds:
            if not (force or self.due(feed, now)):
                continue
            # Reading a feed is guarded where it is read. What follows it is
            # not: remembering, announcing and checking all use the store, and
            # one "database is locked" there ended the loop's thread for good
            # while the server went on serving. Each feed's pass stands alone.
            try:
                status = self.read_feed(feed)
                if status.state != "down":
                    self.review_feed(feed)
            except Exception as exc:
                logger.exception("watch: the pass over feed %s failed", feed.name)
                status = self._down(
                    FeedStatus(feed=feed.name, title=feed.title, checked_at=now),
                    "its reading could not be recorded",
                    exc,
                )
            statuses.append(status)
        return statuses

    def run_forever(
        self, *, tick_seconds: float = 30.0, stop: threading.Event | None = None
    ) -> None:
        """Read each feed at its cadence until ``stop`` is set.

        Args:
            tick_seconds: How often to look for a feed that is due.
            stop: Set it to end the loop after the read in progress.
        """
        halt = stop if stop is not None else threading.Event()
        self._looping = True
        try:
            while not halt.is_set():
                try:
                    if self.leads():
                        if self._standing_by:
                            logger.info("watch: %s has taken over the watch", self.holder)
                        self._standing_by = False
                        self.run_once(force=False)
                    elif not self._standing_by:
                        self._standing_by = True
                        logger.info(
                            "watch: %s is watching this store; %s stands by",
                            self.store.lease().get("holder", "another watch"),
                            self.holder,
                        )
                except Exception:  # nothing a pass raises may end the watch
                    logger.exception("watch: a pass failed; trying again at the next tick")
                halt.wait(tick_seconds)
        finally:
            self._looping = False
            try:
                self.store.release_lease(self.holder)
            except Exception as exc:  # stopping must not raise; the lease lapses on its own
                logger.warning("watch: could not release the lease: %s", exc)
