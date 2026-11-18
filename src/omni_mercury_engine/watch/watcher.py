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
from typing import TYPE_CHECKING, Any, ClassVar

from omni_mercury_engine.watch.canary import Canary
from omni_mercury_engine.watch.feeds import FEEDS, Feed
from omni_mercury_engine.watch.guidance import with_guidance
from omni_mercury_engine.watch.model import FeedStatus, Finding, Outcome
from omni_mercury_engine.watch.notify import RETRY_FIRST, Notifier
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

    Announcing to open consoles is every watcher's job, not only the one that
    read the feed: each tells its own process's consoles what the store says
    is new (:meth:`announce`), whichever watcher recorded it.

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
        on_announced: Callable[[dict[str, Any]], None] | None = None,
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
            on_announced: Called with each finding another watch recorded in
                this store, as the store holds it. ``mercury-agent watch``
                beside a server says "new findings appear below"; without
                this, the ones the server read never did.
        """
        self.store = store if store is not None else WatchStore()
        self.feeds = list(feeds) if feeds is not None else [feed() for feed in FEEDS.values()]
        self.on_finding = on_finding
        self.on_announced = on_announced
        #: Announcements of findings this watcher itself reported, by key.
        self._own: list[str] = []
        self._clock = clock if clock is not None else (lambda: datetime.now(UTC))
        self._last_read: dict[str, datetime] = {}
        #: When each feed was last read by any watch of this store, as recorded.
        self._read_by_anyone: dict[str, datetime] = {}
        #: The last full delivery pass: when, and the newest announcement then.
        self._delivered_at: datetime | None = None
        self._delivered_to = 0
        self._delivered_mark: tuple[Any, ...] = ()
        #: This watcher's name on the lease: unique to it, readable by a person.
        self.holder = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._standing_by = False
        self._looping = False
        self._stopping = False
        self.notifier = (
            notifier if notifier is not None else Notifier(self.store, clock=self._clock)
        )
        #: Test messages sent the way findings are, to learn whether one arrives.
        self.canary = Canary(self.store, self.notifier, clock=self._clock)
        #: The last announcement this watcher has told its consoles of. It
        #: starts at the store's newest: what was found before this process
        #: began is the page's to show, not a fresh alert.
        self._told = self.store.last_announcement()
        for feed in self.feeds:
            feed.bind(self.store)

    def due(self, feed: Feed, now: datetime) -> bool:
        """Whether ``feed`` should be read at ``now``.

        It is, when nobody has read it for its cadence: this watcher, or
        another one of this store (as of this watcher's last pass). Judged by
        this watcher's reads alone, a worker that had never read a feed found
        it due at every tick and took its lease the moment it lapsed. A lease
        lasts fifteen minutes and four feeds are read every thirty to sixty,
        so the workers of ``serve --workers 4`` took turns reading each of
        those four times as often as it asks to be read.
        """
        reads = [
            read
            for read in (self._last_read.get(feed.name), self._read_by_anyone.get(feed.name))
            if read is not None
        ]
        return not reads or now - max(reads) >= feed.cadence

    def _note_reads(self) -> None:
        """Learn from the store when each feed was last read, by any watch."""
        try:
            statuses = self.store.statuses()
        except Exception as exc:  # the store being busy must not stop a pass
            logger.warning("watch: could not read when the feeds were last read: %s", exc)
            return
        for status in statuses:
            try:
                self._read_by_anyone[str(status["feed"])] = datetime.fromisoformat(
                    str(status["checked_at"])
                )
            except (KeyError, ValueError):
                continue

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
            if self.store.report(finding, now, feed.episode_window()):
                announced += 1
                self._own.append(finding.key)
                if self.on_finding is not None:
                    self.on_finding(finding)
        self.announce()
        if not judgment.partial or judgment.unchecked_places:
            # A read that could not look everywhere ends nothing it could not
            # have seen: what it does not show may only be what it could not
            # see, and a finding ended for one read and found again would be
            # told as if it were new. When the read says which places it
            # could not look at, only the findings in effect there are kept.
            self.store.close_absent(
                feed.name,
                {finding.key for finding in findings if finding.ongoing},
                now,
                judgment.unchecked_places,
                judgment.listed,
            )
        try:
            # Only the watch that holds the lease delivers. One that was run
            # beside it (``watch --once`` while a server watches) has recorded
            # what it read; the holder's next pass sends it, once.
            if self.leads():
                self.notifier.sync(heartbeat=self.leads, feed=feed.name)
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
        """Take or renew the lease for telling people. Returns whether this watcher holds it.

        A watcher with no way of telling anyone does not take it. A
        ``mercury-agent watch`` run in a shell without the server's mail
        settings used to, and while it held the lease nobody was told.
        """
        if self._stopping or not any(self.notifier.channels().values()):
            return False
        return self.store.hold_lease(self.holder, self._clock(), self.LEASE)

    def reads(self, feed: Feed) -> bool:
        """Take or renew the lease for reading ``feed``. Returns whether this watcher holds it.

        Each feed has its own. One lease for everything meant the first watch
        to start did all the reading, whatever it had been asked to read: a
        ``watch --feed earthquakes`` left the server's eight other feeds unread.
        """
        if self._stopping:
            return False
        return self.store.hold_lease(
            self.holder, self._clock(), self.LEASE, name=f"feed:{feed.name}"
        )

    def stop(self) -> None:
        """Give every lease up now and take none again: this watcher is ending.

        A server being restarted while a feed is mid-read cannot wait for the
        read. Its leases were left to lapse, and for up to :attr:`LEASE` no
        watch could read or tell anyone.
        """
        self._stopping = True
        try:
            self.store.release_lease(self.holder)
        except Exception as exc:  # stopping must not raise; the leases lapse on their own
            logger.warning("watch: could not release the leases: %s", exc)

    def announce(self) -> int:
        """Tell this process's open consoles of what the store says is new.

        The alert bus is in-process. When only the watcher that read a feed
        announced on it, ``serve --workers 4`` reached the consoles of one
        worker in four, and a ``mercury-agent watch`` running beside the
        server reached none. Every watcher, leading or standing by, now reads
        the store's announcements and tells its own.

        Returns:
            How many findings were put on the bus.
        """
        published = 0
        for number, finding in self.store.announcements_after(self._told):
            self._told = number
            published += int(self.notifier.broadcast(finding))
            if str(finding["key"]) in self._own:
                self._own.remove(str(finding["key"]))  # its own: already said
            elif self.on_announced is not None:
                self.on_announced(finding)
        return published

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
        if not force:
            self._note_reads()
        statuses = []
        for feed in self.feeds:
            if not (force or self.due(feed, now)):
                continue
            # A pass that was asked for (``watch --once``) reads; the loop reads
            # a feed only while it holds that feed's lease.
            if not force and not self.reads(feed):
                continue  # another watch is reading this one
            # Reading a feed is guarded where it is read. What follows it is
            # not: remembering, announcing and checking all use the store, and
            # one "database is locked" there ended the loop's thread for good
            # while the server went on serving. Each feed's pass stands alone.
            try:
                status = self.read_feed(feed)
            except Exception as exc:
                logger.exception("watch: the pass over feed %s failed", feed.name)
                status = self._down(
                    FeedStatus(feed=feed.name, title=feed.title, checked_at=now),
                    "its reading could not be recorded",
                    exc,
                )
            else:
                if status.state != "down":
                    # Checking past findings is its own step. A failure in it
                    # used to overwrite the reading just recorded with "down".
                    try:
                        self.review_feed(feed)
                    except Exception:
                        logger.exception("watch: checking %s's past findings failed", feed.name)
            statuses.append(status)
        try:
            # Once more over everything: a delivery whose wait has passed, a
            # subscriber who joined since, whichever feed their finding is from.
            #
            # And what another watch read. This pass used to run only when
            # this watcher had itself read a feed: with ``mercury-agent watch``
            # reading beside a server (it has no mail settings and cannot
            # tell anyone), the server delivered a tornado warning only when
            # it next happened to read something, up to an hour later.
            if self._owes_a_delivery_pass(bool(statuses), now) and self.leads():
                # What the pass is about to read, noted before it reads: a
                # subscriber who joins while it runs is new to the next one.
                newest = self.store.last_announcement()
                mark = self.store.delivery_mark()[1]
                held = [True]

                def still_leads(held: list[bool] = held) -> bool:
                    held[0] = held[0] and self.leads()
                    return held[0]

                self.notifier.sync(heartbeat=still_leads)
                if held[0]:
                    # A pass that ended because the lease went has not been
                    # made. Asked of the pass itself: asked of the lease
                    # afterwards, a lease lost and taken back in between
                    # counted a pass that had stopped half way.
                    self._delivered_at, self._delivered_to = now, newest
                    self._delivered_mark = mark
        except Exception as exc:  # telling people must not stop the watching
            logger.warning("watch: delivery pass failed: %s", exc)
        try:
            # Whether a message would arrive, asked of the channels themselves:
            # a ledger that says "sent" says only that a provider took it.
            self.canary.run(self.leads)
        except Exception as exc:  # proving delivery must not stop the watching either
            logger.warning("watch: the delivery canary failed: %s", exc)
        return statuses

    def _owes_a_delivery_pass(self, read_a_feed: bool, now: datetime) -> bool:
        """Whether there may be something to send, asked of the store in a few small queries.

        A feed this watcher read, a finding any watch recorded, a change in
        who is told or where (:meth:`WatchStore.delivery_mark`), or a failed
        delivery whose wait may have passed. A full pass loads every
        subscriber, everything standing and the whole ledger: once a minute
        for nothing was several times the work the watch had been doing.
        """
        if read_a_feed or self._delivered_at is None:
            return True
        if self.store.last_announcement() > self._delivered_to:
            return True
        failed, mark = self.store.delivery_mark()
        if mark != self._delivered_mark:
            return True
        return failed > 0 and now - self._delivered_at >= RETRY_FIRST

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
                    # Each feed is read by whichever watch holds its lease, and
                    # people are told by whichever holds the lease for that.
                    self.run_once(force=False)
                    leading = self.store.lease().get("holder") == self.holder
                    if leading and self._standing_by:
                        logger.info("watch: %s has taken over the telling", self.holder)
                    elif not leading and not self._standing_by:
                        logger.info(
                            "watch: %s tells people from this store; %s stands by for that",
                            self.store.lease().get("holder", "another watch"),
                            self.holder,
                        )
                    self._standing_by = not leading
                    self.announce()  # what any other watch found, to this process
                except Exception:  # nothing a pass raises may end the watch
                    logger.exception("watch: a pass failed; trying again at the next tick")
                halt.wait(tick_seconds)
        finally:
            self._looping = False
            try:
                self.store.release_lease(self.holder)
            except Exception as exc:  # stopping must not raise; the lease lapses on its own
                logger.warning("watch: could not release the lease: %s", exc)
