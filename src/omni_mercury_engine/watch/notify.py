# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Telling people: who a finding concerns, and getting it to them.

The watch saw a river in major flood and told no one: its findings were
printed where it ran, stored and served to whoever came to look. This module
delivers them.

* **To whom.** A subscriber gives an address and, usually, a place. A
  finding with coordinates goes to the subscribers whose place is within
  their radius of it; a warning the National Weather Service says is in
  effect at a subscriber's place goes to that subscriber; a finding with no
  place (a geomagnetic storm) goes to everyone. A subscriber with no place
  receives everything.
* **How.** By email through the deployment's mailer, by text message to a
  number its owner has verified (:mod:`~omni_mercury_engine.watch.sms`), to
  the console's alert dock through the alert bus, and to one operator
  webhook (a phone-push or chat service).
* **Once, and provably.** Every delivery is recorded. A finding is sent to
  a subscriber once, and again only if it becomes more severe than what they
  were told; one that failed is tried again, at widening intervals for about
  six hours while the finding stands, and the failure is shown, not
  swallowed. Delivery works from the
  findings that stand, not from the moment they appeared, so someone who
  subscribes while a flood is under way is told about it. A single
  occurrence (an earthquake) stands for a day: a new subscriber is not sent
  every earthquake the store remembers.

Email is sent only through a configured SMTP server. Without one the
subscribers cannot be reached, and the watch says so instead of filing the
message somewhere and calling it sent.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.utils.geo import haversine_km
from omni_mercury_engine.watch.guidance import EMERGENCY_LINE
from omni_mercury_engine.watch.model import EVERYWHERE_KINDS, RANKS, Finding, public_view
from omni_mercury_engine.watch.shaking import felt_at
from omni_mercury_engine.watch.sms import (
    SmsOptedOutError,
    SmsSender,
    build_sms_sender,
    text_message,
)
from omni_mercury_engine.watch.store import address_entry

if TYPE_CHECKING:
    from collections.abc import Callable

    from omni_mercury_engine.api.mailer import Mailer
    from omni_mercury_engine.watch.store import WatchStore

logger = logging.getLogger(__name__)

__all__ = [
    "AUTOMATIC",
    "MAX_ATTEMPTS",
    "RETRY_CAP",
    "RETRY_FIRST",
    "Notifier",
    "address_entry",
    "audience",
    "configured_channels",
    "distance_km",
    "notice",
    "one_per_address",
    "shaking_note",
]

#: A delivery that has failed this many times is left as failed. With the
#: waits below that is about six hours of trying while the finding stands.
MAX_ATTEMPTS = 12
#: How long after a first failure the delivery is tried again; the wait
#: doubles with each failure, up to :data:`RETRY_CAP`. There used to be no
#: wait and five attempts: the watch delivers after every feed it reads, nine
#: in a pass, so a mail server that was down for one minute spent all five in
#: that pass and the alert was never sent, to anyone.
RETRY_FIRST = timedelta(minutes=1)
RETRY_CAP = timedelta(hours=1)
#: The operator webhook's entry in the delivery ledger.
WEBHOOK_SUBSCRIBER = 0
WEBHOOK_URL_ENV = "MERCURY_WATCH_WEBHOOK_URL"
WEBHOOK_TOPIC_ENV = "MERCURY_WATCH_WEBHOOK_TOPIC"
WEBHOOK_MIN_SEVERITY_ENV = "MERCURY_WATCH_WEBHOOK_MIN_SEVERITY"
#: Where this server is reached, for the link in a message. The deployment's
#: own setting, which it must already give for account emails; the watch read
#: a name of its own that nothing documented, so no alert carried the link.
PUBLIC_URL_ENV = "MERCURY_PUBLIC_BASE_URL"
#: The name the watch used to read; still honoured.
LEGACY_PUBLIC_URL_ENV = "MERCURY_PUBLIC_URL"

_RANK = RANKS

#: Headers on every message the watch sends. RFC 3834: a message no person
#: wrote says so, and an out-of-office responder does not answer each alert.
AUTOMATIC = {"Auto-Submitted": "auto-generated"}


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points, in kilometres (the shared geo kernel's)."""
    return float(haversine_km(lat1, lon1, lat2, lon2))


def retry_wait(failures: int) -> timedelta:
    """How long to wait after ``failures`` failed attempts before the next."""
    doublings = min(max(failures - 1, 0), 16)  # far past the cap; keeps the shift small
    return min(RETRY_FIRST * (1 << doublings), RETRY_CAP)


def audience(
    finding: dict[str, Any],
    subscribers: list[dict[str, Any]],
    reach: dict[str, Any] | None = None,
) -> list[tuple[dict[str, Any], float | None]]:
    """The subscribers a finding concerns, each with its distance where one is known.

    Args:
        finding: The finding as the store holds it.
        subscribers: Every subscription.
        reach: For an earthquake, where USGS says it was felt
            (:mod:`~omni_mercury_engine.watch.shaking`). With it, a place
            USGS has spoken of is decided by what USGS said, not by the
            subscriber's radius: told when it was felt there however far
            away, not told when the estimated shaking there is below what
            people notice. A place USGS has said nothing of keeps its radius.
    """
    rank = _RANK.get(str(finding.get("severity")), 1)
    places = {int(place) for place in finding.get("place_ids") or ()}
    lat, lon = finding.get("latitude"), finding.get("longitude")
    out: list[tuple[dict[str, Any], float | None]] = []
    for subscriber in subscribers:
        if rank < int(subscriber["min_severity"]):
            continue
        here = subscriber["latitude"] is not None and subscriber["longitude"] is not None
        if not here:
            out.append((subscriber, None))  # no place: everything
        elif int(subscriber["id"]) in places:
            out.append((subscriber, 0.0))  # the source says it is in effect there
        elif lat is not None and lon is not None:
            away = distance_km(
                float(lat), float(lon), subscriber["latitude"], subscriber["longitude"]
            )
            said = (
                felt_at(reach, subscriber["latitude"], subscriber["longitude"])
                if reach is not None and finding.get("kind") == "earthquake"
                else None
            )
            if said.felt if said is not None else away <= float(subscriber["radius_km"]):
                out.append((subscriber, away))
        elif finding.get("kind") in EVERYWHERE_KINDS and not places:
            out.append((subscriber, None))  # nowhere in particular: everyone
        # Anything else happens somewhere. Without its coordinates it cannot be
        # said to be near this place, and it used to be sent to every place on
        # Earth: one failed location lookup mailed each Alaskan volcano to a
        # subscriber in Florida with a 50 km radius.
    return out


def one_per_address(
    reached: list[tuple[dict[str, Any], float | None]],
) -> list[tuple[dict[str, Any], float | None]]:
    """One entry for each address: its place nearest the finding.

    Someone with two places near each other (home and work) was sent the
    same finding twice, once for each. A place with a known distance comes
    before one without; ties go to the longer-held subscription.
    """
    best: dict[str, tuple[dict[str, Any], float | None]] = {}
    for subscriber, away in reached:
        address = str(subscriber["email"]).lower()
        held = best.get(address)
        if held is None or (away is not None and (held[1] is None or away < held[1])):
            best[address] = (subscriber, away)
    return list(best.values())


def shaking_note(
    finding: dict[str, Any], subscriber: dict[str, Any] | None, reach: dict[str, Any] | None
) -> str:
    """For an earthquake, what USGS says of the subscriber's place (``""`` when nothing)."""
    if (
        reach is None
        or subscriber is None
        or finding.get("kind") != "earthquake"
        or subscriber.get("latitude") is None
        or subscriber.get("longitude") is None
    ):
        return ""
    said = felt_at(reach, subscriber["latitude"], subscriber["longitude"])
    place = str(subscriber.get("place") or "your place")
    return said.said(place) if said is not None else ""


def notice(
    finding: dict[str, Any],
    subscriber: dict[str, Any] | None = None,
    away: float | None = None,
    note: str = "",
) -> tuple[str, str]:
    """The subject and body of the message that tells a person about a finding.

    ``note`` is one more sentence about the reader's place (for an
    earthquake, what USGS says of the shaking there).
    """
    severity = str(finding.get("severity", "notable")).capitalize()
    lines = [str(finding["headline"]), "", str(finding["detail"])]
    if finding.get("area"):
        lines += ["", f"Where: {finding['area']}"]
    place = str((subscriber or {}).get("place") or "")
    if place and away is not None:
        lines += [
            "",
            (
                f'This is in effect at "{place}".'
                if away == 0.0
                else f'About {away:,.0f} km from "{place}".'
            ),
        ]
    if note:
        lines += ["", note]
    if finding.get("action"):
        lines += ["", "WHAT TO DO", str(finding["action"])]
        if finding.get("action_source"):
            lines.append(f"({finding['action_source']})")
    lines += ["", f"Source: {finding['source']}.", f"Basis: {finding['basis']}."]
    public = (
        (os.environ.get(PUBLIC_URL_ENV) or os.environ.get(LEGACY_PUBLIC_URL_ENV) or "")
        .strip()
        .rstrip("/")
    )
    if public:
        lines += ["", f"What Mercury is watching now: {public}/watch"]
    lines += ["", EMERGENCY_LINE]
    if subscriber is not None and subscriber.get("email"):
        lines += [
            "",
            f"You receive this because {subscriber['email']} is subscribed to this Mercury "
            "watch" + (f' for "{place}"' if place else "") + ". To stop, ask whoever runs it, "
            "or change it under your account there.",
        ]
    # A headline carries text from the source (a place, an alert's area). A
    # line break in it is refused as a mail header, and the finding would
    # fail every attempt and never be delivered.
    subject = " ".join(f"[Mercury watch] {severity}: {finding['headline']}".split())
    return subject, "\n".join(lines)


def _sms_sender() -> SmsSender | None:
    """The deployment's text-message sender, or ``None`` when it has none it can use."""
    try:
        return build_sms_sender()
    except ValueError as exc:
        # Settings that are present and wrong: say so, and treat texts as off
        # (every surface then says they cannot be sent) instead of failing to start.
        logger.error("watch: text messages are misconfigured and will not be sent: %s", exc)
        return None


def _smtp_mailer() -> Mailer | None:
    """The deployment's SMTP mailer, or ``None`` when it has none."""
    from omni_mercury_engine.api.mailer import SMTP_HOST_ENV, build_mailer

    return build_mailer() if os.environ.get(SMTP_HOST_ENV, "").strip() else None


_channels: tuple[tuple[str, ...], dict[str, bool]] | None = None


def configured_channels() -> dict[str, bool]:
    """Which ways of reaching people this deployment has, worked out once per configuration.

    The public snapshot asked this by building a :class:`Notifier` on every
    request, which built a mailer and a text-message sender each time, and
    logged an error on each when the text settings were wrong.
    """
    global _channels
    from omni_mercury_engine.api.mailer import SMTP_HOST_ENV
    from omni_mercury_engine.watch import sms

    names = (
        SMTP_HOST_ENV, sms.ACCOUNT_SID_ENV, sms.AUTH_TOKEN_ENV, sms.FROM_ENV,
        sms.SERVICE_SID_ENV, WEBHOOK_URL_ENV,
    )  # fmt: skip
    settings = tuple(os.environ.get(name, "") for name in names)
    if _channels is None or _channels[0] != settings:
        _channels = (
            settings,
            {
                "email": bool(os.environ.get(SMTP_HOST_ENV, "").strip()),
                "sms": _sms_sender() is not None,
                "webhook": bool(os.environ.get(WEBHOOK_URL_ENV, "").strip()),
            },
        )
    return dict(_channels[1])


class Notifier:
    """Delivers the findings that stand to the people they concern."""

    def __init__(
        self,
        store: WatchStore,
        *,
        mailer: Mailer | None = None,
        post: Callable[[str, dict[str, Any]], None] | None = None,
        clock: Callable[[], datetime] | None = None,
        sms: SmsSender | None = None,
    ) -> None:
        """Build a notifier.

        Args:
            store: The watch's store: subscribers, findings, the ledger.
            mailer: Sends email. Defaults to the deployment's SMTP mailer;
                with none configured, email is not attempted.
            sms: Sends text messages. Defaults to the deployment's provider
                account (:func:`~omni_mercury_engine.watch.sms.build_sms_sender`);
                with none configured, no text is attempted.
            post: Sends the webhook's JSON body to its URL; replaceable in tests.
            clock: Returns the current UTC time; replaceable in tests.
        """
        self.store = store
        self.mailer = mailer if mailer is not None else _smtp_mailer()
        self.sms = sms if sms is not None else _sms_sender()
        self._post = post if post is not None else _post_json
        self._clock = clock if clock is not None else (lambda: datetime.now(UTC))

    @property
    def webhook_url(self) -> str:
        """The operator's webhook, when one is set."""
        return os.environ.get(WEBHOOK_URL_ENV, "").strip()

    def channels(self) -> dict[str, bool]:
        """Which ways of reaching people this deployment has."""
        return {
            "email": self.mailer is not None,
            "sms": self.sms is not None,
            "webhook": bool(self.webhook_url),
        }

    def broadcast(self, finding: Finding | dict[str, Any]) -> bool:
        """Put a finding on the alert bus, for consoles that are open now.

        The bus reaches every open console, signed in or not. A finding that
        is recorded only because it is in effect at a subscriber's place is
        not put on it (:func:`~omni_mercury_engine.watch.model.public_view`):
        it used to be, with the place ids in the event.

        Returns:
            Whether it was published.
        """
        entry = public_view(finding.to_dict() if isinstance(finding, Finding) else finding)
        if entry is None:
            return False
        try:
            from omni_mercury_engine.narrative.alert_bus import get_alert_bus

            rank = _RANK.get(str(entry.get("severity")), 1)
            feed, action = str(entry.get("feed")), str(entry.get("action") or "")
            get_alert_bus().publish(
                {
                    "domain": f"watch:{feed}",
                    "timestamp": self._clock().timestamp(),
                    "principal": None,
                    "detection": {
                        "anomaly_detected": True,
                        "detection_id": f"watch:{entry['key']}:{rank}",
                        "severity": rank / 3,
                        "severity_label": entry["severity"],
                    },
                    "alert": {
                        "initiative_type": f"watch_{entry['severity']}",
                        "severity": rank / 3,
                        "domain": f"watch:{feed}",
                        "message": str(entry["headline"]) + (f". {action}" if action else ""),
                        "timestamp": self._clock().timestamp(),
                        "event": entry,
                    },
                }
            )
        except Exception as exc:  # an open console is a courtesy; it must not stop delivery
            logger.warning("watch: could not broadcast %s: %s", entry.get("headline"), exc)
            return False
        return True

    def sync(
        self, heartbeat: Callable[[], object] | None = None, feed: str | None = None
    ) -> dict[str, int]:
        """Deliver every standing finding to everyone it concerns and has not reached.

        Args:
            heartbeat: Called after each delivery attempt. A long pass (many
                subscribers, a slow mail server) uses it to renew the
                watch's lease, so no other watch takes the pass for dead.
                When it answers ``False`` the lease is gone and the pass
                ends there: it used to go on sending from the ledger it had
                read, beside the watch that had taken over.
            feed: Only this feed's findings. The watch delivers after each
                feed it reads; going over every feed's findings each time
                was nine full passes where one does.

        Returns:
            How many deliveries were ``sent`` and how many ``failed`` in this pass.
        """
        done = {"sent": 0, "failed": 0}
        mailer = self.mailer
        reachable = mailer is not None or self.sms is not None
        subscribers = self.store.subscribers() if reachable else []
        phones = self.store.verified_phones() if self.sms is not None else {}
        hook = self.webhook_url
        if not subscribers and not hook:
            return done
        try:
            least = int(os.environ.get(WEBHOOK_MIN_SEVERITY_ENV, "2"))
        except ValueError:
            least = 2
        now = self._clock()
        standing = self.store.standing(now=now, limit=1000, feed=feed)
        if not standing:
            return done
        ledger = self.store.standing_deliveries(now)
        shaken = subscribers and any(f.get("kind") == "earthquake" for f in standing)
        reaches = self.store.reaches() if shaken else {}
        # One ledger entry for an address, whichever of its places a finding
        # is nearest. The entry was the place's: when the nearest place
        # changed between passes (USGS's felt reports arriving for "work"
        # after the radius had chosen "home") the same finding went again.
        # Then it was the address's oldest place, which changed when that
        # place was removed: everything standing near the others went again.
        # It is the address's own. What was recorded under one of its places
        # is still read, so nothing already sent is sent for the change.
        places_of: dict[str, list[int]] = {}
        for subscriber in subscribers:
            places_of.setdefault(str(subscriber["email"]).lower(), []).append(int(subscriber["id"]))
        for finding in standing:
            rank = _RANK.get(str(finding.get("severity")), 1)
            key = str(finding["key"])
            # What was sent before it last became more severe was sent of a
            # milder thing and does not count.
            since = str(finding.get("raised_at") or finding.get("first_seen") or "")
            # Where USGS says an earthquake was felt, once it has said.
            reach = (
                reaches.get(str(finding.get("subject")))
                if finding.get("kind") == "earthquake"
                else None
            )
            if reach is not None and not reach.get("grid") and not reach.get("cells"):
                reach = None
            for subscriber, away in one_per_address(audience(finding, subscribers, reach)):
                note = shaking_note(finding, subscriber, reach)
                address = str(subscriber["email"]).lower()
                entry = address_entry(address)
                if (
                    mailer is not None
                    and self._due(ledger, key, rank, entry, "email", now, since)
                    and not any(
                        self._sent(ledger, key, rank, place, "email", since)
                        for place in places_of[address]
                    )
                ):
                    subject, body = notice(finding, subscriber, away, note)
                    self._attempt(
                        key, rank, entry, "email", done,
                        partial(
                            mailer.send, to=str(subscriber["email"]), subject=subject,
                            body=body, headers=AUTOMATIC,
                        ),
                    )  # fmt: skip
                    if heartbeat is not None and heartbeat() is False:
                        return done  # the lease is another watch's now, or this one is stopping
                # A text goes to the address's verified number, for findings at or
                # above the severity its owner chose for texts. The ledger entry is
                # the phone's, so it is texted once whichever place matched.
                phone = phones.get(str(subscriber["email"]).lower())
                if (
                    phone is not None
                    and rank >= int(phone["min_severity"])
                    and self._due(ledger, key, rank, int(phone["id"]), "sms", now, since)
                ):
                    text = text_message(finding, subscriber, away, note)
                    self._attempt(
                        key, rank, int(phone["id"]), "sms", done,
                        partial(self._text, str(phone["number"]), text),
                    )  # fmt: skip
                    if heartbeat is not None and heartbeat() is False:
                        return done  # the lease is another watch's now, or this one is stopping
            if (
                hook
                and rank >= least
                and self._due(ledger, key, rank, WEBHOOK_SUBSCRIBER, "webhook", now, since)
            ):
                subject, body = notice(finding)
                payload: dict[str, Any] = {
                    "title": subject,
                    "message": body,
                    "text": f"*{subject}*\n{body}",
                    "priority": 2 + rank,
                    "tags": [str(finding.get("severity")), str(finding.get("feed"))],
                }
                topic = os.environ.get(WEBHOOK_TOPIC_ENV, "").strip()
                if topic:
                    payload["topic"] = topic
                self._attempt(
                    key, rank, WEBHOOK_SUBSCRIBER, "webhook", done,
                    partial(self._post, hook, payload),
                )  # fmt: skip
        return done

    def _text(self, number: str, text: str) -> None:
        """Send one text; a number whose owner replied STOP is not texted again."""
        if self.sms is None:
            raise RuntimeError("no text-message sender is configured")
        try:
            self.sms.send(to=number, body=text)
        except SmsOptedOutError:
            self.store.unverify_phone(number)
            raise

    @staticmethod
    def _due(
        ledger: dict[tuple[str, int, int, str], dict[str, Any]],
        key: str,
        rank: int,
        subscriber: int,
        channel: str,
        now: datetime,
        since: str = "",
    ) -> bool:
        """Whether this finding, at this severity, is still owed on this channel.

        Not when it was sent at this severity or a higher one since it last
        became more severe (``since``): a flood that falls from major to
        moderate to minor is one flood the reader has been told of, and each
        step down used to go out as a new alert. One that falls and then
        climbs back is told again, because what was sent was sent before
        that. A failed attempt is tried again once its wait has passed, up to
        :data:`MAX_ATTEMPTS`.
        """
        if Notifier._sent(ledger, key, rank, subscriber, channel, since):
            return False
        record = ledger.get((key, rank, subscriber, channel))
        if not record or str(record["at"]) < since:
            return True
        failures = int(record["attempts"])
        if failures >= MAX_ATTEMPTS:
            return False
        return now - datetime.fromisoformat(str(record["at"])) >= retry_wait(failures)

    @staticmethod
    def _sent(
        ledger: dict[tuple[str, int, int, str], dict[str, Any]],
        key: str,
        rank: int,
        subscriber: int,
        channel: str,
        since: str,
    ) -> bool:
        """Whether it reached ``subscriber`` at ``rank`` or above since ``since``."""
        for told in range(rank, max(RANKS.values()) + 1):
            sent = ledger.get((key, told, subscriber, channel))
            if sent and sent["status"] == "sent" and str(sent["at"]) >= since:
                return True
        return False

    def _attempt(
        self,
        key: str,
        rank: int,
        subscriber: int,
        channel: str,
        done: dict[str, int],
        send: Callable[[], object],
    ) -> None:
        error: str | None = None
        try:
            send()
        except Exception as exc:  # one address failing must not cost the others theirs
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
            logger.warning("watch: %s delivery of %s failed: %s", channel, key, error)
        self.store.record_delivery(key, rank, subscriber, channel, error=error, now=self._clock())
        done["failed" if error else "sent"] += 1


def _post_json(url: str, payload: dict[str, Any]) -> None:
    """POST ``payload`` to the operator's webhook through the egress gate."""
    from omni_mercury_engine.security.safe_http import SafeHTTPClient

    SafeHTTPClient.post(url, json_body=payload, timeout=20.0, user_configured=True).close()
