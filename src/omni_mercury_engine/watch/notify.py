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
* **How.** By email through the deployment's mailer, to the console's alert
  dock through the alert bus, and to one operator webhook (a phone-push or
  chat service).
* **Once, and provably.** Every delivery is recorded. A finding is sent to
  a subscriber once per severity; one that failed is tried again on later
  passes, and the failure is shown, not swallowed. Delivery works from the
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
import math
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.watch.guidance import EMERGENCY_LINE

if TYPE_CHECKING:
    from collections.abc import Callable

    from omni_mercury_engine.api.mailer import Mailer
    from omni_mercury_engine.watch.model import Finding
    from omni_mercury_engine.watch.store import WatchStore

logger = logging.getLogger(__name__)

__all__ = ["AUTOMATIC", "MAX_ATTEMPTS", "Notifier", "audience", "distance_km", "notice"]

#: A delivery that has failed this many times is left as failed.
MAX_ATTEMPTS = 5
#: The operator webhook's entry in the delivery ledger.
WEBHOOK_SUBSCRIBER = 0
WEBHOOK_URL_ENV = "MERCURY_WATCH_WEBHOOK_URL"
WEBHOOK_TOPIC_ENV = "MERCURY_WATCH_WEBHOOK_TOPIC"
WEBHOOK_MIN_SEVERITY_ENV = "MERCURY_WATCH_WEBHOOK_MIN_SEVERITY"
PUBLIC_URL_ENV = "MERCURY_PUBLIC_URL"

_RANK = {"notable": 1, "significant": 2, "severe": 3}

#: Headers on every message the watch sends. RFC 3834: a message no person
#: wrote says so, and an out-of-office responder does not answer each alert.
AUTOMATIC = {"Auto-Submitted": "auto-generated"}


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points, in kilometres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    half = (
        math.sin((phi2 - phi1) / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0088 * math.asin(min(1.0, math.sqrt(half)))


def audience(
    finding: dict[str, Any], subscribers: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], float | None]]:
    """The subscribers a finding concerns, each with its distance where one is known."""
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
            if away <= float(subscriber["radius_km"]):
                out.append((subscriber, away))
        elif finding.get("kind") != "nws_alert" and not places:
            out.append((subscriber, None))  # nowhere in particular: everyone
    return out


def notice(
    finding: dict[str, Any], subscriber: dict[str, Any] | None = None, away: float | None = None
) -> tuple[str, str]:
    """The subject and body of the message that tells a person about a finding."""
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
    if finding.get("action"):
        lines += ["", "WHAT TO DO", str(finding["action"])]
        if finding.get("action_source"):
            lines.append(f"({finding['action_source']})")
    lines += ["", f"Source: {finding['source']}.", f"Basis: {finding['basis']}."]
    public = os.environ.get(PUBLIC_URL_ENV, "").strip().rstrip("/")
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


def _smtp_mailer() -> Mailer | None:
    """The deployment's SMTP mailer, or ``None`` when it has none."""
    from omni_mercury_engine.api.mailer import SMTP_HOST_ENV, build_mailer

    return build_mailer() if os.environ.get(SMTP_HOST_ENV, "").strip() else None


class Notifier:
    """Delivers the findings that stand to the people they concern."""

    def __init__(
        self,
        store: WatchStore,
        *,
        mailer: Mailer | None = None,
        post: Callable[[str, dict[str, Any]], None] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Build a notifier.

        Args:
            store: The watch's store: subscribers, findings, the ledger.
            mailer: Sends email. Defaults to the deployment's SMTP mailer;
                with none configured, email is not attempted.
            post: Sends the webhook's JSON body to its URL; replaceable in tests.
            clock: Returns the current UTC time; replaceable in tests.
        """
        self.store = store
        self.mailer = mailer if mailer is not None else _smtp_mailer()
        self._post = post if post is not None else _post_json
        self._clock = clock if clock is not None else (lambda: datetime.now(UTC))

    @property
    def webhook_url(self) -> str:
        """The operator's webhook, when one is set."""
        return os.environ.get(WEBHOOK_URL_ENV, "").strip()

    def channels(self) -> dict[str, bool]:
        """Which ways of reaching people this deployment has."""
        return {"email": self.mailer is not None, "webhook": bool(self.webhook_url)}

    def broadcast(self, finding: Finding) -> None:
        """Put a new finding on the alert bus, for consoles that are open now."""
        try:
            from omni_mercury_engine.narrative.alert_bus import get_alert_bus

            entry = finding.to_dict()
            rank = int(finding.severity)
            get_alert_bus().publish(
                {
                    "domain": f"watch:{finding.feed}",
                    "timestamp": self._clock().timestamp(),
                    "principal": None,
                    "detection": {
                        "anomaly_detected": True,
                        "detection_id": f"watch:{finding.key}:{rank}",
                        "severity": rank / 3,
                        "severity_label": entry["severity"],
                    },
                    "alert": {
                        "initiative_type": f"watch_{entry['severity']}",
                        "severity": rank / 3,
                        "domain": f"watch:{finding.feed}",
                        "message": finding.headline
                        + (f". {finding.action}" if finding.action else ""),
                        "timestamp": self._clock().timestamp(),
                        "event": entry,
                    },
                }
            )
        except Exception as exc:  # an open console is a courtesy; it must not stop delivery
            logger.warning("watch: could not broadcast %s: %s", finding.headline, exc)

    def sync(self, heartbeat: Callable[[], object] | None = None) -> dict[str, int]:
        """Deliver every standing finding to everyone it concerns and has not reached.

        Args:
            heartbeat: Called after each delivery attempt. A long pass (many
                subscribers, a slow mail server) uses it to renew the
                watch's lease, so no other watch takes the pass for dead.

        Returns:
            How many deliveries were ``sent`` and how many ``failed`` in this pass.
        """
        done = {"sent": 0, "failed": 0}
        subscribers = self.store.subscribers() if self.mailer is not None else []
        hook = self.webhook_url
        if not subscribers and not hook:
            return done
        try:
            least = int(os.environ.get(WEBHOOK_MIN_SEVERITY_ENV, "2"))
        except ValueError:
            least = 2
        ledger = self.store.standing_deliveries()
        for finding in self.store.standing(now=self._clock(), limit=1000):
            rank = _RANK.get(str(finding.get("severity")), 1)
            key = str(finding["key"])
            for subscriber, away in audience(finding, subscribers):
                if self._due(ledger, key, rank, int(subscriber["id"]), "email"):
                    subject, body = notice(finding, subscriber, away)
                    self._attempt(
                        key, rank, int(subscriber["id"]), "email", done,
                        lambda s=subscriber, t=subject, b=body: self.mailer.send(  # type: ignore[union-attr,misc]
                            to=s["email"], subject=t, body=b, headers=AUTOMATIC
                        ),
                    )  # fmt: skip
                    if heartbeat is not None:
                        heartbeat()
            if (
                hook
                and rank >= least
                and self._due(ledger, key, rank, WEBHOOK_SUBSCRIBER, "webhook")
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
                    lambda u=hook, p=payload: self._post(u, p),  # type: ignore[misc]
                )  # fmt: skip
        return done

    @staticmethod
    def _due(
        ledger: dict[tuple[str, int, int, str], dict[str, Any]],
        key: str,
        rank: int,
        subscriber: int,
        channel: str,
    ) -> bool:
        record = ledger.get((key, rank, subscriber, channel))
        return not record or (
            record["status"] == "failed" and int(record["attempts"]) < MAX_ATTEMPTS
        )

    def _attempt(
        self,
        key: str,
        rank: int,
        subscriber: int,
        channel: str,
        done: dict[str, int],
        send: Callable[[], None],
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
