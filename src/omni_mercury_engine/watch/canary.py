# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Proof that a message arrives: a test sent the way an alert is, and what came back.

The watch records a delivery as sent when the mail server or the text
provider accepted it. Accepted is not arrived: a message a provider takes
and then drops, files as spam or cannot hand to the handset is "sent" in the
ledger and was read by no one. Nothing in the watch could tell the two apart.

A canary is a message that says it is a test, sent through the same mailer,
the same text sender and the same webhook a finding is sent through. It is
*proven* only when something outside the watch says it arrived:

* **acknowledged**: someone opened the link in it, which this deployment
  serves, or typed its code into ``mercury-agent alerts ack``;
* **carrier**: the text provider reports the message delivered to the
  handset (:meth:`~omni_mercury_engine.watch.sms.TwilioSms.status`);
* **mailbox**: the message is found in the mailbox it was sent to, read over
  IMAP with credentials the operator supplies (:class:`ImapMailbox`).

A webhook that answers 2xx has *accepted* the message, which is all a
webhook can say; it is proven when someone opens the link.

The watch sends one every ``MERCURY_WATCH_CANARY_HOURS`` (24) to
``MERCURY_WATCH_CANARY_EMAIL`` and ``MERCURY_WATCH_CANARY_PHONE``, and to the
webhook when ``MERCURY_WATCH_CANARY_WEBHOOK`` is on; ``mercury-agent alerts
canary`` sends one now. One that is not proven within :data:`PROVE_WITHIN`
is *unproven*, and that is said where the watch says everything else: the
snapshot, the command line and the log. No address is kept with a canary:
only which channel it went by, when, and what came back.
"""

from __future__ import annotations

import imaplib
import logging
import os
import re
import ssl
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from omni_mercury_engine.watch.notify import AUTOMATIC, LEGACY_PUBLIC_URL_ENV, PUBLIC_URL_ENV
from omni_mercury_engine.watch.sms import E164, SmsRefusedError, SmsUnreachableError, two_segments
from omni_mercury_engine.watch.store import address_entry

if TYPE_CHECKING:
    from collections.abc import Callable

    from omni_mercury_engine.watch.notify import Notifier
    from omni_mercury_engine.watch.store import WatchStore

logger = logging.getLogger(__name__)

__all__ = [
    "ASK_EVERY",
    "ASK_FOR",
    "CHANNELS",
    "HEADER",
    "PROVE_WITHIN",
    "RETRY_FAILING",
    "Canary",
    "ImapMailbox",
    "Mailbox",
    "Targets",
    "build_mailbox",
    "proof_of_delivery",
    "targets",
]

EMAIL_ENV = "MERCURY_WATCH_CANARY_EMAIL"
PHONE_ENV = "MERCURY_WATCH_CANARY_PHONE"
WEBHOOK_ENV = "MERCURY_WATCH_CANARY_WEBHOOK"
HOURS_ENV = "MERCURY_WATCH_CANARY_HOURS"
IMAP_HOST_ENV = "MERCURY_WATCH_CANARY_IMAP_HOST"
IMAP_PORT_ENV = "MERCURY_WATCH_CANARY_IMAP_PORT"
IMAP_USERNAME_ENV = "MERCURY_WATCH_CANARY_IMAP_USERNAME"
IMAP_PASSWORD_ENV = "MERCURY_WATCH_CANARY_IMAP_PASSWORD"  # noqa: S105 - the variable's name
IMAP_FOLDER_ENV = "MERCURY_WATCH_CANARY_IMAP_FOLDER"

#: The ways a canary goes, which are the ways a finding goes.
CHANNELS = ("email", "sms", "webhook")
#: The header a canary email carries its id in, and a mailbox is searched for.
HEADER = "X-Mercury-Canary"
#: How long a canary has to be proven before it is called unproven. A mail
#: server that greylists holds a first message for up to about half an hour,
#: and a person asked to open a link is not at the keyboard at once.
PROVE_WITHIN = timedelta(hours=1)
#: How long the carrier and the mailbox go on being asked about a canary.
#: One that arrives after it was called unproven is still worth knowing of.
ASK_FOR = timedelta(hours=24)
#: The least time between two askings about one canary; it widens to a tenth
#: of the canary's age, so a day-old one is asked about every couple of hours
#: and not every minute.
ASK_EVERY = timedelta(minutes=1)
#: How soon a channel that refused a canary is sent another.
RETRY_FAILING = timedelta(hours=1)
#: How often the watch sends one when ``MERCURY_WATCH_CANARY_HOURS`` does not say.
EVERY_HOURS = 24.0
#: What a text provider calls a message that reached the handset, and one that will not.
_DELIVERED = frozenset({"delivered", "read"})
_NOT_DELIVERED = frozenset({"undelivered", "failed", "canceled"})


@dataclass(frozen=True)
class Targets:
    """Where the watch's own canary goes, from the deployment's settings.

    Attributes:
        email: The operator's address, or ``""``.
        phone: The operator's number in E.164, or ``""``.
        webhook: Whether the webhook is sent one too.
        every: How often.
    """

    email: str = ""
    phone: str = ""
    webhook: bool = False
    every: timedelta = timedelta(hours=EVERY_HOURS)

    def __bool__(self) -> bool:
        """Whether a canary is asked for at all."""
        return bool(self.email or self.phone or self.webhook)


def targets() -> Targets:
    """The deployment's canary settings. Unset, the watch sends none.

    A number that is not E.164 or an interval that is not a number is said in
    the log and left out: the watch runs without that canary, not without a
    watch.
    """
    phone = os.environ.get(PHONE_ENV, "").strip()
    if phone and not E164.fullmatch(phone):
        logger.error("watch: %s is not a number like +14155550123; no canary is texted", PHONE_ENV)
        phone = ""
    hours = EVERY_HOURS
    raw = os.environ.get(HOURS_ENV, "").strip()
    if raw:
        try:
            hours = float(raw)
        except ValueError:
            hours = 0.0
        if not 1.0 <= hours <= 24.0 * 31:
            logger.error(
                "watch: %s must be between 1 and 744 hours; using %g", HOURS_ENV, EVERY_HOURS
            )
            hours = EVERY_HOURS
    return Targets(
        email=os.environ.get(EMAIL_ENV, "").strip(),
        phone=phone,
        webhook=os.environ.get(WEBHOOK_ENV, "").strip().lower() in ("1", "true", "yes", "on"),
        every=timedelta(hours=hours),
    )


@runtime_checkable
class Mailbox(Protocol):
    """Somewhere a canary email can be looked for."""

    def holds(self, canary: str) -> bool:
        """Whether a message carrying the canary's id is in the mailbox."""
        ...


class ImapMailbox:
    """A mailbox read over IMAP with TLS from the first byte (RFC 8314)."""

    def __init__(
        self,
        *,
        host: str,
        username: str,
        password: str,
        port: int = 993,
        folder: str = "INBOX",
        timeout: float = 20.0,
        context: ssl.SSLContext | None = None,
    ) -> None:
        """Hold where the mailbox is and who reads it.

        Args:
            host: The IMAP server.
            username: The mailbox's login.
            password: Its password, or an app password.
            port: The server's IMAP-over-TLS port.
            folder: The folder a canary arrives in.
            timeout: Seconds to wait on the server.
            context: How the server's certificate is checked; the system's
                trust store when not given.
        """
        self._host, self._port, self._folder = host, port, folder
        self._username, self._password, self._timeout = username, password, timeout
        self._context = context

    def holds(self, canary: str) -> bool:
        """Whether a message with the canary's id in :data:`HEADER` is in the folder.

        Raises:
            OSError: If the server cannot be reached.
            imaplib.IMAP4.error: If it refuses the login, the folder or the search.
            ValueError: If ``canary`` is not an id this module issued.
        """
        if not re.fullmatch(r"[0-9a-f]{16}", canary):
            raise ValueError("not a canary's id")
        context = self._context or ssl.create_default_context()
        with imaplib.IMAP4_SSL(
            self._host, self._port, ssl_context=context, timeout=self._timeout
        ) as box:
            box.login(self._username, self._password)
            status, _ = box.select(self._folder, readonly=True)
            if status != "OK":
                raise imaplib.IMAP4.error(f"the folder {self._folder!r} could not be opened")
            status, found = box.search(None, "HEADER", HEADER, canary)
            if status != "OK":
                raise imaplib.IMAP4.error("the server refused the search")
            return bool(found and found[0] and found[0].split())


def build_mailbox() -> Mailbox | None:
    """The mailbox the deployment's canary email is looked for in, or ``None``."""
    host = os.environ.get(IMAP_HOST_ENV, "").strip()
    username = os.environ.get(IMAP_USERNAME_ENV, "").strip()
    password = os.environ.get(IMAP_PASSWORD_ENV, "")
    if not (host and username and password):
        return None
    try:
        port = int(os.environ.get(IMAP_PORT_ENV, "993"))
    except ValueError:
        logger.error("watch: %s is not a port; the canary's mailbox is not read", IMAP_PORT_ENV)
        return None
    return ImapMailbox(
        host=host,
        port=port,
        username=username,
        password=password,
        folder=os.environ.get(IMAP_FOLDER_ENV, "").strip() or "INBOX",
    )


class Canary:
    """Sends test messages the way findings are sent, and learns whether they arrived."""

    def __init__(
        self,
        store: WatchStore,
        notifier: Notifier,
        *,
        mailbox: Mailbox | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Build one over the watch's store and the notifier findings go through.

        Args:
            store: The watch's store.
            notifier: What delivers findings. Its mailer, its text sender and
                its webhook are the ones a canary is sent with.
            mailbox: Where a canary email is looked for. Defaults to the
                deployment's (:func:`build_mailbox`).
            clock: Returns the current UTC time; replaceable in tests.
        """
        self.store = store
        self.notifier = notifier
        self.mailbox = mailbox if mailbox is not None else build_mailbox()
        self._clock = clock if clock is not None else (lambda: datetime.now(UTC))
        #: Canaries this watch has already said were not proven: said once each.
        self._said: set[str] = set()
        #: When each open canary was last asked about, and when the watch last looked.
        self._asked: dict[str, datetime] = {}
        self._looked_at: datetime | None = None
        #: Channels a canary is asked for by and this deployment lacks: said once each.
        self._cannot: set[str] = set()

    def send(
        self, *, email: str = "", phone: str = "", webhook: bool = False
    ) -> list[dict[str, Any]]:
        """Send one canary by each channel named, and record what the channel said.

        Returns:
            One entry per channel: its ``channel``, the canary's ``id``, the
            ``code`` that acknowledges it by hand, and ``error`` when the
            channel did not take it. The code is returned here and nowhere
            else: the store keeps a hash of it.

        Raises:
            ValueError: If a channel is named that this deployment cannot
                send by, or ``phone`` is not E.164.
        """
        if phone and not E164.fullmatch(phone):
            raise ValueError("a phone number must be written like +14155550123")
        have = self.notifier.channels()
        asked = {"email": bool(email), "sms": bool(phone), "webhook": webhook}
        missing = [channel for channel in CHANNELS if asked[channel] and not have[channel]]
        if missing:
            raise ValueError(f"this deployment cannot send by {', '.join(missing)}")
        sent = []
        for channel, address in (("email", email), ("sms", phone), ("webhook", "webhook")):
            if not asked[channel]:
                continue
            now = self._clock()
            issued = self.store.issue_canary(channel, address_entry(address), now)
            receipt, error = "", None
            try:
                receipt = self._deliver(channel, address, issued, now) or ""
            except Exception as exc:  # whatever the channel raised is the finding
                error = f"{type(exc).__name__}: {str(exc)[:200]}"
                logger.error("watch: the %s canary could not be sent: %s", channel, error)
            self.store.record_canary(issued["id"], receipt=receipt, error=error)
            sent.append(
                {"channel": channel, "id": issued["id"], "code": issued["code"], "error": error}
            )
        return sent

    def _deliver(self, channel: str, address: str, issued: dict[str, str], now: datetime) -> str:
        """Send one, by the same means a finding goes. Returns the channel's receipt."""
        link = _link(issued["token"])
        how = (
            f"Open this to confirm it arrived: {link}"
            if link
            else f"Confirm it arrived with: mercury-agent alerts ack {issued['code']}"
        )
        if channel == "email":
            mailer = self.notifier.mailer
            if mailer is None:  # pragma: no cover - send() has checked
                raise RuntimeError("no mailer is configured")
            mailer.send(
                to=address,
                subject="[Mercury watch] Test: is this reaching you?",
                body=(
                    "This is a test from the Mercury watch. Nothing has happened.\n\n"
                    "It was sent the way an alert is sent, to find out whether one would "
                    f"reach you. {how}\n\n"
                    f"Sent {now:%Y-%m-%d %H:%M} UTC. Test {issued['id']}, code {issued['code']}."
                ),
                headers={**AUTOMATIC, HEADER: issued["id"]},
            )
            return "accepted by the mail server"
        if channel == "sms":
            sender = self.notifier.sms
            if sender is None:  # pragma: no cover - send() has checked
                raise RuntimeError("no text-message sender is configured")
            text = f"Mercury watch test. Nothing has happened. Code {issued['code']}."
            return sender.send(to=address, body=two_segments(f"{text} {link}".strip())) or ""
        self.notifier.post_webhook(
            {
                "title": "[Mercury watch] Test: is this reaching you?",
                "message": f"This is a test from the Mercury watch. Nothing has happened. {how}",
                "text": f"*Mercury watch test.* Nothing has happened. {how}",
                "priority": 2,
                "tags": ["test"],
            }
        )
        return "accepted by the webhook"

    def acknowledge(self, *, token: str = "", code: str = "") -> bool:
        """Prove the canary a link's ``token`` or a typed ``code`` belongs to.

        Returns:
            Whether there was one. Acknowledging one twice is still ``True``.
        """
        return self.store.prove_canary(token=token, code=code, by="acknowledged", now=self._clock())

    def settle(self, every: timedelta = ASK_EVERY) -> dict[str, int]:
        """Ask the carrier and the mailbox about every canary that is still open.

        Args:
            every: The least time between two askings about one canary by
                this object (see :data:`ASK_EVERY`).

        Returns:
            How many were ``proven`` and how many ``failed`` by what was learned.
        """
        learned = {"proven": 0, "failed": 0}
        now = self._clock()
        sender = self.notifier.sms
        status_of = getattr(sender, "status", None)
        for canary in self.store.open_canaries(now - ASK_FOR):
            asked = self._asked.get(canary["id"])
            age = now - datetime.fromisoformat(canary["sent_at"])
            if asked is not None and now - asked < max(every, age / 10):
                continue
            self._asked[canary["id"]] = now
            try:
                if canary["channel"] == "sms" and canary["receipt"] and callable(status_of):
                    status = str(status_of(canary["receipt"]))
                    if status in _DELIVERED:
                        self.store.prove_canary(canary=canary["id"], by="carrier", now=now)
                        learned["proven"] += 1
                    elif status in _NOT_DELIVERED:
                        self.store.record_canary(
                            canary["id"],
                            receipt=canary["receipt"],
                            error=f"the carrier reports it {status}",
                        )
                        learned["failed"] += 1
                        logger.error("watch: the text canary was not delivered (%s)", status)
                elif canary["channel"] == "email" and self.mailbox is not None:
                    if self.mailbox.holds(canary["id"]):
                        self.store.prove_canary(canary=canary["id"], by="mailbox", now=now)
                        learned["proven"] += 1
            except (
                OSError, imaplib.IMAP4.error, SmsRefusedError, SmsUnreachableError, ValueError
            ) as exc:  # fmt: skip
                # Not being able to ask is not an answer either way: the
                # canary stays open and becomes unproven with time.
                logger.warning(
                    "watch: could not ask whether the %s canary arrived: %s: %s",
                    canary["channel"], type(exc).__name__, str(exc)[:200],
                )  # fmt: skip
        return learned

    def due(self, wanted: Targets) -> set[str]:
        """The channels the watch owes a canary by.

        One it has not sent a canary by for ``wanted.every``; and one whose
        last canary the channel itself refused, after :data:`RETRY_FAILING`,
        so that a mail server that was down for an hour is not called
        failing for a day.
        """
        now = self._clock()
        latest = self.store.latest_canaries()
        owed = set()
        for channel, asked in (
            ("email", wanted.email), ("sms", wanted.phone), ("webhook", wanted.webhook)
        ):  # fmt: skip
            last = latest.get(channel)
            if not asked:
                continue
            if last is None:
                owed.add(channel)
                continue
            wait = min(wanted.every, RETRY_FAILING) if last["error"] else wanted.every
            if now - datetime.fromisoformat(last["sent_at"]) >= wait:
                owed.add(channel)
        return owed

    def run(self, leads: Callable[[], bool] = lambda: True) -> None:
        """What the watch does on each pass: ask about open canaries, send one when due.

        Costs nothing when no canary is asked for, and two small reads of the
        store at most once in :data:`ASK_EVERY` when one is.

        Args:
            leads: Asked only when there is something to do; ``False`` means
                another watch of this store tells people, and sends the
                canaries too.
        """
        now = self._clock()
        if self._looked_at is not None and now - self._looked_at < ASK_EVERY:
            return
        self._looked_at = now
        wanted = targets()
        if not wanted:
            return
        owed = self.due(wanted)
        if not (owed or self.store.open_canaries(now - ASK_FOR)) or not leads():
            return
        self.settle()
        for channel, shown in proof_of_delivery(self.store, now, detail=True).items():
            if shown["state"] in ("unproven", "failing") and shown["id"] not in self._said:
                self._said.add(shown["id"])
                logger.error(
                    "watch: the %s canary sent %s is %s: nothing shows that an alert sent "
                    "by %s would arrive%s",
                    channel, shown["sent_at"], shown["state"], channel,
                    f" ({shown['error']})" if shown.get("error") else "",
                )  # fmt: skip
        have = self.notifier.channels()
        for channel in sorted(owed):
            if not have[channel] and channel not in self._cannot:
                self._cannot.add(channel)
                logger.error(
                    "watch: a %s canary is asked for and this deployment cannot send by %s",
                    channel, channel,
                )  # fmt: skip
        owed = {channel for channel in owed if have[channel]}
        if owed:
            self.send(
                email=wanted.email if "email" in owed else "",
                phone=wanted.phone if "sms" in owed else "",
                webhook="webhook" in owed,
            )


def _link(token: str) -> str:
    """Where a canary is acknowledged, when this deployment says where it is reached."""
    public = (
        (os.environ.get(PUBLIC_URL_ENV) or os.environ.get(LEGACY_PUBLIC_URL_ENV) or "")
        .strip()
        .rstrip("/")
    )
    return f"{public}/api/v1/watch/ack/{token}" if public else ""


def proof_of_delivery(
    store: WatchStore, now: datetime | None = None, *, detail: bool = False
) -> dict[str, dict[str, Any]]:
    """For each channel, what its latest canary shows. No address is in it.

    A channel's ``state`` is one of:

    * ``never``: no canary has been sent by it;
    * ``failing``: the channel refused the latest one, or its carrier
      reports it not delivered;
    * ``awaiting``: the latest was accepted less than :data:`PROVE_WITHIN`
      ago and nothing has yet said it arrived;
    * ``unproven``: it was accepted longer ago than that and nothing has;
    * ``proven``: something outside the watch said it arrived (``by``).

    ``last_proven_at`` is the newest canary of the channel that was proven,
    which can be older than the latest one sent. With ``detail``, the
    canary's ``id`` and the channel's own ``error`` are given too: an error
    is a mail server's or a provider's words, which can quote the address,
    so the public snapshot is served without it.
    """
    at = now if now is not None else datetime.now(UTC)
    shown: dict[str, dict[str, Any]] = {}
    latest = store.latest_canaries()
    for channel in CHANNELS:
        canary = latest.get(channel)
        if canary is None:
            shown[channel] = {"state": "never"}
            continue
        sent_at = datetime.fromisoformat(canary["sent_at"])
        if canary["proven_at"]:
            state = "proven"
        elif canary["error"]:
            state = "failing"
        else:
            state = "awaiting" if at - sent_at < PROVE_WITHIN else "unproven"
        shown[channel] = {
            "state": state,
            "sent_at": canary["sent_at"],
            "proven_at": canary["proven_at"],
            "by": canary["proven_by"],
            "last_proven_at": canary["last_proven_at"],
        }
        if detail:
            shown[channel].update(id=canary["id"], error=canary["error"])
    return shown
