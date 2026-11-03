# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Transactional email delivery for account flows.

Verification and password-reset emails go through a small :class:`Mailer`
seam so the auth service never touches SMTP directly and stays fully testable
with no network and no secrets:

* :class:`ConsoleMailer` — files the message in a local outbox instead of
  sending it (``MERCURY_MAIL_OUTBOX``, default ``~/.local/state/mercury-agent/outbox``). The default
  when no SMTP host is configured, so local/dev/CI runs never try to reach a
  mail server.
* :class:`SmtpMailer` — real delivery. **Every credential is read from the
  environment, never hard-coded**, so secrets live only in the deployment's
  secret store (see the module note below), not in the repo.

Configuration (all via environment):

* ``MERCURY_SMTP_HOST``      — SMTP server hostname. If unset, delivery falls
  back to :class:`ConsoleMailer` (no email is sent; the outbox file is the delivery).
* ``MERCURY_SMTP_PORT``      — port (default ``587`` for STARTTLS; use ``465``
  for implicit TLS).
* ``MERCURY_SMTP_USERNAME``  — login user (for a Wix mailbox this is the full
  address, e.g. ``contact@mercuryagent.global``).
* ``MERCURY_SMTP_PASSWORD``  — the mailbox password / app password.
* ``MERCURY_SMTP_FROM``      — the From address (defaults to the username).
* ``MERCURY_SMTP_STARTTLS``  — ``"true"`` (default) to upgrade the connection
  with STARTTLS on the connect port; set ``"false"`` only when the port is
  already implicit-TLS (465).

These names are the whole contract between the code and the operator: the code
reads them, the deployment supplies them as secrets. Nothing here needs to
change to point at a different mailbox — only the environment does.
"""

from __future__ import annotations

import hashlib
import logging
import os
import smtplib
import ssl
from datetime import UTC, datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = [
    "OUTBOX_ENV",
    "ConsoleMailer",
    "Mailer",
    "SmtpMailer",
    "build_mailer",
    "default_outbox",
]

SMTP_HOST_ENV = "MERCURY_SMTP_HOST"
#: Directory :class:`ConsoleMailer` files messages in (default: :func:`default_outbox`).
OUTBOX_ENV = "MERCURY_MAIL_OUTBOX"


@runtime_checkable
class Mailer(Protocol):
    """Contract for sending a single transactional email."""

    def send(
        self,
        *,
        to: str,
        subject: str,
        body: str,
        html_body: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Send an email: plaintext ``body``, optional HTML alternative.

        ``headers`` carries deliverability metadata (e.g. ``List-Unsubscribe``)
        that the transport should attach verbatim.
        """
        ...


def default_outbox() -> Path:
    """Where :class:`ConsoleMailer` files messages when no outbox is configured.

    ``$XDG_STATE_HOME/mercury-agent/outbox`` (``~/.local/state`` by default),
    the same state root ``mercury-agent decouple status`` already inventories
    and ``decouple purge`` removes.
    """
    xdg_state = os.getenv("XDG_STATE_HOME", "").strip()
    root = Path(xdg_state) if xdg_state else Path.home() / ".local" / "state"
    return root / "mercury-agent" / "outbox"


class ConsoleMailer:
    """Mailer that files each message in a local outbox instead of sending it.

    The development and test default. Every message is written as one
    plain-text file (headers, blank line, body; an HTML alternative beside it)
    in ``outbox``, created owner-only, and the log carries the recipient,
    subject and path but never the body: the engine's log formatters redact
    credentials, so a verification or reset link logged there would arrive
    as ``token=<redacted>`` and the account could never be verified. The file
    is the delivery. A write failure propagates, so the caller's delivery
    audit records it like an SMTP outage.
    """

    def __init__(self, outbox: Path | str | None = None) -> None:
        """Use ``outbox``, or :data:`OUTBOX_ENV`, or :func:`default_outbox`."""
        chosen = outbox if outbox is not None else os.getenv(OUTBOX_ENV, "").strip() or None
        self.outbox = Path(chosen).expanduser() if chosen else default_outbox()

    def send(
        self,
        *,
        to: str,
        subject: str,
        body: str,
        html_body: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """File the message and log where it went; never touches the network."""
        self.outbox.mkdir(parents=True, exist_ok=True, mode=0o700)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        digest = hashlib.blake2b(f"{to}\n{subject}\n{body}".encode(), digest_size=4).hexdigest()
        path = self.outbox / f"{stamp}-{digest}.txt"
        lines = [f"To: {to}", f"Subject: {subject}", f"Date: {stamp}"]
        lines.extend(f"{name}: {value}" for name, value in (headers or {}).items())
        _write_private(path, "\n".join(lines) + "\n\n" + body)
        if html_body is not None:
            _write_private(path.with_suffix(".html"), html_body)
        logger.info("ConsoleMailer -> %s | %s | filed at %s", to, subject, path)


def _write_private(path: Path, text: str) -> None:
    """Create ``path`` readable by its owner only and write ``text`` to it."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


class SmtpMailer:
    """Mailer that delivers over SMTP using environment-supplied credentials.

    Instances are cheap and stateless; a fresh connection is opened per send so
    a dropped connection never wedges the sender.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        username: str | None,
        password: str | None,
        from_addr: str,
        use_starttls: bool,
        timeout: float = 15.0,
    ) -> None:
        """Store the connection parameters resolved from the environment."""
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._from_addr = from_addr
        self._use_starttls = use_starttls
        self._timeout = timeout

    def send(
        self,
        *,
        to: str,
        subject: str,
        body: str,
        html_body: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Deliver one message, opening and closing a fresh SMTP session.

        Raises:
            smtplib.SMTPException: If the server rejects the message or the
                connection fails (the caller decides whether that is fatal).
        """
        message = EmailMessage()
        message["From"] = self._from_addr
        message["To"] = to
        message["Subject"] = subject
        for name, value in (headers or {}).items():
            message[name] = value
        message.set_content(body)
        if html_body is not None:
            # multipart/alternative: clients render the HTML part, and the
            # plaintext part keeps deliverability scoring and text-only
            # clients working.
            message.add_alternative(html_body, subtype="html")

        context = ssl.create_default_context()
        if self._use_starttls:
            with smtplib.SMTP(self._host, self._port, timeout=self._timeout) as server:
                server.starttls(context=context)
                self._authenticate_and_send(server, message)
        else:
            with smtplib.SMTP_SSL(
                self._host, self._port, timeout=self._timeout, context=context
            ) as server:
                self._authenticate_and_send(server, message)

    def _authenticate_and_send(self, server: smtplib.SMTP, message: EmailMessage) -> None:
        """Log in (when credentials are set) and hand the message to the server."""
        if self._username and self._password:
            server.login(self._username, self._password)
        server.send_message(message)


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean-ish environment flag."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def build_mailer() -> Mailer:
    """Construct the configured mailer from the environment.

    Returns:
        A :class:`SmtpMailer` when ``MERCURY_SMTP_HOST`` is set, otherwise a
        :class:`ConsoleMailer` filing into ``MERCURY_MAIL_OUTBOX`` (so dev,
        tests, and un-provisioned deployments deliver to a local outbox
        instead of failing).
    """
    host = os.getenv(SMTP_HOST_ENV, "").strip()
    if not host:
        return ConsoleMailer()
    username = os.getenv("MERCURY_SMTP_USERNAME") or None
    from_addr = os.getenv("MERCURY_SMTP_FROM") or username or "no-reply@localhost"
    return SmtpMailer(
        host=host,
        port=int(os.getenv("MERCURY_SMTP_PORT", "587")),
        username=username,
        password=os.getenv("MERCURY_SMTP_PASSWORD") or None,
        from_addr=from_addr,
        use_starttls=_env_flag("MERCURY_SMTP_STARTTLS", default=True),
    )
