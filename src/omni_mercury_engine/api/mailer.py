# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Transactional email delivery for account flows.

Verification and password-reset emails go through a small :class:`Mailer`
seam so the auth service never touches SMTP directly and stays fully testable
with no network and no secrets:

* :class:`ConsoleMailer` — logs the message instead of sending it. The default
  when no SMTP host is configured, so local/dev/CI runs never try to reach a
  mail server.
* :class:`SmtpMailer` — real delivery. **Every credential is read from the
  environment, never hard-coded**, so secrets live only in the deployment's
  secret store (see the module note below), not in the repo.

Configuration (all via environment):

* ``MERCURY_SMTP_HOST``      — SMTP server hostname. If unset, delivery falls
  back to :class:`ConsoleMailer` (no email is sent).
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

import logging
import os
import smtplib
import ssl
from email.message import EmailMessage
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = [
    "ConsoleMailer",
    "Mailer",
    "SmtpMailer",
    "build_mailer",
]

SMTP_HOST_ENV = "MERCURY_SMTP_HOST"


@runtime_checkable
class Mailer(Protocol):
    """Contract for sending a single plaintext email."""

    def send(self, *, to: str, subject: str, body: str) -> None:
        """Send an email to ``to`` with ``subject`` and plaintext ``body``."""
        ...


class ConsoleMailer:
    """Mailer that logs messages instead of sending them (dev/test default)."""

    def send(self, *, to: str, subject: str, body: str) -> None:
        """Log the message at INFO; never touches the network."""
        logger.info("ConsoleMailer -> %s | %s\n%s", to, subject, body)


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

    def send(self, *, to: str, subject: str, body: str) -> None:
        """Deliver one message, opening and closing a fresh SMTP session.

        Raises:
            smtplib.SMTPException: If the server rejects the message or the
                connection fails (the caller decides whether that is fatal).
        """
        message = EmailMessage()
        message["From"] = self._from_addr
        message["To"] = to
        message["Subject"] = subject
        message.set_content(body)

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
        :class:`ConsoleMailer` (so dev, tests, and un-provisioned deployments
        degrade to logging instead of failing).
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
