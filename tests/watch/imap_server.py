# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A small IMAP server for tests: real sockets, real TLS, real protocol.

The mailbox it serves is whatever a :class:`~tests.watch.smtp_server.SmtpServer`
accepted for one address, so a test can send a message through the
deployment's mailer and then look for it the way the deployment does: with
``imaplib``, over TLS from the first byte.

It speaks enough IMAP4rev1 (RFC 3501) for that: the greeting, CAPABILITY,
LOGIN, SELECT and EXAMINE, SEARCH with HEADER keys, NOOP and LOGOUT.
"""

from __future__ import annotations

import shlex
import socketserver
import ssl
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from email.message import EmailMessage
    from pathlib import Path


@dataclass
class ImapServer:
    """A listening IMAP-over-TLS server on a free local port.

    Attributes:
        certificate: PEM certificate the server presents.
        key: Its private key.
        messages: Returns the messages in the mailbox, oldest first.
        credentials: The one ``(username, password)`` LOGIN accepts.
        folders: The folders that exist; each holds the same messages.
        searches: Every SEARCH the server was sent, as its arguments.
    """

    certificate: Path
    key: Path
    messages: Callable[[], list[EmailMessage]]
    credentials: tuple[str, str] = ("operator@example.org", "mailbox-password")
    folders: tuple[str, ...] = ("INBOX",)
    searches: list[list[str]] = field(default_factory=list)
    port: int = 0
    _server: Any = None

    def __enter__(self) -> ImapServer:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(self.certificate, self.key)
        owner = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                connection = context.wrap_socket(self.request, server_side=True)
                stream = connection.makefile("rwb")

                def say(line: str) -> None:
                    stream.write(line.encode() + b"\r\n")
                    stream.flush()

                say("* OK IMAP4rev1 test server ready")
                signed_in = selected = False
                while True:
                    raw = stream.readline()
                    if not raw:
                        return
                    tag, _, rest = raw.decode("utf-8", "replace").rstrip("\r\n").partition(" ")
                    verb, _, rest = rest.partition(" ")
                    verb = verb.upper()
                    arguments = shlex.split(rest)
                    if verb == "CAPABILITY":
                        say("* CAPABILITY IMAP4rev1 AUTH=PLAIN")
                        say(f"{tag} OK CAPABILITY completed")
                    elif verb == "LOGIN":
                        if tuple(arguments[:2]) == owner.credentials:
                            signed_in = True
                            say(f"{tag} OK LOGIN completed")
                        else:
                            say(f"{tag} NO [AUTHENTICATIONFAILED] Invalid credentials")
                    elif verb in ("SELECT", "EXAMINE") and signed_in:
                        if arguments and arguments[0] in owner.folders:
                            selected = True
                            say(f"* {len(owner.messages())} EXISTS")
                            say("* 0 RECENT")
                            say("* OK [UIDVALIDITY 1] UIDs valid")
                            say(f"{tag} OK [READ-ONLY] {verb} completed")
                        else:
                            say(f"{tag} NO [NONEXISTENT] No such folder")
                    elif verb == "SEARCH" and selected:
                        owner.searches.append(arguments)
                        if len(arguments) == 3 and arguments[0].upper() == "HEADER":
                            found = [
                                str(number)
                                for number, message in enumerate(owner.messages(), start=1)
                                if arguments[2].lower()
                                in str(message.get(arguments[1], "")).lower()
                            ]
                            say("* SEARCH" + "".join(f" {number}" for number in found))
                            say(f"{tag} OK SEARCH completed")
                        else:
                            say(f"{tag} BAD Only HEADER searches are served")
                    elif verb == "NOOP":
                        say(f"{tag} OK NOOP completed")
                    elif verb == "LOGOUT":
                        say("* BYE Logging out")
                        say(f"{tag} OK LOGOUT completed")
                        return
                    else:
                        say(f"{tag} BAD Command not recognised or not allowed now")

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

            def handle_error(self, request: Any, client_address: Any) -> None:
                """A client that refuses the certificate hangs up mid-handshake: expected."""

        self._server = Server(("127.0.0.1", 0), Handler)
        self.port = int(self._server.server_address[1])
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *_: object) -> None:
        self._server.shutdown()
        self._server.server_close()
