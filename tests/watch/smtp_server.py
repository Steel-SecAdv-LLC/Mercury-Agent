# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A small SMTP server for tests: real sockets, real TLS, real protocol.

The watch's email had only ever been "sent" to a Python object that recorded
the call. This server speaks enough ESMTP (RFC 5321) for ``smtplib`` to
deliver through it exactly as it would through a provider: greeting, EHLO,
STARTTLS (RFC 3207) or TLS from the first byte, AUTH PLAIN (RFC 4616),
MAIL/RCPT/DATA with dot-unstuffing, QUIT. It keeps what it was sent, and can
be told to refuse a recipient.

The certificate is minted per test with the system ``openssl`` (as
``tests/distributed/test_tcp_hardening.py`` does), so no key is committed.
"""

from __future__ import annotations

import base64
import shutil
import socketserver
import ssl
import subprocess
import threading
from dataclasses import dataclass, field
from email import message_from_bytes, policy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import socket
    from email.message import EmailMessage
    from pathlib import Path

OPENSSL = shutil.which("openssl")


def mint_certificate(directory: Path, name: str = "localhost") -> tuple[Path, Path]:
    """Mint a self-signed certificate for ``name``. Returns ``(certificate, key)``."""
    certificate, key = directory / f"{name}.crt", directory / f"{name}.key"
    subprocess.run(  # fixed argv, no shell
        [
            str(OPENSSL),
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "2",
            "-keyout",
            str(key),
            "-out",
            str(certificate),
            "-subj",
            f"/CN={name}",
            "-addext",
            f"subjectAltName=DNS:{name},IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )
    return certificate, key


@dataclass
class Received:
    """One message the server accepted."""

    sender: str
    recipients: list[str]
    message: EmailMessage
    encrypted: bool
    authenticated_as: str | None


@dataclass
class SmtpServer:
    """A listening SMTP server on a free local port.

    Attributes:
        certificate: PEM certificate the server presents.
        key: Its private key.
        implicit_tls: TLS from the first byte (port-465 style) instead of STARTTLS.
        credentials: The one ``(username, password)`` AUTH accepts.
        refuse: Recipients answered ``550``.
        received: Messages accepted, in order.
    """

    certificate: Path
    key: Path
    implicit_tls: bool = False
    credentials: tuple[str, str] = ("watch@example.org", "app-password")
    refuse: set[str] = field(default_factory=set)
    received: list[Received] = field(default_factory=list)
    port: int = 0
    _server: Any = None

    def __enter__(self) -> SmtpServer:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(self.certificate, self.key)
        owner = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                connection: socket.socket = self.request
                encrypted = False
                if owner.implicit_tls:
                    connection = context.wrap_socket(connection, server_side=True)
                    encrypted = True
                stream = connection.makefile("rwb")

                def say(line: str) -> None:
                    stream.write(line.encode() + b"\r\n")
                    stream.flush()

                user: str | None = None
                sender = ""
                recipients: list[str] = []
                say("220 localhost ESMTP test")
                while True:
                    raw = stream.readline()
                    if not raw:
                        return
                    line = raw.decode("utf-8", "replace").rstrip("\r\n")
                    verb, _, rest = line.partition(" ")
                    verb = verb.upper()
                    if verb in ("EHLO", "HELO"):
                        extensions = ["localhost", "8BITMIME", "SMTPUTF8"]
                        extensions.append("AUTH PLAIN" if encrypted else "STARTTLS")
                        for extension in extensions[:-1]:
                            say(f"250-{extension}")
                        say(f"250 {extensions[-1]}")
                    elif verb == "STARTTLS" and not encrypted:
                        say("220 Ready to start TLS")
                        connection = context.wrap_socket(connection, server_side=True)
                        stream, encrypted = connection.makefile("rwb"), True
                    elif verb == "AUTH" and encrypted:
                        mechanism, _, blob = rest.partition(" ")
                        _, name, secret = base64.b64decode(blob).decode().split("\x00")
                        if mechanism.upper() == "PLAIN" and (name, secret) == owner.credentials:
                            user = name
                            say("235 2.7.0 Authentication successful")
                        else:
                            say("535 5.7.8 Authentication credentials invalid")
                    elif verb == "MAIL":
                        if not encrypted:
                            say("530 5.7.0 Must issue a STARTTLS command first")
                            continue
                        sender, recipients = rest.partition(":")[2].split()[0].strip("<>"), []
                        say("250 2.1.0 OK")
                    elif verb == "RCPT":
                        address = rest.partition(":")[2].strip().strip("<>")
                        if address in owner.refuse:
                            say(
                                "550 5.1.1 The email account that you tried to reach does not exist"
                            )
                        else:
                            recipients.append(address)
                            say("250 2.1.5 OK")
                    elif verb == "DATA":
                        say("354 End data with <CR><LF>.<CR><LF>")
                        lines = []
                        while (data := stream.readline()) not in (b".\r\n", b""):
                            lines.append(data[1:] if data.startswith(b"..") else data)
                        owner.received.append(
                            Received(
                                sender,
                                list(recipients),
                                message_from_bytes(b"".join(lines), policy=policy.default),
                                encrypted,
                                user,
                            )
                        )
                        say("250 2.0.0 OK queued")
                    elif verb == "RSET":
                        sender, recipients = "", []
                        say("250 2.0.0 OK")
                    elif verb == "QUIT":
                        say("221 2.0.0 Bye")
                        return
                    else:
                        say("502 5.5.1 Command not implemented")

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
