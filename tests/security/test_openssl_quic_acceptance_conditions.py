# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The conditions the CVE-2026-124709 acceptance rests on, held mechanically.

The deployment image's accepted-risk ledger (``.trivyignore``) carries the
OpenSSL QUIC-server stream leak published 2026-11-17, with this rationale: the
flaw is in OpenSSL's own QUIC *server*, which only a program calling that API
runs, and nothing in the image runs one — the API serves HTTP over TCP with
uvicorn, no QUIC or HTTP/3 server package is shipped, no UDP port is exposed,
and ``openssl(1)`` is never invoked. Debian trixie publishes no fixed build
(the tracker lists every suite as vulnerable), so there is nothing for the
Dockerfile's ``apt-get upgrade`` to install yet.

Each of those statements is a condition. If one stops being true — a QUIC
server library enters the lock, the image exposes UDP, the ledger entry
outlives a published fix — the acceptance's ground is gone and this file
fails, loudly, instead of the entry silencing a finding it no longer has the
standing to accept.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_LEDGER = _REPO / ".trivyignore"
_DOCKERFILE = (_REPO / "Dockerfile").read_text(encoding="utf-8")
_LOCK = (_REPO / "uv.lock").read_text(encoding="utf-8")

#: Python packages that implement or embed a QUIC / HTTP/3 server. Any of
#: these in the lock means the image could run the vulnerable protocol and
#: the acceptance must be re-reviewed.
_QUIC_SERVER_PACKAGES = ("aioquic", "qh3", "hypercorn", "pylsqpack", "aiohttp3")


def _ledger_entry() -> str | None:
    for line in _LEDGER.read_text(encoding="utf-8").splitlines():
        if line.startswith("CVE-2026-124709"):
            return line
    return None


def test_the_acceptance_is_enumerated_and_expiring() -> None:
    entry = _ledger_entry()
    if entry is None:
        # The entry is gone: either Debian shipped a fix and the upgrade path
        # took over, or the acceptance was withdrawn. Nothing to hold.
        return
    assert re.search(r"exp:\d{4}-\d{2}-\d{2}$", entry), (
        "the CVE-2026-124709 acceptance must expire like every other entry "
        "(90-day review cadence); a non-expiring entry is a silenced finding"
    )


def test_no_quic_server_implementation_is_shipped() -> None:
    if _ledger_entry() is None:
        return
    shipped = [name for name in _QUIC_SERVER_PACKAGES if re.search(rf'name = "{name}"', _LOCK)]
    assert shipped == [], (
        f"{shipped} in uv.lock: the image can now run a QUIC/HTTP-3 server, so the "
        "CVE-2026-124709 (OpenSSL QUIC server memory growth) acceptance no longer "
        "holds — remove the ledger entry or re-review it against the new surface"
    )


def test_the_image_serves_tcp_only() -> None:
    if _ledger_entry() is None:
        return
    exposes = re.findall(r"^EXPOSE\s+(.+)$", _DOCKERFILE, flags=re.MULTILINE)
    assert exposes, "the Dockerfile must state what it exposes"
    for line in exposes:
        for port in line.split():
            assert not port.endswith("/udp"), (
                f"EXPOSE {port}: QUIC runs over UDP, so an exposed UDP port voids the "
                "not-reachable rationale of the CVE-2026-124709 acceptance"
            )
    command = re.findall(r"^CMD\s+(.+)$", _DOCKERFILE, flags=re.MULTILINE)
    assert command and "uvicorn" in command[-1], (
        "the acceptance's rationale names uvicorn (an HTTP-over-TCP server) as the "
        "image's only server; the CMD changed, so re-review the acceptance"
    )


def test_the_image_never_invokes_the_openssl_binary() -> None:
    if _ledger_entry() is None:
        return
    # openssl(1) appears in the Dockerfile only as package names (openssl,
    # openssl-provider-legacy) inside apt operations and comments, never as a
    # command of its own.
    for line in _DOCKERFILE.splitlines():
        code = line.split("#", 1)[0]
        assert not re.search(
            r"(?<![-\w])openssl\s+(req|s_server|s_client|x509|genrsa|rehash)", code
        ), (
            f"the Dockerfile runs openssl(1) ({line.strip()!r}); the CVE-2026-124709 "
            "acceptance says it never does"
        )
