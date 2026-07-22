# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Trusted-proxy client IP resolution for rate limiting and audit.

Deriving the client identity from ``X-Forwarded-For`` is only safe when the
value is read from the *right* position. The header is client-writable: a
caller can send ``X-Forwarded-For: 1.2.3.4`` and, if the server naively takes
the left-most entry, rotate that value to mint a fresh rate-limit bucket per
request — a total rate-limiter bypass. The entries a deployment may trust are
exactly the ones appended by its *own* proxies, which are the right-most hops.

The rule implemented here (the standard one — see e.g. the MDN
``X-Forwarded-For`` guidance and nginx's ``real_ip_recursive``):

* ``MERCURY_TRUSTED_PROXY_HOPS = 0`` (default): the header is untrusted and
  ignored entirely; the TCP peer address is the client. Correct for a server
  exposed directly to the internet.
* ``MERCURY_TRUSTED_PROXY_HOPS = N`` (N >= 1): the deployment guarantees the
  request traverses exactly N trusted proxies, each appending one entry. The
  client is the N-th entry *from the right*. Anything further left is
  client-supplied noise and never consulted.

Every resolved value is validated as an IP literal (``ipaddress`` — no DNS,
no ambiguity) and normalised, so a spoofed garbage header can neither poison
rate-limit keys nor smuggle log-injection payloads into audit trails. On any
malformed input the resolver fails closed to the directly connected peer
address.
"""

from __future__ import annotations

import ipaddress
import os

__all__ = [
    "TRUSTED_PROXY_HOPS_ENV",
    "resolve_client_ip",
    "trusted_proxy_hops",
]

#: Environment variable declaring how many trailing ``X-Forwarded-For``
#: entries were appended by this deployment's own proxy tier.
TRUSTED_PROXY_HOPS_ENV = "MERCURY_TRUSTED_PROXY_HOPS"


def trusted_proxy_hops() -> int:
    """Read the configured number of trusted proxy hops (default 0).

    Returns:
        The non-negative hop count. A malformed or negative value degrades to
        0 (header untrusted) — the fail-closed direction: misconfiguration
        must never widen trust.
    """
    raw = os.getenv(TRUSTED_PROXY_HOPS_ENV, "0").strip()
    try:
        hops = int(raw)
    except ValueError:
        return 0
    return max(0, hops)


def _normalize_ip(candidate: str) -> str | None:
    """Validate and canonicalise one IP literal, or return ``None``.

    Accepts bare IPv4/IPv6 literals plus the two forms proxies commonly
    forward: an IPv4 ``host:port`` pair and a bracketed ``[v6]:port`` pair.

    Args:
        candidate: A single ``X-Forwarded-For`` list entry, already split.

    Returns:
        The canonical ``str(ip_address(...))`` form, or ``None`` if the entry
        is not a valid IP literal.
    """
    entry = candidate.strip()
    if not entry:
        return None
    # [v6]:port / [v6]
    if entry.startswith("["):
        closing = entry.find("]")
        if closing == -1:
            return None
        entry = entry[1:closing]
    elif entry.count(":") == 1 and "." in entry:
        # IPv4 host:port (a lone colon with dots present cannot be IPv6).
        entry = entry.split(":", 1)[0]
    try:
        return str(ipaddress.ip_address(entry))
    except ValueError:
        return None


def resolve_client_ip(
    direct_peer: str | None,
    forwarded_for: str | None,
    *,
    hops: int | None = None,
) -> str:
    """Resolve the real client IP behind ``hops`` trusted proxies.

    Args:
        direct_peer: The TCP peer address of the connection (``request.client
            .host``), or ``None`` when unavailable (some test transports).
        forwarded_for: The raw ``X-Forwarded-For`` header value, or ``None``.
        hops: Trusted proxy hop count; defaults to :func:`trusted_proxy_hops`
            (the ``MERCURY_TRUSTED_PROXY_HOPS`` environment variable).

    Returns:
        The canonical client IP string. Resolution order:

        1. ``hops == 0`` or no header → the validated direct peer.
        2. ``hops >= 1`` → the ``hops``-th entry from the right of the
           header, validated as an IP literal.
        3. Any malformed/missing candidate → fall back to the direct peer.
        4. No usable value at all → the sentinel ``"unknown"`` (a single
           shared bucket — throttles rather than bypasses).
    """
    if hops is None:
        hops = trusted_proxy_hops()

    fallback = _normalize_ip(direct_peer) if direct_peer else None

    if hops <= 0 or not forwarded_for:
        return fallback or "unknown"

    entries = [part for part in forwarded_for.split(",")]
    if len(entries) < hops:
        # Fewer entries than trusted hops: the request did not come through
        # the declared proxy tier (direct hit on the origin). The peer *is*
        # the client — and is possibly an attacker probing with a forged
        # header, which must not be honoured.
        return fallback or "unknown"

    candidate = _normalize_ip(entries[len(entries) - hops])
    return candidate or fallback or "unknown"
