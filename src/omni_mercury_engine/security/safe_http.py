# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""SafeHTTPClient -- the single egress point for outbound HTTP in Mercury Agent.

This module replaces every ad-hoc ``urllib.request.urlopen`` call in
``src/`` with a centrally enforced gate built on the ``requests``
library.  Centralising egress lets the static-analysis surface for
``B310 urllib_urlopen`` collapse to zero: ``requests`` is not on the
bandit dangerous-call list, and every user-configurable URL is
validated before the network call happens.

Gates enforced on every call:

1. **Scheme allowlist** -- only ``https://`` is accepted by default.
   ``http://`` is rejected unless the caller passes
   ``allow_http=True`` (reserved for documented research mirrors
   that publish over plain HTTP; never accepted for arbitrary user
   input).

2. **TRUSTED_DOMAINS allowlist** -- the host must be in
   :attr:`TrustedEndpoints.TRUSTED_DOMAINS` for class-constant
   dataset URLs.  User-configured endpoints (Ollama base_url, SearXNG
   instance, custom inference backends) skip the allowlist but still
   pass the loopback / private-network gate (see #3).

3. **Private-network / IMDS block for user-configured URLs** --
   when ``user_configured=True``, the resolved host is checked
   against RFC1918, link-local (169.254/16, including the AWS /
   GCP / Azure IMDS at 169.254.169.254), loopback, and IPv6 ULA
   ranges.  This blocks SSRF pivots to the metadata service or
   internal infrastructure.

4. **Loopback-only enforcement for on-box adapters** -- callers
   that are talking to a local daemon (Ollama at 127.0.0.1:11434,
   Redis sidecar) pass ``loopback_only=True``; any non-loopback
   host raises immediately.

The result is that the only ``urlopen`` call in ``src/`` lives in
this module's tests (and even that uses ``requests``); the original
B310 finding has nowhere left to fire.

Usage
-----

For trusted-allowlist GET (the dataset / API loader case)::

    from omni_mercury_engine.security.safe_http import SafeHTTPClient

    body: bytes = SafeHTTPClient.get_bytes(
        "https://earthquake.usgs.gov/fdsnws/event/1/query",
        params={"format": "geojson", "limit": "100"},
        timeout=30,
    )

For a user-configured base URL (the Ollama / SearXNG case)::

    text: str = SafeHTTPClient.post_json(
        f"{ollama_base_url}/api/generate",
        json_body={"model": "llama3", "prompt": "hi"},
        timeout=30,
        user_configured=True,
        loopback_only=True,
    )

The helpers always raise on a 4xx/5xx response (``raise_for_status``)
and always emit a ``User-Agent`` header.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode, urlparse

if TYPE_CHECKING:
    import requests

from omni_mercury_engine.security.input_validation import TrustedEndpoints
from omni_mercury_engine.security.redaction import (
    redact_env_secrets,
    redact_text,
    redact_url,
)


def _redact_full(text: str) -> str:
    """Canonical scrub for operator-facing diagnostics.

    Structural redaction (``redact_url`` / ``redact_text``) removes
    credential-named query values and userinfo; the env-value pass on
    top catches credentials that ride in URL PATH segments (the FIRMS
    MAP-key shape), which no structural rule can recognise.  Applied to
    every URL or transport-exception text this module surfaces.
    """
    return redact_env_secrets(text)


logger = logging.getLogger(__name__)

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; Mercury-Agent/1.0; "
    "+https://github.com/Steel-SecAdv-LLC/Mercury-Agent)"
)

# Schemes the gate will ever consider. ``file://``, ``ftp://``,
# ``gopher://``, ``data://`` are never permitted; the gate raises
# before any handler runs.
_HTTPS_ONLY = frozenset({"https"})
_HTTP_AND_HTTPS = frozenset({"http", "https"})


class UnsafeURLError(ValueError):
    """An outbound URL violated a SafeHTTPClient gate."""


def _parse_and_check_scheme(url: str, *, allow_http: bool) -> tuple[str, str]:
    """Return (scheme, host) after enforcing the scheme allowlist.

    Raises:
        UnsafeURLError: scheme is not in the configured allowlist
            (or netloc is empty so we cannot extract a host).
    """
    parsed = urlparse(url)
    allowed = _HTTP_AND_HTTPS if allow_http else _HTTPS_ONLY
    # Refusal messages carry the URL credential-REDACTED: a keyed URL that
    # trips a gate would otherwise leak its credential into exception text
    # (the exact channel the loader-layer chain-severing fix closed).
    if parsed.scheme not in allowed:
        raise UnsafeURLError(
            f"SafeHTTPClient: refusing URL '{_redact_full(redact_url(url))}' -- scheme "
            f"'{parsed.scheme}' is not in the allowlist {sorted(allowed)}."
        )
    host = parsed.hostname or ""
    if not host:
        raise UnsafeURLError(
            f"SafeHTTPClient: URL '{_redact_full(redact_url(url))}' has no host component."
        )
    return parsed.scheme, host


def _resolve_ips(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Resolve a host to its IPs for the private-network gate.

    IP literals are returned as-is so the gate cannot be bypassed
    by passing ``127.0.0.1`` directly.  Hostnames are resolved via
    ``getaddrinfo`` so the gate sees every A/AAAA the system would
    actually connect to (including DNS-rebinding attempts).

    Raises:
        UnsafeURLError: the host did not resolve.
    """
    try:
        return [ipaddress.ip_address(host)]
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as exc:
        raise UnsafeURLError(f"SafeHTTPClient: host '{host}' did not resolve: {exc}.") from exc
    ips: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in infos:
        sockaddr = info[4]
        ip_str = sockaddr[0]
        if isinstance(ip_str, str):
            try:
                ips.append(ipaddress.ip_address(ip_str))
            except ValueError:
                continue
    if not ips:
        raise UnsafeURLError(f"SafeHTTPClient: host '{host}' resolved to no usable IPs.")
    return ips


def _is_private_or_imds(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if ``ip`` is in a range we refuse for user-configured URLs.

    Covers RFC1918 (10/8, 172.16/12, 192.168/16), link-local
    (169.254/16, which includes the AWS / GCP / Azure IMDS at
    169.254.169.254), loopback (127/8, ::1), the IPv4-mapped
    equivalents of those, IPv6 ULA (fc00::/7), IPv6 link-local
    (fe80::/10), the unspecified address (0.0.0.0, ::), AND the
    RFC 6598 shared address space (100.64.0.0/10, CGNAT).

    ``ipaddress`` does not classify the RFC 6598 block as private or
    reserved, so the convenience attributes used elsewhere on the
    standard library type miss it. A user-configured URL resolving
    to a CGNAT / internal shared address would otherwise pass the
    SSRF gate; we add the network explicitly here.
    """
    if isinstance(ip, ipaddress.IPv4Address) and ip in _SHARED_CGNAT_V4:
        return True
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_unspecified
        or ip.is_reserved
        or ip.is_multicast
    )


# RFC 6598 shared CGNAT address space. Promoted to a module-level
# constant so both the "private/IMDS" predicate (default policy) and
# the "always-blocked" set (defence-in-depth for ``allow_private=True``)
# can reference the same range. Network ISPs use this block for carrier
# NAT, and corporate environments sometimes use it for internal services;
# either way it is not public Internet and a user-configured URL pointed
# at it is an SSRF pivot we refuse.
_SHARED_CGNAT_V4 = ipaddress.IPv4Network("100.64.0.0/10")

# IPv4 and IPv6 ranges that we refuse even when the caller opted into
# ``allow_private=True`` for an on-VPC deployment.  The link-local
# block (169.254/16) is the AWS / GCP / Azure metadata service home --
# RFC1918 lateral movement is one thing, but ``169.254.169.254`` is
# the actual SSRF prize and we never permit it.  The loopback,
# unspecified, and shared-CGNAT ranges are kept on the refuse-list
# because they cannot correspond to a real on-VPC service either.
_ALWAYS_BLOCKED_V4 = (
    ipaddress.IPv4Network("169.254.0.0/16"),
    ipaddress.IPv4Network("127.0.0.0/8"),
    ipaddress.IPv4Network("0.0.0.0/8"),
    ipaddress.IPv4Network("224.0.0.0/4"),  # multicast
    ipaddress.IPv4Network("240.0.0.0/4"),  # reserved
    _SHARED_CGNAT_V4,  # RFC 6598 shared / CGNAT
)
_ALWAYS_BLOCKED_V6 = (
    ipaddress.IPv6Network("::1/128"),  # loopback
    ipaddress.IPv6Network("fe80::/10"),  # link-local
    ipaddress.IPv6Network("ff00::/8"),  # multicast
    ipaddress.IPv6Network("::/128"),  # unspecified
)


def _is_always_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if ``ip`` is in a range refused even with ``allow_private=True``.

    See :data:`_ALWAYS_BLOCKED_V4` / ``_V6`` for the policy.  The
    point of the carve-out is to let operators reach a SearXNG /
    Ollama / internal-API host on RFC1918 from inside a private VPC,
    while still slamming the door on the cloud metadata service and
    on the obviously-bogus reserved / loopback / multicast ranges.
    """
    if isinstance(ip, ipaddress.IPv4Address):
        return any(ip in net for net in _ALWAYS_BLOCKED_V4)
    return any(ip in net for net in _ALWAYS_BLOCKED_V6)


def _is_loopback(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if ``ip`` is on 127/8 or ``::1``."""
    return ip.is_loopback


def _host_ip_literal(
    host: str,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Return ``host`` as an IP address if it is an IP literal, else None.

    Used by the VPC-air-gap gate to classify a target *without DNS*: a public IP
    literal is refused pre-resolution, while a private literal / named host is
    deferred to the private-only resolve (which fails closed if DNS is down).
    """
    h = host.strip().strip("[]")
    try:
        return ipaddress.ip_address(h)
    except ValueError:
        return None


def _is_loopback_host(host: str) -> bool:
    """Return True if ``host`` is a loopback target decidable without DNS.

    Recognises loopback IP literals (127/8, ``::1``) and the literal
    ``localhost`` name only. The air-gap gate uses this to permit
    on-box adapters (a local Ollama model, a Redis sidecar) while
    performing **no** network resolution -- so the offline guarantee
    holds even when no resolver is reachable. A hostname that is not a
    loopback literal is treated as non-loopback here; the strict
    ``loopback_only`` gate still resolves and re-checks it downstream
    when the caller has opted into on-box enforcement.

    ``*.localhost`` SUBDOMAINS are deliberately **not** recognised:
    RFC 6761 only says resolvers SHOULD answer them locally, and on
    glibc without systemd-resolved they are forwarded to the configured
    resolver -- an ``/etc/hosts`` entry, a dnsmasq wildcard, or a
    hostile resolver can map ``foo.localhost`` to a public address,
    which would turn this DNS-free permit into an egress bypass under
    ``MERCURY_OFFLINE`` for callsites that hand the URL to their own
    transport. Operators should address on-box services as ``localhost``
    or a loopback IP literal (the documented carve-out).
    """
    h = host.strip().strip("[]").lower()  # tolerate bracketed IPv6 literals
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def enforce_offline_egress(url: str) -> None:
    """Air-gap gate for callsites that keep their own HTTP transport.

    Most egress goes through :class:`SafeHTTPClient`, whose
    :meth:`~SafeHTTPClient.validate_url` enforces ``MERCURY_OFFLINE``
    already.  A handful of callsites legitimately keep a different
    transport (httpx sources, aiohttp adapters, the wfdb PhysioNet
    downloader, webhook callbacks) and must apply the same policy
    before their transport opens a socket.  This helper is that policy:
    under ``MERCURY_OFFLINE`` every non-loopback destination raises,
    decided **without DNS** (the same pre-resolution predicate the
    ``SafeHTTPClient`` gate uses) so the refusal holds where no
    resolver is reachable, while a loopback target (a local Ollama
    model, a Prometheus sidecar) stays permitted.

    Args:
        url: The destination the caller is about to contact.

    Raises:
        OfflineModeError: ``MERCURY_OFFLINE`` is set and ``url`` does
            not target a loopback host.
    """
    # Lazy import -- the codebase-wide pattern -- so this security
    # primitive never pulls the heavy datasets package at module load.
    from omni_mercury_engine.datasets.exceptions import (
        OfflineModeError,
        offline_mode_active,
    )

    if not offline_mode_active():
        return
    host = urlparse(url).hostname or ""
    if not _is_loopback_host(host):
        raise OfflineModeError(url)


class _PinnedDNSHTTPAdapter:
    """Lazy proxy for the real requests-based HTTPAdapter.

    The actual subclass is built the first time :meth:`build` is
    called so the ``requests`` import stays deferred. Importing
    safe_http should not require ``requests`` (see commentary near
    ``_request`` below).
    """

    @staticmethod
    def build(hostname: str, validated_ip: str) -> Any:
        """Build a requests HTTPAdapter that pins the TCP target to ``validated_ip``.

        The adapter overrides connection acquisition so urllib3
        connects to ``validated_ip`` (the pre-vetted address) while
        TLS SNI and certificate verification still use ``hostname``
        (the operator-meaningful name on the cert). The result: DNS
        cannot be rebound between the SafeHTTPClient validation
        step and the actual TCP connect, because the second DNS
        lookup that ``requests`` would otherwise do never happens.

        We work in three steps inside the adapter:

        1. ``poolmanager.connection_from_host(host=ip, ...)`` returns
           a connection pool keyed on the IP (not the hostname).
        2. ``pool_kwargs`` carries ``server_hostname`` and
           ``assert_hostname`` so the TLS handshake validates the
           certificate against the original name, not the IP literal.
        3. ``Host:`` header on the outgoing HTTP request is set to
           the hostname (urllib3 derives it from the URL by default;
           we never substitute the IP into the URL, so this is free).
        """
        import requests
        from urllib3 import HTTPConnectionPool, HTTPSConnectionPool
        from urllib3.poolmanager import PoolManager

        class _Adapter(requests.adapters.HTTPAdapter):
            def __init__(self, _hostname: str, _ip: str) -> None:
                """Initialize the instance."""
                self._hostname = _hostname
                self._ip = _ip
                super().__init__()

            def get_connection_with_tls_context(
                self,
                request: Any,
                verify: Any,
                proxies: Any | None = None,
                cert: Any | None = None,
            ) -> Any:
                # requests >= 2.32 path. ``verify``/``cert`` are
                # routed to the pool by the framework; we only need
                # to swap the host -> IP and feed SNI through pool_kwargs.
                return self._pinned_pool(request.url)

            def get_connection(self, url: str | bytes, proxies: Any | None = None) -> Any:
                # requests < 2.32 fallback. ``url``'s type matches
                # ``requests.adapters.HTTPAdapter.get_connection``'s
                # signature (which accepts ``str | bytes`` because
                # urllib3's older API tolerated either); decode here
                # so the URL parser downstream sees a real ``str``.
                if isinstance(url, bytes):
                    url = url.decode("ascii")
                return self._pinned_pool(url)

            def _pinned_pool(self, url: str) -> Any:
                from urllib.parse import urlparse as _parse

                parsed = _parse(url)
                scheme = parsed.scheme
                port = parsed.port or (443 if scheme == "https" else 80)
                pool_kwargs: dict[str, Any] = {}
                if scheme == "https":
                    pool_kwargs["server_hostname"] = self._hostname
                    pool_kwargs["assert_hostname"] = self._hostname
                # ``connection_from_host`` returns a cached pool keyed
                # on (scheme, host, port); since we use the IP as the
                # host, distinct hostnames that share an IP get
                # distinct pools (correct for SNI).
                return self.poolmanager.connection_from_host(
                    host=self._ip,
                    port=port,
                    scheme=scheme,
                    pool_kwargs=pool_kwargs,
                )

        # Reference the imports to silence unused-import linters in
        # alternative urllib3 versions where the symbols might not be
        # reachable.
        _ = HTTPSConnectionPool, HTTPConnectionPool, PoolManager
        return _Adapter(hostname, validated_ip)


class SafeHTTPClient:
    """Centralised outbound HTTP gate.

    All Mercury Agent egress goes through this class.  See the
    module docstring for the gates that fire on every call.
    """

    @classmethod
    def validate_url(
        cls,
        url: str,
        *,
        allow_http: bool = False,
        user_configured: bool = False,
        loopback_only: bool = False,
        allow_private: bool = False,
    ) -> None:
        """Run every gate without actually sending a request.

        Useful at config-validation time (e.g. when an operator sets
        ``OLLAMA_HOST`` we can fail loudly at startup instead of on
        first inference call).

        Args:
            url: The URL to check.
            allow_http: Permit ``http://`` URLs.  Default False
                (HTTPS-only).
            user_configured: The URL came from operator config /
                env-var, so it must pass the private-network /
                IMDS block.  Class-constant dataset URLs pass
                ``user_configured=False``.
            loopback_only: The host must be on 127/8 or ``::1``.
                Use for on-box adapters (Ollama default, Redis
                sidecar).  Wins over ``allow_private``.
            allow_private: Permit resolved IPs on RFC1918 /
                IPv6-ULA / IPv4-private ranges.  Use for
                self-hosted services that live inside the operator's
                VPC (SearXNG, an internal Redis, an on-prem inference
                backend).  Even with this flag set, the IMDS
                (169.254/16), loopback, multicast, and reserved
                ranges still raise -- those are never legitimate
                production endpoints and the metadata service is the
                primary SSRF target.

        Raises:
            UnsafeURLError: any gate failed.
        """
        scheme, host = _parse_and_check_scheme(url, allow_http=allow_http)

        # Air-gap gate (fires first, before any allowlist or DNS work).
        # When MERCURY_OFFLINE is set, the only egress permitted is to a
        # loopback target -- an on-box model (Ollama) or a local sidecar
        # (Redis). Every other destination is refused HERE, before a
        # resolver is touched or a socket is opened, so the guarantee
        # holds in a true air-gap where no DNS is reachable. Imported
        # lazily (the codebase-wide pattern) so this security primitive
        # never pulls the heavy datasets package at module load.
        from omni_mercury_engine.datasets.exceptions import (
            OfflineModeError,
            offline_allow_private_active,
            offline_mode_active,
        )

        # VPC-air-gap mode: offline (public internet cut) but the caller has
        # explicitly opted into reaching an on-prem RFC1918 service AND the
        # operator has set MERCURY_OFFLINE_ALLOW_PRIVATE. In that mode a private
        # target survives the offline gate and is enforced private-only by the
        # allow_private branch below (which still refuses IMDS and, under
        # air-gap, any PUBLIC resolution). Everything else stays loopback-only.
        vpc_offline = offline_mode_active() and allow_private and offline_allow_private_active()
        if offline_mode_active() and not _is_loopback_host(host):
            if not vpc_offline:
                raise OfflineModeError(url)
            # A public/IMDS IP LITERAL is refused pre-DNS here (no resolver
            # needed); a private IP literal and named hosts fall through to the
            # private-only resolve below, which fails closed if DNS is down.
            literal = _host_ip_literal(host)
            if literal is not None and not (literal.is_private and not _is_always_blocked(literal)):
                raise OfflineModeError(url)

        # Trusted-allowlist gate -- runs for *both* schemes when the
        # URL is not user-configured.  Previously this only fired for
        # https://, which let an ``allow_http=True`` dataset mirror
        # reach an arbitrary host with no allowlist or private-network
        # check.  Now: http:// is treated identically -- the host must
        # be in TRUSTED_DOMAINS unless it is an explicit operator-
        # configured endpoint, and every http:// URL also goes through
        # the private-network / IMDS gate below regardless of
        # ``user_configured`` because plain HTTP is exactly the
        # transport an attacker would use to bounce through internal
        # infrastructure.
        if not user_configured:
            try:
                TrustedEndpoints.validate_url_host(host)
            except ValueError as exc:
                raise UnsafeURLError(str(exc)) from exc

        # Private-network / IMDS gate.  Runs whenever:
        #   * the URL came from operator config (user_configured=True),
        #   * the caller asked for loopback-only enforcement,
        #   * the URL is plain HTTP (allow_http=True), because the
        #     trusted-allowlist gate is the only thing standing
        #     between an http:// URL and an SSRF pivot, and we want
        #     defence-in-depth even when the host is allowlisted.
        # ``allow_private`` permits RFC1918 for explicitly configured
        # internal services; it never unlocks IMDS / loopback /
        # multicast / reserved ranges.
        # Class-constant https:// dataset URLs still skip the resolve
        # because they're public hostnames we explicitly allowlisted;
        # resolving them on every call is pure overhead.
        needs_ip_gate = user_configured or loopback_only or scheme == "http"
        if needs_ip_gate:
            ips = _resolve_ips(host)
            if loopback_only:
                non_lo = [str(ip) for ip in ips if not _is_loopback(ip)]
                if non_lo:
                    raise UnsafeURLError(
                        f"SafeHTTPClient: host '{host}' resolves to "
                        f"non-loopback IPs {non_lo}; loopback_only=True "
                        f"refuses any address outside 127/8 or ::1."
                    )
            elif allow_private:
                # The caller has acknowledged that the target is on
                # their private network. RFC1918 is now permitted, but
                # IMDS / loopback / multicast / reserved still raise.
                bad = [str(ip) for ip in ips if _is_always_blocked(ip)]
                if bad:
                    raise UnsafeURLError(
                        f"SafeHTTPClient: host '{host}' resolves to "
                        f"always-blocked address(es) {bad} (IMDS / loopback / "
                        f"multicast / reserved). allow_private=True does NOT "
                        f"unlock these."
                    )
                if vpc_offline:
                    # VPC-air-gap: cut from the PUBLIC internet, so an
                    # allow_private target must resolve onto the private
                    # network. A public resolution is refused as egress —
                    # the air-gap holds; only on-prem RFC1918 is reachable.
                    public = [str(ip) for ip in ips if not ip.is_private and not _is_loopback(ip)]
                    if public:
                        raise OfflineModeError(url)
            else:
                bad = [str(ip) for ip in ips if _is_private_or_imds(ip)]
                if bad:
                    raise UnsafeURLError(
                        f"SafeHTTPClient: host '{host}' resolves to "
                        f"private/link-local/IMDS address(es) {bad}; "
                        f"refusing SSRF pivot."
                    )

    # ----------------------------------------------------------------
    # Request helpers.  Every callsite that previously did its own
    # urllib.request.Request + urlopen now uses one of these.
    # ----------------------------------------------------------------

    @classmethod
    def _request(
        cls,
        method: str,
        url: str,
        *,
        params: dict[str, str] | None = None,
        json_body: Any = None,
        data: bytes | str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        allow_http: bool = False,
        user_configured: bool = False,
        loopback_only: bool = False,
        allow_private: bool = False,
        stream: bool = False,
        deadline: float | None = None,
    ) -> requests.Response:
        """Shared body of ``get``/``post``/``request``."""
        cls.validate_url(
            url,
            allow_http=allow_http,
            user_configured=user_configured,
            loopback_only=loopback_only,
            allow_private=allow_private,
        )

        # Close the DNS-rebinding / TOCTOU window between
        # ``validate_url`` and the actual network call.  Even when the
        # URL is in TRUSTED_DOMAINS, an attacker who can poison DNS
        # for that hostname (compromised upstream resolver, cache
        # poisoning, a hostile recursive on a misconfigured network)
        # could otherwise rotate the answer between our validation
        # resolve and ``requests``'s second resolve. We resolve once
        # here, re-apply the IP policy to the result (defence in
        # depth -- catches a hostile answer that slipped past
        # ``validate_url``'s conditional resolve), and pin the TCP
        # connection to the validated IP via a custom adapter. TLS
        # SNI and certificate verification still use the operator-
        # meaningful hostname so cert pinning continues to work.
        parsed = urlparse(url)
        host = parsed.hostname or ""
        resolved_ips = _resolve_ips(host)

        # Re-check policy at request time.  ``validate_url`` only
        # resolves when its ``needs_ip_gate`` predicate fires; here we
        # ALWAYS resolve so a hostname whose DNS now returns a private
        # / IMDS address cannot slip through.  The class of blocks we
        # apply mirrors ``validate_url`` exactly so the DNS-rebinding
        # fix does not leave a weaker policy than the pre-flight check:
        #
        #   * ``loopback_only``         -> only 127/8 or ::1 acceptable
        #   * ``allow_private=True``    -> RFC1918 OK, always-blocked
        #                                  set (IMDS / loopback /
        #                                  multicast / reserved / CGNAT)
        #                                  still refused
        #   * default                   -> private/IMDS/CGNAT/loopback
        #                                  all refused
        #
        # Trusted-allowlist class-constant URLs use the default branch
        # (loopback_only=False, allow_private=False); a public host that
        # rebinds to RFC1918 between validation and request is caught
        # by the ``_is_private_or_imds`` filter, not just by the
        # always-blocked subset.
        if loopback_only:
            non_lo = [str(ip) for ip in resolved_ips if not _is_loopback(ip)]
            if non_lo:
                raise UnsafeURLError(
                    f"SafeHTTPClient: host '{host}' resolved to non-loopback "
                    f"IPs {non_lo} at request time; loopback_only=True refuses."
                )
        elif allow_private:
            bad = [str(ip) for ip in resolved_ips if _is_always_blocked(ip)]
            if bad:
                raise UnsafeURLError(
                    f"SafeHTTPClient: host '{host}' resolved to "
                    f"always-blocked address(es) {bad} at request time. This "
                    "is a DNS-rebinding signature (different answer between "
                    "validation and request); refusing connection. "
                    "allow_private=True does NOT open the IMDS / loopback / "
                    "multicast / reserved / CGNAT ranges."
                )
            # VPC-air-gap re-check: under MERCURY_OFFLINE + allow_private the
            # target must stay on the private network. A private->public
            # rebind between validation and request is refused as egress, so
            # the air-gap holds through the TOCTOU window too.
            from omni_mercury_engine.datasets.exceptions import (
                OfflineModeError,
                offline_allow_private_active,
                offline_mode_active,
            )

            if offline_mode_active() and offline_allow_private_active():
                public = [
                    str(ip) for ip in resolved_ips if not ip.is_private and not _is_loopback(ip)
                ]
                if public:
                    raise OfflineModeError(url)
        else:
            bad = [str(ip) for ip in resolved_ips if _is_private_or_imds(ip)]
            if bad:
                raise UnsafeURLError(
                    f"SafeHTTPClient: host '{host}' resolved to "
                    f"private/link-local/IMDS/CGNAT address(es) {bad} at "
                    "request time. This is a DNS-rebinding signature "
                    "(different answer between validation and request); "
                    "refusing SSRF pivot."
                )

        request_headers: dict[str, str] = {"User-Agent": _DEFAULT_USER_AGENT}
        if headers:
            request_headers.update(headers)
        # Force the HTTP ``Host`` header to the original hostname+port.
        # urllib3 derives ``Host`` from the connection pool's host
        # field; since the pinned adapter sets that to the IP, virtual-
        # hosted upstreams would receive ``Host: <ip>`` and either
        # serve the wrong vhost or return 400.  Setting the header
        # explicitly (urllib3 honours caller-supplied ``Host`` and
        # skips synthesising one) keeps HTTP-level routing correct
        # even though TCP is pinned by IP.  The port must be preserved
        # for non-default ports (e.g. ``http://localhost:11434`` for
        # Ollama, or any reverse-proxied service on a non-80/443 port);
        # dropping the port produces ``Host: localhost`` which routes
        # to the wrong upstream on a multi-tenant proxy.  The port is
        # omitted only when it matches the scheme default (80/http,
        # 443/https) so virtual-hosted endpoints that key off the bare
        # hostname continue to work.
        explicit_port = parsed.port
        scheme_default_port = 443 if parsed.scheme == "https" else 80
        if explicit_port is None or explicit_port == scheme_default_port:
            host_header_value = host
        else:
            host_header_value = f"{host}:{explicit_port}"
        request_headers.setdefault("Host", host_header_value)

        # Deferred import: ``requests`` is a core dependency for any
        # caller that actually issues a network request, but it is
        # *not* a dependency of the security package's import surface.
        # The migrate_pkl hardened subprocess scrubs PYTHONNOUSERSITE
        # and only forwards a fixed env allow-list; if ``requests`` is
        # installed in user-site (dev/CI), the child process loses
        # access to it, and an eager module-level import here would
        # break ``from omni_mercury_engine.security.safe_load import
        # sign_npz`` for the signing path -- which has nothing to do
        # with HTTP. Importing here keeps ``safe_load`` resilient.
        import requests

        # Try each validated IP in order.  Pinning to a single IP
        # closes the DNS-rebinding window; iterating the list
        # preserves the multi-IP resilience the stdlib socket layer
        # would have given us if we had not pinned. ``last_exc`` is
        # re-raised if every validated IP is unreachable so the
        # operator sees a real connection error, not a confusing
        # silent failure.
        last_exc: Exception | None = None
        response = None
        for candidate_ip in resolved_ips:
            if deadline is None:
                request_timeout = timeout
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"SafeHTTPClient: deadline exceeded for '{host}'")
                request_timeout = min(timeout, remaining)

            session = requests.Session()
            adapter = _PinnedDNSHTTPAdapter.build(host, str(candidate_ip))
            # Mount for both schemes; the adapter inspects the URL to
            # decide which urllib3 connection pool flavour to use.
            session.mount("https://", adapter)
            session.mount("http://", adapter)

            try:
                response = session.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    data=data,
                    headers=request_headers,
                    timeout=request_timeout,
                    stream=stream,
                    allow_redirects=False,
                )
                break
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                # Redacted: the transport error's text embeds the request
                # path + query, which carries the credential for keyed URLs.
                logger.debug(
                    "SafeHTTPClient: pinned-IP %s failed (%s); trying next.",
                    candidate_ip,
                    _redact_full(redact_text(str(exc))),
                )
                session.close()
                continue
        if response is None:
            if last_exc is None:
                # The loop runs at least once because _resolve_ips
                # raises rather than returning an empty list, so every
                # iteration either assigns response or last_exc. This
                # branch is therefore unreachable; surface it as an
                # explicit RuntimeError so a future regression fails
                # loudly instead of falling through with response=None.
                raise RuntimeError(
                    f"SafeHTTPClient: exhausted all validated IPs for '{host}' "
                    "without recording an exception (loop invariant violated)."
                )
            # The raw transport error embeds the request path + query in
            # its message (urllib3: "Max retries exceeded with url:
            # /query?apikey=..."), so for a keyed caller re-raising the
            # original object leaks the credential into exception text.
            # Re-raise the same TYPE (existing ``except requests.
            # ConnectionError`` / ``Timeout`` handlers keep working) with
            # redacted text and a severed chain — the original message IS
            # the leak, so it must not ride along as ``__cause__``.
            raise type(last_exc)(_redact_full(redact_text(str(last_exc)))) from None
        # Reject 3xx explicitly.  ``allow_redirects=False`` blocks
        # ``requests`` from following the redirect, but
        # ``raise_for_status()`` only fires on 4xx/5xx, so without this
        # branch a 301/302/303/307/308 returns as "success" and the
        # callers below consume the redirect body (often an HTML stub)
        # instead of the resource.  Real-world bite: GitHub
        # ``.../raw/...`` URLs redirect to ``raw.githubusercontent.com``
        # and silently corrupt downloads.  We refuse rather than chase
        # because every URL the loaders hit is an explicit final
        # destination in ``TRUSTED_DOMAINS``; a 3xx from one of them
        # means the source URL has drifted and the right fix is to
        # update the URL in the loader (and add the redirect target to
        # the allowlist if necessary), not to re-validate and follow.
        if 300 <= response.status_code < 400:
            location = response.headers.get("Location", "<no Location header>")
            # Both URLs are credential-redacted: the request URL because a
            # keyed loader composes its credential into it, and the Location
            # header because upstreams answer with signed/tokenised redirect
            # targets — either one in exception text is a disclosure.
            raise UnsafeURLError(
                f"SafeHTTPClient: refused {response.status_code} redirect "
                f"from '{_redact_full(redact_url(url))}' to '{_redact_full(redact_url(location))}'. "
                f"SafeHTTPClient does not follow redirects; update the "
                f"source URL to the final destination (and confirm it is "
                f"in TRUSTED_DOMAINS)."
            )
        # requests renders HTTPError as "<status> ... Error: <reason> for
        # url: <fully-composed URL>" — for a keyed caller (Alpha Vantage
        # ``apikey``, OpenWeatherMap ``appid``, NASA ``api_key`` in query
        # params) that URL carries the live credential. Re-raise the same
        # exception type with the URL credential-redacted and the chain
        # severed (the original message is the leak); ``response`` is
        # preserved so status-code branching keeps working.
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise requests.HTTPError(
                _redact_full(redact_text(str(exc))), response=response
            ) from None
        return response

    @classmethod
    def get(
        cls,
        url: str,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        allow_http: bool = False,
        user_configured: bool = False,
        loopback_only: bool = False,
        allow_private: bool = False,
        stream: bool = False,
        deadline: float | None = None,
    ) -> requests.Response:
        """Issue a validated GET; the returned response is closed by the caller."""
        return cls._request(
            "GET",
            url,
            params=params,
            headers=headers,
            timeout=timeout,
            allow_http=allow_http,
            user_configured=user_configured,
            loopback_only=loopback_only,
            allow_private=allow_private,
            stream=stream,
            deadline=deadline,
        )

    @classmethod
    def get_bytes(
        cls,
        url: str,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        allow_http: bool = False,
        user_configured: bool = False,
        loopback_only: bool = False,
        allow_private: bool = False,
        deadline: float | None = None,
    ) -> bytes:
        """GET and return the response body as bytes."""
        with cls.get(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
            allow_http=allow_http,
            user_configured=user_configured,
            loopback_only=loopback_only,
            allow_private=allow_private,
            deadline=deadline,
        ) as response:
            body: bytes = response.content
            return body

    @classmethod
    def get_json(
        cls,
        url: str,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        allow_http: bool = False,
        user_configured: bool = False,
        loopback_only: bool = False,
        allow_private: bool = False,
    ) -> Any:
        """GET and decode the response body as JSON."""
        with cls.get(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
            allow_http=allow_http,
            user_configured=user_configured,
            loopback_only=loopback_only,
            allow_private=allow_private,
        ) as response:
            return response.json()

    @classmethod
    def get_text(
        cls,
        url: str,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        allow_http: bool = False,
        user_configured: bool = False,
        loopback_only: bool = False,
        allow_private: bool = False,
    ) -> str:
        """GET and return the decoded response text."""
        with cls.get(
            url,
            params=params,
            headers=headers,
            timeout=timeout,
            allow_http=allow_http,
            user_configured=user_configured,
            loopback_only=loopback_only,
            allow_private=allow_private,
        ) as response:
            text: str = response.text
            return text

    @classmethod
    def post_json(
        cls,
        url: str,
        *,
        json_body: Any,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        allow_http: bool = False,
        user_configured: bool = False,
        loopback_only: bool = False,
        allow_private: bool = False,
    ) -> Any:
        """POST a JSON body and decode the response as JSON."""
        with cls._request(
            "POST",
            url,
            json_body=json_body,
            headers=headers,
            timeout=timeout,
            allow_http=allow_http,
            user_configured=user_configured,
            loopback_only=loopback_only,
            allow_private=allow_private,
        ) as response:
            return response.json()

    @classmethod
    def post_form(
        cls,
        url: str,
        *,
        form_data: dict[str, str],
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        allow_http: bool = False,
        user_configured: bool = False,
        loopback_only: bool = False,
        allow_private: bool = False,
    ) -> Any:
        """POST an ``application/x-www-form-urlencoded`` body, decode JSON.

        OAuth 2.0 token endpoints (RFC 6749 §4.1.3) require form-encoded
        request bodies.  This helper centralises the ``urlencode`` +
        ``Content-Type`` plumbing so every SafeHTTPClient gate (scheme
        allowlist, private-network block, DNS-rebinding pin, redirect
        refusal) runs in front of the OAuth call.  The response is
        decoded as JSON because every OAuth token endpoint replies in
        JSON; a non-JSON response surfaces as :class:`ValueError` from
        the underlying ``requests.Response.json``.
        """
        encoded = urlencode(form_data).encode("utf-8")
        request_headers: dict[str, str] = {
            "Content-Type": "application/x-www-form-urlencoded",
        }
        if headers:
            request_headers.update(headers)
        with cls._request(
            "POST",
            url,
            data=encoded,
            headers=request_headers,
            timeout=timeout,
            allow_http=allow_http,
            user_configured=user_configured,
            loopback_only=loopback_only,
            allow_private=allow_private,
        ) as response:
            return response.json()


__all__ = [
    "SafeHTTPClient",
    "UnsafeURLError",
    "enforce_offline_egress",
]
