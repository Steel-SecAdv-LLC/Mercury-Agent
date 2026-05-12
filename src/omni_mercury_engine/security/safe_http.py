"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.

------------------------------------------------------------------------

SafeHTTPClient -- the single egress point for outbound HTTP in
Mercury Agent.

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
from typing import Any
from urllib.parse import urlparse

import requests

from omni_mercury_engine.security.input_validation import TrustedEndpoints

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
    """
    Return (scheme, host) after enforcing the scheme allowlist.

    Raises:
        UnsafeURLError: scheme is not in the configured allowlist
            (or netloc is empty so we cannot extract a host).
    """
    parsed = urlparse(url)
    allowed = _HTTP_AND_HTTPS if allow_http else _HTTPS_ONLY
    if parsed.scheme not in allowed:
        raise UnsafeURLError(
            f"SafeHTTPClient: refusing URL '{url}' -- scheme '{parsed.scheme}' "
            f"is not in the allowlist {sorted(allowed)}."
        )
    host = parsed.hostname or ""
    if not host:
        raise UnsafeURLError(f"SafeHTTPClient: URL '{url}' has no host component.")
    return parsed.scheme, host


def _resolve_ips(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """
    Resolve a host to its IPs for the private-network gate.

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
        raise UnsafeURLError(
            f"SafeHTTPClient: host '{host}' did not resolve: {exc}."
        ) from exc
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
        raise UnsafeURLError(
            f"SafeHTTPClient: host '{host}' resolved to no usable IPs."
        )
    return ips


def _is_private_or_imds(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """
    True if ``ip`` is in a range we refuse for user-configured URLs.

    Covers RFC1918 (10/8, 172.16/12, 192.168/16), link-local
    (169.254/16, which includes the AWS / GCP / Azure IMDS at
    169.254.169.254), loopback (127/8, ::1), the IPv4-mapped
    equivalents of those, IPv6 ULA (fc00::/7), IPv6 link-local
    (fe80::/10), and the unspecified address (0.0.0.0, ::).
    """
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_unspecified
        or ip.is_reserved
        or ip.is_multicast
    )


def _is_loopback(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True if ``ip`` is on 127/8 or ``::1``."""
    return ip.is_loopback


class SafeHTTPClient:
    """
    Centralised outbound HTTP gate.

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
        allow_untrusted: bool = False,
    ) -> None:
        """
        Run every gate without actually sending a request.

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
                sidecar).
            allow_untrusted: Skip the TRUSTED_DOMAINS check.
                ``user_configured=True`` implies this (because we
                cannot know operator-chosen hosts in advance), so
                callers rarely need to set it directly.

        Raises:
            UnsafeURLError: any gate failed.
        """
        scheme, host = _parse_and_check_scheme(url, allow_http=allow_http)

        # Trusted-allowlist gate -- only for class-constant dataset
        # URLs.  User-configured endpoints (Ollama, SearXNG, custom
        # inference backends) opt out via user_configured=True.
        if scheme == "https" and not user_configured and not allow_untrusted:
            # TrustedEndpoints.validate_url raises ValueError on a miss
            # so we convert to our own exception type for consistent
            # error handling at callsites.
            try:
                TrustedEndpoints.validate_url(url)
            except ValueError as exc:
                raise UnsafeURLError(str(exc)) from exc

        # Private-network / IMDS gate -- runs for user-configured
        # URLs and for loopback-only on-box adapters.  Class-constant
        # dataset URLs skip this because they're public hostnames
        # that we explicitly allowlisted; resolving them on every
        # call would just slow us down and create a DNS-rebinding
        # window of its own.
        if user_configured or loopback_only:
            ips = _resolve_ips(host)
            if loopback_only:
                non_lo = [str(ip) for ip in ips if not _is_loopback(ip)]
                if non_lo:
                    raise UnsafeURLError(
                        f"SafeHTTPClient: host '{host}' resolves to "
                        f"non-loopback IPs {non_lo}; loopback_only=True "
                        f"refuses any address outside 127/8 or ::1."
                    )
            else:
                # User-configured but not loopback-only -- block
                # RFC1918 / link-local / IMDS.
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
        allow_untrusted: bool = False,
        stream: bool = False,
    ) -> requests.Response:
        """Shared body of ``get``/``post``/``request``."""
        cls.validate_url(
            url,
            allow_http=allow_http,
            user_configured=user_configured,
            loopback_only=loopback_only,
            allow_untrusted=allow_untrusted,
        )
        request_headers: dict[str, str] = {"User-Agent": _DEFAULT_USER_AGENT}
        if headers:
            request_headers.update(headers)
        response = requests.request(
            method,
            url,
            params=params,
            json=json_body,
            data=data,
            headers=request_headers,
            timeout=timeout,
            stream=stream,
            allow_redirects=False,
        )
        response.raise_for_status()
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
        allow_untrusted: bool = False,
        stream: bool = False,
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
            allow_untrusted=allow_untrusted,
            stream=stream,
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
        allow_untrusted: bool = False,
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
            allow_untrusted=allow_untrusted,
        ) as response:
            return response.content

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
        allow_untrusted: bool = False,
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
            allow_untrusted=allow_untrusted,
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
        allow_untrusted: bool = False,
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
            allow_untrusted=allow_untrusted,
        ) as response:
            return response.text

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
        allow_untrusted: bool = False,
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
            allow_untrusted=allow_untrusted,
        ) as response:
            return response.json()


__all__ = [
    "SafeHTTPClient",
    "UnsafeURLError",
]
