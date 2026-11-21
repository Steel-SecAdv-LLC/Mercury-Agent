# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""An opt-in transport fallback for the calibration scripts, and only for them.

The loaders fetch through ``SafeHTTPClient``, which connects to the address it
validated rather than through ``HTTPS_PROXY``. On a host whose only way out is
that proxy, the pinned connection is answered by the egress itself with
``503 upstream connect error`` and the loader reports the source unreachable,
while the same URL through the proxy answers. The table measured on 2026-11-20
was built on such a host: USGS water services (the Flood row) never answered
on the pinned route, and NASA SPDF (the OMNI2 series behind three Energy
events) stopped answering part-way through.

With ``MERCURY_CALIBRATION_PROXY_FALLBACK_HOSTS`` set to a comma-separated list
of host names, and for those hosts only, a request that the pinned route
answers with exactly that egress error is repeated once through the session's
proxy: the same URL, parameters and headers, after ``SafeHTTPClient``'s own
URL validation (HTTPS, the trusted-domain list), with redirects still refused.
Only the route changes; the bytes are the source's.

This is not in the loaders and must not be: it gives up the pinned address,
which is the client's protection against DNS rebinding. It exists so that a
measurement made on such a host can be repeated, and says so here. Unset, it
does nothing.
"""

from __future__ import annotations

import os
import sys
from typing import Any
from urllib.parse import urlparse

ENV_VAR = "MERCURY_CALIBRATION_PROXY_FALLBACK_HOSTS"
_EGRESS_ERROR = "upstream connect error"


def install_from_env() -> frozenset[str]:
    """Install the fallback for the hosts named in :data:`ENV_VAR`.

    Returns:
        The host names it was installed for; empty when the variable is unset.
    """
    hosts = frozenset(h.strip() for h in os.environ.get(ENV_VAR, "").split(",") if h.strip())
    if not hosts:
        return hosts

    import requests

    from omni_mercury_engine.security.safe_http import SafeHTTPClient

    pinned = SafeHTTPClient.get_bytes.__func__  # type: ignore[attr-defined]

    def get_bytes(
        cls: Any,
        url: str,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ) -> bytes:
        try:
            body: bytes = pinned(
                cls, url, params=params, headers=headers, timeout=timeout, **kwargs
            )
            return body
        except requests.HTTPError as exc:
            response = getattr(exc, "response", None)
            host = urlparse(url).hostname
            if (
                host not in hosts
                or response is None
                or response.status_code != 503
                or _EGRESS_ERROR not in (response.text or "")
            ):
                raise
            cls.validate_url(url)
            answer = requests.get(
                url, params=params, headers=headers, timeout=timeout, allow_redirects=False
            )
            if 300 <= answer.status_code < 400:
                raise
            answer.raise_for_status()
            print(
                f"  [proxy fallback] {host}: {len(answer.content)} bytes through the proxy",
                file=sys.stderr,
                flush=True,
            )
            return bytes(answer.content)

    SafeHTTPClient.get_bytes = classmethod(get_bytes)  # type: ignore[method-assign,assignment]
    print(f"proxy fallback installed for: {', '.join(sorted(hosts))}", file=sys.stderr, flush=True)
    return hosts
