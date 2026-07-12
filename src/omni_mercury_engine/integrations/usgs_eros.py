# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""USGS EROS Machine-to-Machine (M2M) API client.

A thin, keyed client for the USGS/EROS Inventory Service M2M API
(``m2m.cr.usgs.gov``) — the machine interface behind EarthExplorer for
searching Landsat and other USGS remote-sensing scene inventories.

**Authentication.** The legacy ``login`` (username + password) endpoint was
deprecated on 2026-02-26; this client uses ``login-token``, which requires
**both**:

* an **ERS username** — read from ``EROSERS_USERNAME`` (or passed explicitly), and
* a **64-character application token** — read from ``USGS_KEY`` (or passed
  explicitly), generated at https://ers.cr.usgs.gov/password/appgenerate and
  used in place of the ERS password.

``login-token`` returns a short-lived **API key** which is sent as the
``X-Auth-Token`` header on every subsequent request. Both credentials come from
the environment (wired as GitHub Actions repository secrets ``USGS_KEY`` and
``EROSERS_USERNAME``); neither the token nor the API key is ever logged.

The client fetches only through :class:`SafeHTTPClient`, so the host must be on
the ``TrustedEndpoints.TRUSTED_DOMAINS`` allowlist (``m2m.cr.usgs.gov``) and the
scheme/redirect/private-network gates apply.

Usage::

    client = USGSErosM2MClient()          # reads EROSERS_USERNAME + USGS_KEY
    if client.available():
        datasets = client.dataset_search("landsat")   # logs in on demand
        client.logout()
"""

from __future__ import annotations

import os
from typing import Any

from omni_mercury_engine.security.safe_http import SafeHTTPClient

#: M2M stable JSON API base (no trailing slash).
M2M_BASE_URL = "https://m2m.cr.usgs.gov/api/api/json/stable"

#: Env vars the client reads. ``USGS_KEY`` is the application token; the M2M
#: ``login-token`` endpoint additionally requires the ERS username.
TOKEN_ENV_VAR = "USGS_KEY"  # noqa: S105 - env var name, not a secret
USERNAME_ENV_VAR = "EROSERS_USERNAME"


class USGSErosError(RuntimeError):
    """A USGS EROS M2M request failed (auth, transport, or API error code)."""


class USGSErosM2MClient:
    """Authenticated client for the USGS EROS M2M inventory API.

    Credentials default to the environment (``EROSERS_USERNAME`` + ``USGS_KEY``)
    but may be passed explicitly for testing. The API key obtained from
    ``login-token`` is held only in memory and never logged; :meth:`logout`
    invalidates it server-side.
    """

    def __init__(
        self,
        username: str | None = None,
        token: str | None = None,
        *,
        base_url: str = M2M_BASE_URL,
        timeout: float = 60.0,
    ) -> None:
        """Initialize the client (credentials default to the environment).

        Args:
            username: ERS username; defaults to ``$EROSERS_USERNAME``.
            token: 64-char application token; defaults to ``$USGS_KEY``.
            base_url: M2M stable JSON API base URL.
            timeout: Per-request timeout in seconds.
        """
        self._username = (
            username if username is not None else os.environ.get(USERNAME_ENV_VAR, "")
        ).strip()
        self._token = (token if token is not None else os.environ.get(TOKEN_ENV_VAR, "")).strip()
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._api_key: str | None = None

    @classmethod
    def available(cls) -> bool:
        """True when BOTH the ERS username and application token are configured.

        ``login-token`` needs both, so a run with only one is not usable; callers
        (e.g. the CI live-data smoke) should SKIP rather than attempt a login.
        """
        return bool(
            os.environ.get(USERNAME_ENV_VAR, "").strip()
            and os.environ.get(TOKEN_ENV_VAR, "").strip()
        )

    @property
    def authenticated(self) -> bool:
        """True once :meth:`login` has obtained an API key."""
        return self._api_key is not None

    def _post(self, endpoint: str, payload: dict[str, Any], *, authed: bool) -> Any:
        """POST one M2M request and return its ``data`` field (raises on error).

        The M2M API always replies HTTP 200 with a JSON envelope
        ``{"data": ..., "errorCode": ..., "errorMessage": ...}``; a non-null
        ``errorCode`` is the real failure signal and is surfaced as
        :class:`USGSErosError`. Key material is never included in the message.
        """
        headers = {"Content-Type": "application/json"}
        if authed:
            if not self._api_key:
                raise USGSErosError("not authenticated; call login() first")
            headers["X-Auth-Token"] = self._api_key
        try:
            envelope = SafeHTTPClient.post_json(
                f"{self._base_url}/{endpoint}",
                json_body=payload,
                headers=headers,
                timeout=self._timeout,
            )
        except Exception as exc:  # transport / TLS / allowlist / decode
            raise USGSErosError(
                f"M2M {endpoint} transport error: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(envelope, dict):
            raise USGSErosError(f"M2M {endpoint} returned a non-object response")
        error_code = envelope.get("errorCode")
        if error_code:
            raise USGSErosError(
                f"M2M {endpoint} failed: {error_code}: {envelope.get('errorMessage', '')}"
            )
        return envelope.get("data")

    def login(self) -> str:
        """Authenticate via ``login-token`` and cache the returned API key.

        Returns:
            The API key (also stored for subsequent requests).

        Raises:
            USGSErosError: If either credential is missing or the login is
                rejected.
        """
        if not self._username or not self._token:
            raise USGSErosError(
                "USGS EROS M2M needs BOTH an ERS username and an application token "
                f"(set {USERNAME_ENV_VAR} and {TOKEN_ENV_VAR}); the deprecated "
                "password login is not supported"
            )
        api_key = self._post(
            "login-token",
            {"username": self._username, "token": self._token},
            authed=False,
        )
        if not isinstance(api_key, str) or not api_key:
            raise USGSErosError("login-token succeeded but returned no API key")
        self._api_key = api_key
        return api_key

    def dataset_search(self, dataset_name: str = "landsat") -> list[dict[str, Any]]:
        """Search the inventory for datasets matching ``dataset_name``.

        Logs in on demand. This is the low-risk reachability call the live-data
        smoke uses to prove the credentials are effective.

        Args:
            dataset_name: Case-insensitive dataset-name filter (e.g. ``landsat``).

        Returns:
            The list of matching dataset records (possibly empty).
        """
        if not self._api_key:
            self.login()
        data = self._post("dataset-search", {"datasetName": dataset_name}, authed=True)
        return list(data) if isinstance(data, list) else []

    def logout(self) -> None:
        """Invalidate the server-side API key (best effort) and clear it."""
        if not self._api_key:
            return
        try:
            self._post("logout", {}, authed=True)
        except USGSErosError:
            pass
        finally:
            self._api_key = None
