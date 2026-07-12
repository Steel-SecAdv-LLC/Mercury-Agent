# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for the USGS EROS M2M client (mocked transport; no network).

These exercise the ``login-token`` auth flow, the ``X-Auth-Token`` header
plumbing, the two-secret requirement, API error-code surfacing, and that no
secret material leaks into error messages -- all against a fake
``SafeHTTPClient.post_json`` so nothing touches the network.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from omni_mercury_engine.integrations.usgs_eros import (
    TOKEN_ENV_VAR,
    USERNAME_ENV_VAR,
    USGSErosError,
    USGSErosM2MClient,
)

if TYPE_CHECKING:
    from collections.abc import Callable

#: A fake application token used throughout; a variable (not a literal at the
#: call site) so flake8-bandit does not flag every ``token=`` argument.
_FAKE_TOKEN = "fake-app-token"


class _FakeM2M:
    """Fake M2M transport: records calls, replies with canned envelopes."""

    def __init__(self, envelopes: dict[str, Any]) -> None:
        self.envelopes = envelopes
        self.calls: list[tuple[str, Any, dict[str, str]]] = []

    def post_json(
        self, url: str, *, json_body: Any, headers: dict[str, str] | None = None, **_: Any
    ) -> Any:
        endpoint = url.rsplit("/", 1)[-1]
        self.calls.append((endpoint, json_body, dict(headers or {})))
        if endpoint not in self.envelopes:
            raise AssertionError(f"unexpected M2M endpoint: {endpoint}")
        return self.envelopes[endpoint]


@pytest.fixture
def install_fake(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[dict[str, Any]], _FakeM2M]:
    """Install a fake transport over SafeHTTPClient.post_json; return it."""

    def _install(envelopes: dict[str, Any]) -> _FakeM2M:
        fake = _FakeM2M(envelopes)
        monkeypatch.setattr(
            "omni_mercury_engine.security.safe_http.SafeHTTPClient.post_json",
            fake.post_json,
        )
        return fake

    return _install


def test_available_requires_both_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
    monkeypatch.delenv(USERNAME_ENV_VAR, raising=False)
    assert USGSErosM2MClient.available() is False
    monkeypatch.setenv(TOKEN_ENV_VAR, _FAKE_TOKEN)
    assert USGSErosM2MClient.available() is False  # username still missing
    monkeypatch.setenv(USERNAME_ENV_VAR, "user")
    assert USGSErosM2MClient.available() is True


def test_login_sends_credentials_and_caches_key(install_fake: Any) -> None:
    fake = install_fake({"login-token": {"data": "APIKEY123", "errorCode": None}})
    client = USGSErosM2MClient(username="ers_user", token=_FAKE_TOKEN)
    key = client.login()
    assert key == "APIKEY123"
    assert client.authenticated is True
    endpoint, payload, headers = fake.calls[0]
    assert endpoint == "login-token"
    assert payload == {"username": "ers_user", "token": _FAKE_TOKEN}
    assert "X-Auth-Token" not in headers  # the login call itself is unauthenticated


def test_dataset_search_logs_in_and_sends_auth_header(install_fake: Any) -> None:
    fake = install_fake(
        {
            "login-token": {"data": "APIKEY123", "errorCode": None},
            "dataset-search": {"data": [{"datasetAlias": "landsat_ot_c2_l2"}], "errorCode": None},
        }
    )
    client = USGSErosM2MClient(username="ers_user", token=_FAKE_TOKEN)
    datasets = client.dataset_search("landsat")
    assert datasets == [{"datasetAlias": "landsat_ot_c2_l2"}]
    # second call is the authenticated dataset-search
    endpoint, payload, headers = fake.calls[1]
    assert endpoint == "dataset-search"
    assert payload == {"datasetName": "landsat"}
    assert headers["X-Auth-Token"] == "APIKEY123"


def test_login_without_both_credentials_raises_without_leaking(install_fake: Any) -> None:
    install_fake({})  # no call should be made
    client = USGSErosM2MClient(username="", token=_FAKE_TOKEN)
    with pytest.raises(USGSErosError) as exc:
        client.login()
    msg = str(exc.value)
    assert USERNAME_ENV_VAR in msg and TOKEN_ENV_VAR in msg
    assert _FAKE_TOKEN not in msg  # never leak the token value


def test_api_error_code_surfaces_without_leaking_token(install_fake: Any) -> None:
    install_fake(
        {"login-token": {"data": None, "errorCode": "AUTH_INVALID", "errorMessage": "bad token"}}
    )
    client = USGSErosM2MClient(username="ers_user", token=_FAKE_TOKEN)
    with pytest.raises(USGSErosError) as exc:
        client.login()
    msg = str(exc.value)
    assert "AUTH_INVALID" in msg
    assert _FAKE_TOKEN not in msg
    assert client.authenticated is False


def test_dataset_search_non_list_data_returns_empty(install_fake: Any) -> None:
    install_fake(
        {
            "login-token": {"data": "APIKEY123", "errorCode": None},
            "dataset-search": {"data": None, "errorCode": None},
        }
    )
    client = USGSErosM2MClient(username="u", token=_FAKE_TOKEN)
    assert client.dataset_search() == []


def test_logout_clears_key_and_swallows_api_error(install_fake: Any) -> None:
    install_fake(
        {
            "login-token": {"data": "APIKEY123", "errorCode": None},
            "logout": {"data": None, "errorCode": "SERVER_ERROR", "errorMessage": "x"},
        }
    )
    client = USGSErosM2MClient(username="u", token=_FAKE_TOKEN)
    client.login()
    assert client.authenticated is True
    client.logout()  # must not raise despite the logout errorCode
    assert client.authenticated is False
