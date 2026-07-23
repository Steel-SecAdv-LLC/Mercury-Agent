# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for account-scoped API-key issuance and the email-change throttle.

Covers the self-service API-key surface (``POST``/``GET``/``DELETE
/api/v1/auth/api-keys``): one-time raw-key reveal, ownership isolation, the
per-account active-key cap, the permission whitelist, CSRF enforcement, and
that an issued key authenticates and is owned by the issuing account. Also
covers the per-action throttle newly wired onto the email-change request (an
outbound-email abuse vector) and the quota middleware resolving an API key's
owning-account tier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from omni_mercury_engine.api import auth
from omni_mercury_engine.api.auth import Permission
from omni_mercury_engine.api.auth_service import AuthService
from omni_mercury_engine.api.identity_store import Account, InMemoryIdentityStore
from omni_mercury_engine.api.quota_middleware import QuotaMiddleware
from omni_mercury_engine.api.routes import accounts

if TYPE_CHECKING:
    from collections.abc import Iterator


class RecordingMailer:
    """Collects sent messages so tests can extract verification tokens."""

    def __init__(self) -> None:
        """Start with an empty outbox."""
        self.sent: list[dict[str, str]] = []

    def send(
        self,
        *,
        to: str,
        subject: str,
        body: str,
        html_body: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Record the message."""
        self.sent.append({"to": to, "subject": subject, "body": body})

    def last_token(self) -> str:
        """Extract the token from the most recent email body."""
        import re

        match = re.search(r"token=([\w\-]+)", self.sent[-1]["body"])
        assert match is not None
        return match.group(1)


#: What the ``setup`` fixture yields (fully annotated for the strict test gate).
Setup = tuple[TestClient, RecordingMailer, AuthService]


@pytest.fixture
def setup(monkeypatch: pytest.MonkeyPatch) -> Iterator[Setup]:
    """TestClient + service over an in-memory store, fresh global key store."""
    monkeypatch.setenv("MERCURY_SESSION_COOKIE_SECURE", "false")
    # The API-key routes use the process-wide key store singleton; reset it so
    # keys never bleed between tests (MERCURY_KEYSTORE_PATH is unset → in-memory).
    auth._api_key_store = None
    mailer = RecordingMailer()
    service = AuthService(InMemoryIdentityStore(), mailer)
    app = FastAPI()
    app.include_router(accounts.router)
    app.dependency_overrides[accounts.get_auth_service] = lambda: service
    with TestClient(app) as client:
        yield client, mailer, service
    auth._api_key_store = None


def _register_verify_login(
    client: TestClient, mailer: RecordingMailer, email: str = "u@b.com"
) -> str:
    """Register, verify, and log in; return the session's CSRF token."""
    assert (
        client.post(
            "/api/v1/auth/register", json={"email": email, "password": "a-strong-pw"}
        ).status_code
        == 201
    )
    assert (
        client.post("/api/v1/auth/verify-email", json={"token": mailer.last_token()}).status_code
        == 200
    )
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": "a-strong-pw"})
    assert resp.status_code == 200
    return str(resp.json()["csrf_token"])


class TestApiKeyIssuance:
    """Creating, listing, and the one-time reveal contract."""

    def test_create_reveals_raw_key_once_and_it_authenticates(self, setup: Setup) -> None:
        """The raw key is returned once and validates against the key store."""
        client, mailer, _service = setup
        csrf = _register_verify_login(client, mailer)
        resp = client.post(
            "/api/v1/auth/api-keys",
            json={"name": "ci-token"},
            headers={"X-CSRF-Token": csrf},
        )
        assert resp.status_code == 201
        body = resp.json()
        raw = body["api_key"]
        assert raw and body["key"]["name"] == "ci-token"
        assert body["key"]["is_active"] is True
        # The raw key authenticates and is owned by the issuing account.
        account_id = client.get("/api/v1/auth/me").json()["id"]
        stored = auth.get_api_key_store().get_by_key(raw)
        assert stored is not None and stored.user_id == account_id

    def test_list_shows_metadata_never_the_secret(self, setup: Setup) -> None:
        """GET returns metadata only — never the raw key or its hash."""
        client, mailer, _service = setup
        csrf = _register_verify_login(client, mailer)
        client.post("/api/v1/auth/api-keys", json={"name": "k1"}, headers={"X-CSRF-Token": csrf})
        resp = client.get("/api/v1/auth/api-keys")
        assert resp.status_code == 200
        keys = resp.json()
        assert len(keys) == 1
        assert keys[0]["name"] == "k1"
        assert "api_key" not in keys[0] and "key_hash" not in keys[0]

    def test_default_permissions_are_read_and_detect(self, setup: Setup) -> None:
        """A create naming no permissions gets the safe default set."""
        client, mailer, _service = setup
        csrf = _register_verify_login(client, mailer)
        resp = client.post(
            "/api/v1/auth/api-keys", json={"name": "k"}, headers={"X-CSRF-Token": csrf}
        )
        assert sorted(resp.json()["key"]["permissions"]) == ["detect", "read"]

    def test_expiry_is_recorded(self, setup: Setup) -> None:
        """An expires_in_days request stamps the metadata."""
        client, mailer, _service = setup
        csrf = _register_verify_login(client, mailer)
        resp = client.post(
            "/api/v1/auth/api-keys",
            json={"name": "k", "expires_in_days": 30},
            headers={"X-CSRF-Token": csrf},
        )
        assert resp.json()["key"]["expires_at"] is not None


class TestApiKeyPermissionWhitelist:
    """A self-service key cannot be granted privileges its owner lacks."""

    @pytest.mark.parametrize("perm", ["read", "detect", "export"])
    def test_whitelisted_permissions_accepted(self, setup: Setup, perm: str) -> None:
        """Every whitelisted permission is accepted."""
        client, mailer, _service = setup
        csrf = _register_verify_login(client, mailer)
        resp = client.post(
            "/api/v1/auth/api-keys",
            json={"name": "k", "permissions": [perm]},
            headers={"X-CSRF-Token": csrf},
        )
        assert resp.status_code == 201
        assert resp.json()["key"]["permissions"] == [perm]

    @pytest.mark.parametrize("perm", ["admin", "write", "delete"])
    def test_privileged_permissions_rejected(self, setup: Setup, perm: str) -> None:
        """Privileged verbs are rejected 400 (no self-granted escalation)."""
        client, mailer, _service = setup
        csrf = _register_verify_login(client, mailer)
        resp = client.post(
            "/api/v1/auth/api-keys",
            json={"name": "k", "permissions": [perm]},
            headers={"X-CSRF-Token": csrf},
        )
        assert resp.status_code == 400

    def test_unknown_permission_rejected(self, setup: Setup) -> None:
        """An unknown permission name is a 400, not a silent drop."""
        client, mailer, _service = setup
        csrf = _register_verify_login(client, mailer)
        resp = client.post(
            "/api/v1/auth/api-keys",
            json={"name": "k", "permissions": ["superuser"]},
            headers={"X-CSRF-Token": csrf},
        )
        assert resp.status_code == 400


class TestApiKeyOwnership:
    """List and revoke are strictly scoped to the calling account."""

    def test_list_is_isolated_per_account(self, setup: Setup) -> None:
        """Each account sees only its own keys."""
        client, mailer, _service = setup
        csrf_a = _register_verify_login(client, mailer, email="a@b.com")
        client.post(
            "/api/v1/auth/api-keys", json={"name": "a-key"}, headers={"X-CSRF-Token": csrf_a}
        )
        client.post("/api/v1/auth/logout")

        csrf_b = _register_verify_login(client, mailer, email="b@b.com")
        client.post(
            "/api/v1/auth/api-keys", json={"name": "b-key"}, headers={"X-CSRF-Token": csrf_b}
        )
        names = [k["name"] for k in client.get("/api/v1/auth/api-keys").json()]
        assert names == ["b-key"]

    def test_revoke_own_key_deactivates_it(self, setup: Setup) -> None:
        """A revoked key stops authenticating and reads back inactive."""
        client, mailer, _service = setup
        csrf = _register_verify_login(client, mailer)
        created = client.post(
            "/api/v1/auth/api-keys", json={"name": "k"}, headers={"X-CSRF-Token": csrf}
        ).json()
        raw, key_id = created["api_key"], created["key"]["key_id"]

        resp = client.request(
            "DELETE", f"/api/v1/auth/api-keys/{key_id}", headers={"X-CSRF-Token": csrf}
        )
        assert resp.status_code == 200
        revoked = auth.get_api_key_store().get_by_key(raw)
        assert revoked is not None and revoked.is_active is False

    def test_revoke_other_accounts_key_is_404(self, setup: Setup) -> None:
        """Account B cannot revoke account A's key (existence not leaked)."""
        client, mailer, _service = setup
        csrf_a = _register_verify_login(client, mailer, email="a@b.com")
        key_id = client.post(
            "/api/v1/auth/api-keys", json={"name": "a-key"}, headers={"X-CSRF-Token": csrf_a}
        ).json()["key"]["key_id"]
        client.post("/api/v1/auth/logout")

        csrf_b = _register_verify_login(client, mailer, email="b@b.com")
        resp = client.request(
            "DELETE", f"/api/v1/auth/api-keys/{key_id}", headers={"X-CSRF-Token": csrf_b}
        )
        assert resp.status_code == 404
        # A's key is untouched.
        assert auth.get_api_key_store().get_by_id(key_id).is_active is True  # type: ignore[union-attr]

    def test_revoke_unknown_key_is_404(self, setup: Setup) -> None:
        """Revoking a nonexistent key id is a 404."""
        client, mailer, _service = setup
        csrf = _register_verify_login(client, mailer)
        resp = client.request(
            "DELETE", "/api/v1/auth/api-keys/deadbeef", headers={"X-CSRF-Token": csrf}
        )
        assert resp.status_code == 404


class TestApiKeyCap:
    """The per-account active-key cap bounds blast radius."""

    def test_cap_blocks_then_revoke_frees_a_slot(
        self, setup: Setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """At the cap a create is 409; revoking one frees a slot again."""
        monkeypatch.setenv("MERCURY_MAX_API_KEYS_PER_ACCOUNT", "2")
        client, mailer, _service = setup
        csrf = _register_verify_login(client, mailer)

        first = client.post(
            "/api/v1/auth/api-keys", json={"name": "k1"}, headers={"X-CSRF-Token": csrf}
        )
        client.post("/api/v1/auth/api-keys", json={"name": "k2"}, headers={"X-CSRF-Token": csrf})
        over = client.post(
            "/api/v1/auth/api-keys", json={"name": "k3"}, headers={"X-CSRF-Token": csrf}
        )
        assert first.status_code == 201 and over.status_code == 409

        # Revoke one → the cap admits a new key again (revoked keys don't count).
        key_id = first.json()["key"]["key_id"]
        client.delete(f"/api/v1/auth/api-keys/{key_id}", headers={"X-CSRF-Token": csrf})
        again = client.post(
            "/api/v1/auth/api-keys", json={"name": "k4"}, headers={"X-CSRF-Token": csrf}
        )
        assert again.status_code == 201


class TestApiKeyCsrfAndVerification:
    """CSRF is required on mutations; unverified accounts cannot issue keys."""

    def test_create_requires_csrf_header(self, setup: Setup) -> None:
        """A create without the CSRF header is rejected 403."""
        client, mailer, _service = setup
        _register_verify_login(client, mailer)
        assert client.post("/api/v1/auth/api-keys", json={"name": "k"}).status_code == 403

    def test_revoke_requires_csrf_header(self, setup: Setup) -> None:
        """A revoke without the CSRF header is rejected 403."""
        client, mailer, _service = setup
        csrf = _register_verify_login(client, mailer)
        key_id = client.post(
            "/api/v1/auth/api-keys", json={"name": "k"}, headers={"X-CSRF-Token": csrf}
        ).json()["key"]["key_id"]
        assert client.delete(f"/api/v1/auth/api-keys/{key_id}").status_code == 403

    def test_unverified_account_cannot_issue(self, setup: Setup) -> None:
        """The verified-only guard holds even if a session outlives verification.

        Login itself requires a verified account, so this exercises the guard
        via the one path that can produce an unverified session: an account
        that loses its verified flag after login (defense-in-depth).
        """
        client, mailer, service = setup
        csrf = _register_verify_login(client, mailer)
        account_id = client.get("/api/v1/auth/me").json()["id"]
        account = service.get_account(account_id)
        assert account is not None
        account.is_verified = False
        service._store.update_account(account)
        resp = client.post(
            "/api/v1/auth/api-keys", json={"name": "k"}, headers={"X-CSRF-Token": csrf}
        )
        assert resp.status_code == 403


class _NullMailer:
    """A mailer that drops everything (for service wiring in unit tests)."""

    def send(self, **kwargs: object) -> None:
        """Ignore the message."""


async def _dummy_asgi(scope: object, receive: object, send: object) -> None:  # pragma: no cover
    """A no-op ASGI app to satisfy the middleware constructor."""


def _request_with_api_key(raw_key: str) -> Request:
    """A real Starlette request carrying only an ``X-API-Key`` header.

    Built from an ASGI scope (no cookies, no client) so ``_resolve_principal``
    exercises exactly the API-key branch.
    """
    return Request({"type": "http", "headers": [(b"x-api-key", raw_key.encode())]})


class TestApiKeyQuotaTier:
    """The quota middleware charges an API key at its owning account's tier."""

    def test_key_principal_inherits_owner_tier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A key on a non-free account resolves to (account.id, that tier)."""
        store = InMemoryIdentityStore()
        account = Account(
            id="acct-1",
            email="p@b.com",
            password_hash="h",  # noqa: S106
            is_verified=True,
            tier="supporter",
        )
        store.create_account(account)
        service = AuthService(store, _NullMailer())
        monkeypatch.setattr(accounts, "_service", service)

        key_store = auth.APIKeyStore()
        raw, _key = key_store.create_key("k", "acct-1")
        monkeypatch.setattr(auth, "_api_key_store", key_store)

        middleware = QuotaMiddleware(app=_dummy_asgi)
        principal, tier = middleware._resolve_principal(_request_with_api_key(raw))
        assert principal == "acct-1"
        assert tier == "supporter"

    def test_key_for_deleted_owner_falls_back_to_free(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A key whose account vanished still resolves (to the free tier)."""
        store = InMemoryIdentityStore()
        service = AuthService(store, _NullMailer())
        monkeypatch.setattr(accounts, "_service", service)

        key_store = auth.APIKeyStore()
        raw, _key = key_store.create_key("k", "ghost-account")
        monkeypatch.setattr(auth, "_api_key_store", key_store)

        middleware = QuotaMiddleware(app=_dummy_asgi)
        principal, tier = middleware._resolve_principal(_request_with_api_key(raw))
        assert principal == "ghost-account"
        assert tier == "free"


class TestEmailChangeThrottle:
    """The email-change request is throttled per acting account."""

    def test_rules_are_registered(self) -> None:
        """Unregistered actions are silently allowed, so the rules must exist."""
        from omni_mercury_engine.api.rate_limit_store import DEFAULT_ACTION_RULES

        assert "email_change_ip" in DEFAULT_ACTION_RULES
        assert "email_change_account" in DEFAULT_ACTION_RULES

    def test_request_throttled_after_account_limit(self, setup: Setup) -> None:
        """Past the per-account ceiling the request is 429 with Retry-After."""
        client, mailer, _service = setup
        csrf = _register_verify_login(client, mailer)
        limit = 3  # DEFAULT email_change_account max_attempts
        for _ in range(limit):
            ok = client.post(
                "/api/v1/auth/email-change/request",
                json={"new_email": "new@b.com", "current_password": "a-strong-pw"},
                headers={"X-CSRF-Token": csrf},
            )
            assert ok.status_code == 202
        blocked = client.post(
            "/api/v1/auth/email-change/request",
            json={"new_email": "new@b.com", "current_password": "a-strong-pw"},
            headers={"X-CSRF-Token": csrf},
        )
        assert blocked.status_code == 429
        assert "Retry-After" in blocked.headers


def test_api_key_permission_constants_are_a_safe_subset() -> None:
    """The whitelist excludes every privileged verb (guards a future edit)."""
    assert accounts._SELF_SERVICE_KEY_PERMISSIONS.issubset(
        {Permission.READ, Permission.DETECT, Permission.EXPORT}
    )
    assert Permission.ADMIN not in accounts._SELF_SERVICE_KEY_PERMISSIONS
    assert Permission.WRITE not in accounts._SELF_SERVICE_KEY_PERMISSIONS
    assert Permission.DELETE not in accounts._SELF_SERVICE_KEY_PERMISSIONS
