# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A person manages their own watch subscriptions from their account.

The watch's messages ended "To stop, ask whoever runs it, or change it under
your account there", and there was nothing under the account: only the
operator's command line could subscribe or unsubscribe anyone. An account's
address is one it proved it receives, so the account may subscribe that
address, and only that address, to the places it names.

These tests pin: that the address is always the account's own; the two
bounds (per account, and the places one alert read looks up); CSRF and
ownership; that a server which cannot send says so; and that the
subscriptions follow the account when its address changes, are in its data
export, and are gone before the account is.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omni_mercury_engine.api.auth_service import AuthService
from omni_mercury_engine.api.identity_store import InMemoryIdentityStore
from omni_mercury_engine.api.routes import accounts
from omni_mercury_engine.watch.feeds import WarningsFeed
from omni_mercury_engine.watch.store import (
    WatchStore,
    carry_address,
    existing_store,
    forget_address,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

URL = "/api/v1/auth/watch-subscriptions"
COCOA = {"place": "Home", "latitude": 28.37, "longitude": -80.87}


class _Mail:
    """Collects sent messages so tests can take the tokens out of them."""

    def __init__(self) -> None:
        self.sent: list[dict[str, str]] = []

    def send(self, *, to: str, subject: str, body: str, **_: object) -> None:
        self.sent.append({"to": to, "subject": subject, "body": body})

    def last_token(self) -> str:
        match = re.search(r"token=([\w\-]+)", self.sent[-1]["body"])
        assert match is not None
        return match.group(1)


Setup = tuple[TestClient, _Mail]


@pytest.fixture
def setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Setup]:
    monkeypatch.setenv("MERCURY_SESSION_COOKIE_SECURE", "false")
    monkeypatch.setenv("MERCURY_WATCH_DIR", str(tmp_path / "watch"))
    monkeypatch.delenv("MERCURY_SMTP_HOST", raising=False)
    monkeypatch.delenv("MERCURY_WATCH_MAX_PLACES_PER_ACCOUNT", raising=False)
    mailer = _Mail()
    service = AuthService(
        InMemoryIdentityStore(),
        mailer,
        on_email_changed=carry_address,
        on_account_deleting=forget_address,
    )
    app = FastAPI()
    app.include_router(accounts.router)
    app.dependency_overrides[accounts.get_auth_service] = lambda: service
    with TestClient(app) as client:
        yield client, mailer


def _sign_in(client: TestClient, mailer: _Mail, email: str = "ana@example.org") -> dict[str, str]:
    """Register, verify and log in; return the CSRF header for the session."""
    credentials = {"email": email, "password": "a-strong-pw"}
    assert client.post("/api/v1/auth/register", json=credentials).status_code == 201
    token = {"token": mailer.last_token()}
    assert client.post("/api/v1/auth/verify-email", json=token).status_code == 200
    answer = client.post("/api/v1/auth/login", json=credentials)
    assert answer.status_code == 200
    return {"X-CSRF-Token": str(answer.json()["csrf_token"])}


def _store() -> WatchStore:
    store = existing_store()
    assert store is not None
    return store


class TestSubscribing:
    def test_looking_creates_nothing_and_says_what_the_server_can_do(self, setup: Setup) -> None:
        client, mailer = setup
        _sign_in(client, mailer)

        answer = client.get(URL)

        assert answer.status_code == 200
        assert answer.json() == {
            "subscriptions": [],
            "limit": 5,
            "email_delivery": False,
            "watching": False,
        }
        assert existing_store() is None  # reading never makes the store

    def test_a_place_is_subscribed_for_the_accounts_own_address(self, setup: Setup) -> None:
        client, mailer = setup
        csrf = _sign_in(client, mailer)

        answer = client.post(
            URL,
            headers=csrf,
            json={**COCOA, "radius_km": 50, "min_severity": "significant", "email": "x@evil.org"},
        )

        assert answer.status_code == 201
        (listed,) = answer.json()["subscriptions"]
        assert listed == {
            "id": listed["id"], "place": "Home", "latitude": 28.37, "longitude": -80.87,
            "radius_km": 50.0, "min_severity": "significant", "created": listed["created"],
        }  # fmt: skip
        (row,) = _store().subscribers()
        assert (row["email"], row["min_severity"]) == ("ana@example.org", 2)

    def test_naming_a_place_again_changes_it(self, setup: Setup) -> None:
        client, mailer = setup
        csrf = _sign_in(client, mailer)
        client.post(URL, headers=csrf, json=COCOA)

        answer = client.post(URL, headers=csrf, json={**COCOA, "latitude": 28.5, "radius_km": 25})

        (listed,) = answer.json()["subscriptions"]
        assert (listed["latitude"], listed["radius_km"]) == (28.5, 25.0)

    def test_the_server_says_when_it_can_send(
        self, setup: Setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, mailer = setup
        _sign_in(client, mailer)
        monkeypatch.setenv("MERCURY_SMTP_HOST", "smtp.example.org")

        assert client.get(URL).json()["email_delivery"] is True

    @pytest.mark.parametrize(
        "bad",
        [
            {"latitude": 91},
            {"longitude": -181},
            {"radius_km": 0.5},
            {"radius_km": 2001},
            {"min_severity": "everything"},
            {"place": "   "},
            {"place": "x" * 81},
            {"place": "Home\x00"},
        ],
    )
    def test_what_cannot_be_a_place_is_refused(self, setup: Setup, bad: dict[str, object]) -> None:
        client, mailer = setup
        csrf = _sign_in(client, mailer)

        assert client.post(URL, headers=csrf, json={**COCOA, **bad}).status_code == 422
        assert existing_store() is None

    def test_a_name_is_kept_to_one_line(self, setup: Setup) -> None:
        client, mailer = setup
        csrf = _sign_in(client, mailer)

        answer = client.post(URL, headers=csrf, json={**COCOA, "place": "  My\n  house "})

        assert answer.json()["subscriptions"][0]["place"] == "My house"


class TestBounds:
    def test_an_account_is_told_about_at_most_its_limit_of_places(
        self, setup: Setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, mailer = setup
        csrf = _sign_in(client, mailer)
        monkeypatch.setenv("MERCURY_WATCH_MAX_PLACES_PER_ACCOUNT", "2")
        for name in ("Home", "Work"):
            assert client.post(URL, headers=csrf, json={**COCOA, "place": name}).status_code == 201

        third = client.post(URL, headers=csrf, json={**COCOA, "place": "Cabin"})

        assert third.status_code == 409
        assert third.json()["detail"]["message"] == "an account can be told about at most 2 places"
        # Changing one it has is not a third.
        assert client.post(URL, headers=csrf, json={**COCOA, "radius_km": 10}).status_code == 201
        assert len(_store().subscribers()) == 2

    def test_self_service_cannot_pass_the_places_one_alert_read_looks_up(
        self, setup: Setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Past that number the alert feed leaves places unchecked."""
        client, mailer = setup
        csrf = _sign_in(client, mailer)
        monkeypatch.setattr(WarningsFeed, "MAX_PLACES", 1)
        assert client.post(URL, headers=csrf, json=COCOA).status_code == 201

        elsewhere = client.post(
            URL, headers=csrf, json={"place": "Work", "latitude": 47.6, "longitude": -122.3}
        )
        same_point = client.post(URL, headers=csrf, json={**COCOA, "place": "Also home"})

        assert elsewhere.status_code == 409 and "as many places as it can" in elsewhere.text
        assert same_point.status_code == 201  # a place already checked costs nothing more

    def test_the_limits_hold_in_the_store_itself(self, tmp_path: Path) -> None:
        from omni_mercury_engine.watch.store import SubscriptionLimitError

        store = WatchStore(tmp_path)
        store.add_subscriber("a@example.org", place="A", latitude=1.0, longitude=1.0)

        with pytest.raises(SubscriptionLimitError) as address:
            store.add_subscriber(
                "a@example.org", place="B", latitude=1.0, longitude=1.0, address_limit=1
            )
        with pytest.raises(SubscriptionLimitError) as places:
            store.add_subscriber(
                "b@example.org", place="B", latitude=2.0, longitude=2.0, place_limit=1
            )

        assert (address.value.scope, places.value.scope) == ("address", "places")
        assert len(store.subscribers()) == 1  # a refusal writes nothing


class TestWhoMay:
    def test_no_session_no_subscriptions(self, setup: Setup) -> None:
        client, _ = setup

        assert client.get(URL).status_code == 401
        assert client.post(URL, json=COCOA).status_code == 401
        assert client.delete(f"{URL}/1").status_code == 401

    def test_changing_anything_needs_the_csrf_token(self, setup: Setup) -> None:
        client, mailer = setup
        csrf = _sign_in(client, mailer)
        client.post(URL, headers=csrf, json=COCOA)
        (listed,) = client.get(URL).json()["subscriptions"]

        assert client.post(URL, json=COCOA).status_code == 403
        assert client.delete(f"{URL}/{listed['id']}").status_code == 403
        assert len(_store().subscribers()) == 1

    def test_an_account_sees_and_ends_only_its_own(self, setup: Setup) -> None:
        client, mailer = setup
        csrf = _sign_in(client, mailer)
        client.post(URL, headers=csrf, json=COCOA)
        theirs = _store().add_subscriber("bo@example.org", place="Home", latitude=1, longitude=2)
        (mine,) = client.get(URL).json()["subscriptions"]

        assert client.delete(f"{URL}/{theirs}", headers=csrf).status_code == 404
        assert client.delete(f"{URL}/{mine['id']}", headers=csrf).status_code == 200
        assert client.delete(f"{URL}/{mine['id']}", headers=csrf).status_code == 404

        assert [row["email"] for row in _store().subscribers()] == ["bo@example.org"]
        assert client.get(URL).json()["subscriptions"] == []

    def test_one_the_operator_added_for_the_address_is_the_accounts_to_end(
        self, setup: Setup, tmp_path: Path
    ) -> None:
        """The message says "change it under your account": this is that."""
        client, mailer = setup
        added = WatchStore(tmp_path / "watch").add_subscriber("Ana@Example.org")  # everything
        csrf = _sign_in(client, mailer)

        (listed,) = client.get(URL).json()["subscriptions"]

        assert (listed["id"], listed["place"], listed["latitude"]) == (added, "", None)
        assert client.delete(f"{URL}/{added}", headers=csrf).status_code == 200


class TestTheyFollowTheAccount:
    def test_they_are_in_the_data_export(self, setup: Setup) -> None:
        client, mailer = setup
        csrf = _sign_in(client, mailer)
        assert client.get("/api/v1/auth/export").json()["watch_subscriptions"] == []
        client.post(URL, headers=csrf, json=COCOA)

        (exported,) = client.get("/api/v1/auth/export").json()["watch_subscriptions"]

        assert (exported["place"], exported["latitude"]) == ("Home", 28.37)

    def test_they_move_when_the_accounts_address_changes(self, setup: Setup) -> None:
        """Was: nothing moved them; the old mailbox would have gone on being told."""
        client, mailer = setup
        csrf = _sign_in(client, mailer)
        client.post(URL, headers=csrf, json=COCOA)
        asked = client.post(
            "/api/v1/auth/email-change/request",
            headers=csrf,
            json={"new_email": "ana@new.example.org", "current_password": "a-strong-pw"},
        )
        assert asked.status_code in (200, 202), asked.text

        confirmed = client.post(
            "/api/v1/auth/email-change/confirm", json={"token": mailer.last_token()}
        )

        assert confirmed.status_code == 200
        assert [row["email"] for row in _store().subscribers()] == ["ana@new.example.org"]

    def test_they_are_gone_before_the_account_is(self, setup: Setup) -> None:
        """Was: a deleted account's address went on receiving the watch's messages."""
        client, mailer = setup
        csrf = _sign_in(client, mailer)
        client.post(URL, headers=csrf, json=COCOA)
        _store().add_subscriber("bo@example.org")

        wrong = client.post(
            "/api/v1/auth/account/delete", headers=csrf, json={"current_password": "not-it"}
        )
        assert wrong.status_code == 401 and len(_store().subscribers()) == 2

        gone = client.post(
            "/api/v1/auth/account/delete", headers=csrf, json={"current_password": "a-strong-pw"}
        )

        assert gone.status_code == 200
        assert [row["email"] for row in _store().subscribers()] == ["bo@example.org"]

    def test_an_account_is_not_deleted_while_its_address_is_still_subscribed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the subscriptions cannot be ended, the deletion does not go ahead."""
        from omni_mercury_engine.api import passwords

        def cannot(email: str) -> None:
            raise OSError("simulated: the watch's store cannot be written")

        store = InMemoryIdentityStore()
        service = AuthService(store, _Mail(), on_account_deleting=cannot)
        account = service.register("ana@example.org", "a-strong-pw")
        assert passwords.verify_password("a-strong-pw", account.password_hash)

        with pytest.raises(OSError, match="cannot be written"):
            service.delete_account(account.id, "a-strong-pw")

        assert service.get_account(account.id) is not None

    def test_the_built_service_is_wired_to_the_watch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from omni_mercury_engine.api.auth_service import build_auth_service

        monkeypatch.setenv("MERCURY_WATCH_DIR", str(tmp_path / "watch"))
        for name in ("MERCURY_KEYSTORE_PATH", "MERCURY_SMTP_HOST", "MERCURY_AUDIT_LOG_DIR"):
            monkeypatch.delenv(name, raising=False)
        service = build_auth_service()

        assert service._on_email_changed is carry_address
        assert service._on_account_deleting is forget_address
