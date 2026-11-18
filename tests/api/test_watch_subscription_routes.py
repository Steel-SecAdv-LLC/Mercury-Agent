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
from typing import TYPE_CHECKING, Any

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
            "sms_delivery": False,
            "phone": None,
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


# --------------------------------------------------------------------------- #
# the account's phone number for text messages
# --------------------------------------------------------------------------- #
PHONE = "/api/v1/auth/watch-phone"
NUMBER = "+14155550123"


class _Texts:
    """A text-message sender that records, and can be told to refuse."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.refuse: Exception | None = None

    def send(self, *, to: str, body: str) -> None:
        if self.refuse is not None:
            raise self.refuse
        self.sent.append((to, body))

    def code(self) -> str:
        found = re.search(r"your code is (\d{6})", self.sent[-1][1])
        assert found is not None
        return found.group(1)


@pytest.fixture
def texts(monkeypatch: pytest.MonkeyPatch) -> _Texts:
    sender = _Texts()
    monkeypatch.setattr(accounts, "_sms_sender", lambda: sender)
    return sender


class TestPhone:
    def test_a_server_without_a_provider_says_so_and_texts_nothing(self, setup: Setup) -> None:
        client, mailer = setup
        csrf = _sign_in(client, mailer)

        answer = client.post(PHONE, headers=csrf, json={"number": NUMBER})

        assert answer.status_code == 409
        assert answer.json()["detail"]["message"] == (
            "this server is not set up to send text messages"
        )
        assert existing_store() is None

    def test_a_number_is_texted_a_code_and_is_not_verified_until_it_is_typed_back(
        self, setup: Setup, texts: _Texts
    ) -> None:
        client, mailer = setup
        csrf = _sign_in(client, mailer)

        asked = client.post(PHONE, headers=csrf, json={"number": NUMBER, "min_severity": "severe"})

        assert asked.status_code == 202
        assert asked.json()["sms_delivery"] is True
        assert asked.json()["phone"] == {
            "number": NUMBER, "verified": False, "min_severity": "severe", "texting": None,
        }  # fmt: skip
        ((to, body),) = texts.sent
        assert to == NUMBER and "good for 10 minutes" in body
        assert _store().verified_phones() == {}

        wrong = client.post(f"{PHONE}/confirm", headers=csrf, json={"code": "000000"})
        right = client.post(f"{PHONE}/confirm", headers=csrf, json={"code": texts.code()})

        assert wrong.status_code == 400 and "not right" in wrong.json()["detail"]["message"]
        assert right.status_code == 200 and right.json()["phone"]["verified"] is True
        assert list(_store().verified_phones()) == ["ana@example.org"]
        # A code is used once.
        again = client.post(f"{PHONE}/confirm", headers=csrf, json={"code": texts.code()})
        assert again.status_code == 400

    def test_changing_only_the_severity_of_a_verified_number_sends_nothing(
        self, setup: Setup, texts: _Texts
    ) -> None:
        client, mailer = setup
        csrf = _sign_in(client, mailer)
        client.post(PHONE, headers=csrf, json={"number": NUMBER})
        client.post(f"{PHONE}/confirm", headers=csrf, json={"code": texts.code()})

        changed = client.post(
            PHONE, headers=csrf, json={"number": NUMBER, "min_severity": "notable"}
        )

        assert changed.status_code == 202 and len(texts.sent) == 1
        assert changed.json()["phone"] == {
            "number": NUMBER, "verified": True, "min_severity": "notable", "texting": None,
        }  # fmt: skip

    def test_codes_cannot_be_asked_for_faster_than_one_a_minute(
        self, setup: Setup, texts: _Texts
    ) -> None:
        """A form that texts on request is a way to spend money and ring strangers."""
        client, mailer = setup
        csrf = _sign_in(client, mailer)
        assert client.post(PHONE, headers=csrf, json={"number": NUMBER}).status_code == 202

        again = client.post(PHONE, headers=csrf, json={"number": "+14155550199"})

        assert again.status_code == 429 and 1 <= int(again.headers["Retry-After"]) <= 61
        assert len(texts.sent) == 1 and _store().phone_of("ana@example.org")["number"] == NUMBER

    def test_a_number_the_provider_will_not_text_is_said_not_claimed(
        self, setup: Setup, texts: _Texts
    ) -> None:
        from omni_mercury_engine.watch.sms import SmsRefusedError

        client, mailer = setup
        csrf = _sign_in(client, mailer)
        texts.refuse = SmsRefusedError("Twilio error 21211: Invalid 'To' Phone Number", 21211)

        answer = client.post(PHONE, headers=csrf, json={"number": NUMBER})

        assert answer.status_code == 502 and "21211" in answer.json()["detail"]["message"]
        assert _store().verified_phones() == {}

    @pytest.mark.parametrize(
        "bad", ["4155550123", "+0415555", "+1 415 555 0123", "+1415555012345678"]
    )
    def test_what_is_not_an_international_number_is_refused(
        self, setup: Setup, texts: _Texts, bad: str
    ) -> None:
        client, mailer = setup
        csrf = _sign_in(client, mailer)

        assert client.post(PHONE, headers=csrf, json={"number": bad}).status_code == 422
        assert texts.sent == []

    def test_who_may(self, setup: Setup, texts: _Texts) -> None:
        client, mailer = setup
        assert client.post(PHONE, json={"number": NUMBER}).status_code == 401
        csrf = _sign_in(client, mailer)

        assert client.post(PHONE, json={"number": NUMBER}).status_code == 403  # no CSRF token
        assert client.post(f"{PHONE}/confirm", json={"code": "123456"}).status_code == 403
        assert client.delete(PHONE).status_code == 403
        assert client.delete(PHONE, headers=csrf).status_code == 404  # none set
        assert texts.sent == []

    def test_it_is_removed_exported_and_follows_the_account(
        self, setup: Setup, texts: _Texts
    ) -> None:
        client, mailer = setup
        csrf = _sign_in(client, mailer)
        client.post(PHONE, headers=csrf, json={"number": NUMBER})
        client.post(f"{PHONE}/confirm", headers=csrf, json={"code": texts.code()})

        exported = client.get("/api/v1/auth/export").json()["watch_phone"]
        assert exported["number"] == NUMBER and exported["verified_at"]
        assert exported["pending_number"] is None

        client.post(
            "/api/v1/auth/email-change/request",
            headers=csrf,
            json={"new_email": "ana@new.example.org", "current_password": "a-strong-pw"},
        )
        client.post("/api/v1/auth/email-change/confirm", json={"token": mailer.last_token()})
        assert list(_store().verified_phones()) == ["ana@new.example.org"]

        login = client.post(
            "/api/v1/auth/login", json={"email": "ana@new.example.org", "password": "a-strong-pw"}
        )
        csrf = {"X-CSRF-Token": str(login.json()["csrf_token"])}
        assert client.delete(PHONE, headers=csrf).status_code == 200
        assert _store().phone_of("ana@new.example.org") == {}

    def test_a_number_awaiting_its_code_is_in_the_export_too(
        self, setup: Setup, texts: _Texts
    ) -> None:
        """Was: held against the address in the store, and left out of what its owner is given."""
        from datetime import UTC, datetime, timedelta

        client, mailer = setup
        csrf = _sign_in(client, mailer)
        client.post(PHONE, headers=csrf, json={"number": NUMBER})
        client.post(f"{PHONE}/confirm", headers=csrf, json={"code": texts.code()})
        address = next(iter(_store().verified_phones()))
        other = "+14155550199"
        _store().request_phone(address, other, now=datetime.now(UTC) + timedelta(minutes=5))

        exported = client.get("/api/v1/auth/export").json()["watch_phone"]

        assert (exported["number"], exported["pending_number"]) == (NUMBER, other)

    def test_it_is_gone_before_the_account_is(self, setup: Setup, texts: _Texts) -> None:
        client, mailer = setup
        csrf = _sign_in(client, mailer)
        client.post(PHONE, headers=csrf, json={"number": NUMBER})
        client.post(f"{PHONE}/confirm", headers=csrf, json={"code": texts.code()})

        client.post(
            "/api/v1/auth/account/delete", headers=csrf, json={"current_password": "a-strong-pw"}
        )

        assert _store().phone_of("ana@example.org") == {}


class TestWhatTheReviewFound:
    """Each of these was an internal error or a silent wrong state on the code as reviewed."""

    ODD = "o'brien@example.org"  # an address the account system accepts and verifies

    def test_an_account_whose_address_has_an_apostrophe_can_subscribe_and_add_a_phone(
        self, setup: Setup, texts: _Texts
    ) -> None:
        """Was: 500 on both. The watch had its own, stricter, idea of an address."""
        client, mailer = setup
        csrf = _sign_in(client, mailer, self.ODD)

        subscribed = client.post(URL, headers=csrf, json=COCOA)
        phoned = client.post(PHONE, headers=csrf, json={"number": NUMBER})

        assert subscribed.status_code == 201, subscribed.text
        assert phoned.status_code == 202, phoned.text
        assert [row["email"] for row in _store().subscribers()] == [self.ODD]

    def test_a_change_to_such_an_address_carries_the_subscriptions(self, setup: Setup) -> None:
        """Was: the address changed, the request failed, and the old mailbox went on being told."""
        client, mailer = setup
        csrf = _sign_in(client, mailer)
        client.post(URL, headers=csrf, json=COCOA)
        client.post(
            "/api/v1/auth/email-change/request",
            headers=csrf,
            json={"new_email": self.ODD, "current_password": "a-strong-pw"},
        )

        confirmed = client.post(
            "/api/v1/auth/email-change/confirm", json={"token": mailer.last_token()}
        )

        assert confirmed.status_code == 200, confirmed.text
        assert [row["email"] for row in _store().subscribers()] == [self.ODD]

    def test_an_address_change_drops_every_session_whatever_follows_it_does(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Was: a failure in what follows left the old sessions valid and no audit record.

        The hook ran between committing the new address and dropping the
        sessions. Whatever it raised (the watch's store locked, say) ended
        the request there.
        """
        monkeypatch.setenv("MERCURY_SESSION_COOKIE_SECURE", "false")
        monkeypatch.setenv("MERCURY_WATCH_DIR", str(tmp_path / "watch"))
        mailer = _Mail()
        recorded: list[tuple[str, str]] = []

        def locked(previous: str, current: str) -> None:
            raise RuntimeError("database is locked")

        service = AuthService(InMemoryIdentityStore(), mailer, on_email_changed=locked)
        real = service._auditor.record

        def record(event: str, **details: Any) -> Any:
            recorded.append((event, str(details.get("outcome", ""))))
            return real(event, **details)

        monkeypatch.setattr(service._auditor, "record", record)
        app = FastAPI()
        app.include_router(accounts.router)
        app.dependency_overrides[accounts.get_auth_service] = lambda: service
        with TestClient(app) as client:
            csrf = _sign_in(client, mailer)
            assert client.get(URL).status_code == 200
            client.post(
                "/api/v1/auth/email-change/request",
                headers=csrf,
                json={"new_email": "ana@new.example.org", "current_password": "a-strong-pw"},
            )

            confirmed = client.post(
                "/api/v1/auth/email-change/confirm", json={"token": mailer.last_token()}
            )

            assert confirmed.status_code == 200, confirmed.text
            assert client.get(URL).status_code == 401  # the old session is gone
        assert ("email_change", "success") in recorded
        assert ("email_change_followup", "failure") in recorded

    def test_a_provider_that_cannot_be_reached_is_said_and_the_code_is_not_counted(
        self, setup: Setup, texts: _Texts
    ) -> None:
        """Was: an unhandled 500, no audit event, and the requester locked out for a text never sent."""
        from omni_mercury_engine.watch.sms import SmsUnreachableError

        client, mailer = setup
        csrf = _sign_in(client, mailer)
        texts.refuse = SmsUnreachableError("Twilio could not be reached (Timeout)")

        answers = [client.post(PHONE, headers=csrf, json={"number": NUMBER}) for _ in range(7)]

        assert [answer.status_code for answer in answers] == [502] * 7  # never 429, never 500
        assert "could not be reached" in answers[0].json()["detail"]["message"]
        texts.refuse = None
        assert client.post(PHONE, headers=csrf, json={"number": NUMBER}).status_code == 202

    def test_removing_and_adding_a_number_does_not_send_a_sixth_code(
        self, setup: Setup, texts: _Texts, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Was: DELETE then POST, as often as wanted: a paid text to a stranger each time."""
        from datetime import timedelta

        client, mailer = setup
        csrf = _sign_in(client, mailer)
        monkeypatch.setattr(WatchStore, "CODE_GAP", timedelta(0))  # the daily limit is the subject

        statuses = []
        for _ in range(8):
            statuses.append(client.post(PHONE, headers=csrf, json={"number": NUMBER}).status_code)
            client.delete(PHONE, headers=csrf)

        assert statuses == [202] * 5 + [429] * 3
        assert len(texts.sent) == 5
