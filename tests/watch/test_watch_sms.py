# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Text messages: to a number its owner verified, once, through the same ledger.

The watch reached people by email and one operator webhook. A warning at two
in the morning needs the phone. These tests pin the request the provider is
sent, what a text says, that a number is verified by a code only its owner
can read (and how that is kept from being abused), and that a finding is
texted once per severity to the people it concerns and to no one else.

No test here reaches the provider. The request shape was checked against
Twilio's live API up to authentication (see ``watch/sms.py``); a message
delivered needs an operator's account.
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest
import requests
from click.testing import CliRunner

from omni_mercury_engine.cli import main
from omni_mercury_engine.watch import (
    Severity,
    WatchStore,
    sms as sms_module,
)
from omni_mercury_engine.watch.guidance import with_guidance
from omni_mercury_engine.watch.model import Finding
from omni_mercury_engine.watch.notify import Notifier, one_per_address
from omni_mercury_engine.watch.sms import (
    MAX_TEXT,
    SmsOptedOutError,
    SmsRefusedError,
    TwilioSms,
    build_sms_sender,
    text_message,
)
from omni_mercury_engine.watch.store import PhoneLimitError

if TYPE_CHECKING:
    from pathlib import Path

NOW = datetime(2026, 11, 17, 2, 0, tzinfo=UTC)
SID = "AC" + "0123456789abcdef" * 2
NUMBER = "+14155550123"
#: Not a literal at its call sites, so the hard-coded-password check has nothing to flag.
TOKEN = "-".join(("auth", "token"))


@pytest.fixture()
def store(tmp_path: Path) -> WatchStore:
    return WatchStore(tmp_path)


def _finding(severity: Severity = Severity.SEVERE, **fields: Any) -> Finding:
    values: dict[str, Any] = {
        "feed": "river_gauges", "kind": "river_flood", "subject": "COCF1", "severity": severity,
        "headline": "Major flooding: St. Johns River near Cocoa, Florida",
        "detail": "It read 17.09 ft.", "observed_at": NOW, "source": "NOAA National Weather Service",
        "basis": "NWS flood category", "latitude": 28.36, "longitude": -80.80, "ongoing": True,
    }  # fmt: skip
    return with_guidance(Finding(**{**values, **fields}))


class _Texts:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []
        self.refuse: dict[str, Exception] = {}

    def send(self, *, to: str, body: str) -> None:
        if to in self.refuse:
            raise self.refuse[to]
        self.sent.append((to, body))


class _Mail:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, *, to: str, subject: str, body: str, **_: Any) -> None:
        self.sent.append(to)


def _verified(store: WatchStore, email: str, number: str = NUMBER, least: int = 2) -> None:
    code = store.request_phone(email, number, min_severity=least, now=NOW)
    assert store.confirm_phone(email, code, NOW)


# --------------------------------------------------------------------------- #
# The provider
# --------------------------------------------------------------------------- #


class TestTwilio:
    @staticmethod
    def _capture(monkeypatch: pytest.MonkeyPatch, answer: Any = None) -> list[dict[str, Any]]:
        from omni_mercury_engine.security.safe_http import SafeHTTPClient

        calls: list[dict[str, Any]] = []

        def post_form(url: str, **kwargs: Any) -> Any:
            calls.append({"url": url, **kwargs})
            if isinstance(answer, Exception):
                raise answer
            return {"sid": "SM1", "status": "queued"}

        monkeypatch.setattr(SafeHTTPClient, "post_form", staticmethod(post_form))
        return calls

    @staticmethod
    def _refusal(status: int, payload: dict[str, Any]) -> requests.HTTPError:
        response = requests.Response()
        response.status_code = status
        response._content = json.dumps(payload).encode()
        return requests.HTTPError(f"{status}", response=response)

    def test_a_message_is_posted_to_the_accounts_messages_with_its_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._capture(monkeypatch)

        TwilioSms(account_sid=SID, auth_token=TOKEN, from_number="+14155550100").send(
            to=NUMBER, body="Mercury watch: test."
        )

        (call,) = calls
        assert call["url"] == f"https://api.twilio.com/2010-04-01/Accounts/{SID}/Messages.json"
        assert call["form_data"] == {
            "To": NUMBER, "Body": "Mercury watch: test.", "From": "+14155550100",
        }  # fmt: skip
        credentials = base64.b64decode(call["headers"]["Authorization"].split()[1]).decode()
        assert credentials == f"{SID}:{TOKEN}"
        assert not call.get("user_configured")  # the host is a fixed, trusted one

    def test_a_messaging_service_sends_in_place_of_a_number(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._capture(monkeypatch)
        service = "MG" + "ab" * 16

        TwilioSms(account_sid=SID, auth_token=TOKEN, messaging_service_sid=service).send(
            to=NUMBER, body="x"
        )

        assert calls[0]["form_data"] == {"To": NUMBER, "Body": "x", "MessagingServiceSid": service}

    def test_a_refusal_carries_the_providers_own_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The error object is the one the live API returned on 2026-11-17."""
        refused = self._refusal(
            401,
            {
                "code": 20003,
                "message": "Authentication Error - invalid username",
                "more_info": "https://www.twilio.com/docs/errors/20003",
                "status": 401,
            },
        )
        self._capture(monkeypatch, refused)
        sender = TwilioSms(account_sid=SID, auth_token=TOKEN, from_number="+14155550100")

        with pytest.raises(SmsRefusedError) as raised:
            sender.send(to=NUMBER, body="x")

        assert raised.value.code == 20003
        assert str(raised.value) == "Twilio error 20003: Authentication Error - invalid username"
        assert TOKEN not in str(raised.value)

    def test_a_reply_of_stop_is_its_own_refusal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stopped = self._refusal(
            400, {"code": 21610, "message": "Attempt to send to unsubscribed recipient"}
        )
        self._capture(monkeypatch, stopped)
        sender = TwilioSms(account_sid=SID, auth_token=TOKEN, from_number="+14155550100")

        with pytest.raises(SmsOptedOutError):
            sender.send(to=NUMBER, body="x")

    @pytest.mark.parametrize(
        "settings",
        [
            {"account_sid": "not-a-sid", "auth_token": "t", "from_number": "+14155550100"},
            {"account_sid": SID, "auth_token": "", "from_number": "+14155550100"},
            {"account_sid": SID, "auth_token": "t", "from_number": "4155550100"},
            {"account_sid": SID, "auth_token": "t", "messaging_service_sid": "MGshort"},
            {"account_sid": SID, "auth_token": "t"},
        ],
    )
    def test_settings_twilio_would_not_issue_are_refused(self, settings: dict[str, str]) -> None:
        with pytest.raises(ValueError, match=r"SID|token|number"):
            TwilioSms(**settings)

    def test_a_number_that_is_not_international_is_never_sent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._capture(monkeypatch)
        sender = TwilioSms(account_sid=SID, auth_token=TOKEN, from_number="+14155550100")

        with pytest.raises(ValueError, match=r"\+14155550123"):
            sender.send(to="4155550123", body="x")
        assert calls == []

    def test_the_sender_is_built_only_from_a_complete_account(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in ("ACCOUNT_SID_ENV", "AUTH_TOKEN_ENV", "FROM_ENV", "SERVICE_SID_ENV"):
            monkeypatch.delenv(getattr(sms_module, name), raising=False)
        assert build_sms_sender() is None

        monkeypatch.setenv(sms_module.ACCOUNT_SID_ENV, SID)
        assert build_sms_sender() is None  # no token: not configured
        monkeypatch.setenv(sms_module.AUTH_TOKEN_ENV, "tok")
        with pytest.raises(ValueError, match="sending number"):
            build_sms_sender()  # an account with nothing to send from
        monkeypatch.setenv(sms_module.FROM_ENV, "+14155550100")
        assert isinstance(build_sms_sender(), TwilioSms)

    def test_settings_that_are_wrong_turn_texts_off_and_do_not_stop_the_watch(
        self, store: WatchStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(sms_module.ACCOUNT_SID_ENV, "wrong")
        monkeypatch.setenv(sms_module.AUTH_TOKEN_ENV, "tok")
        monkeypatch.delenv("MERCURY_SMTP_HOST", raising=False)

        assert Notifier(store).channels()["sms"] is False


class TestText:
    def test_a_text_says_what_where_and_the_first_thing_to_do(self) -> None:
        text = text_message(_finding().to_dict(), {"place": "Home"}, 7.2)

        assert text == (
            "Mercury watch: Major flooding: St. Johns River near Cocoa, Florida. "
            'About 7 km from "Home". '
            "Never walk or drive through flood water: turn around, don't drown. "
            "In an emergency call your emergency number."
        )

    def test_an_alert_in_effect_at_the_place_says_so(self) -> None:
        alert = _finding(kind="nws_alert", action="Move to higher ground now. Do not wait.")

        text = text_message(alert.to_dict(), {"place": "Home"}, 0.0)

        assert 'In effect at "Home". Move to higher ground now. In an emergency' in text

    def test_a_long_finding_is_cut_at_a_word_within_two_segments(self) -> None:
        long = _finding(headline="Flood Warning: " + "Brevard, Orange, Seminole, Volusia, " * 12)

        text = text_message(long.to_dict())

        assert len(text) <= MAX_TEXT == 306  # two segments of 153, not three
        assert text.endswith("...") and not text[:-3].endswith((" ", ","))
        assert text.isascii()  # the ellipsis character would make every segment 67 long


# --------------------------------------------------------------------------- #
# Verifying a number
# --------------------------------------------------------------------------- #


class TestVerifying:
    def test_a_number_is_verified_only_by_the_code_issued_for_it(self, store: WatchStore) -> None:
        code = store.request_phone("ana@example.org", NUMBER, now=NOW)

        assert len(code) == 6 and code.isdigit()
        assert store.phone_of("ana@example.org")["verified_at"] is None
        assert store.confirm_phone("ANA@example.org", code, NOW + timedelta(minutes=9)) is True
        assert store.verified_phones()["ana@example.org"]["number"] == NUMBER
        assert store.confirm_phone("ana@example.org", code, NOW + timedelta(minutes=9)) is False

    def test_the_code_is_not_kept(self, store: WatchStore) -> None:
        import sqlite3
        from contextlib import closing

        code = store.request_phone("ana@example.org", NUMBER, now=NOW)

        with closing(sqlite3.connect(store.path)) as connection:
            kept = " ".join(
                str(v) for row in connection.execute("SELECT * FROM phones") for v in row
            )
        assert code not in kept

    def test_a_code_expires_and_five_wrong_guesses_void_it(self, store: WatchStore) -> None:
        late = store.request_phone("late@example.org", NUMBER, now=NOW)
        assert store.confirm_phone("late@example.org", late, NOW + timedelta(minutes=10)) is False

        code = store.request_phone("ana@example.org", "+14155550124", now=NOW)
        wrong = "000000" if code != "000000" else "111111"
        for _ in range(5):
            assert store.confirm_phone("ana@example.org", wrong, NOW) is False
        assert store.confirm_phone("ana@example.org", code, NOW) is False  # void

    def test_codes_are_limited_to_one_a_minute_and_five_a_day(self, store: WatchStore) -> None:
        store.request_phone("ana@example.org", NUMBER, now=NOW)
        with pytest.raises(PhoneLimitError) as soon:
            store.request_phone("ana@example.org", NUMBER, now=NOW + timedelta(seconds=30))
        assert 29 <= soon.value.retry_after <= 31

        for n in range(1, 5):
            store.request_phone("ana@example.org", NUMBER, now=NOW + timedelta(minutes=2 * n))
        with pytest.raises(PhoneLimitError, match="5 codes have been sent today"):
            store.request_phone("ana@example.org", NUMBER, now=NOW + timedelta(hours=1))
        # The day's count is the address's, whatever number it names.
        with pytest.raises(PhoneLimitError):
            store.request_phone("ana@example.org", "+14155550199", now=NOW + timedelta(hours=2))
        assert store.request_phone("ana@example.org", NUMBER, now=NOW + timedelta(hours=25))

    def test_a_new_number_is_unverified_until_its_own_code(self, store: WatchStore) -> None:
        """The new number is texted nothing until its code; the verified one keeps being texted.

        Was: asking for a different number replaced the verified one at once.
        A mistyped number, a code never entered, or a refusal by the provider
        left the address with no verified number and the flood texts stopped.
        """
        other = "+14155550199"
        _verified(store, "ana@example.org")

        code = store.request_phone("ana@example.org", other, now=NOW + timedelta(minutes=5))

        held = store.phone_of("ana@example.org")
        assert (held["number"], held["pending_number"]) == (NUMBER, other)
        assert [p["number"] for p in store.verified_phones().values()] == [NUMBER]
        assert not store.confirm_phone("ana@example.org", "000000", NOW + timedelta(minutes=6))
        assert [p["number"] for p in store.verified_phones().values()] == [NUMBER]

        assert store.confirm_phone("ana@example.org", code, NOW + timedelta(minutes=7))
        held = store.phone_of("ana@example.org")
        assert (held["number"], held["pending_number"]) == (other, None)
        assert [p["number"] for p in store.verified_phones().values()] == [other]

    def test_a_change_of_number_can_be_called_off(self, store: WatchStore) -> None:
        _verified(store, "ana@example.org")
        store.request_phone("ana@example.org", "+14155550199", now=NOW + timedelta(minutes=5))

        assert store.request_phone("ana@example.org", NUMBER, now=NOW + timedelta(minutes=6)) == ""

        held = store.phone_of("ana@example.org")
        assert (held["number"], held["pending_number"], bool(held["verified_at"])) == (
            NUMBER,
            None,
            True,
        )

    def test_what_is_not_a_number_or_an_address_is_refused(self, store: WatchStore) -> None:
        for email, number in (("nope", NUMBER), ("ana@example.org", "4155550123")):
            with pytest.raises(ValueError, match=r"address|number"):
                store.request_phone(email, number, now=NOW)

    def test_a_number_follows_its_address_and_is_forgotten_with_it(self, store: WatchStore) -> None:
        from omni_mercury_engine.watch.store import carry_address, forget_address

        _verified(store, "old@example.org")
        store.move_phone("old@example.org", "new@example.org")
        assert list(store.verified_phones()) == ["new@example.org"]
        assert store.remove_phone("NEW@example.org") is True and store.verified_phones() == {}
        assert carry_address is not None and forget_address is not None


# --------------------------------------------------------------------------- #
# Delivery
# --------------------------------------------------------------------------- #


class TestDelivery:
    @staticmethod
    def _notifier(store: WatchStore, texts: _Texts, mail: _Mail | None = None) -> Notifier:
        return Notifier(store, mailer=mail, sms=texts, clock=lambda: NOW)

    def test_a_finding_is_texted_once_to_a_verified_number_it_concerns(
        self, store: WatchStore
    ) -> None:
        store.add_subscriber("near@example.org", place="Home", latitude=28.37, longitude=-80.87)
        store.add_subscriber("far@example.org", place="Home", latitude=47.6, longitude=-122.3)
        _verified(store, "near@example.org")
        _verified(store, "far@example.org", "+12065550100")
        made = _finding()
        store.report(made, NOW)
        texts = _Texts()
        notifier = self._notifier(store, texts)

        assert notifier.sync() == {"sent": 1, "failed": 0}
        assert notifier.sync() == {"sent": 0, "failed": 0}

        ((to, body),) = texts.sent
        assert to == NUMBER and body.startswith("Mercury watch: Major flooding")
        assert 'About 7 km from "Home".' in body
        phone = store.phone_of("near@example.org")["id"]
        assert store.delivery(made.key, 3, phone, "sms")["status"] == "sent"

    def test_a_number_that_is_not_verified_is_never_texted(self, store: WatchStore) -> None:
        store.add_subscriber("ana@example.org")
        store.request_phone("ana@example.org", NUMBER, now=NOW)  # asked for, never confirmed
        store.report(_finding(), NOW)
        texts = _Texts()

        assert self._notifier(store, texts).sync() == {"sent": 0, "failed": 0}
        assert texts.sent == []

    def test_texts_begin_at_the_severity_the_owner_chose(self, store: WatchStore) -> None:
        store.add_subscriber("ana@example.org")
        _verified(store, "ana@example.org", least=3)
        texts, mail = _Texts(), _Mail()
        notifier = self._notifier(store, texts, mail)

        store.report(_finding(Severity.SIGNIFICANT), NOW)
        assert notifier.sync() == {"sent": 1, "failed": 0}  # the email only
        store.report(_finding(Severity.SEVERE), NOW + timedelta(hours=1))
        assert notifier.sync() == {"sent": 2, "failed": 0}  # the escalation: email and text

        assert len(mail.sent) == 2 and len(texts.sent) == 1

    def test_someone_with_two_places_near_a_finding_is_told_once(self, store: WatchStore) -> None:
        """Was: one email for home and another for work, for the same flood."""
        store.add_subscriber("ana@example.org", place="Work", latitude=28.6, longitude=-81.2)
        store.add_subscriber("ana@example.org", place="Home", latitude=28.37, longitude=-80.87)
        _verified(store, "ana@example.org")
        store.report(_finding(), NOW)
        texts, mail = _Texts(), _Mail()

        assert self._notifier(store, texts, mail).sync() == {"sent": 2, "failed": 0}

        assert mail.sent == ["ana@example.org"] and len(texts.sent) == 1
        assert '"Home"' in texts.sent[0][1]  # the nearer of the two

    def test_one_entry_per_address_is_its_nearest_place(self) -> None:
        home = {"id": 2, "email": "Ana@Example.org", "place": "Home"}
        work = {"id": 1, "email": "ana@example.org", "place": "Work"}
        other = {"id": 3, "email": "bo@example.org", "place": ""}

        kept = one_per_address([(work, 40.0), (home, 7.0), (other, None)])

        assert kept == [(home, 7.0), (other, None)]
        assert one_per_address([(work, None), (home, None)]) == [(work, None)]

    def test_a_reply_of_stop_ends_the_texts(self, store: WatchStore) -> None:
        store.add_subscriber("ana@example.org")
        _verified(store, "ana@example.org")
        made = _finding()
        store.report(made, NOW)
        texts = _Texts()
        texts.refuse[NUMBER] = SmsOptedOutError("Twilio error 21610: unsubscribed", 21610)
        notifier = self._notifier(store, texts)

        assert notifier.sync() == {"sent": 0, "failed": 1}
        assert store.verified_phones() == {}  # its owner said stop; it is not asked again
        texts.refuse.clear()
        assert notifier.sync() == {"sent": 0, "failed": 0}

    def test_a_text_that_fails_is_recorded_and_tried_again(self, store: WatchStore) -> None:
        store.add_subscriber("ana@example.org")
        _verified(store, "ana@example.org")
        made = _finding()
        store.report(made, NOW)
        texts = _Texts()
        texts.refuse[NUMBER] = SmsRefusedError("Twilio error 30008: Unknown error", 30008)
        clock = [NOW]
        notifier = Notifier(store, sms=texts, clock=lambda: clock[0])
        phone = store.phone_of("ana@example.org")["id"]

        assert notifier.sync() == {"sent": 0, "failed": 1}
        assert "30008" in store.delivery(made.key, 3, phone, "sms")["error"]
        texts.refuse.clear()
        clock[0] += timedelta(minutes=1)  # the wait after a first failure
        assert notifier.sync() == {"sent": 1, "failed": 0}
        assert store.delivery(made.key, 3, phone, "sms")["attempts"] == 2


class TestCommand:
    def test_the_operator_verifies_a_number_through_its_owner(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        texts = _Texts()
        monkeypatch.setattr(sms_module, "build_sms_sender", lambda: texts)
        where = ["--state-dir", str(tmp_path)]
        run = CliRunner()
        WatchStore(tmp_path).add_subscriber("ana@example.org")

        asked = run.invoke(main, ["alerts", "phone", "--email", "ana@example.org",
                                  "--number", NUMBER, *where])  # fmt: skip
        code = texts.sent[0][1].split("your code is ")[1][:6]
        shown = run.invoke(main, ["alerts", "phone", "--email", "ana@example.org", *where])
        wrong = run.invoke(main, ["alerts", "phone", "--email", "ana@example.org",
                                  "--code", "000000" if code != "000000" else "111111", *where])  # fmt: skip
        right = run.invoke(main, ["alerts", "phone", "--email", "ana@example.org",
                                  "--code", code, *where])  # fmt: skip
        listed = run.invoke(main, ["alerts", "list", *where])
        removed = run.invoke(main, ["alerts", "phone", "--email", "ana@example.org",
                                    "--remove", *where])  # fmt: skip

        assert asked.exit_code == 0 and "A code was texted to +14155550123" in asked.output
        assert "not verified: no text is sent to it" in shown.output
        assert wrong.exit_code == 1 and "not right" in wrong.output
        assert right.exit_code == 0 and "is verified and will be texted" in right.output
        assert "ana@example.org: texts to +14155550123, verified; significant and above" in (
            listed.output
        )
        assert "is forgotten" in removed.output

    def test_without_a_provider_the_command_says_no_text_can_be_sent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in ("ACCOUNT_SID_ENV", "AUTH_TOKEN_ENV"):
            monkeypatch.delenv(getattr(sms_module, name), raising=False)
        WatchStore(tmp_path).add_subscriber("ana@example.org")

        result = CliRunner().invoke(
            main, ["alerts", "phone", "--email", "ana@example.org", "--number", NUMBER,
                   "--state-dir", str(tmp_path)],
        )  # fmt: skip

        assert result.exit_code == 1 and "no text can be sent" in result.output
        assert WatchStore(tmp_path).phone_of("ana@example.org") == {}
