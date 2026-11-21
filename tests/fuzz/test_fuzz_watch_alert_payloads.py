# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fuzz the watch and alert payload paths: what a finding becomes on its way out.

A detection or a watch finding leaves the process four ways -- the alert bus
and its server-sent-event frames, the mail subject, the public broadcast, and
the delivery error kept when a webhook fails.  Hypothesis drives arbitrary
payloads through each and asserts what must hold for *any* of them:

* **Alert frames** -- ``to_alert`` is total over arbitrary entries, and every
  frame is one SSE event whose data line is strict JSON.
* **Visibility** -- an anonymous subscriber is only ever handed unowned
  entries; an account only its own and the unowned; no queue outgrows its bound.
* **Mail subject** -- one line, no control character, whatever the headline.
* **Broadcast** -- never raises, and never puts ``place_ids`` (where a
  subscriber is) on the public bus.
* **Webhook secret** -- wherever it sits in the URL and however the failure
  is worded, no part of it survives into what is kept.

Defects the fuzzer found, each pinned by a test below that fails on the
prior code:

* **A webhook secret survived every connection failure.**  urllib3 words one
  as ``"Max retries exceeded with url: /services/T0/B0/<secret>"`` -- the
  path alone, no scheme, no host.  The value pass looked for the whole URL
  and the structural pass for ``https://``, so the secret reached the
  transport error, the WARNING log and the stored delivery error.
* The structural pass also kept ``user:<secret>@`` in front of a host, and
  stopped at an apostrophe (a legal path character), leaving the rest.
* ``to_alert`` raised on an entry whose ``detection`` was not a mapping, or
  whose fields were not the types the wording reads, ending the subscriber's
  stream; and a frame carrying a non-finite float was not JSON, so the
  console dropped it without a word.
* A mail subject kept raw control characters from a headline (a sender must
  not generate them: RFC 5322 sections 2.2 and 4).
"""

from __future__ import annotations

import asyncio
import json
import os
import unicodedata
from datetime import UTC, datetime
from typing import Any
from unittest import mock

import pytest
from hypothesis import (
    HealthCheck,
    assume,
    given,
    settings,
    strategies as st,
)

pytest.importorskip("fastapi", reason="the alert stream needs the [api] extra")

from omni_mercury_engine.api.voice import _sse
from omni_mercury_engine.narrative.alert_bus import (
    QUEUE_SIZE,
    UNREADABLE_RECORD,
    AlertBus,
    principal_may_see,
    to_alert,
)
from omni_mercury_engine.security.redaction import redact_env_secrets, redact_text
from omni_mercury_engine.security.safe_http import _redact_full
from omni_mercury_engine.watch.model import Severity
from omni_mercury_engine.watch.notify import (
    WEBHOOK_URL_ENV,
    Notifier,
    _scrub,
    _strip_url_paths,
    notice,
)

_SLOW = [HealthCheck.too_slow]
_leaves = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**9), max_value=10**9),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(max_size=48),
)
_json_values = st.recursive(
    _leaves,
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(st.text(max_size=12), children, max_size=5),
    ),
    max_leaves=16,
)
# Every field the wording of a detection reads, each free to be the wrong
# type or absent, plus noise.
_RECORD_KEYS = (
    "anomaly_detected",
    "severity",
    "severity_label",
    "detection_id",
    "anomaly_score",
    "anomaly_count",
    "n_samples",
    "threshold",
    "indices",
    "scope",
    "method",
    "confidence",
)
_record = st.fixed_dictionaries(
    {},
    optional={
        **{key: st.one_of(_leaves, st.lists(_leaves, max_size=3)) for key in _RECORD_KEYS},
        "noise": _json_values,
    },
)
_entry = st.fixed_dictionaries(
    {},
    optional={
        "alert": st.one_of(_json_values, st.dictionaries(st.text(max_size=12), _json_values)),
        "detection": st.one_of(_record, _leaves, st.lists(_leaves, max_size=3)),
        "timestamp": _leaves,
        "domain": _leaves,
        "principal": st.one_of(st.none(), st.text(max_size=24)),
        "noise": _json_values,
    },
)


def _parse_strict(text: str) -> Any:
    def refuse(token: str) -> Any:
        raise ValueError(f"not JSON: {token}")

    return json.loads(text, parse_constant=refuse)


# --------------------------------------------------------------------------- #
# Alert frames
# --------------------------------------------------------------------------- #
class TestAlertFrames:
    @settings(max_examples=600, deadline=None, suppress_health_check=_SLOW)
    @given(entry=_entry)
    def test_every_entry_becomes_one_strict_json_frame(self, entry: dict[str, Any]) -> None:
        alert = to_alert(entry)  # total: must not raise
        assert isinstance(alert, dict)
        frame = _sse("alert", alert).decode()
        head, data, blank, end = frame.split("\n")
        assert (head, blank, end) == ("event: alert", "", "")
        assert data.startswith("data: ")
        _parse_strict(data[len("data: ") :])  # what the console's JSON.parse needs

    def test_a_non_mapping_detection_is_an_empty_record(self) -> None:
        """Fails on the prior code (TypeError / ValueError from ``dict(...)``)."""
        for detection in (5, 0.5, True, "flagged", ["a"]):
            alert = to_alert({"detection": detection, "domain": "general"})
            assert alert["initiative_type"] == "clean_run"
            assert alert["event"] == {}

    def test_a_record_that_cannot_be_worded_still_alerts(self) -> None:
        """Fails on the prior code (the wording raised inside the stream)."""
        mistyped = {"anomaly_detected": True, "anomaly_count": 1, "n_samples": 9, "indices": 7}
        alert = to_alert({"detection": mistyped})
        assert alert["initiative_type"] == "anomaly_detected"  # the flag is kept
        assert alert["message"] == UNREADABLE_RECORD  # and it says what it could not do
        # A time that is not a number could not be worded either.
        assert to_alert({"timestamp": float("-inf")})["message"] == UNREADABLE_RECORD

    def test_a_non_finite_value_is_sent_as_null(self) -> None:
        """Fails on the prior code (``NaN`` / ``Infinity`` are not JSON)."""
        frame = _sse("alert", {"severity": float("nan"), "event": {"scores": [1.0, float("inf")]}})
        payload = _parse_strict(frame.decode().split("data: ", 1)[1])
        assert payload == {"severity": None, "event": {"scores": [1.0, None]}}


# --------------------------------------------------------------------------- #
# Visibility
# --------------------------------------------------------------------------- #
_ANONYMOUS = (None, "", "anonymous")
_principals = st.one_of(st.sampled_from(_ANONYMOUS), st.sampled_from(["alice", "bob", "carol"]))


class TestAlertBusVisibility:
    @settings(max_examples=150, deadline=None, suppress_health_check=_SLOW)
    @given(
        subscribers=st.lists(_principals, min_size=1, max_size=5),
        owners=st.lists(st.one_of(st.none(), _principals), min_size=1, max_size=20),
    )
    def test_each_subscriber_is_handed_only_what_it_may_see(
        self, subscribers: list[str | None], owners: list[str | None]
    ) -> None:
        async def run() -> list[list[Any]]:
            bus = AlertBus()
            queues = [bus.subscribe(principal)[1] for principal in subscribers]
            for number, owner in enumerate(owners):
                bus.publish({"principal": owner, "n": number})  # must not raise
            await asyncio.sleep(0)  # let the staged entries drain
            return [[q.get_nowait()["principal"] for _ in range(q.qsize())] for q in queues]

        expected: list[str | None]
        for viewer, received in zip(subscribers, asyncio.run(run()), strict=True):
            assert len(received) <= QUEUE_SIZE
            if viewer in _ANONYMOUS:
                # No identity to own records with: unowned entries only.
                assert all(owner is None for owner in received)
                expected = [owner for owner in owners if owner is None]
            else:
                assert all(owner is None or owner == viewer for owner in received)
                expected = [owner for owner in owners if owner is None or owner == viewer]
            assert received == expected

    @settings(max_examples=300)
    @given(viewer=st.one_of(st.none(), st.text(max_size=12)), owner=st.text(max_size=12))
    def test_an_owned_entry_is_visible_to_its_authenticated_owner_alone(
        self, viewer: str | None, owner: str
    ) -> None:
        anonymous = viewer in _ANONYMOUS
        assert principal_may_see(viewer, owner) is (not anonymous and viewer == owner)
        assert principal_may_see(viewer, None) is True


# --------------------------------------------------------------------------- #
# Mail subject and public broadcast
# --------------------------------------------------------------------------- #
_finding = st.fixed_dictionaries(
    {"headline": _leaves, "detail": _leaves, "source": _leaves, "basis": _leaves},
    optional={
        "severity": _leaves,
        "area": _leaves,
        "action": _leaves,
        "action_source": _leaves,
        "feed": _leaves,
        "key": _leaves,
        "kind": st.one_of(_leaves, st.just("nws_alert")),
        "place_ids": _json_values,
    },
)


class TestMailSubject:
    @settings(max_examples=600, deadline=None, suppress_health_check=_SLOW)
    @given(
        finding=_finding,
        subscriber=st.one_of(
            st.none(), st.fixed_dictionaries({}, optional={"place": _leaves, "email": _leaves})
        ),
        away=st.one_of(st.none(), st.floats(allow_nan=True, allow_infinity=True)),
        note=st.text(max_size=30),
    )
    def test_the_subject_is_one_line_of_printable_text(
        self, finding: dict[str, Any], subscriber: dict[str, Any] | None, away: Any, note: str
    ) -> None:
        subject, body = notice(finding, subscriber, away, note)  # must not raise
        assert subject.startswith("[Mercury watch] ")
        assert not any(unicodedata.category(char) == "Cc" for char in subject)
        assert subject == " ".join(subject.split())  # one line, single-spaced
        assert isinstance(body, str)

    def test_control_characters_in_a_headline_are_dropped(self) -> None:
        """Fails on the prior code, which removed line breaks only."""
        finding = {
            "headline": "Flood\x00 warning\x1b[31m for\x7f Kent\x85 County",
            "detail": "d",
            "source": "s",
            "basis": "b",
        }
        assert notice(finding)[0] == "[Mercury watch] Notable: Flood warning [31m for Kent County"


class _Broadcaster:
    """The part of a Notifier that ``broadcast`` reads."""

    @staticmethod
    def _clock() -> datetime:
        return datetime.now(UTC)


class TestPublicBroadcast:
    @settings(max_examples=400, deadline=None, suppress_health_check=_SLOW)
    @given(finding=_finding)
    def test_a_broadcast_never_raises_and_never_says_where_a_subscriber_is(
        self, finding: dict[str, Any]
    ) -> None:
        bus = mock.Mock()
        with mock.patch("omni_mercury_engine.narrative.alert_bus.get_alert_bus", return_value=bus):
            published = Notifier.broadcast(_Broadcaster(), finding)  # type: ignore[arg-type]
        assert isinstance(published, bool)
        assert bus.publish.call_count == (1 if published else 0)
        if finding.get("place_ids"):
            # Recorded only because it is in effect at a subscriber's place:
            # public only as an Extreme alert of the National Weather Service.
            extreme = (
                finding.get("kind") == "nws_alert"
                and finding.get("severity") == Severity.SEVERE.label
            )
            assert published is False or extreme
        if published:
            entry = bus.publish.call_args.args[0]
            assert entry["principal"] is None  # unowned: every open console may see it
            assert "place_ids" not in entry["alert"]["event"]
            _parse_strict(_sse("alert", to_alert(entry)).decode().split("data: ", 1)[1])


# --------------------------------------------------------------------------- #
# Webhook secret
# --------------------------------------------------------------------------- #
# RFC 3986 path characters (unreserved, sub-delims, ":" and "@"): what a
# secret in a URL may be made of. Includes the apostrophe.
_PCHAR = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~!$&'()*+,;=:@"
_secret = st.text(alphabet=_PCHAR, min_size=12, max_size=40)
_host = st.sampled_from(
    ["hooks.example.invalid", "ntfy.example.invalid:8443", "192.0.2.7", "[2001:db8::1]"]
)
_position = st.sampled_from(["path", "query", "userinfo", "query_without_path"])
# How a failure names the URL: bare, requests' HTTPError, quoted, angled.
_wording = st.sampled_from(
    [
        "{url}",
        "404 Client Error: Not Found for url: {url}",
        "url='{url}' failed",
        'POST "{url}" -> 500',
        "see <{url}>",
    ]
)


def _url(secret: str, host: str, position: str) -> str:
    return {
        "path": f"https://{host}/services/T0000000/{secret}",
        "query": f"https://{host}/hook?k={secret}",
        "userinfo": f"https://user:{secret}@{host}/topic",
        "query_without_path": f"https://{host}?k={secret}",
    }[position]


def _survives(secret: str, text: str) -> bool:
    """Whether any 6 consecutive characters of the secret are still in ``text``."""
    return any(secret[i : i + 6] in text for i in range(len(secret) - 5))


class TestWebhookSecret:
    @settings(max_examples=800, deadline=None, suppress_health_check=_SLOW)
    @given(secret=_secret, host=_host, position=_position, wording=_wording)
    def test_no_url_keeps_more_than_its_host(
        self, secret: str, host: str, position: str, wording: str
    ) -> None:
        """Structure alone, nothing known about the secret. Fails on the prior code."""
        kept = _strip_url_paths(wording.format(url=_url(secret, host, position)))
        assert not _survives(secret, kept)
        assert f"https://{host}/..." in kept  # the host stays, for diagnosis

    @settings(max_examples=400, deadline=None, suppress_health_check=_SLOW)
    @given(secret=_secret, host=_host, position=_position)
    def test_a_connection_failure_names_the_path_alone_and_is_still_scrubbed(
        self, secret: str, host: str, position: str
    ) -> None:
        """urllib3's wording: the request target, no scheme, no host. Fails on the prior code."""
        assume(position != "userinfo")  # userinfo is never part of a request target
        url = _url(secret, host, position)
        target = url.split(host, 1)[1] or "/"
        raw = (
            f"HTTPSConnectionPool(host='{host}', port=443): Max retries exceeded with url: "
            f"{target} (Caused by NewConnectionError('Connection refused'))"
        )
        with mock.patch.dict(os.environ, {WEBHOOK_URL_ENV: url}):
            assert not _survives(secret, redact_env_secrets(raw))  # the root
            assert not _survives(secret, _redact_full(redact_text(raw)))  # the transport
            assert not _survives(secret, _scrub(raw))  # the delivery layer
            assert "Max retries exceeded" in _scrub(raw)  # the diagnosis survives

    def test_a_path_with_no_secret_is_not_struck_from_ordinary_text(self) -> None:
        """A short path holds no secret; redacting it would corrupt diagnostics."""
        with mock.patch.dict(os.environ, {WEBHOOK_URL_ENV: "https://hooks.example.invalid/hook"}):
            assert redact_env_secrets("POST /hook failed: 503") == "POST /hook failed: 503"
