# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline, deterministic unit tests for ``scripts/verify_data_credentials.py``.

The live-delivery checks in :mod:`tests.test_data_credentials_network` need real
secrets and are ``@pytest.mark.network`` (skipped in the default lane). These
tests exercise the *failure-reporting* logic with no network at all, by driving
each checker through a monkeypatched ``_get``. They exist to lock in the fix for
the Copilot review finding that a transport failure was being collapsed to the
useless string ``"HTTP 0"``: on a transport error (``_get`` returns status ``0``)
the actual DNS/TLS/timeout reason must reach the CI report so a credential-
delivery failure is diagnosable rather than opaque.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_data_credentials.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("verify_data_credentials", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod() -> Any:
    return _load_module()


# --- _fail_detail: the reduction from (status, body) to a report line ---------


class TestFailDetail:
    def test_transport_error_surfaces_the_exception_text(self, mod: Any) -> None:
        # status 0 is _get's transport-error sentinel; the body holds the reason.
        detail = mod._fail_detail(0, b"HTTPSConnectionPool: Max retries exceeded (NXDOMAIN)")
        assert "transport error" in detail
        assert "NXDOMAIN" in detail
        # The regression we are guarding against: a bare "HTTP 0" with no reason.
        assert detail != "HTTP 0"
        assert "HTTP 0" not in detail

    def test_transport_error_with_empty_body_is_still_labelled(self, mod: Any) -> None:
        detail = mod._fail_detail(0, b"")
        assert "transport error" in detail
        assert "HTTP 0" not in detail

    def test_http_error_reports_status_and_body_preview(self, mod: Any) -> None:
        detail = mod._fail_detail(403, b'{"error":"invalid api key"}')
        assert detail.startswith("HTTP 403")
        assert "invalid api key" in detail

    def test_http_error_with_empty_body_is_just_the_status(self, mod: Any) -> None:
        assert mod._fail_detail(500, b"") == "HTTP 500"

    def test_long_bodies_are_truncated(self, mod: Any) -> None:
        # Neither branch may dump an unbounded provider body into the log.
        assert len(mod._fail_detail(0, b"x" * 5000)) <= len("transport error: ") + 200
        assert len(mod._fail_detail(400, b"y" * 5000)) <= len("HTTP 400: ") + 160


# --- Each HTTP checker must route its non-200 path through _fail_detail --------

_TRANSPORT_ERR = "ConnectTimeoutError: connection timed out"


@pytest.mark.parametrize(
    "checker",
    ["check_eia", "check_fred", "check_nasa", "check_alpha_vantage", "check_openweathermap"],
)
def test_checkers_surface_transport_error_not_http_zero(
    mod: Any, monkeypatch: pytest.MonkeyPatch, checker: str
) -> None:
    """Every keyed HTTP checker reports the real transport reason, never "HTTP 0".

    The verdict is ``UNREACH``, not ``FAIL``: a connection that never completed
    is the upstream's problem and says nothing about the credential. Only
    ``FAIL`` fails the lane, so mis-classifying this would make an outage look
    like a bad key.
    """
    monkeypatch.setattr(mod, "_get", lambda *a, **k: (0, _TRANSPORT_ERR.encode()))
    verdict, detail = getattr(mod, checker)("dummy-key")
    assert verdict == mod.UNREACHABLE
    assert "HTTP 0" not in detail, f"{checker} collapsed a transport error to 'HTTP 0'"
    assert "transport error" in detail and "timed out" in detail


@pytest.mark.parametrize(
    "checker",
    ["check_eia", "check_fred", "check_nasa", "check_alpha_vantage", "check_openweathermap"],
)
def test_checkers_surface_http_status_on_real_http_error(
    mod: Any, monkeypatch: pytest.MonkeyPatch, checker: str
) -> None:
    """A genuine HTTP error still reports its status (and any provider message).

    A 401 is the provider rejecting the credential we presented, which is the
    one thing this lane exists to catch -- so it must be ``FAIL``, the verdict
    that fails the lane, and never be softened to ``UNREACH``.
    """
    monkeypatch.setattr(mod, "_get", lambda *a, **k: (401, b'{"error":"bad key"}'))
    verdict, detail = getattr(mod, checker)("dummy-key")
    assert verdict == mod.FAIL
    assert detail.startswith("HTTP 401")


# --- Credential redaction: a transport error must never leak the key ----------

_SECRET = "SUPERSECRETKEY12345"


class TestRedaction:
    @pytest.mark.parametrize(
        "raw",
        [
            f"?api_key={_SECRET}&frequency=hourly",
            f"?apikey={_SECRET}",
            f"?appid={_SECRET}&q=London",
            f"?token={_SECRET}",
            f"&password={_SECRET}&x=1",
            f"?API_KEY={_SECRET}",  # case-insensitive
        ],
    )
    def test_redact_strips_credential_query_values(self, mod: Any, raw: str) -> None:
        out = mod._redact(raw)
        assert _SECRET not in out, f"secret leaked through _redact: {out}"
        assert "***" in out

    def test_redact_preserves_non_credential_text(self, mod: Any) -> None:
        assert mod._redact("host='api.eia.gov' port=443 frequency=hourly") == (
            "host='api.eia.gov' port=443 frequency=hourly"
        )

    def test_fail_detail_does_not_leak_key_from_transport_error(self, mod: Any) -> None:
        # A realistic requests transport error embeds the full URL incl. the key.
        exc = (
            "HTTPSConnectionPool(host='api.eia.gov', port=443): Max retries exceeded with url: "
            f"/v2/electricity/rto/region-data/data/?api_key={_SECRET}&frequency=hourly "
            "(Caused by NewConnectionError('Failed to establish a new connection'))"
        )
        detail = mod._fail_detail(0, exc.encode())
        assert _SECRET not in detail, f"_fail_detail leaked the API key: {detail}"


@pytest.mark.parametrize(
    "checker",
    ["check_eia", "check_fred", "check_nasa", "check_alpha_vantage", "check_openweathermap"],
)
def test_checkers_never_leak_key_on_transport_error(
    mod: Any, monkeypatch: pytest.MonkeyPatch, checker: str
) -> None:
    """No keyed checker may echo the credential value into its report line."""
    leak = (
        "Max retries exceeded with url: /path?api_key="
        + _SECRET
        + "&apikey="
        + _SECRET
        + "&appid="
        + _SECRET
        + "&token="
        + _SECRET
    )
    monkeypatch.setattr(mod, "_get", lambda *a, **k: (0, leak.encode()))
    verdict, detail = getattr(mod, checker)(_SECRET)
    assert verdict != mod.DELIVERS
    assert _SECRET not in detail, f"{checker} leaked the credential: {detail}"


# --- The FAIL / UNREACH boundary ---------------------------------------------
# Only FAIL fails the lane, so where a status lands decides whether an operator
# is asked to act. Getting this wrong in either direction is costly: classify an
# outage as FAIL and the lane fails for something nobody can fix (which is how a
# gate gets ignored); classify a rejected credential as UNREACH and a genuinely
# broken key ships silently.


@pytest.mark.parametrize(
    "status,expected",
    [
        (0, "UNREACHABLE"),  # transport failure: DNS, TLS, timeout, reset
        (500, "UNREACHABLE"),
        (502, "UNREACHABLE"),
        (503, "UNREACHABLE"),  # the real NASA outage that turned the lane red
        (504, "UNREACHABLE"),
        (429, "UNREACHABLE"),  # rate limit: the provider RECOGNISED the key
        (400, "FAIL"),
        (401, "FAIL"),  # rejected credential
        (403, "FAIL"),
        (404, "FAIL"),
    ],
)
def test_non_200_classification(mod: Any, status: int, expected: str) -> None:
    verdict, _detail = mod._non_200(status, b"body")
    assert verdict == getattr(mod, expected)


@pytest.mark.parametrize(
    "checker",
    ["check_eia", "check_fred", "check_nasa", "check_alpha_vantage", "check_openweathermap"],
)
@pytest.mark.parametrize(
    "body",
    [
        b"",  # the real 2026-10-12 NASA case: HTTP 200, empty body
        b"<html><body>503 Service Unavailable</body></html>",  # CDN error page
        b"upstream connect error or disconnect/reset before headers",
    ],
    ids=["empty-body", "html-error-page", "proxy-text"],
)
def test_http_200_with_non_json_body_is_unreachable(
    mod: Any, monkeypatch: pytest.MonkeyPatch, checker: str, body: bytes
) -> None:
    """An HTTP 200 whose body is not JSON is an outage, not a credential verdict.

    Every provider in this lane answers a rejected credential with a non-200 or
    a parseable JSON error body, so a 200 that cannot be parsed at all carries
    no information about the key. Measured 2026-10-12 in the scheduled run:
    NASA DONKI returned HTTP 200 with an unparseable body and turned the lane
    red for a key that had delivered the day before.
    """
    monkeypatch.setattr(mod, "_get", lambda *a, **k: (200, body))
    verdict, detail = getattr(mod, checker)("dummy-key")
    assert verdict == mod.UNREACHABLE, f"{checker} treated upstream noise as a credential verdict"
    assert "non-JSON" in detail


@pytest.mark.parametrize(
    "checker,body",
    [
        ("check_eia", b'{"response": {}}'),  # parses, but no ["response"]["data"]
        ("check_fred", b'{"error_message": "api_key missing"}'),  # no ["observations"]
        ("check_nasa", b'{"error": "unexpected"}'),  # dict where a list is expected
        ("check_alpha_vantage", b"[1, 2]"),  # list where a dict is expected
        ("check_openweathermap", b"[1, 2]"),
    ],
)
def test_http_200_with_wrong_json_shape_stays_fail(
    mod: Any, monkeypatch: pytest.MonkeyPatch, checker: str, body: bytes
) -> None:
    """A body that parses but lacks the expected shape is still a hard FAIL.

    A parseable-but-unexpected answer can be the provider's own credential
    error, which is exactly what this lane exists to surface -- only genuinely
    unparseable bodies are excused as upstream noise.
    """
    monkeypatch.setattr(mod, "_get", lambda *a, **k: (200, body))
    verdict, detail = getattr(mod, checker)("dummy-key")
    assert verdict == mod.FAIL
    assert "unexpected response shape" in detail


def test_non_json_200_detail_is_redacted(mod: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The body preview in a non-JSON-200 detail must never leak a credential."""
    monkeypatch.setenv("DEMO_API_KEY", _SECRET)
    data, outage = mod._parse_200_body(f"redirecting... ?api_key={_SECRET}".encode())
    assert data is None and outage is not None
    verdict, detail = outage
    assert verdict == mod.UNREACHABLE
    assert _SECRET not in detail


def test_all_sources_unreachable_is_not_a_clean_run(
    mod: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """If every keyed source was unreachable the run verified nothing.

    Exiting 0 there would be a green light with no evidence behind it, so this
    is the one case where UNREACH still fails.
    """
    monkeypatch.setenv("EIA_API_KEY", "dummy-key-value")
    for var in ("FRED_API_KEY", "NASA_API_KEY", "ALPHA_VANTAGE_API_KEY", "OPENWEATHERMAP_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    for var in ("EROSERS_USERNAME", "USGS_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(mod, "_get", lambda *a, **k: (503, b"upstream down"))

    assert mod.run() == 1
    assert "verified no credential" in capsys.readouterr().err


def test_unreachable_alone_does_not_fail_when_another_source_delivered(
    mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An outage beside a real delivery must not fail the lane.

    This is the case the three-state split exists for: measured 2026-08-05,
    five sources delivered and NASA returned 503, and the lane went red on the
    one result that carried no information about any credential.
    """
    monkeypatch.setenv("EIA_API_KEY", "dummy-key-value")
    monkeypatch.setenv("NASA_API_KEY", "dummy-key-value")
    for var in ("FRED_API_KEY", "ALPHA_VANTAGE_API_KEY", "OPENWEATHERMAP_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    for var in ("EROSERS_USERNAME", "USGS_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(mod, "check_eia", lambda _k: (mod.DELIVERS, "7 rows"))
    monkeypatch.setattr(mod, "check_nasa", lambda _k: (mod.UNREACHABLE, "HTTP 503"))
    monkeypatch.setitem(mod.CHECKS, "EIA", (("EIA_API_KEY",), mod.check_eia))
    monkeypatch.setitem(mod.CHECKS, "NASA", (("NASA_API_KEY",), mod.check_nasa))

    assert mod.run() == 0
