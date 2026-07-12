# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
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
    """Every keyed HTTP checker reports the real transport reason, never "HTTP 0"."""
    monkeypatch.setattr(mod, "_get", lambda *a, **k: (0, _TRANSPORT_ERR.encode()))
    ok, detail = getattr(mod, checker)("dummy-key")
    assert ok is False
    assert "HTTP 0" not in detail, f"{checker} collapsed a transport error to 'HTTP 0'"
    assert "transport error" in detail and "timed out" in detail


@pytest.mark.parametrize(
    "checker",
    ["check_eia", "check_fred", "check_nasa", "check_alpha_vantage", "check_openweathermap"],
)
def test_checkers_surface_http_status_on_real_http_error(
    mod: Any, monkeypatch: pytest.MonkeyPatch, checker: str
) -> None:
    """A genuine HTTP error still reports its status (and any provider message)."""
    monkeypatch.setattr(mod, "_get", lambda *a, **k: (401, b'{"error":"bad key"}'))
    ok, detail = getattr(mod, checker)("dummy-key")
    assert ok is False
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
    ok, detail = getattr(mod, checker)(_SECRET)
    assert ok is False
    assert _SECRET not in detail, f"{checker} leaked the credential: {detail}"
