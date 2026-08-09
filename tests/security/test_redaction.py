# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Contracts for the canonical credential-redaction primitives.

These pin the leak-closure behaviour every consumer relies on:
``safe_http`` message redaction, the ``data_sources`` transport scrub,
the ``datasets`` exception constructors, and ``scripts/live_data_smoke``.
Every message shape asserted here was reproduced against the real
transport libraries (httpx / requests / urllib3) before being written
down — the shapes are measurements, not guesses.
"""

from __future__ import annotations

import pytest

from omni_mercury_engine.security.redaction import (
    CREDENTIAL_QUERY_PARAMS,
    REDACTED,
    redact_env_secrets,
    redact_secrets,
    redact_text,
    redact_url,
)


class TestRedactUrl:
    """Structural URL redaction: credential positions go, diagnostics stay."""

    @pytest.mark.parametrize(
        ("url", "secret"),
        [
            # NASA DONKI / NeoWs / EIA style: api_key query param.
            ("https://api.nasa.gov/DONKI/GST?api_key=SEC11&startDate=2026-01-01", "SEC11"),
            # AirNow style: upper-case API_KEY.
            ("https://www.airnowapi.org/aq/data/?API_KEY=SEC22&parameters=OZONE", "SEC22"),
            # Alpha Vantage style: apikey.
            ("https://www.alphavantage.co/query?apikey=SEC33&function=DAILY", "SEC33"),
            # OpenWeatherMap style: appid.
            ("https://api.openweathermap.org/data/2.5/weather?appid=SEC44&q=x", "SEC44"),
            # Hyphenated header-style name.
            ("https://h.example/v1?x-api-key=SEC55", "SEC55"),
            # OAuth-ish token names.
            ("https://h.example/cb?access_token=SEC66&state=ok", "SEC66"),
        ],
    )
    def test_credential_query_values_redacted(self, url: str, secret: str) -> None:
        redacted = redact_url(url)
        assert secret not in redacted
        assert REDACTED in redacted

    def test_diagnostics_survive(self) -> None:
        """Scheme, host, path and non-credential params must remain readable."""
        redacted = redact_url("https://api.eia.gov/v2/electricity/rto?api_key=K12345&freq=daily")
        assert redacted.startswith("https://api.eia.gov/v2/electricity/rto?")
        assert "freq=daily" in redacted
        assert "K12345" not in redacted

    def test_userinfo_redacted(self) -> None:
        redacted = redact_url("https://user:hunter2@host.example/data")
        assert "hunter2" not in redacted
        assert "user" not in redacted  # usernames are frequently API keys
        assert "host.example/data" in redacted

    def test_relative_reference_query_redacted(self) -> None:
        assert "tok99" not in redact_url("/path?token=tok99&keep=this")
        assert "keep=this" in redact_url("/path?token=tok99&keep=this")

    def test_non_url_text_passes_through(self) -> None:
        assert redact_url("<no Location header>") == "<no Location header>"

    def test_unparseable_fails_closed(self) -> None:
        """A URL urlsplit cannot parse must NOT pass through unredacted."""
        assert redact_url("https://[invalid-v6?api_key=SEC") == "<unparseable-url:redacted>"

    def test_no_credential_params_byte_identical(self) -> None:
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&limit=100"
        assert redact_url(url) == url

    def test_param_matching_is_case_and_separator_insensitive(self) -> None:
        for name in ("API_KEY", "Api-Key", "APIKEY", "map_key", "MAP-KEY"):
            assert "S3CRET" not in redact_url(f"https://h.example/p?{name}=S3CRET"), name

    def test_credential_param_set_covers_real_upstreams(self) -> None:
        """The names Mercury's real keyed upstreams use must be in the set."""
        for required in ("api_key", "apikey", "appid", "map_key", "token", "signature"):
            assert required in CREDENTIAL_QUERY_PARAMS


class TestRedactText:
    """Free-text redaction: both URL shapes transport libraries produce."""

    def test_httpx_status_error_shape(self) -> None:
        """httpx renders '... for url \\'<full url>\\'' — measured shape."""
        msg = (
            "Client error '403 Forbidden' for url "
            "'https://api.nasa.gov/DONKI/GST?api_key=SECRET123&startDate=2026-01-01'"
        )
        redacted = redact_text(msg)
        assert "SECRET123" not in redacted
        assert "api.nasa.gov/DONKI/GST" in redacted
        assert "403 Forbidden" in redacted

    def test_urllib3_schemeless_shape(self) -> None:
        """urllib3 renders 'Max retries exceeded with url: /path?query' — no scheme."""
        msg = (
            "HTTPSConnectionPool(host='www.alphavantage.co', port=443): "
            "Max retries exceeded with url: /query?apikey=SECRET123&function=DAILY "
            "(Caused by NewConnectionError('...'))"
        )
        redacted = redact_text(msg)
        assert "SECRET123" not in redacted
        assert "function=DAILY" in redacted
        assert "www.alphavantage.co" in redacted

    def test_benign_prose_byte_identical(self) -> None:
        prose = "why?not sure. also x?y and end? plus a=b&c=d alone."
        assert redact_text(prose) == prose

    def test_idempotent(self) -> None:
        """Layers stack (scrub → exception constructor → script redaction):
        a second pass over already-redacted text must be a no-op, never a
        doubled token."""
        once = redact_text(
            "err for url 'https://h.example/p?api_key=SEC&x=1' and "
            "with url: /q?token=TOK123 tail"
        )
        assert redact_text(once) == once
        assert "SEC" not in once and "TOK123" not in once

    def test_empty_and_none_shapes(self) -> None:
        assert redact_text("") == ""


class TestRedactSecrets:
    """Value-based redaction: the only defence for path-segment keys."""

    def test_long_value_replaced_everywhere(self) -> None:
        text = "GET /api/area/csv/LONGMAPKEY99/VIIRS_SNPP_NRT failed; body echoed LONGMAPKEY99"
        redacted = redact_secrets(text, ("NASA_FIRMS_MAP_KEY",), ("LONGMAPKEY99",))
        assert "LONGMAPKEY99" not in redacted
        assert "<NASA_FIRMS_MAP_KEY:redacted>" in redacted
        assert "/VIIRS_SNPP_NRT" in redacted

    def test_url_encoded_form_replaced(self) -> None:
        """A key with special characters appears percent-encoded in composed URLs."""
        redacted = redact_secrets("enc LONGSECRET%2B99 form", ("K",), ("LONGSECRET+99",))
        assert "LONGSECRET%2B99" not in redacted
        assert "<K:redacted>" in redacted

    def test_short_value_boundary_guarded(self) -> None:
        """4-7 char values are replaced standalone but never mangle words
        that merely contain them — the fix for the old 8-char floor that
        skipped short-but-real keys entirely."""
        redacted = redact_secrets("key=ABC123 ok; but FABC123X stays", ("V",), ("ABC123",))
        assert "key=<V:redacted> ok" in redacted
        assert "FABC123X" in redacted

    def test_degenerate_value_skipped(self) -> None:
        """<4 chars cannot be a real credential; replacing would corrupt text."""
        text = "do not mangle: monkey key on and 1"
        assert redact_secrets(text, ("V", "W", "X"), ("on", "1", "")) == text

    def test_none_and_whitespace_ignored(self) -> None:
        text = "nothing to do"
        assert redact_secrets(text, ("A", "B"), (None, "   ")) == text

    def test_mismatched_channel_lengths_raise(self) -> None:
        """Silent zip-truncation would drop a value and let it through
        unredacted — a length mismatch is a programming error, not a
        redaction outcome."""
        with pytest.raises(ValueError, match="position-matched"):
            redact_secrets("text", ("A", "B"), ("LONGSECRET99",))

    def test_bare_string_channel_raises(self) -> None:
        """A ``str`` is itself a ``Sequence[str]``; iterating one would
        silently redact per character instead of per label."""
        with pytest.raises(TypeError, match="not a single string"):
            redact_secrets("text", "LABEL", ("LONGSECRET99",))
        with pytest.raises(TypeError, match="not a single string"):
            redact_secrets("text", ("LABEL",), "LONGSECRET99")


class TestRedactEnvSecrets:
    """Env-driven value redaction — no per-callsite registry to forget."""

    def test_configured_key_scrubbed_from_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A FIRMS-style path-segment key is invisible to structural
        redaction; the env pass is what catches it."""
        monkeypatch.setenv("NASA_FIRMS_MAP_KEY", "PATHKEY777")
        redacted = redact_env_secrets(
            "refusing https://firms.modaps.eosdis.nasa.gov/api/area/csv/PATHKEY777/VIIRS/1"
        )
        assert "PATHKEY777" not in redacted
        assert "<NASA_FIRMS_MAP_KEY:redacted>" in redacted

    @pytest.mark.parametrize(
        "var",
        ["EIA_API_KEY", "FHIR_BEARER_TOKEN", "DEXCOM_REFRESH_TOKEN", "MY_CLIENT_SECRET"],
    )
    def test_credential_shaped_names_covered(
        self, var: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(var, "SENSITIVE-VALUE-1")
        assert "SENSITIVE-VALUE-1" not in redact_env_secrets("saw SENSITIVE-VALUE-1 here")

    def test_non_credential_names_not_scrubbed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AUTHOR / TOKENIZER_PATH style names must not trigger scrubbing."""
        monkeypatch.setenv("AUTHOR", "some-benign-value")
        monkeypatch.setenv("TOKENIZER_PATH", "another-benign-value")
        text = "some-benign-value and another-benign-value"
        assert redact_env_secrets(text) == text
