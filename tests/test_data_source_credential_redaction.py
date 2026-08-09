# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The data_sources transport never leaks a composed credential.

``httpx.HTTPStatusError``'s str embeds the fully-composed request URL
(measured: ``"Client error '403 Forbidden' for url
'https://api.nasa.gov/DONKI/GST?api_key=SECRET...'"``), and the keyed
sources (NASA DONKI / NeoWs ``api_key``, AirNow ``API_KEY``) merge their
credential into that URL as a query parameter.  These tests pin the
transport-layer contract for both ``_http_get`` variants:

* the raised error's chain is SEVERED (``__cause__ is None`` and the
  implicit context suppressed) — the httpx analogue of the loaders-layer
  ``_fetch_url`` fix;
* no credential survives into the raised message, the retry log lines,
  or the exhausted-retries wrap;
* the safe diagnostics (HTTP status, ``status_code`` attribute,
  exception class name) DO survive, so callers can still branch on the
  upstream verdict.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pytest

from omni_mercury_engine.data_sources.base import (
    DataSourceBase,
    DataSourceConfig,
    DataSourceError,
    DataSourceType,
    RateLimitConfig,
    SourceUnreachableError,
)

_SECRET = "SUPERSECRETKEY123"
_KEYED_URL = f"https://api.example.test/v1/data?api_key={_SECRET}&window=7d"


class _KeyedSource(DataSourceBase):
    """Minimal concrete source carrying a live-shaped api_key."""

    DEFAULT_BASE_URL = "https://api.example.test"

    @property
    def source_id(self) -> str:
        return "keyed_test_source"

    @property
    def default_source_types(self) -> list[DataSourceType]:
        return [DataSourceType.SOLAR_FLARE]

    async def _fetch_impl(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


def _make_source(retry_attempts: int = 1) -> _KeyedSource:
    return _KeyedSource(
        DataSourceConfig(
            api_key=_SECRET,
            retry_attempts=retry_attempts,
            rate_limit=RateLimitConfig(min_interval_seconds=0.0),
        )
    )


def _status_response(status_code: int, body: str) -> httpx.Response:
    """A real httpx.Response whose raise_for_status carries the keyed URL."""
    request = httpx.Request("GET", _KEYED_URL)
    return httpx.Response(status_code, request=request, text=body)


class _StubClient:
    """Stands in for httpx.Client/AsyncClient: returns or raises per call."""

    def __init__(self, outcomes: list[httpx.Response | Exception]) -> None:
        self._outcomes = outcomes
        self.calls = 0

    def _next(self) -> httpx.Response:
        outcome = self._outcomes[min(self.calls, len(self._outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _SyncStub(_StubClient):
    def get(self, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
        return self._next()


class _AsyncStub(_StubClient):
    async def get(self, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
        return self._next()


class TestSyncHTTPGetRedaction:
    def test_4xx_chain_severed_and_message_scrubbed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 401 whose body echoes the key raises DataSourceError with the
        chain severed and no credential anywhere in the rendered exception."""
        source = _make_source()
        stub = _SyncStub([_status_response(401, f"Invalid key {_SECRET} rejected")])
        monkeypatch.setattr(source, "_get_sync_client", lambda: stub)

        with pytest.raises(DataSourceError) as exc_info:
            source._http_get_sync("/v1/data", params={"api_key": _SECRET})

        exc = exc_info.value
        assert exc.__cause__ is None, "transport exception chained — URL leak reopened"
        assert exc.__suppress_context__ is True
        assert _SECRET not in str(exc)
        assert exc.status_code == 401

    def test_429_maps_to_unreachable_and_stays_scrubbed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = _make_source()
        stub = _SyncStub([_status_response(429, "slow down")])
        monkeypatch.setattr(source, "_get_sync_client", lambda: stub)

        with pytest.raises(SourceUnreachableError) as exc_info:
            source._http_get_sync("/v1/data", params={"api_key": _SECRET})

        assert exc_info.value.__cause__ is None
        assert exc_info.value.retryable is True
        assert _SECRET not in str(exc_info.value)

    def test_exhausted_5xx_names_class_not_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Retry exhaustion on a 5xx used to interpolate the raw
        HTTPStatusError — whose str embeds the credentialed URL — into the
        message.  The class name survives; the credential does not."""
        source = _make_source(retry_attempts=1)
        stub = _SyncStub([_status_response(503, "upstream sad")])
        monkeypatch.setattr(source, "_get_sync_client", lambda: stub)

        with pytest.raises(SourceUnreachableError) as exc_info:
            source._http_get_sync("/v1/data", params={"api_key": _SECRET})

        message = str(exc_info.value)
        assert "HTTPStatusError" in message
        assert _SECRET not in message
        assert "503" in message

    def test_transport_error_retry_log_scrubbed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Worst-case transport error text carrying the composed URL must
        not reach the retry warning nor the final wrap."""
        source = _make_source(retry_attempts=2)
        error = httpx.ConnectError(f"boom contacting {_KEYED_URL}")
        stub = _SyncStub([error, error])
        monkeypatch.setattr(source, "_get_sync_client", lambda: stub)
        monkeypatch.setattr("time.sleep", lambda _s: None)

        with (
            caplog.at_level(logging.WARNING, logger="omni_mercury_engine.data_sources.base"),
            pytest.raises(SourceUnreachableError) as exc_info,
        ):
            source._http_get_sync("/v1/data", params={"api_key": _SECRET})

        assert _SECRET not in caplog.text
        assert _SECRET not in str(exc_info.value)
        assert "ConnectError" in str(exc_info.value)


class TestAsyncHTTPGetRedaction:
    async def test_4xx_chain_severed_and_message_scrubbed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = _make_source()
        stub = _AsyncStub([_status_response(401, f"Invalid key {_SECRET} rejected")])

        async def _client() -> _AsyncStub:
            return stub

        monkeypatch.setattr(source, "_get_client", _client)

        with pytest.raises(DataSourceError) as exc_info:
            await source._http_get("/v1/data", params={"api_key": _SECRET})

        exc = exc_info.value
        assert exc.__cause__ is None, "transport exception chained — URL leak reopened"
        assert exc.__suppress_context__ is True
        assert _SECRET not in str(exc)
        assert exc.status_code == 401

    async def test_exhausted_5xx_names_class_not_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        source = _make_source(retry_attempts=1)
        stub = _AsyncStub([_status_response(500, "boom")])

        async def _client() -> _AsyncStub:
            return stub

        monkeypatch.setattr(source, "_get_client", _client)

        with pytest.raises(SourceUnreachableError) as exc_info:
            await source._http_get("/v1/data", params={"api_key": _SECRET})

        message = str(exc_info.value)
        assert "HTTPStatusError" in message
        assert _SECRET not in message


class TestFetchResultErrorFunnel:
    """fetch() holds the no-credential contract even for subclass-raised errors.

    Transport errors arrive origin-scrubbed from _http_get, but a
    _fetch_impl can raise DataSourceError with its own composed text —
    the funnel that builds FetchResult.error and the warning log must
    scrub regardless of who raised.
    """

    async def test_subclass_raised_error_scrubbed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        source = _make_source()

        async def _raise_impl(*args: Any, **kwargs: Any) -> list[Any]:
            raise DataSourceError(
                f"upstream call {_KEYED_URL} failed hard",
                source_id="keyed_test_source",
            )

        monkeypatch.setattr(source, "_fetch_impl", _raise_impl)
        with caplog.at_level(logging.WARNING, logger="omni_mercury_engine.data_sources.base"):
            result = await source.fetch(use_cache=False)

        assert result.success is False
        assert _SECRET not in (result.error or "")
        assert _SECRET not in caplog.text
        assert "api.example.test" in (result.error or "")  # diagnostics survive


class TestScrubDiagnosticSink:
    """Direct coverage of ``_scrub_diagnostic``, the funnel every log site uses.

    Every ``logger.warning``/``logger.error`` in this module's error paths
    passes its text through this one helper, so it is the single point where a
    credential either survives into a log line or does not. It had no direct
    test: the surrounding tests exercise it only through ``fetch``, which cannot
    distinguish "the helper scrubbed it" from "the path never carried it".

    It is also what CodeQL's eight `py/clear-text-logging-sensitive-data` alerts
    point at. Those are a sanitizer-modelling artifact rather than leaks -- the
    flow CodeQL follows runs through ``redaction.py``'s
    ``replacement = f"<{label}:redacted>"``, so what it tracks to the log is the
    environment variable *name* taken from ``os.environ.items()``, not the
    credential. ``test_env_var_name_is_what_reaches_the_log`` pins exactly that
    distinction, so the claim is enforced here rather than only asserted in a
    review comment.
    """

    def _source(self) -> _KeyedSource:
        return _make_source()

    def test_query_parameter_credential_is_structurally_redacted(self) -> None:
        src = self._source()
        text = (
            "HTTPSConnectionPool(host='api.example.test'): Max retries exceeded "
            f"with url: /v1/data?api_key={_SECRET}&window=7d "
            "(Caused by NewConnectionError('failed to establish a connection'))"
        )
        out = src._scrub_diagnostic(text)
        assert _SECRET not in out
        # Diagnostics survive: host and path are what name the failing artifact.
        assert "api.example.test" in out
        assert "window=7d" in out

    def test_path_segment_credential_is_value_redacted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """NASA FIRMS puts the key in a URL *path segment*.

        No query-shaped structural rule can see that, so only the value-based
        pass catches it. The value used here is deliberately NOT this source's
        configured ``api_key`` -- that pass runs first and would mask which
        mechanism did the work -- so what is exercised is the environment
        sweep, the one that covers another service's key echoed into a body.
        """
        other = "OTHER_SERVICE_KEY_ABCDEFGH"
        monkeypatch.setenv("MERCURY_PROBE_API_KEY", other)
        src = self._source()
        out = src._scrub_diagnostic(
            f"404 Client Error for url 'https://firms.test/api/area/csv/{other}/VIIRS/1'"
        )
        assert other not in out
        assert "firms.test" in out

    def test_sources_own_configured_key_is_redacted(self) -> None:
        """A key held in config, never in the environment, is still scrubbed."""
        src = self._source()
        out = src._scrub_diagnostic(f"401 for url 'https://h/p?apikey={_SECRET}'")
        assert _SECRET not in out

    def test_env_var_name_is_what_reaches_the_log(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The label is the variable NAME, and the value is absent.

        This is deliberate -- the reader learns *which* credential was scrubbed
        without learning the credential -- and it is the whole content of the
        eight open CodeQL alerts against these log sites.
        """
        other = "OTHER_SERVICE_KEY_ABCDEFGH"
        monkeypatch.setenv("MERCURY_PROBE_API_KEY", other)
        src = self._source()
        out = src._scrub_diagnostic(f"failed fetching https://h/v1/{other}/rows")
        assert "<MERCURY_PROBE_API_KEY:redacted>" in out
        assert other not in out

    def test_scrubbing_is_idempotent(self) -> None:
        """Layers stack: a scrubbed transport string is re-scrubbed downstream."""
        src = self._source()
        once = src._scrub_diagnostic(f"url 'https://h/p?api_key={_SECRET}'")
        assert src._scrub_diagnostic(once) == once
