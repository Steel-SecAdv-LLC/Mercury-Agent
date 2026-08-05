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
