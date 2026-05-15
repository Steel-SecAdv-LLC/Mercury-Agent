"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests for ``omni_mercury_engine.datasets.base.http_get_with_retry`` and
``safe_urlretrieve``.

These two helpers are the bulk-loader entry points (BATADAL, NAB, SMD,
SMAP/MSL, UCR, ADRepository) into the central :class:`SafeHTTPClient`
gate.  Coverage here pins:

* the removal of the ``allow_untrusted`` per-call escape hatch (PR
  #210) at the dataset-helper API surface;
* the retry-on-transient / no-retry-on-permanent split that lets a
  benchmark run fail fast on a configuration fault and fail over on
  a flaky CDN;
* the User-Agent injection that public CDNs (raw.githubusercontent.com,
  www.fema.gov, www.ncei.noaa.gov) require to avoid silent 403/rate
  limiting;
* the ``safe_urlretrieve`` HTTPS-only contract (no ``allow_http``
  carve-out at the helper level).
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest

from omni_mercury_engine.datasets.base import (
    http_get_with_retry,
    safe_urlretrieve,
)
from omni_mercury_engine.security.safe_http import UnsafeURLError


class TestAllowUntrustedRemovedFromHelper:
    """The dataset-helper API MUST not accept ``allow_untrusted``.

    PR #210 deletes the escape hatch from ``http_get_with_retry``.
    These tests pin the removal at the public signature so a future
    refactor cannot silently reintroduce it.
    """

    def test_http_get_with_retry_rejects_allow_untrusted_kwarg(self) -> None:
        with pytest.raises(TypeError, match="allow_untrusted"):
            http_get_with_retry(  # type: ignore[call-arg]
                "https://earthquake.usgs.gov/fdsnws/event/1/query",
                allow_untrusted=True,
            )

    def test_http_get_with_retry_signature_has_no_allow_untrusted(self) -> None:
        """Belt-and-braces: the kwarg name is not in the signature."""
        sig = inspect.signature(http_get_with_retry)
        assert "allow_untrusted" not in sig.parameters


class TestSchemeGate:
    """``safe_urlretrieve`` and ``http_get_with_retry`` are HTTPS-only by default."""

    @pytest.mark.parametrize(
        "url",
        [
            "ftp://example.com/data",
            "file:///etc/passwd",
            "data:text/plain,hello",
            "javascript:alert(1)",
            "http://earthquake.usgs.gov/path",  # http rejected by default
        ],
    )
    def test_http_get_with_retry_rejects_bad_scheme(self, url: str) -> None:
        """Helper refuses the URL before any network attempt."""
        with patch(
            "omni_mercury_engine.security.safe_http.requests.Session"
        ) as session_factory:
            with pytest.raises((UnsafeURLError, ValueError)):
                http_get_with_retry(url, retries=1, backoff=0.0)
            assert session_factory.call_count == 0, (
                "Bad scheme reached requests.Session despite the gate."
            )

    def test_safe_urlretrieve_refuses_http(self, tmp_path) -> None:
        """``safe_urlretrieve`` does not opt into ``allow_http``."""
        with pytest.raises(UnsafeURLError, match="scheme 'http'"):
            safe_urlretrieve(
                "http://earthquake.usgs.gov/path",
                tmp_path / "out.bin",
            )

    def test_safe_urlretrieve_refuses_unlisted_https_host(self, tmp_path) -> None:
        """``safe_urlretrieve`` enforces TRUSTED_DOMAINS for https too."""
        with pytest.raises(UnsafeURLError, match="not in trusted allowlist"):
            safe_urlretrieve(
                "https://attacker.example.com/exfil",
                tmp_path / "out.bin",
            )


class TestRetrySemantics:
    """Permanent failures fail fast; transient failures retry."""

    def test_permanent_4xx_does_not_retry(self) -> None:
        """A 404 (not in ``retry_on_status``) raises on attempt 1."""
        import requests

        response = MagicMock()
        response.status_code = 404
        error = requests.HTTPError("404 client error", response=response)

        with patch(
            "omni_mercury_engine.security.safe_http.SafeHTTPClient.get_bytes",
            side_effect=error,
        ) as get_bytes:
            with pytest.raises(requests.HTTPError):
                http_get_with_retry(
                    "https://earthquake.usgs.gov/path",
                    retries=3,
                    backoff=0.0,
                )
        assert get_bytes.call_count == 1, (
            "Permanent 4xx should not retry; the helper kept hammering."
        )

    def test_transient_429_retries_until_success(self) -> None:
        """A 429 in ``retry_on_status`` retries up to the limit then succeeds."""
        import requests

        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited_error = requests.HTTPError(
            "429 too many requests", response=rate_limited
        )

        # First two attempts: 429. Third: bytes payload.
        side_effects = [rate_limited_error, rate_limited_error, b"payload"]

        with patch(
            "omni_mercury_engine.security.safe_http.SafeHTTPClient.get_bytes",
            side_effect=side_effects,
        ) as get_bytes:
            result = http_get_with_retry(
                "https://earthquake.usgs.gov/path",
                retries=3,
                backoff=0.0,
            )
        assert result == b"payload"
        assert get_bytes.call_count == 3

    def test_transient_socket_error_retries_then_raises_last(self) -> None:
        """Socket errors retry up to the limit, then re-raise the last one."""
        import requests

        attempts: list[Exception] = [
            TimeoutError("attempt 1"),
            ConnectionError("attempt 2"),
            requests.RequestException("attempt 3"),
        ]

        with patch(
            "omni_mercury_engine.security.safe_http.SafeHTTPClient.get_bytes",
            side_effect=attempts,
        ) as get_bytes:
            with pytest.raises(requests.RequestException, match="attempt 3"):
                http_get_with_retry(
                    "https://earthquake.usgs.gov/path",
                    retries=3,
                    backoff=0.0,
                )
        assert get_bytes.call_count == 3

    def test_configuration_fault_does_not_retry(self) -> None:
        """A ``UnsafeURLError`` (ValueError subclass) raises on attempt 1.

        ``http_get_with_retry`` does not catch ``ValueError`` in its
        retry block (it catches ``requests.HTTPError`` and transient
        socket exceptions). The original raise from the inner
        ``SafeHTTPClient.get_bytes`` propagates straight up.
        """
        with patch(
            "omni_mercury_engine.security.safe_http.SafeHTTPClient.get_bytes",
            side_effect=UnsafeURLError("test: forced refusal"),
        ) as get_bytes:
            with pytest.raises(UnsafeURLError, match="forced refusal"):
                http_get_with_retry(
                    "https://earthquake.usgs.gov/path",
                    retries=3,
                    backoff=0.0,
                )
        assert get_bytes.call_count == 1


class TestUserAgentInjection:
    """The helper injects a Mercury-Agent User-Agent unless the caller overrides it.

    Many public dataset CDNs return 403 to requests with no UA. The
    helper enforces an identifying UA suffix so the loader bulk path
    is not silently throttled.
    """

    def test_default_user_agent_is_mercury(self) -> None:
        with patch(
            "omni_mercury_engine.security.safe_http.SafeHTTPClient.get_bytes",
            return_value=b"ok",
        ) as get_bytes:
            http_get_with_retry(
                "https://earthquake.usgs.gov/path",
                retries=1,
                backoff=0.0,
            )
        headers = get_bytes.call_args.kwargs["headers"]
        assert "User-Agent" in headers
        assert "Mercury-Agent" in headers["User-Agent"]

    def test_caller_user_agent_overrides_default(self) -> None:
        with patch(
            "omni_mercury_engine.security.safe_http.SafeHTTPClient.get_bytes",
            return_value=b"ok",
        ) as get_bytes:
            http_get_with_retry(
                "https://earthquake.usgs.gov/path",
                headers={"User-Agent": "custom-agent/0.1"},
                retries=1,
                backoff=0.0,
            )
        headers = get_bytes.call_args.kwargs["headers"]
        assert headers["User-Agent"] == "custom-agent/0.1"


class TestSafeUrlretrieve:
    """``safe_urlretrieve`` writes the response body to disk atomically.

    The helper creates the target parent directory and writes the
    bytes via ``open(..., 'wb')``; the SafeHTTPClient gate fires
    inside the delegated ``http_get_with_retry`` call.
    """

    def test_writes_body_to_target_path(self, tmp_path) -> None:
        target = tmp_path / "sub" / "out.bin"  # parent does not exist yet
        with patch(
            "omni_mercury_engine.security.safe_http.SafeHTTPClient.get_bytes",
            return_value=b"hello-mercury",
        ):
            safe_urlretrieve(
                "https://earthquake.usgs.gov/fdsnws/event/1/query",
                target,
            )
        assert target.read_bytes() == b"hello-mercury"

    def test_uses_120s_timeout(self, tmp_path) -> None:
        """The helper sets a 120s default timeout for slow CDNs."""
        with patch(
            "omni_mercury_engine.security.safe_http.SafeHTTPClient.get_bytes",
            return_value=b"ok",
        ) as get_bytes:
            safe_urlretrieve(
                "https://earthquake.usgs.gov/fdsnws/event/1/query",
                tmp_path / "out.bin",
            )
        assert get_bytes.call_args.kwargs["timeout"] == 120
