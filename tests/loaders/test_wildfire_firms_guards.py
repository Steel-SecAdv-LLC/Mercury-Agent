# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""FIRMS transport guards for WildfireLoader.

Pins the two failure modes NASA FIRMS actually exhibits under key/quota
problems, both previously invisible to the loader:

* an HTTP 429 whose ``requests.HTTPError`` message embeds the full URL —
  whose path segment IS the MAP key — and which the blanket retry loop used
  to burn ~4 minutes against before surfacing as a bare ``ConnectionError``;
* HTTP-200 text bodies ("Invalid MAP_KEY.", transaction-limit messages)
  that ``pd.read_csv`` silently parsed into a nonsense one-column frame.
"""

from __future__ import annotations

import traceback
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from omni_mercury_engine.loaders.base import FetchHTTPError
from omni_mercury_engine.loaders.wildfire_loader import WildfireLoader

_TEST_KEY = "test-map-key-12345"

_FIRMS_CSV = (
    b"latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
    b"instrument,confidence,version,bright_ti5,frp,daynight\n"
    b"-16.5,124.2,331.2,0.4,0.6,2026-08-01,0130,N,VIIRS,n,2.0NRT,290.1,4.2,N\n"
    b"-16.6,124.3,345.9,0.4,0.6,2026-08-01,0130,N,VIIRS,h,2.0NRT,301.4,12.8,N\n"
)


def _loader(tmp_path: Any) -> WildfireLoader:
    return WildfireLoader(cache_dir=tmp_path / "cache", api_key=_TEST_KEY, max_retries=0)


class TestFIRMSRateLimitGuard:
    def test_429_raises_informative_error_without_leaking_key(self, tmp_path: Any) -> None:
        loader = _loader(tmp_path)
        # The chained HTTPError's message embeds the full URL, key included —
        # exactly what the loader-layer redaction must keep out of tracebacks.
        original = FetchHTTPError(
            f"wildfire: Failed to fetch data after 1 attempt (HTTPError, HTTP 429) "
            f"url=https://firms.example/{_TEST_KEY}/VIIRS",
            status_code=429,
        )
        with (
            patch(
                "omni_mercury_engine.loaders.base.BaseDomainLoader._fetch_url",
                side_effect=original,
            ),
            pytest.raises(ConnectionError) as exc_info,
        ):
            loader.fetch_realtime()
        raised = exc_info.value
        message = str(raised)
        assert "429" in message
        assert "rate limit" in message
        assert exc_info.value.__cause__ is None, (
            "the suppressed cause is the loader-layer key redaction; chaining "
            "it would leak the MAP key via the embedded URL"
        )
        # Full leak contract, same bar as the base layer: context
        # suppressed, and the key absent from every rendering a log
        # sink can produce (str, repr, full traceback with chains).
        assert raised.__suppress_context__ is True
        assert _TEST_KEY not in message
        assert _TEST_KEY not in repr(raised)
        rendered = "".join(traceback.format_exception(raised))
        assert _TEST_KEY not in rendered

    def test_non_429_fetch_errors_propagate_unchanged(self, tmp_path: Any) -> None:
        loader = _loader(tmp_path)
        original = FetchHTTPError("wildfire: Failed to fetch data", status_code=503)
        with (
            patch(
                "omni_mercury_engine.loaders.base.BaseDomainLoader._fetch_url",
                side_effect=original,
            ),
            pytest.raises(FetchHTTPError) as exc_info,
        ):
            loader.fetch_realtime()
        assert exc_info.value is original

    def test_non_429_errors_escape_with_no_key_anywhere_in_chain(self, tmp_path: Any) -> None:
        """End-to-end reproduction of the non-429 re-raise leak finding.

        The REAL ``_fetch_url`` runs (only the transport is stubbed), and
        the simulated ``requests.HTTPError`` embeds the actual request
        URL — whose path segment IS the MAP key. The ``FetchHTTPError``
        escaping ``_fetch_firms_csv``'s non-429 re-raise must sever the
        chain (``__cause__ is None``, context suppressed) and render
        key-free in ``str``, ``repr``, and the full traceback.
        """
        loader = _loader(tmp_path)
        seen_urls: list[str] = []

        def explode(url: str, **kwargs: Any) -> bytes:
            seen_urls.append(url)
            exc = OSError(f"503 Server Error: Service Unavailable for url: {url}")
            exc.response = type("_Resp", (), {"status_code": 503})()  # type: ignore[attr-defined]
            raise exc

        with (
            patch(
                "omni_mercury_engine.loaders.base.SafeHTTPClient.get_bytes",
                side_effect=explode,
            ),
            pytest.raises(FetchHTTPError) as exc_info,
        ):
            loader.fetch_realtime()

        # The scenario is real: the request URL genuinely carried the key.
        assert seen_urls and all(_TEST_KEY in u for u in seen_urls)

        exc = exc_info.value
        assert exc.status_code == 503
        assert exc.__cause__ is None
        assert exc.__suppress_context__ is True
        assert _TEST_KEY not in str(exc)
        assert _TEST_KEY not in repr(exc)
        rendered = "".join(traceback.format_exception(exc))
        assert _TEST_KEY not in rendered, (
            "the rendered traceback (cause and context chains included) "
            "must not contain the FIRMS MAP key"
        )


class TestFIRMSErrorBodyGuard:
    @pytest.mark.parametrize(
        "body",
        [b"Invalid MAP_KEY.", b"Error: exceeded transaction limit for key"],
    )
    def test_http_200_error_bodies_fail_closed(self, tmp_path: Any, body: bytes) -> None:
        loader = _loader(tmp_path)
        with (
            patch(
                "omni_mercury_engine.loaders.base.BaseDomainLoader._fetch_url",
                return_value=body,
            ),
            pytest.raises(ValueError, match="non-CSV body") as exc_info,
        ):
            loader.fetch_realtime()
        assert _TEST_KEY not in str(exc_info.value)

    def test_real_firms_csv_still_parses(self, tmp_path: Any) -> None:
        loader = _loader(tmp_path)
        with patch(
            "omni_mercury_engine.loaders.base.BaseDomainLoader._fetch_url",
            return_value=_FIRMS_CSV,
        ):
            df = loader.fetch_realtime()
        assert len(df) == 2
        assert "frp" in df.columns
        feats = loader.engineer_features(df)
        assert feats.ndim == 2 and feats.shape[0] == 2
        assert np.isfinite(feats).all()

    def test_empty_body_returns_empty_frame(self, tmp_path: Any) -> None:
        loader = _loader(tmp_path)
        with patch(
            "omni_mercury_engine.loaders.base.BaseDomainLoader._fetch_url",
            return_value=b"",
        ):
            df = loader.fetch_realtime()
        assert df.empty

    def test_error_body_echoing_key_is_scrubbed(self, tmp_path: Any) -> None:
        """Observed FIRMS error bodies do not echo the MAP key, but that is
        upstream behaviour, not a contract — if one ever does, the value
        scrub must keep it out of the raised message."""
        loader = _loader(tmp_path)
        with (
            patch(
                "omni_mercury_engine.loaders.base.BaseDomainLoader._fetch_url",
                return_value=f"Bad key {_TEST_KEY} rejected by FIRMS".encode(),
            ),
            pytest.raises(ValueError, match="non-CSV body") as exc_info,
        ):
            loader.fetch_realtime()
        assert _TEST_KEY not in str(exc_info.value)
        assert "<NASA_FIRMS_MAP_KEY:redacted>" in str(exc_info.value)
