# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""OfflineModeError / DataSourceUnavailableError never carry a credential.

Both classes render operator-facing messages from URLs that keyed
loaders compose their credentials into (query params for NASA / EIA /
AirNow / Alpha Vantage; a path segment for FIRMS), and
``DataSourceUnavailableError.reason`` is routinely built from transport
exception text that embeds the fully-composed request URL.  Redaction
happens in the constructors — the single funnel — so no raise site can
forget it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omni_mercury_engine.datasets.exceptions import (
    DataSourceUnavailableError,
    OfflineModeError,
)

if TYPE_CHECKING:
    import pytest


class TestOfflineModeErrorRedaction:
    def test_query_credential_never_in_message_or_attribute(self) -> None:
        exc = OfflineModeError("https://api.nasa.gov/neo/rest/v1/feed?api_key=TOPSECRET9&start=x")
        assert "TOPSECRET9" not in str(exc)
        assert "TOPSECRET9" not in exc.url

    def test_path_and_remediation_survive(self) -> None:
        """The message must keep naming the refused artifact and the fix —
        the contract test_error_carries_url_and_remediation pins from the
        consumer side."""
        exc = OfflineModeError("https://h.example/some.npz?token=TKN99")
        assert "some.npz" in str(exc)
        assert "prefetch_datasets" in str(exc)
        assert "TKN99" not in str(exc)

    def test_path_segment_key_scrubbed_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FIRMS embeds the MAP key as a PATH segment — invisible to
        structural redaction, caught by the env-value pass."""
        monkeypatch.setenv("NASA_FIRMS_MAP_KEY", "FIRMSKEY42")
        exc = OfflineModeError(
            "https://firms.modaps.eosdis.nasa.gov/api/area/csv/FIRMSKEY42/VIIRS_SNPP_NRT/1"
        )
        assert "FIRMSKEY42" not in str(exc)
        assert "firms.modaps.eosdis.nasa.gov" in str(exc)


class TestDataSourceUnavailableErrorRedaction:
    def test_source_url_credential_redacted(self) -> None:
        exc = DataSourceUnavailableError(
            "sepsis",
            source_url="https://physionet.example/data?token=TKN12345",
            reason="HTTP 403",
        )
        assert "TKN12345" not in str(exc)
        assert "TKN12345" not in exc.source_url
        assert "physionet.example/data" in str(exc)

    def test_rewrapped_transport_text_in_reason_redacted(self) -> None:
        """Loaders re-wrap transport errors whose str embeds the composed
        URL (httpx: \"... for url '...'\"); the reason funnel must scrub it."""
        exc = DataSourceUnavailableError(
            "volcanic",
            reason=(
                "Client error '401 Unauthorized' for url "
                "'https://api.example/v1/events?api_key=VOLCKEY77&window=7d'"
            ),
        )
        assert "VOLCKEY77" not in str(exc)
        assert "VOLCKEY77" not in exc.reason
        assert "401" in str(exc)

    def test_loader_name_status_and_plain_reason_survive(self) -> None:
        exc = DataSourceUnavailableError(
            "network_security",
            source_url="https://mirror.example/cicids/2017.csv",
            reason="mirror returned truncated archive",
            status_code=502,
        )
        message = str(exc)
        assert "network_security" in message
        assert "mirror.example/cicids/2017.csv" in message
        assert "truncated archive" in message
        assert "HTTP 502" in message
        assert exc.status_code == 502
