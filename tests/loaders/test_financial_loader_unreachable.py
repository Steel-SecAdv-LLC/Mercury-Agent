# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``FinancialLoader`` raises when FRED is unreachable instead of caching nothing as data.

With a FRED key configured and the transport failing, ``_fetch_merged_series``
logged each series' ``ConnectionError`` and returned an empty frame.
``fetch_realtime`` then wrote that empty frame to the cache and returned it,
so the outage was served as "no financial data" for the next hour -- even
after FRED came back -- and ``fetch_historical`` returned an empty frame that
read as an event with no observations.

All tests are offline: the shared ``SafeHTTPClient.get_bytes`` transport is
patched; nothing here touches the network.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError
from omni_mercury_engine.loaders.financial_loader import FinancialLoader

if TYPE_CHECKING:
    from pathlib import Path

_GET_BYTES = "omni_mercury_engine.loaders.base.SafeHTTPClient.get_bytes"
_KEY = "unit-test-fred-key-0123456789"


def _unreachable(*_args: Any, **_kwargs: Any) -> bytes:
    raise ConnectionError("simulated: network unreachable")


def _loader(tmp_path: Path) -> FinancialLoader:
    return FinancialLoader(cache_dir=tmp_path, api_key=_KEY, max_retries=0)


def test_unreachable_fred_raises_and_caches_nothing_realtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was: an empty frame, returned and cached under ``financial_realtime``."""
    monkeypatch.setattr(_GET_BYTES, _unreachable)
    loader = _loader(tmp_path)

    with pytest.raises(DataSourceUnavailableError) as excinfo:
        loader.fetch_realtime()

    message = str(excinfo.value)
    assert excinfo.value.loader_name == "FinancialLoader"
    assert "api.stlouisfed.org" in message
    assert _KEY not in message
    assert loader._read_cache("financial_realtime") is None
    assert list(tmp_path.glob("*.json")) == []


def test_unreachable_fred_raises_for_a_historical_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was: an empty frame, indistinguishable from an event with no observations."""
    monkeypatch.setattr(_GET_BYTES, _unreachable)
    loader = _loader(tmp_path)

    with pytest.raises(DataSourceUnavailableError, match="FRED"):
        loader.fetch_historical("gfc_2008")

    assert list(tmp_path.glob("*.json")) == []


def test_reachable_fred_with_one_failed_series_still_returns_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single failed series stays a logged gap: only a total failure is an outage."""

    def _vix_only(_url: str, *, params: dict[str, str], **_kwargs: Any) -> bytes:
        if params["series_id"] != "VIXCLS":
            raise ConnectionError("simulated: series endpoint down")
        observations = [
            {"date": "2008-09-15", "value": "31.70"},
            {"date": "2008-09-16", "value": "30.30"},
        ]
        return json.dumps({"observations": observations}).encode()

    monkeypatch.setattr(_GET_BYTES, _vix_only)

    frame = _loader(tmp_path).fetch_historical("gfc_2008")

    assert len(frame) == 2
    assert frame["vix"].tolist() == [31.70, 30.30]
