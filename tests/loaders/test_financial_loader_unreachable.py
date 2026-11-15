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
from omni_mercury_engine.loaders.financial_loader import (
    INCOMPLETE_SERIES_ATTR,
    FinancialLoader,
)

if TYPE_CHECKING:
    from pathlib import Path

_GET_BYTES = "omni_mercury_engine.loaders.base.SafeHTTPClient.get_bytes"
_KEY = "unit-test-fred-key-0123456789"

_ALL_SERIES = ("VIXCLS", "T10Y2Y", "BAMLH0A0HYM2", "DFF", "SOFR")


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


def _serve_only(*served: str) -> Any:
    """A transport on which only ``served`` FRED series answer; the rest fail."""

    def _transport(_url: str, *, params: dict[str, str], **_kwargs: Any) -> bytes:
        if params["series_id"] not in served:
            raise ConnectionError("simulated: series endpoint down")
        observations = [
            {"date": "2008-09-15", "value": "31.70"},
            {"date": "2008-09-16", "value": "30.30"},
        ]
        return json.dumps({"observations": observations}).encode()

    return _transport


def test_a_partial_outage_is_returned_but_never_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was: the frame with three all-NaN columns was cached and served for an hour."""
    monkeypatch.setattr(_GET_BYTES, _serve_only("VIXCLS", "T10Y2Y"))
    loader = _loader(tmp_path)

    frame = loader.fetch_historical("gfc_2008")

    assert len(frame) == 2
    assert frame.attrs[INCOMPLETE_SERIES_ATTR] == [
        "high_yield_spread",
        "fed_funds_rate",
        "funding_rate",
    ]
    assert list(tmp_path.glob("*.json")) == []

    # Once FRED answers again the next call fetches the full record and caches it.
    monkeypatch.setattr(_GET_BYTES, _serve_only(*_ALL_SERIES))
    recovered = loader.fetch_historical("gfc_2008")
    assert INCOMPLETE_SERIES_ATTR not in recovered.attrs
    assert not recovered["high_yield_spread"].isna().any()
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_a_partial_realtime_outage_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The realtime path wrote the same partial frame under its own key."""
    monkeypatch.setattr(_GET_BYTES, _serve_only("VIXCLS"))

    frame = _loader(tmp_path).fetch_realtime()

    assert len(frame) == 2
    assert "yield_curve_10y2y" in frame.attrs[INCOMPLETE_SERIES_ATTR]
    assert list(tmp_path.glob("*.json")) == []


@pytest.mark.parametrize("served", [("DFF",), ("VIXCLS",), ("T10Y2Y", "DFF")])
def test_labels_are_refused_when_a_label_series_was_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, served: tuple[str, ...]
) -> None:
    """Was: ``[0, 0]``. The 2008 crisis was labelled normal because of a network fault."""
    monkeypatch.setattr(_GET_BYTES, _serve_only(*served))

    with pytest.raises(DataSourceUnavailableError, match="cannot label event 'gfc_2008'"):
        _loader(tmp_path).get_ground_truth("gfc_2008")


def test_labels_survive_the_loss_of_a_series_they_do_not_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Losing the credit spread does not stop a label that never reads it."""

    def _inverted(_url: str, *, params: dict[str, str], **_kwargs: Any) -> bytes:
        values = {"VIXCLS": ["46.00", "20.00"], "T10Y2Y": ["-0.50", "0.40"]}
        if params["series_id"] not in values:
            raise ConnectionError("simulated: series endpoint down")
        dates = ["2008-09-15", "2008-09-16"]
        observations = [{"date": d, "value": v} for d, v in zip(dates, values[params["series_id"]])]
        return json.dumps({"observations": observations}).encode()

    monkeypatch.setattr(_GET_BYTES, _inverted)

    assert _loader(tmp_path).get_ground_truth("gfc_2008").tolist() == [1, 0]


def test_a_complete_frame_is_cached_and_served_without_a_second_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was: the cache write failed on the Timestamp column, so nothing was ever cached."""
    monkeypatch.setattr(_GET_BYTES, _serve_only(*_ALL_SERIES))
    loader = _loader(tmp_path)
    first = loader.fetch_historical("gfc_2008")
    assert len(list(tmp_path.glob("*.json"))) == 1

    monkeypatch.setattr(_GET_BYTES, _unreachable)
    second = loader.fetch_historical("gfc_2008")

    assert second["date"].tolist() == first["date"].tolist()
    assert second["vix"].tolist() == first["vix"].tolist()
    assert str(second["date"].dtype).startswith("datetime64")
