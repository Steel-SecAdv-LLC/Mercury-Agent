# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Loaders whose frames hold timestamps really cache them.

``_write_cache(frame.to_dict(orient="list"))`` cannot serialise a pandas
``Timestamp``; the writer drops such a payload, and it logged that at DEBUG.
Measured before this file existed: ``TsunamiLoader.fetch_historical`` and all
three ``FloodLoader`` cache writes failed on every call, so the tsunami series
was produced again and USGS Water Services was queried again each time. Both loaders
now go through the frame cache, which round-trips datetimes.

Offline: both transports are patched; the tsunami archive is a recorded NDBC file.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from omni_mercury_engine.loaders.flood_loader import FloodLoader
from omni_mercury_engine.loaders.tsunami_loader import TsunamiLoader

if TYPE_CHECKING:
    import pytest

_GET_BYTES = "omni_mercury_engine.loaders.base.SafeHTTPClient.get_bytes"
_DART_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "dart" / "21418t2011_tohoku.txt"


def _refuse(*_args: Any, **_kwargs: Any) -> bytes:
    raise AssertionError("the cached frame should have been served without a request")


def test_the_tsunami_series_is_cached_and_read_back_with_its_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = gzip.compress(_DART_FIXTURE.read_bytes())
    monkeypatch.setattr(_GET_BYTES, lambda *_a, **_k: archive)
    loader = TsunamiLoader(cache_dir=tmp_path, max_retries=0)

    first = loader.fetch_historical("tohoku_2011")
    assert len(list(tmp_path.glob("*.json"))) == 1

    monkeypatch.setattr(_GET_BYTES, _refuse)
    second = loader.fetch_historical("tohoku_2011")
    assert str(second["timestamp"].dtype) == str(first["timestamp"].dtype)
    assert second["timestamp"].tolist() == first["timestamp"].tolist()
    assert second["bpr"].tolist() == first["bpr"].tolist()
    assert second["measured"].tolist() == first["measured"].tolist()
    assert (second["bpr_series_source"] == "ndbc_dart_archive").all()
    assert loader.get_ground_truth("tohoku_2011").sum() > 0


def _usgs_instantaneous_values() -> bytes:
    """Two gauge-height readings for one site, in the USGS IV JSON layout."""
    series = {
        "sourceInfo": {"siteName": "TEST RIVER", "siteCode": [{"value": "08074000"}]},
        "variable": {"variableCode": [{"value": "00065"}]},
        "values": [
            {
                "value": [
                    {"value": "12.5", "dateTime": "2017-08-27T00:00:00.000-05:00"},
                    {"value": "30.1", "dateTime": "2017-08-27T01:00:00.000-05:00"},
                ]
            }
        ],
    }
    return json.dumps({"value": {"timeSeries": [series]}}).encode()


def test_flood_realtime_is_cached_and_served_without_a_second_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _usgs_instantaneous_values()
    monkeypatch.setattr(_GET_BYTES, lambda *_a, **_k: payload)
    loader = FloodLoader(cache_dir=tmp_path, max_retries=0)

    first = loader.fetch_realtime()
    assert len(first) == 2
    assert pd.api.types.is_datetime64_any_dtype(first["datetime"])
    assert len(list(tmp_path.glob("*.json"))) == 1

    monkeypatch.setattr(_GET_BYTES, _refuse)
    second = loader.fetch_realtime()

    assert pd.api.types.is_datetime64_any_dtype(second["datetime"])
    assert second["datetime"].tolist() == first["datetime"].tolist()
    assert second["gauge_height_ft"].tolist() == [12.5, 30.1]
