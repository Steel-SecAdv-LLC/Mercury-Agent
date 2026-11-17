# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Loaders whose frames hold timestamps really cache them.

``_write_cache(frame.to_dict(orient="list"))`` cannot serialise a pandas
``Timestamp``; the writer drops such a payload, and it logged that at DEBUG.
Measured before this file existed: ``TsunamiLoader.fetch_historical`` and all
three ``FloodLoader`` cache writes failed on every call, so the tsunami series
was produced again and USGS Water Services was queried again each time. Both loaders
now go through the frame cache, which round-trips datetimes.

Measured later, by reading every loader against its live source with an
empty cache directory: ``EnergyLoader.fetch_realtime`` had the same fault
(three SWPC requests on every call), and ``PandemicLoader`` kept its 60 MB
download as Parquet, which needs an engine this package never installs, so
that file was never kept either.

Offline: the transports are patched; the tsunami archive is a recorded NDBC
file and the SWPC answers are recorded ones.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from omni_mercury_engine.loaders.energy_loader import EnergyLoader
from omni_mercury_engine.loaders.flood_loader import FloodLoader
from omni_mercury_engine.loaders.network_security_loader import NetworkSecurityLoader
from omni_mercury_engine.loaders.pandemic_loader import PandemicLoader
from omni_mercury_engine.loaders.tsunami_loader import TsunamiLoader

_GET_BYTES = "omni_mercury_engine.loaders.base.SafeHTTPClient.get_bytes"
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_DART_FIXTURE = _FIXTURES / "dart" / "21418t2011_tohoku.txt"
_SWPC = {
    "noaa-planetary-k-index.json": "swpc_planetary_k_index_2026_11_17.json",
    "rtsw_wind_1m.json": "swpc_rtsw_wind_1m_2026_11_17_head.json",
    "ace_swepam_1h.json": "swpc_ace_swepam_1h_2026_11_17_head.json",
}


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


def _swpc(url: str, **_kwargs: Any) -> bytes:
    """The recorded SWPC answer for whichever of the three feeds is asked for."""
    name = url.rsplit("/", 1)[-1]
    return (_FIXTURES / "energy" / _SWPC[name]).read_bytes()


def test_energy_realtime_is_cached_and_served_without_a_second_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was: "Cache write failed for energy_realtime: Object of type Timestamp ..."."""
    monkeypatch.setattr(_GET_BYTES, _swpc)
    loader = EnergyLoader(cache_dir=tmp_path, max_retries=0)

    first = loader.fetch_realtime()
    assert len(first) == 62  # one row for each three-hour Kp value
    assert first["solar_wind_speed"].notna().any()
    assert len(list(tmp_path.glob("*.json"))) == 1

    monkeypatch.setattr(_GET_BYTES, _refuse)
    second = loader.fetch_realtime()

    assert str(second["timestamp"].dtype) == str(first["timestamp"].dtype)
    assert second["timestamp"].tolist() == first["timestamp"].tolist()
    pd.testing.assert_frame_equal(second, first)


def _owid_csv() -> bytes:
    """Three rows in the layout of the OWID COVID-19 file."""
    return (
        b"iso_code,continent,location,date,new_cases,new_deaths\n"
        b"USA,North America,United States,2020-03-01,7.0,1.0\n"
        b"USA,North America,United States,2020-03-02,23.0,5.0\n"
        b"OWID_WRL,,World,2020-03-02,2000.0,70.0\n"
    )


def test_the_owid_download_is_kept_and_read_back_without_a_second_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was: nothing kept (no Parquet engine), so 60 MB came down on every call."""
    payload = _owid_csv()
    monkeypatch.setattr(_GET_BYTES, lambda *_a, **_k: payload)
    loader = PandemicLoader(cache_dir=tmp_path, max_retries=0)

    first = loader._load_owid_data()
    kept = tmp_path / "owid_covid_data.csv"
    assert kept.read_bytes() == payload
    assert [p.name for p in tmp_path.iterdir()] == [kept.name]  # no part file left

    monkeypatch.setattr(_GET_BYTES, _refuse)
    second = loader._load_owid_data()

    pd.testing.assert_frame_equal(second, first)


def test_a_stale_or_unreadable_owid_copy_is_fetched_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _owid_csv()
    asked: list[str] = []

    def serve(url: str, **_kwargs: Any) -> bytes:
        asked.append(url)
        return payload

    monkeypatch.setattr(_GET_BYTES, serve)
    loader = PandemicLoader(cache_dir=tmp_path, max_retries=0)
    kept = tmp_path / "owid_covid_data.csv"

    loader._load_owid_data()
    stale = kept.stat().st_mtime - loader.CACHE_TTL - 1
    os.utime(kept, (stale, stale))
    loader._load_owid_data()
    assert len(asked) == 2

    kept.write_bytes(b"")  # a copy that cannot be parsed
    frame = loader._load_owid_data()
    assert len(asked) == 3
    assert len(frame) == 3
    assert kept.read_bytes() == payload


def test_a_download_that_does_not_parse_is_not_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_GET_BYTES, lambda *_a, **_k: b"")
    loader = PandemicLoader(cache_dir=tmp_path, max_retries=0)

    with pytest.raises(ValueError, match="No columns"):
        loader._load_owid_data()
    assert list(tmp_path.iterdir()) == []


def test_network_features_are_the_same_from_a_cached_frame_as_from_a_fetched_one(
    tmp_path: Path,
) -> None:
    """Was: up to 9.2e-7 apart on NSL-KDD -- float32 when fetched, float64 from the cache.

    The same rows must score the same whether or not the cache is warm.
    """
    loader = NetworkSecurityLoader(cache_dir=tmp_path, max_retries=0)
    rng = np.random.default_rng(7)
    columns = [*loader.FEATURE_COLUMNS, *loader._LOG_TRANSFORM_COLUMNS]
    fetched = pd.DataFrame(
        {name: rng.uniform(0.0, 5.0e4, 64).astype(np.float32) for name in dict.fromkeys(columns)}
    )
    assert {str(t) for t in fetched.dtypes} == {"float32"}

    loader._write_cache("frame", fetched.to_dict(orient="list"))
    cached = pd.DataFrame(loader._read_cache("frame"))
    assert {str(t) for t in cached.dtypes} == {"float64"}

    np.testing.assert_array_equal(
        loader.engineer_features(fetched), loader.engineer_features(cached)
    )
