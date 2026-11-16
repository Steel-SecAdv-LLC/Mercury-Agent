# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The tsunami loader returns what the DART buoys recorded.

Two defects, both measured against NDBC on 2026-11-16.

``fetch_historical`` never made a request. Its docstring said NDBC files
rotate out after about 45 days, and it built a sine wave with a decaying
oscillation in the event window for every event, unmarked. NDBC keeps a
yearly archive of every DART station (``data/historical/dart/``), and the
archive holds each catalogued event, event-mode samples included: 857
recorded minutes for Tohoku 2011 at station 21418.

``fetch_realtime`` parsed NDBC's ``YY MM DD hh mm ss T HEIGHT`` layout as if
there were no seconds column. Every field after the minute shifted by one, so
``bpr`` held the measurement-type code: across ten stations and 32,497 live
rows its mean was 1.02 and its maximum 3, where the water column is 3,265 m
to 5,886 m deep.

Offline: the transport is patched to serve files recorded from NDBC
(``tests/fixtures/dart``).
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError
from omni_mercury_engine.loaders.base import FetchHTTPError, SyntheticDataNotDeleakable
from omni_mercury_engine.loaders.tsunami_loader import TsunamiLoader

if TYPE_CHECKING:
    from collections.abc import Callable

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "dart"
_ARCHIVE = _FIXTURES / "21418t2011_tohoku.txt"
_REALTIME = _FIXTURES / "46402_realtime_head.dart"
_GET_BYTES = "omni_mercury_engine.loaders.base.SafeHTTPClient.get_bytes"


def _serve(payload: bytes, seen: list[str] | None = None) -> Callable[..., bytes]:
    def _transport(url: str, **_kwargs: Any) -> bytes:
        if seen is not None:
            seen.append(url)
        return payload

    return _transport


def _unreachable(*_args: Any, **_kwargs: Any) -> bytes:
    raise ConnectionError("simulated: network unreachable")


def _loader(tmp_path: Path) -> TsunamiLoader:
    return TsunamiLoader(cache_dir=tmp_path, max_retries=0)


def test_a_historical_event_is_the_recorded_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was: a synthesised sine wave around 5000.0, with no request made."""
    seen: list[str] = []
    monkeypatch.setattr(_GET_BYTES, _serve(gzip.compress(_ARCHIVE.read_bytes()), seen))
    loader = _loader(tmp_path)

    frame = loader.fetch_historical("tohoku_2011")

    assert seen == ["https://www.ndbc.noaa.gov/data/historical/dart/21418t2011.txt.gz"]
    assert not loader.is_synthetic(frame)
    assert (frame["bpr_series_source"] == "ndbc_dart_archive").all()
    assert (frame["station_id"] == "21418").all()
    # One row per minute across the event window and its baseline.
    assert frame["timestamp"].diff().dropna().eq(pd.Timedelta(minutes=1)).all()
    assert frame["timestamp"].iloc[0] == pd.Timestamp("2011-03-10T23:46:00Z")
    assert frame["timestamp"].iloc[-1] == pd.Timestamp("2011-03-11T18:00:00Z")
    # The recorded depth at 21418 and the recorded wave: about 5,662.8 m of
    # water, and a swing of nearly three metres as the tsunami passed.
    assert 5661.0 < frame["bpr"].min() < 5661.5
    assert 5664.0 < frame["bpr"].max() < 5664.5
    assert not frame["bpr"].isna().any()
    # Most minutes hold a recorded sample; the rest are interpolated and say so.
    assert 800 < int(frame["measured"].sum()) < len(frame)
    quiet = frame[frame["timestamp"] < pd.Timestamp("2011-03-11T05:00:00Z")]
    assert quiet["bpr"].max() - quiet["bpr"].min() < 1.0


def test_the_recorded_wave_is_inside_the_labelled_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The labels are a catalogued window; the recorded series has to agree with it."""
    monkeypatch.setattr(_GET_BYTES, _serve(gzip.compress(_ARCHIVE.read_bytes())))
    loader = _loader(tmp_path)

    frame = loader.fetch_historical("tohoku_2011")
    labels = loader.get_ground_truth("tohoku_2011")

    assert len(labels) == len(frame)
    step = np.abs(np.diff(frame["bpr"].to_numpy(), prepend=frame["bpr"].iloc[0]))
    assert labels[int(step.argmax())] == 1
    # Minute-to-minute movement is several times larger inside the window at
    # every quantile. The maximum is not compared: the record has three
    # single-sample dips of 0.6 m to 1 m after the window (12:07, 14:07 and
    # 15:07 UTC), which are the buoy's, not the loader's, and stay in.
    for quantile in (50, 90):
        inside = np.percentile(step[labels == 1], quantile)
        outside = np.percentile(step[labels == 0], quantile)
        assert inside > 5 * outside, (quantile, inside, outside)
    assert loader.engineer_scored_features(frame).shape == (len(frame), 6)


def test_samples_ndbc_marks_as_not_received_are_not_readings() -> None:
    """``9999.000`` is NDBC's placeholder; kept, it would be a 4 km wave."""
    parsed = TsunamiLoader._parse_dart_text(_ARCHIVE.read_text(encoding="utf-8"), "21418")

    assert "9999.000" in _ARCHIVE.read_text(encoding="utf-8")
    assert parsed["bpr"].max() < 6000.0
    assert parsed["timestamp"].is_monotonic_increasing
    assert set(parsed["sample_type"]) == {1, 2, 3}
    # 15-second samples keep their own timestamps.
    assert parsed["timestamp"].dt.second.isin([15, 30, 45]).any()


def test_an_unreachable_archive_is_refused_without_the_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
    monkeypatch.setattr(_GET_BYTES, _unreachable)
    loader = _loader(tmp_path)

    with pytest.raises(DataSourceUnavailableError, match="NDBC DART archive"):
        loader.fetch_historical("tohoku_2011")

    assert list(tmp_path.glob("*.json")) == []


def test_the_opted_in_stand_in_is_marked_and_never_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was: the only path, unmarked, so it was scored as a recorded event."""
    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "1")
    monkeypatch.setattr(_GET_BYTES, _unreachable)
    loader = _loader(tmp_path)

    frame = loader.fetch_historical("tohoku_2011")

    assert loader.is_synthetic(frame)
    assert (frame["bpr_series_source"] == "synthetic").all()
    assert not frame["measured"].any()
    assert list(tmp_path.glob("*.json")) == []
    with pytest.raises(SyntheticDataNotDeleakable):
        loader.engineer_scored_features(frame)


def test_a_live_reading_is_the_water_column_height_not_the_type_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was: ``bpr`` in {1, 2, 3} on every live row."""
    monkeypatch.setattr(_GET_BYTES, _serve(_REALTIME.read_bytes()))

    frame = _loader(tmp_path).fetch_realtime()

    assert frame["bpr"].min() > 4000.0
    assert set(frame["sample_type"]) <= {1, 2, 3}
    one_station = frame[frame["station_id"] == "46402"]
    assert one_station["timestamp"].is_monotonic_increasing


def test_features_do_not_difference_across_two_buoys(tmp_path: Path) -> None:
    """Was: the seam between stations at different depths read as a kilometre-high step."""
    times = pd.date_range("2026-11-15", periods=80, freq="15min", tz="UTC")
    shallow = pd.DataFrame({"timestamp": times, "bpr": 3265.0, "station_id": "46410"})
    deep = pd.DataFrame({"timestamp": times, "bpr": 5886.0, "station_id": "52402"})
    loader = _loader(tmp_path)

    features = loader.engineer_features(pd.concat([shallow, deep], ignore_index=True))

    columns = TsunamiLoader.FEATURE_COLUMNS
    assert features.shape == (160, len(columns))
    assert features[:80, columns.index("bpr")].tolist() == [3265.0] * 80
    for name in ("tidal_deviation", "rate_of_change", "rolling_std", "short_energy"):
        assert np.abs(features[:, columns.index(name)]).max() == 0.0


def test_a_missing_file_is_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Was: a retired station's 404 was retried through the whole backoff."""

    class _NotFound(OSError):
        response = type("R", (), {"status_code": 404})()

    calls: list[str] = []

    def _transport(url: str, **_kwargs: Any) -> bytes:
        calls.append(url)
        raise _NotFound("404")

    monkeypatch.setattr(_GET_BYTES, _transport)
    loader = TsunamiLoader(cache_dir=tmp_path, max_retries=3, retry_backoff=0.0)

    with pytest.raises(FetchHTTPError) as caught:
        loader._fetch_url("https://www.ndbc.noaa.gov/data/realtime2/32412.dart")

    assert caught.value.status_code == 404
    assert len(calls) == 1
