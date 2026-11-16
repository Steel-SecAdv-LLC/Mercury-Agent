# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The tsunami loader returns what the DART buoys recorded.

Two defects, both measured against NDBC on 2026-11-16.

``fetch_historical`` never made a request. Its docstring said NDBC files
rotate out after about 45 days, and it built a sine wave with a decaying
oscillation in the event window for every event, unmarked. NDBC keeps a
yearly archive of every DART station (``data/historical/dart/``), and the
archive holds each catalogued event: 405 of the 409 quarter hours around
Tohoku 2011 at station 21418.

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
    # One row per quarter hour, the buoy's standard cadence, across the event
    # window and two days of baseline either side. A finer grid would record
    # when the buoy switched to event mode, which is the answer, not a reading.
    assert frame["timestamp"].diff().dropna().eq(pd.Timedelta(minutes=15)).all()
    assert frame["timestamp"].iloc[0] == pd.Timestamp("2011-03-09T06:00:00Z")
    assert frame["timestamp"].iloc[-1] == pd.Timestamp("2011-03-13T12:00:00Z")
    # The recorded depth at 21418: about 5,662.6 m of water, a tide of under a
    # metre, and nothing interpolated across more than a few steps.
    assert 5662.0 < frame["bpr"].min() < 5662.3
    assert 5663.0 < frame["bpr"].max() < 5663.1
    assert not frame["bpr"].isna().any()
    assert int(frame["measured"].sum()) == 405
    assert len(frame) == 409


def test_whether_a_step_was_measured_does_not_give_the_window_away(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was: on a 1-minute grid ``measured`` alone ranked the window at AUROC 0.60-0.67.

    The buoy reports every 15 minutes until it detects an event and every
    minute or faster afterwards, so at one minute "a sample exists" is the
    buoy's own detection, handed to the detector as a feature.
    """
    monkeypatch.setattr(_GET_BYTES, _serve(gzip.compress(_ARCHIVE.read_bytes())))
    loader = _loader(tmp_path)

    frame = loader.fetch_historical("tohoku_2011")
    labels = loader.get_ground_truth("tohoku_2011")

    inside = frame["measured"].to_numpy()[labels == 1]
    outside = frame["measured"].to_numpy()[labels == 0]
    assert inside.all()
    assert outside.mean() > 0.98


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
    # Step-to-step movement is several times larger inside the window than
    # the tide moves the column outside it (measured: 4.0x at the median,
    # 5.0x at the 90th percentile).
    for quantile in (50, 90):
        inside = np.percentile(step[labels == 1], quantile)
        outside = np.percentile(step[labels == 0], quantile)
        assert inside > 3.5 * outside, (quantile, inside, outside)
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


# ---------------------------------------------------------------------------
# Robustness: found by an independent review of the loader.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cut", [20, 2000], ids=["header-cut", "body-cut"])
def test_a_download_cut_short_is_refused_not_raised_bare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cut: int
) -> None:
    """Was: ``EOFError`` / ``zlib.error`` escaped ``fetch_historical`` as a traceback."""
    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
    whole = gzip.compress(_ARCHIVE.read_bytes())
    monkeypatch.setattr(_GET_BYTES, _serve(whole[:cut]))

    with pytest.raises(DataSourceUnavailableError, match="not a complete gzip file"):
        _loader(tmp_path).fetch_historical("tohoku_2011")
    assert list(tmp_path.glob("*.json")) == []


def test_a_span_that_is_mostly_gap_is_not_drawn_in_and_called_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Thirty samples used to be enough for four days of interpolated "record"."""
    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
    lines = _ARCHIVE.read_text(encoding="utf-8").splitlines()
    one_day = [line for line in lines if line.startswith(("#", "2011 03 11"))]
    monkeypatch.setattr(_GET_BYTES, _serve(gzip.compress("\n".join(one_day).encode())))

    with pytest.raises(DataSourceUnavailableError, match="at least 90% are needed"):
        _loader(tmp_path).fetch_historical("tohoku_2011")
    assert list(tmp_path.glob("*.json")) == []


def test_a_file_with_only_its_header_parses_to_no_samples() -> None:
    """Was: ``EmptyDataError`` from pandas."""
    header = "#YY  MM DD hh mm ss T   HEIGHT\n#yr  mo dy hr mn  s -        m\n"

    for text in (header, ""):
        parsed = TsunamiLoader._parse_dart_text(text, "21418")
        assert parsed.empty
        assert list(parsed.columns) == ["timestamp", "bpr", "sample_type", "station_id"]


def test_rows_that_cannot_be_read_are_dropped_and_the_rest_kept() -> None:
    """Was: one malformed row failed the whole file (``ParserError`` / ``ValueError``)."""
    text = (
        "#YY  MM DD hh mm ss T   HEIGHT\n"
        "2011 03 11 05 00 00 1 5662.500\n"
        "2011 03 11 05 15 00 1 5662.600 extra tokens here\n"
        "2011 13 41 05 30 00 1 5662.700\n"
        "2011 03 11 05 45 00 1 MM\n"
        "2011 03 11 06 00\n"
        "2011 03 11 06 15 00 1 0.250\n"
        "2011 03 11 06 30 00 1 5662.800\n"
    )

    parsed = TsunamiLoader._parse_dart_text(text, "21418")

    assert parsed["bpr"].tolist() == [5662.5, 5662.8]
    assert parsed["timestamp"].dt.strftime("%H:%M").tolist() == ["05:00", "06:30"]
