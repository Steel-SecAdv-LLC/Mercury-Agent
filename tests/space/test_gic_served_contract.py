# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What ``GICDetector`` and its USGS source serve, on real Gannon-storm data.

Each test fails on the code as it was before: 1-second data were differenced
at 1 s (FRD read 765 nT/min "severe" where 1-minute data read ~55), a field
in tesla read "low", the circular FFT put a late step's field at sample 0, a
5-minute series was scored as if it were 1-minute, and the USGS source called
a 404 endpoint and returned an empty list labelled "live".
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.core.exceptions import InvalidObservationError
from omni_mercury_engine.data_sources.base import DataSourceError
from omni_mercury_engine.data_sources.geomagnetic import USGSGeomagnetismSource
from omni_mercury_engine.space.gic_detector import GICDetector

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "space_weather"


def _load(name: str) -> tuple[list[datetime], np.ndarray, np.ndarray]:
    data = json.loads((FIXTURES / name).read_text())["data"]
    times = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in data["times"]]
    series = {v["id"]: np.asarray(v["values"], dtype=float) for v in data["values"]}
    return times, series["X"], series["Y"]


def test_one_second_data_is_assessed_at_one_minute() -> None:
    times, bx, by = _load("usgs_geomag_frd_1s_2024_05_10.json")
    raw_peak = float(np.max(np.hypot(np.diff(bx), np.diff(by)))) * 60.0
    assert raw_peak > 700.0  # what 1-second differencing reads
    result = GICDetector().assess(times, bx, by, observatory="FRD")
    # 21 wall-clock minutes are touched, but the 17:20 bin holds a single
    # second of data -- a spot sample, not a 1-minute mean -- so it is
    # dropped rather than differenced against the 17:19 mean.
    assert result.n_samples == 20
    assert 30.0 < result.peak_dbdt_nt_per_min < 90.0
    assert result.risk_level == "low"
    assert any("1-minute means" in note for note in result.notes)
    assert any("partial edge minute" in note for note in result.notes)


def test_partial_edge_minutes_are_dropped_not_scored_as_means() -> None:
    """A partial first/last wall-clock minute must not enter the dB/dt steps.

    Regression: bins were labelled at their wall-clock minute start and a
    series starting at :30 put a half-minute mean in the first bin, so the
    first dB/dt step of a steady 60 nT/min ramp read ~45 nT/min (the two
    "minute" centres sit 45 s apart but the step divides by 60 s).
    """
    start = datetime(2024, 5, 10, 17, 0, 30, tzinfo=UTC)  # mid-minute start
    n = 60 * 4 + 40  # ends mid-minute too (17:04:50)
    times = [start + timedelta(seconds=i) for i in range(n)]
    ramp = np.arange(n, dtype=np.float64)  # 1 nT/s = 60 nT/min, exactly
    result = GICDetector().assess(times, ramp, np.zeros(n), observatory="FRD")
    # Only the full minutes 17:01-17:04 remain; every step reads the true rate.
    assert result.n_samples == 4
    assert result.peak_dbdt_nt_per_min == pytest.approx(60.0)
    assert result.sustained_dbdt_nt_per_min is None or result.sustained_dbdt_nt_per_min <= 60.0
    assert any("partial edge minute" in note for note in result.notes)


def test_gaps_are_counted_on_the_input_samples() -> None:
    times, bx, by = _load("usgs_geomag_frd_1s_2024_05_10.json")
    bx[100:400:3] = np.nan  # 100 X samples, scattered over minutes 1-6
    by[600:660] = np.nan  # 60 Y samples: all of 17:10
    result = GICDetector().assess(times, bx, by, observatory="FRD")
    assert result.n_gaps == 160
    assert "160 non-finite input samples skipped in the 1-minute means." in result.notes
    assert any(note.startswith("1 minute bins") for note in result.notes)


def test_minute_data_pass_through_unchanged() -> None:
    times, bx, by = _load("usgs_geomag_bou_2024_05_10.json")
    result = GICDetector().assess(times, bx, by, observatory="BOU")
    expected = float(np.nanmax(np.hypot(np.diff(bx), np.diff(by))))
    assert result.peak_dbdt_nt_per_min == pytest.approx(expected)
    assert result.risk_level == "moderate"  # ~254 nT/min, below the 300 nT/min tier


def test_field_in_tesla_is_refused() -> None:
    times, bx, by = _load("usgs_geomag_bou_2024_05_10.json")
    with pytest.raises(InvalidObservationError, match="nT"):
        GICDetector().assess(times, bx * 1e-9, by * 1e-9)


def test_coarser_than_minute_cadence_is_refused() -> None:
    times, bx, by = _load("usgs_geomag_bou_2024_05_10.json")
    with pytest.raises(InvalidObservationError, match="cadence"):
        GICDetector().assess(times[::5], bx[::5], by[::5])


def test_geoelectric_field_is_causal() -> None:
    times, bx, by = _load("usgs_geomag_bou_2024_05_10.json")
    det = GICDetector()
    stepped = bx.copy()
    stepped[-5:] += 500.0  # a step in the last five minutes
    ex0, ey0 = det.geoelectric_plane_wave_v_per_km(bx, by, 60.0)
    ex1, ey1 = det.geoelectric_plane_wave_v_per_km(stepped, by, 60.0)
    early = slice(0, len(bx) - 5)
    np.testing.assert_allclose(ex1[early], ex0[early], atol=1e-9)
    np.testing.assert_allclose(ey1[early], ey0[early], atol=1e-9)
    assert float(np.max(np.abs(ey1[-5:] - ey0[-5:]))) > 0.1


def _source_with(
    payload: Any, *, fail: bool = False, sampling: str = "minute"
) -> tuple[USGSGeomagnetismSource, list[tuple[str, dict[str, Any] | None]]]:
    calls: list[tuple[str, dict[str, Any] | None]] = []

    class _Response:
        def json(self) -> Any:
            return payload

    async def _get(endpoint: str, params: dict[str, Any] | None = None) -> _Response:
        calls.append((endpoint, params))
        if fail:
            raise DataSourceError("HTTP 503", source_id="usgs_geomagnetism")
        return _Response()

    source = USGSGeomagnetismSource(sampling=sampling)
    source._http_get = _get  # type: ignore[assignment, method-assign, unused-ignore]
    return source, calls


def test_source_requests_the_live_endpoint_with_vector_components() -> None:
    payload = json.loads((FIXTURES / "usgs_geomag_bou_2024_05_10.json").read_text())["data"]
    source, calls = _source_with(payload)
    end = datetime(2024, 5, 10, 18, tzinfo=UTC)
    points = asyncio.run(source._fetch_impl(start_time=end - timedelta(hours=1), end_time=end))
    endpoint, params = calls[0]
    assert endpoint == "data/"
    assert params is not None and params["sampling_period"] == "60"
    assert params["elements"].split(",")[:2] == ["X", "Y"]
    assert points and {"X", "Y"} <= set(points[0].data["elements"])


def test_source_returns_the_whole_requested_window() -> None:
    payload = json.loads((FIXTURES / "usgs_geomag_frd_1s_2024_05_10.json").read_text())["data"]
    source, calls = _source_with(payload, sampling="second")
    start = datetime(2024, 5, 10, 17, tzinfo=UTC)
    points = asyncio.run(
        source._fetch_impl(start_time=start, end_time=start + timedelta(minutes=20))
    )
    assert calls[0][1] is not None and calls[0][1]["sampling_period"] == "1"
    assert len(points) == len(payload["times"]) == 1201
    assert (points[0].timestamp, points[-1].timestamp) == (
        start,
        start + timedelta(minutes=20),
    )
    (result,) = GICDetector().assess_from_datapoints(points).values()
    # Every full minute of the window (not just the last 100 s); the 21st
    # mark is a 1-second partial bin and is dropped, not scored as a mean.
    assert result.n_samples == 20


def test_source_failure_is_not_an_empty_live_feed() -> None:
    source, _calls = _source_with(None, fail=True)
    end = datetime(2024, 5, 10, 18, tzinfo=UTC)
    with pytest.raises(DataSourceError):
        asyncio.run(source._fetch_impl(start_time=end - timedelta(hours=1), end_time=end))
