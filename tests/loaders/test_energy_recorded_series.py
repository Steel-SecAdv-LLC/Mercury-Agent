# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The energy loader's historical events are the recorded series.

Before this, no historical event was fetched. The three geomagnetic storms
were a synthetic Kp profile, on the stated ground that they predate the SWPC
real-time feeds; NASA's OMNI2 archive holds hourly Kp and solar wind for
every year since 1963, each storm included. The grid event (Texas, February
2021) needed ``EIA_API_KEY`` and, with one, queried the EIA API without a
region filter, set ``kp`` to a constant, the solar wind to 400 km/s and
5 N/cm^3, the X-ray class to 0, ``grid_supply`` to 95% of demand and a
missing reading to 0 MW, then labelled ``demand > supply``: every row. EIA
publishes the same grid operator's hours in keyless EIA-930 files.

Measured against the live archives on 2026-11-16: Kp reaches 9.0 in all
three storms, and ERCOT's served demand on 15 to 18 February 2021 sits 20%
to 40% below its own day-ahead forecast (72 to 75 GW forecast, 44 to 51 GW
served), the only such hours in the three-month window.

Offline: the transport is patched to serve files recorded from NASA and EIA
(``tests/fixtures/energy``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError
from omni_mercury_engine.loaders.energy_loader import (
    _EVENT_CATALOG,
    DOE_DISTURBANCE_COLUMN,
    GRID_FEATURE_COLUMNS,
    GRID_SHORTFALL_THRESHOLD,
    EnergyLoader,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "energy"
_GET_BYTES = "omni_mercury_engine.loaders.base.SafeHTTPClient.get_bytes"


class _NotFound(Exception):
    """What the transport raises for HTTP 404 (``requests.HTTPError`` in shape)."""

    response = type("Response", (), {"status_code": 404})()


def _archives(seen: list[str] | None = None) -> Callable[..., bytes]:
    """A transport serving the recorded files by the URL the loader asks for."""

    def _transport(url: str, **_kwargs: Any) -> bytes:
        if seen is not None:
            seen.append(url)
        if url.endswith("omni2_1989.dat"):
            return (_FIXTURES / "omni2_1989_quebec.dat").read_bytes()
        if url.endswith("goes-xrs-report_1989.txt"):
            return (_FIXTURES / "goes_xrs_1989_quebec.txt").read_bytes()
        if "goes-xrs-report_" in url:
            # The flare archive ends in 2017: later years answer 404.
            raise _NotFound("404")
        if url.endswith("omni2_2021.dat"):
            return (_FIXTURES / "omni2_2021_uri.dat").read_bytes()
        if url.endswith("EIA930_BALANCE_2021_Jan_Jun.csv"):
            return (_FIXTURES / "eia930_balance_2021_uri.csv").read_bytes()
        if url.endswith("omni2_2023.dat"):
            return (_FIXTURES / "omni2_2023_ice_storm.dat").read_bytes()
        if url.endswith("EIA930_BALANCE_2023_Jan_Jun.csv"):
            return (_FIXTURES / "eia930_balance_2023_ice_storm.csv").read_bytes()
        if url.endswith("EIA930_BALANCE_2023_Jul_Dec.csv"):
            # The second half of the year: the header and no ERCOT rows.
            header = (_FIXTURES / "eia930_balance_2023_ice_storm.csv").read_bytes().split(b"\n")[0]
            return header + b"\n"
        if "oe-417-annual-summaries" in url:
            document = json.loads((_FIXTURES / "oe417_2023.json").read_text(encoding="utf-8"))
            offset = int((_kwargs.get("params") or {}).get("offset", 0))
            limit = int((_kwargs.get("params") or {}).get("limit", 100))
            page = {
                "total_count": document["total_count"],
                "results": document["results"][offset : offset + limit],
            }
            return json.dumps(page).encode()
        raise ConnectionError(f"no recorded file for {url}")

    return _transport


#: The days each fixture holds. The catalogued windows are months long; the
#: tests narrow them to the recorded days so that the loader's completeness
#: check (at least 95% of a grid event's hours) sees a complete span.
_FIXTURE_WINDOWS = {
    "quebec_1989": ("1989-03-11", "1989-03-15"),
    "texas_2021": ("2021-02-13", "2021-02-19"),
    "texas_2023": ("2023-01-28", "2023-02-08"),
}


@pytest.fixture(autouse=True)
def _fixture_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    for event_id, (start, end) in _FIXTURE_WINDOWS.items():
        monkeypatch.setitem(_EVENT_CATALOG[event_id], "start", start)
        monkeypatch.setitem(_EVENT_CATALOG[event_id], "end", end)


def _unreachable(*_args: Any, **_kwargs: Any) -> bytes:
    raise ConnectionError("simulated: archive unreachable")


def _loader(tmp_path: Path) -> EnergyLoader:
    return EnergyLoader(cache_dir=tmp_path, max_retries=0)


def test_a_storm_is_read_from_omni2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Was: a synthetic Kp profile; no request was made."""
    seen: list[str] = []
    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
    monkeypatch.setattr(_GET_BYTES, _archives(seen))
    loader = _loader(tmp_path)

    frame = loader.fetch_historical("quebec_1989")

    assert seen == [
        "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_1989.dat",
        "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/"
        "x-rays/goes/xrs/goes-xrs-report_1989.txt",
    ]
    assert not loader.is_synthetic(frame)
    assert (frame["series_source"] == "nasa_omni2").all()
    # Five recorded days in 3-hour steps.
    assert len(frame) == 40
    assert frame["timestamp"].diff().dropna().eq(pd.Timedelta(hours=3)).all()
    # The storm that took down Hydro-Quebec: Kp 9 on 13 and 14 March 1989.
    assert frame["kp"].max() == 9.0
    peak = frame.loc[frame["kp"].idxmax(), "timestamp"]
    assert pd.Timestamp("1989-03-13", tz="UTC") <= peak < pd.Timestamp("1989-03-15", tz="UTC")
    # No spacecraft measured the solar wind for much of it; that is NaN, not 0.
    assert frame["solar_wind_speed"].isna().any()
    assert frame["solar_wind_speed"].dropna().between(250.0, 1200.0).all()
    assert not frame["xray_class"].isna().any()


def test_storm_labels_and_features_come_from_the_recorded_kp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_GET_BYTES, _archives())
    loader = _loader(tmp_path)

    frame = loader.fetch_historical("quebec_1989")
    labels = loader.get_ground_truth("quebec_1989")
    features = loader.engineer_features(frame)

    assert labels.tolist() == (frame["kp"] >= 6.67).astype(int).tolist()  # NOAA G3 and up
    assert 0 < labels.sum() < len(labels)
    assert np.isfinite(features).all()
    columns = EnergyLoader.FEATURE_COLUMNS
    speed = features[:, columns.index("solar_wind_speed")]
    # A step with no coverage takes the recorded median, never zero.
    assert speed.min() > 250.0
    assert features[:, columns.index("xray_class")].max() == 5.0  # recorded X-class flares
    assert loader.engineer_scored_features(frame).shape == (len(frame), 3)


def test_the_omni2_reader_agrees_with_the_training_code() -> None:
    """Two readers of one archive: the same Kp, speed and density, hour for hour."""
    torch = pytest.importorskip("torch")
    assert torch is not None
    from omni_mercury_engine.ml.hazard_training.solar_storm import parse_omni2

    path = _FIXTURES / "omni2_1989_quebec.dat"
    reference = parse_omni2([path])
    parsed = EnergyLoader._parse_omni2(path.read_text(encoding="ascii"))

    np.testing.assert_array_equal(parsed["kp"].to_numpy(), reference.kp)
    np.testing.assert_array_equal(parsed["solar_wind_speed"].to_numpy(), reference.speed)
    np.testing.assert_array_equal(parsed["solar_wind_density"].to_numpy(), reference.density)
    assert parsed["timestamp"].iloc[0] == pd.Timestamp("1989-03-11T00:00:00Z")


def test_the_grid_event_is_the_operators_recorded_hours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was: refused without ``EIA_API_KEY``; with one, demand beside invented columns."""
    seen: list[str] = []
    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    monkeypatch.setattr(_GET_BYTES, _archives(seen))
    loader = _loader(tmp_path)

    frame = loader.fetch_historical("texas_2021")

    assert any(url.startswith("https://www.eia.gov/electricity/gridmonitor/") for url in seen)
    assert not loader.is_synthetic(frame)
    assert (frame["series_source"] == "eia_930+nasa_omni2").all()
    assert "grid_supply" not in frame.columns
    # ERCOT only: the other operator in the file is 20 to 30 GW, ERCOT 40 to 75.
    assert frame["grid_demand"].min() > 35_000.0
    assert 72_000.0 < frame["grid_demand_forecast"].max() < 76_000.0
    # The space-weather columns are what was measured in February 2021.
    assert frame["kp"].nunique() > 3
    assert frame["solar_wind_speed"].nunique() > 10
    assert not (frame["solar_wind_speed"] == 400.0).all()


def test_the_grid_label_is_the_measured_shortfall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was: ``demand > 0.95 * demand``, true on every row."""
    monkeypatch.setattr(_GET_BYTES, _archives())
    loader = _loader(tmp_path)

    frame = loader.fetch_historical("texas_2021")
    labels = loader.get_ground_truth("texas_2021")

    shortfall = 1.0 - frame["grid_demand"] / frame["grid_demand_forecast"]
    assert labels.tolist() == (shortfall > GRID_SHORTFALL_THRESHOLD).astype(int).tolist()
    labelled = frame["timestamp"][labels == 1]
    assert labelled.iloc[0] == pd.Timestamp("2021-02-15T12:00:00Z")
    assert labelled.iloc[-1] == pd.Timestamp("2021-02-18T12:00:00Z")
    # Not every step: the days before the load shed are normal operation.
    assert 15 < labels.sum() < 30
    assert labels[frame["timestamp"] < pd.Timestamp("2021-02-15", tz="UTC")].sum() == 0


def test_a_cell_the_operator_did_not_report_is_missing_not_zero() -> None:
    """Was: ``value: null`` became a demand of 0.0 MW."""
    raw = (
        b'"Balancing Authority","UTC Time at End of Hour","Demand Forecast (MW)","Demand (MW)",'
        b'"Net Generation (MW)","Total Interchange (MW)"\n'
        b'"ERCO","02/15/2021 2:00:00 PM","72,183","","50,588","-814"\n'
        b'"CISO","02/15/2021 2:00:00 PM","25,000","24,000","20,000","-4,000"\n'
    )

    (record,) = EnergyLoader._parse_eia930(raw, "ERCO")

    assert record["timestamp"] == pd.Timestamp("2021-02-15T13:00:00Z")
    assert record["grid_demand_forecast"] == 72183.0
    assert np.isnan(record["grid_demand"])
    assert record["grid_interchange"] == -814.0


def test_an_unreachable_archive_is_refused_without_the_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
    monkeypatch.setattr(_GET_BYTES, _unreachable)

    for event_id in ("quebec_1989", "texas_2021"):
        with pytest.raises(DataSourceUnavailableError, match=event_id):
            _loader(tmp_path).fetch_historical(event_id)
    assert list(tmp_path.glob("*.json")) == []


def test_a_recorded_frame_is_cached_and_served_without_a_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_GET_BYTES, _archives())
    loader = _loader(tmp_path)
    first = loader.fetch_historical("texas_2021")

    monkeypatch.setattr(_GET_BYTES, _unreachable)
    second = loader.fetch_historical("texas_2021")

    assert second["timestamp"].tolist() == first["timestamp"].tolist()
    assert second["grid_demand"].tolist() == first["grid_demand"].tolist()
    assert not loader.is_synthetic(second)


# ---------------------------------------------------------------------------
# DOE OE-417: grid disturbances as the Department of Energy recorded them.
# ---------------------------------------------------------------------------


def test_disturbance_reports_are_read_from_the_doe_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(_GET_BYTES, _archives(seen))
    loader = _loader(tmp_path)

    reports = loader.fetch_disturbance_reports()
    texas = loader.fetch_disturbance_reports("TRE")

    # 341 records in the 2023 summary, 32 of them undated; four pages of 100.
    assert len(reports) == 309
    assert set(reports["began"].dt.year) == {2023}
    assert len([url for url in seen if "oe-417" in url]) == 4
    assert len(texas) == 41
    assert (texas["nerc_region"] == "TRE").all()
    ice_storm = texas[texas["customers_affected"] == 360000]
    assert ice_storm["began"].iloc[0] == pd.Timestamp("2023-01-31", tz="UTC")
    assert ice_storm["restored"].iloc[0] == pd.Timestamp("2023-02-06", tz="UTC")
    # Blank on the form is missing, not zero.
    assert np.isnan(ice_storm["demand_loss_mw"].iloc[0])


def test_a_doe_labelled_grid_event_takes_its_labels_from_the_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Labels that no scored column produces: the operator's hours, DOE's incidents."""
    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
    monkeypatch.setattr(_GET_BYTES, _archives())
    loader = _loader(tmp_path)

    frame = loader.fetch_historical("texas_2023")
    labels = loader.get_ground_truth("texas_2023")

    assert not loader.is_synthetic(frame)
    assert labels.tolist() == frame[DOE_DISTURBANCE_COLUMN].tolist()
    stamps = frame["timestamp"]
    # The recorded span is 28 January to 8 February. DOE: Harris County 24 to
    # 26 January (before the span), the ice storm 31 January to 6 February,
    # Travis County 1 to 4 February.
    assert labels[stamps < pd.Timestamp("2023-01-31", tz="UTC")].sum() == 0
    inside = (stamps >= pd.Timestamp("2023-01-31", tz="UTC")) & (
        stamps < pd.Timestamp("2023-02-07", tz="UTC")
    )
    assert labels[inside].all()
    assert labels[stamps >= pd.Timestamp("2023-02-07", tz="UTC")].sum() == 0
    # The label is not a function of the scored matrix's inputs.
    assert DOE_DISTURBANCE_COLUMN not in EnergyLoader.FEATURE_COLUMNS
    assert loader.engineer_features(frame).shape == (len(frame), 8)


def test_only_reports_with_a_stated_loss_are_disturbances() -> None:
    """Vandalism with no customers or load lost is a security filing, not an outage."""
    stamps = pd.Series(pd.date_range("2023-03-01", periods=16, freq="3h", tz="UTC"))
    reports = pd.DataFrame(
        {
            "began": pd.to_datetime(["2023-03-01", "2023-03-02", "2023-03-02"], utc=True),
            "restored": pd.to_datetime(["2023-03-01", None, "2023-03-01"], utc=True),
            "demand_loss_mw": [np.nan, 120.0, np.nan],
            "customers_affected": [0.0, np.nan, np.nan],
        }
    )
    start, end = pd.Timestamp("2023-03-01", tz="UTC"), pd.Timestamp("2023-03-03", tz="UTC")

    labels = EnergyLoader._disturbance_labels(stamps, reports, start, end)

    # Only the second report counts, and with no restoration date it covers
    # its first day alone.
    assert labels.tolist() == [0] * 8 + [1] * 8


def test_a_year_the_reports_do_not_cover_is_refused_not_labelled_quiet() -> None:
    """The hub's copy is 2023 only; 2021 must not read as a year without disturbances."""
    stamps = pd.Series(pd.date_range("2021-02-15", periods=8, freq="3h", tz="UTC"))
    reports = pd.DataFrame(
        {
            "began": pd.to_datetime(["2023-03-02"], utc=True),
            "restored": pd.to_datetime(["2023-03-04"], utc=True),
            "demand_loss_mw": [np.nan],
            "customers_affected": [470000.0],
        }
    )

    with pytest.raises(ValueError, match="refusing to label an uncovered period"):
        EnergyLoader._disturbance_labels(
            stamps,
            reports,
            pd.Timestamp("2021-02-15", tz="UTC"),
            pd.Timestamp("2021-02-16", tz="UTC"),
        )


def test_a_stand_in_frame_has_no_doe_labels_to_give(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "1")
    monkeypatch.setattr(_GET_BYTES, _unreachable)
    loader = _loader(tmp_path)

    assert loader.is_synthetic(loader.fetch_historical("texas_2023"))
    with pytest.raises(ValueError, match="labelled by DOE OE-417"):
        loader.get_ground_truth("texas_2023")


# ---------------------------------------------------------------------------
# Robustness: found by an independent review of the loader.
# ---------------------------------------------------------------------------


def test_a_span_with_most_of_its_hours_missing_is_not_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was: a truncated download gave a short frame that was cached as recorded."""
    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
    monkeypatch.setitem(_EVENT_CATALOG["texas_2021"], "start", "2021-01-01")
    monkeypatch.setitem(_EVENT_CATALOG["texas_2021"], "end", "2021-03-31")
    monkeypatch.setattr(_GET_BYTES, _archives())
    loader = _loader(tmp_path)

    with pytest.raises(DataSourceUnavailableError, match="168 of 2160 hours"):
        loader.fetch_historical("texas_2021")
    assert list(tmp_path.glob("*.json")) == []


def test_a_row_cut_short_and_a_non_numeric_cell_do_not_fail_the_file() -> None:
    """Was: ``IndexError`` for a truncated row, ``ValueError`` for an "NA" cell."""
    raw = (
        b'"Balancing Authority","UTC Time at End of Hour","Demand Forecast (MW)","Demand (MW)",'
        b'"Net Generation (MW)","Total Interchange (MW)"\n'
        b'"ERCO","02/15/2021 2:00:00 PM","72,183","NA","50,588","-814"\n'
        b'"ERCO","not a time","1","2","3","4"\n'
        b'"ERCO","02/15/2021 3:00:00 PM","72,0'
    )

    records = EnergyLoader._parse_eia930(raw, "ERCO")

    assert len(records) == 1
    assert np.isnan(records[0]["grid_demand"])
    assert records[0]["grid_net_generation"] == 50588.0


def test_a_window_starting_on_a_file_boundary_reads_the_neighbouring_half(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EIA cuts its files at local midnight, so the first UTC hours sit in the other half."""
    seen: list[str] = []

    def _transport(url: str, **_kwargs: Any) -> bytes:
        seen.append(url.rsplit("/", 1)[1])
        raise ConnectionError("not recorded")

    monkeypatch.setattr(_GET_BYTES, _transport)
    start, end = pd.Timestamp("2021-07-01", tz="UTC"), pd.Timestamp("2021-07-03", tz="UTC")

    with pytest.raises(ConnectionError):
        _loader(tmp_path)._fetch_eia930("ERCO", start, end)

    # The January-June file is tried first and tolerated when absent; the
    # July-December file holds the window's own months and is required.
    assert seen == ["EIA930_BALANCE_2021_Jan_Jun.csv", "EIA930_BALANCE_2021_Jul_Dec.csv"]


def test_one_missing_kp_does_not_blank_a_day_of_features() -> None:
    """Was: a NaN Kp propagated through the 24-hour sum and maximum for 8 steps."""
    kp = np.array([1.0, 2.0, 3.0, 9.0, np.nan, 8.0, 4.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    rolling_max = EnergyLoader._compute_rolling_max(kp, window=8)
    rolling_sum = EnergyLoader._compute_rolling_sum(kp, window=8)

    assert rolling_max[4:11].tolist() == [9.0] * 7
    assert not np.isnan(rolling_sum).any()
    assert rolling_sum[5] == 1.0 + 2.0 + 3.0 + 9.0 + 8.0


def test_a_step_with_no_kp_is_not_a_storm_and_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was: ``ValueError: kp must be finite`` from the labels of a frame with one gap."""
    monkeypatch.setattr(_GET_BYTES, _archives())
    loader = _loader(tmp_path)
    frame = loader.fetch_historical("quebec_1989")
    peak = int(frame["kp"].idxmax())
    frame.loc[peak, "kp"] = np.nan
    monkeypatch.setattr(loader, "fetch_historical", lambda _event_id: frame)

    labels = loader.get_ground_truth("quebec_1989")

    assert labels[peak] == 0
    assert labels.sum() > 0


# ---------------------------------------------------------------------------
# A grid event is scored on the grid.
# ---------------------------------------------------------------------------


def test_a_grid_event_is_scored_on_its_own_series_not_on_the_solar_wind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was: three solar-wind columns, for a winter storm and a year of outages."""
    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
    monkeypatch.setattr(_GET_BYTES, _archives())
    loader = _loader(tmp_path)
    frame = loader.fetch_historical("texas_2023")

    grid = loader.engineer_grid_features(frame)
    scored = loader.engineer_scored_features(frame)

    assert loader.is_grid_frame(frame)
    assert grid.shape == (len(frame), len(GRID_FEATURE_COLUMNS))
    assert np.isfinite(grid).all()
    demand = grid[:, GRID_FEATURE_COLUMNS.index("grid_demand")]
    assert np.allclose(demand, frame["grid_demand"].to_numpy(dtype=float))
    change = grid[:, GRID_FEATURE_COLUMNS.index("grid_demand_change")]
    assert change[0] == 0.0
    assert np.allclose(change[1:], np.diff(demand))
    error = grid[:, GRID_FEATURE_COLUMNS.index("grid_forecast_error")]
    forecast = frame["grid_demand_forecast"].to_numpy(dtype=float)
    assert np.allclose(error, (forecast - demand) / forecast)
    # DOE's reports label this event and no column produces them: all six score.
    assert np.array_equal(scored, grid)
    # The space-weather matrix is unchanged for callers that ask for it.
    assert loader.engineer_features(frame).shape == (len(frame), 8)


def test_a_shortfall_labelled_grid_event_is_not_scored_on_demand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Texas 2021 is labelled by demand against forecast; neither is scored."""
    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
    monkeypatch.setattr(_GET_BYTES, _archives())
    loader = _loader(tmp_path)
    frame = loader.fetch_historical("texas_2021")

    grid = loader.engineer_grid_features(frame)
    scored = loader.engineer_scored_features(frame)

    kept = ["grid_net_generation", "grid_interchange", "grid_generation_change"]
    assert scored.shape == (len(frame), 3)
    for position, name in enumerate(kept):
        assert np.array_equal(scored[:, position], grid[:, GRID_FEATURE_COLUMNS.index(name)])


def test_a_storm_frame_has_no_grid_view(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_GET_BYTES, _archives())
    loader = _loader(tmp_path)
    frame = loader.fetch_historical("quebec_1989")

    assert not loader.is_grid_frame(frame)
    with pytest.raises(ValueError, match="not a grid-event frame"):
        loader.engineer_grid_features(frame)


# ---------------------------------------------------------------------------
# The X-ray class of a historical storm is the recorded one.
# ---------------------------------------------------------------------------


def test_a_storm_frame_carries_the_recorded_flare_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was: NaN in every step, on the ground that OMNI2 does not hold it.

    NOAA NCEI keeps the GOES XRS flare reports by year. Region 5395, whose
    X4.5 flare of 1989-03-10 preceded the Quebec storm, produced an X1.2 flare
    peaking at 03:20 UTC on 1989-03-13, the day of the blackout.
    """
    monkeypatch.setattr(_GET_BYTES, _archives())
    loader = _loader(tmp_path)

    frame = loader.fetch_historical("quebec_1989")

    assert not frame["xray_class"].isna().any()
    assert set(frame["xray_class"].unique()) <= {0.0, 1.0, 2.0, 3.0, 4.0, 5.0}
    step = frame[frame["timestamp"] == pd.Timestamp("1989-03-13T03:00:00Z")]
    assert step["xray_class"].iloc[0] == 5.0  # the X-class flare
    quiet = frame[frame["timestamp"] == pd.Timestamp("1989-03-15T09:00:00Z")]
    assert quiet["xray_class"].iloc[0] < 5.0
    features = loader.engineer_features(frame)
    column = EnergyLoader.FEATURE_COLUMNS.index("xray_class")
    assert features[:, column].max() == 5.0


def test_a_year_outside_the_flare_archive_is_unknown_not_quiet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The archive ends in 2017; a 2021 frame must not claim there were no flares."""
    monkeypatch.setattr(_GET_BYTES, _archives())

    frame = _loader(tmp_path).fetch_historical("texas_2021")

    assert frame["xray_class"].isna().all()


def test_flare_report_rows_that_do_not_fit_are_skipped() -> None:
    """Lines as NOAA writes them, then three that are not flare records."""
    recorded = (_FIXTURES / "goes_xrs_1989_quebec.txt").read_text(encoding="ascii").splitlines()
    x_flare = next(line for line in recorded if line.startswith("31777890310  1848"))
    m_flare = next(line for line in recorded if line[59] == "M")
    text = "\n".join(
        [
            x_flare,
            m_flare,
            x_flare[:23] + "    " + x_flare[27:],  # no peak time: the start time stands
            "short line",
            x_flare[:5] + "89XX10" + x_flare[11:],  # unreadable date
            x_flare[:59] + "?" + x_flare[60:],  # no such class
        ]
    )

    parsed = EnergyLoader._parse_flare_report(text)

    assert parsed["xray_class"].tolist() == [5.0, 4.0, 5.0]
    assert parsed["timestamp"].iloc[0] == pd.Timestamp("1989-03-10T19:12:00Z")
    assert parsed["timestamp"].iloc[2] == pd.Timestamp("1989-03-10T18:48:00Z")
