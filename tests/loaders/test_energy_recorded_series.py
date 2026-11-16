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

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError
from omni_mercury_engine.loaders.energy_loader import GRID_SHORTFALL_THRESHOLD, EnergyLoader

if TYPE_CHECKING:
    from collections.abc import Callable

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "energy"
_GET_BYTES = "omni_mercury_engine.loaders.base.SafeHTTPClient.get_bytes"


def _archives(seen: list[str] | None = None) -> Callable[..., bytes]:
    """A transport serving the recorded files by the URL the loader asks for."""

    def _transport(url: str, **_kwargs: Any) -> bytes:
        if seen is not None:
            seen.append(url)
        if url.endswith("omni2_1989.dat"):
            return (_FIXTURES / "omni2_1989_quebec.dat").read_bytes()
        if url.endswith("omni2_2021.dat"):
            return (_FIXTURES / "omni2_2021_uri.dat").read_bytes()
        if url.endswith("EIA930_BALANCE_2021_Jan_Jun.csv"):
            return (_FIXTURES / "eia930_balance_2021_uri.csv").read_bytes()
        raise ConnectionError(f"no recorded file for {url}")

    return _transport


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

    assert seen == ["https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_1989.dat"]
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
    assert frame["xray_class"].isna().all()


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
    assert (features[:, columns.index("xray_class")] == 0.0).all()
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
