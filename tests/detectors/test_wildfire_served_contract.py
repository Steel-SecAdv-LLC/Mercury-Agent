# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""What ``WildfireDetector.predict_wildfire`` serves, on a real FIRMS raster.

Each test fails on the detector as it was before: any 3-channel array went to
the next-day FIRMS forecaster (a brightness image with a 380 K fire read
p=0.078, not detected), NaN and inf passed through, the risk ladder summed
floats so three weather flags with no fire read "very_high", spread was
modelled with no fire, fuel moisture was a fixed 10% whatever the weather
(and then the "NFDRS" value from shelter weather, not the air at the fuel),
and smoke read "not detected" without being assessed.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.core.exceptions import InvalidObservationError
from omni_mercury_engine.detectors.geological import wildfire
from omni_mercury_engine.detectors.geological.wildfire import (
    WildfireDetector,
    WildfirePredictionResult,
)
from omni_mercury_engine.ml.hazard_training import wildfire_ignition as wf

#: The documented ``weather_data`` key of the NFDRS state-of-weather code.
SOW = "nfdrs_state_of_weather"

FIXTURE_CSV = (
    Path(__file__).resolve().parents[1] / "fixtures" / "hazard_training" / "wildfire"
) / "viirs_snpp_2023_ca_excerpt.csv"


@pytest.fixture(scope="module")
def firms_patch() -> np.ndarray:
    """A wildfire-firms-v1 raster around the hottest 2023-08-30 VIIRS detection."""
    grids = wf.rasterize_daily([wf.parse_firms_csv(FIXTURE_CSV)])
    t = dt.date(2023, 8, 30).toordinal() - grids.day0
    cells = grids.days[t]
    j = int(np.argmax(cells.bt_max))
    return wf.build_patch(grids, t, int(cells.rows[j]), int(cells.cols[j]))


@pytest.fixture(scope="module")
def served() -> WildfireDetector:
    return WildfireDetector()


def _brightness_scene(fire_k: float = 380.0) -> np.ndarray:
    scene = np.full((3, 32, 32), 295.0)
    scene[:, 14:16, 14:16] = fire_k
    return scene


def test_firms_raster_is_scored_as_a_next_day_forecast(
    served: WildfireDetector, firms_patch: np.ndarray
) -> None:
    result = served.predict_wildfire({"thermal_image": firms_patch})
    assert result.assessment == "next_day_forecast"
    assert 0.0 <= result.confidence <= 1.0


def test_brightness_scene_with_a_fire_is_detected_as_active_fire(
    served: WildfireDetector,
) -> None:
    result = served.predict_wildfire({"thermal_image": _brightness_scene()})
    assert result.assessment == "active_fire"
    assert result.fire_detected is True


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_raster_is_refused(served: WildfireDetector, bad: float) -> None:
    scene = _brightness_scene()
    scene[0, 0, 0] = bad
    with pytest.raises(InvalidObservationError, match="non-finite"):
        served.predict_wildfire({"thermal_image": scene})


def test_plain_list_is_accepted_as_an_array(served: WildfireDetector) -> None:
    result = served.predict_wildfire({"thermal_image": _brightness_scene().tolist()})
    assert result.fire_detected is True


def test_three_weather_flags_without_fire_are_high_not_very_high(
    served: WildfireDetector,
) -> None:
    weather = {"wind_speed_kmh": 45.0, "relative_humidity_pct": 10.0, "temperature_c": 40.0}
    result = served.predict_wildfire({"weather_data": weather})
    assert result.fire_detected is False
    assert result.risk_level == "high"


def test_risk_ladder_with_fire(served: WildfireDetector) -> None:
    weather = {"wind_speed_kmh": 45.0, "relative_humidity_pct": 10.0, "temperature_c": 40.0}
    scene = {"thermal_image": _brightness_scene()}
    assert served.predict_wildfire({**scene, "weather_data": weather}).risk_level == "extreme"
    two = {"wind_speed_kmh": 45.0, "relative_humidity_pct": 10.0}
    assert served.predict_wildfire({**scene, "weather_data": two}).risk_level == "very_high"


def test_no_spread_without_a_fire(served: WildfireDetector) -> None:
    result = served.predict_wildfire({"weather_data": {"wind_speed_kmh": 60.0}})
    assert result.spread_rate_km_hr is None


def test_weather_outside_its_units_is_refused(served: WildfireDetector) -> None:
    with pytest.raises(InvalidObservationError, match="relative_humidity_pct"):
        served.predict_wildfire({"weather_data": {"relative_humidity_pct": 0.15e3}})


def test_fire_perimeter_area_is_pixel_count_times_pixel_area(served: WildfireDetector) -> None:
    result = served.predict_wildfire({"thermal_image": _brightness_scene(), "pixel_size_km": 0.5})
    assert result.fire_perimeter_km2 == pytest.approx(4 * 0.25)


def test_every_result_field_of_the_released_api_is_kept() -> None:
    fields = set(WildfirePredictionResult.__dataclass_fields__)
    assert {
        "fire_perimeter_km2",
        "smoke_detected",
        "fuel_moisture",
        "weather_factors",
        "evacuation_zones",
        "containment_strategy",
    } <= fields
    # Positional construction of the released field order still works.
    positional = WildfirePredictionResult(True, 0.9, "high", [(1.0, 2.0)], 3.0)
    assert (positional.ignition_locations, positional.fire_perimeter_km2) == ([(1.0, 2.0)], 3.0)


@pytest.mark.parametrize(
    ("temperature_c", "humidity", "emc"),
    [
        # GTR INT-169 Appendix C at 90 degF (32.2 degC) and 60 degF (15.6 degC).
        ((90 - 32) * 5 / 9, 5.0, 0.03299 + 0.281073 * 5 - 0.000578 * 5 * 90),
        ((90 - 32) * 5 / 9, 15.0, 2.22749 + 0.160107 * 15 - 0.01478 * 90),
        ((60 - 32) * 5 / 9, 70.0, 21.0606 + 0.005565 * 4900 - 0.00035 * 70 * 60 - 0.483199 * 70),
    ],
)
def test_equilibrium_moisture_follows_nfdrs(
    temperature_c: float, humidity: float, emc: float
) -> None:
    assert wildfire.equilibrium_moisture_content(temperature_c, humidity) == pytest.approx(emc)


def test_fuel_moisture_is_estimated_from_the_weather_and_drives_spread(
    served: WildfireDetector,
) -> None:
    weather = {
        "temperature_c": 32.0,
        "relative_humidity_pct": 12.0,
        "wind_speed_kmh": 25.0,
        SOW: 0,  # clear
    }
    result = served.predict_wildfire(
        {"thermal_image": _brightness_scene(), "weather_data": weather}
    )
    # GTR INT-169 Table 6, clear: +25 degF and RH x 0.75 at the fuel (h = 9 < 10).
    t_f, h = 32.0 * 9 / 5 + 32 + 25, 12.0 * 0.75
    emc = 0.03299 + 0.281073 * h - 0.000578 * h * t_f
    assert result.weather_factors["equilibrium_moisture_pct"] == pytest.approx(emc)
    assert result.weather_factors["wind_speed_kmh"] == 25.0
    assert "wind_direction_deg" not in result.weather_factors  # not supplied, not invented
    assert result.fuel_moisture == pytest.approx(1.03 * emc)
    # Dry fuel spreads faster than the fixed 10% the spread model assumed.
    fixed = served.predict_wildfire(
        {
            "thermal_image": _brightness_scene(),
            "weather_data": weather,
            "fire_data": {"fuel_moisture_pct": 10.0},
        }
    )
    assert fixed.fuel_moisture == 10.0
    assert result.spread_rate_km_hr is not None and fixed.spread_rate_km_hr is not None
    assert result.spread_rate_km_hr > fixed.spread_rate_km_hr


@pytest.mark.parametrize(
    ("state_of_weather", "add_f", "humidity_factor"),
    # GTR INT-169 Table 6: clear, scattered, broken, overcast.
    [(0, 25.0, 0.75), (1, 19.0, 0.83), (2, 12.0, 0.91), (3, 5.0, 1.0)],
)
def test_nfdrs_fuel_moisture_uses_the_air_at_the_fuel(
    served: WildfireDetector, state_of_weather: int, add_f: float, humidity_factor: float
) -> None:
    # Shelter 25 degC (77 degF), 30% RH: the 10 <= h < 50 regression (eq. C-2).
    weather = {
        "temperature_c": 25.0,
        "relative_humidity_pct": 30.0,
        SOW: state_of_weather,
    }
    result = served.predict_wildfire(
        {"thermal_image": _brightness_scene(), "weather_data": weather}
    )
    t_f, h = 77.0 + add_f, 30.0 * humidity_factor
    emc = 2.22749 + 0.160107 * h - 0.01478 * t_f
    assert result.fuel_moisture == pytest.approx(1.03 * emc)
    assert result.weather_factors == pytest.approx(
        {
            "temperature_c": 25.0,
            "relative_humidity_pct": 30.0,
            SOW: float(state_of_weather),
            "fuel_interface_temperature_c": (t_f - 32.0) * 5.0 / 9.0,
            "fuel_interface_relative_humidity_pct": h,
            "equilibrium_moisture_pct": emc,
        }
    )


def test_hot_shelter_reading_under_clear_sky_is_estimated(served: WildfireDetector) -> None:
    # 48 degC in the shelter is about 62 degC at the fuel, past the 60 degC
    # limit on a shelter temperature; it is still a valid observation.
    weather = {"temperature_c": 48.0, "relative_humidity_pct": 10.0, SOW: 0}
    result = served.predict_wildfire(
        {"thermal_image": _brightness_scene(), "weather_data": weather}
    )
    t_f, h = 48.0 * 9 / 5 + 32 + 25, 7.5
    emc = 0.03299 + 0.281073 * h - 0.000578 * h * t_f
    assert result.weather_factors["fuel_interface_temperature_c"] == pytest.approx(
        48.0 + 25 * 5 / 9
    )
    assert result.fuel_moisture == pytest.approx(1.03 * emc)


def test_shelter_weather_alone_gives_no_nfdrs_fuel_moisture(served: WildfireDetector) -> None:
    weather = {"temperature_c": 32.0, "relative_humidity_pct": 12.0, "wind_speed_kmh": 25.0}
    result = served.predict_wildfire(
        {"thermal_image": _brightness_scene(), "weather_data": weather}
    )
    assert result.fuel_moisture is None
    assert result.weather_factors == weather  # what was supplied, nothing derived
    # The spread model keeps the fixed 10% it used before any EMC was computed.
    fixed = served.predict_wildfire(
        {
            "thermal_image": _brightness_scene(),
            "weather_data": weather,
            "fire_data": {"fuel_moisture_pct": 10.0},
        }
    )
    assert result.spread_rate_km_hr is not None
    assert result.spread_rate_km_hr == fixed.spread_rate_km_hr


@pytest.mark.parametrize("code", [4, 6, 1.5, -1, "clear"])
def test_state_of_weather_without_an_interface_adjustment_is_refused(
    served: WildfireDetector, code: object
) -> None:
    weather = {
        "temperature_c": 32.0,
        "relative_humidity_pct": 12.0,
        SOW: code,
    }
    with pytest.raises(InvalidObservationError, match=SOW):
        served.predict_wildfire({"thermal_image": _brightness_scene(), "weather_data": weather})


def test_unassessed_fields_say_so(served: WildfireDetector) -> None:
    result = served.predict_wildfire({"thermal_image": _brightness_scene()})
    assert result.smoke_detected is None
    assert result.fuel_moisture is None
    assert (result.evacuation_zones, result.containment_strategy) == ([], [])
    official = served.predict_wildfire(
        {"thermal_image": _brightness_scene(), "evacuation_zones": ["Zone 4B"]}
    )
    assert official.evacuation_zones == ["Zone 4B"]
