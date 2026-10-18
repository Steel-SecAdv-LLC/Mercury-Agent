# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""What ``WildfireDetector.predict_wildfire`` serves, on a real FIRMS raster.

Each test fails on the detector as it was before: any 3-channel array went to
the next-day FIRMS forecaster (a brightness image with a 380 K fire read
p=0.078, not detected), NaN and inf passed through, the risk ladder summed
floats so three weather flags with no fire read "very_high", spread was
modelled with no fire, and the hotspot area was named a perimeter.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.core.exceptions import InvalidObservationError
from omni_mercury_engine.detectors.geological.wildfire import (
    WildfireDetector,
    WildfirePredictionResult,
)
from omni_mercury_engine.ml.hazard_training import wildfire_ignition as wf

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


def test_hotspot_area_is_named_an_area(served: WildfireDetector) -> None:
    result = served.predict_wildfire({"thermal_image": _brightness_scene(), "pixel_size_km": 0.5})
    assert result.hotspot_area_km2 == pytest.approx(4 * 0.25)
    fields = set(WildfirePredictionResult.__dataclass_fields__)
    assert "fire_perimeter_km2" not in fields
    for never_computed in (
        "smoke_detected",
        "fuel_moisture",
        "weather_factors",
        "evacuation_zones",
        "containment_strategy",
    ):
        assert never_computed not in fields
