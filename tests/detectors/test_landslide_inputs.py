# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``LandslideDetector.predict_landslide`` refuses inputs it would grade wrongly.

Before :func:`validate_landslide_data`, a NaN rainfall or PGA failed every
threshold comparison and read as a silent all-clear, a negative duration
crashed on a complex square root, a 500 deg slope was graded, and a misspelt
or empty section defaulted to quiet values (0 mm/h, 0 g).
"""

from __future__ import annotations

from typing import Any

import pytest

from omni_mercury_engine.core.exceptions import InputModalityError, InvalidObservationError
from omni_mercury_engine.detectors.geological.landslide import (
    LANDSLIDE_INPUTS,
    LandslideDetector,
)


@pytest.fixture(scope="module")
def detector() -> LandslideDetector:
    return LandslideDetector()


def _storm() -> dict[str, Any]:
    return {
        "rainfall_data": {
            "intensity_mm_hr": 40.0,
            "duration_hours": 12.0,
            "antecedent_7day_mm": 200.0,
        },
        "slope_data": {"slope_angle_deg": 35.0},
        "sensor_data": {"soil_saturation_pct": 95.0, "displacement_rate_mm_day": 20.0},
    }


def test_a_measured_storm_is_graded(detector: LandslideDetector) -> None:
    result = detector.predict_landslide(_storm())
    assert result.rainfall_trigger
    assert result.landslide_imminent


@pytest.mark.parametrize(
    ("section", "key"),
    [(section, key) for section, fields in LANDSLIDE_INPUTS.items() for key in fields],
)
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_value_is_refused(
    detector: LandslideDetector, section: str, key: str, bad: float
) -> None:
    with pytest.raises(InvalidObservationError, match=key):
        detector.predict_landslide({section: {key: bad}})


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("rainfall_data", "duration_hours", -1.0),
        ("slope_data", "slope_angle_deg", 500.0),
        ("sensor_data", "soil_saturation_pct", 140.0),
        ("seismic_data", "pga_g", -0.1),
    ],
)
def test_non_physical_value_is_refused(
    detector: LandslideDetector, section: str, key: str, value: float
) -> None:
    with pytest.raises(InvalidObservationError, match=key):
        detector.predict_landslide({section: {key: value}})


def test_misspelt_key_is_refused(detector: LandslideDetector) -> None:
    with pytest.raises(InvalidObservationError, match="unknown keys"):
        detector.predict_landslide({"rainfall_data": {"intensity_mm_h": 80.0}})


def test_empty_section_is_refused(detector: LandslideDetector) -> None:
    with pytest.raises(InvalidObservationError, match="no measurement"):
        detector.predict_landslide({"seismic_data": {}})


@pytest.mark.parametrize("record", [[1.0, 2.0], {"rainfall_data": [40.0]}])
def test_non_mapping_is_refused(detector: LandslideDetector, record: Any) -> None:
    with pytest.raises(InputModalityError, match="mapping"):
        detector.predict_landslide(record)


def test_record_without_a_section_is_refused(detector: LandslideDetector) -> None:
    with pytest.raises(InvalidObservationError, match="no slope measured"):
        detector.predict_landslide({"metadata": {"site": "x"}})
