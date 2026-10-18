# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""What ``FloodDetector`` serves, offline and on recorded NWS/NWPS responses.

Each test fails on the detector as it was before: an empty record read as at
flood stage ("MINOR FLOODING"), invented category stages overrode the gauge's
own, severity ignored the gauge category, injected noise moved the decision,
advisories counted as floods, and a quiet gauge feed reported an arbitrary
gauge's stage.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.core.exceptions import InvalidObservationError
from omni_mercury_engine.data_sources.earth_science import NOAANWPSSource, NWSWeatherAlertsSource
from omni_mercury_engine.detectors.geological.flood_detector import (
    FloodDetector,
    FloodPredictionResult,
    FloodType,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "live_wiring"


class _Response:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def json(self) -> Any:
        return self._payload


def _replay(source: Any, payload: Any) -> Any:
    async def _get(endpoint: str, params: dict[str, Any] | None = None) -> _Response:
        return _Response(payload)

    source._http_get = _get
    return source


def _alerts(payload: Any) -> NWSWeatherAlertsSource:
    return _replay(NWSWeatherAlertsSource(), payload)


@pytest.fixture(scope="module")
def recorded_alerts() -> dict[str, Any]:
    return json.loads((FIXTURES / "nws_alerts_flood.json").read_text())


def test_empty_record_is_no_flood() -> None:
    result = FloodDetector().predict_flood({})
    assert result.flood_likely is False
    assert result.severity == "no_flood"
    assert result.warning_actions == []


def test_gauge_needs_its_own_flood_stage() -> None:
    with pytest.raises(InvalidObservationError, match="flood_stage_ft"):
        FloodDetector().predict_flood({"gauge_data": {"current_stage_ft": 8.0}})


def test_small_creek_uses_its_own_categories() -> None:
    gauge = {
        "current_stage_ft": 9.0,
        "flood_stage_ft": 6.0,
        "moderate_flood_stage_ft": 7.5,
        "major_flood_stage_ft": 8.5,
    }
    result = FloodDetector().predict_flood({"gauge_data": gauge})
    assert result.flood_likely is True
    assert result.severity == "major"


def test_flash_rain_with_gauge_at_minor_is_minor_not_record() -> None:
    data = {
        "precip_data": {"precipitation_1h_inches": 2.5},
        "gauge_data": {"current_stage_ft": 15.0, "flood_stage_ft": 15.0},
    }
    result = FloodDetector().predict_flood(data)
    assert result.flood_likely is True
    assert result.severity == "minor"
    assert result.flood_type == FloodType.FLASH.value


def test_out_of_order_categories_are_refused() -> None:
    gauge = {"current_stage_ft": 9.0, "flood_stage_ft": 10.0, "major_flood_stage_ft": 8.0}
    with pytest.raises(InvalidObservationError, match="out of order"):
        FloodDetector().predict_flood({"gauge_data": gauge})


@pytest.mark.parametrize(
    ("data", "fragment"),
    [
        ({"soil_data": {"soil_moisture_pct": 150.0}}, "soil_moisture_pct"),
        ({"precip_data": {"precipitation_24h_inches": 254.0}}, "precipitation_24h_inches"),
        ({"precip_data": {"precipitation_1h_inches": float("nan")}}, "finite"),
    ],
)
def test_values_outside_their_units_are_refused(data: dict[str, Any], fragment: str) -> None:
    with pytest.raises(InvalidObservationError, match=fragment):
        FloodDetector().predict_flood(data)


def test_signal_noise_carries_no_decision_weight() -> None:
    base = {"precip_data": {"precipitation_24h_inches": 4.5}}
    noisy = {**base, "signal_data": np.random.default_rng(2).normal(0.0, 3.0, 512)}
    a, b = FloodDetector().predict_flood(base), FloodDetector().predict_flood(noisy)
    assert (a.flood_likely, a.confidence, a.severity) == (b.flood_likely, b.confidence, b.severity)


def test_recorded_flood_warning_counts(recorded_alerts: dict[str, Any]) -> None:
    result = FloodDetector(alerts_source=_alerts(recorded_alerts)).detect_live()
    assert result.flood_likely is True
    assert result.live_context["flood_alert_events"] == ["Flood Warning"]
    assert "Coastal Flood Advisory" in result.live_context["other_flood_products"]
    assert result.flood_type == FloodType.RIVER.value


def test_advisories_alone_are_not_a_flood(recorded_alerts: dict[str, Any]) -> None:
    advisories = copy.deepcopy(recorded_alerts)
    advisories["features"] = [
        f for f in advisories["features"] if f["properties"]["event"] != "Flood Warning"
    ]
    assert advisories["features"]
    result = FloodDetector(alerts_source=_alerts(advisories)).detect_live()
    assert result.flood_likely is False
    assert result.severity == "no_flood"
    assert result.flood_type == FloodType.NO_FLOOD.value


def test_quiet_gauges_report_no_stage() -> None:
    gauges = json.loads((FIXTURES / "nwps_gauges.json").read_text())
    assert all(g["status"]["observed"]["floodCategory"] == "no_flooding" for g in gauges["gauges"])
    result = FloodDetector(gauge_source=_replay(NOAANWPSSource(), gauges)).detect_live()
    assert result.flood_likely is False
    assert result.river_stage_ft == 0.0
    assert "worst_gauge" not in result.live_context


def test_action_stage_is_not_flooding() -> None:
    gauges = json.loads((FIXTURES / "nwps_gauges.json").read_text())
    gauges["gauges"][0]["status"]["observed"]["floodCategory"] = "action"
    result = FloodDetector(gauge_source=_replay(NOAANWPSSource(), gauges)).detect_live()
    assert result.flood_likely is False
    assert result.severity == "no_flood"


def test_every_gauge_in_the_box_is_read() -> None:
    gauges = json.loads((FIXTURES / "nwps_gauges.json").read_text())
    template = gauges["gauges"][0]
    many = {"gauges": [{**copy.deepcopy(template), "lid": f"G{i:03d}"} for i in range(60)]}
    many["gauges"][59]["status"]["observed"]["floodCategory"] = "major"
    source = _replay(NOAANWPSSource(bbox=(-96.0, 28.0, -93.0, 31.0)), many)
    result = FloodDetector(gauge_source=source).detect_live()
    assert result.live_context["gauge_count"] == 60
    assert result.severity == "major"


def test_never_computed_fields_are_gone() -> None:
    fields = set(FloodPredictionResult.__dataclass_fields__)
    for name in (
        "peak_discharge_cfs",
        "time_to_peak_hours",
        "affected_area_sq_mi",
        "population_at_risk",
    ):
        assert name not in fields
