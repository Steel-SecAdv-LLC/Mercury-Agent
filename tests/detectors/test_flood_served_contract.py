# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What ``FloodDetector`` serves, offline and on recorded NWS/NWPS responses.

Each test fails on the detector as it was before: an empty record read as at
flood stage ("MINOR FLOODING"), invented category stages overrode the gauge's
own, severity ignored the gauge category, injected noise moved the decision,
advisories counted as floods, a quiet gauge feed reported an arbitrary
gauge's stage, and peak discharge, time to peak, area and population read 0
without being assessed.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

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
    return cast("NWSWeatherAlertsSource", _replay(NWSWeatherAlertsSource(), payload))


@pytest.fixture(scope="module")
def recorded_alerts() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads((FIXTURES / "nws_alerts_flood.json").read_text())
    return payload


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
    assert result.live_context is not None
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
    assert result.live_context is not None
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
    assert result.live_context is not None
    assert result.live_context["gauge_count"] == 60
    assert result.severity == "major"


GAUGE = {"current_stage_ft": 12.0, "flood_stage_ft": 15.0}


def test_released_result_fields_are_kept() -> None:
    fields = set(FloodPredictionResult.__dataclass_fields__)
    assert {
        "peak_discharge_cfs",
        "time_to_peak_hours",
        "affected_area_sq_mi",
        "population_at_risk",
    } <= fields
    result = FloodDetector().predict_flood({"gauge_data": GAUGE})
    # Not assessed is None; 0.0 claimed a dry channel and nothing affected.
    assert (result.peak_discharge_cfs, result.time_to_peak_hours) == (None, None)
    assert (result.affected_area_sq_mi, result.population_at_risk) == (None, None)


def test_peak_discharge_is_the_largest_supplied_flow() -> None:
    gauge = {
        **GAUGE,
        "discharge_cfs": 8_200.0,
        "discharge_history_cfs": [5_000.0, 7_900.0],
        "forecast_discharge_cfs": [9_100.0, 12_400.0, 11_000.0],
    }
    assert FloodDetector().predict_flood({"gauge_data": gauge}).peak_discharge_cfs == 12_400.0


def test_time_to_peak_is_the_forecast_crest_lead() -> None:
    rising = {
        **GAUGE,
        "forecast_stage_ft": [13.0, 16.5, 17.2, 16.0],
        "forecast_lead_h": [6, 12, 18, 24],
    }
    assert FloodDetector().predict_flood({"gauge_data": rising}).time_to_peak_hours == 18.0
    crested = {**GAUGE, "forecast_stage_ft": [11.5, 10.0], "forecast_lead_h": [6, 12]}
    assert FloodDetector().predict_flood({"gauge_data": crested}).time_to_peak_hours == 0.0


@pytest.mark.parametrize(
    ("extra", "fragment"),
    [
        ({"forecast_stage_ft": [16.0, 17.0]}, "forecast_lead_h"),
        ({"forecast_stage_ft": [16.0, 17.0], "forecast_lead_h": [12, 6]}, "increasing"),
        ({"discharge_cfs": -5.0}, "discharge_cfs"),
    ],
)
def test_malformed_hydrograph_is_refused(extra: dict[str, Any], fragment: str) -> None:
    with pytest.raises(InvalidObservationError, match=fragment):
        FloodDetector().predict_flood({"gauge_data": {**GAUGE, **extra}})


def test_live_worst_gauge_reports_its_observed_flow() -> None:
    gauges = json.loads((FIXTURES / "nwps_gauges.json").read_text())
    gauges["gauges"][0]["status"]["observed"]["floodCategory"] = "minor"
    result = FloodDetector(gauge_source=_replay(NOAANWPSSource(), gauges)).detect_live()
    assert result.live_context is not None
    assert result.live_context["worst_gauge"]["observed_flow_cfs"] == pytest.approx(1_340.0)


def test_soil_products_are_never_fabricated_from_defaults() -> None:
    """Regression: empty soil input was served as a measured 50%-moist loam
    with a 0.8 runoff coefficient; an unknown soil class read as loam."""
    from omni_mercury_engine.detectors.geological.flood_detector import SoilSaturationModel

    model = SoilSaturationModel()
    out = model.analyze_soil_conditions({})
    assert out["runoff_coefficient"] is None
    assert out["saturation_status"] == "unknown"
    assert set(out["missing_inputs"]) == {"soil_moisture_pct", "soil_type"}
    with pytest.raises(InvalidObservationError, match="soil_type"):
        model.analyze_soil_conditions({"soil_moisture_pct": 60.0, "soil_type": "basalt"})


class TestPrecipitationFlashFloodGuidance:
    """Flash-flood gate against per-location NWS Flash Flood Guidance."""

    def _analyzer(self) -> Any:
        from omni_mercury_engine.detectors.geological.flood_detector import (
            PrecipitationAnalyzer,
        )

        return PrecipitationAnalyzer()

    def test_rain_at_or_above_guidance_fires(self) -> None:
        """1.0 in/h over an 0.8 in/h FFG is flash-flood potential, though the
        fixed 2.0 in/h screen would have stayed silent."""
        result = self._analyzer().analyze_precipitation(
            {"precipitation_1h_inches": 1.0, "ffg_1h_inches": 0.8}
        )
        assert result["flash_flood_risk"] is True
        assert result["flash_flood_basis"] == "nws_flash_flood_guidance"

    def test_rain_below_guidance_holds(self) -> None:
        """2.2 in/h under a 2.5 in/h FFG is not flash-flood potential, though
        the fixed screen would have fired."""
        result = self._analyzer().analyze_precipitation(
            {"precipitation_1h_inches": 2.2, "ffg_1h_inches": 2.5}
        )
        assert result["flash_flood_risk"] is False
        assert result["flash_flood_basis"] == "nws_flash_flood_guidance"

    def test_longer_window_guidance_is_checked(self) -> None:
        result = self._analyzer().analyze_precipitation(
            {
                "precipitation_1h_inches": 0.4,
                "precipitation_6h_inches": 3.4,
                "ffg_6h_inches": 3.0,
            }
        )
        assert result["flash_flood_risk"] is True

    def test_without_guidance_the_fixed_screen_is_labelled(self) -> None:
        result = self._analyzer().analyze_precipitation({"precipitation_1h_inches": 2.5})
        assert result["flash_flood_risk"] is True
        assert result["flash_flood_basis"] == "fixed_screen_unsourced"

    def test_decision_source_declares_both_bases(self) -> None:
        published, fallback = self._analyzer().decision_source
        assert published.kind == "published"
        assert "Flash Flood Guidance" in published.citation
        assert fallback.kind == "experimental"
