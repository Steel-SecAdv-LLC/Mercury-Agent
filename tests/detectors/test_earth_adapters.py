# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Geophysical, hydro-meteorological and marine adapters serve their detector's own verdict.

Every adapter is checked against a direct call to the domain detector it
wraps (the adapter adds no judgement of its own), on hazardous and quiet
inputs, and must refuse malformed input instead of grading it.  Recorded real
observations are used where the repository holds them: the USGS Tohoku 2011
event, NOAA CO-OPS water levels at The Battery (8518750), the CHIRPS 2018
rainfall at the Jodupala landslide cell, and a VIIRS 2023 FIRMS raster.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.exceptions import InputModalityError, InvalidObservationError
from omni_mercury_engine.detectors.adapters import (
    AvalancheProfileAdapter,
    EqTsunamiCascadeAdapter,
    FireDebrisFlowAdapter,
    LandslideAdapter,
    MarineEcosystemAdapter,
    RockfallAdapter,
    SubsidenceAdapter,
    SurgeFloodCascadeAdapter,
    VolcanicAdapter,
    WildfireSceneAdapter,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
NAN = float("nan")

ADAPTERS = [
    AvalancheProfileAdapter,
    EqTsunamiCascadeAdapter,
    FireDebrisFlowAdapter,
    LandslideAdapter,
    MarineEcosystemAdapter,
    RockfallAdapter,
    SubsidenceAdapter,
    SurgeFloodCascadeAdapter,
    VolcanicAdapter,
    WildfireSceneAdapter,
]


@pytest.mark.parametrize("cls", ADAPTERS)
def test_every_earth_adapter_is_a_base_detector(cls: Any) -> None:
    assert issubclass(cls, BaseDetector)
    for method in ("fit", "detect", "extract_features", "predict"):
        assert callable(getattr(cls, method))


@pytest.mark.parametrize("cls", ADAPTERS)
def test_scalar_payload_is_refused(cls: Any) -> None:
    with pytest.raises((InputModalityError, InvalidObservationError)):
        cls().detect(3.0)


def _served(adapter: BaseDetector, data: Any) -> tuple[float, bool, Any]:
    out = adapter.detect(data)
    assert out["scores"].shape == (1,)
    return float(out["scores"][0]), bool(out["flags"][0]), out["metadata"]["results"][0]


# ---------------------------------------------------------------------------
# Volcanic
# ---------------------------------------------------------------------------


class TestVolcanic:
    UNREST: dict[str, Any] = {
        "gas_data": {"so2_tons_per_day": 5000.0, "co2_tons_per_day": 8000.0},
        "insar_data": {"vertical_displacement_cm": 30.0},
        "thermal_data": {"brightness_temperature_k": np.full((8, 8), 420.0)},
    }
    QUIET: dict[str, Any] = {
        "gas_data": {"so2_tons_per_day": 100.0, "co2_tons_per_day": 500.0},
        "insar_data": {"vertical_displacement_cm": 0.2},
    }

    @pytest.mark.parametrize("record", [UNREST, QUIET])
    def test_matches_the_detector(self, record: dict[str, Any]) -> None:
        from omni_mercury_engine.detectors.geological.volcanic import VolcanicEruptionDetector

        direct = VolcanicEruptionDetector().predict_eruption(record)
        score, flag, result = _served(VolcanicAdapter(), record)
        assert score == pytest.approx(direct.confidence)
        assert flag is direct.eruption_imminent
        assert result.alert_level == direct.alert_level

    def test_unrest_outranks_quiet(self) -> None:
        adapter = VolcanicAdapter()
        out = adapter.detect([self.UNREST, self.QUIET])
        assert out["scores"][0] > out["scores"][1]
        features = adapter.extract_features([self.UNREST, self.QUIET])
        names = VolcanicAdapter.FEATURE_NAMES
        assert features[0, names.index("gas_flux_anomaly")] == 1.0
        assert features[0, names.index("deformation_detected")] == 1.0
        assert features[1, names.index("gas_flux_anomaly")] == 0.0

    @pytest.mark.parametrize(
        "record",
        [{}, {"gass_data": {"so2_tons_per_day": 5000.0}}, {"gas_data": {"so2": 5000.0}}],
    )
    def test_empty_or_misspelt_record_is_refused(self, record: dict[str, Any]) -> None:
        with pytest.raises(InvalidObservationError):
            VolcanicAdapter().detect(record)


# ---------------------------------------------------------------------------
# Landslide
# ---------------------------------------------------------------------------


def _jodupala_rain_row(day: dt.date) -> list[float]:
    """CHIRPS daily rain at the Jodupala landslide cell, as the trained benchmark maps it."""
    z = np.load(FIXTURES / "hazard_training" / "landslide" / "chirps_cell_fixture.npz")
    i = int(np.flatnonzero(z["era2018_dates"] == day.toordinal())[0])
    mm = z["era2018_mm"]
    return [float(mm[i]) / 24.0, 24.0, float(mm[i - 7 : i].sum())] + [NAN] * 7


class TestLandslide:
    STORM = [40.0, 12.0, 200.0, NAN, NAN, NAN, 35.0, 95.0, 20.0, NAN]
    CALM = [0.0, 0.0, 0.0, 0.0, 3.0, 500.0, 10.0, 20.0, 0.0, NAN]

    @staticmethod
    def _direct(row: list[float]) -> Any:
        from omni_mercury_engine.detectors.geological.landslide import LandslideDetector

        sections = LandslideAdapter._SECTIONS
        record = {}
        for section, keys in sections.items():
            values = {
                k: row[LandslideAdapter.COLUMNS.index(k)]
                for k in keys
                if not np.isnan(row[LandslideAdapter.COLUMNS.index(k)])
            }
            if values:
                record[section] = values
        return LandslideDetector().predict_landslide(record)

    @staticmethod
    def _adapted(row: list[float]) -> tuple[float, bool, dict[str, float]]:
        adapter = LandslideAdapter()
        out = adapter.detect(np.array([row]))
        features = adapter.extract_features(np.array([row]))[0]
        return (
            float(out["scores"][0]),
            bool(out["flags"][0]),
            dict(zip(LandslideAdapter.FEATURE_NAMES, features.tolist(), strict=True)),
        )

    @pytest.mark.parametrize("row", [STORM, CALM])
    def test_matches_the_detector(self, row: list[float]) -> None:
        direct = self._direct(row)
        score, flag, features = self._adapted(row)
        assert score == pytest.approx(direct.slope_failure_probability)
        assert flag is direct.landslide_imminent
        assert features["confidence"] == pytest.approx(direct.confidence)

    def test_storm_on_a_saturated_steep_slope_is_imminent(self) -> None:
        out = LandslideAdapter().detect(np.array([self.STORM, self.CALM]))
        assert out["flags"].tolist() == [True, False]

    @pytest.mark.parametrize("day", [dt.date(2018, 8, 17), dt.date(2018, 2, 15)])
    def test_recorded_chirps_rain_matches_the_detector(self, day: dt.date) -> None:
        """Real rain on an unmonitored slope: the adapter serves the detector's view."""
        row = _jodupala_rain_row(day)
        direct = self._direct(row)
        score, flag, features = self._adapted(row)
        assert score == pytest.approx(direct.slope_failure_probability)
        assert flag is direct.landslide_imminent
        assert features["rainfall_trigger"] == float(direct.rainfall_trigger)

    def test_unmeasured_row_and_non_finite_values_are_refused(self) -> None:
        with pytest.raises(InvalidObservationError, match="no slope measured"):
            LandslideAdapter().detect(np.array([[NAN] * 10]))
        with pytest.raises(InvalidObservationError):
            LandslideAdapter().detect(np.array([[np.inf] + self.STORM[1:]]))

    def test_non_physical_value_is_set_aside_and_the_verdict_is_partial(self) -> None:
        """Regression: the adapter dropped the domain detector's partial marker.

        A 500 deg slope beside a measured storm is set aside by the domain
        detector and the storm is still graded; the served verdict must say
        it is partial, row by row.
        """
        bad = list(self.STORM)
        bad[LandslideAdapter.COLUMNS.index("slope_angle_deg")] = 500.0
        out = LandslideAdapter().detect(np.array([self.STORM, bad]))
        assert out["partial"] is True
        assert set(out["set_aside"]) == {1}
        assert out["set_aside"][1][0]["where"] == "slope_data.slope_angle_deg"
        clean = LandslideAdapter().detect(np.array([self.STORM]))
        assert "partial" not in clean and "set_aside" not in clean


# ---------------------------------------------------------------------------
# Wildfire
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def firms_patch() -> np.ndarray[Any, Any]:
    """A wildfire-firms-v1 raster around the hottest 2023-08-30 VIIRS detection."""
    from omni_mercury_engine.ml.hazard_training import wildfire_ignition as wf

    csv = FIXTURES / "hazard_training" / "wildfire" / "viirs_snpp_2023_ca_excerpt.csv"
    grids = wf.rasterize_daily([wf.parse_firms_csv(csv)])
    t = dt.date(2023, 8, 30).toordinal() - grids.day0
    cells = grids.days[t]
    j = int(np.argmax(cells.bt_max))
    return wf.build_patch(grids, t, int(cells.rows[j]), int(cells.cols[j]))


class TestWildfire:
    @staticmethod
    def _scene(fire_k: float | None) -> np.ndarray[Any, Any]:
        scene = np.full((3, 32, 32), 295.0)
        if fire_k is not None:
            scene[:, 14:16, 14:16] = fire_k
        return scene

    @pytest.mark.parametrize("fire_k", [380.0, None])
    def test_matches_the_detector(self, fire_k: float | None) -> None:
        from omni_mercury_engine.detectors.geological.wildfire import WildfireDetector

        record = {"thermal_image": self._scene(fire_k), "pixel_size_km": 0.375}
        direct = WildfireDetector().predict_wildfire(record)
        score, flag, result = _served(WildfireSceneAdapter(), record)
        assert score == pytest.approx(direct.confidence)
        assert flag is direct.fire_detected
        assert flag is (fire_k is not None)
        assert result.assessment == "active_fire"

    def test_recorded_firms_raster_is_a_next_day_forecast(
        self, firms_patch: np.ndarray[Any, Any]
    ) -> None:
        from omni_mercury_engine.detectors.geological.wildfire import WildfireDetector

        direct = WildfireDetector().predict_wildfire({"thermal_image": firms_patch})
        score, flag, result = _served(WildfireSceneAdapter(), {"thermal_image": firms_patch})
        assert result.assessment == "next_day_forecast"
        assert score == pytest.approx(direct.confidence)
        assert flag is direct.fire_detected

    def test_weather_only_or_misspelt_scene_is_refused(self) -> None:
        with pytest.raises(InvalidObservationError, match="missing"):
            WildfireSceneAdapter().detect({"weather_data": {"wind_speed_kmh": 40.0}})
        # A misspelt weather key beside a scene is set aside; the scene is graded.
        out = WildfireSceneAdapter().detect(
            {"thermal_image": self._scene(None), "weather_data": {"humidity": 5.0}}
        )
        assert out["partial"] is True
        assert out["set_aside"][0][0]["where"] == "weather_data.humidity"
        with pytest.raises(InputModalityError):
            WildfireSceneAdapter().detect({"thermal_image": "not a raster"})


# ---------------------------------------------------------------------------
# Avalanche
# ---------------------------------------------------------------------------


class TestAvalanche:
    SLAB = [
        {"thickness_m": 0.4, "density_kg_m3": 200.0, "temperature_c": -6.0},
        {
            "thickness_m": 0.02,
            "density_kg_m3": 250.0,
            "temperature_c": -5.0,
            "grain_type": "facets",
        },
        {"thickness_m": 0.8, "density_kg_m3": 320.0, "temperature_c": -2.0},
    ]
    LOADED: dict[str, Any] = {
        "layers": SLAB,
        "weak_layer_index": 1,
        "new_snow_24h_cm": 50.0,
        "wind_speed_10m_ms": 12.0,
    }
    CALM: dict[str, Any] = {"new_snow_24h_cm": 0.0, "wind_speed_10m_ms": 2.0}

    @pytest.mark.parametrize("record", [LOADED, CALM])
    def test_matches_the_detector_on_the_eaws_scale(self, record: dict[str, Any]) -> None:
        from omni_mercury_engine.detectors.geological.avalanche_detector import (
            AvalancheDetector,
            SnowLayer,
        )

        kwargs = dict(record)
        if "layers" in kwargs:
            kwargs["layers"] = [SnowLayer(**layer) for layer in kwargs["layers"]]
        direct = AvalancheDetector().predict_avalanche(**kwargs)
        score, flag, _ = _served(AvalancheProfileAdapter(), record)
        assert score == pytest.approx((direct.danger_level - 1) / 4)
        assert flag is direct.avalanche_likely

    def test_loaded_slab_is_considerable_or_worse(self) -> None:
        out = AvalancheProfileAdapter().detect([self.LOADED, self.CALM])
        assert out["flags"].tolist() == [True, False]
        assert out["scores"][0] >= 0.5  # EAWS 3, considerable

    def test_malformed_profile_is_refused(self) -> None:
        with pytest.raises(InvalidObservationError, match="snow layer"):
            AvalancheProfileAdapter().detect(
                {"layers": [{"thickness": 0.4}], "weak_layer_index": 0}
            )
        with pytest.raises(InputModalityError):
            AvalancheProfileAdapter().detect({"layers": "deep", "weak_layer_index": 0})
        with pytest.raises(InvalidObservationError, match="unknown keys"):
            AvalancheProfileAdapter().detect({"new_snow_cm": 50.0})


# ---------------------------------------------------------------------------
# Rockfall
# ---------------------------------------------------------------------------


class TestRockfall:
    @staticmethod
    def _accelerating() -> dict[str, Any]:
        """Fukuzono inverse-velocity approach to failure at t = 100 h."""
        t = np.linspace(0.0, 95.0, 60)
        return {"displacement_mm": -np.log(1.0 - t / 100.0) * 10.0, "displacement_time": t}

    @pytest.mark.parametrize(
        "record",
        [{"rain_mm": np.zeros(24)}, {"rain_mm": np.full(24, 8.0)}, None],
    )
    def test_matches_the_detector_ladder(self, record: dict[str, Any] | None) -> None:
        from omni_mercury_engine.detectors.geological.rockfall_detector import RockfallDetector

        record = record if record is not None else self._accelerating()
        direct = RockfallDetector().predict_rockfall(dict(record))
        score, flag, _ = _served(RockfallAdapter(), record)
        ladder = ("low", "moderate", "high", "critical")
        assert score == pytest.approx(ladder.index(direct.hazard_level) / 3)
        assert flag is (direct.hazard_level != "low")

    def test_decision_is_exactly_above_low(self) -> None:
        out = RockfallAdapter().detect([{"rain_mm": np.zeros(24)}, {"rain_mm": np.full(24, 8.0)}])
        assert out["flags"].tolist() == [False, True]
        assert out["scores"][0] == 0.0

    def test_no_channel_or_misspelt_channel_is_refused(self) -> None:
        with pytest.raises(InvalidObservationError):
            RockfallAdapter().detect({})
        with pytest.raises(InvalidObservationError, match="nothing usable: unknown keys"):
            RockfallAdapter().detect({"rainfall_mm": np.zeros(24)})

    def test_misspelt_channel_beside_a_measured_one_is_set_aside(self) -> None:
        """Regression: one unknown key refused a record with a measured rain channel."""
        out = RockfallAdapter().detect(
            [{"rain_mm": np.full(24, 8.0), "rainfall_mm": np.zeros(24)}, {"rain_mm": np.zeros(24)}]
        )
        assert out["flags"].tolist() == [True, False]  # the rain channel is still graded
        assert out["partial"] is True
        assert out["set_aside"] == {
            0: [{"where": "record[0].rainfall_mm", "reason": out["set_aside"][0][0]["reason"]}]
        }
        assert out["set_aside"][0][0]["reason"].startswith("unknown key; accepted:")


# ---------------------------------------------------------------------------
# Subsidence
# ---------------------------------------------------------------------------


class TestSubsidence:
    T = np.linspace(2018.0, 2024.0, 60)

    def _scene(self, velocity_mm_yr: float) -> dict[str, Any]:
        rng = np.random.default_rng(0)
        d = velocity_mm_yr * (self.T - self.T[0]) + rng.normal(0.0, 0.5, self.T.size)
        return {"displacements_mm": d[None, :], "epochs_years": self.T}

    @pytest.mark.parametrize("velocity", [-30.0, 0.0])
    def test_matches_the_detector(self, velocity: float) -> None:
        from omni_mercury_engine.detectors.geological.subsidence_detector import (
            SubsidenceDetector,
        )

        scene = self._scene(velocity)
        direct = SubsidenceDetector().analyze(**scene)
        score, flag, _ = _served(SubsidenceAdapter(), scene)
        assert score == pytest.approx(direct.confidence)
        assert flag is direct.anomaly_detected
        assert flag is (velocity != 0.0)

    def test_missing_epochs_and_non_finite_coherence_are_refused(self) -> None:
        scene = self._scene(-30.0)
        with pytest.raises(InvalidObservationError, match="missing"):
            SubsidenceAdapter().detect({"displacements_mm": scene["displacements_mm"]})
        with pytest.raises(InvalidObservationError, match="coherence"):
            SubsidenceAdapter().detect({**scene, "coherence": np.array([NAN])})


# ---------------------------------------------------------------------------
# Earthquake -> tsunami cascade
# ---------------------------------------------------------------------------


def _tohoku() -> dict[str, Any]:
    data = json.loads((FIXTURES / "geological" / "usgs_tohoku_2011_event.json").read_text())
    feature: dict[str, Any] = data["features"][0]
    return feature


def _tide(n: int = 400, tsunami_m: float = 0.0) -> np.ndarray[Any, Any]:
    t = np.arange(n) * 60.0
    series = 0.5 * np.sin(2.0 * np.pi * t / (12.42 * 3600.0))  # M2 tide
    if tsunami_m:
        onset = int(n * 0.85)
        series[onset:] += tsunami_m * np.sin(2.0 * np.pi * (t[onset:] - t[onset]) / 1200.0)
    return np.asarray(series)


class TestEqTsunamiCascade:
    def test_recorded_tohoku_reaches_watch_then_threat_on_a_dart_departure(self) -> None:
        adapter = EqTsunamiCascadeAdapter()
        watch = {"events": [_tohoku()], "offshore": True}
        tide = {**watch, "water_level_m": _tide(), "sampling_interval_s": 60.0}
        wave = {**watch, "water_level_m": _tide(tsunami_m=0.5), "sampling_interval_s": 60.0}
        out = adapter.detect([watch, tide, wave])
        assert out["scores"].tolist() == pytest.approx([2 / 3, 2 / 3, 1.0])
        assert out["flags"].tolist() == [True, True, True]
        stages = [r.stage for r in out["metadata"]["results"]]
        assert stages == ["tsunami_watch", "tsunami_watch", "tsunami_threat"]

    def test_matches_a_direct_cascade(self) -> None:
        from omni_mercury_engine.detectors.geological.eq_tsunami_cascade import (
            EqTsunamiCascadeDetector,
        )

        direct = EqTsunamiCascadeDetector()
        direct.process_event(_tohoku(), offshore=True)
        state = direct.confirm_water_level(_tide(tsunami_m=0.5), 60.0, run_spectral_analysis=False)
        _, _, served = _served(
            EqTsunamiCascadeAdapter(),
            {
                "events": [_tohoku()],
                "offshore": True,
                "water_level_m": _tide(tsunami_m=0.5),
                "sampling_interval_s": 60.0,
            },
        )
        assert served.stage == state.stage
        assert served.water_level is not None and state.water_level is not None
        assert served.water_level.max_residual_m == pytest.approx(state.water_level.max_residual_m)

    def test_each_record_is_a_fresh_cascade(self) -> None:
        """A watch in one record never leaks into the next."""
        small = {"events": [{"magnitude": 5.0, "depth_km": 10.0}]}
        out = EqTsunamiCascadeAdapter().detect([{"events": [_tohoku()], "offshore": True}, small])
        assert out["flags"].tolist() == [True, False]

    def test_water_level_without_a_watch_is_set_aside_and_the_events_graded(self) -> None:
        """Regression: a tide reading with no watch refused the whole record.

        The earthquake is still graded; the water level, which has no watch
        to confirm, is set aside with that reason and the verdict is partial.
        """
        record = {
            "events": [{"magnitude": 5.0, "depth_km": 10.0}],
            "water_level_m": _tide(),
            "sampling_interval_s": 60.0,
        }
        out = EqTsunamiCascadeAdapter().detect(record)
        assert out["metadata"]["results"][0].stage != "tsunami_watch"
        assert out["partial"] is True
        (entry,) = out["set_aside"][0]
        assert entry["where"] == "water_level_m"
        assert "no watch to confirm" in entry["reason"]

    def test_half_a_gauge_series_is_set_aside_and_the_watch_still_served(self) -> None:
        out = EqTsunamiCascadeAdapter().detect(
            {"events": [_tohoku()], "offshore": True, "water_level_m": _tide()}
        )
        assert out["metadata"]["results"][0].stage == "tsunami_watch"
        (entry,) = out["set_aside"][0]
        assert entry["where"] == "water_level_m" and "come together" in entry["reason"]

    def test_unusable_gauge_series_is_set_aside_and_the_watch_still_served(self) -> None:
        tide = _tide()
        tide[5] = float("nan")
        out = EqTsunamiCascadeAdapter().detect(
            {
                "events": [_tohoku()],
                "offshore": True,
                "water_level_m": tide,
                "sampling_interval_s": 60.0,
            }
        )
        assert out["metadata"]["results"][0].stage == "tsunami_watch"
        assert out["partial"] is True
        assert out["set_aside"][0][0]["where"] == "water_level_m"

    def test_bad_events_are_refused(self) -> None:
        with pytest.raises(InputModalityError):
            EqTsunamiCascadeAdapter().detect({"events": []})
        with pytest.raises(InvalidObservationError, match="offshore"):
            EqTsunamiCascadeAdapter().detect({"events": [_tohoku()], "offshore": "yes"})


# ---------------------------------------------------------------------------
# Wildfire -> debris flow cascade
# ---------------------------------------------------------------------------


class TestFireDebrisFlow:
    BASIN = {"t_proportion": 0.6, "dnbr_mean": 0.6, "kf_factor": 0.3}
    STORM = [0.0, 1.0, 2.0, 8.0, 8.0, 8.0, 2.0, 1.0, 0.0]  # 24 mm / 15 min peak

    @pytest.mark.parametrize("rain", [STORM, [0.0, 0.1, 0.0]])
    def test_matches_the_staley_m1_detector(self, rain: list[float]) -> None:
        from omni_mercury_engine.detectors.geological.fire_debris_flow_cascade import (
            FireDebrisFlowCascadeDetector,
        )

        record = {**self.BASIN, "rain_mm": rain, "step_minutes": 5.0}
        direct = FireDebrisFlowCascadeDetector().assess(
            **self.BASIN, rain_mm=np.asarray(rain), step_minutes=5.0
        )
        score, flag, _ = _served(FireDebrisFlowAdapter(), record)
        assert score == pytest.approx(direct.likelihood)
        assert flag is direct.threshold_exceeded

    def test_configured_duration_and_region_reach_the_detector(self) -> None:
        adapter = FireDebrisFlowAdapter({"primary_duration_min": 30, "cannon_region": "colorado"})
        _, _, result = _served(adapter, {**self.BASIN, "rain_mm": self.STORM, "step_minutes": 5.0})
        assert result.primary_duration_min == 30
        with pytest.raises(ValueError, match="cannon_region"):
            FireDebrisFlowAdapter({"cannon_region": "iceland"}).detect(
                {**self.BASIN, "rain_mm": self.STORM, "step_minutes": 5.0}
            )

    def test_missing_basin_term_is_refused(self) -> None:
        with pytest.raises(InvalidObservationError, match="missing"):
            FireDebrisFlowAdapter().detect({"rain_mm": self.STORM, "step_minutes": 5.0})


# ---------------------------------------------------------------------------
# Hurricane -> surge -> compound flood cascade
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def battery() -> tuple[Any, Any]:
    """Recorded NOAA CO-OPS water level and predictions at The Battery, NY."""
    coops = FIXTURES / "coops"
    observed = json.loads((coops / "observed_8518750.json").read_text())
    predicted = json.loads((coops / "predicted_8518750.json").read_text())
    return observed, predicted


class TestSurgeFloodCascade:
    HURRICANE = {
        "pressure_data": {"central_pressure_mb": 940.0, "environmental_pressure_mb": 1010.0},
        "sst_data": {"sst_celsius": 29.0},
    }

    def test_real_battery_levels_with_and_without_a_hurricane(self, battery: Any) -> None:
        observed, predicted = battery
        surge = {"coops_observed": observed, "coops_predicted": predicted, "station_id": "8518750"}
        out = SurgeFloodCascadeAdapter().detect([{**surge, "hurricane": self.HURRICANE}, surge])
        # A calm day at The Battery: the hurricane opens WATCH, the surge gate
        # stays shut; without the hurricane the chain never leaves QUIET.
        assert out["scores"].tolist() == pytest.approx([1 / 3, 0.0])
        assert out["flags"].tolist() == [True, False]

    def test_partial_hurricane_evidence_keeps_the_cascade_verdict_partial(
        self, battery: Any
    ) -> None:
        """Regression: the cascade's served result dropped the hurricane's set-aside input."""
        observed, predicted = battery
        surge = {"coops_observed": observed, "coops_predicted": predicted, "station_id": "8518750"}
        kelvin_sst = {**self.HURRICANE, "sst_data": {"sst_celsius": 302.15}}
        out = SurgeFloodCascadeAdapter().detect([{**surge, "hurricane": kelvin_sst}, surge])
        assert out["flags"].tolist() == [True, False]  # the pressure deficit still decides
        assert out["partial"] is True
        assert set(out["set_aside"]) == {0}
        assert out["set_aside"][0][0]["where"] == "hurricane.sst_data.sst_celsius"

    def test_matches_a_direct_cascade(self, battery: Any) -> None:
        from omni_mercury_engine.detectors.geological.hurricane_detector import HurricaneDetector
        from omni_mercury_engine.detectors.meteorological.surge_flood_cascade import (
            SurgeFloodCascade,
        )

        observed, predicted = battery
        cascade = SurgeFloodCascade()
        cascade.update_hurricane_evidence(HurricaneDetector().predict_hurricane(self.HURRICANE))
        cascade.update_surge_evidence(observed, predicted, "8518750")
        direct = cascade.evaluate()
        _, _, served = _served(
            SurgeFloodCascadeAdapter(),
            {
                "hurricane": self.HURRICANE,
                "coops_observed": observed,
                "coops_predicted": predicted,
                "station_id": "8518750",
            },
        )
        assert served.stage is direct.stage
        assert served.surge is not None and direct.surge is not None
        assert served.surge.max_residual_m == pytest.approx(direct.surge.max_residual_m)

    def test_record_without_evidence_is_refused(self) -> None:
        with pytest.raises(InvalidObservationError, match="needs evidence"):
            SurgeFloodCascadeAdapter().detect({"station_id": "8518750"})

    def test_half_a_coops_pair_or_corrupt_level_is_refused(self, battery: Any) -> None:
        observed, _ = battery
        with pytest.raises(InvalidObservationError, match="come together"):
            SurgeFloodCascadeAdapter().detect({"coops_observed": observed})
        corrupt = json.loads(json.dumps(observed))
        corrupt["data"][0]["v"] = "inf"
        with pytest.raises(InvalidObservationError, match="non-finite"):
            SurgeFloodCascadeAdapter().detect(
                {"coops_observed": corrupt, "coops_predicted": battery[1]}
            )


# ---------------------------------------------------------------------------
# Marine ecosystem
# ---------------------------------------------------------------------------


class TestMarineEcosystem:
    def test_matches_the_detector_and_its_health_ladder(self) -> None:
        from omni_mercury_engine.detectors.marine.biodiversity_detector import (
            MarineBiodiversityDetector,
        )

        rows = np.array([[1.5, 9.0, 7.8, 2.5], [NAN, NAN, 8.1, NAN], [0.2, 0.5, NAN, NAN]])
        out = MarineEcosystemAdapter().detect(rows)
        direct = MarineBiodiversityDetector()
        expected = [
            direct.predict_biodiversity_threat(
                {
                    "coral_data": {"sst_anomaly_c": 1.5, "degree_heating_weeks": 9.0},
                    "chemistry_data": {"ph": 7.8},
                    "temperature_data": {"anomaly_c": 2.5},
                }
            ),
            direct.predict_biodiversity_threat({"chemistry_data": {"ph": 8.1}}),
            direct.predict_biodiversity_threat(
                {"coral_data": {"sst_anomaly_c": 0.2, "degree_heating_weeks": 0.5}}
            ),
        ]
        ladder = ("healthy", "stressed", "critical")
        assert out["scores"].tolist() == pytest.approx(
            [ladder.index(r.health_status) / 2 for r in expected]
        )
        assert out["flags"].tolist() == [r.ecosystem_threatened for r in expected]
        assert out["flags"].tolist() == [True, False, False]

    def test_decision_is_exactly_score_at_least_half(self) -> None:
        rows = np.array([[NAN, NAN, 7.7, NAN], [NAN, NAN, 7.7, 3.0], [NAN, NAN, 8.1, 0.5]])
        out = MarineEcosystemAdapter().detect(rows)
        assert out["flags"].tolist() == (out["scores"] >= 0.5).tolist()

    def test_unmeasured_row_or_half_coral_pair_is_refused(self) -> None:
        with pytest.raises(InvalidObservationError, match="no indicator measured"):
            MarineEcosystemAdapter().detect(np.array([[NAN] * 4]))
        with pytest.raises(InvalidObservationError, match="missing"):
            MarineEcosystemAdapter().detect(np.array([[1.5, NAN, NAN, NAN]]))
