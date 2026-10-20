# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioral tests for the enhanced geological detectors (3R integration).

Covers, with exact value/boundary pins so the module mutation floor has teeth:
- LandslideDetector: rainfall/seismic trigger models, the geotechnical physics
  stability assessment, the failure-type ladder, risk/zone/warning/cascade logic,
  the trained-checkpoint neural path and its input-contract guards, and the
  3R Recursion multi-scale machinery.
- WildfireDetector: NFDRS fuel moisture (EMC + fuel-interface adjustment), the
  wind-driven spread model, NDVI fuel/drought/burn metrics, the
  brightness-temperature physics detection, the FIRMS raster contract, the
  integer-point risk ladder, and the 3R Resonance analyzer.
- VolcanicEruptionDetector: thermal/gas/InSAR component thresholds, the
  activity-state HMM (forward update, eruption probability, reset), the 3R
  Refactoring optimizer, the physics swarm/forecast paths, the USGS-style
  alert ladder, and zone/warning/evacuation generation.

Two latent defects of the previous suite are fixed here rather than preserved:
the HMM tests passed a dict where ``update_belief`` takes a binary observation
array (iterating dict KEYS, all truthy), and the volcanic high/low-risk tests
passed ``gas_emissions``/``deformation_mm`` where ``predict_eruption`` reads
``gas_data``/``insar_data`` (so gas and deformation were never exercised).
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.core.exceptions import InputModalityError, InvalidObservationError

# Optional torch import
HAS_TORCH = importlib.util.find_spec("torch") is not None

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


# =============================================================================
# Landslide: trigger models and recursion machinery
# =============================================================================


class TestRainfallTriggerModel:
    """Intensity-duration threshold (I*sqrt(D) vs 10*sqrt(6)) and severity ladder."""

    @pytest.fixture(scope="class")
    @classmethod
    def model(cls) -> Any:
        from omni_mercury_engine.detectors.geological.landslide import RainfallTriggerModel

        return RainfallTriggerModel()

    def test_trigger_boundary_is_strict(self, model: Any) -> None:
        # At exactly the critical intensity-duration product the trigger is OFF.
        at = model.assess_rainfall_trigger({"intensity_mm_hr": 10.0, "duration_hours": 6.0})
        assert at["rainfall_trigger"] is False
        over = model.assess_rainfall_trigger({"intensity_mm_hr": 10.1, "duration_hours": 6.0})
        assert over["rainfall_trigger"] is True

    def test_probability_scales_with_id_ratio(self, model: Any) -> None:
        half = model.assess_rainfall_trigger({"intensity_mm_hr": 5.0, "duration_hours": 6.0})
        assert half["trigger_probability"] == pytest.approx(0.5)
        assert half["id_threshold"] == pytest.approx(5.0 * 6.0**0.5)

    def test_antecedent_rain_boosts_probability(self, model: Any) -> None:
        boosted = model.assess_rainfall_trigger(
            {"intensity_mm_hr": 5.0, "duration_hours": 6.0, "antecedent_7day_mm": 50.0}
        )
        assert boosted["trigger_probability"] == pytest.approx(0.75)

    def test_probability_caps_at_one(self, model: Any) -> None:
        capped = model.assess_rainfall_trigger(
            {"intensity_mm_hr": 5.0, "duration_hours": 6.0, "antecedent_7day_mm": 200.0}
        )
        assert capped["trigger_probability"] == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("intensity", "severity"),
        [(8.5, "extreme"), (7.0, "high"), (5.0, "moderate"), (3.0, "low")],
    )
    def test_severity_ladder(self, model: Any, intensity: float, severity: str) -> None:
        result = model.assess_rainfall_trigger(
            {"intensity_mm_hr": intensity, "duration_hours": 6.0}
        )
        assert result["severity"] == severity

    def test_no_rain_is_low_and_untriggered(self, model: Any) -> None:
        result = model.assess_rainfall_trigger({})
        assert result["rainfall_trigger"] is False
        assert result["trigger_probability"] == 0.0
        assert result["severity"] == "low"


class TestSeismicTriggerModel:
    """PGA threshold (0.15 g, strict) and magnitude/distance scaling."""

    @pytest.fixture(scope="class")
    @classmethod
    def model(cls) -> Any:
        from omni_mercury_engine.detectors.geological.landslide import SeismicTriggerModel

        return SeismicTriggerModel()

    def test_pga_threshold_is_strict(self, model: Any) -> None:
        assert model.assess_seismic_trigger({"pga_g": 0.15})["seismic_trigger"] is False
        assert model.assess_seismic_trigger({"pga_g": 0.16})["seismic_trigger"] is True

    def test_probability_baseline(self, model: Any) -> None:
        # ratio 0.5, magnitude factor 0 (M5), distance factor 0 (50 km).
        result = model.assess_seismic_trigger(
            {"pga_g": 0.075, "magnitude": 5.0, "distance_km": 50.0}
        )
        assert result["trigger_probability"] == pytest.approx(0.5)

    def test_magnitude_and_proximity_compound_then_cap(self, model: Any) -> None:
        result = model.assess_seismic_trigger({"pga_g": 0.075, "magnitude": 8.0, "distance_km": 0})
        # 0.5 * (1 + 1) * (1 + 1) = 2.0, capped at 1.0.
        assert result["trigger_probability"] == pytest.approx(1.0)

    def test_small_magnitude_and_far_distance_add_nothing(self, model: Any) -> None:
        result = model.assess_seismic_trigger(
            {"pga_g": 0.03, "magnitude": 4.0, "distance_km": 100.0}
        )
        assert result["trigger_probability"] == pytest.approx(0.2)


class TestRecursionMultiScaleAnalyzer:
    """Golden-ratio scale weighting and the 8-feature scale statistics."""

    @pytest.fixture(scope="class")
    @classmethod
    def analyzer(cls) -> Any:
        from omni_mercury_engine.detectors.geological.landslide import (
            RecursionMultiScaleAnalyzer,
        )

        return RecursionMultiScaleAnalyzer()

    def test_default_scales_and_normalized_decaying_weights(self, analyzer: Any) -> None:
        assert analyzer.scales == [1, 4, 16, 64]
        weights = analyzer._scale_weights
        assert weights.sum() == pytest.approx(1.0)
        assert all(weights[i] > weights[i + 1] for i in range(len(weights) - 1))

    def test_scale_features_exact_statistics(self, analyzer: Any) -> None:
        feats = analyzer._compute_scale_features(np.array([1.0, 2.0, 3.0]), scale=1)
        assert feats[0] == pytest.approx(2.0)  # mean
        assert feats[1] == pytest.approx(np.sqrt(2.0 / 3.0))  # std
        assert feats[2] == pytest.approx(2.0)  # range
        assert feats[3] == pytest.approx(1.6)  # p90 - p10
        assert feats[4] == pytest.approx(1.0)  # mean diff
        assert feats[5] == pytest.approx(0.0)  # diff volatility
        assert feats[6] == pytest.approx(1.0)  # fraction increasing
        assert feats[7] == pytest.approx(1.0)  # max |diff|

    def test_scale_features_too_short_is_zero_vector(self, analyzer: Any) -> None:
        assert np.array_equal(analyzer._compute_scale_features(np.array([5.0]), 1), np.zeros(8))

    def test_multi_scale_output_shape_and_bounds(self, analyzer: Any) -> None:
        rng = np.random.default_rng(0)
        features = analyzer.extract_multi_scale_features(rng.normal(size=500))
        for scale in (1, 4, 16, 64):
            assert f"scale_{scale}h" in features
        assert 0.0 <= features["recursion_score"] <= 1.0


class TestTemporalLagFeatureExtractor:
    """Lagged value, cumulative sum, and lag correlation per configured lag."""

    def test_default_lags(self) -> None:
        from omni_mercury_engine.detectors.geological.landslide import (
            TemporalLagFeatureExtractor,
        )

        assert TemporalLagFeatureExtractor().lag_hours == [1, 6, 12, 24, 48, 72]

    def test_one_hour_lag_features_exact(self) -> None:
        from omni_mercury_engine.detectors.geological.landslide import (
            TemporalLagFeatureExtractor,
        )

        extractor = TemporalLagFeatureExtractor()
        series = np.arange(8000, dtype=float)
        feats = extractor.extract_lag_features(series, sample_rate_hz=1.0)
        # Only the 1 h lag (3600 samples) fits an 8000-sample record.
        assert feats[0] == pytest.approx(4400.0)  # series[-3600]
        assert feats[1] == pytest.approx((4400 + 7999) * 3600 / 2)  # cumulative sum
        assert feats[2] == pytest.approx(1.0)  # linear series: perfect lag correlation
        # The 6 h+ lags exceed the record and stay zero.
        assert np.array_equal(feats[3:], np.zeros(15))

    def test_series_shorter_than_every_lag_is_all_zero(self) -> None:
        from omni_mercury_engine.detectors.geological.landslide import (
            TemporalLagFeatureExtractor,
        )

        extractor = TemporalLagFeatureExtractor()
        feats = extractor.extract_lag_features(np.arange(10, dtype=float), sample_rate_hz=1.0)
        assert np.array_equal(feats, np.zeros(18))


class TestSVMRFEnsembleClassifier:
    """Unfitted default probabilities and the 0.4/0.6 fitted ensemble."""

    def test_unfitted_predicts_uniform(self) -> None:
        from omni_mercury_engine.detectors.geological.landslide import SVMRFEnsembleClassifier

        clf = SVMRFEnsembleClassifier()
        assert clf.is_fitted is False
        assert clf.ensemble_weights == (0.4, 0.6)
        proba = clf.predict_proba(np.zeros((3, 2)))
        assert np.array_equal(proba, np.full((3, 2), 0.5))
        assert clf.get_feature_importance().size == 0

    def test_fitted_separable_classification(self) -> None:
        from omni_mercury_engine.detectors.geological.landslide import SVMRFEnsembleClassifier

        clf = SVMRFEnsembleClassifier()
        X = np.array([[0.0], [0.1], [0.2], [5.0], [5.1], [5.2]])
        y = np.array([0, 0, 0, 1, 1, 1])
        assert clf.fit(X, y) is clf
        assert clf.is_fitted is True
        proba = clf.predict_proba(np.array([[5.1], [0.05]]))
        assert proba[0, 1] > 0.5  # clear class-1 point
        assert proba[1, 0] > 0.5  # clear class-0 point
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        assert clf.get_feature_importance().shape == (1,)


# =============================================================================
# Landslide: detector-level behavior
# =============================================================================


class TestLandslideDetector:
    """Physics stability math, failure-type ladder, alert and risk logic."""

    @pytest.fixture(scope="class")
    @classmethod
    def detector(cls) -> Any:
        from omni_mercury_engine.detectors.geological.landslide import LandslideDetector

        return LandslideDetector(enable_recursion=True)

    def test_detector_initialization(self, detector: Any) -> None:
        assert detector.enable_recursion is True
        assert detector.recursion_analyzer is not None
        assert detector.lag_extractor is not None
        # The shipped landslide_coolr checkpoint loads at construction.
        assert detector._neural_trained is True

    def test_physics_probability_geotechnical_term(self, detector: Any) -> None:
        result = detector.predict_landslide(
            {
                "slope_data": {"slope_angle_deg": 30.0},
                "sensor_data": {"soil_saturation_pct": 50.0, "displacement_rate_mm_day": 0.0},
            }
        )
        # slope severity (30-15)/30 = 0.5 times saturation 0.5 -> 0.25.
        assert result.slope_failure_probability == pytest.approx(0.25)
        assert result.landslide_type == "rotational_slide"
        assert result.confidence == pytest.approx(0.25)
        assert result.risk_level == "moderate"
        assert result.landslide_imminent is False

    def test_physics_probability_displacement_term(self, detector: Any) -> None:
        result = detector.predict_landslide({"sensor_data": {"displacement_rate_mm_day": 25.0}})
        # displacement severity 25/50 = 0.5, no geotechnical term.
        assert result.slope_failure_probability == pytest.approx(0.5)
        # No triggers, probability > 0.4 -> high.
        assert result.risk_level == "high"
        assert result.evacuation_zones == []

    def test_physics_probability_saturates(self, detector: Any) -> None:
        result = detector.predict_landslide(
            {
                "slope_data": {"slope_angle_deg": 45.0},
                "sensor_data": {"soil_saturation_pct": 100.0, "displacement_rate_mm_day": 50.0},
            }
        )
        assert result.slope_failure_probability == pytest.approx(1.0)

    def test_no_data_is_low_risk_nothing_imputed(self, detector: Any) -> None:
        result = detector.predict_landslide({})
        assert result.slope_failure_probability == 0.0
        assert result.risk_level == "low"
        assert result.landslide_imminent is False
        assert result.early_warning_actions == []
        assert result.cascade_risks == []

    def test_imminent_requires_a_trigger_and_probability(self, detector: Any) -> None:
        # High probability alone (no trigger) must not fire the alert.
        no_trigger = detector.predict_landslide(
            {
                "slope_data": {"slope_angle_deg": 45.0},
                "sensor_data": {"soil_saturation_pct": 90.0, "displacement_rate_mm_day": 40.0},
            }
        )
        assert no_trigger.slope_failure_probability > 0.9
        assert no_trigger.landslide_imminent is False

        triggered = detector.predict_landslide(
            {
                "rainfall_data": {"intensity_mm_hr": 20.0, "duration_hours": 6.0},
                "slope_data": {"slope_angle_deg": 45.0},
                "sensor_data": {"soil_saturation_pct": 90.0, "displacement_rate_mm_day": 40.0},
            }
        )
        assert triggered.rainfall_trigger is True
        assert triggered.landslide_imminent is True
        assert triggered.risk_level == "very_high"
        assert "immediate_downslope_area" in triggered.evacuation_zones
        assert "potential_runout_path" in triggered.evacuation_zones
        assert triggered.early_warning_actions[0] == "LANDSLIDE WATCH: Prepare for evacuation"

    def test_two_triggers_and_high_probability_is_extreme(self, detector: Any) -> None:
        # Slope 30 deg stays below the 35 deg rock-slide rule, so the
        # rainfall-on-unsaturated-ground branch classifies a debris flow.
        result = detector.predict_landslide(
            {
                "rainfall_data": {"intensity_mm_hr": 20.0, "duration_hours": 6.0},
                "seismic_data": {"pga_g": 0.3, "magnitude": 7.0, "distance_km": 10.0},
                "slope_data": {"slope_angle_deg": 30.0},
                "sensor_data": {"soil_saturation_pct": 70.0, "displacement_rate_mm_day": 40.0},
                "infrastructure_data": {"dam_present": True},
                "population_data": {"population_density": 200},
            }
        )
        assert result.seismic_trigger is True
        # slope severity 0.5 * saturation 0.7 = 0.35; displacement 0.8:
        # 1 - 0.2*0.65 = 0.87 > 0.8 with two triggers -> extreme.
        assert result.slope_failure_probability == pytest.approx(0.87)
        assert result.risk_level == "extreme"
        assert result.early_warning_actions[0] == "LANDSLIDE WARNING: Immediate evacuation required"
        # Saturation 0.7 < 0.8 with a rainfall trigger -> debris flow,
        # which adds the drainage-channel zone and the river-blockage cascade.
        assert result.landslide_type == "debris_flow"
        assert "drainage_channels" in result.evacuation_zones
        assert set(result.cascade_risks) == {
            "dam_failure_risk",
            "river_blockage_flooding",
            "high_casualty_potential",
        }

    def test_snowmelt_trigger_boundary_and_half_weight(self, detector: Any) -> None:
        at = detector.predict_landslide({"weather_data": {"snowmelt_mm_day": 20.0}})
        assert at.snowmelt_trigger is False
        over = detector.predict_landslide(
            {
                "weather_data": {"snowmelt_mm_day": 25.0},
                "slope_data": {"slope_angle_deg": 45.0},
                "sensor_data": {"soil_saturation_pct": 100.0, "displacement_rate_mm_day": 50.0},
            }
        )
        assert over.snowmelt_trigger is True
        assert over.landslide_type == "snow_avalanche"
        # Snowmelt counts only half a trigger: probability 1.0 but no alert.
        assert over.slope_failure_probability == pytest.approx(1.0)
        assert over.landslide_imminent is False

    def test_failure_type_ladder(self, detector: Any) -> None:
        rock = detector.predict_landslide(
            {
                "seismic_data": {"pga_g": 0.3, "magnitude": 7.0, "distance_km": 10.0},
                "slope_data": {"slope_angle_deg": 35.0},
            }
        )
        assert rock.landslide_type == "rock_slide"
        shallow = detector.predict_landslide(
            {
                "seismic_data": {"pga_g": 0.3, "magnitude": 7.0, "distance_km": 10.0},
                "slope_data": {"slope_angle_deg": 34.0},
            }
        )
        assert shallow.landslide_type == "rotational_slide"
        mud = detector.predict_landslide(
            {
                "rainfall_data": {"intensity_mm_hr": 20.0, "duration_hours": 6.0},
                "sensor_data": {"soil_saturation_pct": 80.0},
            }
        )
        assert mud.landslide_type == "mud_flow"
        earth = detector.predict_landslide({"sensor_data": {"soil_saturation_pct": 60.0}})
        assert earth.landslide_type == "earth_flow"

    def test_trained_neural_path_contract(self, detector: Any) -> None:
        expected_width = detector.stability_model.feature_encoder[0].in_features
        features = np.zeros(expected_width)
        result = detector.predict_landslide({"slope_features": features})
        assert 0.0 <= result.slope_failure_probability <= 1.0
        # The already-batched (1, W) spelling is the same single sample.
        batched = detector.predict_landslide({"slope_features": features.reshape(1, -1)})
        assert batched.slope_failure_probability == pytest.approx(result.slope_failure_probability)

    def test_off_contract_features_fall_back_to_physics(self, detector: Any) -> None:
        result = detector.predict_landslide({"slope_features": np.zeros(7)})
        # Physics with no geotechnical fields: probability exactly 0.
        assert result.slope_failure_probability == 0.0
        assert result.landslide_type == "rotational_slide"

    def test_diagnostics_only_on_neural_path(self) -> None:
        from omni_mercury_engine.detectors.geological.landslide import LandslideDetector

        detector = LandslideDetector(keep_diagnostics=True)
        assert detector.stability_model is not None
        width = detector.stability_model.feature_encoder[0].in_features
        neural = detector.predict_landslide({"slope_features": np.zeros(width)})
        assert neural.diagnostics is not None
        probs = neural.diagnostics.arrays["failure_type_probs"]
        assert probs.shape == (6,)
        assert probs.sum() == pytest.approx(1.0)
        physics = detector.predict_landslide({"sensor_data": {"soil_saturation_pct": 50.0}})
        assert physics.diagnostics is None


class TestLandslideDetectorWithoutRecursion:
    """Tests for LandslideDetector without 3R Recursion."""

    @pytest.fixture
    def detector_no_recursion(self) -> Any:
        from omni_mercury_engine.detectors.geological.landslide import LandslideDetector

        return LandslideDetector(enable_recursion=False)

    def test_detector_without_recursion(self, detector_no_recursion: Any) -> None:
        assert detector_no_recursion.enable_recursion is False
        assert detector_no_recursion.recursion_analyzer is None
        assert detector_no_recursion.lag_extractor is None

    def test_predict_without_recursion(self, detector_no_recursion: Any) -> None:
        result = detector_no_recursion.predict_landslide(
            {"sensor_data": {"displacement_rate_mm_day": 25.0}}
        )
        assert result.slope_failure_probability == pytest.approx(0.5)


# =============================================================================
# Wildfire: NFDRS fuel moisture, spread, NDVI, resonance
# =============================================================================


class TestNFDRSFuelMoisture:
    """Simard EMC regressions and the fuel-interface adjustment (GTR INT-169)."""

    def test_emc_low_humidity_branch(self) -> None:
        from omni_mercury_engine.detectors.geological.wildfire import (
            equilibrium_moisture_content,
        )

        # T=30 C (86 F), h=5%: 0.03299 + 0.281073*5 - 0.000578*5*86.
        assert equilibrium_moisture_content(30.0, 5.0) == pytest.approx(1.189815, abs=1e-6)

    def test_emc_mid_humidity_branch(self) -> None:
        from omni_mercury_engine.detectors.geological.wildfire import (
            equilibrium_moisture_content,
        )

        # T=20 C (68 F), h=30%: 2.22749 + 0.160107*30 - 0.01478*68.
        assert equilibrium_moisture_content(20.0, 30.0) == pytest.approx(6.02566, abs=1e-5)

    def test_emc_high_humidity_branch(self) -> None:
        from omni_mercury_engine.detectors.geological.wildfire import (
            equilibrium_moisture_content,
        )

        # T=20 C (68 F), h=80%: 21.0606 + 0.005565*6400 - 0.00035*80*68 - 0.483199*80.
        assert equilibrium_moisture_content(20.0, 80.0) == pytest.approx(16.11668, abs=1e-5)

    def test_emc_branch_boundaries(self) -> None:
        from omni_mercury_engine.detectors.geological.wildfire import _simard_emc

        # h=10 goes to the mid regression (strict h < 10), h=50 to the high one.
        assert _simard_emc(68.0, 10.0) == pytest.approx(2.22749 + 1.60107 - 0.01478 * 68.0)
        assert _simard_emc(68.0, 50.0) == pytest.approx(
            21.0606 + 0.005565 * 2500 - 0.00035 * 50 * 68 - 0.483199 * 50
        )

    def test_emc_rejects_out_of_range(self) -> None:
        from omni_mercury_engine.detectors.geological.wildfire import (
            equilibrium_moisture_content,
        )

        with pytest.raises(InvalidObservationError):
            equilibrium_moisture_content(61.0, 50.0)
        with pytest.raises(InvalidObservationError):
            equilibrium_moisture_content(20.0, 101.0)
        with pytest.raises(InvalidObservationError):
            equilibrium_moisture_content(float("nan"), 50.0)

    @pytest.mark.parametrize(
        ("code", "add_f", "humidity_factor"),
        [(0, 25.0, 0.75), (1, 19.0, 0.83), (2, 12.0, 0.91), (3, 5.0, 1.0)],
    )
    def test_fuel_interface_adjustment_per_code(
        self, code: int, add_f: float, humidity_factor: float
    ) -> None:
        from omni_mercury_engine.detectors.geological.wildfire import fuel_interface_moisture

        out = fuel_interface_moisture(20.0, 50.0, code)
        assert out["fuel_interface_temperature_c"] == pytest.approx(20.0 + add_f * 5.0 / 9.0)
        assert out["fuel_interface_relative_humidity_pct"] == pytest.approx(50.0 * humidity_factor)

    def test_fuel_interface_emc_code2_exact(self) -> None:
        from omni_mercury_engine.detectors.geological.wildfire import fuel_interface_moisture

        out = fuel_interface_moisture(20.0, 50.0, 2)
        # Interface: 80 F, 45.5% -> mid regression.
        assert out["equilibrium_moisture_pct"] == pytest.approx(
            2.22749 + 0.160107 * 45.5 - 0.01478 * 80.0
        )

    # bad_code is deliberately off-type (1.5): the function must refuse
    # non-integer state-of-weather codes at runtime, so the parameter is
    # typed Any rather than the signature's int.
    @pytest.mark.parametrize("bad_code", [4, -1, 1.5])
    def test_invalid_state_of_weather_refused(self, bad_code: Any) -> None:
        from omni_mercury_engine.detectors.geological.wildfire import fuel_interface_moisture

        with pytest.raises(InvalidObservationError):
            fuel_interface_moisture(20.0, 50.0, bad_code)


class TestFireSpreadModel:
    """Wind-gated spread with exact factor arithmetic and the 0.1 km/h floor."""

    @pytest.fixture(scope="class")
    @classmethod
    def model(cls) -> Any:
        from omni_mercury_engine.detectors.geological.wildfire import FireSpreadModel

        return FireSpreadModel()

    def test_no_wind_measurement_means_no_spread_prediction(self, model: Any) -> None:
        out = model.predict_spread({}, {})
        assert out["spread_rate_km_hr"] is None
        assert out["spread_direction_deg"] is None
        assert out["wind_driven"] is False
        assert out["missing_inputs"] == ["wind_speed_kmh"]

    def test_full_inputs_exact_rate(self, model: Any) -> None:
        out = model.predict_spread(
            {"fuel_load_tons_ha": 20.0, "fuel_moisture_pct": 10.0},
            {
                "wind_speed_kmh": 20.0,
                "wind_direction_deg": 90.0,
                "relative_humidity_pct": 50.0,
                "temperature_c": 20.0,
            },
        )
        # 0.5 * 2.0 (wind) * 0.75 (RH) * 1.0 (T) * 1.5 (fuel) * 0.9 (moisture).
        assert out["spread_rate_km_hr"] == pytest.approx(1.0125)
        assert out["spread_direction_deg"] == 90.0
        assert out["wind_driven"] is False  # 20 km/h is not > 20
        assert out["missing_inputs"] == []

    def test_defaults_fill_missing_factors_but_never_direction(self, model: Any) -> None:
        out = model.predict_spread({}, {"wind_speed_kmh": 20.0})
        # Defaults: RH 50 -> 0.75, T 25 -> 1.1, fuel 20 -> 1.5, moisture 10 -> 0.9.
        assert out["spread_rate_km_hr"] == pytest.approx(0.5 * 2.0 * 0.75 * 1.1 * 1.5 * 0.9)
        assert out["spread_direction_deg"] is None
        assert set(out["missing_inputs"]) == {
            "wind_direction_deg",
            "relative_humidity_pct",
            "temperature_c",
            "fuel_load_tons_ha",
            "fuel_moisture_pct",
        }

    def test_wind_driven_threshold_strict(self, model: Any) -> None:
        assert model.predict_spread({}, {"wind_speed_kmh": 25.0})["wind_driven"] is True

    def test_rate_floor(self, model: Any) -> None:
        out = model.predict_spread(
            {"fuel_load_tons_ha": 0.0, "fuel_moisture_pct": 100.0},
            {"wind_speed_kmh": 0.0, "relative_humidity_pct": 100.0, "temperature_c": -60.0},
        )
        assert out["spread_rate_km_hr"] == pytest.approx(0.1)

    def test_nonfinite_wind_refused(self, model: Any) -> None:
        with pytest.raises(InvalidObservationError):
            model.predict_spread({}, {"wind_speed_kmh": float("inf")})


class TestNDVIProcessor:
    """NDVI arithmetic, fuel-load relation, drought and burn fractions."""

    @pytest.fixture(scope="class")
    @classmethod
    def processor(cls) -> Any:
        from omni_mercury_engine.detectors.geological.wildfire import NDVIProcessor

        return NDVIProcessor()

    def test_compute_ndvi_exact(self, processor: Any) -> None:
        ndvi = processor.compute_ndvi(np.array([0.8]), np.array([0.2]))
        assert ndvi[0] == pytest.approx(0.6)

    def test_compute_ndvi_zero_denominator_safe(self, processor: Any) -> None:
        ndvi = processor.compute_ndvi(np.array([0.0]), np.array([0.0]))
        assert ndvi[0] == pytest.approx(0.0)

    def test_fuel_load_quadratic_relation(self, processor: Any) -> None:
        out = processor.estimate_fuel_load(np.full((4, 4), 0.5))
        assert out["mean_ndvi"] == pytest.approx(0.5)
        assert out["fuel_load_tons_ha"] == pytest.approx(50.0 * 0.25)
        assert out["drought_fraction"] == 0.0
        assert out["is_drought"] is False
        assert out["high_fuel_fraction"] == 1.0  # 0.5 > fuel_threshold 0.4

    def test_nonpositive_mean_ndvi_has_zero_fuel(self, processor: Any) -> None:
        out = processor.estimate_fuel_load(np.full((4, 4), -0.1))
        assert out["fuel_load_tons_ha"] == 0.0
        assert out["is_drought"] is True  # every cell below 0.2

    def test_drought_fraction_threshold_strict(self, processor: Any) -> None:
        # Exactly 30% of cells in drought does not flag (strict > 0.3).
        ndvi = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        out = processor.estimate_fuel_load(ndvi)
        assert out["drought_fraction"] == pytest.approx(0.3)
        assert out["is_drought"] is False

    def test_ndvi_change_burn_metrics(self, processor: Any) -> None:
        out = processor.compute_ndvi_change(np.full(5, 0.1), np.full(5, 0.5))
        assert out["mean_ndvi_change"] == pytest.approx(-0.4)
        assert out["burned_fraction"] == 1.0
        assert out["recovery_potential"] == 0.0
        # A drop of exactly 0.2 is not burned (strict < -0.2).
        boundary = processor.compute_ndvi_change(np.full(5, 0.3), np.full(5, 0.5))
        assert boundary["burned_fraction"] == 0.0


class TestResonanceFrequencyAnalyzer:
    """FFT dominant frequency and smoke-trend classification."""

    @pytest.fixture(scope="class")
    @classmethod
    def analyzer(cls) -> Any:
        from omni_mercury_engine.detectors.geological.wildfire import (
            ResonanceFrequencyAnalyzer,
        )

        return ResonanceFrequencyAnalyzer()

    def test_short_series_exact_default(self, analyzer: Any) -> None:
        out = analyzer.analyze_thermal_resonance(np.array([1.0, 2.0, 3.0]))
        assert out == {
            "resonance_score": 0.0,
            "dominant_frequency": 0.0,
            "fire_behavior_detected": False,
        }

    def test_dominant_frequency_of_pure_tone(self, analyzer: Any) -> None:
        t = np.arange(256)
        signal = np.sin(2 * np.pi * 0.25 * t)
        out = analyzer.analyze_thermal_resonance(signal)
        assert out["dominant_frequency"] == pytest.approx(0.25, abs=1e-6)
        assert 0.0 <= out["resonance_score"] <= 1.0

    def test_smoke_trend_increasing(self, analyzer: Any) -> None:
        out = analyzer.analyze_smoke_patterns(np.array([0.0, 0.2, 0.4, 0.6, 0.8]))
        assert out["smoke_trend"] == "increasing"
        assert out["spread_indicator"] == pytest.approx(0.4)  # min(1, 0.2 * 2)
        assert out["rapid_change_detected"] is False

    def test_smoke_trend_decreasing_and_stable(self, analyzer: Any) -> None:
        down = analyzer.analyze_smoke_patterns(np.array([0.8, 0.6, 0.4, 0.2, 0.0]))
        assert down["smoke_trend"] == "decreasing"
        assert down["spread_indicator"] == 0.0
        flat = analyzer.analyze_smoke_patterns(np.array([0.5, 0.52, 0.48, 0.5, 0.51]))
        assert flat["smoke_trend"] == "stable"
        assert flat["spread_indicator"] == pytest.approx(0.3)

    def test_rapid_change_detection(self, analyzer: Any) -> None:
        out = analyzer.analyze_smoke_patterns(np.array([0.1, 0.1, 0.8, 0.8, 0.8]))
        assert out["rapid_change_detected"] is True
        assert out["max_change_rate"] == pytest.approx(0.7)

    def test_too_short_smoke_series(self, analyzer: Any) -> None:
        out = analyzer.analyze_smoke_patterns(np.array([0.1, 0.9]))
        assert out == {"smoke_trend": "stable", "spread_indicator": 0.0}


# =============================================================================
# Wildfire: detector-level behavior
# =============================================================================


class TestWildfireDetector:
    """Brightness-temperature physics, FIRMS contract, risk ladder, spread gating."""

    @pytest.fixture(scope="class")
    @classmethod
    def detector(cls) -> Any:
        from omni_mercury_engine.detectors.geological.wildfire import WildfireDetector

        return WildfireDetector(enable_resonance=True)

    def test_detector_initialization(self, detector: Any) -> None:
        assert detector.enable_resonance is True
        assert detector.resonance_analyzer is not None
        assert detector.ndvi_processor is not None
        # The shipped wildfire_firms checkpoint loads at construction.
        assert detector._neural_trained is True

    def test_cool_uniform_scene_no_fire(self, detector: Any) -> None:
        result = detector.predict_wildfire({"thermal_image": np.full((16, 16), 300.0)})
        assert result.fire_detected is False
        assert result.confidence == 0.0
        assert result.thermal_hotspots == 0
        assert result.assessment == "active_fire"  # 1-channel image -> physics path
        assert result.risk_level == "low"

    def test_sub_contextual_peak_no_fire(self, detector: Any) -> None:
        scene = np.full((16, 16), 300.0)
        scene[0, 0] = 329.0  # below the 330 K contextual floor
        result = detector.predict_wildfire({"thermal_image": scene})
        assert result.fire_detected is False

    def test_absolute_criterion_and_full_confidence(self, detector: Any) -> None:
        scene = np.full((16, 16), 290.0)
        scene[4, 4] = 400.0
        result = detector.predict_wildfire({"thermal_image": scene, "pixel_size_km": 2.0})
        assert result.fire_detected is True
        assert result.confidence == pytest.approx(1.0)  # (400-330)/60 clipped to 1
        assert result.thermal_hotspots == 1  # single pixel over 350 K
        assert result.ignition_locations == [(4.0, 4.0)]
        assert result.fire_perimeter_km2 == pytest.approx(4.0)  # 1 pixel * (2 km)^2

    def test_contextual_criterion_confidence_exact(self, detector: Any) -> None:
        scene = np.full((16, 16), 290.0)
        scene[2, 2] = 340.0  # > 330 and far above the flat background
        result = detector.predict_wildfire({"thermal_image": scene})
        assert result.fire_detected is True
        assert result.confidence == pytest.approx((340.0 - 330.0) / 60.0)
        assert result.thermal_hotspots == 0  # 340 < 350 K hotspot threshold

    def test_two_hot_regions_two_centroids(self, detector: Any) -> None:
        scene = np.full((16, 16), 290.0)
        scene[0:2, 0:2] = 400.0
        scene[10:12, 10:12] = 400.0
        result = detector.predict_wildfire({"thermal_image": scene})
        assert result.thermal_hotspots == 8
        assert sorted(result.ignition_locations) == [(0.5, 0.5), (10.5, 10.5)]

    def test_one_dimensional_series_counts_but_has_no_geometry(self, detector: Any) -> None:
        result = detector.predict_wildfire({"thermal_image": np.array([300.0, 400.0, 360.0])})
        assert result.fire_detected is True
        assert result.thermal_hotspots == 2  # 400 and 360 exceed 350 K
        assert result.ignition_locations == []

    def test_nonfinite_thermal_image_refused(self, detector: Any) -> None:
        with pytest.raises(InvalidObservationError):
            detector.predict_wildfire({"thermal_image": np.array([[300.0, np.nan]])})

    def test_risk_ladder_integer_points(self, detector: Any) -> None:
        fire_scene = np.full((8, 8), 290.0)
        fire_scene[0, 0] = 400.0
        extreme = detector.predict_wildfire(
            {
                "thermal_image": fire_scene,
                "weather_data": {
                    "wind_speed_kmh": 40.0,
                    "relative_humidity_pct": 10.0,
                    "temperature_c": 40.0,
                },
            }
        )
        assert extreme.risk_level == "extreme"  # 2 + 1 + 1 + 1 = 5 points
        assert extreme.early_warning_actions[0] == (
            "EXTREME FIRE DANGER - Evacuations may be required"
        )

        very_high = detector.predict_wildfire(
            {
                "thermal_image": fire_scene,
                "weather_data": {
                    "wind_speed_kmh": 40.0,
                    "relative_humidity_pct": 10.0,
                    "temperature_c": 20.0,
                },
            }
        )
        assert very_high.risk_level == "very_high"  # 4 points

        high = detector.predict_wildfire(
            {
                "weather_data": {
                    "wind_speed_kmh": 40.0,
                    "relative_humidity_pct": 10.0,
                    "temperature_c": 40.0,
                },
            }
        )
        assert high.risk_level == "high"  # weather-only: 3 points
        assert high.early_warning_actions == ["High fire danger - Prepare for rapid response"]

        moderate = detector.predict_wildfire({"thermal_image": fire_scene})
        assert moderate.risk_level == "moderate"  # fire alone: 2 points

    def test_risk_point_boundaries_are_strict(self, detector: Any) -> None:
        result = detector.predict_wildfire(
            {
                "weather_data": {
                    "wind_speed_kmh": 30.0,  # not > 30
                    "relative_humidity_pct": 20.0,  # not < 20
                    "temperature_c": 35.0,  # not > 35
                },
            }
        )
        assert result.risk_level == "low"  # 0 points

    def test_spread_only_for_a_detected_fire(self, detector: Any) -> None:
        weather = {"wind_speed_kmh": 20.0, "wind_direction_deg": 90.0}
        no_fire = detector.predict_wildfire(
            {"thermal_image": np.full((8, 8), 300.0), "weather_data": weather}
        )
        assert no_fire.spread_rate_km_hr is None

        fire_scene = np.full((8, 8), 290.0)
        fire_scene[0, 0] = 400.0
        fire = detector.predict_wildfire(
            {
                "thermal_image": fire_scene,
                "weather_data": weather,
                "fire_data": {"fuel_load_tons_ha": 20.0, "fuel_moisture_pct": 10.0},
            }
        )
        # Defaults RH 50 / T 25: 0.5 * 2.0 * 0.75 * 1.1 * 1.5 * 0.9.
        assert fire.spread_rate_km_hr == pytest.approx(0.5 * 2.0 * 0.75 * 1.1 * 1.5 * 0.9)
        assert fire.spread_direction_deg == 90.0

    def test_weather_factors_and_nfdrs_fuel_moisture(self, detector: Any) -> None:
        result = detector.predict_wildfire(
            {
                "weather_data": {
                    "temperature_c": 20.0,
                    "relative_humidity_pct": 50.0,
                    "nfdrs_state_of_weather": 2,
                },
            }
        )
        emc = 2.22749 + 0.160107 * 45.5 - 0.01478 * 80.0
        assert result.weather_factors["nfdrs_state_of_weather"] == 2.0
        assert result.weather_factors["equilibrium_moisture_pct"] == pytest.approx(emc)
        assert result.fuel_moisture == pytest.approx(1.03 * emc)

    def test_measured_fuel_moisture_wins_over_nfdrs(self, detector: Any) -> None:
        result = detector.predict_wildfire(
            {
                "weather_data": {
                    "temperature_c": 20.0,
                    "relative_humidity_pct": 50.0,
                    "nfdrs_state_of_weather": 2,
                },
                "fire_data": {"fuel_moisture_pct": 7.0},
            }
        )
        assert result.fuel_moisture == 7.0

    def test_firms_raster_routes_to_forecaster(self, detector: Any) -> None:
        raster = np.zeros((3, 32, 32))
        raster[0, 15:17, 15:17] = 360.0  # detections in the centre cells
        raster[1, 15:17, 15:17] = 5.0  # log1p FRP
        raster[2, 15:17, 15:17] = 2.0  # log1p count
        result = detector.predict_wildfire({"thermal_image": raster})
        assert result.assessment == "next_day_forecast"
        assert 0.0 <= result.confidence <= 1.0
        assert result.fire_detected == (result.confidence > detector._fire_prob_threshold)
        assert result.thermal_hotspots == 4  # channel-0 360 K cells over 350 K

    def test_brightness_three_band_image_is_not_a_firms_raster(self, detector: Any) -> None:
        image = np.full((3, 32, 32), 300.0)
        image[:, 0, 0] = 400.0
        result = detector.predict_wildfire({"thermal_image": image})
        assert result.assessment == "active_fire"  # physics, not the forecaster
        assert result.fire_detected is True

    def test_is_firms_raster_predicate(self) -> None:
        from omni_mercury_engine.detectors.geological.wildfire import _is_firms_raster

        good = np.zeros((3, 8, 8))
        good[0, 2, 2] = 350.0
        good[1, 2, 2] = 5.0
        good[2, 2, 2] = 2.0
        assert _is_firms_raster(good) is True
        assert _is_firms_raster(np.full((3, 8, 8), 300.0)) is False

    def test_diagnostics_capture_on_request(self) -> None:
        from omni_mercury_engine.detectors.geological.wildfire import WildfireDetector

        detector = WildfireDetector(keep_diagnostics=True)
        scene = np.full((8, 8), 290.0)
        scene[3, 3] = 400.0
        result = detector.predict_wildfire({"thermal_image": scene})
        assert result.diagnostics is not None
        assert result.diagnostics.arrays["hotspot_mask"].sum() == 1
        assert result.diagnostics.context["hotspot_count"] == 1


class TestWildfireDetectorWithoutResonance:
    """Tests for WildfireDetector without 3R Resonance."""

    @pytest.fixture
    def detector_no_resonance(self) -> Any:
        from omni_mercury_engine.detectors.geological.wildfire import WildfireDetector

        return WildfireDetector(enable_resonance=False)

    def test_detector_without_resonance(self, detector_no_resonance: Any) -> None:
        assert detector_no_resonance.enable_resonance is False
        assert detector_no_resonance.resonance_analyzer is None

    def test_predict_without_resonance(self, detector_no_resonance: Any) -> None:
        scene = np.full((8, 8), 290.0)
        scene[0, 0] = 400.0
        result = detector_no_resonance.predict_wildfire({"thermal_image": scene})
        assert result.fire_detected is True


# =============================================================================
# Volcanic: component analyzers
# =============================================================================


class TestThermalHotspotDetector:
    """20 K-above-baseline anomaly rule and the intensity ladder."""

    @pytest.fixture(scope="class")
    @classmethod
    def detector(cls) -> Any:
        from omni_mercury_engine.detectors.geological.volcanic import ThermalHotspotDetector

        return ThermalHotspotDetector()

    def test_empty_input_defaults(self, detector: Any) -> None:
        out = detector.detect_thermal_anomaly({})
        assert out == {"anomaly_detected": False, "max_temp_k": 288.0}

    def test_anomaly_threshold_is_baseline_plus_20(self, detector: Any) -> None:
        cold = detector.detect_thermal_anomaly({"brightness_temperature_k": np.full(10, 305.0)})
        assert cold["anomaly_detected"] is False
        hot = detector.detect_thermal_anomaly({"brightness_temperature_k": np.full(10, 309.0)})
        assert hot["anomaly_detected"] is True

    @pytest.mark.parametrize(
        ("max_temp", "intensity"),
        [(300.0, "low"), (330.0, "moderate"), (360.0, "high"), (410.0, "extreme")],
    )
    def test_intensity_ladder(self, detector: Any, max_temp: float, intensity: str) -> None:
        out = detector.detect_thermal_anomaly(
            {"brightness_temperature_k": np.array([290.0, max_temp])}
        )
        assert out["intensity"] == intensity
        assert out["max_temp_k"] == max_temp

    def test_hotspot_pixel_count(self, detector: Any) -> None:
        temps = np.array([300.0] * 99 + [400.0])
        out = detector.detect_thermal_anomaly({"brightness_temperature_k": temps})
        assert out["hotspot_pixel_count"] == 1

    def test_out_of_range_brightness_refused(self, detector: Any) -> None:
        with pytest.raises(InvalidObservationError):
            detector.detect_thermal_anomaly({"brightness_temperature_k": np.array([np.inf])})


class TestGasEmissionAnalyzer:
    """SO2 3x / CO2 2x anomaly ratios and the degassing-index ladder."""

    @pytest.fixture(scope="class")
    @classmethod
    def analyzer(cls) -> Any:
        from omni_mercury_engine.detectors.geological.volcanic import GasEmissionAnalyzer

        return GasEmissionAnalyzer()

    def test_baselines_give_unit_index_and_no_anomaly(self, analyzer: Any) -> None:
        out = analyzer.analyze_gas_emissions({})
        assert out["degassing_index"] == pytest.approx(1.0)
        assert out["so2_anomaly"] is False
        assert out["co2_anomaly"] is False
        assert out["risk_level"] == "low"

    def test_anomaly_ratios_are_strict(self, analyzer: Any) -> None:
        at = analyzer.analyze_gas_emissions({"so2_tons_per_day": 300.0, "co2_tons_per_day": 1000.0})
        assert at["so2_anomaly"] is False  # ratio exactly 3
        assert at["co2_anomaly"] is False  # ratio exactly 2
        over = analyzer.analyze_gas_emissions(
            {"so2_tons_per_day": 301.0, "co2_tons_per_day": 1001.0}
        )
        assert over["so2_anomaly"] is True
        assert over["co2_anomaly"] is True

    @pytest.mark.parametrize(
        ("so2", "co2", "level"),
        [
            (100.0, 500.0, "low"),  # index 1.0
            (300.0, 1000.0, "moderate"),  # index 2.5
            (500.0, 1500.0, "high"),  # index 4.0
            (700.0, 2500.0, "critical"),  # index 6.0
        ],
    )
    def test_degassing_ladder(self, analyzer: Any, so2: float, co2: float, level: str) -> None:
        out = analyzer.analyze_gas_emissions({"so2_tons_per_day": so2, "co2_tons_per_day": co2})
        assert out["risk_level"] == level

    def test_non_mapping_refused(self, analyzer: Any) -> None:
        with pytest.raises(InputModalityError):
            analyzer.analyze_gas_emissions([100.0, 500.0])


class TestInSARDeformationDetector:
    """Euclidean displacement, the strict 5 cm gate, and the severity ladder."""

    @pytest.fixture(scope="class")
    @classmethod
    def detector(cls) -> Any:
        from omni_mercury_engine.detectors.geological.volcanic import InSARDeformationDetector

        return InSARDeformationDetector()

    def test_total_displacement_is_euclidean(self, detector: Any) -> None:
        out = detector.detect_deformation(
            {"vertical_displacement_cm": 3.0, "horizontal_displacement_cm": 4.0}
        )
        assert out["total_displacement_cm"] == pytest.approx(5.0)
        assert out["deformation_detected"] is False  # strict > 5

    @pytest.mark.parametrize(
        ("vertical", "horizontal", "severity"),
        [(3.0, 4.0, "low"), (6.0, 8.0, "moderate"), (9.0, 12.0, "high"), (15.0, 20.0, "critical")],
    )
    def test_severity_ladder(
        self, detector: Any, vertical: float, horizontal: float, severity: str
    ) -> None:
        out = detector.detect_deformation(
            {"vertical_displacement_cm": vertical, "horizontal_displacement_cm": horizontal}
        )
        assert out["severity"] == severity

    def test_deformation_type_sign(self, detector: Any) -> None:
        up = detector.detect_deformation({"vertical_displacement_cm": 6.0})
        assert up["deformation_type"] == "inflation"
        down = detector.detect_deformation({"vertical_displacement_cm": -6.0})
        assert down["deformation_type"] == "deflation"


# =============================================================================
# Volcanic: HMM state tracker
# =============================================================================


class TestVolcanicStateHMM:
    """Forward-algorithm belief updates over the published matrices."""

    @pytest.fixture
    def hmm_tracker(self) -> Any:
        from omni_mercury_engine.detectors.geological.volcanic import VolcanicStateHMM

        return VolcanicStateHMM()

    def test_hmm_initialization(self, hmm_tracker: Any) -> None:
        assert hmm_tracker.state_names == [
            "QUIESCENT",
            "UNREST",
            "PRE_ERUPTIVE",
            "ERUPTIVE",
            "POST_ERUPTIVE",
        ]
        assert np.allclose(hmm_tracker.transition_matrix.sum(axis=1), 1.0)
        assert np.array_equal(
            hmm_tracker.initial_distribution, np.array([0.7, 0.2, 0.05, 0.03, 0.02])
        )
        assert np.array_equal(hmm_tracker.state_belief, hmm_tracker.initial_distribution)

    def test_all_precursors_observed_points_to_eruptive(self, hmm_tracker: Any) -> None:
        belief = hmm_tracker.update_belief(np.array([True, True, True, True]))
        assert belief.sum() == pytest.approx(1.0)
        state_idx, state_name, prob = hmm_tracker.get_most_likely_state()
        assert state_idx == hmm_tracker.ERUPTIVE
        assert state_name == "ERUPTIVE"
        assert prob == pytest.approx(float(belief[hmm_tracker.ERUPTIVE]))
        assert hmm_tracker.state_history == [hmm_tracker.ERUPTIVE]

    def test_no_observations_stays_quiescent(self, hmm_tracker: Any) -> None:
        hmm_tracker.update_belief(np.array([False, False, False, False]))
        state_idx, state_name, _ = hmm_tracker.get_most_likely_state()
        assert state_name == "QUIESCENT"
        assert state_idx == hmm_tracker.QUIESCENT

    def test_update_matches_forward_algorithm_exactly(self, hmm_tracker: Any) -> None:
        obs = np.array([True, False, True, False])
        prior = hmm_tracker.state_belief.copy()
        T = hmm_tracker.transition_matrix
        E = hmm_tracker.emission_matrix
        likelihood = E[:, 0] * (1 - E[:, 1]) * E[:, 2] * (1 - E[:, 3])
        expected = likelihood * (T.T @ prior)
        expected = expected / (expected.sum() + 1e-10)
        belief = hmm_tracker.update_belief(obs)
        assert np.allclose(belief, expected)

    def test_predict_next_state_uses_transition_matrix(self, hmm_tracker: Any) -> None:
        hmm_tracker.update_belief(np.array([True, True, True, True]))
        next_belief = hmm_tracker.transition_matrix.T @ hmm_tracker.state_belief
        idx, name, prob = hmm_tracker.predict_next_state()
        assert idx == int(np.argmax(next_belief))
        assert name == hmm_tracker.state_names[idx]
        assert prob == pytest.approx(float(next_belief[idx]))

    def test_eruption_probability_formula(self, hmm_tracker: Any) -> None:
        hmm_tracker.update_belief(np.array([True, True, True, True]))
        belief = hmm_tracker.state_belief
        expected = 0.6 * belief[hmm_tracker.ERUPTIVE] + 0.4 * (
            hmm_tracker.transition_matrix[:, hmm_tracker.ERUPTIVE] @ belief
        )
        assert hmm_tracker.get_eruption_probability() == pytest.approx(float(expected))

    def test_hmm_reset(self, hmm_tracker: Any) -> None:
        hmm_tracker.update_belief(np.array([True, True, True, True]))
        hmm_tracker.reset()
        assert np.array_equal(hmm_tracker.state_belief, hmm_tracker.initial_distribution)
        assert hmm_tracker.state_history == []


# =============================================================================
# Volcanic: 3R Refactoring optimizer
# =============================================================================


class TestRefactoringAdaptiveOptimizer:
    """Error accounting, adaptation thresholds, and the clipped calibrations."""

    @pytest.fixture
    def optimizer(self) -> Any:
        from omni_mercury_engine.detectors.geological.volcanic import (
            RefactoringAdaptiveOptimizer,
        )

        return RefactoringAdaptiveOptimizer()

    def test_record_prediction_without_outcome_tracks_no_error(self, optimizer: Any) -> None:
        optimizer.record_prediction({"eruption_imminent": False, "confidence": 0.7})
        assert len(optimizer.prediction_history) == 1
        assert optimizer.error_history == []

    def test_history_window_is_enforced(self, optimizer: Any) -> None:
        for _ in range(optimizer.history_window + 5):
            optimizer.record_prediction({"eruption_imminent": False, "confidence": 0.5})
        assert len(optimizer.prediction_history) == optimizer.history_window

    def test_prediction_error_exact_cases(self, optimizer: Any) -> None:
        perfect = optimizer._compute_prediction_error(
            {"eruption_imminent": True, "confidence": 1.0, "vei_estimate": 4},
            {"eruption_occurred": True, "actual_vei": 4},
        )
        assert perfect == 0.0
        worst = optimizer._compute_prediction_error(
            {"eruption_imminent": False, "confidence": 0.0},
            {"eruption_occurred": True},
        )
        assert worst == 1.0
        vei_only = optimizer._compute_prediction_error({"vei_estimate": 2}, {"actual_vei": 6})
        assert vei_only == pytest.approx(0.5)
        assert optimizer._compute_prediction_error({}, {}) == 0.5

    def test_adapt_parameters_needs_ten_errors(self, optimizer: Any) -> None:
        params = optimizer.adapt_parameters()
        assert params == {
            "confidence_calibration": 1.0,
            "threshold_adjustment": 0.0,
            "refactoring_score": 0.5,
        }

    def test_high_error_reduces_calibration(self, optimizer: Any) -> None:
        for _ in range(12):
            optimizer.record_prediction(
                {"eruption_imminent": True, "confidence": 1.0},
                {"eruption_occurred": False},
            )
        params = optimizer.adapt_parameters()
        assert params["confidence_calibration"] == pytest.approx(1.0 - optimizer.learning_rate)
        assert params["refactoring_score"] == pytest.approx(0.0)

    def test_low_error_raises_calibration_by_phi_fraction(self, optimizer: Any) -> None:
        for _ in range(12):
            optimizer.record_prediction(
                {"eruption_imminent": True, "confidence": 1.0},
                {"eruption_occurred": True},
            )
        params = optimizer.adapt_parameters()
        assert params["confidence_calibration"] == pytest.approx(
            1.0 + optimizer.learning_rate / optimizer.phi
        )
        assert params["refactoring_score"] == pytest.approx(1.0)

    def test_adapted_confidence_scales_and_clips(self, optimizer: Any) -> None:
        assert optimizer.get_adapted_confidence(0.7) == pytest.approx(0.7)
        optimizer.confidence_calibration = 1.5
        assert optimizer.get_adapted_confidence(0.8) == pytest.approx(1.0)  # clipped

    def test_adapted_threshold_clips_to_operating_band(self, optimizer: Any) -> None:
        assert optimizer.get_adapted_threshold(0.5) == pytest.approx(0.5)
        assert optimizer.get_adapted_threshold(0.95) == pytest.approx(0.9)
        assert optimizer.get_adapted_threshold(0.1) == pytest.approx(0.3)


# =============================================================================
# Volcanic: detector-level behavior
# =============================================================================


class TestVolcanicEruptionDetector:
    """Physics forecast, USGS alert ladder, zones/warnings, determinism."""

    @pytest.fixture(scope="class")
    @classmethod
    def detector(cls) -> Any:
        from omni_mercury_engine.detectors.geological.volcanic import (
            VolcanicEruptionDetector,
        )

        return VolcanicEruptionDetector(enable_refactoring=True)

    def test_detector_initialization(self, detector: Any) -> None:
        assert detector.enable_refactoring is True
        assert detector.refactoring_optimizer is not None
        assert detector.seismic_detector is not None
        assert detector.state_hmm is not None
        # The shipped volcanic_avo_seismic checkpoint loads at construction.
        assert detector._neural_trained is True

    def test_non_mapping_input_refused(self, detector: Any) -> None:
        with pytest.raises(InputModalityError):
            detector.predict_eruption(np.zeros(10))

    def test_quiet_volcano_reports_normal(self, detector: Any) -> None:
        result = detector.predict_eruption(
            {
                "seismic_sequence": np.full(100, 1.0),
                "thermal_data": {"brightness_temperature_k": np.full(50, 290.0)},
                "gas_data": {"so2_tons_per_day": 100.0, "co2_tons_per_day": 500.0},
                "insar_data": {"vertical_displacement_cm": 0.0},
            }
        )
        assert result.eruption_imminent is False
        assert result.alert_level == "normal"
        assert result.eruption_type == "no_eruption"
        assert result.seismic_swarm_detected is False
        assert result.gas_flux_anomaly is False
        assert result.deformation_detected is False
        assert result.hazard_zones == []
        assert result.early_warning_actions == []
        assert result.evacuation_recommendations == []

    def test_multi_precursor_crisis_reaches_warning(self, detector: Any) -> None:
        result = detector.predict_eruption(
            {
                "thermal_data": {
                    "brightness_temperature_k": np.full(50, 380.0),
                    "radiant_heat_mw": 900.0,
                },
                "gas_data": {"so2_tons_per_day": 2000.0, "co2_tons_per_day": 5000.0},
                "insar_data": {
                    "vertical_displacement_cm": 15.0,
                    "horizontal_displacement_cm": 20.0,
                },
            }
        )
        assert result.thermal_anomaly_detected is True
        assert result.gas_flux_anomaly is True
        assert result.deformation_detected is True
        assert result.eruption_imminent is True
        assert result.confidence == pytest.approx(1.0)  # degassing term saturates the noisy-OR
        assert result.alert_level == "warning"
        assert result.eruption_type == "undetermined"
        assert result.hazard_zones == ["crater_vicinity"]  # no VEI estimate -> no radius zones
        assert result.early_warning_actions[0] == (
            "VOLCANIC WARNING: hazardous eruption imminent, underway or suspected"
        )
        assert result.evacuation_recommendations[0] == "Evacuate high-risk zones immediately"

    def test_prediction_is_deterministic_across_calls(self, detector: Any) -> None:
        data = {
            "thermal_data": {"brightness_temperature_k": np.full(50, 340.0)},
            "gas_data": {"so2_tons_per_day": 400.0, "co2_tons_per_day": 1200.0},
        }
        first = detector.predict_eruption(dict(data))
        second = detector.predict_eruption(dict(data))
        # persist_hmm_state defaults False: each call restarts from the prior.
        assert second.confidence == pytest.approx(first.confidence)
        assert second.alert_level == first.alert_level

    def test_schumann_is_reported_but_never_decides(self, detector: Any) -> None:
        quiet = {
            "thermal_data": {"brightness_temperature_k": np.full(50, 290.0)},
            "gas_data": {"so2_tons_per_day": 100.0, "co2_tons_per_day": 500.0},
        }
        with_elf = detector.predict_eruption({**quiet, "schumann_elf": np.full(100, 9.83)})
        assert with_elf.schumann_elf_correlation == pytest.approx(1.0)  # |9.83-7.83|/2
        assert with_elf.alert_level == "normal"

    def test_alert_ladder_direct(self, detector: Any) -> None:
        assert detector._determine_alert_level(3, 0.85) == "warning"
        assert detector._determine_alert_level(3, 0.8) == "watch"  # strict > 0.8
        assert detector._determine_alert_level(2, 0.65) == "watch"
        assert detector._determine_alert_level(1, 0.45) == "advisory"
        assert detector._determine_alert_level(1, 0.4) == "normal"  # strict > 0.4
        assert detector._determine_alert_level(0, 0.99) == "normal"

    def test_hazard_zones_scale_with_vei(self, detector: Any) -> None:
        from omni_mercury_engine.detectors.geological.volcanic import VolcanicPredictionResult

        def result_with(vei: int | None, imminent: bool = True) -> VolcanicPredictionResult:
            return VolcanicPredictionResult(
                eruption_imminent=imminent,
                confidence=0.9,
                alert_level="warning",
                eruption_type="undetermined",
                vei_estimate=vei,
            )

        assert detector._identify_hazard_zones(result_with(None)) == ["crater_vicinity"]
        vei3 = detector._identify_hazard_zones(result_with(3))
        assert vei3 == ["crater_vicinity", "10km_radius", "ashfall_region"]
        vei4 = detector._identify_hazard_zones(result_with(4))
        assert "pyroclastic_flow_paths" in vei4 and "lahar_drainage_channels" in vei4
        assert detector._identify_hazard_zones(result_with(4, imminent=False)) == []

    def test_swarm_physics_exact(self, detector: Any) -> None:
        constant = detector._detect_swarm_physics(np.full(100, 3.0))
        assert constant["swarm_detected"] is False
        assert constant["confidence"] == 0.0

        spiky = detector._detect_swarm_physics(np.array([0.0] * 95 + [1.0] * 5))
        assert spiky["swarm_detected"] is True  # 5% of samples far above the median
        assert spiky["confidence"] == pytest.approx(0.5)  # 0.05 / 0.10

        short = detector._detect_swarm_physics(np.array([1.0, 2.0]))
        assert short == {"swarm_detected": False, "confidence": 0.0, "attention_weights": []}

    def test_forecast_physics_noisy_or(self, detector: Any) -> None:
        none_seen = detector._forecast_eruption_physics({})
        assert none_seen == {"eruption_imminent": False, "confidence": 0.0, "method": "physics"}

        saturated = detector._forecast_eruption_physics({"degassing_index": 5.0})
        assert saturated["confidence"] == pytest.approx(1.0)
        assert saturated["eruption_imminent"] is True

        combined = detector._forecast_eruption_physics(
            {"seismic_confidence": 0.5, "total_displacement_cm": 10.0}
        )
        assert combined["confidence"] == pytest.approx(0.75)
        assert combined["eruption_imminent"] is True  # 0.75 > 0.7

        single = detector._forecast_eruption_physics({"seismic_confidence": 0.5})
        assert single["eruption_imminent"] is False

    def test_schumann_correlation_formula(self, detector: Any) -> None:
        assert detector._correlate_schumann_elf(np.full(10, 7.83)) == pytest.approx(0.0)
        assert detector._correlate_schumann_elf(np.full(10, 8.83)) == pytest.approx(0.5)

    def test_vei_estimate_bounds_when_present(self, detector: Any) -> None:
        result = detector.predict_eruption(
            {"thermal_data": {"brightness_temperature_k": np.full(10, 290.0)}}
        )
        assert result.vei_estimate is None or 0 <= result.vei_estimate <= 8


class TestVolcanicDetectorWithoutRefactoring:
    """Tests for VolcanicEruptionDetector without 3R Refactoring."""

    @pytest.fixture
    def detector_no_refactoring(self) -> Any:
        from omni_mercury_engine.detectors.geological.volcanic import (
            VolcanicEruptionDetector,
        )

        return VolcanicEruptionDetector(enable_refactoring=False)

    def test_detector_without_refactoring(self, detector_no_refactoring: Any) -> None:
        assert detector_no_refactoring.enable_refactoring is False
        assert detector_no_refactoring.refactoring_optimizer is None

    def test_predict_without_refactoring(self, detector_no_refactoring: Any) -> None:
        result = detector_no_refactoring.predict_eruption(
            {"gas_data": {"so2_tons_per_day": 2000.0, "co2_tons_per_day": 5000.0}}
        )
        assert result.gas_flux_anomaly is True


# =============================================================================
# Integration
# =============================================================================


class TestEnhancedDetectorIntegration:
    """All three enhanced detectors work end to end with their 3R synapses."""

    @pytest.fixture(scope="class")
    @classmethod
    def all_detectors(cls) -> Any:
        from omni_mercury_engine.detectors.geological.landslide import LandslideDetector
        from omni_mercury_engine.detectors.geological.volcanic import (
            VolcanicEruptionDetector,
        )
        from omni_mercury_engine.detectors.geological.wildfire import WildfireDetector

        return {
            "landslide": LandslideDetector(enable_recursion=True),
            "wildfire": WildfireDetector(enable_resonance=True),
            "volcanic": VolcanicEruptionDetector(enable_refactoring=True),
        }

    def test_3r_synapses_enabled(self, all_detectors: Any) -> None:
        assert all_detectors["landslide"].enable_recursion is True
        assert all_detectors["wildfire"].enable_resonance is True
        assert all_detectors["volcanic"].enable_refactoring is True

    def test_end_to_end_predictions_are_coherent(self, all_detectors: Any) -> None:
        landslide = all_detectors["landslide"].predict_landslide(
            {
                "rainfall_data": {"intensity_mm_hr": 20.0, "duration_hours": 6.0},
                "sensor_data": {"soil_saturation_pct": 90.0, "displacement_rate_mm_day": 40.0},
                "slope_data": {"slope_angle_deg": 45.0},
            }
        )
        assert landslide.landslide_imminent is True

        fire_scene = np.full((8, 8), 290.0)
        fire_scene[0, 0] = 400.0
        wildfire = all_detectors["wildfire"].predict_wildfire({"thermal_image": fire_scene})
        assert wildfire.fire_detected is True

        volcanic = all_detectors["volcanic"].predict_eruption(
            {"gas_data": {"so2_tons_per_day": 2000.0, "co2_tons_per_day": 5000.0}}
        )
        assert volcanic.gas_flux_anomaly is True

        # Every result carries a bounded confidence.
        for result in (landslide, wildfire, volcanic):
            assert 0.0 <= result.confidence <= 1.0


# =============================================================================
# Mutation-floor kill tests: exact constants and internals
# =============================================================================


class TestMutationKillPins:
    """Exact-value pins that close the gaps the first enumeration measured."""

    def test_hmm_matrices_are_the_published_constants(self) -> None:
        from omni_mercury_engine.detectors.geological.volcanic import VolcanicStateHMM

        hmm = VolcanicStateHMM()
        expected_transitions = np.array(
            [
                [0.90, 0.08, 0.01, 0.005, 0.005],
                [0.15, 0.70, 0.12, 0.02, 0.01],
                [0.05, 0.20, 0.50, 0.20, 0.05],
                [0.01, 0.05, 0.10, 0.60, 0.24],
                [0.30, 0.40, 0.10, 0.05, 0.15],
            ]
        )
        assert np.allclose(hmm.transition_matrix, expected_transitions, atol=1e-12)
        expected_emissions = np.array(
            [
                [0.05, 0.02, 0.03, 0.02],
                [0.40, 0.20, 0.30, 0.25],
                [0.70, 0.50, 0.60, 0.55],
                [0.90, 0.85, 0.80, 0.75],
                [0.30, 0.40, 0.25, 0.20],
            ]
        )
        assert np.array_equal(hmm.emission_matrix, expected_emissions)

    def test_thermal_anomaly_boundary_is_baseline_plus_20_exact(self) -> None:
        from omni_mercury_engine.detectors.geological.volcanic import ThermalHotspotDetector

        detector = ThermalHotspotDetector()
        at = detector.detect_thermal_anomaly({"brightness_temperature_k": np.array([308.0])})
        assert at["anomaly_detected"] is False
        over = detector.detect_thermal_anomaly({"brightness_temperature_k": np.array([308.1])})
        assert over["anomaly_detected"] is True

    @pytest.mark.parametrize(
        ("max_temp", "intensity"),
        [(320.0, "low"), (350.0, "moderate"), (400.0, "high")],
    )
    def test_thermal_intensity_cuts_are_strict(self, max_temp: float, intensity: str) -> None:
        from omni_mercury_engine.detectors.geological.volcanic import ThermalHotspotDetector

        out = ThermalHotspotDetector().detect_thermal_anomaly(
            {"brightness_temperature_k": np.array([290.0, max_temp])}
        )
        assert out["intensity"] == intensity
        assert out["mean_temp_k"] == pytest.approx((290.0 + max_temp) / 2)

    def test_gas_fluxes_echo_and_negative_refused(self) -> None:
        from omni_mercury_engine.detectors.geological.volcanic import GasEmissionAnalyzer

        out = GasEmissionAnalyzer().analyze_gas_emissions(
            {"so2_tons_per_day": 250.0, "co2_tons_per_day": 750.0}
        )
        assert out["so2_flux_tons_day"] == 250.0
        assert out["co2_flux_tons_day"] == 750.0
        assert out["degassing_index"] == pytest.approx((2.5 + 1.5) / 2)
        with pytest.raises(ValueError):
            GasEmissionAnalyzer().analyze_gas_emissions({"so2_tons_per_day": -1.0})

    def test_deformation_details(self) -> None:
        from omni_mercury_engine.detectors.geological.volcanic import InSARDeformationDetector

        detector = InSARDeformationDetector()
        out = detector.detect_deformation(
            {"vertical_displacement_cm": 0.0, "deformation_rate_cm_day": 2.5}
        )
        assert out["deformation_type"] == "deflation"  # strictly positive means inflation
        assert out["deformation_rate_cm_day"] == 2.5
        at_20 = detector.detect_deformation({"vertical_displacement_cm": 20.0})
        assert at_20["severity"] == "high"  # critical needs strictly more than 20 cm

    def test_single_indicator_never_forecasts_eruption(self) -> None:
        """The physics forecast gate needs two concurrent indicators."""
        from omni_mercury_engine.detectors.geological.volcanic import VolcanicEruptionDetector

        detector = VolcanicEruptionDetector()
        gas_only = detector.predict_eruption(
            {"gas_data": {"so2_tons_per_day": 2000.0, "co2_tons_per_day": 5000.0}}
        )
        assert gas_only.gas_flux_anomaly is True
        assert gas_only.eruption_imminent is False
        assert gas_only.confidence == 0.0
        assert gas_only.alert_level == "normal"

        two = detector.predict_eruption(
            {
                "gas_data": {"so2_tons_per_day": 2000.0, "co2_tons_per_day": 5000.0},
                "thermal_data": {
                    "brightness_temperature_k": np.full(10, 340.0),
                    "radiant_heat_mw": 100.0,
                },
            }
        )
        assert two.eruption_imminent is True
        assert two.confidence == pytest.approx(1.0)
        assert two.alert_level == "watch"

    def test_refactoring_threshold_adjustment_follows_the_error_trend(self) -> None:
        from omni_mercury_engine.detectors.geological.volcanic import (
            RefactoringAdaptiveOptimizer,
        )

        worsening = RefactoringAdaptiveOptimizer()
        # Confidence-calibration errors 0.0, 0.05, ..., strictly increasing.
        for i in range(12):
            worsening.record_prediction(
                {"eruption_imminent": False, "confidence": min(1.0, i * 0.05)},
                {"eruption_occurred": False},
            )
        params = worsening.adapt_parameters()
        assert params["threshold_adjustment"] == pytest.approx(worsening.learning_rate * 0.1)

        improving = RefactoringAdaptiveOptimizer()
        for i in range(12):
            improving.record_prediction(
                {"eruption_imminent": False, "confidence": max(0.0, 0.9 - i * 0.05)},
                {"eruption_occurred": False},
            )
        params = improving.adapt_parameters()
        assert params["threshold_adjustment"] == pytest.approx(-improving.learning_rate * 0.05)

    def test_refactoring_calibration_clips_at_its_floor(self) -> None:
        from omni_mercury_engine.detectors.geological.volcanic import (
            RefactoringAdaptiveOptimizer,
        )

        optimizer = RefactoringAdaptiveOptimizer()
        optimizer.confidence_calibration = 0.504
        for _ in range(12):
            optimizer.record_prediction(
                {"eruption_imminent": True, "confidence": 1.0},
                {"eruption_occurred": False},
            )
        assert optimizer.adapt_parameters()["confidence_calibration"] == pytest.approx(0.5)

    def test_refactoring_midband_error_leaves_calibration_alone(self) -> None:
        from omni_mercury_engine.detectors.geological.volcanic import (
            RefactoringAdaptiveOptimizer,
        )

        optimizer = RefactoringAdaptiveOptimizer()
        for _ in range(12):
            # errors [0.0, 0.4] -> mean 0.2 per record... use conf 0.8 on a
            # non-event: errors are [0, 0.8] -> mean 0.4, inside (0.3, 0.5].
            optimizer.record_prediction(
                {"eruption_imminent": False, "confidence": 0.8},
                {"eruption_occurred": False},
            )
        params = optimizer.adapt_parameters()
        assert params["confidence_calibration"] == pytest.approx(1.0)
        assert params["refactoring_score"] == pytest.approx(0.6)

    def test_prediction_error_treats_missing_vei_as_zero(self) -> None:
        from omni_mercury_engine.detectors.geological.volcanic import (
            RefactoringAdaptiveOptimizer,
        )

        optimizer = RefactoringAdaptiveOptimizer()
        error = optimizer._compute_prediction_error({"vei_estimate": None}, {"actual_vei": 4})
        assert error == pytest.approx(0.5)  # |0 - 4| / 8

    def test_resonance_scores_fire_frequencies_with_phi_weights(self) -> None:
        from omni_mercury_engine.detectors.geological.wildfire import (
            ResonanceFrequencyAnalyzer,
        )

        analyzer = ResonanceFrequencyAnalyzer()
        t = np.arange(240)
        out = analyzer.analyze_thermal_resonance(np.sin(2 * np.pi * (1 / 24) * t))
        # The diurnal band holds the dominant peak (score ~1), the convective
        # band only leakage; phi-weighting gives ~phi/(phi+1) = 0.618.
        assert out["dominant_frequency"] == pytest.approx(1 / 24, abs=1e-4)
        assert out["resonance_score"] == pytest.approx(0.618, abs=0.02)
        assert out["fire_behavior_detected"] is True
        # All four bands are reported by name (scores keyed by band since the
        # 2026-10-20 misattribution fix); bands with no FFT bin read 0.0.
        assert list(out["frequency_components"]) == [
            "diurnal",
            "wind_gust",
            "convective",
            "spotting",
        ]
        assert out["frequency_components"]["wind_gust"] == 0.0

    def test_smoke_trend_boundaries_are_strict(self) -> None:
        from omni_mercury_engine.detectors.geological.wildfire import (
            ResonanceFrequencyAnalyzer,
        )

        analyzer = ResonanceFrequencyAnalyzer()
        # Mean diff exactly +0.1 is "stable" (strictly greater is required).
        flat = analyzer.analyze_smoke_patterns(np.array([0.0, 0.1, 0.2, 0.3, 0.4]))
        assert flat["smoke_trend"] == "stable"
        assert flat["spread_indicator"] == pytest.approx(0.3)
        # A max jump of exactly 0.5 is not "rapid" (strictly greater required).
        edge = analyzer.analyze_smoke_patterns(np.array([0.0, 0.5, 0.5, 0.5, 0.5]))
        assert edge["rapid_change_detected"] is False
        assert edge["max_change_rate"] == pytest.approx(0.5)

    def test_undetected_fire_confidence_is_capped_at_half(self) -> None:
        """A hot-but-uniform scene keeps confidence at the 0.5 cap."""
        from omni_mercury_engine.detectors.geological.wildfire import WildfireDetector

        detector = WildfireDetector(load_shipped_weights=False)
        result = detector.predict_wildfire({"thermal_image": np.full((8, 8), 355.0)})
        # Uniform 355 K: no contrast, below the 360 K absolute criterion.
        assert result.fire_detected is False
        assert result.confidence == pytest.approx(0.5)  # min(raw 1.0, cap 0.5)
        assert result.thermal_hotspots == 64  # every pixel exceeds 350 K

    def test_noisy_background_defeats_the_contextual_criterion(self) -> None:
        from omni_mercury_engine.detectors.geological.wildfire import WildfireDetector

        detector = WildfireDetector(load_shipped_weights=False)
        scene = np.tile(np.array([[280.0, 320.0]]), (8, 4))
        scene[0, 0] = 340.0  # 40 K over the median, far under 4x the MAD scale
        result = detector.predict_wildfire({"thermal_image": scene})
        assert result.fire_detected is False

    def test_firms_raster_predicate_boundaries(self) -> None:
        from omni_mercury_engine.detectors.geological.wildfire import _is_firms_raster

        base = np.zeros((3, 8, 8))
        base[0, 2, 2] = 200.0  # detected min exactly at the 200 K floor: OK
        base[1, 2, 2] = 15.0  # log1p channels at their 15.0 ceiling: OK
        base[2, 2, 2] = 15.0
        assert _is_firms_raster(base) is True

        cold_detection = base.copy()
        cold_detection[0, 2, 2] = 199.9  # a nonzero detection below 200 K
        assert _is_firms_raster(cold_detection) is False

        hot_frp = base.copy()
        hot_frp[1, 2, 2] = 15.1
        assert _is_firms_raster(hot_frp) is False

        all_empty = np.zeros((3, 8, 8))
        assert _is_firms_raster(all_empty) is True  # nothing detected is a valid raster

    def test_landslide_alert_bar_is_exceeded_not_met(self) -> None:
        from omni_mercury_engine.detectors.geological.landslide import LandslideDetector

        detector = LandslideDetector(load_shipped_weights=False)
        trigger = {"rainfall_data": {"intensity_mm_hr": 20.0, "duration_hours": 6.0}}
        # Physics probability exactly 0.6 (displacement 30 mm/day) is NOT
        # above the 0.6 alert bar.
        at_bar = detector.predict_landslide(
            {**trigger, "sensor_data": {"displacement_rate_mm_day": 30.0}}
        )
        assert at_bar.slope_failure_probability == pytest.approx(0.6)
        assert at_bar.landslide_imminent is False
        over_bar = detector.predict_landslide(
            {**trigger, "sensor_data": {"displacement_rate_mm_day": 32.0}}
        )
        assert over_bar.slope_failure_probability == pytest.approx(0.64)
        assert over_bar.landslide_imminent is True

    def test_landslide_risk_ladder_boundaries_exact(self) -> None:
        from omni_mercury_engine.detectors.geological.landslide import LandslideDetector

        detector = LandslideDetector(load_shipped_weights=False)
        two_triggers = {
            "rainfall_data": {"intensity_mm_hr": 20.0, "duration_hours": 6.0},
            "seismic_data": {"pga_g": 0.3, "magnitude": 7.0, "distance_km": 10.0},
        }
        # p exactly 0.8 with two triggers: extreme needs p STRICTLY above 0.8.
        at_extreme_bar = detector.predict_landslide(
            {**two_triggers, "sensor_data": {"displacement_rate_mm_day": 40.0}}
        )
        assert at_extreme_bar.slope_failure_probability == pytest.approx(0.8)
        assert at_extreme_bar.risk_level == "very_high"
        # p exactly 0.6 with one trigger: very_high needs p strictly above 0.6.
        one_trigger = detector.predict_landslide(
            {
                "rainfall_data": {"intensity_mm_hr": 20.0, "duration_hours": 6.0},
                "sensor_data": {"displacement_rate_mm_day": 30.0},
            }
        )
        assert one_trigger.risk_level == "high"
        # p exactly 0.4 with no trigger: high needs p strictly above 0.4.
        no_trigger = detector.predict_landslide({"sensor_data": {"displacement_rate_mm_day": 20.0}})
        assert no_trigger.slope_failure_probability == pytest.approx(0.4)
        assert no_trigger.risk_level == "moderate"
        # p exactly 0.2: moderate needs p strictly above 0.2.
        low = detector.predict_landslide({"sensor_data": {"displacement_rate_mm_day": 10.0}})
        assert low.risk_level == "low"

    def test_landslide_confidence_is_the_max_of_trigger_and_stability(self) -> None:
        from omni_mercury_engine.detectors.geological.landslide import LandslideDetector

        detector = LandslideDetector(load_shipped_weights=False)
        result = detector.predict_landslide(
            {
                # trigger probability 0.5 (below-threshold rain)...
                "rainfall_data": {"intensity_mm_hr": 5.0, "duration_hours": 6.0},
                # ...and stability probability 0.8.
                "sensor_data": {"displacement_rate_mm_day": 40.0},
            }
        )
        # The sub-threshold trigger contributes no confidence; stability does.
        assert result.rainfall_trigger is False
        assert result.confidence == pytest.approx(0.8)

    def test_cascade_population_threshold_is_strict(self) -> None:
        from omni_mercury_engine.detectors.geological.landslide import LandslideDetector

        detector = LandslideDetector(load_shipped_weights=False)
        base = {
            "rainfall_data": {"intensity_mm_hr": 20.0, "duration_hours": 6.0},
            "sensor_data": {"soil_saturation_pct": 90.0, "displacement_rate_mm_day": 40.0},
        }
        at_100 = detector.predict_landslide(
            {**base, "population_data": {"population_density": 100}}
        )
        assert at_100.landslide_imminent is True
        assert "high_casualty_potential" not in at_100.cascade_risks
        # Saturation 0.9 makes this a mud flow, not a debris flow: no
        # river-blockage cascade either.
        assert at_100.landslide_type == "mud_flow"
        assert "river_blockage_flooding" not in at_100.cascade_risks


class TestVolcanicSurvivorPins:
    """Round-22 kill pins for the volcanic enumeration's surviving mutants.

    Each pin targets measured survivors of the 2026-10-20 full enumeration:
    published constants to the digit, exact Poisson-MLE recovery, mid-band
    threshold probes (a value just past each served boundary), the live
    alert-state decision ladder, and the adaptive optimizer's arithmetic.
    """

    # ---- failure forecast methods ----

    def test_ffm_published_constants(self) -> None:
        from omni_mercury_engine.detectors.geological import volcanic

        assert volcanic.FFM_MIN_R2 == 0.8
        assert volcanic.FFM_MIN_POINTS == 5
        # 95% point of chi-squared with one degree of freedom.
        assert volcanic.FFM_LR_CRITICAL == 3.841

    def test_poisson_mle_recovers_a_known_failure_time(self) -> None:
        """Expected counts of k/(t_f - t) recover t_f to sub-millihour error.

        Bins are 0.5 h wide (kills a loosened edge-increase guard), the first
        bin holds 0.197 expected events (kills an occupancy cut at n > 1),
        and the recovered time and likelihood ratio are pinned tightly
        enough that any mutation of the profile search, the closed-form rate,
        the log terms or the LR scale moves them out of tolerance.
        """
        from omni_mercury_engine.detectors.geological.volcanic import failure_forecast_poisson

        tf, k = 10.4, 4.0
        edges = np.arange(0, 10.5, 0.5)
        counts = k * np.log((tf - edges[:-1]) / (tf - edges[1:]))
        result = failure_forecast_poisson(edges, counts)
        assert result is not None
        assert result["failure_time_h"] == pytest.approx(10.4, abs=1e-3)
        assert result["time_to_failure_h"] == pytest.approx(0.4, abs=1e-3)
        assert result["n_bins"] == 20.0
        # Deterministic statistic for expected counts; measured 2026-10-20.
        assert result["likelihood_ratio"] == pytest.approx(10.4262, abs=0.01)

    def test_poisson_rejects_malformed_and_refuses_no_signal(self) -> None:
        from omni_mercury_engine.detectors.geological.volcanic import failure_forecast_poisson

        edges = np.arange(0, 10.5, 0.5)
        with pytest.raises(InvalidObservationError, match="bin edges"):
            failure_forecast_poisson(np.arange(0, 3.5, 0.5), np.ones(5))
        with pytest.raises(InvalidObservationError, match="increase"):
            failure_forecast_poisson(np.array([0.0, 1.0, 1.0, 2.0, 3.0, 4.0]), np.ones(5))
        # All-zero counts carry no signal: None, never a forecast.
        assert failure_forecast_poisson(edges, np.zeros(20)) is None
        # Deceleration must not beat the constant-rate model.
        decel = 4.0 * np.log((edges[1:] + 5.0) / (edges[:-1] + 5.0))
        assert failure_forecast_poisson(edges, decel) is None

    # ---- live alert state ----

    @staticmethod
    def _live_detector(levels: list[str]) -> tuple[Any, dict[str, Any]]:
        from datetime import UTC, datetime

        from omni_mercury_engine.data_sources.base import DataPoint, DataSourceType
        from omni_mercury_engine.data_sources.live_ingestion import LiveFetch
        from omni_mercury_engine.detectors.geological.volcanic import VolcanicEruptionDetector

        detector = VolcanicEruptionDetector()
        seen: dict[str, Any] = {}
        points = [
            DataPoint(
                source_id="stub",
                source_type=DataSourceType.VOLCANO,
                event_id=f"v{i}",
                timestamp=datetime.now(UTC),
                data={"name": f"Volcano {i}", "alert_level": level, "observatory": "OBS"},
            )
            for i, level in enumerate(levels)
        ]

        def fake_fetch(*, allow_simulated: bool = False, **kwargs: Any) -> LiveFetch:
            seen["allow_simulated"] = allow_simulated
            return LiveFetch(
                source_id="stub",
                data_points=points,
                data_provenance="live",
                fetched_at=datetime.now(UTC),
            )

        detector.fetch_live_data = fake_fetch  # type: ignore[method-assign]
        return detector, seen

    def test_live_quiet_feed_is_normal_with_no_asserted_confidence(self) -> None:
        detector, seen = self._live_detector(["normal", "normal"])
        result = detector.detect_live()
        assert result.alert_level == "normal"
        assert result.confidence == 0.0
        assert result.eruption_imminent is False
        assert result.eruption_type == "no_eruption"
        assert result.early_warning_actions == []
        assert result.live_context["alert_level_counts"] == {"normal": 2}
        assert result.live_context["volcanoes_reported"] == 2
        # Simulated sources stay opt-in: the default fetch asks for real data.
        assert seen["allow_simulated"] is False

    def test_live_advisory_asserts_official_confidence_without_actions(self) -> None:
        detector, _ = self._live_detector(["normal", "advisory"])
        result = detector.detect_live()
        assert result.alert_level == "advisory"
        assert result.confidence == pytest.approx(0.98)
        assert result.eruption_imminent is False
        assert result.eruption_type == "no_eruption"
        assert result.early_warning_actions == []

    def test_live_watch_carries_guidance_but_not_imminence(self) -> None:
        detector, _ = self._live_detector(["watch"])
        result = detector.detect_live()
        assert result.alert_level == "watch"
        assert result.eruption_imminent is False
        assert result.eruption_type == "undetermined"
        assert result.early_warning_actions == ["USGS WATCH for Volcano 0: follow OBS guidance."]

    def test_live_warning_is_the_only_imminent_level(self) -> None:
        detector, _ = self._live_detector(["advisory", "warning"])
        result = detector.detect_live()
        assert result.alert_level == "warning"
        assert result.eruption_imminent is True
        assert result.confidence == pytest.approx(0.98)
        assert result.live_context["alert_level_counts"] == {"advisory": 1, "warning": 1}
        assert result.live_context["highest_alert_volcano"]["name"] == "Volcano 1"

    def test_live_unknown_levels_assert_nothing(self) -> None:
        detector, _ = self._live_detector(["unassigned"])
        result = detector.detect_live()
        assert result.alert_level == "normal"
        assert result.confidence == 0.0
        assert result.live_context["alert_level_counts"] == {"unassigned": 1}

    def test_live_name_filter_is_case_insensitive(self) -> None:
        detector, _ = self._live_detector(["advisory", "watch"])
        result = detector.detect_live(volcano_name="volcano 0")
        assert result.live_context["volcanoes_reported"] == 1
        assert result.alert_level == "advisory"

    # ---- component mid-band threshold probes ----

    def test_thermal_intensity_mid_band_probes(self) -> None:
        from omni_mercury_engine.detectors.geological.volcanic import ThermalHotspotDetector

        detector = ThermalHotspotDetector()
        for temp, intensity in ((320.5, "moderate"), (350.5, "high"), (400.5, "extreme")):
            result = detector.detect_thermal_anomaly({"brightness_temperature_k": np.array([temp])})
            assert result["intensity"] == intensity, temp

    def test_thermal_validation_band_edges(self) -> None:
        from omni_mercury_engine.detectors.geological.volcanic import ThermalHotspotDetector

        detector = ThermalHotspotDetector()
        # Just inside the physical band on both sides: accepted.
        ok = detector.detect_thermal_anomaly(
            {"brightness_temperature_k": np.array([150.05, 1999.95]), "radiant_heat_mw": 0.05}
        )
        assert ok["max_temp_k"] == pytest.approx(1999.95)
        # Just outside: refused.
        with pytest.raises(InvalidObservationError):
            detector.detect_thermal_anomaly({"brightness_temperature_k": np.array([2000.05])})
        with pytest.raises(InvalidObservationError):
            detector.detect_thermal_anomaly({"brightness_temperature_k": np.array([149.95])})

    def test_thermal_hotspot_count_uses_three_sigma(self) -> None:
        """One pixel sits between 3 and 4 robust sigmas: exactly 2 hotspots."""
        from omni_mercury_engine.detectors.geological.volcanic import ThermalHotspotDetector

        values = np.full(100, 300.0)
        values[0] = 340.0
        values[1] = 315.0  # above mean+3*std (313.26), below mean+4*std (317.50)
        result = ThermalHotspotDetector().detect_thermal_anomaly(
            {"brightness_temperature_k": values}
        )
        assert result["hotspot_pixel_count"] == 2

    def test_gas_mid_band_probes(self) -> None:
        from omni_mercury_engine.detectors.geological.volcanic import GasEmissionAnalyzer

        analyzer = GasEmissionAnalyzer()
        so2_only = analyzer.analyze_gas_emissions(
            {
                "so2_tons_per_day": 3.05 * analyzer.baseline_so2_tons_day,
                "co2_tons_per_day": analyzer.baseline_co2_tons_day,
            }
        )
        assert so2_only["so2_anomaly"] is True
        assert so2_only["co2_anomaly"] is False
        assert so2_only["degassing_index"] == pytest.approx(2.025)
        assert so2_only["risk_level"] == "moderate"
        critical = analyzer.analyze_gas_emissions(
            {
                "so2_tons_per_day": 8.05 * analyzer.baseline_so2_tons_day,
                "co2_tons_per_day": 2.05 * analyzer.baseline_co2_tons_day,
            }
        )
        assert critical["co2_anomaly"] is True
        assert critical["degassing_index"] == pytest.approx(5.05)
        assert critical["risk_level"] == "critical"

    def test_insar_mid_band_probes(self) -> None:
        from omni_mercury_engine.detectors.geological.volcanic import InSARDeformationDetector

        detector = InSARDeformationDetector()
        moderate = detector.detect_deformation(
            {
                "vertical_displacement_cm": 0.5,
                "horizontal_displacement_cm": 5.0252,
                "deformation_rate_cm_day": 0.05,
            }
        )
        assert moderate["deformation_detected"] is True
        assert moderate["deformation_type"] == "inflation"
        assert moderate["severity"] == "moderate"
        high = detector.detect_deformation({"vertical_displacement_cm": -10.05})
        assert high["severity"] == "high"
        assert high["deformation_type"] == "deflation"
        critical = detector.detect_deformation({"vertical_displacement_cm": 20.05})
        assert critical["severity"] == "critical"

    # ---- deterministic swarm physics ----

    def test_swarm_physics_exact_confidence_and_bar(self) -> None:
        from omni_mercury_engine.detectors.geological.volcanic import VolcanicEruptionDetector

        physics = VolcanicEruptionDetector._detect_swarm_physics
        # 4 of 100 samples exceed 3 robust sigma: confidence 0.04/0.10 = 0.4.
        swarm = physics(np.array([0.0, 1.0] * 48 + [100.0] * 4))
        assert swarm["swarm_detected"] is True
        assert swarm["confidence"] == pytest.approx(0.4)
        # Exactly 2% exceedance sits ON the bar: not a swarm (strict >).
        at_bar = physics(np.concatenate([np.zeros(49), np.ones(49), np.full(2, 100.0)]))
        assert at_bar["swarm_detected"] is False
        assert at_bar["confidence"] == pytest.approx(0.2)
        # The minimum window is exactly 4 samples, and confidence caps at 1.0.
        four = physics(np.array([1.0, 1.0, 2.0, 50.0]))
        assert four["swarm_detected"] is True
        assert four["confidence"] == 1.0
        three = physics(np.array([1.0, 1.0, 50.0]))
        assert three["swarm_detected"] is False
        assert three["confidence"] == 0.0

    # ---- deterministic physics forecast ----

    def test_physics_forecast_clips_and_bar(self) -> None:
        from omni_mercury_engine.detectors.geological.volcanic import VolcanicEruptionDetector

        forecast = VolcanicEruptionDetector._forecast_eruption_physics
        # Exactly 0.7 is not imminent (strict >).
        at_bar = forecast({"hmm_eruption_prob": 0.7})
        assert at_bar["eruption_imminent"] is False
        assert at_bar["confidence"] == pytest.approx(0.7)
        # Each severity clips to exactly 1.0 at saturation; an unclipped term
        # would push the noisy-OR off 1.0.
        for signals in (
            {"seismic_confidence": 1.05},
            {"degassing_index": 5.4},
            {"total_displacement_cm": 21.0},
            {"radiant_heat_mw": 1050.0},
        ):
            saturated = forecast(signals)
            assert saturated["confidence"] == 1.0, signals
            assert saturated["eruption_imminent"] is True
        # Baseline gas (index 1.0) contributes nothing.
        quiet = forecast({"degassing_index": 1.0})
        assert quiet["confidence"] == 0.0

    def test_alert_ladder_strict_boundaries(self) -> None:
        from omni_mercury_engine.detectors.geological.volcanic import VolcanicEruptionDetector

        detector = VolcanicEruptionDetector(enable_hmm=False)
        assert detector._determine_alert_level(3, 0.8) == "watch"
        assert detector._determine_alert_level(2, 0.6) == "advisory"
        assert detector._determine_alert_level(1, 0.4) == "normal"

    # ---- adaptive optimizer arithmetic ----

    def test_optimizer_defaults_and_adaptation_arithmetic(self) -> None:
        from omni_mercury_engine.detectors.geological.volcanic import (
            RefactoringAdaptiveOptimizer,
        )

        fresh = RefactoringAdaptiveOptimizer()
        assert fresh.learning_rate == 0.01
        assert fresh.phi == pytest.approx(1.618033988749895)
        assert fresh.history_window == 100
        assert fresh.threshold_adjustment == 0.0

        # Fewer than 10 errors: parameters returned unchanged.
        fresh.error_history = [0.9] * 9
        assert fresh.adapt_parameters()["confidence_calibration"] == 1.0
        # Exactly 10: adaptation runs.
        fresh.error_history = [0.9] * 10
        assert fresh.adapt_parameters()["confidence_calibration"] == pytest.approx(0.99)

        # Mean exactly 0.5 adapts in neither direction.
        flat = RefactoringAdaptiveOptimizer()
        flat.error_history = [0.5] * 20
        assert flat.adapt_parameters()["confidence_calibration"] == 1.0

        # Low error raises calibration by lr/phi exactly.
        low = RefactoringAdaptiveOptimizer()
        low.error_history = [0.1] * 20
        adapted = low.adapt_parameters()
        assert adapted["confidence_calibration"] == pytest.approx(1 + 0.01 / low.phi)
        assert adapted["refactoring_score"] == pytest.approx(0.9)

        # The error window is the last 20: five older 1.0s must not leak in.
        windowed = RefactoringAdaptiveOptimizer()
        windowed.error_history = [1.0] * 5 + [0.6] * 20
        assert windowed.adapt_parameters()["refactoring_score"] == pytest.approx(0.4)

        # Rising errors nudge the threshold up by lr * 0.1 exactly.
        rising = RefactoringAdaptiveOptimizer()
        rising.error_history = list(np.linspace(0.0, 0.38, 20))
        assert rising.adapt_parameters()["threshold_adjustment"] == pytest.approx(0.001)

        # Calibration clips at 1.5.
        capped = RefactoringAdaptiveOptimizer()
        capped.confidence_calibration = 1.4999
        capped.error_history = [0.1] * 20
        assert capped.adapt_parameters()["confidence_calibration"] == 1.5

    def test_optimizer_threshold_and_error_metrics(self) -> None:
        from omni_mercury_engine.detectors.geological.volcanic import (
            RefactoringAdaptiveOptimizer,
        )

        optimizer = RefactoringAdaptiveOptimizer()
        assert optimizer.get_adapted_threshold(0.5) == 0.5
        optimizer.threshold_adjustment = 0.1
        assert optimizer.get_adapted_threshold(0.5) == pytest.approx(0.6)
        # VEI error: a None estimate counts as 0, normalised by max VEI 8.
        assert optimizer._compute_prediction_error(
            {"vei_estimate": None}, {"actual_vei": 2}
        ) == pytest.approx(0.25)
        # A prediction without the paired key contributes nothing (and the
        # key pairing is AND, never OR: no KeyError on partial inputs).
        assert optimizer._compute_prediction_error(
            {"eruption_imminent": True}, {"eruption_occurred": True}
        ) == pytest.approx(0.0)
        assert optimizer._compute_prediction_error({}, {}) == 0.5


class TestWildfireSurvivorPins:
    """Round-22 kill pins for the wildfire enumeration's surviving mutants."""

    # ---- module constants (served thresholds) ----

    def test_module_constants(self) -> None:
        from omni_mercury_engine.detectors.geological import wildfire

        assert wildfire.HOTSPOT_THRESHOLD_K == 350.0
        assert wildfire.FIRE_PROB_THRESHOLD_DEFAULT == 0.6
        assert wildfire.ONE_HOUR_FUEL_MOISTURE_PER_EMC == 1.03
        # GTR INT-169 Table 6, by NFDRS state-of-weather code.
        assert wildfire.FUEL_INTERFACE_ADJUSTMENT == {
            0: (25.0, 0.75),
            1: (19.0, 0.83),
            2: (12.0, 0.91),
            3: (5.0, 1.0),
        }
        assert wildfire._WEATHER_INPUTS == {
            "wind_speed_kmh": ("km/h", 0.0, 500.0),
            "wind_direction_deg": ("deg", 0.0, 360.0),
            "relative_humidity_pct": ("%", 0.0, 100.0),
            "temperature_c": ("degC", -90.0, 60.0),
        }

    def test_firms_raster_check_boundaries(self) -> None:
        from omni_mercury_engine.detectors.geological.wildfire import _is_firms_raster

        raster = np.zeros((3, 4, 4))
        raster[0, 0, 0] = 700.0  # inclusive upper brightness bound
        assert _is_firms_raster(raster) is True
        raster[0, 0, 0] = 700.05
        assert _is_firms_raster(raster) is False
        raster[0, 0, 0] = 400.0
        raster[1, 0, 0] = 15.05  # log1p(FRP) cannot reach 15
        assert _is_firms_raster(raster) is False

    # ---- brightness-temperature ignition physics ----

    def test_contextual_fire_just_over_330(self) -> None:
        """330.05 K over a flat 300 K background is a (weak) contextual fire."""
        from omni_mercury_engine.detectors.geological.wildfire import WildfireDetector

        values = np.full(100, 300.0)
        values[0] = 330.05
        result = WildfireDetector._detect_ignition_physics(values)
        assert result["fire_detected"] is True
        assert result["confidence"] == pytest.approx(0.05 / 60.0)

    def test_contextual_needs_four_times_the_robust_scale(self) -> None:
        """A 27.5 K excursion over scale 11.12 is NOT 4-sigma contextual.

        4 * scale = 44.48 while 4 + scale = 15.12: the multiplicative form
        rejects this scene, the additive mutation would accept it.
        """
        from omni_mercury_engine.detectors.geological.wildfire import WildfireDetector

        values = np.concatenate([np.full(50, 300.0), np.full(49, 315.0), [335.0]])
        result = WildfireDetector._detect_ignition_physics(values)
        assert result["fire_detected"] is False

    def test_absolute_fire_just_over_360(self) -> None:
        """A flat 360.05 K scene is absolute combustion (and all-hot: conf 1)."""
        from omni_mercury_engine.detectors.geological.wildfire import WildfireDetector

        result = WildfireDetector._detect_ignition_physics(np.full(50, 360.05))
        assert result["fire_detected"] is True
        assert result["confidence"] == 1.0

    def test_confidence_noisy_or_of_temperature_and_hot_fraction(self) -> None:
        """98x300 K + 2x380 K: ts = 50/60, hf = 0.02*20 = 0.4, conf = 0.9."""
        from omni_mercury_engine.detectors.geological.wildfire import WildfireDetector

        values = np.concatenate([np.full(98, 300.0), [380.0, 380.0]])
        result = WildfireDetector._detect_ignition_physics(values)
        assert result["confidence"] == pytest.approx(0.9)
        # A 10% hot fraction saturates the fraction term exactly at 1.0.
        saturated = WildfireDetector._detect_ignition_physics(
            np.concatenate([np.full(90, 300.0), np.full(10, 380.0)])
        )
        assert saturated["confidence"] == 1.0

    def test_physics_spatial_intermediates_on_a_2d_raster(self) -> None:
        """Two separate hot clusters: exact mask, centroids and sizes."""
        from omni_mercury_engine.detectors.geological.wildfire import WildfireDetector

        image = np.full((8, 8), 300.0)
        image[0, 0:2] = 380.0
        image[6:8, 6] = 380.0
        result = WildfireDetector._detect_ignition_physics(image)
        assert result["hotspot_count"] == 4
        assert result["hotspot_mask"].shape == (8, 8)
        assert result["ignition_centroids"].tolist() == [[0.0, 0.5], [6.5, 6.0]]
        assert result["ignition_component_sizes"].tolist() == [2, 2]
        # A bare 1-D series has no pixel geometry: flat exceedance count only.
        series = WildfireDetector._detect_ignition_physics(np.array([300.0, 351.0, 352.0]))
        assert series["hotspot_count"] == 2
        assert series["hotspot_mask"].shape == (0, 0)

    # ---- thermal resonance and smoke ----

    def test_resonance_exact_pins_on_a_pure_diurnal_tone(self) -> None:
        """96 h of a 24-h sine: dominant bin pins exactly.

        Power of the dominant bin is (n*A/2)^2 = (96*5/2)^2 = 57600; the
        diurnal band holds the dominant tone so its ratio is 1.0 and the
        phi-weighted aggregate is 1/phi = 0.618034 (second band weight
        normalisation), measured 2026-10-20.
        """
        from omni_mercury_engine.detectors.geological.wildfire import (
            ResonanceFrequencyAnalyzer,
        )

        analyzer = ResonanceFrequencyAnalyzer()
        assert analyzer.fire_frequencies == {
            "diurnal": pytest.approx(1.0 / 24.0),
            "wind_gust": 2.0,
            "convective": 0.5,
            "spotting": 4.0,
        }
        t = np.arange(96.0)
        series = 5.0 * np.sin(2 * np.pi * t / 24.0) + 100.0
        result = analyzer.analyze_thermal_resonance(series)
        assert result["dominant_frequency"] == pytest.approx(1.0 / 24.0)
        assert result["dominant_power"] == pytest.approx(57600.0)
        assert result["resonance_score"] == pytest.approx(0.618034, abs=1e-5)
        assert result["fire_behavior_detected"] is True
        # Regression (fixed 2026-10-20): scores used to be zipped against the
        # band names BY POSITION, so with no bin in the wind_gust band the
        # convective band's score was reported as "wind_gust" and the last
        # two bands vanished from the dict.
        components = result["frequency_components"]
        assert list(components) == ["diurnal", "wind_gust", "convective", "spotting"]
        assert components["diurnal"] == pytest.approx(1.0, abs=1e-9)
        assert components["wind_gust"] == 0.0
        assert components["spotting"] == 0.0

    def test_resonance_minimum_window_is_four_samples(self) -> None:
        """Exactly 4 samples runs the full path (the early return lacks keys)."""
        from omni_mercury_engine.detectors.geological.wildfire import (
            ResonanceFrequencyAnalyzer,
        )

        full = ResonanceFrequencyAnalyzer().analyze_thermal_resonance(
            np.array([1.0, 5.0, 2.0, 8.0])
        )
        assert "dominant_power" in full
        short = ResonanceFrequencyAnalyzer().analyze_thermal_resonance(np.array([1.0, 5.0, 2.0]))
        assert "dominant_power" not in short
        assert short["resonance_score"] == 0.0

    def test_smoke_trend_boundaries_and_cap(self) -> None:
        from omni_mercury_engine.detectors.geological.wildfire import (
            ResonanceFrequencyAnalyzer,
        )

        smoke = ResonanceFrequencyAnalyzer().analyze_smoke_patterns
        rising = smoke(np.array([0.0, 0.6, 1.2, 1.8, 2.4]))
        assert rising["smoke_trend"] == "increasing"
        assert rising["spread_indicator"] == 1.0  # min(1.0, 0.6 * 2)
        assert rising["rapid_change_detected"] is True  # 0.6 > 0.5 strictly
        # A mean change of exactly -0.1 is stable, not decreasing (strict <).
        at_bar = smoke(np.array([0.4, 0.3, 0.2, 0.1, 0.0]))
        assert at_bar["smoke_trend"] == "stable"
        assert at_bar["spread_indicator"] == pytest.approx(0.3)
        # Exactly 4 samples is enough for the full analysis.
        four = smoke(np.array([0.0, 0.2, 0.4, 0.6]))
        assert four["smoke_trend"] == "increasing"

    # ---- NFDRS interface validation band edges ----

    def test_fuel_moisture_validation_band_edges(self) -> None:
        from omni_mercury_engine.detectors.geological.wildfire import (
            equilibrium_moisture_content,
            fuel_interface_moisture,
        )

        # Just inside every bound: accepted.
        assert equilibrium_moisture_content(59.95, 99.95) > 0
        assert equilibrium_moisture_content(-89.95, 0.05) > 0
        assert fuel_interface_moisture(59.95, 99.95, 0)["equilibrium_moisture_pct"] > 0
        # Just outside: refused.
        for temp, hum in ((60.05, 50.0), (-90.05, 50.0), (20.0, 100.05), (20.0, -0.05)):
            with pytest.raises(InvalidObservationError):
                equilibrium_moisture_content(temp, hum)
            with pytest.raises(InvalidObservationError):
                fuel_interface_moisture(temp, hum, 0)

    # ---- spread model bounds and wind gate ----

    def test_spread_bounds_band_edges_and_wind_gate(self) -> None:
        from omni_mercury_engine.detectors.geological.wildfire import FireSpreadModel

        model = FireSpreadModel()
        # Just-inside values for every validated field are accepted.
        result = model.predict_spread(
            fire_data={"fuel_load_tons_ha": 499.95, "fuel_moisture_pct": 99.95},
            weather_data={
                "wind_speed_kmh": 299.95,
                "wind_direction_deg": 359.95,
                "relative_humidity_pct": 0.05,
                "temperature_c": 59.95,
            },
        )
        assert result["spread_rate_km_hr"] is not None
        assert result["spread_direction_deg"] == pytest.approx(359.95)
        assert result["wind_driven"] is True
        # Just-outside values are refused.
        with pytest.raises(InvalidObservationError):
            model.predict_spread(fire_data={}, weather_data={"wind_speed_kmh": 300.05})
        with pytest.raises(InvalidObservationError):
            model.predict_spread(
                fire_data={"fuel_load_tons_ha": 500.05},
                weather_data={"wind_speed_kmh": 10.0},
            )
        # Wind of exactly 20 km/h is not wind-driven (strict >).
        calm = model.predict_spread(fire_data={}, weather_data={"wind_speed_kmh": 20.0})
        assert calm["wind_driven"] is False

    # ---- fire risk ladder mid-band probes ----

    def test_fire_risk_points_mid_band(self) -> None:
        from omni_mercury_engine.detectors.geological.wildfire import (
            WildfireDetector,
            WildfirePredictionResult,
        )

        detector = WildfireDetector(load_shipped_weights=False)
        fire = WildfirePredictionResult(fire_detected=True, confidence=0.9, risk_level="low")
        no_fire = WildfirePredictionResult(fire_detected=False, confidence=0.0, risk_level="low")
        weather_hot = {
            "weather_data": {
                "wind_speed_kmh": 30.5,
                "relative_humidity_pct": 19.5,
                "temperature_c": 35.5,
            }
        }
        # 2 (fire) + 3 weather points = 5 -> extreme; without fire 3 -> high.
        assert detector._assess_fire_risk(weather_hot, fire) == "extreme"
        assert detector._assess_fire_risk(weather_hot, no_fire) == "high"
        # Boundary values score no weather points (strict comparisons).
        weather_at_bars = {
            "weather_data": {
                "wind_speed_kmh": 30.0,
                "relative_humidity_pct": 20.0,
                "temperature_c": 35.0,
            }
        }
        assert detector._assess_fire_risk(weather_at_bars, fire) == "moderate"
        # Missing weather defaults to benign (wind 0, humidity 100, temp 0).
        assert detector._assess_fire_risk({}, no_fire) == "low"
        # Weather just inside its physical bounds is accepted.
        assert (
            detector._assess_fire_risk(
                {
                    "weather_data": {
                        "wind_speed_kmh": 499.95,
                        "relative_humidity_pct": 0.05,
                        "temperature_c": 59.95,
                    }
                },
                no_fire,
            )
            == "high"
        )

    # ---- architecture = checkpoint compatibility ----

    def test_ignition_cnn_checkpoint_shape_contract(self) -> None:
        from omni_mercury_engine.detectors.geological.wildfire import FireIgnitionDetector

        shapes = {
            k: tuple(v.shape)
            for k, v in FireIgnitionDetector().state_dict().items()
            if "weight" in k and v.dim() >= 2
        }
        assert shapes == {
            "thermal_cnn.0.weight": (32, 3, 3, 3),
            "thermal_cnn.4.weight": (64, 32, 3, 3),
            "thermal_cnn.8.weight": (128, 64, 3, 3),
            "fire_classifier.0.weight": (256, 2048),
            "fire_classifier.3.weight": (64, 256),
            "fire_classifier.5.weight": (1, 64),
        }

    def test_wildfire_cnn_checkpoint_shape_contract(self) -> None:
        from omni_mercury_engine.detectors.geological.wildfire import WildfireCNN

        model = WildfireCNN()
        shapes = {
            k: tuple(v.shape)
            for k, v in model.state_dict().items()
            if "weight" in k and v.dim() >= 2
        }
        assert shapes == {
            "conv_block1.0.weight": (32, 4, 3, 3),
            "conv_block1.3.weight": (32, 32, 3, 3),
            "conv_block2.0.weight": (64, 32, 3, 3),
            "conv_block2.3.weight": (64, 64, 3, 3),
            "conv_block3.0.weight": (128, 64, 3, 3),
            "conv_block3.3.weight": (207, 128, 3, 3),  # int(128 * phi)
            "fire_head.0.weight": (256, 3312),
            "fire_head.3.weight": (64, 256),
            "fire_head.5.weight": (1, 64),
            "severity_head.0.weight": (256, 3312),
            "severity_head.3.weight": (5, 256),
        }
        assert len(model.state_dict()) == 52


class TestLandslideSurvivorPins:
    """Round-22 kill pins for the landslide enumeration's surviving mutants."""

    def test_result_dataclass_defaults(self) -> None:
        from omni_mercury_engine.detectors.geological.landslide import (
            LandslidePredictionResult,
        )

        result = LandslidePredictionResult(
            landslide_imminent=False,
            confidence=0.0,
            risk_level="low",
            landslide_type="debris_flow",
        )
        assert result.slope_failure_probability == 0.0
        assert result.rainfall_trigger is False
        assert result.seismic_trigger is False
        assert result.snowmelt_trigger is False

    def test_rainfall_defaults_contribute_nothing(self) -> None:
        """Duration without intensity is zero intensity, zero probability."""
        from omni_mercury_engine.detectors.geological.landslide import RainfallTriggerModel

        result = RainfallTriggerModel().assess_rainfall_trigger({"duration_hours": 4.0})
        assert result["rainfall_trigger"] is False
        assert result["trigger_probability"] == 0.0

    def test_seismic_trigger_arithmetic_pins(self) -> None:
        from omni_mercury_engine.detectors.geological.landslide import SeismicTriggerModel

        model = SeismicTriggerModel()
        # (0.03/0.15) * (1 + (6.5-5)/3) * (1 + (1 - 25/50)) = 0.2*1.5*1.5.
        mid = model.assess_seismic_trigger({"pga_g": 0.03, "magnitude": 6.5, "distance_km": 25.0})
        assert mid["seismic_trigger"] is False
        assert mid["trigger_probability"] == pytest.approx(0.45)
        # Probability caps at 1.0, and the trigger needs pga > 0.15 g.
        capped = model.assess_seismic_trigger(
            {"pga_g": 0.09, "magnitude": 6.5, "distance_km": 25.0}
        )
        assert capped["seismic_trigger"] is False
        assert capped["trigger_probability"] == 1.0
        # Defaults: magnitude 0 and distance 100 km contribute no factors.
        bare = model.assess_seismic_trigger({"pga_g": 0.06})
        assert bare["trigger_probability"] == pytest.approx(0.4)

    def test_lag_features_default_sample_rate_and_window_boundary(self) -> None:
        from omni_mercury_engine.detectors.geological.landslide import (
            TemporalLagFeatureExtractor,
        )

        extractor = TemporalLagFeatureExtractor()
        assert extractor.lag_hours == [1, 6, 12, 24, 48, 72]
        # Default 1 Hz: the 1-hour lag on arange(8000) reads sample -3600.
        features = extractor.extract_lag_features(np.arange(8000.0))
        assert features[0] == 4400.0
        assert features[1] == (4400.0 + 7999.0) * 3600 / 2
        assert features[2] == pytest.approx(1.0)
        # A lag of exactly the series length is skipped (features stay 0).
        exact = extractor.extract_lag_features(np.arange(21600.0))
        assert exact[3] == 0.0 and exact[4] == 0.0 and exact[5] == 0.0
        # One sample longer: the 6 h lag engages with the exact lagged value.
        over = extractor.extract_lag_features(np.arange(21601.0))
        assert over[3] == 1.0  # series[-21600]
        assert over[4] == (1.0 + 21600.0) * 21600 / 2

    def test_ensemble_classifier_reproducibility_contract(self) -> None:
        from omni_mercury_engine.detectors.geological.landslide import (
            SVMRFEnsembleClassifier,
        )

        ensemble = SVMRFEnsembleClassifier()
        assert ensemble.svm.probability is True
        assert ensemble.svm.random_state == 42
        assert ensemble.rf.n_estimators == 100
        assert ensemble.rf.random_state == 42
        assert ensemble.is_fitted is False

    def test_untrained_when_shipped_weights_declined(self) -> None:
        from omni_mercury_engine.detectors.geological.landslide import LandslideDetector

        detector = LandslideDetector(load_shipped_weights=False)
        assert detector._neural_trained is False

    def test_snowmelt_trigger_mid_band(self) -> None:
        from omni_mercury_engine.detectors.geological.landslide import LandslideDetector

        detector = LandslideDetector(load_shipped_weights=False)
        result = detector.predict_landslide({"weather_data": {"snowmelt_mm_day": 20.05}})
        assert result.snowmelt_trigger is True
        assert result.landslide_type == "snow_avalanche"
        at_bar = detector.predict_landslide({"weather_data": {"snowmelt_mm_day": 20.0}})
        assert at_bar.snowmelt_trigger is False

    def test_physics_probability_exact_and_risk_ladder_boundaries(self) -> None:
        from omni_mercury_engine.detectors.geological.landslide import (
            LandslideDetector,
            LandslidePredictionResult,
        )

        detector = LandslideDetector(load_shipped_weights=False)
        result = LandslidePredictionResult(
            landslide_imminent=False,
            confidence=0.0,
            risk_level="low",
            landslide_type="debris_flow",
        )
        # displacement 20 mm/day alone: severity 20/50 = 0.4 exactly.
        result.displacement_rate_mm_day = 20.0
        physics = detector._assess_slope_stability_physics(result)
        assert physics["failure_probability"] == 0.4
        # p exactly 0.4 with no triggers: moderate (strict >).
        result.slope_failure_probability = 0.4
        assert detector._determine_risk_level(0, result) == "moderate"
        # One trigger with p just over 0.6: very_high; exactly 0.6: high.
        result.slope_failure_probability = 0.65
        assert detector._determine_risk_level(1, result) == "very_high"
        result.slope_failure_probability = 0.6
        assert detector._determine_risk_level(1, result) == "high"
        # p exactly 0.2 with nothing else: low (strict >).
        result.slope_failure_probability = 0.2
        assert detector._determine_risk_level(0, result) == "low"

    def test_imminence_needs_trigger_and_bar_jointly(self) -> None:
        """p > 0.6 with no trigger, and a trigger with p <= 0.6, both stay calm."""
        from omni_mercury_engine.detectors.geological.landslide import LandslideDetector

        detector = LandslideDetector(load_shipped_weights=False)
        # Saturated steep slope: p = 1 - (1-0)(1-geo) with geo high, no trigger.
        quiet = detector.predict_landslide(
            {
                "slope_data": {"slope_angle_deg": 45.0},
                "sensor_data": {"soil_saturation_pct": 95.0, "displacement_rate_mm_day": 45.0},
            }
        )
        assert quiet.slope_failure_probability > 0.6
        assert quiet.landslide_imminent is False
        # A rainfall trigger with a quiet slope: no imminence either.
        triggered = detector.predict_landslide(
            {"rainfall_data": {"intensity_mm_hr": 30.0, "duration_hours": 8.0}}
        )
        assert triggered.rainfall_trigger is True
        assert triggered.slope_failure_probability <= 0.6
        assert triggered.landslide_imminent is False

    def test_evacuation_zones_follow_type(self) -> None:
        from omni_mercury_engine.detectors.geological.landslide import (
            LandslideDetector,
            LandslidePredictionResult,
        )

        detector = LandslideDetector(load_shipped_weights=False)
        result = LandslidePredictionResult(
            landslide_imminent=True,
            confidence=0.9,
            risk_level="extreme",
            landslide_type="debris_flow",
        )
        zones = detector._identify_evacuation_zones(result)
        assert zones == [
            "immediate_downslope_area",
            "potential_runout_path",
            "drainage_channels",
        ]
        result.landslide_type = "snow_avalanche"
        assert "avalanche_path" in detector._identify_evacuation_zones(result)

    def test_cascade_population_strictly_above_100(self) -> None:
        from omni_mercury_engine.detectors.geological.landslide import (
            LandslideDetector,
            LandslidePredictionResult,
        )

        detector = LandslideDetector(load_shipped_weights=False)
        result = LandslidePredictionResult(
            landslide_imminent=True,
            confidence=0.9,
            risk_level="extreme",
            landslide_type="earth_flow",
        )
        dense = detector._assess_cascade_risks(
            result, {"population_data": {"population_density": 100.5}}
        )
        assert "high_casualty_potential" in dense
        at_bar = detector._assess_cascade_risks(
            result, {"population_data": {"population_density": 100}}
        )
        assert "high_casualty_potential" not in at_bar

    def test_stability_model_checkpoint_shape_contract(self) -> None:
        from omni_mercury_engine.detectors.geological.landslide import SlopeStabilityModel

        shapes = {
            k: tuple(v.shape)
            for k, v in SlopeStabilityModel().state_dict().items()
            if "weight" in k and v.dim() >= 2
        }
        assert shapes == {
            "feature_encoder.0.weight": (207, 64),  # int(128 * phi)
            "feature_encoder.4.weight": (103, 207),
            "feature_encoder.7.weight": (64, 103),
            "stability_predictor.0.weight": (32, 64),
            "stability_predictor.2.weight": (1, 32),
            "type_classifier.0.weight": (32, 64),
            "type_classifier.2.weight": (6, 32),
        }


class TestVolcanicArchitecturePins:
    """Checkpoint-compatibility shape contracts for the volcanic networks."""

    def test_seismic_swarm_lstm_shape_contract(self) -> None:
        from omni_mercury_engine.detectors.geological.volcanic import SeismicSwarmDetector

        model = SeismicSwarmDetector()
        shapes = {
            k: tuple(v.shape)
            for k, v in model.state_dict().items()
            if "weight" in k and v.dim() >= 2
        }
        # 3 bidirectional LSTM layers, hidden 64, input 32.
        assert shapes["lstm.weight_ih_l0"] == (256, 32)
        assert shapes["lstm.weight_hh_l0"] == (256, 64)
        assert shapes["lstm.weight_ih_l2_reverse"] == (256, 128)
        assert shapes["attention.0.weight"] == (64, 128)
        assert shapes["attention.2.weight"] == (1, 64)
        assert shapes["swarm_classifier.0.weight"] == (64, 128)
        assert shapes["swarm_classifier.3.weight"] == (1, 64)
        assert len(model.state_dict()) == 32

    def test_seismic_swarm_forward_contract(self) -> None:
        """Attention normalises over TIME and the output is one probability."""
        import torch

        from omni_mercury_engine.detectors.geological.volcanic import SeismicSwarmDetector

        torch.manual_seed(7)
        model = SeismicSwarmDetector()
        model.eval()
        with torch.no_grad():
            prob, attention = model(torch.randn(2, 12, 32))
        assert prob.shape == (2, 1)
        assert attention.shape == (2, 12)
        assert float(prob.min()) >= 0.0 and float(prob.max()) <= 1.0
        # Softmax over the sequence dimension: weights sum to 1 per sample.
        sums = attention.sum(dim=1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-6)

    def test_eruption_forecast_shape_contract(self) -> None:
        from omni_mercury_engine.detectors.geological.volcanic import EruptionForecastModel

        model = EruptionForecastModel()
        shapes = {
            k: tuple(v.shape)
            for k, v in model.state_dict().items()
            if "weight" in k and v.dim() >= 2
        }
        assert shapes == {
            "feature_fusion.0.weight": (414, 128),  # int(256 * 1.618)
            "feature_fusion.4.weight": (207, 414),  # int(128 * 1.618)
            "feature_fusion.8.weight": (128, 207),
            "eruption_predictor.0.weight": (64, 128),
            "eruption_predictor.2.weight": (1, 64),
            "vei_estimator.0.weight": (32, 128),
            "vei_estimator.2.weight": (8, 32),
            "time_predictor.0.weight": (32, 128),
            "time_predictor.2.weight": (1, 32),
        }
        assert len(model.state_dict()) == 28


class TestCheckpointAndIntegrationPins:
    """Round-22 pins: operating-point validation, neural guards, warn-once."""

    @staticmethod
    def _zeroed(module: Any, **overrides: Any) -> dict[str, Any]:
        """Zero state dict (running_var kept 1) with optional tensor overrides."""
        import torch

        state = {}
        for key, value in module.state_dict().items():
            if "running_var" in key:
                state[key] = torch.ones_like(value)
            else:
                state[key] = torch.zeros_like(value)
        for key, value in overrides.items():
            state[key] = value
        return state

    def test_volcanic_operating_point_validation_and_decision(self, tmp_path: Any) -> None:
        import torch

        from omni_mercury_engine.detectors.geological.volcanic import VolcanicEruptionDetector

        detector = VolcanicEruptionDetector()
        zeroed = self._zeroed(detector.eruption_model)
        # Invalid thresholds are refused outright.
        for tau in (0.0, 1.0, 1.05, float("nan")):
            path = tmp_path / f"bad_{tau}.pt"
            torch.save(
                {"eruption_model": zeroed, "operating_point": {"eruption_prob_threshold": tau}},
                path,
            )
            with pytest.raises(ValueError, match="probability"):
                detector.load_neural_weights(str(path))
        # A small-but-valid tau is accepted exactly.
        good = tmp_path / "good.pt"
        torch.save(
            {"eruption_model": zeroed, "operating_point": {"eruption_prob_threshold": 0.05}},
            good,
        )
        detector.load_neural_weights(str(good))
        assert detector._operating_point == {"eruption_prob_threshold": 0.05}
        assert detector._neural_trained is True

    def test_volcanic_neural_guard_matrix_and_thresholds(self, tmp_path: Any) -> None:
        """Zero weights give probability exactly 0.5: the decision pins land.

        tau = 0.5 is an inclusive decision (0.5 >= 0.5 -> imminent); the
        legacy rule without an operating point is strict > 0.7, pinned with a
        bias-1.0 head (sigmoid(1) = 0.731, inside (0.7, 0.8)).
        """
        import torch

        from omni_mercury_engine.detectors.geological.volcanic import VolcanicEruptionDetector

        detector = VolcanicEruptionDetector()
        path = tmp_path / "tau_half.pt"
        torch.save(
            {
                "eruption_model": self._zeroed(detector.eruption_model),
                "operating_point": {"eruption_prob_threshold": 0.5},
            },
            path,
        )
        detector.load_neural_weights(str(path))
        on_contract = detector._forecast_eruption({"fused_features": np.zeros(128)}, 0, {})
        assert on_contract == {
            "eruption_imminent": True,
            "confidence": pytest.approx(0.5),
            "method": "neural",
        }
        batched = detector._forecast_eruption({"fused_features": np.zeros((1, 128))}, 0, {})
        assert batched["method"] == "neural"
        # Off-contract shapes fall to physics, never crash the model.
        for shape in ((2, 128), (64,), (1, 1, 128)):
            off = detector._forecast_eruption({"fused_features": np.zeros(shape)}, 0, {})
            assert off["method"] == "physics", shape

        # Regression (fixed 2026-10-20): loading a checkpoint WITHOUT an
        # operating point over one that carried a tau used to keep the stale
        # tau in charge of the new weights. It must reset to the legacy rule.
        legacy = VolcanicEruptionDetector()  # auto-loads the shipped tau
        assert legacy._operating_point is not None
        legacy_path = tmp_path / "legacy.pt"
        import torch as _torch

        torch.save(
            {
                "eruption_model": self._zeroed(
                    legacy.eruption_model,
                    **{"eruption_predictor.2.bias": _torch.tensor([1.0])},
                )
            },
            legacy_path,
        )
        legacy.load_neural_weights(str(legacy_path))
        assert legacy._operating_point is None
        out = legacy._forecast_eruption({"fused_features": np.zeros(128)}, 0, {})
        assert out["eruption_imminent"] is True
        assert out["confidence"] == pytest.approx(0.7310586, abs=1e-6)

    def test_wildfire_operating_point_validation(self, tmp_path: Any) -> None:
        import torch

        from omni_mercury_engine.detectors.geological.wildfire import (
            FIRE_PROB_THRESHOLD_DEFAULT,
            WildfireDetector,
        )

        detector = WildfireDetector(load_shipped_weights=False)
        zeroed = self._zeroed(detector.ignition_detector)
        for tau in (0.0, 1.05):
            path = tmp_path / f"bad_{tau}.pt"
            torch.save(
                {"ignition_detector": zeroed, "operating_point": {"fire_prob_threshold": tau}},
                path,
            )
            with pytest.raises(ValueError, match="probability"):
                detector.load_neural_weights(str(path))
        good = tmp_path / "good.pt"
        torch.save(
            {"ignition_detector": zeroed, "operating_point": {"fire_prob_threshold": 0.25}},
            good,
        )
        detector.load_neural_weights(str(good))
        assert detector._fire_prob_threshold == 0.25
        # A checkpoint without an operating point restores the fixed default.
        plain = tmp_path / "plain.pt"
        torch.save({"ignition_detector": zeroed}, plain)
        detector.load_neural_weights(str(plain))
        assert detector._fire_prob_threshold == FIRE_PROB_THRESHOLD_DEFAULT

    def test_landslide_operating_point_validated_before_any_state_mutates(
        self, tmp_path: Any
    ) -> None:
        import torch

        from omni_mercury_engine.detectors.geological.landslide import LandslideDetector

        detector = LandslideDetector(load_shipped_weights=False)
        zeroed = self._zeroed(detector.stability_model)
        bad = tmp_path / "bad.pt"
        torch.save(
            {"stability_model": zeroed, "operating_point": {"detection_threshold": 1.05}}, bad
        )
        with pytest.raises(ValueError, match="probability"):
            detector.load_neural_weights(str(bad))
        # The invalid rule must not have half-loaded.
        assert detector._neural_trained is False
        assert detector._operating_point is None
        good = tmp_path / "good.pt"
        torch.save(
            {"stability_model": zeroed, "operating_point": {"detection_threshold": 0.35}}, good
        )
        detector.load_neural_weights(str(good))
        assert detector._operating_point == {"detection_threshold": 0.35}
        assert detector._neural_trained is True

    def test_untrained_warning_fires_exactly_once_per_detector(self, caplog: Any) -> None:
        import logging as _logging

        from omni_mercury_engine.detectors.geological.landslide import LandslideDetector
        from omni_mercury_engine.detectors.geological.volcanic import VolcanicEruptionDetector
        from omni_mercury_engine.detectors.geological.wildfire import WildfireDetector

        with caplog.at_level(_logging.WARNING):
            landslide = LandslideDetector(load_shipped_weights=False)
            landslide.predict_landslide({"slope_data": {"slope_angle_deg": 20.0}})
            landslide.predict_landslide({"slope_data": {"slope_angle_deg": 25.0}})
            wildfire = WildfireDetector(load_shipped_weights=False)
            wildfire.predict_wildfire({"thermal_image": np.full((4, 4), 300.0)})
            wildfire.predict_wildfire({"thermal_image": np.full((4, 4), 301.0)})
            volcanic = VolcanicEruptionDetector(load_shipped_weights=False)
            volcanic.predict_eruption({"seismic_sequence": np.ones(16)})
            volcanic.predict_eruption({"seismic_sequence": np.full(16, 2.0)})
        untrained = [r for r in caplog.records if "untrained" in r.getMessage()]
        assert len(untrained) == 3  # one per detector, not one per call

    def test_wildfire_perimeter_diagnostics_and_interface_moisture(self) -> None:
        from omni_mercury_engine.detectors.geological.wildfire import WildfireDetector

        detector = WildfireDetector(load_shipped_weights=False, keep_diagnostics=True)
        image = np.full((8, 8), 300.0)
        image[0, 0:2] = 380.0
        result = detector.predict_wildfire({"thermal_image": image, "pixel_size_km": 0.5})
        assert result.thermal_hotspots == 2
        assert result.fire_perimeter_km2 == pytest.approx(0.5)  # 2 * 0.5^2
        assert result.ignition_locations == [(0.0, 0.5)]
        assert result.diagnostics is not None
        assert result.diagnostics.arrays["hotspot_mask"].shape == (8, 8)
        # Without a pixel size there is no ground-area estimate.
        bare = detector.predict_wildfire({"thermal_image": image})
        assert bare.fire_perimeter_km2 is None
        with pytest.raises(InvalidObservationError):
            detector.predict_wildfire({"thermal_image": image, "pixel_size_km": 0.0})

    def test_interface_moisture_needs_temperature_and_humidity_jointly(self) -> None:
        from omni_mercury_engine.detectors.geological.wildfire import WildfireDetector

        detector = WildfireDetector(load_shipped_weights=False)
        image = np.full((4, 4), 300.0)
        partial = detector.predict_wildfire(
            {
                "thermal_image": image,
                "weather_data": {"nfdrs_state_of_weather": 1, "relative_humidity_pct": 40.0},
            }
        )
        assert sorted(partial.weather_factors) == ["relative_humidity_pct"]
        assert partial.fuel_moisture is None
        full = detector.predict_wildfire(
            {
                "thermal_image": image,
                "weather_data": {
                    "nfdrs_state_of_weather": 1,
                    "relative_humidity_pct": 40.0,
                    "temperature_c": 25.0,
                },
            }
        )
        assert full.fuel_moisture == pytest.approx(6.3079, abs=1e-3)
        assert "equilibrium_moisture_pct" in full.weather_factors

    def test_detector_enable_flags_default_on(self) -> None:
        from omni_mercury_engine.detectors.geological.landslide import LandslideDetector
        from omni_mercury_engine.detectors.geological.volcanic import VolcanicEruptionDetector
        from omni_mercury_engine.detectors.geological.wildfire import WildfireDetector

        volcanic = VolcanicEruptionDetector()
        assert volcanic.enable_seismic and volcanic.enable_thermal
        assert volcanic.enable_gas and volcanic.enable_insar
        assert volcanic.keep_diagnostics is False
        wildfire = WildfireDetector(load_shipped_weights=False)
        assert wildfire.enable_ignition and wildfire.enable_spread
        assert wildfire.keep_diagnostics is False
        landslide = LandslideDetector(load_shipped_weights=False)
        assert landslide.enable_rainfall and landslide.enable_seismic
        assert landslide.enable_stability
        assert landslide.keep_diagnostics is False
