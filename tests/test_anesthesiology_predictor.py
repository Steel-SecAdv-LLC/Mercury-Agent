"""Tests for the anesthesiology predictor and its rule monitors."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

# ``omni_mercury_engine.medical.anesthesiology_predictor`` imports
# ``torch`` at module level; skip cleanly at collection time so the
# rest of the suite is still discoverable without the optional ``ml``
# extra installed.
pytest.importorskip("torch")

from omni_mercury_engine.medical.anesthesiology_predictor import (
    AnesthesiaPredictionResult,
    AnesthesiologyPredictor,
    HemodynamicMonitor,
    SmartInfusionController,
    TIVAMonitoringSystem,
    count_tiva_parameters,
    get_anesthesiology_predictor,
)
from omni_mercury_engine.medical.data_sources import (
    ConfigurationError,
    VitalsDataSource,
    VitalsReading,
)


class _StaticVitalsSource(VitalsDataSource):
    """In-process vitals source backed by a static reading list."""

    name = "static_test_vitals"

    def __init__(self, readings: list[VitalsReading]) -> None:
        self._readings = readings
        self.calls = 0
        self.last_window: int | None = None

    def fetch_recent_vitals(self, window_minutes: int = 5) -> list[VitalsReading]:
        self.calls += 1
        self.last_window = window_minutes
        return list(self._readings)


def _stable_vitals(n: int = 3) -> list[VitalsReading]:
    start = datetime(2024, 3, 22, 13, 30, 0, tzinfo=UTC)
    return [
        VitalsReading(
            timestamp=start + timedelta(seconds=15 * i),
            map_mmhg=82.0,
            hr_bpm=72.0,
            spo2_pct=98.0,
            etco2_mmhg=38.0,
            source="static_test_vitals",
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------------- #
# TIVAMonitoringSystem (neural network) structural tests
# --------------------------------------------------------------------------- #


class TestTIVAMonitoringSystem:
    """TIVA Bi-LSTM structural tests."""

    def test_parameter_count_matches_reference(self) -> None:
        count = count_tiva_parameters()
        # Reference: 164K parameters (164,000 ± small linear-layer drift).
        assert 160_000 <= count <= 170_000, f"Got {count} parameters"

    def test_forward_pass_shapes(self) -> None:
        import torch

        model = TIVAMonitoringSystem()
        x = torch.zeros((1, 60, 8))
        model.eval()
        with torch.no_grad():
            depth, risks, attn = model(x)
        assert depth.shape == (1, 1)
        assert risks.shape == (1, 7)
        assert attn.shape == (1, 60)

    def test_input_dim_is_eight(self) -> None:
        model = TIVAMonitoringSystem()
        assert model.lstm.input_size == 8

    def test_bidirectional(self) -> None:
        model = TIVAMonitoringSystem()
        assert model.lstm.bidirectional is True


# --------------------------------------------------------------------------- #
# SmartInfusionController (PID) tests
# --------------------------------------------------------------------------- #


class TestSmartInfusionController:
    """PID infusion controller behaviour."""

    def test_pid_gains_match_reference(self) -> None:
        controller = SmartInfusionController()
        assert controller.kp == 0.5
        assert controller.ki == 0.1
        assert controller.kd == 0.2
        assert controller.target_bis == 50.0
        assert controller.bis_range == (40.0, 60.0)

    def test_in_range_no_anomaly(self) -> None:
        controller = SmartInfusionController()
        result = controller.compute_infusion_adjustment(50.0, 100.0, 0.2)
        assert result["anomaly_detected"] is False

    def test_low_bis_triggers_deep_anesthesia_alert(self) -> None:
        controller = SmartInfusionController()
        result = controller.compute_infusion_adjustment(30.0, 150.0, 0.3)
        assert result["anomaly_detected"] is True
        joined = " ".join(result["recommendations"])
        assert "Deep anesthesia" in joined or "BIS < 40" in joined

    def test_high_bis_triggers_light_anesthesia_alert(self) -> None:
        controller = SmartInfusionController()
        result = controller.compute_infusion_adjustment(70.0, 80.0, 0.1)
        assert result["anomaly_detected"] is True
        joined = " ".join(result["recommendations"])
        assert "awareness" in joined.lower() or "Light anesthesia" in joined

    def test_propofol_clamped_to_limits(self) -> None:
        controller = SmartInfusionController()
        result = controller.compute_infusion_adjustment(10.0, 0.0, 0.0)
        assert 0.0 <= result["propofol_rate_mcg_kg_min"] <= 200.0

    def test_remifentanil_clamped_to_limits(self) -> None:
        controller = SmartInfusionController()
        result = controller.compute_infusion_adjustment(10.0, 0.0, 0.0)
        assert 0.0 <= result["remifentanil_rate_mcg_kg_min"] <= 0.5

    def test_reset_clears_integrator(self) -> None:
        controller = SmartInfusionController()
        controller.compute_infusion_adjustment(70.0, 100.0, 0.2)
        controller.reset()
        assert controller.integral_error == 0.0
        assert controller.previous_error == 0.0

    def test_dt_must_be_positive(self) -> None:
        controller = SmartInfusionController()
        with pytest.raises(ValueError):
            controller.compute_infusion_adjustment(50.0, 100.0, 0.2, dt=0.0)
        with pytest.raises(ValueError):
            controller.compute_infusion_adjustment(50.0, 100.0, 0.2, dt=-1.0)

    def test_pid_responds_to_persistent_error(self) -> None:
        controller = SmartInfusionController()
        first = controller.compute_infusion_adjustment(70.0, 100.0, 0.2)
        second = controller.compute_infusion_adjustment(70.0, 100.0, 0.2)
        assert first["propofol_adjustment"] != second["propofol_adjustment"]


# --------------------------------------------------------------------------- #
# HemodynamicMonitor tests
# --------------------------------------------------------------------------- #


class TestHemodynamicMonitor:
    """Hemodynamic monitor clinical-range tests."""

    def test_normal_vitals_high_stability(self) -> None:
        monitor = HemodynamicMonitor()
        result = monitor.assess_hemodynamics(
            {
                "mean_arterial_pressure_mmhg": 80.0,
                "heart_rate_bpm": 70.0,
                "oxygen_saturation_pct": 98.0,
                "end_tidal_co2_mmhg": 38.0,
            }
        )
        assert result["hemodynamic_stability"] == pytest.approx(1.0)

    def test_hypotension_detected(self) -> None:
        monitor = HemodynamicMonitor()
        result = monitor.assess_hemodynamics({"mean_arterial_pressure_mmhg": 40.0})
        assert result["risk_scores"]["hypotension"] > 0.0
        assert any("Hypotension" in a for a in result["alerts"])

    def test_hypertension_detected(self) -> None:
        monitor = HemodynamicMonitor()
        result = monitor.assess_hemodynamics({"mean_arterial_pressure_mmhg": 150.0})
        assert result["risk_scores"]["hypertension"] > 0.0

    def test_hypoxemia_triggers_intervention(self) -> None:
        monitor = HemodynamicMonitor()
        result = monitor.assess_hemodynamics({"oxygen_saturation_pct": 85.0})
        assert result["risk_scores"]["hypoxemia"] > 0.0
        assert result["intervention_needed"] is True

    def test_bradycardia_recommendations(self) -> None:
        monitor = HemodynamicMonitor()
        result = monitor.assess_hemodynamics({"heart_rate_bpm": 35.0})
        assert result["risk_scores"]["bradycardia"] > 0.0
        joined = " ".join(result["recommendations"])
        assert "atropine" in joined.lower() or "Bradycardia" in joined

    def test_tachycardia(self) -> None:
        monitor = HemodynamicMonitor()
        result = monitor.assess_hemodynamics({"heart_rate_bpm": 130.0})
        assert result["risk_scores"]["tachycardia"] > 0.0

    def test_hypercarbia_and_hypocarbia(self) -> None:
        monitor = HemodynamicMonitor()
        high = monitor.assess_hemodynamics({"end_tidal_co2_mmhg": 60.0})
        low = monitor.assess_hemodynamics({"end_tidal_co2_mmhg": 20.0})
        assert high["risk_scores"]["hypercarbia"] > 0.0
        assert low["risk_scores"]["hypocarbia"] > 0.0


# --------------------------------------------------------------------------- #
# AnesthesiologyPredictor integration
# --------------------------------------------------------------------------- #


class TestPredictorConfiguration:
    """Configuration semantics + ConfigurationError contract."""

    def test_missing_data_source_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="VitalsDataSource"):
            AnesthesiologyPredictor()

    def test_factory_raises_without_data_source(self) -> None:
        with pytest.raises(ConfigurationError, match="VitalsDataSource"):
            get_anesthesiology_predictor()

    def test_hemodynamics_disabled_allows_no_source(self) -> None:
        predictor = AnesthesiologyPredictor(enable_hemodynamics=False)
        assert predictor.data_source is None
        assert predictor.hemodynamic_monitor is None

    def test_with_source_constructs(self) -> None:
        source = _StaticVitalsSource(_stable_vitals())
        predictor = AnesthesiologyPredictor(source)
        assert predictor.data_source is source
        assert predictor.hemodynamic_monitor is not None

    def test_non_subclass_data_source_rejected(self) -> None:
        with pytest.raises(TypeError, match="VitalsDataSource"):
            AnesthesiologyPredictor(object())  # type: ignore[arg-type]

    def test_factory_returns_predictor_when_disabled(self) -> None:
        predictor = get_anesthesiology_predictor(enable_tiva=False, enable_hemodynamics=False)
        assert isinstance(predictor, AnesthesiologyPredictor)
        assert predictor.tiva_monitor is None


class TestPredictorPipeline:
    """Predictor pipeline behaviour with real VitalsReading inputs."""

    def test_no_data_returns_safe_defaults(self) -> None:
        predictor = AnesthesiologyPredictor(enable_hemodynamics=False)
        result = predictor.predict_anesthesia_risk({})
        assert isinstance(result, AnesthesiaPredictionResult)
        assert result.risk_detected is False

    def test_hemodynamic_path_runs(self) -> None:
        critical = [
            VitalsReading(
                timestamp=datetime(2024, 3, 22, 13, 30, tzinfo=UTC),
                map_mmhg=40.0,
                hr_bpm=130.0,
                spo2_pct=92.5,
                etco2_mmhg=38.0,
                source="static_test_vitals",
            )
        ]
        source = _StaticVitalsSource(critical)
        predictor = AnesthesiologyPredictor(source, enable_tiva=False, enable_pid=False)
        result = predictor.predict_anesthesia_risk({"vitals_readings": critical})
        assert result.intervention_needed is True
        assert result.vitals_source == "static_test_vitals"
        assert result.vitals_snapshot_count == 1

    def test_fetch_and_predict_uses_adapter(self) -> None:
        readings = _stable_vitals(3)
        source = _StaticVitalsSource(readings)
        predictor = AnesthesiologyPredictor(source, enable_tiva=False, enable_pid=False)
        result = predictor.fetch_and_predict(window_minutes=2)
        assert source.calls == 1
        assert source.last_window == 2
        assert result.vitals_source == "static_test_vitals"
        assert result.vitals_snapshot_count == 3

    def test_fetch_and_predict_requires_enabled_hemodynamics(self) -> None:
        predictor = AnesthesiologyPredictor(enable_hemodynamics=False)
        with pytest.raises(ConfigurationError, match="enable_hemodynamics"):
            predictor.fetch_and_predict()

    def test_vitals_readings_must_be_vitalsreading_instances(self) -> None:
        source = _StaticVitalsSource(_stable_vitals())
        predictor = AnesthesiologyPredictor(source, enable_tiva=False, enable_pid=False)
        with pytest.raises(TypeError, match="VitalsReading"):
            predictor.predict_anesthesia_risk({"vitals_readings": [{"map": 80}]})

    def test_legacy_vitals_payload_still_works(self) -> None:
        source = _StaticVitalsSource(_stable_vitals())
        predictor = AnesthesiologyPredictor(source, enable_tiva=False, enable_pid=False)
        result = predictor.predict_anesthesia_risk(
            {
                "vitals": {
                    "mean_arterial_pressure_mmhg": 80.0,
                    "heart_rate_bpm": 72.0,
                    "oxygen_saturation_pct": 98.0,
                    "end_tidal_co2_mmhg": 38.0,
                }
            }
        )
        assert result.intervention_needed is False
        assert result.vitals_source == "preloaded"

    def test_infusion_path_runs(self) -> None:
        predictor = AnesthesiologyPredictor(
            enable_tiva=False,
            enable_hemodynamics=False,
            enable_pid=True,
        )
        result = predictor.predict_anesthesia_risk(
            {
                "infusion": {
                    "current_bis": 70.0,
                    "current_propofol_rate": 100.0,
                    "current_remifentanil_rate": 0.2,
                }
            }
        )
        assert result.risk_detected is True
        assert result.bis_score == pytest.approx(70.0)

    def test_tiva_path_runs(self) -> None:
        predictor = AnesthesiologyPredictor(enable_pid=False, enable_hemodynamics=False)
        sequence = np.zeros((60, 8), dtype=np.float64)
        result = predictor.predict_anesthesia_risk({"anesthesia_sequence": sequence})
        assert 0.0 <= result.depth_of_anesthesia <= 1.0
        assert 0.0 <= result.risk_score <= 1.0

    def test_tiva_sequence_shape_validated(self) -> None:
        predictor = AnesthesiologyPredictor(enable_pid=False, enable_hemodynamics=False)
        bad = np.zeros((60, 4), dtype=np.float64)
        with pytest.raises(ValueError, match=r"shape \(time, 8\)"):
            predictor.predict_anesthesia_risk({"anesthesia_sequence": bad})

    def test_overall_risk_in_range(self) -> None:
        predictor = AnesthesiologyPredictor(enable_hemodynamics=False)
        result = AnesthesiaPredictionResult(
            risk_detected=False,
            confidence=0.0,
            risk_type="none",
            risk_score=0.0,
            depth_of_anesthesia=0.5,
            hemodynamic_stability=1.0,
            respiratory_adequacy=1.0,
        )
        score = predictor._calculate_overall_risk(result)
        assert 0.0 <= score <= 1.0

    def test_mixed_sources_label(self) -> None:
        start = datetime(2024, 3, 22, 13, 30, 0, tzinfo=UTC)
        mixed = [
            VitalsReading(
                timestamp=start,
                map_mmhg=80.0,
                hr_bpm=70.0,
                spo2_pct=98.0,
                etco2_mmhg=38.0,
                source="fhir_observation",
            ),
            VitalsReading(
                timestamp=start + timedelta(seconds=30),
                map_mmhg=82.0,
                hr_bpm=72.0,
                spo2_pct=98.0,
                etco2_mmhg=38.0,
                source="philips_intellivue",
            ),
        ]
        predictor = AnesthesiologyPredictor(
            _StaticVitalsSource(mixed), enable_tiva=False, enable_pid=False
        )
        result = predictor.predict_anesthesia_risk({"vitals_readings": mixed})
        assert result.vitals_source == "mixed"
