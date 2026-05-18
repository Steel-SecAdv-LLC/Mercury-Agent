"""Tests for the anesthesiology predictor."""

from __future__ import annotations

import io
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from omni_mercury_engine.medical.anesthesiology_predictor import (
    AnesthesiaPredictionResult,
    AnesthesiologyPredictor,
    HemodynamicMonitor,
    SmartInfusionController,
    TIVAMonitoringSystem,
    VitalDBClient,
    VitalDBClientError,
    count_tiva_parameters,
    get_anesthesiology_predictor,
)


class TestTIVAMonitoringSystem:
    """TIVA Bi-LSTM structural tests."""

    def test_parameter_count_matches_reference(self) -> None:
        """The TIVA model must have ~164K trainable parameters."""
        count = count_tiva_parameters()
        # Reference: 164K parameters (164,000 +/- small linear-layer drift)
        assert 160_000 <= count <= 170_000, f"Got {count} parameters"

    def test_forward_pass_shapes(self) -> None:
        """forward returns depth, risks, and attention with correct shapes."""
        model = TIVAMonitoringSystem()
        import torch

        x = torch.zeros((1, 60, 8))
        model.eval()
        with torch.no_grad():
            depth, risks, attn = model(x)
        assert depth.shape == (1, 1)
        assert risks.shape == (1, 7)
        assert attn.shape == (1, 60)

    def test_input_dim_is_eight(self) -> None:
        """The first LSTM input dim is 8 (propofol, remi, MAP, HR, SpO2, ...)."""
        model = TIVAMonitoringSystem()
        assert model.lstm.input_size == 8

    def test_bidirectional(self) -> None:
        """The LSTM is bidirectional."""
        model = TIVAMonitoringSystem()
        assert model.lstm.bidirectional is True


class TestSmartInfusionController:
    """PID infusion controller behaviour."""

    def test_pid_gains_match_reference(self) -> None:
        """PID gains match the verified Omni-AXA values."""
        controller = SmartInfusionController()
        assert controller.kp == 0.5
        assert controller.ki == 0.1
        assert controller.kd == 0.2
        assert controller.target_bis == 50.0
        assert controller.bis_range == (40.0, 60.0)

    def test_in_range_no_anomaly(self) -> None:
        """In-range BIS does not flag an anomaly."""
        controller = SmartInfusionController()
        result = controller.compute_infusion_adjustment(50.0, 100.0, 0.2)
        assert result["anomaly_detected"] is False

    def test_low_bis_triggers_deep_anesthesia_alert(self) -> None:
        """BIS below 40 produces a deep-anesthesia alert."""
        controller = SmartInfusionController()
        result = controller.compute_infusion_adjustment(30.0, 150.0, 0.3)
        assert result["anomaly_detected"] is True
        joined = " ".join(result["recommendations"])
        assert "Deep anesthesia" in joined or "BIS < 40" in joined

    def test_high_bis_triggers_light_anesthesia_alert(self) -> None:
        """BIS above 60 produces a light-anesthesia (awareness risk) alert."""
        controller = SmartInfusionController()
        result = controller.compute_infusion_adjustment(70.0, 80.0, 0.1)
        assert result["anomaly_detected"] is True
        joined = " ".join(result["recommendations"])
        assert "awareness" in joined.lower() or "Light anesthesia" in joined

    def test_propofol_clamped_to_limits(self) -> None:
        """Propofol output stays within the configured limits."""
        controller = SmartInfusionController()
        # Push BIS far below target -> negative error -> dose decreases
        result = controller.compute_infusion_adjustment(10.0, 0.0, 0.0)
        assert 0.0 <= result["propofol_rate_mcg_kg_min"] <= 200.0

    def test_reset_clears_integrator(self) -> None:
        """reset() clears accumulated integral error."""
        controller = SmartInfusionController()
        controller.compute_infusion_adjustment(70.0, 100.0, 0.2)
        controller.reset()
        assert controller.integral_error == 0.0
        assert controller.previous_error == 0.0

    def test_dt_must_be_positive(self) -> None:
        """Non-positive dt raises ValueError."""
        controller = SmartInfusionController()
        with pytest.raises(ValueError):
            controller.compute_infusion_adjustment(50.0, 100.0, 0.2, dt=0.0)
        with pytest.raises(ValueError):
            controller.compute_infusion_adjustment(50.0, 100.0, 0.2, dt=-1.0)

    def test_pid_responds_to_persistent_error(self) -> None:
        """Persistent error grows the integral term and changes propofol."""
        controller = SmartInfusionController()
        first = controller.compute_infusion_adjustment(70.0, 100.0, 0.2)
        second = controller.compute_infusion_adjustment(70.0, 100.0, 0.2)
        # second adjustment magnitude should differ from first due to integral
        assert first["propofol_adjustment"] != second["propofol_adjustment"]


class TestHemodynamicMonitor:
    """Hemodynamic monitor vital ranges."""

    def test_normal_vitals_high_stability(self) -> None:
        """Normal vitals produce hemodynamic stability near 1.0."""
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
        """MAP < 65 triggers hypotension."""
        monitor = HemodynamicMonitor()
        result = monitor.assess_hemodynamics({"mean_arterial_pressure_mmhg": 40.0})
        assert result["risk_scores"]["hypotension"] > 0.0
        assert any("Hypotension" in a for a in result["alerts"])

    def test_hypertension_detected(self) -> None:
        """MAP > 110 triggers hypertension."""
        monitor = HemodynamicMonitor()
        result = monitor.assess_hemodynamics({"mean_arterial_pressure_mmhg": 150.0})
        assert result["risk_scores"]["hypertension"] > 0.0

    def test_hypoxemia_detected(self) -> None:
        """SpO2 < 92% triggers hypoxemia and a critical recommendation."""
        monitor = HemodynamicMonitor()
        result = monitor.assess_hemodynamics({"oxygen_saturation_pct": 85.0})
        assert result["risk_scores"]["hypoxemia"] > 0.0
        assert result["intervention_needed"] is True

    def test_bradycardia_recommendations(self) -> None:
        """Bradycardia produces anticholinergic recommendations."""
        monitor = HemodynamicMonitor()
        result = monitor.assess_hemodynamics({"heart_rate_bpm": 35.0})
        assert result["risk_scores"]["bradycardia"] > 0.0
        joined = " ".join(result["recommendations"])
        assert "atropine" in joined.lower() or "Bradycardia" in joined

    def test_tachycardia(self) -> None:
        """HR > 100 bpm flags tachycardia."""
        monitor = HemodynamicMonitor()
        result = monitor.assess_hemodynamics({"heart_rate_bpm": 130.0})
        assert result["risk_scores"]["tachycardia"] > 0.0

    def test_hypercarbia_and_hypocarbia(self) -> None:
        """EtCO2 outside [30, 45] triggers the appropriate risk."""
        monitor = HemodynamicMonitor()
        high = monitor.assess_hemodynamics({"end_tidal_co2_mmhg": 60.0})
        low = monitor.assess_hemodynamics({"end_tidal_co2_mmhg": 20.0})
        assert high["risk_scores"]["hypercarbia"] > 0.0
        assert low["risk_scores"]["hypocarbia"] > 0.0


class TestVitalDBClient:
    """VitalDB client behaviour (no real network in tests)."""

    def _fake_response(self, body: bytes) -> Any:
        class _Resp:
            status = 200

            def __init__(self, data: bytes) -> None:
                self._buf = io.BytesIO(data)

            def read(self) -> bytes:
                return self._buf.read()

            def __enter__(self) -> _Resp:
                return self

            def __exit__(self, *args: Any) -> None:
                return None

        return _Resp(body)

    def test_list_cases_parses_csv(self) -> None:
        """list_cases parses VitalDB CSV into dicts."""
        body = b"caseid,age,sex\n1,42,M\n2,55,F\n"
        client = VitalDBClient()
        with patch(
            "omni_mercury_engine.medical.anesthesiology_predictor.urlopen",
            return_value=self._fake_response(body),
        ):
            cases = client.list_cases()
        assert cases == [
            {"caseid": "1", "age": "42", "sex": "M"},
            {"caseid": "2", "age": "55", "sex": "F"},
        ]

    def test_list_tracks_query_param(self) -> None:
        """list_tracks issues the right query string."""
        body = b"tid,tname\nabc,Solar8000/HR\n"
        client = VitalDBClient()
        with patch(
            "omni_mercury_engine.medical.anesthesiology_predictor.urlopen",
            return_value=self._fake_response(body),
        ) as mocked:
            client.list_tracks(case_id=1)
            req = mocked.call_args[0][0]
            assert "caseid=1" in req.full_url

    def test_fetch_case_track_parses_pairs(self) -> None:
        """fetch_case_track returns an (N, 2) float array."""
        body = b"Time,HR\n0.0,72.5\n1.0,73.0\n2.0,71.0\n"
        client = VitalDBClient()
        with patch(
            "omni_mercury_engine.medical.anesthesiology_predictor.urlopen",
            return_value=self._fake_response(body),
        ):
            arr = client.fetch_case_track("abc")
        assert arr.shape == (3, 2)
        assert arr[0, 1] == pytest.approx(72.5)

    def test_request_error_wrapped(self) -> None:
        """OSError is wrapped as VitalDBClientError."""
        client = VitalDBClient()
        with (
            patch(
                "omni_mercury_engine.medical.anesthesiology_predictor.urlopen",
                side_effect=OSError("offline"),
            ),
            pytest.raises(VitalDBClientError),
        ):
            client.list_cases()


class TestAnesthesiologyPredictor:
    """End-to-end predictor behaviour."""

    def test_no_data_returns_safe_defaults(self) -> None:
        """Empty input yields a non-risk result."""
        predictor = AnesthesiologyPredictor()
        result = predictor.predict_anesthesia_risk({})
        assert isinstance(result, AnesthesiaPredictionResult)
        assert result.risk_detected is False

    def test_hemodynamic_path_runs(self) -> None:
        """Critical vitals trigger intervention_needed."""
        predictor = AnesthesiologyPredictor(enable_tiva=False, enable_smart_infusion=False)
        result = predictor.predict_anesthesia_risk(
            {
                "current_vitals": {
                    "mean_arterial_pressure_mmhg": 40.0,
                    "heart_rate_bpm": 130.0,
                }
            }
        )
        assert result.intervention_needed is True

    def test_infusion_path_runs(self) -> None:
        """BIS out of range triggers infusion anomaly."""
        predictor = AnesthesiologyPredictor(
            enable_tiva=False, enable_hemodynamic=False, enable_smart_infusion=True
        )
        result = predictor.predict_anesthesia_risk(
            {
                "bis_score": 70.0,
                "infusion_rates": {
                    "propofol_mcg_kg_min": 100.0,
                    "remifentanil_mcg_kg_min": 0.2,
                },
            }
        )
        assert result.risk_detected is True
        assert result.bis_score == pytest.approx(70.0)

    def test_tiva_path_runs(self) -> None:
        """TIVA Bi-LSTM inference returns a valid depth and risk score."""
        predictor = AnesthesiologyPredictor(enable_smart_infusion=False, enable_hemodynamic=False)
        sequence = np.zeros((60, 8), dtype=np.float64)
        result = predictor.predict_anesthesia_risk({"anesthesia_sequence": sequence})
        assert 0.0 <= result.depth_of_anesthesia <= 1.0
        assert 0.0 <= result.risk_score <= 1.0

    def test_overall_risk_in_range(self) -> None:
        """The overall risk score stays in [0, 1]."""
        predictor = AnesthesiologyPredictor()
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


class TestFactory:
    """Factory function behaviour."""

    def test_factory_returns_predictor(self) -> None:
        """get_anesthesiology_predictor returns a predictor."""
        predictor = get_anesthesiology_predictor(enable_tiva=False)
        assert isinstance(predictor, AnesthesiologyPredictor)
        assert predictor.tiva_monitor is None
