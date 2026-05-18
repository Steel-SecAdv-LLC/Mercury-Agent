"""Tests for the endocrinology detector."""

from __future__ import annotations

import io
import json
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from omni_mercury_engine.medical.endocrinology_detector import (
    CGMAnalyzer,
    DexcomCredentials,
    EndocrinologyDetector,
    EndocrinologyPredictionResult,
    GLP1TherapyMonitor,
    GlycemicState,
    InhaledInsulinMonitor,
    SmartInsulinPenMonitor,
    TidepoolClient,
    TidepoolClientError,
    count_cgm_parameters,
    get_endocrinology_detector,
)


class TestCGMAnalyzer:
    """CGM Bi-LSTM structural tests."""

    def test_parameter_count_in_reference_band(self) -> None:
        """CGM model parameter count is ~155K."""
        count = count_cgm_parameters()
        # Reference: 155K parameters (tolerance for linear-layer drift)
        assert 145_000 <= count <= 165_000, f"Got {count} parameters"

    def test_input_dim_is_one(self) -> None:
        """The first LSTM input dim is 1 (glucose only)."""
        model = CGMAnalyzer()
        assert model.lstm.input_size == 1

    def test_bidirectional(self) -> None:
        """The LSTM is bidirectional."""
        model = CGMAnalyzer()
        assert model.lstm.bidirectional is True

    def test_forward_shapes(self) -> None:
        """forward returns 5-class logits, scalar trend, and attention weights."""
        import torch

        model = CGMAnalyzer()
        model.eval()
        x = torch.zeros((1, 60, 1))
        with torch.no_grad():
            logits, trend, attn = model(x)
        assert logits.shape == (1, 5)
        assert trend.shape == (1, 1)
        assert attn.shape == (1, 60)


class TestSmartInsulinPenMonitor:
    """Smart insulin pen monitor + dose-stacking guard."""

    def test_normal_dose_safe(self) -> None:
        """A normal-sized dose at a sensible spacing is safe."""
        monitor = SmartInsulinPenMonitor()
        result = monitor.monitor_insulin_delivery(
            {
                "dose_units": 5.0,
                "insulin_type": "rapid_acting",
                "time_since_last_dose_hours": 4.0,
                "daily_total_units": 30.0,
            }
        )
        assert result["insulin_delivery_safe"] is True

    def test_large_bolus_flagged(self) -> None:
        """A bolus > max_bolus_units triggers an alert."""
        monitor = SmartInsulinPenMonitor()
        result = monitor.monitor_insulin_delivery(
            {
                "dose_units": 25.0,
                "insulin_type": "rapid_acting",
                "time_since_last_dose_hours": 4.0,
            }
        )
        assert result["insulin_delivery_safe"] is False
        assert any("Large bolus" in a for a in result["alerts"])

    def test_dose_stacking_under_two_hours_blocked(self) -> None:
        """Rapid-acting doses < 2h apart trip the dose-stacking guard."""
        monitor = SmartInsulinPenMonitor()
        result = monitor.monitor_insulin_delivery(
            {
                "dose_units": 5.0,
                "insulin_type": "rapid_acting",
                "time_since_last_dose_hours": 1.0,
            }
        )
        assert result["insulin_delivery_safe"] is False
        assert any("Dose stacking" in a for a in result["alerts"])

    def test_dose_stacking_only_applies_to_rapid_acting(self) -> None:
        """Basal insulin within 2h is not flagged as stacking."""
        monitor = SmartInsulinPenMonitor()
        result = monitor.monitor_insulin_delivery(
            {
                "dose_units": 5.0,
                "insulin_type": "basal",
                "time_since_last_dose_hours": 0.5,
                "daily_total_units": 25.0,
            }
        )
        assert all("Dose stacking" not in a for a in result["alerts"])

    def test_adherence_calculation(self) -> None:
        """Adherence is doses_taken / doses_prescribed, capped at 1.0."""
        monitor = SmartInsulinPenMonitor()
        result = monitor.monitor_insulin_delivery(
            {
                "doses_taken_last_week": 18,
                "doses_prescribed_last_week": 21,
                "dose_units": 5.0,
                "insulin_type": "rapid_acting",
                "time_since_last_dose_hours": 4.0,
            }
        )
        assert result["adherence_score"] == pytest.approx(18 / 21)


class TestGLP1TherapyMonitor:
    """GLP-1 therapy monitor with pancreatitis discontinuation."""

    def test_pancreatitis_forces_discontinuation(self) -> None:
        """Pancreatitis in side_effects sets continue_therapy=False."""
        monitor = GLP1TherapyMonitor()
        result = monitor.monitor_glp1_therapy(
            {
                "medication": "semaglutide",
                "dose_mg": 1.0,
                "duration_weeks": 16,
                "a1c_percent": 6.8,
                "weight_change_percent": -8.0,
                "side_effects": ["pancreatitis"],
            }
        )
        assert result["continue_therapy"] is False
        assert any(
            "Pancreatitis" in r or "discontinue" in r.lower() for r in result["recommendations"]
        )

    def test_pancreatitis_keyword_case_insensitive(self) -> None:
        """The pancreatitis check is case-insensitive."""
        monitor = GLP1TherapyMonitor()
        result = monitor.monitor_glp1_therapy({"side_effects": ["PANCREATITIS"]})
        assert result["continue_therapy"] is False

    def test_normal_therapy_continues(self) -> None:
        """GLP-1 therapy without pancreatitis continues."""
        monitor = GLP1TherapyMonitor()
        result = monitor.monitor_glp1_therapy(
            {
                "a1c_percent": 6.9,
                "weight_change_percent": -10.5,
                "side_effects": [],
            }
        )
        assert result["continue_therapy"] is True

    def test_nausea_recommendations(self) -> None:
        """GI side effects yield management recommendations."""
        monitor = GLP1TherapyMonitor()
        result = monitor.monitor_glp1_therapy({"side_effects": ["nausea"], "duration_weeks": 4})
        joined = " ".join(result["recommendations"]).lower()
        assert "food" in joined or "titration" in joined

    def test_high_a1c_after_12_weeks(self) -> None:
        """A1C > target at 12+ weeks triggers dose-escalation recommendation."""
        monitor = GLP1TherapyMonitor()
        result = monitor.monitor_glp1_therapy(
            {"a1c_percent": 8.5, "duration_weeks": 14, "side_effects": []}
        )
        assert any("dose escalation" in r.lower() for r in result["recommendations"])


class TestInhaledInsulinMonitor:
    """Inhaled insulin monitor + Afrezza FEV1 contraindication."""

    @pytest.mark.parametrize(
        ("fev1", "appropriate"),
        [
            (70.1, True),  # just above the threshold
            (70.0, True),  # exactly at the threshold
            (69.9, False),  # just below -> contraindicated
            (50.0, False),  # clearly contraindicated
        ],
    )
    def test_fev1_threshold(self, fev1: float, appropriate: bool) -> None:
        """The Afrezza contraindication fires at FEV1 < 70%."""
        monitor = InhaledInsulinMonitor()
        result = monitor.monitor_inhaled_insulin({"pulmonary_function_fev1_percent": fev1})
        assert result["inhaled_insulin_appropriate"] is appropriate

    def test_high_dose_recommends_subcutaneous(self) -> None:
        """Doses above max_dose_units recommend SC insulin."""
        monitor = InhaledInsulinMonitor()
        result = monitor.monitor_inhaled_insulin(
            {"dose_units": 20, "pulmonary_function_fev1_percent": 90.0}
        )
        joined = " ".join(result["recommendations"]).lower()
        assert "subcutaneous" in joined

    def test_poor_technique_warning(self) -> None:
        """Technique score < 0.7 produces a retraining recommendation."""
        monitor = InhaledInsulinMonitor()
        result = monitor.monitor_inhaled_insulin(
            {
                "inhalation_technique_score": 0.5,
                "pulmonary_function_fev1_percent": 90.0,
            }
        )
        joined = " ".join(result["recommendations"]).lower()
        assert "retrain" in joined or "technique" in joined


class TestTidepoolClient:
    """Tidepool client behaviour (no real network in tests)."""

    def _fake_response(self, body: bytes, status: int = 200) -> Any:
        class _Resp:
            def __init__(self, data: bytes, code: int) -> None:
                self._buf = io.BytesIO(data)
                self.status = code

            def read(self) -> bytes:
                return self._buf.read()

            def __enter__(self) -> _Resp:
                return self

            def __exit__(self, *args: Any) -> None:
                return None

        return _Resp(body, status)

    def test_info_returns_parsed_json(self) -> None:
        """info() parses Tidepool /info as JSON."""
        body = json.dumps({"version": "1.0.0"}).encode("utf-8")
        client = TidepoolClient()
        with patch(
            "omni_mercury_engine.medical.endocrinology_detector.urlopen",
            return_value=self._fake_response(body),
        ):
            data = client.info()
        assert data == {"version": "1.0.0"}

    def test_bearer_token_attached(self) -> None:
        """A configured token is sent as a bearer header."""
        body = json.dumps({}).encode("utf-8")
        client = TidepoolClient(token="abc")  # noqa: S106 - test fixture
        with patch(
            "omni_mercury_engine.medical.endocrinology_detector.urlopen",
            return_value=self._fake_response(body),
        ) as mocked:
            client.info()
            req = mocked.call_args[0][0]
            assert req.get_header("Authorization") == "Bearer abc"

    def test_oserror_wrapped(self) -> None:
        """Network errors raise TidepoolClientError."""
        client = TidepoolClient()
        with (
            patch(
                "omni_mercury_engine.medical.endocrinology_detector.urlopen",
                side_effect=OSError("offline"),
            ),
            pytest.raises(TidepoolClientError),
        ):
            client.info()


class TestDexcomCredentials:
    """DexcomCredentials environment loading."""

    def test_missing_env_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing env vars raise RuntimeError."""
        monkeypatch.delenv("DEXCOM_CLIENT_ID", raising=False)
        monkeypatch.delenv("DEXCOM_CLIENT_SECRET", raising=False)
        with pytest.raises(RuntimeError):
            DexcomCredentials.from_environment()

    def test_present_env_loaded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Present env vars are loaded."""
        monkeypatch.setenv("DEXCOM_CLIENT_ID", "client-id")
        monkeypatch.setenv("DEXCOM_CLIENT_SECRET", "secret")
        creds = DexcomCredentials.from_environment()
        assert creds.client_id == "client-id"
        assert creds.client_secret == "secret"


class TestEndocrinologyDetector:
    """End-to-end detector behaviour."""

    def test_empty_input_safe_default(self) -> None:
        """Empty input yields no detected anomaly."""
        detector = EndocrinologyDetector()
        result = detector.detect_endocrine_anomaly({})
        assert isinstance(result, EndocrinologyPredictionResult)
        assert result.anomaly_detected is False
        assert result.glycemic_state == GlycemicState.NORMAL.value

    def test_cgm_sequence_runs_inference(self) -> None:
        """A 60-point CGM sequence runs the Bi-LSTM and reports time-in-range."""
        detector = EndocrinologyDetector(
            enable_smart_pen=False, enable_glp1=False, enable_inhaled_insulin=False
        )
        sequence = np.full(60, 130.0, dtype=np.float64)
        result = detector.detect_endocrine_anomaly({"cgm_sequence": sequence})
        assert result.time_in_range_percent is not None
        assert result.time_in_range_percent == pytest.approx(100.0)

    def test_time_in_range_below_70(self) -> None:
        """Values below 70 mg/dL are not in the target range."""
        detector = EndocrinologyDetector(
            enable_smart_pen=False, enable_glp1=False, enable_inhaled_insulin=False
        )
        sequence = np.array([60.0] * 30 + [120.0] * 30, dtype=np.float64)
        result = detector.detect_endocrine_anomaly({"cgm_sequence": sequence})
        assert result.time_in_range_percent == pytest.approx(50.0)

    def test_insulin_delivery_path(self) -> None:
        """Smart-pen unsafe dosing triggers intervention_needed."""
        detector = EndocrinologyDetector(
            enable_cgm=False, enable_glp1=False, enable_inhaled_insulin=False
        )
        result = detector.detect_endocrine_anomaly(
            {
                "insulin_delivery": {
                    "dose_units": 25.0,
                    "insulin_type": "rapid_acting",
                    "time_since_last_dose_hours": 1.0,
                }
            }
        )
        assert result.intervention_needed is True
        assert result.anomaly_detected is True

    def test_glp1_pancreatitis_triggers_intervention(self) -> None:
        """Pancreatitis in GLP-1 therapy triggers intervention."""
        detector = EndocrinologyDetector(
            enable_cgm=False, enable_smart_pen=False, enable_inhaled_insulin=False
        )
        result = detector.detect_endocrine_anomaly(
            {"glp1_therapy": {"side_effects": ["pancreatitis"]}}
        )
        assert result.intervention_needed is True
        assert result.anomaly_detected is True

    def test_afrezza_contraindication_triggers_intervention(self) -> None:
        """FEV1 below 70% triggers intervention."""
        detector = EndocrinologyDetector(
            enable_cgm=False, enable_smart_pen=False, enable_glp1=False
        )
        result = detector.detect_endocrine_anomaly(
            {
                "inhaled_insulin": {
                    "pulmonary_function_fev1_percent": 65.0,
                }
            }
        )
        assert result.intervention_needed is True

    def test_overall_risk_capped(self) -> None:
        """Risk score is capped at 1.0."""
        detector = EndocrinologyDetector()
        result = EndocrinologyPredictionResult(
            anomaly_detected=True,
            confidence=0.9,
            glycemic_state=GlycemicState.HYPERGLYCEMIA.value,
            risk_score=0.0,
            hypoglycemia_risk=1.0,
            hyperglycemia_risk=1.0,
            dka_risk=1.0,
        )
        assert detector._calculate_overall_risk(result) == pytest.approx(1.0)


class TestFactory:
    """Factory function."""

    def test_factory_returns_detector(self) -> None:
        """get_endocrinology_detector returns a detector."""
        detector = get_endocrinology_detector(enable_cgm=False)
        assert isinstance(detector, EndocrinologyDetector)
        assert detector.cgm_analyzer is None
