"""Tests for the endocrinology detector and its rule monitors."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from omni_mercury_engine.medical.data_sources import (
    CGMDataSource,
    CGMReading,
    ConfigurationError,
)
from omni_mercury_engine.medical.endocrinology_detector import (
    CGMAnalyzer,
    EndocrinologyDetector,
    EndocrinologyPredictionResult,
    GLP1TherapyMonitor,
    GlycemicState,
    InhaledInsulinMonitor,
    SmartInsulinPenMonitor,
    count_cgm_parameters,
    get_endocrinology_detector,
)


class _StaticCGMSource(CGMDataSource):
    """In-process CGM source backed by a static reading list.

    Used purely to drive end-to-end tests of the detector's integration with
    a configured adapter; no synthetic data ever enters production paths.
    """

    name = "static_test_cgm"

    def __init__(self, readings: list[CGMReading]) -> None:
        self._readings = readings
        self.calls = 0
        self.last_window: int | None = None

    def fetch_recent_readings(self, window_minutes: int = 180) -> list[CGMReading]:
        self.calls += 1
        self.last_window = window_minutes
        return list(self._readings)


def _readings_at_glucose(value: float, n: int = 60) -> list[CGMReading]:
    start = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
    return [
        CGMReading(
            timestamp=start + timedelta(minutes=5 * i),
            value_mg_dl=value,
            source="static_test_cgm",
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------------- #
# CGMAnalyzer (neural network) structural tests
# --------------------------------------------------------------------------- #


class TestCGMAnalyzer:
    """CGM Bi-LSTM structural tests."""

    def test_parameter_count_in_reference_band(self) -> None:
        count = count_cgm_parameters()
        # Reference: 155K parameters (tolerance for linear-layer drift).
        assert 145_000 <= count <= 165_000, f"Got {count} parameters"

    def test_input_dim_is_one(self) -> None:
        model = CGMAnalyzer()
        assert model.lstm.input_size == 1

    def test_bidirectional(self) -> None:
        model = CGMAnalyzer()
        assert model.lstm.bidirectional is True

    def test_forward_shapes(self) -> None:
        import torch

        model = CGMAnalyzer()
        model.eval()
        x = torch.zeros((1, 60, 1))
        with torch.no_grad():
            logits, trend, attn = model(x)
        assert logits.shape == (1, 5)
        assert trend.shape == (1,)
        assert attn.shape == (1, 60)


# --------------------------------------------------------------------------- #
# Rule-engine monitors
# --------------------------------------------------------------------------- #


class TestSmartInsulinPenMonitor:
    """Smart insulin pen monitor + dose-stacking guard."""

    def test_safe_when_doses_spaced(self) -> None:
        monitor = SmartInsulinPenMonitor()
        result = monitor.monitor_insulin_delivery(
            {
                "recent_doses": [
                    {"time_hours": 0.0, "units": 5.0},
                    {"time_hours": 4.0, "units": 5.0},
                ],
                "adherence_rate": 1.0,
                "patient_glucose": 140.0,
            }
        )
        assert result["insulin_delivery_safe"] is True
        assert result["alerts"] == []

    def test_dose_stacking_blocked_when_glucose_missing(self) -> None:
        """Without verified glucose, stacking < 2 h triggers an alert."""
        monitor = SmartInsulinPenMonitor()
        result = monitor.monitor_insulin_delivery(
            {
                "recent_doses": [
                    {"time_hours": 0.0, "units": 5.0},
                    {"time_hours": 1.0, "units": 5.0},
                ],
                "adherence_rate": 1.0,
            }
        )
        assert result["insulin_delivery_safe"] is False
        joined = " ".join(result["alerts"])
        assert "stacking" in joined.lower()

    def test_dose_stacking_blocked_when_glucose_high(self) -> None:
        """Glucose > 250 mg/dL signals hyperglycaemia; stacking is alerted."""
        monitor = SmartInsulinPenMonitor()
        result = monitor.monitor_insulin_delivery(
            {
                "recent_doses": [
                    {"time_hours": 0.0, "units": 5.0},
                    {"time_hours": 1.0, "units": 5.0},
                ],
                "adherence_rate": 1.0,
                "patient_glucose": 320.0,
            }
        )
        assert result["insulin_delivery_safe"] is False

    def test_dose_stacking_allowed_when_glucose_verified_safe(self) -> None:
        """When measured glucose is ≤ 250 mg/dL the stacking guard does not fire."""
        monitor = SmartInsulinPenMonitor()
        result = monitor.monitor_insulin_delivery(
            {
                "recent_doses": [
                    {"time_hours": 0.0, "units": 5.0},
                    {"time_hours": 1.0, "units": 5.0},
                ],
                "adherence_rate": 1.0,
                "patient_glucose": 180.0,
            }
        )
        assert result["insulin_delivery_safe"] is True

    def test_low_adherence_alert(self) -> None:
        monitor = SmartInsulinPenMonitor()
        result = monitor.monitor_insulin_delivery({"recent_doses": [], "adherence_rate": 0.5})
        assert result["insulin_delivery_safe"] is False
        assert any("adherence" in a.lower() for a in result["alerts"])
        assert result["adherence_rate"] == pytest.approx(0.5)

    def test_adherence_clamped(self) -> None:
        monitor = SmartInsulinPenMonitor()
        result = monitor.monitor_insulin_delivery({"recent_doses": [], "adherence_rate": 1.5})
        assert result["adherence_rate"] == 1.0


class TestGLP1TherapyMonitor:
    """GLP-1 therapy monitor with pancreatitis discontinuation."""

    def test_pancreatitis_forces_discontinuation(self) -> None:
        monitor = GLP1TherapyMonitor()
        result = monitor.monitor_glp1_therapy(
            {
                "medication": "semaglutide",
                "a1c_change_percent": -0.6,
                "weight_loss_kg": 4.0,
                "side_effects": ["pancreatitis"],
            }
        )
        assert result["continue_therapy"] is False
        joined = " ".join(result["recommendations"]).lower()
        assert "discontinue" in joined or "pancreatitis" in joined

    def test_pancreatitis_keyword_case_insensitive(self) -> None:
        monitor = GLP1TherapyMonitor()
        result = monitor.monitor_glp1_therapy({"side_effects": ["PANCREATITIS"]})
        assert result["continue_therapy"] is False

    def test_pancreatitis_substring_match(self) -> None:
        monitor = GLP1TherapyMonitor()
        result = monitor.monitor_glp1_therapy({"side_effects": ["acute pancreatitis history"]})
        assert result["continue_therapy"] is False

    def test_normal_therapy_continues(self) -> None:
        monitor = GLP1TherapyMonitor()
        result = monitor.monitor_glp1_therapy(
            {
                "a1c_change_percent": -1.0,
                "weight_loss_kg": 5.0,
                "side_effects": [],
            }
        )
        assert result["continue_therapy"] is True
        assert result["therapeutic_success"] is True

    def test_subtherapeutic_response_recommended(self) -> None:
        monitor = GLP1TherapyMonitor()
        result = monitor.monitor_glp1_therapy(
            {
                "a1c_change_percent": -0.1,
                "weight_loss_kg": 0.5,
                "side_effects": [],
            }
        )
        assert result["continue_therapy"] is True
        assert result["therapeutic_success"] is False
        joined = " ".join(result["recommendations"]).lower()
        assert "dose" in joined or "adherence" in joined


class TestInhaledInsulinMonitor:
    """Inhaled insulin monitor + Afrezza FEV1 contraindication."""

    @pytest.mark.parametrize(
        ("fev1", "appropriate"),
        [
            (70.1, True),
            (70.0, True),
            (69.9, False),
            (50.0, False),
        ],
    )
    def test_fev1_threshold(self, fev1: float, appropriate: bool) -> None:
        monitor = InhaledInsulinMonitor()
        result = monitor.monitor_inhaled_insulin({"fev1_percent": fev1})
        assert result["inhaled_insulin_appropriate"] is appropriate

    def test_contraindication_message(self) -> None:
        monitor = InhaledInsulinMonitor()
        result = monitor.monitor_inhaled_insulin({"fev1_percent": 50.0})
        assert any("CONTRAINDICATION" in a for a in result["alerts"])
        joined = " ".join(result["recommendations"]).lower()
        assert "subcutaneous" in joined

    def test_high_post_meal_glucose_recommendation(self) -> None:
        monitor = InhaledInsulinMonitor()
        result = monitor.monitor_inhaled_insulin({"fev1_percent": 90.0, "post_meal_glucose": 220.0})
        joined = " ".join(result["recommendations"]).lower()
        assert "post-meal" in joined or "dose adjustment" in joined


# --------------------------------------------------------------------------- #
# EndocrinologyDetector integration
# --------------------------------------------------------------------------- #


class TestDetectorConfiguration:
    """Configuration semantics + ConfigurationError contract."""

    def test_missing_data_source_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="CGMDataSource"):
            EndocrinologyDetector()

    def test_factory_raises_without_data_source(self) -> None:
        with pytest.raises(ConfigurationError, match="CGMDataSource"):
            get_endocrinology_detector()

    def test_cgm_disabled_allows_no_source(self) -> None:
        detector = EndocrinologyDetector(enable_cgm=False)
        assert detector.data_source is None
        assert detector.cgm_analyzer is None

    def test_with_source_constructs(self) -> None:
        source = _StaticCGMSource(_readings_at_glucose(120.0))
        detector = EndocrinologyDetector(source)
        assert detector.data_source is source
        assert detector.cgm_analyzer is not None

    def test_non_subclass_data_source_rejected(self) -> None:
        with pytest.raises(TypeError, match="CGMDataSource"):
            EndocrinologyDetector(object())  # type: ignore[arg-type]

    def test_factory_returns_detector_when_disabled(self) -> None:
        detector = get_endocrinology_detector(enable_cgm=False)
        assert isinstance(detector, EndocrinologyDetector)
        assert detector.cgm_analyzer is None


class TestDetectorPipeline:
    """Detector pipeline behaviour with real CGMReading inputs."""

    def test_empty_input_safe_default(self) -> None:
        detector = EndocrinologyDetector(enable_cgm=False)
        result = detector.detect_endocrine_anomaly({})
        assert isinstance(result, EndocrinologyPredictionResult)
        assert result.anomaly_detected is False
        assert result.glycemic_state == GlycemicState.NORMAL.value

    def test_cgm_readings_run_inference_and_compute_tir(self) -> None:
        source = _StaticCGMSource(_readings_at_glucose(130.0))
        detector = EndocrinologyDetector(
            source,
            enable_smart_pen=False,
            enable_glp1=False,
            enable_inhaled_insulin=False,
        )
        result = detector.detect_endocrine_anomaly({"cgm_readings": source.fetch_recent_readings()})
        assert result.time_in_range_percent == pytest.approx(100.0)
        assert result.cgm_reading_count == 60
        assert result.cgm_source == "static_test_cgm"

    def test_fetch_and_detect_uses_adapter(self) -> None:
        source = _StaticCGMSource(_readings_at_glucose(140.0))
        detector = EndocrinologyDetector(
            source,
            enable_smart_pen=False,
            enable_glp1=False,
            enable_inhaled_insulin=False,
        )
        result = detector.fetch_and_detect(window_minutes=90)
        assert source.calls == 1
        assert source.last_window == 90
        assert result.cgm_reading_count == 60
        assert result.cgm_source == "static_test_cgm"

    def test_fetch_and_detect_requires_enabled_cgm(self) -> None:
        detector = EndocrinologyDetector(enable_cgm=False)
        with pytest.raises(ConfigurationError, match="enable_cgm"):
            detector.fetch_and_detect()

    def test_cgm_readings_must_be_cgmreading_instances(self) -> None:
        source = _StaticCGMSource(_readings_at_glucose(130.0))
        detector = EndocrinologyDetector(source)
        with pytest.raises(TypeError, match="CGMReading"):
            detector.detect_endocrine_anomaly({"cgm_readings": [120.0, 130.0]})

    def test_time_in_range_below_70(self) -> None:
        source = _StaticCGMSource([])
        detector = EndocrinologyDetector(
            source,
            enable_smart_pen=False,
            enable_glp1=False,
            enable_inhaled_insulin=False,
        )
        sequence = np.array([60.0] * 30 + [120.0] * 30, dtype=np.float64)
        result = detector.detect_endocrine_anomaly({"cgm_sequence": sequence})
        assert result.time_in_range_percent == pytest.approx(50.0)

    def test_insulin_delivery_path(self) -> None:
        detector = EndocrinologyDetector(
            enable_cgm=False,
            enable_glp1=False,
            enable_inhaled_insulin=False,
        )
        # Doses < 2 h apart with unverified glucose ⇒ stacking alert.
        result = detector.detect_endocrine_anomaly(
            {
                "insulin_delivery": {
                    "recent_doses": [
                        {"time_hours": 0.0, "units": 5.0},
                        {"time_hours": 1.0, "units": 5.0},
                    ],
                    "adherence_rate": 1.0,
                }
            }
        )
        assert result.intervention_needed is True
        assert result.anomaly_detected is True

    def test_glp1_pancreatitis_triggers_intervention(self) -> None:
        detector = EndocrinologyDetector(
            enable_cgm=False,
            enable_smart_pen=False,
            enable_inhaled_insulin=False,
        )
        result = detector.detect_endocrine_anomaly(
            {"glp1_therapy": {"side_effects": ["pancreatitis"]}}
        )
        assert result.intervention_needed is True
        assert result.anomaly_detected is True

    def test_afrezza_contraindication_triggers_intervention(self) -> None:
        detector = EndocrinologyDetector(
            enable_cgm=False,
            enable_smart_pen=False,
            enable_glp1=False,
        )
        result = detector.detect_endocrine_anomaly({"inhaled_insulin": {"fev1_percent": 65.0}})
        assert result.intervention_needed is True

    def test_overall_risk_capped(self) -> None:
        detector = EndocrinologyDetector(enable_cgm=False)
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

    def test_mixed_sources_label(self) -> None:
        start = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
        mixed = [
            CGMReading(timestamp=start, value_mg_dl=120.0, source="dexcom_v3"),
            CGMReading(
                timestamp=start + timedelta(minutes=5),
                value_mg_dl=121.0,
                source="abbott_libre",
            ),
        ]
        detector = EndocrinologyDetector(
            _StaticCGMSource(mixed),
            enable_smart_pen=False,
            enable_glp1=False,
            enable_inhaled_insulin=False,
        )
        result = detector.detect_endocrine_anomaly({"cgm_readings": mixed})
        assert result.cgm_source == "mixed"
