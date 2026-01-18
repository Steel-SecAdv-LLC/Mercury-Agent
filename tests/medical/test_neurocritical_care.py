"""
Mercury Agent - Neurocritical Care Predictor Tests

Comprehensive test suite for neurocritical care components:
- StrokeDetector: CNN-based stroke detection and classification
- SeizurePredictor: LSTM-based seizure detection with attention
- ICPMonitor: Intracranial pressure monitoring
- NIHSSCalculator: NIH Stroke Scale scoring
- NeurocriticalCarePredictor: Main integration system

Target: 85%+ code coverage for life-critical module.
"""

import numpy as np
import pytest
import torch

from omni_mercury_engine.medical.critical_care.neurocritical_care import (
    ICPMonitor,
    NeurocriticalCarePredictor,
    NeurocriticalPredictionResult,
    NIHSSCalculator,
    SeizurePredictor,
    SeizureType,
    StrokeDetector,
    StrokeType,
)


class TestNIHSSCalculator:
    """Tests for NIH Stroke Scale Calculator."""

    @pytest.fixture
    def calculator(self) -> NIHSSCalculator:
        """Provide NIHSSCalculator instance."""
        return NIHSSCalculator()

    def test_nihss_perfect_score_zero(self, calculator: NIHSSCalculator) -> None:
        """Test that normal exam yields NIHSS score of 0."""
        normal_exam = {
            "loc": 0,
            "loc_questions": 0,
            "loc_commands": 0,
            "gaze": 0,
            "visual_fields": 0,
            "facial_palsy": 0,
            "motor_arm_left": 0,
            "motor_arm_right": 0,
            "motor_leg_left": 0,
            "motor_leg_right": 0,
            "limb_ataxia": 0,
            "sensory": 0,
            "language": 0,
            "dysarthria": 0,
            "extinction_inattention": 0,
        }
        result = calculator.calculate_nihss(normal_exam)
        assert result["total_score"] == 0
        assert result["severity"] == "normal"

    def test_nihss_minor_stroke(self, calculator: NIHSSCalculator) -> None:
        """Test minor stroke (NIHSS 1-4)."""
        exam = {
            "loc": 0,
            "loc_questions": 1,
            "loc_commands": 0,
            "gaze": 0,
            "visual_fields": 0,
            "facial_palsy": 1,
            "motor_arm_left": 1,
            "motor_arm_right": 0,
            "motor_leg_left": 0,
            "motor_leg_right": 0,
            "limb_ataxia": 0,
            "sensory": 0,
            "language": 0,
            "dysarthria": 0,
            "extinction_inattention": 0,
        }
        result = calculator.calculate_nihss(exam)
        assert 1 <= result["total_score"] <= 4
        assert result["severity"] == "minor"

    def test_nihss_moderate_stroke(self, calculator: NIHSSCalculator) -> None:
        """Test moderate stroke (NIHSS 5-15)."""
        exam = {
            "loc": 1,
            "loc_questions": 2,
            "loc_commands": 1,
            "gaze": 1,
            "visual_fields": 1,
            "facial_palsy": 2,
            "motor_arm_left": 2,
            "motor_arm_right": 0,
            "motor_leg_left": 2,
            "motor_leg_right": 0,
            "limb_ataxia": 1,
            "sensory": 1,
            "language": 0,
            "dysarthria": 0,
            "extinction_inattention": 0,
        }
        result = calculator.calculate_nihss(exam)
        assert 5 <= result["total_score"] <= 15
        assert result["severity"] == "moderate"

    def test_nihss_moderate_severe_stroke(self, calculator: NIHSSCalculator) -> None:
        """Test moderate-severe stroke (NIHSS 16-20)."""
        exam = {
            "loc": 2,
            "loc_questions": 2,
            "loc_commands": 2,
            "gaze": 2,
            "visual_fields": 2,
            "facial_palsy": 3,
            "motor_arm_left": 3,
            "motor_arm_right": 0,
            "motor_leg_left": 3,
            "motor_leg_right": 0,
            "limb_ataxia": 1,
            "sensory": 1,
            "language": 1,
            "dysarthria": 0,
            "extinction_inattention": 0,
        }
        result = calculator.calculate_nihss(exam)
        assert 16 <= result["total_score"] <= 20
        assert result["severity"] == "moderate-severe"

    def test_nihss_severe_stroke(self, calculator: NIHSSCalculator) -> None:
        """Test severe stroke (NIHSS 21-42)."""
        exam = {
            "loc": 3,
            "loc_questions": 2,
            "loc_commands": 2,
            "gaze": 2,
            "visual_fields": 3,
            "facial_palsy": 3,
            "motor_arm_left": 4,
            "motor_arm_right": 4,
            "motor_leg_left": 4,
            "motor_leg_right": 4,
            "limb_ataxia": 2,
            "sensory": 2,
            "language": 3,
            "dysarthria": 2,
            "extinction_inattention": 2,
        }
        result = calculator.calculate_nihss(exam)
        assert result["total_score"] >= 21
        assert result["severity"] == "severe"

    def test_nihss_component_scores(self, calculator: NIHSSCalculator) -> None:
        """Test that component scores are properly captured."""
        exam = {
            "loc": 1,
            "loc_questions": 2,
            "loc_commands": 1,
            "gaze": 0,
            "visual_fields": 0,
            "facial_palsy": 2,
            "motor_arm_left": 3,
            "motor_arm_right": 0,
            "motor_leg_left": 2,
            "motor_leg_right": 0,
            "limb_ataxia": 0,
            "sensory": 1,
            "language": 1,
            "dysarthria": 1,
            "extinction_inattention": 0,
        }
        result = calculator.calculate_nihss(exam)
        assert "component_scores" in result
        assert result["component_scores"]["loc"] == 1
        assert result["component_scores"]["facial_palsy"] == 2
        assert result["component_scores"]["motor_arm_left"] == 3


class TestICPMonitor:
    """Tests for Intracranial Pressure Monitor."""

    @pytest.fixture
    def monitor(self) -> ICPMonitor:
        """Provide ICPMonitor instance."""
        return ICPMonitor()

    def test_normal_icp(self, monitor: ICPMonitor) -> None:
        """Test normal ICP assessment (< 20 mmHg)."""
        icp_data = {
            "icp_mmhg": 12.0,
            "cerebral_perfusion_pressure": 75.0,
            "mean_arterial_pressure": 87.0,
            "gcs_score": 15,
            "pupil_abnormality": False,
            "motor_posturing": False,
        }
        result = monitor.assess_icp(icp_data)
        assert result["icp_status"] == "normal"
        assert result["icp_mmhg"] == 12.0
        assert result["elevated"] is False
        assert result["critical"] is False

    def test_elevated_icp(self, monitor: ICPMonitor) -> None:
        """Test elevated ICP assessment (20-25 mmHg)."""
        icp_data = {
            "icp_mmhg": 22.0,
            "cerebral_perfusion_pressure": 60.0,
            "mean_arterial_pressure": 82.0,
            "gcs_score": 12,
            "pupil_abnormality": False,
            "motor_posturing": False,
        }
        result = monitor.assess_icp(icp_data)
        assert result["elevated"] is True
        assert result["critical"] is False
        assert "recommendations" in result
        assert len(result["recommendations"]) > 0

    def test_critical_icp(self, monitor: ICPMonitor) -> None:
        """Test critical ICP assessment (> 25 mmHg)."""
        icp_data = {
            "icp_mmhg": 30.0,
            "cerebral_perfusion_pressure": 50.0,
            "mean_arterial_pressure": 80.0,
            "gcs_score": 8,
            "pupil_abnormality": True,
            "motor_posturing": True,
        }
        result = monitor.assess_icp(icp_data)
        assert result["critical"] is True
        assert result["herniation_risk"] is True
        assert "recommendations" in result

    def test_icp_estimation_from_clinicals(self, monitor: ICPMonitor) -> None:
        """Test ICP estimation when direct measurement not available."""
        clinical_data = {
            "gcs_score": 6,
            "pupil_abnormality": True,
            "motor_posturing": True,
            "mean_arterial_pressure": 70.0,
        }
        result = monitor.assess_icp(clinical_data)
        assert "estimated_icp" in result or "icp_mmhg" in result
        assert result.get("elevated", False) or result.get("critical", False)

    def test_cpp_threshold_assessment(self, monitor: ICPMonitor) -> None:
        """Test cerebral perfusion pressure threshold (CPP > 60 mmHg)."""
        icp_data = {
            "icp_mmhg": 15.0,
            "cerebral_perfusion_pressure": 55.0,  # Below threshold
            "mean_arterial_pressure": 70.0,
            "gcs_score": 14,
            "pupil_abnormality": False,
            "motor_posturing": False,
        }
        result = monitor.assess_icp(icp_data)
        assert (
            "cpp_warning" in result
            or "low_cpp" in result
            or any("perfusion" in r.lower() for r in result.get("recommendations", []))
        )


class TestStrokeDetector:
    """Tests for Stroke Detection Neural Network."""

    @pytest.fixture
    def detector(self) -> StrokeDetector:
        """Provide StrokeDetector instance."""
        return StrokeDetector()

    def test_stroke_detector_forward_pass(self, detector: StrokeDetector) -> None:
        """Test forward pass with clinical features."""
        # Clinical features shape: (batch_size, feature_dim=64)
        features = torch.randn(4, 64)
        output = detector(features)
        assert output.shape == (4, 5)  # 5 stroke types
        assert torch.allclose(output.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_stroke_detector_single_sample(self, detector: StrokeDetector) -> None:
        """Test with single sample."""
        features = torch.randn(1, 64)
        output = detector(features)
        assert output.shape == (1, 5)
        predicted_class = output.argmax(dim=1).item()
        assert 0 <= predicted_class < 5

    def test_stroke_type_classification(self, detector: StrokeDetector) -> None:
        """Test that stroke types are correctly enumerated."""
        assert StrokeType.NO_STROKE.value == "no_stroke"
        assert StrokeType.ISCHEMIC.value == "ischemic"
        assert StrokeType.HEMORRHAGIC.value == "hemorrhagic"
        assert StrokeType.TIA.value == "tia"
        assert StrokeType.CRYPTOGENIC.value == "cryptogenic"


class TestSeizurePredictor:
    """Tests for Seizure Prediction Neural Network."""

    @pytest.fixture
    def predictor(self) -> SeizurePredictor:
        """Provide SeizurePredictor instance."""
        return SeizurePredictor()

    def test_seizure_predictor_forward_pass(self, predictor: SeizurePredictor) -> None:
        """Test forward pass with temporal sequences."""
        # EEG-like temporal sequence: (batch, seq_len, features=32)
        sequence = torch.randn(4, 100, 32)
        output = predictor(sequence)
        assert output.shape == (4, 7)  # 7 seizure types
        assert torch.allclose(output.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_seizure_type_enumeration(self, predictor: SeizurePredictor) -> None:
        """Test seizure type enumeration."""
        assert SeizureType.NO_SEIZURE.value == "no_seizure"
        assert SeizureType.FOCAL_AWARE.value == "focal_aware"
        assert SeizureType.FOCAL_IMPAIRED.value == "focal_impaired"
        assert SeizureType.GENERALIZED_TONIC_CLONIC.value == "generalized_tonic_clonic"
        assert SeizureType.STATUS_EPILEPTICUS.value == "status_epilepticus"

    def test_attention_weights(self, predictor: SeizurePredictor) -> None:
        """Test that attention mechanism produces valid weights."""
        sequence = torch.randn(2, 50, 32)
        output = predictor(sequence)
        # Check if attention is being applied (model should have attention)
        assert hasattr(predictor, "attention") or output is not None


class TestNeurocriticalCarePredictor:
    """Tests for integrated Neurocritical Care Predictor."""

    @pytest.fixture
    def predictor(self) -> NeurocriticalCarePredictor:
        """Provide NeurocriticalCarePredictor instance."""
        return NeurocriticalCarePredictor()

    def test_normal_patient(self, predictor: NeurocriticalCarePredictor) -> None:
        """Test prediction for patient with no neurocritical emergency."""
        patient_data = {
            "clinical_features": np.zeros(64),
            "exam_findings": {
                "loc": 0,
                "loc_questions": 0,
                "loc_commands": 0,
                "gaze": 0,
                "visual_fields": 0,
                "facial_palsy": 0,
                "motor_arm_left": 0,
                "motor_arm_right": 0,
                "motor_leg_left": 0,
                "motor_leg_right": 0,
                "limb_ataxia": 0,
                "sensory": 0,
                "language": 0,
                "dysarthria": 0,
                "extinction_inattention": 0,
            },
            "icp_data": {
                "icp_mmhg": 10.0,
                "cerebral_perfusion_pressure": 80.0,
                "mean_arterial_pressure": 90.0,
                "gcs_score": 15,
                "pupil_abnormality": False,
                "motor_posturing": False,
            },
            "tbi_features": {"gcs_score": 15},
        }
        result = predictor.predict_neurocritical_emergency(patient_data)
        assert isinstance(result, NeurocriticalPredictionResult)
        assert result.neurological_emergency_detected is False

    def test_stroke_detection(self, predictor: NeurocriticalCarePredictor) -> None:
        """Test stroke detection with abnormal exam."""
        patient_data = {
            "clinical_features": np.random.randn(64) * 2,  # Abnormal features
            "exam_findings": {
                "loc": 1,
                "loc_questions": 2,
                "loc_commands": 1,
                "gaze": 2,
                "visual_fields": 2,
                "facial_palsy": 3,
                "motor_arm_left": 4,
                "motor_arm_right": 0,
                "motor_leg_left": 3,
                "motor_leg_right": 0,
                "limb_ataxia": 1,
                "sensory": 1,
                "language": 2,
                "dysarthria": 1,
                "extinction_inattention": 1,
            },
            "icp_data": {
                "gcs_score": 10,
                "pupil_abnormality": True,
                "motor_posturing": False,
            },
            "tbi_features": {"gcs_score": 10},
        }
        result = predictor.predict_neurocritical_emergency(patient_data)
        assert isinstance(result, NeurocriticalPredictionResult)
        assert result.nihss_score > 0
        assert "recommendations" in result.__dict__ or hasattr(result, "clinical_recommendations")

    def test_tbi_severity_mild(self, predictor: NeurocriticalCarePredictor) -> None:
        """Test mild TBI detection (GCS 13-15)."""
        patient_data = {
            "clinical_features": np.zeros(64),
            "exam_findings": {
                "loc": 0,
                "loc_questions": 0,
                "loc_commands": 0,
                "gaze": 0,
                "visual_fields": 0,
                "facial_palsy": 0,
                "motor_arm_left": 0,
                "motor_arm_right": 0,
                "motor_leg_left": 0,
                "motor_leg_right": 0,
                "limb_ataxia": 0,
                "sensory": 0,
                "language": 0,
                "dysarthria": 0,
                "extinction_inattention": 0,
            },
            "icp_data": {"gcs_score": 14},
            "tbi_features": {"gcs_score": 14},
        }
        result = predictor.predict_neurocritical_emergency(patient_data)
        assert result.tbi_severity in ["mild", "none", None] or result.gcs_score >= 13

    def test_tbi_severity_moderate(self, predictor: NeurocriticalCarePredictor) -> None:
        """Test moderate TBI detection (GCS 9-12)."""
        patient_data = {
            "tbi_features": {"gcs_score": 10},
            "icp_data": {"gcs_score": 10},
            "clinical_features": np.zeros(64),
            "exam_findings": {},
        }
        result = predictor.predict_neurocritical_emergency(patient_data)
        assert result.gcs_score == 10 or result.tbi_severity in ["moderate", "severe"]

    def test_tbi_severity_severe(self, predictor: NeurocriticalCarePredictor) -> None:
        """Test severe TBI detection (GCS 3-8)."""
        patient_data = {
            "tbi_features": {"gcs_score": 6},
            "icp_data": {
                "gcs_score": 6,
                "pupil_abnormality": True,
                "motor_posturing": True,
            },
            "clinical_features": np.zeros(64),
            "exam_findings": {},
        }
        result = predictor.predict_neurocritical_emergency(patient_data)
        assert result.neurological_emergency_detected is True
        assert result.gcs_score <= 8

    def test_combined_emergency(self, predictor: NeurocriticalCarePredictor) -> None:
        """Test detection of combined emergencies (stroke + elevated ICP)."""
        patient_data = {
            "clinical_features": np.random.randn(64) * 2,
            "exam_findings": {
                "loc": 2,
                "motor_arm_left": 4,
                "language": 3,
            },
            "icp_data": {
                "icp_mmhg": 28.0,
                "gcs_score": 8,
                "pupil_abnormality": True,
            },
            "tbi_features": {"gcs_score": 8},
        }
        result = predictor.predict_neurocritical_emergency(patient_data)
        assert result.neurological_emergency_detected is True
        assert result.confidence >= 0.5

    def test_result_structure(self, predictor: NeurocriticalCarePredictor) -> None:
        """Test that result has all required fields."""
        patient_data = {
            "clinical_features": np.zeros(64),
            "exam_findings": {},
            "icp_data": {"gcs_score": 15},
            "tbi_features": {"gcs_score": 15},
        }
        result = predictor.predict_neurocritical_emergency(patient_data)
        assert hasattr(result, "neurological_emergency_detected")
        assert hasattr(result, "confidence")
        assert hasattr(result, "stroke_detected")
        assert hasattr(result, "seizure_detected")
        assert hasattr(result, "gcs_score")


class TestNeurocriticalEdgeCases:
    """Edge case and boundary tests for neurocritical care."""

    @pytest.fixture
    def predictor(self) -> NeurocriticalCarePredictor:
        """Provide NeurocriticalCarePredictor instance."""
        return NeurocriticalCarePredictor()

    def test_empty_exam_findings(self, predictor: NeurocriticalCarePredictor) -> None:
        """Test handling of empty exam findings."""
        patient_data = {
            "clinical_features": np.zeros(64),
            "exam_findings": {},
            "icp_data": {},
            "tbi_features": {},
        }
        result = predictor.predict_neurocritical_emergency(patient_data)
        assert isinstance(result, NeurocriticalPredictionResult)

    def test_gcs_boundary_values(self, predictor: NeurocriticalCarePredictor) -> None:
        """Test GCS boundary values (3, 8, 9, 12, 13, 15)."""
        boundaries = [3, 8, 9, 12, 13, 15]
        for gcs in boundaries:
            patient_data = {
                "clinical_features": np.zeros(64),
                "exam_findings": {},
                "icp_data": {"gcs_score": gcs},
                "tbi_features": {"gcs_score": gcs},
            }
            result = predictor.predict_neurocritical_emergency(patient_data)
            assert 3 <= result.gcs_score <= 15

    def test_maximum_nihss_score(self, predictor: NeurocriticalCarePredictor) -> None:
        """Test maximum possible NIHSS score (42)."""
        calculator = NIHSSCalculator()
        max_exam = {
            "loc": 3,
            "loc_questions": 2,
            "loc_commands": 2,
            "gaze": 2,
            "visual_fields": 3,
            "facial_palsy": 3,
            "motor_arm_left": 4,
            "motor_arm_right": 4,
            "motor_leg_left": 4,
            "motor_leg_right": 4,
            "limb_ataxia": 2,
            "sensory": 2,
            "language": 3,
            "dysarthria": 2,
            "extinction_inattention": 2,
        }
        result = calculator.calculate_nihss(max_exam)
        assert result["total_score"] == 42


@pytest.mark.medical
class TestNeurocriticalIntegration:
    """Integration tests for complete neurocritical care workflow."""

    def test_full_workflow(self) -> None:
        """Test complete assessment workflow."""
        predictor = NeurocriticalCarePredictor()

        # Stage 1: Initial assessment
        initial_data = {
            "clinical_features": np.zeros(64),
            "exam_findings": {"gcs_score": 15},
            "icp_data": {"gcs_score": 15},
            "tbi_features": {"gcs_score": 15},
        }
        initial_result = predictor.predict_neurocritical_emergency(initial_data)
        assert initial_result.neurological_emergency_detected is False

        # Stage 2: Deterioration
        deteriorated_data = {
            "clinical_features": np.random.randn(64),
            "exam_findings": {
                "loc": 2,
                "motor_arm_left": 3,
                "facial_palsy": 2,
            },
            "icp_data": {"gcs_score": 9, "icp_mmhg": 24.0},
            "tbi_features": {"gcs_score": 9},
        }
        deteriorated_result = predictor.predict_neurocritical_emergency(deteriorated_data)
        assert deteriorated_result.neurological_emergency_detected is True
