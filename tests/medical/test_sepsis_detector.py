"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

Tests for the sepsis_detector module - medical sepsis detection.
"""

from __future__ import annotations

import pytest

try:
    from omni_mercury_engine.medical.critical_care.sepsis_detector import (
        QuickSOFACalculator,
        SepsisDetector,
        SepsisStage,
        SOFACalculator,
    )

    HAS_SEPSIS = True
except ImportError:
    HAS_SEPSIS = False


pytestmark = pytest.mark.skipif(not HAS_SEPSIS, reason="sepsis_detector not available")


class TestSOFACalculator:
    """Tests for Sequential Organ Failure Assessment scoring."""

    @pytest.fixture
    def calculator(self):
        """Provide SOFA calculator instance."""
        return SOFACalculator()

    def test_respiration_score_normal(self, calculator):
        """Test normal respiration score (PaO2/FiO2 >= 400)."""
        score = calculator._calculate_respiration({"pao2_fio2_ratio": 450})
        assert score == 0

    def test_respiration_score_mild(self, calculator):
        """Test mild respiratory failure (300-399)."""
        score = calculator._calculate_respiration({"pao2_fio2_ratio": 350})
        assert score == 1

    def test_respiration_score_moderate(self, calculator):
        """Test moderate respiratory failure (200-299)."""
        score = calculator._calculate_respiration({"pao2_fio2_ratio": 250})
        assert score == 2

    def test_respiration_score_severe(self, calculator):
        """Test severe respiratory failure (100-199)."""
        score = calculator._calculate_respiration({"pao2_fio2_ratio": 150})
        assert score == 3

    def test_respiration_score_critical(self, calculator):
        """Test critical respiratory failure (<100)."""
        score = calculator._calculate_respiration({"pao2_fio2_ratio": 80})
        assert score == 4

    def test_coagulation_score_normal(self, calculator):
        """Test normal platelet count (>= 150k)."""
        score = calculator._calculate_coagulation({"platelets": 200})
        assert score == 0

    def test_coagulation_score_low(self, calculator):
        """Test low platelet count (100-149k)."""
        score = calculator._calculate_coagulation({"platelets": 120})
        assert score == 1

    def test_coagulation_score_very_low(self, calculator):
        """Test very low platelet count (50-99k)."""
        score = calculator._calculate_coagulation({"platelets": 70})
        assert score == 2

    def test_coagulation_score_critical(self, calculator):
        """Test critical platelet count (<20k)."""
        score = calculator._calculate_coagulation({"platelets": 15})
        assert score == 4

    def test_liver_score_normal(self, calculator):
        """Test normal bilirubin (<1.2 mg/dL)."""
        score = calculator._calculate_liver({"bilirubin": 0.8})
        assert score == 0

    def test_liver_score_elevated(self, calculator):
        """Test elevated bilirubin (1.2-1.9 mg/dL)."""
        score = calculator._calculate_liver({"bilirubin": 1.5})
        assert score == 1

    def test_liver_score_high(self, calculator):
        """Test high bilirubin (2.0-5.9 mg/dL)."""
        score = calculator._calculate_liver({"bilirubin": 4.0})
        assert score == 2

    def test_liver_score_severe(self, calculator):
        """Test severe bilirubin (>=12 mg/dL)."""
        score = calculator._calculate_liver({"bilirubin": 15.0})
        assert score == 4

    def test_cardiovascular_score_normal(self, calculator):
        """Test normal cardiovascular (MAP >= 70)."""
        score = calculator._calculate_cardiovascular({"map": 80, "vasopressors": None})
        assert score == 0

    def test_cardiovascular_score_hypotension(self, calculator):
        """Test hypotension without vasopressors (MAP < 70)."""
        score = calculator._calculate_cardiovascular({"map": 60, "vasopressors": None})
        assert score == 1

    def test_cns_score_normal(self, calculator):
        """Test normal GCS (15)."""
        score = calculator._calculate_cns({"gcs": 15})
        assert score == 0

    def test_cns_score_mild(self, calculator):
        """Test mildly impaired GCS (13-14)."""
        score = calculator._calculate_cns({"gcs": 14})
        assert score == 1

    def test_cns_score_moderate(self, calculator):
        """Test moderately impaired GCS (10-12)."""
        score = calculator._calculate_cns({"gcs": 11})
        assert score == 2

    def test_cns_score_severe(self, calculator):
        """Test severely impaired GCS (<6)."""
        score = calculator._calculate_cns({"gcs": 5})
        assert score == 4

    def test_renal_score_normal(self, calculator):
        """Test normal creatinine (<1.2 mg/dL)."""
        score = calculator._calculate_renal({"creatinine": 1.0, "urine_output": 1000})
        assert score == 0

    def test_renal_score_elevated(self, calculator):
        """Test elevated creatinine (1.2-1.9 mg/dL)."""
        score = calculator._calculate_renal({"creatinine": 1.5, "urine_output": 1000})
        assert score == 1

    def test_calculate_sofa_healthy_patient(self, calculator):
        """Test SOFA calculation for healthy patient."""
        patient_data = {
            "pao2_fio2_ratio": 450,
            "platelets": 250,
            "bilirubin": 0.5,
            "map": 85,
            "vasopressors": None,
            "gcs": 15,
            "creatinine": 0.9,
            "urine_output": 1500,
        }

        result = calculator.calculate_sofa(patient_data)

        assert result["total_score"] == 0
        assert len(result["organ_dysfunctions"]) == 0

    def test_calculate_sofa_septic_patient(self, calculator):
        """Test SOFA calculation for septic patient."""
        patient_data = {
            "pao2_fio2_ratio": 250,  # Score 2
            "platelets": 80,  # Score 2
            "bilirubin": 3.0,  # Score 2
            "map": 60,  # Score 1
            "vasopressors": None,
            "gcs": 14,  # Score 1
            "creatinine": 1.5,  # Score 1
            "urine_output": 800,
        }

        result = calculator.calculate_sofa(patient_data)

        assert result["total_score"] >= 2  # Sepsis threshold
        assert len(result["organ_dysfunctions"]) >= 2


class TestQuickSOFACalculator:
    """Tests for qSOFA bedside screening."""

    @pytest.fixture
    def calculator(self):
        """Provide qSOFA calculator instance."""
        return QuickSOFACalculator()

    def test_qsofa_normal(self, calculator):
        """Test qSOFA for normal vital signs."""
        vitals = {
            "respiratory_rate": 16,
            "gcs": 15,
            "systolic_bp": 120,
        }

        result = calculator.calculate_qsofa(vitals)

        assert result["score"] == 0
        assert result["positive"] is False

    def test_qsofa_tachypnea(self, calculator):
        """Test qSOFA with tachypnea (RR >= 22)."""
        vitals = {
            "respiratory_rate": 24,
            "gcs": 15,
            "systolic_bp": 120,
        }

        result = calculator.calculate_qsofa(vitals)

        assert result["score"] >= 1
        assert "tachypnea" in result["criteria"] or "respiratory" in str(result["criteria"]).lower()

    def test_qsofa_altered_mentation(self, calculator):
        """Test qSOFA with altered mentation (GCS < 15)."""
        vitals = {
            "respiratory_rate": 16,
            "gcs": 13,
            "systolic_bp": 120,
        }

        result = calculator.calculate_qsofa(vitals)

        assert result["score"] >= 1

    def test_qsofa_hypotension(self, calculator):
        """Test qSOFA with hypotension (SBP <= 100)."""
        vitals = {
            "respiratory_rate": 16,
            "gcs": 15,
            "systolic_bp": 95,
        }

        result = calculator.calculate_qsofa(vitals)

        assert result["score"] >= 1

    def test_qsofa_positive_two_criteria(self, calculator):
        """Test qSOFA positive with 2+ criteria."""
        vitals = {
            "respiratory_rate": 25,  # +1
            "gcs": 13,  # +1
            "systolic_bp": 120,
        }

        result = calculator.calculate_qsofa(vitals)

        assert result["score"] >= 2
        assert result["positive"] is True

    def test_qsofa_positive_all_criteria(self, calculator):
        """Test qSOFA with all three criteria met."""
        vitals = {
            "respiratory_rate": 28,  # +1
            "gcs": 12,  # +1
            "systolic_bp": 85,  # +1
        }

        result = calculator.calculate_qsofa(vitals)

        assert result["score"] == 3
        assert result["positive"] is True


class TestSepsisDetector:
    """Tests for the main SepsisDetector class."""

    @pytest.fixture
    def detector(self):
        """Provide SepsisDetector instance."""
        return SepsisDetector()

    def test_detector_initialization(self, detector):
        """Test detector can be initialized."""
        assert detector is not None
        assert hasattr(detector, "detect_sepsis") or hasattr(detector, "detect")

    def test_detect_no_sepsis(self, detector):
        """Test detection for healthy patient."""
        patient_data = {
            "vital_signs": {
                "respiratory_rate": 14,
                "gcs": 15,
                "systolic_bp": 125,
            },
            "labs": {
                "pao2_fio2_ratio": 450,
                "platelets": 280,
                "bilirubin": 0.6,
                "creatinine": 0.8,
            },
            "map": 90,
            "vasopressors": None,
            "urine_output": 1800,
        }

        result = detector.detect_sepsis(patient_data)

        assert result is not None
        assert result.stage in [SepsisStage.NO_SEPSIS, "NO_SEPSIS", 0]

    def test_detect_sepsis(self, detector):
        """Test detection for septic patient."""
        patient_data = {
            "vital_signs": {
                "respiratory_rate": 26,
                "gcs": 13,
                "systolic_bp": 90,
            },
            "labs": {
                "pao2_fio2_ratio": 200,
                "platelets": 90,
                "bilirubin": 3.5,
                "creatinine": 2.5,
            },
            "map": 58,
            "vasopressors": None,
            "urine_output": 300,
        }

        result = detector.detect_sepsis(patient_data)

        assert result is not None
        # Should detect some level of sepsis
        assert result.stage not in [SepsisStage.NO_SEPSIS, "NO_SEPSIS", 0]
        assert result.sofa_score >= 2

    def test_recommendations_generated(self, detector):
        """Test that recommendations are generated for sepsis."""
        patient_data = {
            "vital_signs": {
                "respiratory_rate": 28,
                "gcs": 12,
                "systolic_bp": 85,
            },
            "labs": {
                "pao2_fio2_ratio": 180,
                "platelets": 70,
                "bilirubin": 5.0,
                "creatinine": 3.0,
            },
            "map": 55,
            "vasopressors": "norepinephrine",
            "urine_output": 200,
        }

        result = detector.detect_sepsis(patient_data)

        assert result.recommendations is not None
        assert len(result.recommendations) > 0

    def test_bundle_checklist_generated(self, detector):
        """Test that bundle checklist is generated for sepsis."""
        patient_data = {
            "vital_signs": {
                "respiratory_rate": 24,
                "gcs": 14,
                "systolic_bp": 95,
            },
            "labs": {
                "pao2_fio2_ratio": 280,
                "platelets": 110,
                "bilirubin": 2.0,
                "creatinine": 1.8,
            },
            "map": 62,
            "vasopressors": None,
            "urine_output": 500,
        }

        result = detector.detect_sepsis(patient_data)

        if result.stage not in [SepsisStage.NO_SEPSIS, "NO_SEPSIS", 0]:
            assert result.bundle_checklist is not None


class TestSepsisStage:
    """Tests for SepsisStage enum."""

    def test_stages_exist(self):
        """Test all expected sepsis stages are defined."""
        assert hasattr(SepsisStage, "NO_SEPSIS")
        assert hasattr(SepsisStage, "SEPSIS")
        assert hasattr(SepsisStage, "SEVERE_SEPSIS") or hasattr(SepsisStage, "SEPTIC_SHOCK")

    def test_stage_ordering(self):
        """Test stages have logical ordering."""
        # NO_SEPSIS should be lowest severity
        assert SepsisStage.NO_SEPSIS.value <= SepsisStage.SEPSIS.value
