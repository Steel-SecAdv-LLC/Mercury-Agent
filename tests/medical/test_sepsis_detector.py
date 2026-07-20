# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the sepsis_detector module - medical sepsis detection."""

from __future__ import annotations

from typing import Any

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
    def calculator(self) -> SOFACalculator:
        """Provide SOFA calculator instance."""
        return SOFACalculator()

    def test_respiration_score_normal(self, calculator: Any) -> None:
        """Test normal respiration score (PaO2/FiO2 >= 400)."""
        score = calculator._calculate_respiration({"pao2_fio2_ratio": 450})
        assert score == 0

    def test_respiration_score_mild(self, calculator: Any) -> None:
        """Test mild respiratory failure (300-399)."""
        score = calculator._calculate_respiration({"pao2_fio2_ratio": 350})
        assert score == 1

    def test_respiration_score_moderate(self, calculator: Any) -> None:
        """Test moderate respiratory failure (200-299)."""
        score = calculator._calculate_respiration({"pao2_fio2_ratio": 250})
        assert score == 2

    def test_respiration_score_severe_requires_support(self, calculator: Any) -> None:
        """SOFA 3 (PaO2/FiO2 100-199) requires respiratory support; else caps at 2."""
        # Without respiratory support, <200 caps at 2 (Vincent 1996).
        assert calculator._calculate_respiration({"pao2_fio2_ratio": 150}) == 2
        # With respiratory support, 100-199 scores 3.
        assert (
            calculator._calculate_respiration(
                {"pao2_fio2_ratio": 150, "mechanical_ventilation": True}
            )
            == 3
        )

    def test_respiration_score_critical_requires_support(self, calculator: Any) -> None:
        """SOFA 4 (PaO2/FiO2 <100) requires respiratory support; else caps at 2."""
        assert calculator._calculate_respiration({"pao2_fio2_ratio": 80}) == 2
        assert (
            calculator._calculate_respiration(
                {"pao2_fio2_ratio": 80, "mechanical_ventilation": True}
            )
            == 4
        )

    def test_coagulation_score_normal(self, calculator: Any) -> None:
        """Test normal platelet count (>= 150k)."""
        score = calculator._calculate_coagulation({"platelets_k_ul": 200})
        assert score == 0

    def test_coagulation_score_low(self, calculator: Any) -> None:
        """Test low platelet count (100-149k)."""
        score = calculator._calculate_coagulation({"platelets_k_ul": 120})
        assert score == 1

    def test_coagulation_score_very_low(self, calculator: Any) -> None:
        """Test very low platelet count (50-99k)."""
        score = calculator._calculate_coagulation({"platelets_k_ul": 70})
        assert score == 2

    def test_coagulation_score_critical(self, calculator: Any) -> None:
        """Test critical platelet count (<20k)."""
        score = calculator._calculate_coagulation({"platelets_k_ul": 15})
        assert score == 4

    def test_liver_score_normal(self, calculator: Any) -> None:
        """Test normal bilirubin (<1.2 mg/dL)."""
        score = calculator._calculate_liver({"bilirubin_mg_dl": 0.8})
        assert score == 0

    def test_liver_score_elevated(self, calculator: Any) -> None:
        """Test elevated bilirubin (1.2-1.9 mg/dL)."""
        score = calculator._calculate_liver({"bilirubin_mg_dl": 1.5})
        assert score == 1

    def test_liver_score_high(self, calculator: Any) -> None:
        """Test high bilirubin (2.0-5.9 mg/dL)."""
        score = calculator._calculate_liver({"bilirubin_mg_dl": 4.0})
        assert score == 2

    def test_liver_score_severe(self, calculator: Any) -> None:
        """Test severe bilirubin (>=12 mg/dL)."""
        score = calculator._calculate_liver({"bilirubin_mg_dl": 15.0})
        assert score == 4

    def test_cardiovascular_score_normal(self, calculator: Any) -> None:
        """Test normal cardiovascular (MAP >= 70)."""
        score = calculator._calculate_cardiovascular({"mean_arterial_pressure": 80})
        assert score == 0

    def test_cardiovascular_score_hypotension(self, calculator: Any) -> None:
        """Test hypotension without vasopressors (MAP < 70)."""
        score = calculator._calculate_cardiovascular({"mean_arterial_pressure": 60})
        assert score == 1

    def test_cns_score_normal(self, calculator: Any) -> None:
        """Test normal GCS (15)."""
        score = calculator._calculate_cns({"gcs_score": 15})
        assert score == 0

    def test_cns_score_mild(self, calculator: Any) -> None:
        """Test mildly impaired GCS (13-14)."""
        score = calculator._calculate_cns({"gcs_score": 14})
        assert score == 1

    def test_cns_score_moderate(self, calculator: Any) -> None:
        """Test moderately impaired GCS (10-12)."""
        score = calculator._calculate_cns({"gcs_score": 11})
        assert score == 2

    def test_cns_score_severe(self, calculator: Any) -> None:
        """Test severely impaired GCS (<6)."""
        score = calculator._calculate_cns({"gcs_score": 5})
        assert score == 4

    def test_renal_score_normal(self, calculator: Any) -> None:
        """Test normal creatinine (<1.2 mg/dL)."""
        score = calculator._calculate_renal({"creatinine_mg_dl": 1.0, "urine_output_ml_day": 1000})
        assert score == 0

    def test_renal_score_elevated(self, calculator: Any) -> None:
        """Test elevated creatinine (1.2-1.9 mg/dL)."""
        score = calculator._calculate_renal({"creatinine_mg_dl": 1.5, "urine_output_ml_day": 1000})
        assert score == 1

    def test_calculate_sofa_healthy_patient(self, calculator: Any) -> None:
        """Test SOFA calculation for healthy patient."""
        patient_data = {
            "pao2_fio2_ratio": 450,
            "platelets_k_ul": 250,
            "bilirubin_mg_dl": 0.5,
            "mean_arterial_pressure": 85,
            "gcs_score": 15,
            "creatinine_mg_dl": 0.9,
            "urine_output_ml_day": 1500,
        }

        result = calculator.calculate_sofa(patient_data)

        assert result["sofa_score"] == 0
        assert len(result["organ_dysfunctions"]) == 0

    def test_calculate_sofa_septic_patient(self, calculator: Any) -> None:
        """Test SOFA calculation for septic patient."""
        patient_data = {
            "pao2_fio2_ratio": 250,  # Score 2
            "platelets_k_ul": 80,  # Score 2
            "bilirubin_mg_dl": 3.0,  # Score 2
            "mean_arterial_pressure": 60,  # Score 1
            "gcs_score": 14,  # Score 1
            "creatinine_mg_dl": 1.5,  # Score 1
            "urine_output_ml_day": 800,
        }

        result = calculator.calculate_sofa(patient_data)

        assert result["sofa_score"] >= 2  # Sepsis threshold
        assert len(result["organ_dysfunctions"]) >= 2


class TestQuickSOFACalculator:
    """Tests for qSOFA bedside screening."""

    @pytest.fixture
    def calculator(self) -> QuickSOFACalculator:
        """Provide qSOFA calculator instance."""
        return QuickSOFACalculator()

    def test_qsofa_normal(self, calculator: Any) -> None:
        """Test qSOFA for normal vital signs."""
        vitals = {
            "respiratory_rate_bpm": 16,
            "gcs_score": 15,
            "systolic_bp_mmhg": 120,
        }

        result = calculator.calculate_qsofa(vitals)

        assert result["qsofa_score"] == 0
        assert result["qsofa_positive"] is False

    def test_qsofa_tachypnea(self, calculator: Any) -> None:
        """Test qSOFA with tachypnea (RR >= 22)."""
        vitals = {
            "respiratory_rate_bpm": 24,
            "gcs_score": 15,
            "systolic_bp_mmhg": 120,
        }

        result = calculator.calculate_qsofa(vitals)

        assert result["qsofa_score"] >= 1
        assert (
            "tachypnea" in str(result["criteria_met"]).lower()
            or "respiratory" in str(result["criteria_met"]).lower()
        )

    def test_qsofa_altered_mentation(self, calculator: Any) -> None:
        """Test qSOFA with altered mentation (GCS < 15)."""
        vitals = {
            "respiratory_rate_bpm": 16,
            "gcs_score": 13,
            "systolic_bp_mmhg": 120,
        }

        result = calculator.calculate_qsofa(vitals)

        assert result["qsofa_score"] >= 1

    def test_qsofa_hypotension(self, calculator: Any) -> None:
        """Test qSOFA with hypotension (SBP <= 100)."""
        vitals = {
            "respiratory_rate_bpm": 16,
            "gcs_score": 15,
            "systolic_bp_mmhg": 95,
        }

        result = calculator.calculate_qsofa(vitals)

        assert result["qsofa_score"] >= 1

    def test_qsofa_positive_two_criteria(self, calculator: Any) -> None:
        """Test qSOFA positive with 2+ criteria."""
        vitals = {
            "respiratory_rate_bpm": 25,  # +1
            "gcs_score": 13,  # +1
            "systolic_bp_mmhg": 120,
        }

        result = calculator.calculate_qsofa(vitals)

        assert result["qsofa_score"] >= 2
        assert result["qsofa_positive"] is True

    def test_qsofa_positive_all_criteria(self, calculator: Any) -> None:
        """Test qSOFA with all three criteria met."""
        vitals = {
            "respiratory_rate_bpm": 28,  # +1
            "gcs_score": 12,  # +1
            "systolic_bp_mmhg": 85,  # +1
        }

        result = calculator.calculate_qsofa(vitals)

        assert result["qsofa_score"] == 3
        assert result["qsofa_positive"] is True


class TestSepsisDetector:
    """Tests for the main SepsisDetector class."""

    @pytest.fixture
    def detector(self) -> SepsisDetector:
        """Provide SepsisDetector instance."""
        return SepsisDetector(enable_ml_prediction=False)

    def test_detector_initialization(self, detector: Any) -> None:
        """Test detector can be initialized."""
        assert detector is not None
        assert hasattr(detector, "detect_sepsis") or hasattr(detector, "detect")

    def test_detect_no_sepsis(self, detector: Any) -> None:
        """Test detection for healthy patient."""
        patient_data = {
            "vital_signs": {
                "respiratory_rate_bpm": 14,
                "gcs_score": 15,
                "systolic_bp_mmhg": 125,
            },
            "laboratory_values": {
                "pao2_fio2_ratio": 450,
                "platelets_k_ul": 280,
                "bilirubin_mg_dl": 0.6,
                "creatinine_mg_dl": 0.8,
                "mean_arterial_pressure": 90,
                "urine_output_ml_day": 1800,
            },
        }

        result = detector.detect_sepsis(patient_data)

        assert result is not None
        assert result.sepsis_stage in [SepsisStage.NO_SEPSIS.value, "no_sepsis", 0]

    def test_detect_sepsis(self, detector: Any) -> None:
        """Test detection for septic patient."""
        patient_data = {
            "vital_signs": {
                "respiratory_rate_bpm": 26,
                "gcs_score": 13,
                "systolic_bp_mmhg": 90,
            },
            "laboratory_values": {
                "pao2_fio2_ratio": 200,
                "platelets_k_ul": 90,
                "bilirubin_mg_dl": 3.5,
                "creatinine_mg_dl": 2.5,
                "mean_arterial_pressure": 58,
                "urine_output_ml_day": 300,
            },
        }

        result = detector.detect_sepsis(patient_data)

        assert result is not None
        # Should detect some level of sepsis
        assert result.sepsis_stage not in [SepsisStage.NO_SEPSIS.value, "no_sepsis", 0]
        assert result.sofa_score >= 2

    def test_recommendations_generated(self, detector: Any) -> None:
        """Test that recommendations are generated for sepsis."""
        patient_data = {
            "vital_signs": {
                "respiratory_rate_bpm": 28,
                "gcs_score": 12,
                "systolic_bp_mmhg": 85,
            },
            "laboratory_values": {
                "pao2_fio2_ratio": 180,
                "platelets_k_ul": 70,
                "bilirubin_mg_dl": 5.0,
                "creatinine_mg_dl": 3.0,
                "mean_arterial_pressure": 55,
                "norepinephrine_mcg_kg_min": 0.2,
                "urine_output_ml_day": 200,
            },
        }

        result = detector.detect_sepsis(patient_data)

        assert result.clinical_recommendations is not None
        assert len(result.clinical_recommendations) > 0

    def test_bundle_checklist_generated(self, detector: Any) -> None:
        """Test that bundle checklist is generated for sepsis."""
        patient_data = {
            "vital_signs": {
                "respiratory_rate_bpm": 24,
                "gcs_score": 14,
                "systolic_bp_mmhg": 95,
            },
            "laboratory_values": {
                "pao2_fio2_ratio": 280,
                "platelets_k_ul": 110,
                "bilirubin_mg_dl": 2.0,
                "creatinine_mg_dl": 1.8,
                "mean_arterial_pressure": 62,
                "urine_output_ml_day": 500,
            },
        }

        result = detector.detect_sepsis(patient_data)

        if result.sepsis_stage not in [SepsisStage.NO_SEPSIS.value, "no_sepsis", 0]:
            assert result.bundle_compliance is not None


class TestSOFAFailClosed:
    """Missing inputs must abstain (unassessed), never score as a healthy organ."""

    @pytest.fixture
    def calculator(self) -> SOFACalculator:
        """Provide SOFA calculator instance."""
        return SOFACalculator()

    def test_missing_organ_inputs_return_none(self, calculator: Any) -> None:
        """Each organ scorer returns None when its required input is absent."""
        assert calculator._calculate_respiration({}) is None
        assert calculator._calculate_coagulation({}) is None
        assert calculator._calculate_liver({}) is None
        assert calculator._calculate_cardiovascular({}) is None
        assert calculator._calculate_cns({}) is None
        assert calculator._calculate_renal({}) is None

    def test_empty_input_scores_zero_not_healthy(self, calculator: Any) -> None:
        """An empty SOFA input yields 0 assessed organs, all unassessed — not a healthy 0/24."""
        result = calculator.calculate_sofa({})
        assert result["sofa_score"] == 0
        assert result["sofa_is_lower_bound"] is True
        assert set(result["unassessed_organs"]) == {
            "respiration",
            "coagulation",
            "liver",
            "cardiovascular",
            "cns",
            "renal",
        }
        assert result["assessed_organs"] == []

    def test_partial_input_is_lower_bound(self, calculator: Any) -> None:
        """A partial SOFA is flagged as a lower bound and lists what was unassessed."""
        # Only platelets present (severe thrombocytopenia). Missing organs must
        # not be scored as 0 — the total is a lower bound.
        result = calculator.calculate_sofa({"platelets_k_ul": 15})
        assert result["sofa_score"] == 4  # coagulation only
        assert result["sofa_is_lower_bound"] is True
        assert "coagulation" in result["assessed_organs"]
        assert "renal" in result["unassessed_organs"]

    def test_missing_creatinine_does_not_falsely_normalize_renal(self, calculator: Any) -> None:
        """A missing creatinine with normal urine abstains rather than scoring renal 0."""
        assert calculator._calculate_renal({"urine_output_ml_day": 1500}) is None
        # Severe oliguria alone is still determinable.
        assert calculator._calculate_renal({"urine_output_ml_day": 150}) == 4


class TestQSOFAFailClosed:
    """qSOFA abstains on missing components instead of assuming normal vitals."""

    @pytest.fixture
    def calculator(self) -> QuickSOFACalculator:
        """Provide qSOFA calculator instance."""
        return QuickSOFACalculator()

    def test_missing_components_are_unassessed(self, calculator: Any) -> None:
        """Absent vitals are reported unassessed and do not count as normal."""
        result = calculator.calculate_qsofa({"respiratory_rate_bpm": 28})
        assert result["qsofa_is_lower_bound"] is True
        assert set(result["unassessed_components"]) == {"gcs_score", "systolic_bp_mmhg"}
        assert result["qsofa_score"] == 1  # only the one present, met criterion


class TestSepsisMLGate:
    """The untrained progression network must never surface a number."""

    def test_untrained_model_abstains(self) -> None:
        """With ML enabled but no trained weights, no ML risk is emitted."""
        import numpy as np

        detector = SepsisDetector(enable_ml_prediction=True)
        assert detector.progression_predictor is not None
        assert detector.progression_predictor.is_fitted is False

        patient_data = {
            "vital_signs": {
                "respiratory_rate_bpm": 26,
                "gcs_score": 13,
                "systolic_bp_mmhg": 90,
            },
            "temporal_sequence": np.zeros((8, 32), dtype=np.float32),
        }
        result = detector.detect_sepsis(patient_data)
        assert result.ml_available is False
        # Untrained net contributes nothing to the surfaced risks.
        assert result.septic_shock_risk == 0.0
        assert result.mortality_risk == 0.0

    def test_predict_progression_reports_unavailable(self) -> None:
        """_predict_progression returns available=False when unfitted."""
        import numpy as np

        detector = SepsisDetector(enable_ml_prediction=True)
        out = detector._predict_progression(np.zeros((8, 32), dtype=np.float32))
        assert out["available"] is False
        assert out["shock_risk"] is None


class TestSepsisSafetyEnvelope:
    """Every result carries a disclaimer, provenance, and emergency routing."""

    @pytest.fixture
    def detector(self) -> SepsisDetector:
        """Provide SepsisDetector without ML."""
        return SepsisDetector(enable_ml_prediction=False)

    def test_disclaimer_and_provenance_present(self, detector: Any) -> None:
        """Result carries a decision-support disclaimer and input-hashed provenance."""
        patient_data = {
            "vital_signs": {
                "respiratory_rate_bpm": 16,
                "gcs_score": 15,
                "systolic_bp_mmhg": 120,
            },
            "laboratory_values": {
                "pao2_fio2_ratio": 450,
                "platelets_k_ul": 250,
                "bilirubin_mg_dl": 0.5,
                "mean_arterial_pressure": 85,
                "gcs_score": 15,
                "creatinine_mg_dl": 0.9,
                "urine_output_ml_day": 1500,
            },
        }
        result = detector.detect_sepsis(patient_data)
        assert result.safety.is_decision_support is True
        assert result.safety.is_diagnosis is False
        assert "decision-support" in result.safety.disclaimer
        assert "sofa" in result.safety.provenance
        assert len(result.safety.provenance["sofa"]["input_sha256"]) == 64

    def test_septic_shock_flags_emergency(self, detector: Any) -> None:
        """A septic presentation routes a lay reader to emergency care."""
        patient_data = {
            "vital_signs": {
                "respiratory_rate_bpm": 28,
                "gcs_score": 12,
                "systolic_bp_mmhg": 85,
            },
            "laboratory_values": {
                "pao2_fio2_ratio": 180,
                "platelets_k_ul": 70,
                "bilirubin_mg_dl": 5.0,
                "mean_arterial_pressure": 55,
                "gcs_score": 12,
                "creatinine_mg_dl": 3.0,
                "norepinephrine_mcg_kg_min": 0.2,
                "urine_output_ml_day": 200,
            },
        }
        result = detector.detect_sepsis(patient_data)
        assert result.sepsis_detected is True
        assert result.safety.emergency is True
        assert result.safety.emergency_guidance is not None
        assert "emergency" in result.safety.emergency_guidance.lower()

    def test_healthy_patient_no_emergency(self, detector: Any) -> None:
        """A clearly healthy assessment does not raise the emergency flag."""
        patient_data = {
            "vital_signs": {
                "respiratory_rate_bpm": 14,
                "gcs_score": 15,
                "systolic_bp_mmhg": 125,
            },
            "laboratory_values": {
                "pao2_fio2_ratio": 450,
                "platelets_k_ul": 280,
                "bilirubin_mg_dl": 0.6,
                "creatinine_mg_dl": 0.8,
                "mean_arterial_pressure": 90,
                "gcs_score": 15,
                "urine_output_ml_day": 1800,
            },
        }
        result = detector.detect_sepsis(patient_data)
        assert result.safety.emergency is False


class TestSepsisStage:
    """Tests for SepsisStage enum."""

    def test_stages_exist(self) -> None:
        """Test all expected sepsis stages are defined."""
        assert hasattr(SepsisStage, "NO_SEPSIS")
        assert hasattr(SepsisStage, "SEPSIS")
        assert hasattr(SepsisStage, "SEVERE_SEPSIS") or hasattr(SepsisStage, "SEPTIC_SHOCK")

    def test_stage_ordering(self) -> None:
        """Test stages have logical ordering."""
        # NO_SEPSIS should be lowest severity
        assert SepsisStage.NO_SEPSIS.value <= SepsisStage.SEPSIS.value
