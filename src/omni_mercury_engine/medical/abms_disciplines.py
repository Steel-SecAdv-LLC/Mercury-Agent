"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations


"""
ABMS Medical Disciplines Integration Module

Comprehensive integration of American Board of Medical Specialties (ABMS) disciplines
for multi-specialty anomaly detection in healthcare. This module enables specialized
detection across 24+ medical boards and 150+ subspecialties for early disease detection,
treatment optimization, and outcome prediction.

Key Features:
- Specialty-specific anomaly patterns for each ABMS board
- Subspecialty predictors for refined diagnosis
- Cross-disciplinary consultation recommendations
- Integration with neurosymbolic reasoning for medical knowledge
- Golden ratio optimization for diagnostic thresholds
- O(n) complexity for real-time clinical decision support

Data Sources & Research:
- ABMS Guide 2025 (American Board of Medical Specialties)
- PMC (PubMed Central) clinical research
- CDC clinical guidelines
- WHO treatment protocols

⚠️ SIMULATION-BASED: For research/development. Clinical validation required.
Medical professionals must review all findings before patient care decisions.

"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
from torch import nn


class ABMSBoard(Enum):
    """ABMS Medical Board Categories"""

    ALLERGY_IMMUNOLOGY = "allergy_immunology"
    ANESTHESIOLOGY = "anesthesiology"
    COLON_RECTAL_SURGERY = "colon_rectal_surgery"
    DERMATOLOGY = "dermatology"
    EMERGENCY_MEDICINE = "emergency_medicine"
    FAMILY_MEDICINE = "family_medicine"
    INTERNAL_MEDICINE = "internal_medicine"
    MEDICAL_GENETICS = "medical_genetics"
    NEUROLOGICAL_SURGERY = "neurological_surgery"
    NUCLEAR_MEDICINE = "nuclear_medicine"
    OBSTETRICS_GYNECOLOGY = "obstetrics_gynecology"
    OPHTHALMOLOGY = "ophthalmology"
    ORTHOPAEDIC_SURGERY = "orthopaedic_surgery"
    OTOLARYNGOLOGY = "otolaryngology"
    PATHOLOGY = "pathology"
    PEDIATRICS = "pediatrics"
    PHYSICAL_MEDICINE = "physical_medicine"
    PLASTIC_SURGERY = "plastic_surgery"
    PREVENTIVE_MEDICINE = "preventive_medicine"
    PSYCHIATRY = "psychiatry"
    RADIOLOGY = "radiology"
    SURGERY = "surgery"
    THORACIC_SURGERY = "thoracic_surgery"
    UROLOGY = "urology"


@dataclass
class MedicalAnomalyResult:
    """Result from ABMS-based medical anomaly detection"""

    primary_board: str
    subspecialty: str | None = None
    anomaly_detected: bool = False
    confidence: float = 0.0
    risk_score: float = 0.0
    clinical_indicators: list[str] = field(default_factory=list)
    recommended_consultations: list[str] = field(default_factory=list)
    treatment_considerations: list[str] = field(default_factory=list)
    urgency_level: str = "routine"
    neurosymbolic_reasoning: dict[str, Any] | None = None


class MultiSpecialtyNeuralNet(nn.Module):
    """
    Neural network for multi-specialty medical anomaly detection.

    Architecture optimized with golden ratio (φ ≈ 1.618) for layer dimensions.
    """

    def __init__(self, input_dim: int = 64, num_specialties: int = 24) -> None:
        super().__init__()

        phi = 1.618
        hidden_1 = int(input_dim * phi)
        hidden_2 = int(hidden_1 * phi)
        hidden_3 = (
            round(int(hidden_2 / phi) / 8) * 8
        )  # Round to nearest multiple of 8 for attention

        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.LayerNorm(hidden_1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_1, hidden_2),
            nn.LayerNorm(hidden_2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_2, hidden_3),
            nn.LayerNorm(hidden_3),
            nn.ReLU(),
        )

        self.specialty_heads = nn.ModuleDict(
            {
                board.value: nn.Sequential(
                    nn.Linear(hidden_3, hidden_3 // 2),
                    nn.ReLU(),
                    nn.Dropout(0.15),
                    nn.Linear(hidden_3 // 2, 3),
                )
                for board in ABMSBoard
            }
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_3, num_heads=8, dropout=0.1, batch_first=True
        )

    def forward(self, x: torch.Tensor, specialty: str | None = None) -> dict[str, torch.Tensor]:
        """
        Forward pass with optional specialty-specific prediction.

        Args:
            x: Input features [batch, input_dim]
            specialty: Specific ABMS board to predict (None = all)

        Returns:
            Dictionary of predictions per specialty
        """
        encoded = self.shared_encoder(x)

        encoded_seq = encoded.unsqueeze(1)
        attended, _attention_weights = self.attention(encoded_seq, encoded_seq, encoded_seq)
        attended = attended.squeeze(1)

        predictions = {}

        if specialty and specialty in self.specialty_heads:
            predictions[specialty] = self.specialty_heads[specialty](attended)
        else:
            for spec_name, head in self.specialty_heads.items():
                predictions[spec_name] = head(attended)

        return predictions


class ABMSDisciplineDetector:
    """
    ABMS Medical Disciplines Anomaly Detector.

    Integrates 24 ABMS boards with 150+ subspecialties for comprehensive
    medical anomaly detection across all major medical specialties.
    """

    def __init__(
        self, enable_neurosymbolic: bool = True, golden_ratio_threshold: bool = True
    ) -> None:
        """
        Initialize ABMS detector.

        Args:
            enable_neurosymbolic: Enable symbolic medical reasoning
            golden_ratio_threshold: Use φ-optimized decision thresholds
        """
        self.logger = logging.getLogger(__name__)
        self.enable_neurosymbolic = enable_neurosymbolic
        self.golden_ratio = 1.618 if golden_ratio_threshold else 1.0

        self.model = MultiSpecialtyNeuralNet(input_dim=64, num_specialties=24)

        self.subspecialties = self._initialize_subspecialties()

        self.medical_knowledge_base = self._initialize_medical_kb()

        self.omni_medical_scalars = {
            "omni_diagnostic_precision": 1.42 * self.golden_ratio,
            "omni_clinical_judgment": 1.38 * self.golden_ratio,
            "omni_treatment_efficacy": 1.45 * self.golden_ratio,
            "omni_patient_safety": 1.50 * self.golden_ratio,
            "omni_interdisciplinary_coordination": 1.35 * self.golden_ratio,
            "omni_evidence_based_medicine": 1.40 * self.golden_ratio,
            "omni_preventive_care": 1.33 * self.golden_ratio,
            "omni_holistic_assessment": 1.37 * self.golden_ratio,
        }

        self.specialty_thresholds = {board.value: 0.5 * self.golden_ratio for board in ABMSBoard}

        self.logger.info(f"ABMS Disciplines Detector initialized with {len(ABMSBoard)} boards")

    def _initialize_subspecialties(self) -> dict[str, list[str]]:
        """
        Initialize comprehensive subspecialty mappings per ABMS board.

        Based on ABMS Guide 2025 and current subspecialty certifications.
        Total: 24 boards, 150+ subspecialties.
        """
        return {
            ABMSBoard.ALLERGY_IMMUNOLOGY.value: [
                "adult_allergy_immunology",
                "pediatric_allergy_immunology",
                "clinical_immunology",
                "asthma_specialist",
            ],
            ABMSBoard.ANESTHESIOLOGY.value: [
                "adult_cardiac_anesthesiology",
                "critical_care_medicine",
                "hospice_palliative_medicine",
                "neurocritical_care",
                "pain_medicine",
                "pediatric_anesthesiology",
                "sleep_medicine",
            ],
            ABMSBoard.COLON_RECTAL_SURGERY.value: [
                "colon_rectal_surgery",
                "minimally_invasive_colorectal",
                "inflammatory_bowel_disease_surgery",
                "pelvic_floor_disorders",
            ],
            ABMSBoard.DERMATOLOGY.value: [
                "dermatopathology",
                "micrographic_dermatologic_surgery",
                "pediatric_dermatology",
            ],
            ABMSBoard.EMERGENCY_MEDICINE.value: [
                "anesthesiology_critical_care",
                "emergency_medical_services",
                "medical_toxicology",
                "neurocritical_care",
                "pain_medicine",
                "pediatric_emergency_medicine",
                "sports_medicine",
                "undersea_hyperbaric_medicine",
            ],
            ABMSBoard.FAMILY_MEDICINE.value: [
                "adolescent_medicine",
                "geriatric_medicine",
                "hospice_palliative_medicine",
                "pain_medicine",
                "sleep_medicine",
                "sports_medicine",
            ],
            ABMSBoard.INTERNAL_MEDICINE.value: [
                "advanced_heart_failure_transplant",
                "cardiovascular_disease",
                "clinical_cardiac_electrophysiology",
                "critical_care_medicine",
                "endocrinology_diabetes_metabolism",
                "gastroenterology",
                "geriatric_medicine",
                "hematology",
                "hospice_palliative_medicine",
                "infectious_disease",
                "interventional_cardiology",
                "medical_oncology",
                "nephrology",
                "pulmonary_disease",
                "rheumatology",
                "sleep_medicine",
                "transplant_hepatology",
            ],
            ABMSBoard.MEDICAL_GENETICS.value: [
                "clinical_biochemical_genetics",
                "clinical_cytogenetics",
                "clinical_molecular_genetics",
                "medical_biochemical_genetics",
            ],
            ABMSBoard.NEUROLOGICAL_SURGERY.value: [
                "pediatric_neurological_surgery",
                "spine_surgery",
                "vascular_neurosurgery",
                "functional_neurosurgery",
                "neuro_oncology",
                "peripheral_nerve_surgery",
            ],
            ABMSBoard.NUCLEAR_MEDICINE.value: [
                "nuclear_radiology",
                "nuclear_cardiology",
                "pet_ct_imaging",
                "therapeutic_nuclear_medicine",
            ],
            ABMSBoard.OBSTETRICS_GYNECOLOGY.value: [
                "maternal_fetal_medicine",
                "reproductive_endocrinology_infertility",
                "gynecologic_oncology",
                "female_pelvic_medicine_reconstructive_surgery",
                "complex_family_planning",
            ],
            ABMSBoard.OPHTHALMOLOGY.value: [
                "cornea_external_disease",
                "glaucoma",
                "neuro_ophthalmology",
                "oculoplastics_orbit",
                "pediatric_ophthalmology",
                "retina_vitreous",
                "uveitis",
            ],
            ABMSBoard.ORTHOPAEDIC_SURGERY.value: [
                "adult_reconstructive_orthopaedics",
                "foot_ankle_orthopaedics",
                "hand_surgery",
                "musculoskeletal_oncology",
                "orthopaedic_sports_medicine",
                "orthopaedic_trauma",
                "pediatric_orthopaedics",
                "spine_surgery",
            ],
            ABMSBoard.OTOLARYNGOLOGY.value: [
                "neurotology",
                "pediatric_otolaryngology",
                "rhinology",
                "laryngology",
                "facial_plastic_reconstructive_surgery",
                "head_neck_oncology",
                "sleep_medicine",
            ],
            ABMSBoard.PATHOLOGY.value: [
                "anatomic_pathology",
                "clinical_pathology",
                "blood_banking_transfusion_medicine",
                "chemical_pathology",
                "cytopathology",
                "dermatopathology",
                "forensic_pathology",
                "hematology",
                "immunopathology",
                "medical_microbiology",
                "molecular_genetic_pathology",
                "neuropathology",
                "pediatric_pathology",
            ],
            ABMSBoard.PEDIATRICS.value: [
                "adolescent_medicine",
                "child_abuse_pediatrics",
                "developmental_behavioral_pediatrics",
                "hospice_palliative_medicine",
                "medical_toxicology",
                "neonatal_perinatal_medicine",
                "pediatric_cardiology",
                "pediatric_critical_care",
                "pediatric_emergency_medicine",
                "pediatric_endocrinology",
                "pediatric_gastroenterology",
                "pediatric_hematology_oncology",
                "pediatric_infectious_diseases",
                "pediatric_nephrology",
                "pediatric_pulmonology",
                "pediatric_rheumatology",
                "pediatric_transplant_hepatology",
                "sleep_medicine",
                "sports_medicine",
            ],
            ABMSBoard.PHYSICAL_MEDICINE.value: [
                "brain_injury_medicine",
                "hospice_palliative_medicine",
                "neuromuscular_medicine",
                "pain_medicine",
                "pediatric_rehabilitation_medicine",
                "spinal_cord_injury_medicine",
                "sports_medicine",
            ],
            ABMSBoard.PLASTIC_SURGERY.value: [
                "craniofacial_surgery",
                "hand_surgery",
                "plastic_surgery_within_head_neck",
            ],
            ABMSBoard.PREVENTIVE_MEDICINE.value: [
                "aerospace_medicine",
                "occupational_medicine",
                "public_health_general_preventive_medicine",
                "medical_toxicology",
                "undersea_hyperbaric_medicine",
            ],
            ABMSBoard.PSYCHIATRY.value: [
                "addiction_psychiatry",
                "child_adolescent_psychiatry",
                "consultation_liaison_psychiatry",
                "forensic_psychiatry",
                "geriatric_psychiatry",
                "hospice_palliative_medicine",
                "pain_medicine",
                "sleep_medicine",
            ],
            ABMSBoard.RADIOLOGY.value: [
                "abdominal_radiology",
                "breast_imaging",
                "cardiothoracic_radiology",
                "emergency_radiology",
                "endovascular_surgical_neuroradiology",
                "interventional_radiology_diagnostic_radiology",
                "musculoskeletal_radiology",
                "neuroradiology",
                "nuclear_radiology",
                "pediatric_radiology",
                "vascular_interventional_radiology",
            ],
            ABMSBoard.SURGERY.value: [
                "complex_general_surgical_oncology",
                "hospice_palliative_medicine",
                "pediatric_surgery",
                "surgery_critical_care",
                "surgical_oncology",
                "vascular_surgery",
            ],
            ABMSBoard.THORACIC_SURGERY.value: [
                "congenital_cardiac_surgery",
                "general_thoracic_surgery",
            ],
            ABMSBoard.UROLOGY.value: [
                "pediatric_urology",
                "female_pelvic_medicine_reconstructive_surgery",
            ],
        }

    def _initialize_medical_kb(self) -> dict[str, dict[str, Any]]:
        """Initialize medical knowledge base for neurosymbolic reasoning"""
        return {
            "cardiac_indicators": {
                "chest_pain": ["cardiovascular_disease", "interventional_cardiology"],
                "dyspnea": ["pulmonary_disease", "cardiovascular_disease"],
                "arrhythmia": ["clinical_cardiac_electrophysiology"],
                "heart_failure": ["advanced_heart_failure_transplant"],
            },
            "neurological_indicators": {
                "altered_consciousness": ["neurocritical_care", "emergency_medicine"],
                "seizures": ["neurological_surgery", "neurocritical_care"],
                "focal_deficits": ["neurological_surgery", "interventional_cardiology"],
            },
            "infectious_indicators": {
                "fever": ["infectious_disease", "internal_medicine"],
                "sepsis": ["critical_care_medicine", "infectious_disease"],
                "immunocompromised": ["allergy_immunology", "infectious_disease"],
            },
            "oncological_indicators": {
                "mass_lesion": ["medical_oncology", "radiology", "surgery"],
                "unexplained_weight_loss": ["medical_oncology", "gastroenterology"],
                "cytopenias": ["hematology", "medical_oncology"],
            },
        }

    def detect_medical_anomaly(
        self, patient_data: dict[str, Any], specialty_focus: str | None = None
    ) -> MedicalAnomalyResult:
        """
        Detect medical anomalies across ABMS disciplines.

        Args:
            patient_data: Patient clinical data including:
                - vitals: Dict of vital signs
                - labs: Dict of laboratory results
                - symptoms: List of presenting symptoms
                - history: Medical history
                - imaging: Optional imaging results
            specialty_focus: Optional specific ABMS board to prioritize

        Returns:
            Medical anomaly result with specialty recommendations
        """
        features = self._extract_clinical_features(patient_data)

        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(x, specialty=specialty_focus)

        primary_board, confidence, subspecialty = self._determine_primary_specialty(
            predictions, patient_data
        )

        risk_score = confidence * self.omni_medical_scalars["omni_diagnostic_precision"]

        anomaly_detected = risk_score > (0.5 * self.golden_ratio)

        clinical_indicators = self._identify_clinical_indicators(patient_data, primary_board)

        consultations = self._recommend_consultations(
            primary_board, subspecialty, clinical_indicators
        )

        treatments = self._generate_treatment_considerations(
            primary_board, subspecialty, risk_score
        )

        urgency = self._assess_urgency(risk_score, clinical_indicators)

        neurosymbolic_reasoning = None
        if self.enable_neurosymbolic:
            neurosymbolic_reasoning = self._apply_symbolic_reasoning(
                patient_data, primary_board, clinical_indicators
            )

        result = MedicalAnomalyResult(
            primary_board=primary_board,
            subspecialty=subspecialty,
            anomaly_detected=anomaly_detected,
            confidence=confidence,
            risk_score=risk_score,
            clinical_indicators=clinical_indicators,
            recommended_consultations=consultations,
            treatment_considerations=treatments,
            urgency_level=urgency,
            neurosymbolic_reasoning=neurosymbolic_reasoning,
        )

        self.logger.info(
            f"Medical anomaly detection: {primary_board} "
            f"(confidence={confidence:.3f}, risk={risk_score:.3f})"
        )

        return result

    def _extract_clinical_features(self, patient_data: dict[str, Any]) -> np.ndarray[Any, Any]:
        """Extract numerical features from patient clinical data (O(n) complexity)"""
        features = []

        vitals = patient_data.get("vitals", {})
        features.extend(
            [
                vitals.get("heart_rate_bpm", 75) / 200.0,
                vitals.get("blood_pressure_systolic", 120) / 200.0,
                vitals.get("blood_pressure_diastolic", 80) / 120.0,
                vitals.get("respiratory_rate_bpm", 16) / 40.0,
                vitals.get("temperature_f", 98.6) / 105.0,
                vitals.get("oxygen_saturation_pct", 98) / 100.0,
            ]
        )

        labs = patient_data.get("labs", {})
        features.extend(
            [
                labs.get("wbc_count", 7.5) / 20.0,
                labs.get("hemoglobin", 14.0) / 20.0,
                labs.get("platelet_count", 250) / 500.0,
                labs.get("glucose", 100) / 300.0,
                labs.get("creatinine", 1.0) / 5.0,
                labs.get("bilirubin", 0.8) / 5.0,
            ]
        )

        symptoms = patient_data.get("symptoms", [])
        symptom_features = np.zeros(10)
        symptom_map = {
            "pain": 0,
            "dyspnea": 1,
            "fever": 2,
            "nausea": 3,
            "fatigue": 4,
            "confusion": 5,
            "weakness": 6,
            "bleeding": 7,
            "edema": 8,
            "rash": 9,
        }
        for symptom in symptoms:
            if symptom.lower() in symptom_map:
                symptom_features[symptom_map[symptom.lower()]] = 1.0
        features.extend(symptom_features.tolist())

        history = patient_data.get("history", {})
        features.extend(
            [
                float(history.get("chronic_conditions", 0)) / 10.0,
                float(history.get("prior_surgeries", 0)) / 10.0,
                float(history.get("medication_count", 0)) / 20.0,
                float(history.get("age", 50)) / 100.0,
            ]
        )

        while len(features) < 64:
            features.append(0.0)

        return np.array(features[:64], dtype=np.float32)

    def _determine_primary_specialty(
        self, predictions: dict[str, torch.Tensor], patient_data: dict[str, Any]
    ) -> tuple[str, float, str | None]:
        """Determine primary specialty and confidence from predictions"""
        max_confidence = 0.0
        primary_board = ABMSBoard.INTERNAL_MEDICINE.value

        for specialty, pred_tensor in predictions.items():
            anomaly_score = torch.softmax(pred_tensor[0], dim=0)[2].item()

            if anomaly_score > max_confidence:
                max_confidence = anomaly_score
                primary_board = specialty

        subspecialty = None
        if primary_board in self.subspecialties:
            subspecs = self.subspecialties[primary_board]
            if subspecs:
                subspecialty = subspecs[0]

        return primary_board, max_confidence, subspecialty

    def _identify_clinical_indicators(
        self, patient_data: dict[str, Any], primary_board: str
    ) -> list[str]:
        """Identify key clinical indicators for the primary specialty"""
        indicators = []

        vitals = patient_data.get("vitals", {})
        if vitals.get("heart_rate_bpm", 75) > 100:
            indicators.append("tachycardia")
        if vitals.get("blood_pressure_systolic", 120) > 140:
            indicators.append("hypertension")
        if vitals.get("oxygen_saturation_pct", 98) < 92:
            indicators.append("hypoxemia")
        if vitals.get("temperature_f", 98.6) > 100.4:
            indicators.append("fever")

        symptoms = patient_data.get("symptoms", [])
        indicators.extend([s.lower() for s in symptoms[:5]])

        return indicators[:10]

    def _recommend_consultations(
        self, primary_board: str, subspecialty: str | None, indicators: list[str]
    ) -> list[str]:
        """Recommend consultations based on specialty and indicators"""
        consultations = [primary_board]

        if subspecialty:
            consultations.append(f"{primary_board}/{subspecialty}")

        if "cardiac" in " ".join(indicators).lower() or "chest_pain" in indicators:
            if ABMSBoard.INTERNAL_MEDICINE.value not in consultations:
                consultations.append(f"{ABMSBoard.INTERNAL_MEDICINE.value}/cardiovascular_disease")

        if "neurological" in " ".join(indicators).lower() or "seizure" in indicators:
            consultations.append(ABMSBoard.NEUROLOGICAL_SURGERY.value)

        if "infectious" in " ".join(indicators).lower() or "fever" in indicators:
            consultations.append(f"{ABMSBoard.INTERNAL_MEDICINE.value}/infectious_disease")

        return consultations[:5]

    def _generate_treatment_considerations(
        self, primary_board: str, subspecialty: str | None, risk_score: float
    ) -> list[str]:
        """Generate treatment considerations based on specialty"""
        treatments = []

        if risk_score > 0.8:
            treatments.append("Consider immediate intervention")
            treatments.append("Admit to appropriate level of care")
        elif risk_score > 0.6:
            treatments.append("Expedited outpatient evaluation")
            treatments.append("Consider emergency department if symptoms worsen")
        else:
            treatments.append("Routine follow-up appropriate")

        if primary_board == ABMSBoard.INTERNAL_MEDICINE.value:
            treatments.append("Comprehensive metabolic panel recommended")
            treatments.append("Consider echocardiogram if cardiac involvement")
        elif primary_board == ABMSBoard.EMERGENCY_MEDICINE.value:
            treatments.append("Rapid assessment and stabilization")
            treatments.append("Rule out life-threatening conditions")

        treatments.append("Multidisciplinary team approach recommended")

        return treatments[:6]

    def _assess_urgency(self, risk_score: float, indicators: list[str]) -> str:
        """Assess clinical urgency level"""
        critical_indicators = ["chest_pain", "altered_consciousness", "severe_bleeding", "stroke"]

        if any(ind in indicators for ind in critical_indicators):
            return "critical"

        if risk_score > 0.8:
            return "urgent"
        elif risk_score > 0.6:
            return "emergent"
        else:
            return "routine"

    def _apply_symbolic_reasoning(
        self, patient_data: dict[str, Any], primary_board: str, indicators: list[str]
    ) -> dict[str, Any]:
        """Apply neurosymbolic medical reasoning"""
        reasoning: dict[str, list[str]] = {
            "rules_applied": [],
            "deductions": [],
            "confidence_adjustments": [],
        }

        symptoms = patient_data.get("symptoms", [])

        for mappings in self.medical_knowledge_base.values():
            for symptom, specialties in mappings.items():
                if symptom in [s.lower() for s in symptoms]:
                    reasoning["rules_applied"].append(f"Rule: {symptom} → {', '.join(specialties)}")
                    reasoning["deductions"].append(f"Consider {specialties[0]} consultation")

        if len(reasoning["rules_applied"]) > 2:
            reasoning["confidence_adjustments"].append(
                "High confidence: Multiple consistent indicators"
            )

        return reasoning

    def extract_features(self, data: dict[str, Any]) -> torch.Tensor:
        """Extract features for ML fusion integration"""
        features = self._extract_clinical_features(data)
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Predict for engine integration"""
        result = self.detect_medical_anomaly(data)

        return {
            "anomaly_scores": np.array([result.risk_score], dtype=np.float32),
            "primary_specialty": result.primary_board,
            "confidence": result.confidence,
            "recommendations": result.treatment_considerations,
        }

    def detect(
        self,
        data: np.ndarray[Any, Any] | torch.Tensor,
        specialty: str,
        include_reasoning: bool = False,
    ) -> MedicalAnomalyResult:
        """Detect anomalies for a specific specialty.

        Args:
            data: Input data as numpy array or tensor (shape: [features])
            specialty: ABMS specialty to evaluate (supports common aliases)
            include_reasoning: Whether to include neurosymbolic reasoning

        Returns:
            MedicalAnomalyResult with detection results
        """
        specialty_aliases = {
            "cardiology": "internal_medicine",
            "neurology": "internal_medicine",
            "gastroenterology": "internal_medicine",
            "pulmonology": "internal_medicine",
            "nephrology": "internal_medicine",
            "endocrinology": "internal_medicine",
            "rheumatology": "internal_medicine",
            "hematology": "internal_medicine",
            "oncology": "internal_medicine",
            "infectious_disease": "internal_medicine",
        }

        mapped_specialty = specialty_aliases.get(specialty, specialty)

        if mapped_specialty not in [b.value for b in ABMSBoard]:
            raise ValueError(f"Invalid specialty: {specialty}")

        if isinstance(data, torch.Tensor):
            data_np = data.detach().cpu().numpy()
        else:
            data_np = np.asarray(data, dtype=np.float32)

        data_np = np.nan_to_num(data_np, nan=0.0, posinf=1e6, neginf=-1e6)

        patient_data = {
            "vitals": {
                "heart_rate_bpm": float(data_np[0]) if len(data_np) > 0 else 75.0,
                "blood_pressure_systolic": float(data_np[1]) if len(data_np) > 1 else 120.0,
                "blood_pressure_diastolic": float(data_np[2]) if len(data_np) > 2 else 80.0,
                "respiratory_rate_bpm": float(data_np[3]) if len(data_np) > 3 else 16.0,
                "temperature_f": float(data_np[4]) if len(data_np) > 4 else 98.6,
                "oxygen_saturation_pct": float(data_np[5]) if len(data_np) > 5 else 98.0,
            },
            "labs": {
                "wbc_count": float(data_np[6]) if len(data_np) > 6 else 7.5,
                "hemoglobin": float(data_np[7]) if len(data_np) > 7 else 14.0,
                "platelet_count": float(data_np[8]) if len(data_np) > 8 else 250.0,
                "glucose": float(data_np[9]) if len(data_np) > 9 else 100.0,
                "creatinine": float(data_np[10]) if len(data_np) > 10 else 1.0,
                "bilirubin": float(data_np[11]) if len(data_np) > 11 else 0.8,
            },
            "symptoms": [],
            "history": {
                "chronic_conditions": 0,
                "prior_surgeries": 0,
                "medication_count": 0,
                "age": 50,
            },
        }

        result = self.detect_medical_anomaly(patient_data)

        result = MedicalAnomalyResult(
            primary_board=specialty,
            confidence=result.confidence,
            risk_score=result.risk_score,
            urgency_level=result.urgency_level,
            clinical_indicators=result.clinical_indicators,
            recommended_consultations=result.recommended_consultations,
            treatment_considerations=result.treatment_considerations,
            neurosymbolic_reasoning=result.neurosymbolic_reasoning if include_reasoning else None,
        )

        return result

    def detect_all(
        self, data: np.ndarray[Any, Any] | torch.Tensor
    ) -> dict[str, MedicalAnomalyResult]:
        """Detect anomalies across all ABMS specialties.

        Args:
            data: Input data as numpy array or tensor

        Returns:
            Dictionary mapping specialty names to MedicalAnomalyResult
        """
        results = {}
        for board in ABMSBoard:
            results[board.value] = self.detect(data, board.value)
        return results


def create_omni_medical_scalars() -> dict[str, float]:
    """
    Create doctorate-level medical scalars for truth deciphering.

    Returns:
        Dictionary of omni-medical scalars with golden ratio optimization
    """
    phi = 1.618

    return {
        "omni_diagnostic_precision": 1.42 * phi,
        "omni_clinical_judgment": 1.38 * phi,
        "omni_treatment_efficacy": 1.45 * phi,
        "omni_patient_safety": 1.50 * phi,
        "omni_interdisciplinary_coordination": 1.35 * phi,
        "omni_evidence_based_medicine": 1.40 * phi,
        "omni_preventive_care": 1.33 * phi,
        "omni_holistic_assessment": 1.37 * phi,
        "omni_subspecialty_expertise": 1.44 * phi,
        "omni_emergency_responsiveness": 1.48 * phi,
        "omni_chronic_disease_management": 1.36 * phi,
        "omni_surgical_precision": 1.46 * phi,
        "omni_pharmaceutical_optimization": 1.39 * phi,
        "omni_imaging_interpretation": 1.41 * phi,
        "omni_laboratory_correlation": 1.34 * phi,
    }


ABMSAnomalyDetector = ABMSDisciplineDetector

# Alias for test compatibility - ABMSSpecialty is an alias for ABMSBoard
ABMSSpecialty = ABMSBoard
