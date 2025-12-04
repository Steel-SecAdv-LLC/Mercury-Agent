"""
OMNI ♱ AVA (O♱A)
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

"""
Neurocritical Care Module - Advanced Neurological Emergency Detection

Specialized neurocritical care for humanitarian healthcare:
- Stroke detection & classification (ischemic/hemorrhagic)
- Intracranial pressure (ICP) monitoring & prediction
- Seizure detection & prediction
- Traumatic brain injury (TBI) severity assessment
- Neurological deterioration early warning

⚠️ SIMULATION-BASED: Uses simulated neurological data. Clinical validation required.
Consult neurologists/neurosurgeons before acting on predictions.

Research sources:
- NIH Stroke Scale (NIHSS) methodologies
- Brain Trauma Foundation ICP guidelines
- International League Against Epilepsy seizure classifications
- Glasgow Coma Scale standards

"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
from torch import nn


class StrokeType(Enum):
    """Stroke classifications"""

    NO_STROKE = "no_stroke"
    ISCHEMIC = "ischemic_stroke"
    HEMORRHAGIC = "hemorrhagic_stroke"
    TIA = "transient_ischemic_attack"
    CRYPTOGENIC = "cryptogenic_stroke"


class SeizureType(Enum):
    """Seizure classifications per ILAE"""

    NO_SEIZURE = "no_seizure"
    FOCAL_AWARE = "focal_aware"
    FOCAL_IMPAIRED = "focal_impaired_awareness"
    GENERALIZED_TONIC_CLONIC = "generalized_tonic_clonic"
    ABSENCE = "absence"
    MYOCLONIC = "myoclonic"
    STATUS_EPILEPTICUS = "status_epilepticus"


@dataclass
class NeurocriticalPredictionResult:
    """Neurocritical care prediction results"""

    # Required fields (no defaults) - must come first
    neurological_emergency_detected: bool
    confidence: float
    emergency_type: str
    risk_score: float

    stroke_detected: bool
    stroke_type: str

    seizure_detected: bool
    seizure_type: str

    # Optional fields (with defaults) - must come after required fields
    nihss_score: int | None = None
    seizure_risk_score: float = 0.0

    icp_elevated: bool = False
    icp_mmhg: float | None = None

    tbi_severity: str | None = None
    gcs_score: int | None = None

    clinical_recommendations: list[str] = field(default_factory=list)
    time_sensitive_interventions: list[str] = field(default_factory=list)


class StrokeDetector(nn.Module):
    """
    Neural network for stroke detection and classification.

    Uses multimodal inputs: vital signs, neurological exam, imaging features.
    """

    def __init__(self, input_dim: int = 64):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.stroke_classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 5)
        )

        self.severity_predictor = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for stroke detection.

        Args:
            features: Patient features (vitals, exam, imaging)

        Returns:
            Tuple of (stroke_classification, severity_score)
        """
        extracted = self.feature_extractor(features)
        classification = self.stroke_classifier(extracted)
        severity = self.severity_predictor(extracted)

        return classification, severity


class SeizurePredictor(nn.Module):
    """
    LSTM-based seizure detection and prediction.

    Analyzes EEG-like patterns and clinical features for seizure risk.
    """

    def __init__(self, input_dim: int = 32, hidden_dim: int = 64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )

        self.attention = nn.Sequential(nn.Linear(hidden_dim * 2, 64), nn.Tanh(), nn.Linear(64, 1))

        self.seizure_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, len(SeizureType)),
        )

        self.risk_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for seizure prediction.

        Args:
            sequence: Temporal neurological features

        Returns:
            Tuple of (seizure_classification, risk_score, attention_weights)
        """
        lstm_out, _ = self.lstm(sequence)

        attention_scores = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_scores, dim=1)

        context = torch.sum(lstm_out * attention_weights, dim=1)

        classification = self.seizure_classifier(context)
        risk = self.risk_predictor(context)

        return classification, risk, attention_weights.squeeze(-1)


class ICPMonitor:
    """
    Intracranial Pressure (ICP) monitoring and prediction.

    Monitors for elevated ICP and predicts herniation risk.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.normal_icp_range = (5.0, 15.0)
        self.elevated_threshold = 20.0
        self.critical_threshold = 25.0

    def assess_icp(self, patient_data: dict[str, Any]) -> dict[str, Any]:
        """
        Assess ICP status and risk.

        Args:
            patient_data: Patient vitals and neurological exam

        Returns:
            ICP assessment with recommendations
        """
        measured_icp = patient_data.get("icp_mmhg")
        cpp = patient_data.get("cerebral_perfusion_pressure")

        if measured_icp is None:
            estimated_icp = self._estimate_icp_from_clinicals(patient_data)
            measured_icp = estimated_icp

        icp_elevated = measured_icp > self.elevated_threshold
        icp_critical = measured_icp > self.critical_threshold

        if cpp is None:
            map_val = patient_data.get("mean_arterial_pressure", 90)
            cpp = map_val - measured_icp

        recommendations = self._generate_icp_recommendations(
            measured_icp, cpp, icp_elevated, icp_critical
        )

        return {
            "icp_mmhg": measured_icp,
            "icp_elevated": icp_elevated,
            "icp_critical": icp_critical,
            "cerebral_perfusion_pressure": cpp,
            "herniation_risk": float(measured_icp > 25) * 0.8,
            "recommendations": recommendations,
        }

    def _estimate_icp_from_clinicals(self, patient_data: dict[str, Any]) -> float:
        """Estimate ICP from clinical signs when direct measurement unavailable"""
        baseline_icp = 10.0

        gcs = patient_data.get("gcs_score", 15)
        if gcs < 8:
            baseline_icp += 10.0
        elif gcs < 13:
            baseline_icp += 5.0

        pupil_abnormal = patient_data.get("pupil_abnormality", False)
        if pupil_abnormal:
            baseline_icp += 8.0

        posturing = patient_data.get("motor_posturing", False)
        if posturing:
            baseline_icp += 7.0

        return min(baseline_icp, 40.0)

    def _generate_icp_recommendations(
        self, icp: float, cpp: float, elevated: bool, critical: bool
    ) -> list[str]:
        """Generate ICP management recommendations"""
        recs = []

        if critical:
            recs.append("CRITICAL ICP ELEVATION - IMMEDIATE INTERVENTION REQUIRED")
            recs.append("Elevate head of bed 30 degrees")
            recs.append("Hyperosmolar therapy (mannitol or hypertonic saline)")
            recs.append("Consider EVD placement for CSF drainage")
            recs.append("Neurosurgical consultation for decompressive craniectomy")
        elif elevated:
            recs.append("Elevated ICP detected")
            recs.append("Optimize head position and ventilator settings")
            recs.append("Consider moderate hyperventilation (PCO2 30-35)")
            recs.append("Sedation and analgesia optimization")

        if cpp < 60:
            recs.append(f"Low CPP ({cpp:.1f} mmHg) - risk of cerebral ischemia")
            recs.append("Consider vasopressors to maintain CPP >60 mmHg")

        return recs


class NIHSSCalculator:
    """
    NIH Stroke Scale (NIHSS) calculator.

    Standardized neurological deficit assessment for stroke severity.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_nihss(self, exam_findings: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate NIHSS score from neurological exam.

        Args:
            exam_findings: Neurological exam components

        Returns:
            NIHSS score and severity interpretation
        """
        score = 0

        score += exam_findings.get("loc", 0)
        score += exam_findings.get("loc_questions", 0)
        score += exam_findings.get("loc_commands", 0)
        score += exam_findings.get("gaze", 0)
        score += exam_findings.get("visual_fields", 0)
        score += exam_findings.get("facial_palsy", 0)
        score += exam_findings.get("motor_arm_left", 0)
        score += exam_findings.get("motor_arm_right", 0)
        score += exam_findings.get("motor_leg_left", 0)
        score += exam_findings.get("motor_leg_right", 0)
        score += exam_findings.get("limb_ataxia", 0)
        score += exam_findings.get("sensory", 0)
        score += exam_findings.get("language", 0)
        score += exam_findings.get("dysarthria", 0)
        score += exam_findings.get("extinction_inattention", 0)

        severity = self._interpret_nihss(score)

        return {
            "nihss_score": score,
            "severity": severity["category"],
            "stroke_risk": severity["stroke_risk"],
            "recommendations": severity["recommendations"],
        }

    def _interpret_nihss(self, score: int) -> dict[str, Any]:
        """Interpret NIHSS score"""
        if score == 0:
            return {
                "category": "No stroke symptoms",
                "stroke_risk": 0.0,
                "recommendations": ["No acute stroke treatment needed"],
            }
        elif score <= 4:
            return {
                "category": "Minor stroke",
                "stroke_risk": 0.3,
                "recommendations": [
                    "Consider tPA if within window",
                    "Admission for observation",
                    "Secondary prevention initiation",
                ],
            }
        elif score <= 15:
            return {
                "category": "Moderate stroke",
                "stroke_risk": 0.6,
                "recommendations": [
                    "Immediate tPA administration if eligible",
                    "Consider thrombectomy for large vessel occlusion",
                    "Neurology consultation",
                    "ICU-level monitoring",
                ],
            }
        elif score <= 20:
            return {
                "category": "Moderate-severe stroke",
                "stroke_risk": 0.8,
                "recommendations": [
                    "URGENT: tPA and thrombectomy evaluation",
                    "Comprehensive stroke center transfer",
                    "Neurointensive care",
                    "Early rehabilitation planning",
                ],
            }
        else:
            return {
                "category": "Severe stroke",
                "stroke_risk": 1.0,
                "recommendations": [
                    "CRITICAL: Immediate intervention required",
                    "Mechanical thrombectomy strongly considered",
                    "Neurocritical care unit admission",
                    "Family discussion regarding prognosis",
                ],
            }


class NeurocriticalCarePredictor:
    """
    Comprehensive neurocritical care prediction system integrating stroke,
    seizure, ICP monitoring, and TBI assessment.
    """

    def __init__(
        self,
        enable_stroke: bool = True,
        enable_seizure: bool = True,
        enable_icp: bool = True,
    ):
        self.enable_stroke = enable_stroke
        self.enable_seizure = enable_seizure
        self.enable_icp = enable_icp

        self.stroke_detector = StrokeDetector() if enable_stroke else None
        self.seizure_predictor = SeizurePredictor() if enable_seizure else None
        self.icp_monitor = ICPMonitor() if enable_icp else None
        self.nihss_calculator = NIHSSCalculator()

        self.logger = logging.getLogger(__name__)

    def predict_neurocritical_emergency(
        self, patient_data: dict[str, Any]
    ) -> NeurocriticalPredictionResult:
        """
        Comprehensive neurocritical care prediction.

        Args:
            patient_data: Patient data including:
                - clinical_features: Neurological exam, vitals
                - temporal_sequence: Time-series neurological data
                - exam_findings: NIHSS components
                - icp_data: ICP monitoring data

        Returns:
            Neurocritical prediction with emergency classification
        """
        result = NeurocriticalPredictionResult(
            neurological_emergency_detected=False,
            confidence=0.0,
            emergency_type="none",
            risk_score=0.0,
            stroke_detected=False,
            stroke_type="no_stroke",
            seizure_detected=False,
            seizure_type="no_seizure",
        )

        if self.enable_stroke and "clinical_features" in patient_data:
            stroke_result = self._detect_stroke(patient_data["clinical_features"])
            result.stroke_detected = stroke_result["stroke_detected"]
            result.stroke_type = stroke_result["stroke_type"]
            result.confidence = max(result.confidence, stroke_result["confidence"])
            result.clinical_recommendations.extend(stroke_result["recommendations"])

            if stroke_result["stroke_detected"]:
                result.neurological_emergency_detected = True
                result.emergency_type = "stroke"

        if "exam_findings" in patient_data:
            nihss = self.nihss_calculator.calculate_nihss(patient_data["exam_findings"])
            result.nihss_score = nihss["nihss_score"]
            result.risk_score = max(result.risk_score, nihss["stroke_risk"])
            result.clinical_recommendations.extend(nihss["recommendations"])

        if self.enable_seizure and "temporal_sequence" in patient_data:
            seizure_result = self._predict_seizure(patient_data["temporal_sequence"])
            result.seizure_detected = seizure_result["seizure_detected"]
            result.seizure_type = seizure_result["seizure_type"]
            result.seizure_risk_score = seizure_result["risk_score"]
            result.clinical_recommendations.extend(seizure_result["recommendations"])

            if seizure_result["seizure_type"] == "status_epilepticus":
                result.neurological_emergency_detected = True
                result.emergency_type = "status_epilepticus"
                result.time_sensitive_interventions.append(
                    "URGENT: Status epilepticus - benzodiazepines immediately"
                )

        if self.enable_icp and "icp_data" in patient_data:
            icp_result = self.icp_monitor.assess_icp(patient_data["icp_data"])
            result.icp_elevated = icp_result["icp_elevated"]
            result.icp_mmhg = icp_result["icp_mmhg"]
            result.clinical_recommendations.extend(icp_result["recommendations"])

            if icp_result["icp_critical"]:
                result.neurological_emergency_detected = True
                result.emergency_type = "elevated_icp"

        if "tbi_features" in patient_data:
            tbi_result = self._assess_tbi(patient_data["tbi_features"])
            result.tbi_severity = tbi_result["severity"]
            result.gcs_score = tbi_result["gcs_score"]
            result.clinical_recommendations.extend(tbi_result["recommendations"])

        result.risk_score = max(
            result.risk_score, result.seizure_risk_score, float(result.icp_elevated) * 0.7
        )
        result.confidence = max(result.confidence, result.risk_score)

        self.logger.info(
            f"Neurocritical prediction: {result.emergency_type}, "
            f"stroke={result.stroke_detected}, seizure={result.seizure_detected}"
        )

        return result

    def _detect_stroke(self, features: np.ndarray) -> dict[str, Any]:
        """Detect and classify stroke"""
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        self.stroke_detector.eval()
        with torch.no_grad():
            classification, severity = self.stroke_detector(features_tensor)

        probs = torch.softmax(classification[0], dim=0)
        stroke_idx = torch.argmax(probs).item()
        confidence = float(probs[stroke_idx].item())

        stroke_types = [e.value for e in StrokeType]
        detected_type = stroke_types[stroke_idx]

        stroke_detected = detected_type != "no_stroke"

        recs = []
        if detected_type == "ischemic_stroke":
            recs.append("Ischemic stroke suspected - tPA window assessment")
            recs.append("CT angiography for large vessel occlusion")
        elif detected_type == "hemorrhagic_stroke":
            recs.append("Hemorrhagic stroke suspected - neurosurgical consult")
            recs.append("Blood pressure control critical")

        return {
            "stroke_detected": stroke_detected,
            "stroke_type": detected_type,
            "confidence": confidence,
            "recommendations": recs,
        }

    def _predict_seizure(self, sequence: np.ndarray) -> dict[str, Any]:
        """Predict seizure occurrence and type"""
        seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

        self.seizure_predictor.eval()
        with torch.no_grad():
            classification, risk, attention = self.seizure_predictor(seq_tensor)

        probs = torch.softmax(classification[0], dim=0)
        seizure_idx = torch.argmax(probs).item()
        risk_score = float(risk[0].item())

        seizure_types = [e.value for e in SeizureType]
        detected_type = seizure_types[seizure_idx]

        seizure_detected = detected_type != "no_seizure"

        recs = []
        if detected_type == "status_epilepticus":
            recs.append("STATUS EPILEPTICUS - IMMEDIATE BENZODIAZEPINES")
            recs.append("Prepare for intubation if refractory")
        elif seizure_detected:
            recs.append(f"Seizure detected: {detected_type}")
            recs.append("Antiepileptic medication consideration")

        return {
            "seizure_detected": seizure_detected,
            "seizure_type": detected_type,
            "risk_score": risk_score,
            "recommendations": recs,
        }

    def _assess_tbi(self, tbi_features: dict[str, Any]) -> dict[str, Any]:
        """Assess traumatic brain injury severity"""
        gcs = tbi_features.get("gcs_score", 15)

        if gcs >= 13:
            severity = "mild"
            recs = ["Observation protocol", "CT head if indicated"]
        elif gcs >= 9:
            severity = "moderate"
            recs = ["ICU admission", "Repeat CT in 6-12 hours", "Neurosurgical consult"]
        else:
            severity = "severe"
            recs = [
                "SEVERE TBI - Neurointensive care",
                "ICP monitoring",
                "Immediate neurosurgical evaluation",
            ]

        return {"severity": severity, "gcs_score": gcs, "recommendations": recs}
