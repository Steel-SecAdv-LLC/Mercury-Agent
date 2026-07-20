# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Neurocritical Care Module - Advanced Neurological Emergency Detection.

Specialized neurocritical care for humanitarian healthcare:
- Stroke detection & classification (ischemic/hemorrhagic)
- Intracranial pressure (ICP) monitoring & prediction
- Seizure detection & prediction
- Traumatic brain injury (TBI) severity assessment
- Neurological deterioration early warning

Safety semantics (fail-closed): NIHSS and GCS/TBI are deterministic instruments.
A missing NIHSS exam item is reported unassessed (never scored 0), so a partial
exam cannot under-score toward "no stroke"; the total is a lower bound. ICP is
**never fabricated** — when no ICP is measured, ``icp_mmhg`` is ``None`` and only
a clearly-labelled qualitative clinical-sign concern is reported, not an invented
pressure. The stroke and seizure networks are gated behind ``is_fitted`` and
refuse to emit a number on random weights. Every result carries a decision-support
disclaimer, provenance, and red-flag emergency routing.

Research sources:
- NIH Stroke Scale (NIHSS) methodologies
- Brain Trauma Foundation ICP guidelines
- International League Against Epilepsy seizure classifications
- Glasgow Coma Scale standards
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from omni_mercury_engine.medical.safety import (
    ClinicalSafetyEnvelope,
    build_provenance,
)

if TYPE_CHECKING:
    import numpy as np


class StrokeType(Enum):
    """Stroke classifications."""

    NO_STROKE = "no_stroke"
    ISCHEMIC = "ischemic_stroke"
    HEMORRHAGIC = "hemorrhagic_stroke"
    TIA = "transient_ischemic_attack"
    CRYPTOGENIC = "cryptogenic_stroke"


class SeizureType(Enum):
    """Seizure classifications per ILAE."""

    NO_SEIZURE = "no_seizure"
    FOCAL_AWARE = "focal_aware"
    FOCAL_IMPAIRED = "focal_impaired_awareness"
    GENERALIZED_TONIC_CLONIC = "generalized_tonic_clonic"
    ABSENCE = "absence"
    MYOCLONIC = "myoclonic"
    STATUS_EPILEPTICUS = "status_epilepticus"


@dataclass
class NeurocriticalPredictionResult:
    """Neurocritical care prediction results."""

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

    # Safety / honesty layer.
    ml_stroke_available: bool = False
    ml_seizure_available: bool = False
    icp_measured: bool = False
    nihss_is_lower_bound: bool = False
    safety: ClinicalSafetyEnvelope = field(default_factory=ClinicalSafetyEnvelope)


class StrokeDetector(nn.Module):
    """Neural network for stroke detection and classification.

    Uses multimodal inputs: vital signs, neurological exam, imaging features.
    """

    def __init__(self, input_dim: int = 64) -> None:
        """Initialize the instance."""
        super().__init__()

        # Fail-closed: random weights until real trained weights are loaded.
        self.is_fitted: bool = False

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
        """Forward pass for stroke detection.

        Args:
            features: Patient features (vitals, exam, imaging)

        Returns:
            Tuple of (stroke_classification, severity_score)
        """
        extracted = self.feature_extractor(features)
        classification = self.stroke_classifier(extracted)
        severity = self.severity_predictor(extracted)

        return classification, severity

    def load_trained_weights(self, state_dict: dict[str, Any]) -> None:
        """Load trained weights and mark the model fitted (safe to surface)."""
        self.load_state_dict(state_dict)
        self.is_fitted = True


class SeizurePredictor(nn.Module):
    """LSTM-based seizure detection and prediction.

    Analyzes EEG-like patterns and clinical features for seizure risk.
    """

    def __init__(self, input_dim: int = 32, hidden_dim: int = 64) -> None:
        """Initialize the instance."""
        super().__init__()

        # Fail-closed: random weights until real trained weights are loaded.
        self.is_fitted: bool = False

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
        """Forward pass for seizure prediction.

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

    def load_trained_weights(self, state_dict: dict[str, Any]) -> None:
        """Load trained weights and mark the model fitted (safe to surface)."""
        self.load_state_dict(state_dict)
        self.is_fitted = True


class ICPMonitor:
    """Intracranial Pressure (ICP) monitoring and prediction.

    Monitors for elevated ICP and predicts herniation risk.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)
        self.normal_icp_range = (5.0, 15.0)
        self.elevated_threshold = 20.0
        self.critical_threshold = 25.0

    def assess_icp(self, patient_data: dict[str, Any]) -> dict[str, Any]:
        """Assess ICP status and risk.

        Args:
            patient_data: Patient vitals and neurological exam

        Returns:
            ICP assessment with recommendations
        """
        measured_icp = patient_data.get("icp_mmhg")
        cpp = patient_data.get("cerebral_perfusion_pressure")
        icp_measured = measured_icp is not None

        clinical_concern: str | None = None

        if measured_icp is not None:
            icp_elevated = measured_icp > self.elevated_threshold
            icp_critical = measured_icp > self.critical_threshold
            herniation_risk = float(measured_icp > 25) * 0.8
            if cpp is None:
                map_val = patient_data.get("mean_arterial_pressure")
                cpp = (map_val - measured_icp) if map_val is not None else None
        else:
            # No measured ICP: DO NOT fabricate a pressure. Report a clearly
            # labelled qualitative concern from clinical signs; the numeric ICP
            # and any pressure-derived risk abstain.
            icp_elevated = False
            icp_critical = False
            herniation_risk = 0.0
            clinical_concern = self._clinical_icp_concern(patient_data)

        recommendations = self._generate_icp_recommendations(
            cpp, icp_elevated, icp_critical, icp_measured, clinical_concern
        )

        return {
            "icp_mmhg": measured_icp,  # None when not measured — never invented
            "icp_measured": icp_measured,
            "icp_elevated": icp_elevated,
            "icp_critical": icp_critical,
            "cerebral_perfusion_pressure": cpp,
            "herniation_risk": herniation_risk,
            "clinical_concern": clinical_concern,
            "recommendations": recommendations,
        }

    def _clinical_icp_concern(self, patient_data: dict[str, Any]) -> str:
        """Qualitative ICP concern from clinical signs (NOT a measured pressure).

        Returns a coarse concern band derived from GCS, pupillary abnormality,
        and motor posturing. This is a clinical-sign flag prompting measurement,
        explicitly not a substitute for a monitored ICP value.
        """
        concern = 0
        gcs = patient_data.get("gcs_score")
        if gcs is not None and gcs < 8:
            concern += 2
        elif gcs is not None and gcs < 13:
            concern += 1
        if patient_data.get("pupil_abnormality"):
            concern += 2
        if patient_data.get("motor_posturing"):
            concern += 2

        if concern >= 3:
            return "high_clinical_concern"
        if concern >= 1:
            return "some_clinical_concern"
        return "no_clinical_signs"

    def _generate_icp_recommendations(
        self,
        cpp: float | None,
        elevated: bool,
        critical: bool,
        icp_measured: bool,
        clinical_concern: str | None,
    ) -> list[str]:
        """Generate ICP management recommendations."""
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

        if not icp_measured:
            if clinical_concern == "high_clinical_concern":
                recs.append(
                    "No measured ICP; clinical signs raise concern for raised ICP — "
                    "urgent neurosurgical evaluation and ICP monitoring."
                )
            elif clinical_concern == "some_clinical_concern":
                recs.append(
                    "No measured ICP; some clinical signs present — consider ICP "
                    "monitoring and neurological reassessment."
                )
            else:
                recs.append("No measured ICP available; monitor and reassess.")

        if cpp is not None and cpp < 60:
            recs.append(f"Low CPP ({cpp:.1f} mmHg) - risk of cerebral ischemia")
            recs.append("Consider vasopressors to maintain CPP >60 mmHg")

        return recs


class NIHSSCalculator:
    """NIH Stroke Scale (NIHSS) calculator.

    Standardized neurological deficit assessment for stroke severity.
    """

    #: The 15 NIHSS exam items, in order.
    ITEMS = (
        "loc",
        "loc_questions",
        "loc_commands",
        "gaze",
        "visual_fields",
        "facial_palsy",
        "motor_arm_left",
        "motor_arm_right",
        "motor_leg_left",
        "motor_leg_right",
        "limb_ataxia",
        "sensory",
        "language",
        "dysarthria",
        "extinction_inattention",
    )

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)

    def calculate_nihss(self, exam_findings: dict[str, Any]) -> dict[str, Any]:
        """Calculate NIHSS score from neurological exam.

        Args:
            exam_findings: Neurological exam components

        Returns:
            NIHSS score and severity interpretation. A missing exam item is
            reported unassessed (not scored 0), so a partial exam cannot
            under-score toward "no stroke"; the total is a lower bound.
        """
        score = 0
        assessed: list[str] = []
        unassessed: list[str] = []
        for item in self.ITEMS:
            value = exam_findings.get(item)
            if value is None:
                unassessed.append(item)
            else:
                score += value
                assessed.append(item)

        is_lower_bound = bool(unassessed)
        severity = self._interpret_nihss(score, partial=is_lower_bound)

        return {
            "nihss_score": score,
            "nihss_is_lower_bound": is_lower_bound,
            "assessed_items": assessed,
            "unassessed_items": unassessed,
            "severity": severity["category"],
            "stroke_risk": severity["stroke_risk"],
            "recommendations": severity["recommendations"],
        }

    def _interpret_nihss(self, score: int, *, partial: bool = False) -> dict[str, Any]:
        """Interpret NIHSS score.

        When the exam is partial and the assessed items sum to 0, the score is
        not reported as "no stroke symptoms" — an incomplete exam cannot
        establish a normal neurological status.
        """
        if score == 0 and partial:
            return {
                "category": "Incomplete exam — not scorable as normal",
                "stroke_risk": 0.0,
                "recommendations": [
                    "Complete the NIHSS exam before ruling out stroke",
                ],
            }
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
    """Comprehensive neurocritical care prediction system integrating stroke, seizure, ICP.

    monitoring, and TBI assessment.
    """

    def __init__(
        self,
        enable_stroke: bool = True,
        enable_seizure: bool = True,
        enable_icp: bool = True,
    ):
        """Initialize the instance."""
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
        """Comprehensive neurocritical care prediction.

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

        safety = result.safety

        if self.enable_stroke and "clinical_features" in patient_data:
            stroke_result = self._detect_stroke(patient_data["clinical_features"])
            result.ml_stroke_available = stroke_result["available"]
            if stroke_result["available"]:
                result.stroke_detected = stroke_result["stroke_detected"]
                result.stroke_type = stroke_result["stroke_type"]
                result.confidence = max(result.confidence, stroke_result["confidence"])
                result.clinical_recommendations.extend(stroke_result["recommendations"])

                if stroke_result["stroke_detected"]:
                    result.neurological_emergency_detected = True
                    result.emergency_type = "stroke"
                    safety.flag_emergency(f"possible stroke ({stroke_result['stroke_type']})")

        if "exam_findings" in patient_data:
            nihss = self.nihss_calculator.calculate_nihss(patient_data["exam_findings"])
            result.nihss_score = nihss["nihss_score"]
            result.nihss_is_lower_bound = nihss["nihss_is_lower_bound"]
            result.risk_score = max(result.risk_score, nihss["stroke_risk"])
            result.clinical_recommendations.extend(nihss["recommendations"])
            safety.note_unassessed([f"nihss.{i}" for i in nihss["unassessed_items"]])
            safety.provenance["nihss"] = build_provenance(
                instrument="NIHSS",
                version="v1",
                inputs=patient_data["exam_findings"],
            )
            # A moderate-or-worse deficit is a time-critical neurological
            # emergency (tPA / thrombectomy window) — a real, deterministic
            # signal, independent of any ML model.
            if nihss["stroke_risk"] >= 0.6:
                result.neurological_emergency_detected = True
                if result.emergency_type == "none":
                    result.emergency_type = "stroke"
                safety.flag_emergency("NIHSS indicates moderate-or-worse stroke deficit")

        if self.enable_seizure and "temporal_sequence" in patient_data:
            seizure_result = self._predict_seizure(patient_data["temporal_sequence"])
            result.ml_seizure_available = seizure_result["available"]
            if seizure_result["available"]:
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
                    safety.flag_emergency("possible status epilepticus")

        if self.enable_icp and "icp_data" in patient_data:
            if self.icp_monitor is not None:
                icp_result = self.icp_monitor.assess_icp(patient_data["icp_data"])
                result.icp_measured = icp_result["icp_measured"]
                result.icp_elevated = icp_result["icp_elevated"]
                result.icp_mmhg = icp_result["icp_mmhg"]
                result.clinical_recommendations.extend(icp_result["recommendations"])
                if not icp_result["icp_measured"]:
                    safety.note_unassessed(["icp_data.icp_mmhg"])
                safety.provenance["icp"] = build_provenance(
                    instrument="ICPMonitor",
                    version="v1",
                    inputs=patient_data["icp_data"],
                )

                if icp_result["icp_critical"]:
                    result.neurological_emergency_detected = True
                    result.emergency_type = "elevated_icp"
                    safety.flag_emergency("critical intracranial pressure")
                elif icp_result["clinical_concern"] == "high_clinical_concern":
                    safety.flag_emergency("clinical signs concerning for raised ICP")

        if "tbi_features" in patient_data:
            tbi_result = self._assess_tbi(patient_data["tbi_features"])
            result.tbi_severity = tbi_result["severity"]
            result.gcs_score = tbi_result["gcs_score"]
            result.clinical_recommendations.extend(tbi_result["recommendations"])
            if tbi_result["severity"] == "severe":
                result.neurological_emergency_detected = True
                if result.emergency_type == "none":
                    result.emergency_type = "severe_tbi"
                safety.flag_emergency("severe traumatic brain injury (GCS ≤ 8)")

        result.risk_score = max(
            result.risk_score, result.seizure_risk_score, float(result.icp_elevated) * 0.7
        )
        result.confidence = max(result.confidence, result.risk_score)

        self.logger.info(
            "Neurocritical prediction: %s, stroke=%s (ml=%s), seizure=%s (ml=%s), "
            "icp_measured=%s, unassessed=%d",
            result.emergency_type,
            result.stroke_detected,
            result.ml_stroke_available,
            result.seizure_detected,
            result.ml_seizure_available,
            result.icp_measured,
            len(safety.unassessed_inputs),
        )

        return result

    def _detect_stroke(self, features: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Detect and classify stroke, or abstain if the model is untrained."""
        if self.stroke_detector is None or not getattr(self.stroke_detector, "is_fitted", False):
            # Fail-closed: an untrained network's output is noise, not a
            # prediction. Refuse rather than surface a fabricated stroke class.
            return {
                "available": False,
                "stroke_detected": False,
                "stroke_type": "no_stroke",
                "confidence": 0.0,
                "recommendations": [],
            }

        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        self.stroke_detector.eval()
        with torch.no_grad():
            classification, _severity = self.stroke_detector(features_tensor)

        probs = torch.softmax(classification[0], dim=0)
        stroke_idx = torch.argmax(probs).item()
        confidence = float(probs[stroke_idx].item())  # type: ignore[index, unused-ignore]

        stroke_types = [e.value for e in StrokeType]
        detected_type = stroke_types[stroke_idx]  # type: ignore[index, unused-ignore]

        stroke_detected = detected_type != "no_stroke"

        recs = []
        if detected_type == "ischemic_stroke":
            recs.append("Ischemic stroke suspected - tPA window assessment")
            recs.append("CT angiography for large vessel occlusion")
        elif detected_type == "hemorrhagic_stroke":
            recs.append("Hemorrhagic stroke suspected - neurosurgical consult")
            recs.append("Blood pressure control critical")

        return {
            "available": True,
            "stroke_detected": stroke_detected,
            "stroke_type": detected_type,
            "confidence": confidence,
            "recommendations": recs,
        }

    def _predict_seizure(self, sequence: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Predict seizure occurrence and type, or abstain if untrained."""
        if self.seizure_predictor is None or not getattr(
            self.seizure_predictor, "is_fitted", False
        ):
            # Fail-closed: refuse the untrained network rather than emit noise.
            return {
                "available": False,
                "seizure_detected": False,
                "seizure_type": "no_seizure",
                "risk_score": 0.0,
                "recommendations": [],
            }

        seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

        self.seizure_predictor.eval()
        with torch.no_grad():
            classification, risk, _attention = self.seizure_predictor(seq_tensor)

        probs = torch.softmax(classification[0], dim=0)
        seizure_idx = torch.argmax(probs).item()
        risk_score = float(risk[0].item())

        seizure_types = [e.value for e in SeizureType]
        detected_type = seizure_types[seizure_idx]  # type: ignore[index, unused-ignore]

        seizure_detected = detected_type != "no_seizure"

        recs = []
        if detected_type == "status_epilepticus":
            recs.append("STATUS EPILEPTICUS - IMMEDIATE BENZODIAZEPINES")
            recs.append("Prepare for intubation if refractory")
        elif seizure_detected:
            recs.append(f"Seizure detected: {detected_type}")
            recs.append("Antiepileptic medication consideration")

        return {
            "available": True,
            "seizure_detected": seizure_detected,
            "seizure_type": detected_type,
            "risk_score": risk_score,
            "recommendations": recs,
        }

    def _assess_tbi(self, tbi_features: dict[str, Any]) -> dict[str, Any]:
        """Assess traumatic brain injury severity (abstains without a GCS)."""
        gcs = tbi_features.get("gcs_score")
        if gcs is None:
            return {
                "severity": None,
                "gcs_score": None,
                "recommendations": [
                    "TBI severity requires a Glasgow Coma Scale; none provided.",
                ],
            }

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
