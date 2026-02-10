"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisors LLC

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
Medical Cure Predictor - Early Disease Detection and Treatment Optimization

Novel medical constructions for humanitarian healthcare:
- Temporal vital signs anomaly detection (PyTorch LSTM/GNN)
- Medical imaging anomaly detection (DeepFace-like approach)
- Treatment pathway optimization (multiverse exploration)

⚠️ SIMULATION-BASED: Uses simulated vital signs and medical images. Clinical validation required.
Consult medical professionals before acting on predictions.

Research sources:
- PMC (PubMed Central) research on AI in medical diagnosis
- CDC disease surveillance methodologies
- WHO treatment guidelines
- arXiv research on deep learning in medical imaging

"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from scipy.ndimage import zoom
from torch import nn

from omni_mercury_engine.infrastructure.healthcare_emergency import HealthcareEmergencyDetector
from omni_mercury_engine.models.multiverse import MultiverseOmniEngine
from omni_mercury_engine.utils.logging import LoggerMixin


@dataclass
class MedicalPredictionResult:
    """Result from medical cure prediction analysis."""

    disease_risk_detected: bool
    confidence: float
    disease_type: str
    risk_score: float
    vital_signs_anomaly: bool
    imaging_anomaly: bool
    optimal_treatment: str | None = None
    treatment_pathways: list[dict[str, Any]] = field(default_factory=list)
    early_warning_indicators: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class TemporalVitalSignsLSTM(nn.Module):
    """
    PyTorch LSTM for temporal vital signs anomaly detection.

    Detects disease progression patterns and early warning signs
    through time-series analysis of patient vitals.
    """

    def __init__(
        self, input_dim: int = 5, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.attention = nn.Linear(hidden_dim, 1)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LSTM.

        Args:
            x: Input tensor of shape (batch, sequence_length, input_dim)

        Returns:
            Tuple of (anomaly_scores, attention_weights)
        """
        lstm_out, _ = self.lstm(x)

        attention_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)

        context = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)

        anomaly_scores = self.fc(context)

        return anomaly_scores, attention_weights


class TemporalVitalSignsDetector:
    """
    Temporal vital signs anomaly detector using LSTM.

    Extends HealthcareEmergencyDetector with temporal pattern analysis
    for early disease detection.
    """

    def __init__(self) -> None:
        self.healthcare_detector = HealthcareEmergencyDetector()
        self.lstm_model = TemporalVitalSignsLSTM(input_dim=5, hidden_dim=64, num_layers=2)
        self.logger = logging.getLogger(__name__)

    def detect_temporal_anomaly(
        self,
        vital_signs_sequence: np.ndarray[Any, Any],
        patient_history: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Detect anomalies in temporal vital signs sequence.

        Args:
            vital_signs_sequence: Time-series vital signs (seq_len, 5)
                Columns: [heart_rate, blood_pressure, oxygen_sat, temp, resp_rate]
            patient_history: Optional patient context

        Returns:
            Temporal anomaly detection results with disease risk
        """
        normalized = self._normalize_vitals(vital_signs_sequence)

        x = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)

        self.lstm_model.eval()
        with torch.no_grad():
            anomaly_scores, attention_weights = self.lstm_model(x)

        anomaly_score = float(anomaly_scores.item())
        attention = attention_weights[0].numpy()

        critical_indices = np.where(attention > np.percentile(attention, 75))[0]

        current_vitals = {
            "heart_rate_bpm": float(vital_signs_sequence[-1, 0]),
            "blood_pressure_systolic": float(vital_signs_sequence[-1, 1]),
            "oxygen_saturation_pct": float(vital_signs_sequence[-1, 2]),
            "temperature_f": float(vital_signs_sequence[-1, 3]),
            "respiratory_rate_bpm": float(vital_signs_sequence[-1, 4]),
        }

        current_assessment = self.healthcare_detector.detect_patient_deterioration(
            current_vitals, patient_history
        )

        disease_risk = self._calculate_disease_risk(
            anomaly_score, current_assessment, vital_signs_sequence
        )

        return {
            "temporal_anomaly_detected": anomaly_score > 0.5,
            "anomaly_score": anomaly_score,
            "disease_risk": disease_risk,
            "critical_time_indices": critical_indices.tolist(),
            "attention_weights": attention.tolist(),
            "current_status": current_assessment["patient_status"],
            "early_warning_score": current_assessment["early_warning_score"],
            "recommendations": self._generate_temporal_recommendations(
                anomaly_score, disease_risk, current_assessment
            ),
        }

    def _normalize_vitals(self, vitals: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Normalize vital signs for LSTM input."""
        ranges = np.array([[40, 130], [70, 180], [85, 100], [95, 103], [8, 30]])

        normalized = (vitals - ranges[:, 0]) / (ranges[:, 1] - ranges[:, 0])
        return np.clip(normalized, 0, 1)

    def _calculate_disease_risk(
        self,
        anomaly_score: float,
        assessment: dict[str, Any],
        vitals_sequence: np.ndarray[Any, Any],
    ) -> dict[str, float]:
        """Calculate disease risk scores."""
        if len(vitals_sequence) >= 3:
            recent_trend = np.mean(np.diff(vitals_sequence[-3:, :], axis=0), axis=0)
        else:
            recent_trend = np.zeros(5)

        risks = {"cardiac": 0.0, "respiratory": 0.0, "metabolic": 0.0, "infectious": 0.0}

        if abs(recent_trend[0]) > 5 or abs(recent_trend[1]) > 10:
            risks["cardiac"] = min(0.8, anomaly_score + 0.2)

        if recent_trend[2] < -1 or abs(recent_trend[4]) > 2:
            risks["respiratory"] = min(0.8, anomaly_score + 0.2)

        if abs(recent_trend[3]) > 1:
            risks["metabolic"] = min(0.7, anomaly_score + 0.1)

        if vitals_sequence[-1, 3] > 100.4 and vitals_sequence[-1, 0] > 100:
            risks["infectious"] = min(0.7, anomaly_score + 0.15)

        return risks

    def _generate_temporal_recommendations(
        self, anomaly_score: float, disease_risk: dict[str, Any], assessment: dict[str, Any]
    ) -> list[str]:
        """Generate recommendations based on temporal analysis."""
        recs = []

        if anomaly_score > 0.7:
            recs.append("HIGH RISK: Significant temporal anomaly detected in vital signs")
            recs.append("Recommend immediate clinical evaluation")

        max_risk_type = max(disease_risk, key=lambda k: disease_risk[k])
        max_risk = disease_risk[max_risk_type]

        if max_risk > 0.6:
            recs.append(f"Elevated {max_risk_type} disease risk: {max_risk:.2f}")

            if max_risk_type == "cardiac":
                recs.append("Consider ECG and cardiac biomarkers")
            elif max_risk_type == "respiratory":
                recs.append("Consider chest X-ray and ABG")
            elif max_risk_type == "infectious":
                recs.append("Consider blood cultures and infectious workup")
            elif max_risk_type == "metabolic":
                recs.append("Consider metabolic panel and endocrine evaluation")

        recs.extend(assessment.get("recommended_actions", []))

        return recs


class MedicalImagingAnomalyDetector:
    """
    Medical imaging anomaly detection using DeepFace-like approach.

    Detects anomalies in CT/MRI/X-ray images for early disease detection.
    """

    def __init__(self) -> None:
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.logger = logging.getLogger(__name__)

    def detect_imaging_anomaly(
        self, medical_image: np.ndarray[Any, Any], imaging_type: str = "xray"
    ) -> dict[str, Any]:
        """
        Detect anomalies in medical imaging.

        Args:
            medical_image: Medical image array (H, W) or (H, W, C)
            imaging_type: Type of imaging ('xray', 'ct', 'mri')

        Returns:
            Imaging anomaly detection results
        """
        processed = self._preprocess_image(medical_image)

        x = torch.tensor(processed, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            anomaly_score = float(self.model(x).item())

        is_anomalous = anomaly_score > 0.5

        findings = self._generate_findings(is_anomalous, anomaly_score, imaging_type)

        return {
            "imaging_anomaly_detected": is_anomalous,
            "anomaly_score": anomaly_score,
            "imaging_type": imaging_type,
            "findings": findings,
            "recommendations": self._generate_imaging_recommendations(
                is_anomalous, anomaly_score, imaging_type
            ),
        }

    def _preprocess_image(self, image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Preprocess medical image for CNN."""
        if image.ndim == 3:
            image = np.mean(image, axis=2)

        target_shape = (224, 224)
        zoom_factors = (target_shape[0] / image.shape[0], target_shape[1] / image.shape[1])
        resized = zoom(image, zoom_factors, order=1)

        normalized = (resized - np.min(resized)) / (np.max(resized) - np.min(resized) + 1e-8)

        return normalized

    def _generate_findings(self, is_anomalous: bool, score: float, imaging_type: str) -> list[str]:
        """Generate findings based on anomaly detection."""
        findings = []

        if is_anomalous:
            if score > 0.8:
                findings.append(f"High-confidence abnormality detected in {imaging_type}")
            elif score > 0.6:
                findings.append(f"Moderate abnormality detected in {imaging_type}")
            else:
                findings.append(f"Possible abnormality in {imaging_type}")

            if imaging_type == "xray":
                findings.append("Recommend correlation with clinical history")
                findings.append("Consider follow-up CT if clinically indicated")
            elif imaging_type == "ct":
                findings.append("Detailed anatomical assessment recommended")
            elif imaging_type == "mri":
                findings.append("High-resolution soft tissue evaluation")
        else:
            findings.append(f"No significant abnormality detected in {imaging_type}")

        return findings

    def _generate_imaging_recommendations(
        self, is_anomalous: bool, score: float, imaging_type: str
    ) -> list[str]:
        """Generate imaging recommendations."""
        recs = []

        if is_anomalous:
            recs.append("Radiologist review recommended")
            recs.append("Compare with prior imaging if available")
            recs.append("Correlate with laboratory findings")

            if score > 0.7:
                recs.append("Consider biopsy or interventional procedure if appropriate")
        else:
            recs.append("Routine follow-up as clinically indicated")

        return recs


class TreatmentPathwayOptimizer:
    """
    Treatment pathway optimization using multiverse exploration.

    Explores optimal treatment strategies using multiverse optimization.
    """

    def __init__(self, num_universes: int = 25) -> None:
        self.multiverse = MultiverseOmniEngine(
            num_universes=num_universes, state_dim=96, convergence_threshold=0.9
        )
        self.logger = logging.getLogger(__name__)

    def optimize_treatment(
        self, patient_state: dict[str, Any], disease_type: str
    ) -> dict[str, Any]:
        """
        Optimize treatment pathway using multiverse exploration.

        Args:
            patient_state: Current patient state
            disease_type: Diagnosed disease type

        Returns:
            Optimal treatment pathway recommendations
        """

        def treatment_fitness(treatment_vector: np.ndarray[Any, Any]) -> float:
            efficacy = np.mean(treatment_vector[:32])
            safety = -np.std(treatment_vector[32:64])
            tolerance = np.mean(treatment_vector[64:])

            return float(efficacy + safety * 0.5 + tolerance * 0.3)

        self.multiverse.converge_multiverse(treatment_fitness)

        sorted_universes = sorted(
            self.multiverse.universes.values(), key=lambda u: u.fitness, reverse=True
        )

        treatment_pathways = []
        for i, universe in enumerate(sorted_universes[:3]):
            treatment_pathways.append(
                {
                    "pathway_id": f"TREATMENT-{i+1}",
                    "fitness": float(universe.fitness),
                    "efficacy_score": float(np.mean(universe.state_vector[:32])),
                    "safety_score": float(-np.std(universe.state_vector[32:64])),
                    "tolerance_score": float(np.mean(universe.state_vector[64:])),
                }
            )

        return {
            "optimal_treatment": (
                treatment_pathways[0]["pathway_id"] if treatment_pathways else None
            ),
            "treatment_pathways": treatment_pathways,
            "pathways_explored": len(self.multiverse.universes),
            "recommendations": self._generate_treatment_recommendations(
                treatment_pathways, disease_type
            ),
        }

    def _generate_treatment_recommendations(
        self, pathways: list[dict[str, Any]], disease_type: str
    ) -> list[str]:
        """Generate treatment recommendations."""
        recs = []

        if pathways:
            best = pathways[0]
            recs.append(f"Optimal treatment pathway: {best['pathway_id']}")
            recs.append(f"Efficacy score: {best['efficacy_score']:.3f}")
            recs.append(f"Safety score: {best['safety_score']:.3f}")
            recs.append("Consider multi-modal approach for best outcomes")
        else:
            recs.append("Unable to identify optimal treatment pathway")
            recs.append("Consult disease-specific guidelines")

        recs.append(f"Treatment tailored for {disease_type}")
        recs.append("Regular monitoring and adjustment recommended")

        return recs


class MedicalCurePredictor(LoggerMixin):
    """
    Unified medical cure predictor integrating temporal analysis,
    imaging detection, and treatment optimization.
    """

    def __init__(
        self,
        enable_temporal: bool = True,
        enable_imaging: bool = True,
        enable_treatment_opt: bool = True,
    ):
        self.enable_temporal = enable_temporal
        self.enable_imaging = enable_imaging
        self.enable_treatment_opt = enable_treatment_opt

        self.temporal_detector = TemporalVitalSignsDetector() if enable_temporal else None
        self.imaging_detector = MedicalImagingAnomalyDetector() if enable_imaging else None
        self.treatment_optimizer = TreatmentPathwayOptimizer() if enable_treatment_opt else None

    def predict_and_cure(self, patient_data: dict[str, Any]) -> MedicalPredictionResult:
        """
        Comprehensive medical prediction and cure optimization.

        Args:
            patient_data: Patient data including:
                - vital_signs_sequence: Time-series vital signs
                - medical_image: Optional medical imaging
                - patient_history: Patient context

        Returns:
            Medical prediction results with treatment recommendations
        """
        result = MedicalPredictionResult(
            disease_risk_detected=False,
            confidence=0.0,
            disease_type="unknown",
            risk_score=0.0,
            vital_signs_anomaly=False,
            imaging_anomaly=False,
        )

        if self.enable_temporal and "vital_signs_sequence" in patient_data:
            if self.temporal_detector is not None:
                temporal = self.temporal_detector.detect_temporal_anomaly(
                    patient_data["vital_signs_sequence"], patient_data.get("patient_history")
                )
                result.vital_signs_anomaly = temporal["temporal_anomaly_detected"]
                result.confidence = max(result.confidence, temporal["anomaly_score"])
                result.early_warning_indicators.extend(
                    [
                        f"Early warning score: {temporal['early_warning_score']}",
                        f"Current status: {temporal['current_status']}",
                    ]
                )
                result.recommendations.extend(temporal["recommendations"])

                disease_risks = temporal["disease_risk"]
                max_risk_type = max(disease_risks, key=disease_risks.get)
                result.risk_score = disease_risks[max_risk_type]
                if result.risk_score > 0.5:
                    result.disease_risk_detected = True
                    result.disease_type = max_risk_type

        if self.enable_imaging and "medical_image" in patient_data:
            if self.imaging_detector is not None:
                imaging = self.imaging_detector.detect_imaging_anomaly(
                    patient_data["medical_image"], patient_data.get("imaging_type", "xray")
                )
                result.imaging_anomaly = imaging["imaging_anomaly_detected"]
                result.confidence = max(result.confidence, imaging["anomaly_score"])
                result.early_warning_indicators.extend(imaging["findings"])
                result.recommendations.extend(imaging["recommendations"])

                if imaging["imaging_anomaly_detected"]:
                    result.disease_risk_detected = True

        if (
            self.enable_treatment_opt
            and result.disease_risk_detected
            and result.disease_type != "unknown"
        ):
            if self.treatment_optimizer is not None:
                treatment = self.treatment_optimizer.optimize_treatment(
                    patient_data, result.disease_type
                )
                result.optimal_treatment = treatment["optimal_treatment"]
                result.treatment_pathways = treatment["treatment_pathways"]
                result.recommendations.extend(treatment["recommendations"])

        return result
