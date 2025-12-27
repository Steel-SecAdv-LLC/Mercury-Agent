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

from __future__ import annotations

"""
Cardiology Predictor - Cardiovascular Disease Detection & Risk Assessment

Advanced cardiac anomaly detection for humanitarian healthcare:
- ECG rhythm analysis (PyTorch 1D CNN + LSTM)
- Cardiac biomarker anomaly detection
- Arrhythmia classification (13 rhythm types)
- Myocardial infarction risk prediction
- Heart failure progression modeling

⚠️ SIMULATION-BASED: Uses simulated ECG and biomarker data. Clinical validation required.
Consult cardiologists before acting on predictions.

Research sources:
- MIT-BIH Arrhythmia Database methodologies
- AHA/ACC cardiovascular guidelines
- PTB-XL ECG database architectures
- Framingham Heart Study risk algorithms

"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
from torch import nn


class ArrhythmiaType(Enum):
    """13 cardiac rhythm classifications"""

    NORMAL_SINUS = "normal_sinus_rhythm"
    ATRIAL_FIB = "atrial_fibrillation"
    ATRIAL_FLUTTER = "atrial_flutter"
    SINUS_TACHYCARDIA = "sinus_tachycardia"
    SINUS_BRADYCARDIA = "sinus_bradycardia"
    VENTRICULAR_TACH = "ventricular_tachycardia"
    VENTRICULAR_FIB = "ventricular_fibrillation"
    SVT = "supraventricular_tachycardia"
    PVC = "premature_ventricular_contraction"
    PAC = "premature_atrial_contraction"
    AV_BLOCK = "atrioventricular_block"
    BUNDLE_BRANCH_BLOCK = "bundle_branch_block"
    ST_ELEVATION = "st_elevation_mi"


@dataclass
class CardiologyPredictionResult:
    """Cardiology prediction results"""

    cardiac_risk_detected: bool
    confidence: float
    arrhythmia_type: str
    risk_score: float

    mi_risk: float
    heart_failure_risk: float
    stroke_risk: float

    ecg_anomalies: list[str] = field(default_factory=list)
    biomarker_alerts: list[str] = field(default_factory=list)
    clinical_recommendations: list[str] = field(default_factory=list)

    framingham_score: float | None = None
    acute_intervention_needed: bool = False


class ECGRhythmAnalyzer(nn.Module):
    """
    1D CNN + LSTM for ECG rhythm analysis.

    Detects arrhythmias and cardiac anomalies from raw ECG signals.
    Architecture inspired by PTB-XL and MIT-BIH research.
    """

    def __init__(
        self, input_length: int = 1000, num_leads: int = 12, num_classes: int = 13
    ) -> None:
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(num_leads, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )

        self.attention = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, ecg_signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for ECG analysis.

        Args:
            ecg_signal: ECG tensor (batch, leads, time_steps)

        Returns:
            Tuple of (rhythm_classification, attention_weights)
        """
        conv_features = self.conv_layers(ecg_signal)

        lstm_input = conv_features.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_input)

        attention_scores = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_scores, dim=1)

        context = torch.sum(lstm_out * attention_weights, dim=1)

        classification = self.classifier(context)

        return classification, attention_weights.squeeze(-1)


class CardiacBiomarkerAnalyzer:
    """
    Cardiac biomarker anomaly detection.

    Analyzes troponin, BNP, CK-MB, and other cardiac markers.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        self.normal_ranges = {
            "troponin_i_ng_ml": (0.0, 0.04),
            "troponin_t_ng_ml": (0.0, 0.01),
            "bnp_pg_ml": (0.0, 100.0),
            "nt_probnp_pg_ml": (0.0, 125.0),
            "ck_mb_ng_ml": (0.0, 5.0),
            "myoglobin_ng_ml": (0.0, 90.0),
            "ldh_u_l": (140.0, 280.0),
        }

        self.critical_thresholds = {
            "troponin_i_ng_ml": 0.4,
            "bnp_pg_ml": 400.0,
            "nt_probnp_pg_ml": 900.0,
        }

    def analyze_biomarkers(self, biomarkers: dict[str, float]) -> dict[str, Any]:
        """
        Analyze cardiac biomarkers for anomalies.

        Args:
            biomarkers: Dictionary of biomarker values

        Returns:
            Biomarker analysis with risk assessment
        """
        anomalies = []
        alerts = []
        mi_indicators = 0
        heart_failure_indicators = 0

        for marker, value in biomarkers.items():
            if marker in self.normal_ranges:
                min_val, max_val = self.normal_ranges[marker]

                if value < min_val or value > max_val:
                    anomalies.append(f"{marker}: {value:.3f} (normal: {min_val}-{max_val})")

                if marker in self.critical_thresholds:
                    if value > self.critical_thresholds[marker]:
                        alerts.append(f"CRITICAL: {marker} = {value:.3f}")

                        if "troponin" in marker:
                            mi_indicators += 1
                        if "bnp" in marker or "probnp" in marker:
                            heart_failure_indicators += 1

        mi_risk = min(mi_indicators / 2.0, 1.0)
        hf_risk = min(heart_failure_indicators / 2.0, 1.0)

        acute_mi = (
            biomarkers.get("troponin_i_ng_ml", 0) > 0.4
            or biomarkers.get("troponin_t_ng_ml", 0) > 0.1
        )

        return {
            "biomarker_anomalies": anomalies,
            "critical_alerts": alerts,
            "mi_risk": mi_risk,
            "heart_failure_risk": hf_risk,
            "acute_mi_suspected": acute_mi,
            "recommendations": self._generate_biomarker_recommendations(mi_risk, hf_risk, acute_mi),
        }

    def _generate_biomarker_recommendations(
        self, mi_risk: float, hf_risk: float, acute_mi: bool
    ) -> list[str]:
        """Generate clinical recommendations based on biomarkers"""
        recs = []

        if acute_mi:
            recs.append("URGENT: Acute MI suspected - immediate cardiology consult")
            recs.append("Activate cath lab for urgent PCI consideration")
            recs.append("Administer aspirin, heparin, and antithrombotic therapy")

        if mi_risk > 0.5:
            recs.append("Elevated MI risk - serial troponin monitoring")
            recs.append("12-lead ECG and continuous telemetry")

        if hf_risk > 0.5:
            recs.append("Heart failure suspected - echo evaluation")
            recs.append("Consider BNP-guided diuretic therapy")
            recs.append("Assess for volume overload")

        return recs


class FraminghamRiskCalculator:
    """
    Framingham Risk Score calculator for 10-year CVD risk.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def calculate_risk(self, patient_data: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate Framingham 10-year CVD risk.

        Args:
            patient_data: Patient demographics and risk factors

        Returns:
            Risk score and interpretation
        """
        age = patient_data.get("age", 50)
        gender = patient_data.get("gender", "male")
        total_chol = patient_data.get("total_cholesterol_mg_dl", 200)
        hdl_chol = patient_data.get("hdl_cholesterol_mg_dl", 50)
        systolic_bp = patient_data.get("systolic_bp_mmhg", 120)
        smoker = patient_data.get("smoker", False)
        diabetes = patient_data.get("diabetes", False)

        if gender == "male":
            points = self._calculate_male_points(
                age, total_chol, hdl_chol, systolic_bp, smoker, diabetes
            )
        else:
            points = self._calculate_female_points(
                age, total_chol, hdl_chol, systolic_bp, smoker, diabetes
            )

        risk_percent = self._points_to_risk(points, gender)

        interpretation = self._interpret_risk(risk_percent)

        return {
            "framingham_score": points,
            "10_year_cvd_risk_percent": risk_percent,
            "risk_category": interpretation["category"],
            "recommendations": interpretation["recommendations"],
        }

    def _calculate_male_points(
        self,
        age: int,
        total_chol: float,
        hdl_chol: float,
        systolic_bp: int,
        smoker: bool,
        diabetes: bool,
    ) -> int:
        """Calculate Framingham points for males"""
        points = 0

        if age < 35:
            points += -1
        elif age < 40:
            points += 0
        elif age < 45:
            points += 1
        elif age < 50:
            points += 2
        elif age < 55:
            points += 3
        elif age < 60:
            points += 4
        elif age < 65:
            points += 5
        elif age < 70:
            points += 6
        else:
            points += 7

        if total_chol < 160:
            points += -3
        elif total_chol < 200:
            points += 0
        elif total_chol < 240:
            points += 1
        elif total_chol < 280:
            points += 2
        else:
            points += 3

        if hdl_chol >= 60:
            points += -2
        elif hdl_chol >= 50:
            points += -1
        elif hdl_chol >= 35:
            points += 0
        else:
            points += 2

        if systolic_bp < 120:
            points += -2
        elif systolic_bp < 130:
            points += 0
        elif systolic_bp < 140:
            points += 1
        elif systolic_bp < 160:
            points += 2
        else:
            points += 3

        if smoker:
            points += 2
        if diabetes:
            points += 2

        return points

    def _calculate_female_points(
        self,
        age: int,
        total_chol: float,
        hdl_chol: float,
        systolic_bp: int,
        smoker: bool,
        diabetes: bool,
    ) -> int:
        """Calculate Framingham points for females"""
        points = 0

        if age < 35:
            points += -9
        elif age < 40:
            points += -4
        elif age < 45:
            points += 0
        elif age < 50:
            points += 3
        elif age < 55:
            points += 6
        elif age < 60:
            points += 7
        elif age < 65:
            points += 8
        else:
            points += 9

        if total_chol < 160:
            points += -2
        elif total_chol < 200:
            points += 0
        elif total_chol < 240 or total_chol < 280:
            points += 1
        else:
            points += 2

        if hdl_chol >= 60:
            points += -2
        elif hdl_chol >= 50:
            points += -1
        elif hdl_chol >= 35:
            points += 0
        else:
            points += 2

        if systolic_bp < 120:
            points += -3
        elif systolic_bp < 130:
            points += 0
        elif systolic_bp < 140:
            points += 1
        elif systolic_bp < 160:
            points += 2
        else:
            points += 3

        if smoker:
            points += 2
        if diabetes:
            points += 4

        return points

    def _points_to_risk(self, points: int, gender: str) -> float:
        """Convert points to 10-year CVD risk percentage"""
        if gender == "male":
            risk_map = {
                -3: 1,
                -2: 1,
                -1: 2,
                0: 2,
                1: 3,
                2: 4,
                3: 5,
                4: 7,
                5: 9,
                6: 11,
                7: 14,
                8: 18,
                9: 22,
                10: 27,
                11: 33,
                12: 40,
                13: 47,
            }
        else:
            risk_map = {
                -2: 1,
                -1: 2,
                0: 2,
                1: 2,
                2: 3,
                3: 3,
                4: 4,
                5: 5,
                6: 6,
                7: 7,
                8: 8,
                9: 9,
                10: 11,
                11: 13,
                12: 15,
                13: 17,
                14: 20,
                15: 24,
                16: 27,
            }

        return float(risk_map.get(points, 30 if points > 13 else 1))

    def _interpret_risk(self, risk_percent: float) -> dict[str, Any]:
        """Interpret Framingham risk score"""
        if risk_percent < 10:
            category = "Low Risk"
            recs = [
                "Continue healthy lifestyle",
                "Regular cardiovascular screening",
                "Maintain optimal cholesterol and BP",
            ]
        elif risk_percent < 20:
            category = "Moderate Risk"
            recs = [
                "Consider statin therapy",
                "Lifestyle modification counseling",
                "Aggressive BP control if hypertensive",
                "Smoking cessation if applicable",
            ]
        else:
            category = "High Risk"
            recs = [
                "Initiate statin therapy",
                "Intensive lifestyle modification",
                "Strict BP control (target <130/80)",
                "Consider aspirin prophylaxis",
                "Cardiology referral",
            ]

        return {"category": category, "recommendations": recs}


class CardiologyPredictor:
    """
    Comprehensive cardiology prediction system integrating ECG analysis,
    biomarker detection, and risk stratification.
    """

    def __init__(self, enable_ecg: bool = True, enable_biomarkers: bool = True) -> None:
        self.enable_ecg = enable_ecg
        self.enable_biomarkers = enable_biomarkers

        self.ecg_analyzer = ECGRhythmAnalyzer() if enable_ecg else None
        self.biomarker_analyzer = CardiacBiomarkerAnalyzer() if enable_biomarkers else None
        self.risk_calculator = FraminghamRiskCalculator()

        self.logger = logging.getLogger(__name__)

    def predict_cardiac_risk(self, patient_data: dict[str, Any]) -> CardiologyPredictionResult:
        """
        Comprehensive cardiac risk prediction.

        Args:
            patient_data: Patient data including:
                - ecg_signal: 12-lead ECG (optional)
                - biomarkers: Cardiac biomarkers (optional)
                - demographics: Age, gender, risk factors

        Returns:
            Cardiology prediction with risk stratification
        """
        result = CardiologyPredictionResult(
            cardiac_risk_detected=False,
            confidence=0.0,
            arrhythmia_type="normal_sinus_rhythm",
            risk_score=0.0,
            mi_risk=0.0,
            heart_failure_risk=0.0,
            stroke_risk=0.0,
        )

        if self.enable_ecg and "ecg_signal" in patient_data:
            ecg_result = self._analyze_ecg(patient_data["ecg_signal"])
            result.arrhythmia_type = ecg_result["arrhythmia_type"]
            result.ecg_anomalies = ecg_result["anomalies"]
            result.confidence = max(result.confidence, ecg_result["confidence"])

            if ecg_result["arrhythmia_type"] != "normal_sinus_rhythm":
                result.cardiac_risk_detected = True

        if self.enable_biomarkers and "biomarkers" in patient_data:
            biomarker_result = self.biomarker_analyzer.analyze_biomarkers(
                patient_data["biomarkers"]
            )
            result.biomarker_alerts = biomarker_result["critical_alerts"]
            result.mi_risk = biomarker_result["mi_risk"]
            result.heart_failure_risk = biomarker_result["heart_failure_risk"]
            result.clinical_recommendations.extend(biomarker_result["recommendations"])
            result.acute_intervention_needed = biomarker_result["acute_mi_suspected"]

            if biomarker_result["mi_risk"] > 0.5:
                result.cardiac_risk_detected = True

        if "demographics" in patient_data:
            framingham = self.risk_calculator.calculate_risk(patient_data["demographics"])
            result.framingham_score = framingham["framingham_score"]
            result.clinical_recommendations.extend(framingham["recommendations"])

            cvd_risk = framingham["10_year_cvd_risk_percent"] / 100.0
            result.stroke_risk = cvd_risk * 0.3

        result.risk_score = max(result.mi_risk, result.heart_failure_risk, result.stroke_risk)
        result.confidence = max(result.confidence, result.risk_score)

        self.logger.info(
            f"Cardiology prediction: {result.arrhythmia_type}, "
            f"MI risk={result.mi_risk:.2f}, HF risk={result.heart_failure_risk:.2f}"
        )

        return result

    def _analyze_ecg(self, ecg_signal: np.ndarray[Any, Any]) -> dict[str, Any]:
        """Analyze ECG signal for arrhythmias"""
        ecg_tensor = torch.tensor(ecg_signal, dtype=torch.float32).unsqueeze(0)

        self.ecg_analyzer.eval()
        with torch.no_grad():
            classification, _attention = self.ecg_analyzer(ecg_tensor)

        probs = torch.softmax(classification[0], dim=0)
        rhythm_idx = torch.argmax(probs).item()
        confidence = float(probs[rhythm_idx].item())

        arrhythmia_types = [e.value for e in ArrhythmiaType]
        detected_rhythm = arrhythmia_types[rhythm_idx]

        anomalies = []
        if detected_rhythm != "normal_sinus_rhythm":
            anomalies.append(f"Detected: {detected_rhythm}")

        if detected_rhythm in ["ventricular_tachycardia", "ventricular_fibrillation"]:
            anomalies.append("LIFE-THREATENING ARRHYTHMIA - IMMEDIATE INTERVENTION REQUIRED")

        return {
            "arrhythmia_type": detected_rhythm,
            "confidence": confidence,
            "anomalies": anomalies,
        }
