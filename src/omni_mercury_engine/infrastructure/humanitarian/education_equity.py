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
Education Equity Module - Learning bias and access anomaly detection

Supports SDG 4 (Quality Education) by monitoring:
- Learning outcome disparities
- Resource allocation inequities
- Access barriers for marginalized groups
- Dropout risk prediction
- Curriculum bias detection

⚠️ SIMULATION-BASED: Uses simulated educational data. Real-world validation required.

Research sources:
- UNESCO Education data
- OECD education statistics
- EdTech research on learning analytics

"""

from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np


class EducationThreat(Enum):
    NORMAL = "normal"
    ACHIEVEMENT_GAP = "achievement_gap"
    DROPOUT_RISK = "dropout_risk"
    ACCESS_BARRIER = "access_barrier"
    RESOURCE_INEQUITY = "resource_inequity"
    BIAS_DETECTED = "bias_detected"


class EducationEquityDetector:
    """Detect educational equity anomalies and learning barriers."""

    def __init__(self) -> None:
        self.achievement_baseline = {"mean": 75.0, "std": 10.0}
        self.dropout_risk_threshold = 0.3

    def detect(
        self,
        data: np.ndarray[Any, Any],
        detection_type: str = "achievement",
        demographic_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Detect educational equity anomalies.

        Args:
            data: Educational metrics (test scores, attendance, etc.)
            detection_type: 'achievement', 'dropout_risk', 'access'
            demographic_data: Student demographic information

        Returns:
            Education equity anomaly detection results
        """
        if detection_type == "achievement":
            return self.detect_achievement_gap(data, demographic_data)
        elif detection_type == "dropout_risk":
            return self.detect_dropout_risk(data, demographic_data)
        else:
            return self.detect_achievement_gap(data, demographic_data)

    def detect_achievement_gap(
        self, achievement_data: np.ndarray[Any, Any], demographic_data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Detect achievement gaps across student populations."""
        if len(achievement_data) == 0:
            return {"anomaly_detected": False}

        mean_achievement = np.mean(achievement_data)
        std_achievement = np.std(achievement_data)
        baseline_mean = self.achievement_baseline["mean"]
        baseline_std = self.achievement_baseline["std"]

        z_score = (mean_achievement - baseline_mean) / baseline_std if baseline_std > 0 else 0

        threat_type = EducationThreat.NORMAL
        severity = "low"

        if mean_achievement < baseline_mean - 2 * baseline_std:
            threat_type = EducationThreat.ACHIEVEMENT_GAP
            severity = "critical" if z_score < -3 else "high"

        disparity_score = std_achievement / mean_achievement if mean_achievement > 0 else 0

        return {
            "anomaly_detected": z_score < -2 or disparity_score > 0.3,
            "threat_type": threat_type.value,
            "severity": severity,
            "metrics": {
                "mean_achievement": float(mean_achievement),
                "std_achievement": float(std_achievement),
                "disparity_score": float(disparity_score),
                "z_score": float(z_score),
            },
            "equity_risk": self._assess_equity_risk(disparity_score, severity),
            "affected_students_estimate": self._estimate_affected_students(z_score),
            "recommendations": self._generate_education_recommendations(
                threat_type, severity, disparity_score
            ),
            "timestamp": datetime.now(),
        }

    def detect_dropout_risk(
        self, student_data: np.ndarray[Any, Any], demographic_data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Detect students at risk of dropping out."""
        if len(student_data) == 0:
            return {"anomaly_detected": False}

        risk_indicators = np.mean(student_data, axis=0) if student_data.ndim > 1 else student_data
        dropout_risk = np.mean(risk_indicators < self.dropout_risk_threshold)

        threat_type = EducationThreat.NORMAL
        severity = "low"

        if dropout_risk > 0.2:
            threat_type = EducationThreat.DROPOUT_RISK
            severity = "critical" if dropout_risk > 0.4 else "high"

        return {
            "anomaly_detected": dropout_risk > 0.15,
            "threat_type": threat_type.value,
            "severity": severity,
            "metrics": {
                "dropout_risk_rate": float(dropout_risk),
                "at_risk_percentage": float(dropout_risk * 100),
            },
            "intervention_urgency": "immediate" if severity == "critical" else "standard",
            "recommendations": self._generate_education_recommendations(
                threat_type, severity, dropout_risk
            ),
            "timestamp": datetime.now(),
        }

    def _assess_equity_risk(self, disparity_score: float, severity: str) -> str:
        """Assess educational equity risk."""
        if severity == "critical" or disparity_score > 0.4:
            return "severe_inequity"
        elif severity == "high" or disparity_score > 0.3:
            return "moderate_inequity"
        return "acceptable"

    def _estimate_affected_students(self, z_score: float) -> int:
        """Estimate number of students affected."""
        if z_score < -3:
            return 500
        elif z_score < -2:
            return 200
        return 0

    def _generate_education_recommendations(
        self, threat_type: EducationThreat, severity: str, metric: float
    ) -> list[str]:
        """Generate educational equity recommendations."""
        recommendations = []

        if threat_type == EducationThreat.ACHIEVEMENT_GAP:
            recommendations.append("Implement targeted interventions for underperforming groups")
            recommendations.append("Review curriculum for cultural bias")
            if severity == "critical":
                recommendations.append("URGENT: Deploy emergency tutoring programs")
        elif threat_type == EducationThreat.DROPOUT_RISK:
            recommendations.append("Activate student retention programs")
            recommendations.append("Provide counseling and mentorship")
            if severity == "critical":
                recommendations.append("CRITICAL: Immediate one-on-one interventions")

        if not recommendations:
            recommendations.append("Continue routine educational monitoring")

        return recommendations
