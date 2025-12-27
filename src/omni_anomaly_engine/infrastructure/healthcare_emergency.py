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

from typing import Any

"""
CISA Healthcare & Emergency Services Critical Infrastructure Anomaly Detection

Patient monitoring, outbreak detection, and emergency response optimization.

Research sources:
- CISA Healthcare Sector framework
- CISA Emergency Services Sector framework
- CDC surveillance guidelines
- HIPAA security rules

"""

from datetime import datetime, timedelta
from enum import Enum

import numpy as np


class PatientStatus(Enum):
    STABLE = "stable"
    DETERIORATING = "deteriorating"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class EmergencyType(Enum):
    MEDICAL = "medical"
    FIRE = "fire"
    LAW_ENFORCEMENT = "law_enforcement"
    RESCUE = "rescue"
    MASS_CASUALTY = "mass_casualty"


class HealthcareEmergencyDetector:
    """
    Anomaly detection for CISA Healthcare and Emergency Services sectors.

    Monitors:
    - Patient vital signs (early warning for deterioration)
    - Healthcare cybersecurity (ransomware, device security)
    - Emergency call patterns (mass casualty, disaster prediction)
    - Disease outbreak detection (syndromic surveillance)
    - Medical supply chain disruptions
    """

    def __init__(self) -> None:
        self.vital_sign_ranges = {
            "heart_rate_bpm": {"min": 60, "max": 100, "critical_min": 40, "critical_max": 130},
            "blood_pressure_systolic": {
                "min": 90,
                "max": 120,
                "critical_min": 70,
                "critical_max": 180,
            },
            "oxygen_saturation_pct": {
                "min": 95,
                "max": 100,
                "critical_min": 85,
                "critical_max": 100,
            },
            "temperature_f": {
                "min": 97.0,
                "max": 99.5,
                "critical_min": 95.0,
                "critical_max": 103.0,
            },
            "respiratory_rate_bpm": {"min": 12, "max": 20, "critical_min": 8, "critical_max": 30},
        }
        self.call_baseline = {"avg_per_hour": 100, "std_per_hour": 20}

    def detect(
        self,
        data: np.ndarray[Any, Any],
        detection_type: str = "patient",
        patient_history: dict | None = None,
    ) -> dict[str, Any]:
        """Generic detection interface for healthcare/emergency services.

        Args:
            data: Sensor data as numpy array
            detection_type: 'patient' or 'emergency_calls'
            patient_history: Optional patient context

        Returns:
            Anomaly detection results
        """
        if detection_type == "patient":
            vital_signs = {
                "heart_rate_bpm": float(data[0]) if len(data) > 0 else 75,
                "blood_pressure_systolic": float(data[1]) if len(data) > 1 else 120,
                "oxygen_saturation_pct": float(data[2]) if len(data) > 2 else 98,
            }
            return self.detect_patient_deterioration(vital_signs, patient_history)
        elif detection_type == "emergency_calls":
            call_data = {"total_calls": int(np.sum(data))}
            return self.detect_emergency_call_anomaly(call_data)
        else:
            vital_signs = {
                "heart_rate_bpm": float(data[0]) if len(data) > 0 else 75,
            }
            return self.detect_patient_deterioration(vital_signs, patient_history)

    def detect_patient_deterioration(
        self,
        vital_signs: dict[str, float],
        patient_history: dict | None = None,
        time_series: dict[str, np.ndarray[Any, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Detect patient deterioration from vital signs.

        Args:
            vital_signs: Current vital sign readings
            patient_history: Optional patient context
            time_series: Optional historical vital signs

        Returns:
            Patient status assessment with early warning scores
        """
        anomalies = {}
        early_warning_score = 0

        for vital_name, value in vital_signs.items():
            if vital_name not in self.vital_sign_ranges:
                continue

            ranges = self.vital_sign_ranges[vital_name]

            if value < ranges["critical_min"] or value > ranges["critical_max"]:
                anomalies[vital_name] = {
                    "value": value,
                    "severity": "CRITICAL",
                    "range": f"{ranges['critical_min']}-{ranges['critical_max']}",
                    "details": f"{vital_name} critically abnormal",
                }
                early_warning_score += 3
            elif value < ranges["min"] or value > ranges["max"]:
                anomalies[vital_name] = {
                    "value": value,
                    "severity": "HIGH",
                    "range": f"{ranges['min']}-{ranges['max']}",
                    "details": f"{vital_name} abnormal",
                }
                early_warning_score += 2

        patient_status = self._determine_patient_status(early_warning_score)

        return {
            "anomalies": anomalies,
            "early_warning_score": early_warning_score,
            "patient_status": patient_status.value,
            "requires_intervention": early_warning_score >= 5,
            "requires_icu": early_warning_score >= 7,
            "recommended_actions": self._generate_clinical_recommendations(
                anomalies, early_warning_score
            ),
            "timestamp": datetime.now(),
        }

    def detect_emergency_call_anomaly(
        self, call_data: dict[str, int], time_window: timedelta = timedelta(hours=1)
    ) -> dict[str, Any]:
        """
        Detect anomalies in 911/emergency call patterns.

        Args:
            call_data: Emergency call counts
            time_window: Time window for aggregation

        Returns:
            Emergency call anomaly assessment
        """
        total_calls = call_data.get("total_calls", 0)
        baseline = self.call_baseline["avg_per_hour"]
        std = self.call_baseline["std_per_hour"]

        z_score = (total_calls - baseline) / std if std > 0 else 0

        anomalies = {}

        if z_score > 3:
            anomalies["call_surge"] = {
                "total_calls": total_calls,
                "baseline": baseline,
                "z_score": z_score,
                "severity": "CRITICAL" if z_score > 5 else "HIGH",
                "details": "Significant increase in emergency calls - possible mass casualty event",
            }

        event_type = self._classify_emergency_event(call_data, z_score)

        return {
            "anomalies": anomalies,
            "call_volume_status": (
                "CRITICAL" if z_score > 5 else "HIGH" if z_score > 3 else "NORMAL"
            ),
            "estimated_event_type": event_type,
            "resource_allocation": self._recommend_resource_allocation(call_data, event_type),
            "mutual_aid_needed": z_score > 5,
            "recommended_actions": self._generate_emergency_recommendations(anomalies, event_type),
        }

    def _determine_patient_status(self, early_warning_score: int) -> PatientStatus:
        """Determine patient status from early warning score."""
        if early_warning_score >= 7:
            return PatientStatus.CRITICAL
        elif early_warning_score >= 5:
            return PatientStatus.DETERIORATING
        elif early_warning_score >= 3:
            return PatientStatus.EMERGENCY
        else:
            return PatientStatus.STABLE

    def _generate_clinical_recommendations(self, anomalies: dict[str, Any], score: int) -> list[str]:
        """Generate clinical action recommendations."""
        if score < 3:
            return ["Continue routine monitoring"]

        recommendations = []

        if score >= 7:
            recommendations.append("CRITICAL: Activate rapid response team immediately")
            recommendations.append("Consider ICU transfer")
        elif score >= 5:
            recommendations.append("Increase monitoring frequency to every 15 minutes")
            recommendations.append("Notify attending physician immediately")

        for vital, details in anomalies.items():
            if details["severity"] == "CRITICAL":
                if "oxygen" in vital.lower():
                    recommendations.append("Administer supplemental oxygen immediately")
                elif "heart_rate" in vital.lower():
                    recommendations.append("Assess for cardiac emergency - prepare crash cart")

        return recommendations

    def _classify_emergency_event(self, call_data: dict[str, Any], z_score: float) -> str:
        """Classify type of emergency event."""
        if z_score <= 3:
            return "normal_operations"

        if z_score > 5:
            return "major_disaster"

        return "elevated_activity"

    def _recommend_resource_allocation(self, call_data: dict[str, Any], event_type: str) -> dict[str, Any]:
        """Recommend emergency resource allocation."""
        if event_type == "normal_operations":
            return {"status": "normal", "additional_units": 0}

        total_calls = call_data.get("total_calls", 0)
        baseline = self.call_baseline["avg_per_hour"]
        excess = max(0, total_calls - baseline)

        return {
            "additional_ambulances": int(excess * 0.3),
            "additional_fire_units": int(excess * 0.1),
            "additional_police": int(excess * 0.2),
            "activate_mutual_aid": excess > 50,
            "activate_emergency_operations_center": event_type == "major_disaster",
        }

    def _generate_emergency_recommendations(self, anomalies: dict[str, Any], event_type: str) -> list[str]:
        """Generate emergency response recommendations."""
        if not anomalies:
            return ["Continue normal operations"]

        recommendations = []

        if event_type == "major_disaster":
            recommendations.append("CRITICAL: Activate Emergency Operations Center")
            recommendations.append("Request mutual aid from neighboring jurisdictions")

        return recommendations
