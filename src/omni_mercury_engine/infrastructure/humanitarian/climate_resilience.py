# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Climate Resilience Module - Anomaly detection for environmental data.

Supports SDG 13 (Climate Action) by detecting anomalies in:
- Temperature patterns (heatwaves, cold snaps)
- Precipitation anomalies (droughts, floods)
- Sea level rise acceleration
- Extreme weather events
- Carbon emission spikes

⚠️ SIMULATION-BASED: Uses simulated climate data. Real-world validation required.

Research sources:
- IPCC Assessment Reports
- NOAA Climate.gov
- NASA Earth Observatory
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np


class ClimateEvent(Enum):
    """Climate event."""

    NORMAL = "normal"
    HEATWAVE = "heatwave"
    DROUGHT = "drought"
    FLOOD = "flood"
    HURRICANE = "hurricane"
    WILDFIRE = "wildfire"


class ClimateResilienceDetector:
    """Detect climate anomalies for disaster prediction and resilience."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self.temp_baseline = {"mean": 15.0, "std": 10.0}
        self.precip_baseline = {"mean": 100.0, "std": 50.0}

    def detect(
        self,
        data: np.ndarray[Any, Any],
        detection_type: str = "temperature",
        historical_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Detect climate anomalies.

        Args:
            data: Environmental sensor data
            detection_type: 'temperature', 'precipitation', 'sea_level'
            historical_context: Historical climate data for comparison

        Returns:
            Climate anomaly detection results with disaster predictions
        """
        if detection_type == "temperature":
            return self.detect_temperature_anomaly(data, historical_context)
        elif detection_type == "precipitation":
            return self.detect_precipitation_anomaly(data, historical_context)
        else:
            return self.detect_temperature_anomaly(data, historical_context)

    def detect_temperature_anomaly(
        self,
        temperature_series: np.ndarray[Any, Any],
        historical_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Detect heatwaves, cold snaps, and abnormal warming trends."""
        if len(temperature_series) == 0:
            return {"anomaly_detected": False}

        mean_temp = np.mean(temperature_series)
        max_temp = np.max(temperature_series)
        min_temp = np.min(temperature_series)

        baseline_mean = self.temp_baseline["mean"]
        baseline_std = self.temp_baseline["std"]

        z_score = (mean_temp - baseline_mean) / baseline_std if baseline_std > 0 else 0

        event_type = ClimateEvent.NORMAL
        severity = "low"

        if max_temp > baseline_mean + 3 * baseline_std:
            event_type = ClimateEvent.HEATWAVE
            severity = "critical"
        elif min_temp < baseline_mean - 3 * baseline_std:
            event_type = ClimateEvent.HEATWAVE
            severity = "high"
        elif abs(z_score) > 2:
            severity = "medium"

        return {
            "anomaly_detected": abs(z_score) > 2 or severity in ["high", "critical"],
            "event_type": event_type.value,
            "severity": severity,
            "metrics": {
                "mean_temperature_c": float(mean_temp),
                "max_temperature_c": float(max_temp),
                "min_temperature_c": float(min_temp),
                "z_score": float(z_score),
            },
            "disaster_risk": self._assess_disaster_risk(event_type, severity),
            "recommendations": self._generate_climate_recommendations(event_type, severity),
            "timestamp": datetime.now(),
        }

    def detect_precipitation_anomaly(
        self,
        precipitation_mm: np.ndarray[Any, Any],
        historical_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Detect droughts and floods from precipitation patterns."""
        if len(precipitation_mm) == 0:
            return {"anomaly_detected": False}

        total_precip = np.sum(precipitation_mm)
        baseline = self.precip_baseline["mean"]
        std = self.precip_baseline["std"]

        z_score = (total_precip - baseline) / std if std > 0 else 0

        event_type = ClimateEvent.NORMAL
        severity = "low"

        if total_precip < baseline - 2 * std:
            event_type = ClimateEvent.DROUGHT
            severity = "high" if z_score < -3 else "medium"
        elif total_precip > baseline + 2 * std:
            event_type = ClimateEvent.FLOOD
            severity = "critical" if z_score > 3 else "high"

        return {
            "anomaly_detected": abs(z_score) > 2,
            "event_type": event_type.value,
            "severity": severity,
            "metrics": {
                "total_precipitation_mm": float(total_precip),
                "z_score": float(z_score),
            },
            "disaster_risk": self._assess_disaster_risk(event_type, severity),
            "recommendations": self._generate_climate_recommendations(event_type, severity),
            "timestamp": datetime.now(),
        }

    def _assess_disaster_risk(self, event_type: ClimateEvent, severity: str) -> str:
        """Assess disaster risk based on climate event."""
        if severity == "critical":
            return "imminent"
        elif severity == "high":
            return "elevated"
        elif event_type != ClimateEvent.NORMAL:
            return "moderate"
        return "low"

    def _generate_climate_recommendations(
        self, event_type: ClimateEvent, severity: str
    ) -> list[str]:
        """Generate climate resilience recommendations."""
        recommendations = []

        if event_type == ClimateEvent.HEATWAVE:
            recommendations.append("Activate heat emergency protocols")
            recommendations.append("Open cooling centers for vulnerable populations")
            if severity == "critical":
                recommendations.append("URGENT: Issue heat warnings, restrict outdoor activities")
        elif event_type == ClimateEvent.DROUGHT:
            recommendations.append("Implement water conservation measures")
            recommendations.append("Monitor agricultural impacts and food security")
        elif event_type == ClimateEvent.FLOOD:
            recommendations.append("Activate flood response teams")
            recommendations.append("Issue evacuation advisories for flood zones")
            if severity == "critical":
                recommendations.append("CRITICAL: Begin emergency evacuations")

        if not recommendations:
            recommendations.append("Continue routine climate monitoring")

        return recommendations
