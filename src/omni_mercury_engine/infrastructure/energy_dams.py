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

from typing import Any


"""
CISA Energy & Dams Critical Infrastructure Anomaly Detection

Power grid stability and dam structural monitoring.

Research sources:
- CISA Energy Sector framework
- CISA Dams Sector framework
- NERC (North American Electric Reliability Corporation) standards

"""

from enum import Enum

import numpy as np


class EnergySubsector(Enum):
    ELECTRICITY = "electricity"
    OIL_GAS = "oil_gas"
    RENEWABLE = "renewable"


class DamType(Enum):
    HYDROELECTRIC = "hydroelectric"
    FLOOD_CONTROL = "flood_control"
    WATER_SUPPLY = "water_supply"
    MULTIPURPOSE = "multipurpose"


class EnergyDamsDetector:
    """
    Anomaly detection for CISA Energy and Dams critical infrastructure.

    Monitors:
    - Power grid stability (frequency, voltage, load)
    - SCADA/ICS security
    - Dam structural integrity
    - Cascading failure risks
    - Renewable energy integration
    """

    def __init__(self, subsector: EnergySubsector | None = None) -> None:
        self.subsector = subsector or EnergySubsector.ELECTRICITY
        self.grid_parameters = {
            "frequency_hz": {"nominal": 60.0, "tolerance": 0.05},
            "voltage_kv": {"nominal": 345.0, "tolerance_pct": 5.0},
            "load_mw": {"min": 0, "max": 100000},
        }
        self.dam_thresholds = {
            "seepage_gpm": {"max": 50},
            "displacement_mm": {"max": 10},
            "water_level_ft": {"min": 100, "max": 500},
        }

    def detect(
        self,
        data: np.ndarray[Any, Any],
        detection_type: str = "grid",
        timestamps: np.ndarray[Any, Any] | None = None,
    ) -> dict[str, Any]:
        """Generic detection interface for energy/dams infrastructure.

        Args:
            data: Sensor data as numpy array
            detection_type: 'grid' or 'dam'
            timestamps: Optional timestamps

        Returns:
            Anomaly detection results
        """
        if detection_type == "grid":
            grid_data = {"frequency_hz": data}
            return self.detect_grid_anomaly(grid_data, timestamps)
        elif detection_type == "dam":
            dam_data = {
                "seepage_gpm": float(data[0]) if len(data) > 0 else 0,
                "water_level_ft": float(data[1]) if len(data) > 1 else 250,
            }
            return self.detect_dam_anomaly(dam_data)
        else:
            grid_data = {"frequency_hz": data}
            return self.detect_grid_anomaly(grid_data, timestamps)

    def detect_grid_anomaly(
        self,
        grid_data: dict[str, np.ndarray[Any, Any]],
        timestamps: np.ndarray[Any, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Detect power grid anomalies.

        Args:
            grid_data: Grid measurements dict
            timestamps: Optional timestamps

        Returns:
            Anomaly detection results with cascading impact assessment
        """
        anomalies = {}

        freq_anomaly = self._detect_frequency_deviation(grid_data.get("frequency_hz"))
        if freq_anomaly:
            anomalies["frequency"] = freq_anomaly

        cascading_risk = self._assess_cascading_impact(anomalies)

        return {
            "anomalies": anomalies,
            "grid_status": self._determine_grid_status(anomalies),
            "cascading_risk": cascading_risk,
            "affected_sectors": self._identify_affected_sectors(cascading_risk),
            "recommended_actions": self._generate_grid_recommendations(anomalies, cascading_risk),
        }

    def detect_dam_anomaly(
        self, dam_data: dict[str, float], dam_type: DamType = DamType.MULTIPURPOSE
    ) -> dict[str, Any]:
        """
        Detect dam structural and operational anomalies.

        Args:
            dam_data: Dam measurements dict
            dam_type: Type of dam

        Returns:
            Anomaly detection results with safety assessment
        """
        anomalies = {}

        seepage = dam_data.get("seepage_gpm", 0)
        if seepage > self.dam_thresholds["seepage_gpm"]["max"]:
            anomalies["seepage"] = {
                "value": seepage,
                "threshold": self.dam_thresholds["seepage_gpm"]["max"],
                "severity": "CRITICAL" if seepage > 100 else "HIGH",
                "details": "Excessive seepage - potential structural failure risk",
            }

        water_level = dam_data.get("water_level_ft", 250)
        thresholds = self.dam_thresholds["water_level_ft"]
        if water_level < thresholds["min"] or water_level > thresholds["max"]:
            anomalies["water_level"] = {
                "value": water_level,
                "min": thresholds["min"],
                "max": thresholds["max"],
                "severity": "HIGH" if water_level > thresholds["max"] else "MEDIUM",
                "details": "Flood risk" if water_level > thresholds["max"] else "Low water supply",
            }

        downstream_impact = self._assess_downstream_impact(anomalies, water_level)

        return {
            "anomalies": anomalies,
            "dam_type": dam_type.value,
            "safety_status": (
                "CRITICAL"
                if any(a.get("severity") == "CRITICAL" for a in anomalies.values())
                else "WARNING" if anomalies else "SAFE"
            ),
            "downstream_impact": downstream_impact,
            "evacuation_recommended": any(
                a.get("severity") == "CRITICAL" for a in anomalies.values()
            )
            and downstream_impact["risk"] == "HIGH",
            "recommended_actions": self._generate_dam_recommendations(anomalies),
        }

    def _detect_frequency_deviation(self, frequency: np.ndarray[Any, Any] | None) -> dict[str, Any] | None:
        """Detect grid frequency deviations."""
        if frequency is None or len(frequency) == 0:
            return None

        freq_params = self.grid_parameters["frequency_hz"]
        assert isinstance(freq_params, dict)  # Type guard
        nominal = float(freq_params["nominal"])
        tolerance = float(freq_params["tolerance"])

        deviations = np.abs(frequency - nominal)
        max_deviation = np.max(deviations)

        if max_deviation > tolerance:
            return {
                "current_hz": float(frequency[-1]),
                "nominal_hz": nominal,
                "max_deviation_hz": float(max_deviation),
                "severity": "CRITICAL" if max_deviation > 2 * tolerance else "HIGH",
                "details": "Grid frequency instability - generation/load imbalance",
            }

        return None

    def _assess_cascading_impact(self, anomalies: dict[str, Any]) -> dict[str, Any]:
        """Assess cascading failure risk to other sectors."""
        if not anomalies:
            return {"risk": "LOW", "probability": 0.0}

        critical_count = sum(1 for a in anomalies.values() if a.get("severity") == "CRITICAL")
        high_count = sum(1 for a in anomalies.values() if a.get("severity") == "HIGH")

        if critical_count >= 1 or high_count >= 2:
            return {
                "risk": "HIGH",
                "probability": min(1.0, (critical_count * 0.4 + high_count * 0.2)),
                "message": (
                    "Energy sector failure will cascade to ALL critical " "infrastructure sectors"
                ),
            }

        return {"risk": "LOW", "probability": 0.1}

    def _identify_affected_sectors(self, cascading_risk: dict[str, Any]) -> list[str]:
        """Identify which sectors would be affected by energy failure."""
        if cascading_risk["risk"] == "HIGH":
            return [
                "chemical",
                "communications",
                "critical_manufacturing",
                "dams",
                "defense_industrial",
                "emergency_services",
                "financial_services",
                "food_agriculture",
                "government_facilities",
                "healthcare",
                "information_technology",
                "nuclear",
                "transportation",
                "water",
            ]

        return []

    def _determine_grid_status(self, anomalies: dict[str, Any]) -> str:
        """Determine overall grid status."""
        if not anomalies:
            return "NORMAL"

        if any(a.get("severity") == "CRITICAL" for a in anomalies.values()):
            return "EMERGENCY"
        elif any(a.get("severity") == "HIGH" for a in anomalies.values()):
            return "ALERT"
        else:
            return "WARNING"

    def _assess_downstream_impact(
        self, anomalies: dict[str, Any], water_level: float
    ) -> dict[str, Any]:
        """Assess impact on downstream communities."""
        if any(a.get("severity") == "CRITICAL" for a in anomalies.values()):
            return {
                "risk": "HIGH",
                "population_at_risk": "HIGH",
                "message": "Dam failure could cause catastrophic flooding",
            }

        return {"risk": "LOW", "population_at_risk": "LOW"}

    def _generate_grid_recommendations(
        self, anomalies: dict[str, Any], cascading_risk: dict[str, Any]
    ) -> list[str]:
        """Generate grid anomaly recommendations."""
        if not anomalies:
            return ["Continue normal operations"]

        recommendations = []

        if "frequency" in anomalies:
            recommendations.append("URGENT: Balance generation and load immediately")

        if cascading_risk["risk"] == "HIGH":
            recommendations.append(
                "CRITICAL: Notify all critical infrastructure sectors of potential outage"
            )

        return recommendations

    def _generate_dam_recommendations(self, anomalies: dict[str, Any]) -> list[str]:
        """Generate dam anomaly recommendations."""
        if not anomalies:
            return ["Continue normal monitoring"]

        recommendations = []

        if "seepage" in anomalies:
            recommendations.append("URGENT: Conduct immediate structural inspection")
            recommendations.append("Notify downstream communities of potential risk")

        if "water_level" in anomalies:
            level_anomaly = anomalies["water_level"]
            if level_anomaly["value"] > level_anomaly["max"]:
                recommendations.append("Increase spillway releases to reduce flood risk")

        return recommendations
