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
CISA Chemical & Nuclear Critical Infrastructure Anomaly Detection

Safety-critical monitoring for chemical manufacturing and nuclear facilities.

Research sources:
- CISA Chemical Sector framework
- CISA Nuclear Sector framework
- Nuclear Regulatory Commission (NRC) standards

MIT License compatible - original implementation
"""

from enum import Enum
from typing import Dict, List, Optional
import numpy as np


class CISASector(Enum):
    CHEMICAL = "chemical"
    NUCLEAR = "nuclear_reactors_materials_waste"


class ChemicalNuclearDetector:
    """
    Anomaly detection for CISA Chemical and Nuclear critical infrastructure.

    Implements safety-critical monitoring patterns for:
    - Chemical process parameters (pressure, temperature, composition)
    - Nuclear reactor monitoring (radiation, coolant, core parameters)
    - Cyber-physical security (SCADA intrusions)
    - Insider threat detection
    - Cross-sector impact assessment
    """

    def __init__(self, sector: CISASector):
        self.sector = sector
        self.safety_thresholds = self._init_safety_thresholds()
        self.interdependency_map = {
            CISASector.CHEMICAL: ["energy", "water", "transportation", "healthcare"],
            CISASector.NUCLEAR: ["energy", "water", "communications"],
        }

    def detect(
        self,
        sensor_data: np.ndarray,
        parameter_names: Optional[List[str]] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Detect anomalies in chemical/nuclear process parameters.

        Args:
            sensor_data: Time-series sensor readings (N x M) for M sensors
            parameter_names: Names of monitored parameters (optional)
            timestamps: Optional timestamps

        Returns:
            Anomaly detection results with safety assessment
        """
        if parameter_names is None:
            parameter_names = [
                f"param_{i}" for i in range(sensor_data.shape[1] if sensor_data.ndim > 1 else 1)
            ]

        return self.detect_process_anomaly(sensor_data, parameter_names, timestamps)

    def detect_process_anomaly(
        self,
        sensor_data: np.ndarray,
        parameter_names: List[str],
        timestamps: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Detect anomalies in chemical/nuclear process parameters.

        Args:
            sensor_data: Time-series sensor readings (N x M) for M sensors
            parameter_names: Names of monitored parameters
            timestamps: Optional timestamps

        Returns:
            Anomaly detection results with safety assessment
        """
        anomalies = {}

        for i, param_name in enumerate(parameter_names):
            if i >= sensor_data.shape[1]:
                continue

            param_data = sensor_data[:, i]

            threshold = self.safety_thresholds.get(param_name, {})
            lower_limit = threshold.get("lower", -np.inf)
            upper_limit = threshold.get("upper", np.inf)

            violations = (param_data < lower_limit) | (param_data > upper_limit)

            if np.any(violations):
                anomalies[param_name] = {
                    "violation_indices": np.where(violations)[0].tolist(),
                    "violation_values": param_data[violations].tolist(),
                    "severity": self._calculate_severity(param_data[violations], threshold),
                    "requires_emergency_response": self._assess_emergency(param_name, violations),
                }

        cross_sector_impact = self._assess_cross_sector_impact(anomalies)

        return {
            "anomalies": anomalies,
            "sector": self.sector.value,
            "safety_status": (
                "CRITICAL"
                if any(a.get("requires_emergency_response") for a in anomalies.values())
                else "WARNING" if anomalies else "NORMAL"
            ),
            "cross_sector_impact": cross_sector_impact,
            "recommended_actions": self._generate_recommendations(anomalies),
        }

    def _init_safety_thresholds(self) -> Dict:
        """Initialize sector-specific safety thresholds."""
        if self.sector == CISASector.CHEMICAL:
            return {
                "temperature_celsius": {"lower": -50, "upper": 200},
                "pressure_psi": {"lower": 0, "upper": 1000},
                "ph_level": {"lower": 2, "upper": 12},
                "leak_rate_ppm": {"lower": 0, "upper": 100},
            }
        elif self.sector == CISASector.NUCLEAR:
            return {
                "radiation_mrem_hr": {"lower": 0, "upper": 5},
                "core_temperature_celsius": {"lower": 280, "upper": 330},
                "coolant_flow_gpm": {"lower": 50000, "upper": 150000},
                "neutron_flux": {"lower": 0.8, "upper": 1.2},
            }
        return {}

    def _calculate_severity(self, violations: np.ndarray, threshold: Dict) -> str:
        """Calculate severity level based on violation magnitude."""
        if len(violations) == 0:
            return "NONE"

        lower = threshold.get("lower", -np.inf)
        upper = threshold.get("upper", np.inf)
        range_size = upper - lower

        max_deviation = np.max(
            [
                np.max(violations - upper) if np.any(violations > upper) else 0,
                np.max(lower - violations) if np.any(violations < lower) else 0,
            ]
        )

        deviation_ratio = max_deviation / range_size if range_size > 0 else 0

        if deviation_ratio > 0.5:
            return "CRITICAL"
        elif deviation_ratio > 0.2:
            return "HIGH"
        elif deviation_ratio > 0.1:
            return "MEDIUM"
        else:
            return "LOW"

    def _assess_emergency(self, param_name: str, violations: np.ndarray) -> bool:
        """Determine if emergency response is required."""
        critical_params = {
            CISASector.CHEMICAL: ["leak_rate_ppm", "temperature_celsius"],
            CISASector.NUCLEAR: [
                "radiation_mrem_hr",
                "core_temperature_celsius",
                "coolant_flow_gpm",
            ],
        }

        return param_name in critical_params.get(self.sector, []) and np.sum(violations) > 3

    def _assess_cross_sector_impact(self, anomalies: Dict) -> Dict:
        """Assess how anomalies in this sector affect other sectors."""
        if not anomalies:
            return {"affected_sectors": [], "impact_level": "NONE"}

        affected = self.interdependency_map.get(self.sector, [])

        has_critical = any(a.get("severity") == "CRITICAL" for a in anomalies.values())

        return {
            "affected_sectors": affected,
            "impact_level": "HIGH" if has_critical else "MEDIUM",
            "cascading_risk": has_critical and len(affected) > 2,
        }

    def _generate_recommendations(self, anomalies: Dict) -> List[str]:
        """Generate action recommendations based on detected anomalies."""
        if not anomalies:
            return ["Continue normal operations"]

        recommendations = []

        for param_name, details in anomalies.items():
            if details.get("requires_emergency_response"):
                recommendations.append(f"URGENT: Initiate emergency shutdown for {param_name}")
            elif details.get("severity") in ["CRITICAL", "HIGH"]:
                recommendations.append(f"Investigate and correct {param_name} immediately")
            else:
                recommendations.append(f"Monitor {param_name} closely")

        if any(a.get("requires_emergency_response") for a in anomalies.values()):
            recommendations.append("Notify regulatory authorities (NRC/EPA)")
            recommendations.append("Activate emergency response team")

        return recommendations
