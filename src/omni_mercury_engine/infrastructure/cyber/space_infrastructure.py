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

"""EU Space sector infrastructure anomaly detection.

Monitors satellites, ground stations, and launch facilities for anomalies.
Unique to EU Critical Entities Directive (not in CISA 16 sectors).

Reference: EUR-Lex Directive (EU) 2022/2557
"""

from typing import Any

import numpy as np


class SpaceInfrastructureMonitor:
    """Space infrastructure anomaly detector (EU Critical Entities unique sector).

    Monitors satellites, ground stations, launch facilities for anomalies.
    Unique to EU Critical Entities Directive (not in CISA 16 sectors).
    """

    def __init__(self) -> None:
        """Initialize Space Infrastructure Monitor."""
        self.asset_types = {
            "satellites": {
                "navigation": ["galileo", "gps", "glonass", "beidou"],
                "earth_observation": ["copernicus", "landsat", "sentinel"],
                "communication": ["geostationary", "low_earth_orbit"],
                "weather": ["meteosat", "goes", "noaa"],
            },
            "ground_stations": {
                "tracking": ["antenna_arrays", "telemetry_receivers"],
                "control": ["command_uplink", "mission_control"],
                "data_downlink": ["data_reception", "processing_centers"],
            },
            "launch_facilities": {
                "spaceports": ["kourou", "baikonur", "kennedy", "cape_canaveral"],
                "launch_pads": ["pad_infrastructure", "fueling_systems"],
                "control_centers": ["launch_control", "range_safety"],
            },
        }

    def detect(
        self,
        data: np.ndarray[Any, Any],
        asset_type: str,
        asset_id: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Detect anomalies in space infrastructure using z-score analysis.

        Args:
            data: Sensor/telemetry data from the space asset (numpy array)
            asset_type: 'satellite', 'ground_station', or 'launch_facility'
            asset_id: Specific asset identifier
            context: Additional context (optional)

        Returns:
            Detection results with anomaly score and asset details
        """
        z_scores = self._calculate_z_scores(data)
        anomaly_detected = np.max(z_scores) > 3.0
        anomaly_score = float(np.max(z_scores))
        anomaly_indices = np.where(z_scores > 3.0)[0].tolist()

        anomalies = []
        threat_type = "none"

        if anomaly_detected:
            if asset_type == "satellite":
                if anomaly_score > 5.0:
                    anomalies.append("orbital_deviation")
                    threat_type = "potential_asat_attack_or_debris"
                elif len(anomaly_indices) > 5:
                    anomalies.append("signal_degradation")
                    threat_type = "jamming_or_interference"
                else:
                    anomalies.append("performance_degradation")
                    threat_type = "orbital_perturbation_or_collision_avoidance"

            elif asset_type == "ground_station":
                if len(anomaly_indices) > 10:
                    anomalies.append("security_breach_attempt")
                    threat_type = "cyber_attack_attempted_intrusion"
                elif anomaly_score > 4.5:
                    anomalies.append("network_anomaly")
                    threat_type = "data_exfiltration_or_ddos"
                else:
                    anomalies.append("operational_issue")
                    threat_type = "physical_security_breach"

            elif asset_type == "launch_facility":
                if anomaly_score > 5.0:
                    anomalies.append("critical_safety_hazard")
                    threat_type = "safety_hazard_abort_recommended"
                elif len(anomaly_indices) > 8:
                    anomalies.append("unauthorized_access")
                    threat_type = "sabotage_attempt"
                else:
                    anomalies.append("operational_anomaly")
                    threat_type = "safety_hazard_abort_recommended"

        return {
            "asset_type": asset_type,
            "asset_id": asset_id,
            "anomaly_detected": anomaly_detected,
            "anomaly_score": anomaly_score,
            "confidence": 0.85,
            "threat_type": threat_type,
            "anomalies": anomalies,
            "severity": (
                "critical"
                if anomaly_score > 5.0
                else "high" if anomaly_score > 4.0 else "medium" if anomaly_detected else "low"
            ),
            "details": {
                "anomaly_indices": anomaly_indices,
                "z_score_max": anomaly_score,
            },
            "recommendations": self._generate_space_recommendations(threat_type),
        }

    def _calculate_z_scores(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Calculate z-scores for anomaly detection."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        z_scores = np.abs((data - mean) / std)
        return np.max(z_scores, axis=1) if data.ndim > 1 else z_scores.flatten()

    def _generate_space_recommendations(self, threat_type: str) -> list[str]:
        """Generate recommendations based on threat type."""
        recommendations_map = {
            "jamming_or_interference": [
                "Switch to backup frequencies",
                "Activate anti-jamming protocols",
                "Notify space situational awareness center",
            ],
            "orbital_perturbation_or_collision_avoidance": [
                "Recalculate orbital trajectory",
                "Check for space debris or nearby objects",
                "Prepare collision avoidance maneuver if needed",
            ],
            "potential_asat_attack_or_debris": [
                "CRITICAL: Notify national security authorities",
                "Activate satellite defense protocols",
                "Document all telemetry for incident analysis",
            ],
            "cyber_attack_attempted_intrusion": [
                "Isolate affected systems immediately",
                "Review access logs and authentication records",
                "Engage cybersecurity incident response team",
            ],
            "physical_security_breach": [
                "IMMEDIATE: Lockdown facility",
                "Notify security personnel and law enforcement",
                "Review surveillance footage",
            ],
            "data_exfiltration_or_ddos": [
                "Analyze network traffic patterns",
                "Implement rate limiting and traffic filtering",
                "Check for compromised credentials",
            ],
            "safety_hazard_abort_recommended": [
                "CRITICAL: Initiate abort sequence",
                "Evacuate non-essential personnel",
                "Engage safety systems",
            ],
            "sabotage_attempt": [
                "IMMEDIATE: Halt all operations",
                "Notify law enforcement and counterintelligence",
                "Conduct thorough security sweep",
            ],
        }
        return recommendations_map.get(threat_type, ["Monitor situation and log for analysis"])
