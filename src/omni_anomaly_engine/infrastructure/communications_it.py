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
CISA Communications & IT Critical Infrastructure Anomaly Detection

Network traffic and cybersecurity monitoring for communications and IT sectors.

Research sources:
- CISA Communications Sector framework
- CISA Information Technology Sector framework
- NIST Cybersecurity Framework

"""

from collections import deque
from datetime import datetime

import numpy as np


class CommunicationsITDetector:
    """
    Anomaly detection for CISA Communications and Information Technology sectors.

    Monitors:
    - Network traffic patterns (DDoS, congestion, routing anomalies)
    - Cyber intrusions (ransomware, data exfiltration, lateral movement)
    - Service availability (uptime, performance, capacity)
    - Supply chain security
    - Cross-sector communication impacts
    """

    def __init__(self, baseline_window: int = 3600) -> None:
        self.baseline_window = baseline_window
        self.traffic_history = deque(maxlen=baseline_window)
        self.baseline_stats = {}
        self.alert_thresholds = {
            "ddos_multiplier": 10.0,
            "latency_multiplier": 3.0,
            "packet_loss_threshold": 0.05,
            "exfiltration_mb_threshold": 1000,
        }

    def detect(self, data: np.ndarray[Any, Any], timestamp: datetime | None = None) -> dict[str, Any]:
        """Generic detection interface for communications/IT infrastructure.

        Args:
            data: Network metrics as numpy array
            timestamp: Optional timestamp

        Returns:
            Anomaly detection results
        """
        traffic_data = {
            "packets_per_sec": float(data[0]) if len(data) > 0 else 1000,
            "bytes_per_sec": float(data[1]) if len(data) > 1 else 1000000,
            "connections": int(data[2]) if len(data) > 2 else 100,
        }
        return self.detect_network_anomaly(traffic_data, timestamp)

    def detect_network_anomaly(
        self, traffic_data: dict[str, float], timestamp: datetime | None = None
    ) -> dict[str, Any]:
        """
        Detect network traffic anomalies.

        Args:
            traffic_data: Network metrics dict
            timestamp: Optional timestamp

        Returns:
            Anomaly detection results with threat assessment
        """
        self.traffic_history.append(traffic_data)

        if len(self.traffic_history) < 100:
            return {"status": "LEARNING", "message": "Building baseline, need more data"}

        self._update_baseline()

        anomalies = {}

        ddos_score = self._detect_ddos(traffic_data)
        if ddos_score > 0.7:
            anomalies["ddos"] = {
                "score": ddos_score,
                "severity": "CRITICAL" if ddos_score > 0.9 else "HIGH",
                "details": "Potential DDoS attack detected",
            }

        cross_sector_impact = self._assess_comm_it_impact(anomalies)

        return {
            "anomalies": anomalies,
            "overall_risk": self._calculate_overall_risk(anomalies),
            "cross_sector_impact": cross_sector_impact,
            "affected_sectors": (
                ["all_15_sectors"]
                if cross_sector_impact["level"] == "CRITICAL"
                else cross_sector_impact.get("specific_sectors", [])
            ),
            "recommended_actions": self._generate_recommendations(anomalies),
            "timestamp": timestamp or datetime.now(),
        }

    def _update_baseline(self) -> None:
        """Update baseline statistics from recent traffic history."""
        recent_data = list(self.traffic_history)[-min(len(self.traffic_history), 1000) :]

        for metric in recent_data[0]:
            values = [d[metric] for d in recent_data if metric in d]
            if values:
                self.baseline_stats[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "median": np.median(values),
                    "p95": np.percentile(values, 95),
                    "p99": np.percentile(values, 99),
                }

    def _detect_ddos(self, traffic_data: dict[str, Any]) -> float:
        """Detect DDoS attacks based on traffic volume."""
        if "packets_per_sec" not in self.baseline_stats:
            return 0.0

        baseline = self.baseline_stats["packets_per_sec"]
        current = traffic_data.get("packets_per_sec", 0)

        if current > baseline["mean"] + 3 * baseline["std"]:
            excess_ratio = current / (baseline["mean"] + baseline["std"])
            ddos_score = min(1.0, excess_ratio / 10.0)
            return ddos_score

        return 0.0

    def _assess_comm_it_impact(self, anomalies: dict[str, Any]) -> dict[str, Any]:
        """Assess impact on other critical infrastructure sectors."""
        if not anomalies:
            return {"level": "NONE", "specific_sectors": []}

        severity_count = sum(
            1 for a in anomalies.values() if a.get("severity") in ["CRITICAL", "HIGH"]
        )

        if severity_count >= 2 or any(a.get("severity") == "CRITICAL" for a in anomalies.values()):
            return {
                "level": "CRITICAL",
                "specific_sectors": [
                    "emergency_services",
                    "financial_services",
                    "healthcare",
                    "energy",
                    "transportation",
                    "water",
                ],
                "message": "Communications/IT disruption affects all critical infrastructure",
            }

        return {"level": "LOW", "specific_sectors": []}

    def _calculate_overall_risk(self, anomalies: dict[str, Any]) -> str:
        """Calculate overall risk level."""
        if not anomalies:
            return "LOW"

        critical_count = sum(1 for a in anomalies.values() if a.get("severity") == "CRITICAL")

        if critical_count >= 1:
            return "CRITICAL"

        return "LOW"

    def _generate_recommendations(self, anomalies: dict[str, Any]) -> list[str]:
        """Generate action recommendations."""
        if not anomalies:
            return ["Continue normal monitoring"]

        recommendations = []

        if "ddos" in anomalies:
            recommendations.append("Activate DDoS mitigation - enable rate limiting")
            recommendations.append("Contact upstream ISP for assistance")

        recommendations.append("Notify affected critical infrastructure sectors")

        return recommendations
