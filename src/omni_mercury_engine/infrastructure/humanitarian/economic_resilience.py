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
Economic Resilience Module - Market and financial anomaly detection

Supports SDG 8 (Decent Work and Economic Growth) by monitoring:
- Market volatility and crashes
- Employment anomalies
- Economic inequality patterns
- Supply chain disruptions
- Financial system stress

⚠️ SIMULATION-BASED: Uses simulated economic data. Real-world validation required.

Research sources:
- IMF economic indicators
- World Bank development metrics
- Federal Reserve economic data

"""

from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np


class EconomicThreat(Enum):
    NORMAL = "normal"
    MARKET_CRASH = "market_crash"
    UNEMPLOYMENT_SPIKE = "unemployment_spike"
    INFLATION_SURGE = "inflation_surge"
    RECESSION_RISK = "recession_risk"


class EconomicResilienceDetector:
    """Detect economic anomalies and systemic risks."""

    def __init__(self) -> None:
        self.market_baseline = {"mean": 100.0, "std": 15.0}
        self.unemployment_baseline = {"mean": 5.0, "std": 2.0}

    def detect(
        self,
        data: np.ndarray[Any, Any],
        detection_type: str = "market",
        economic_context: dict | None = None,
    ) -> dict[str, Any]:
        """
        Detect economic anomalies.

        Args:
            data: Economic indicators (prices, employment, etc.)
            detection_type: 'market', 'employment', 'inflation'
            economic_context: Additional economic context

        Returns:
            Economic resilience anomaly detection results
        """
        if detection_type == "market":
            return self.detect_market_anomaly(data, economic_context)
        elif detection_type == "employment":
            return self.detect_employment_anomaly(data, economic_context)
        else:
            return self.detect_market_anomaly(data, economic_context)

    def detect_market_anomaly(
        self, market_data: np.ndarray[Any, Any], economic_context: dict | None = None
    ) -> dict[str, Any]:
        """Detect market crashes and volatility spikes."""
        if len(market_data) == 0:
            return {"anomaly_detected": False}

        current_value = np.mean(market_data)
        volatility = np.std(market_data)
        baseline_mean = self.market_baseline["mean"]
        baseline_std = self.market_baseline["std"]

        z_score = (current_value - baseline_mean) / baseline_std if baseline_std > 0 else 0

        threat_type = EconomicThreat.NORMAL
        severity = "low"

        if current_value < baseline_mean - 2 * baseline_std:
            threat_type = EconomicThreat.MARKET_CRASH
            severity = "critical" if z_score < -3 else "high"
        elif volatility > baseline_std * 2:
            threat_type = EconomicThreat.MARKET_CRASH
            severity = "high"

        return {
            "anomaly_detected": abs(z_score) > 2 or volatility > baseline_std * 2,
            "threat_type": threat_type.value,
            "severity": severity,
            "metrics": {
                "current_value": float(current_value),
                "volatility": float(volatility),
                "z_score": float(z_score),
                "change_pct": (
                    float((current_value - baseline_mean) / baseline_mean * 100)
                    if baseline_mean > 0
                    else 0
                ),
            },
            "systemic_risk": self._assess_systemic_risk(z_score, volatility, baseline_std),
            "economic_impact_estimate": self._estimate_economic_impact(z_score),
            "recommendations": self._generate_economic_recommendations(threat_type, severity),
            "timestamp": datetime.now(),
        }

    def detect_employment_anomaly(
        self, employment_data: np.ndarray[Any, Any], economic_context: dict | None = None
    ) -> dict[str, Any]:
        """Detect unemployment spikes and labor market stress."""
        if len(employment_data) == 0:
            return {"anomaly_detected": False}

        unemployment_rate = np.mean(employment_data)
        baseline = self.unemployment_baseline["mean"]
        std = self.unemployment_baseline["std"]

        z_score = (unemployment_rate - baseline) / std if std > 0 else 0

        threat_type = EconomicThreat.NORMAL
        severity = "low"

        if unemployment_rate > baseline + 2 * std:
            threat_type = EconomicThreat.UNEMPLOYMENT_SPIKE
            severity = "critical" if z_score > 3 else "high"

        return {
            "anomaly_detected": z_score > 2,
            "threat_type": threat_type.value,
            "severity": severity,
            "metrics": {
                "unemployment_rate_pct": float(unemployment_rate),
                "z_score": float(z_score),
            },
            "labor_market_stress": "severe" if severity == "critical" else "moderate",
            "recommendations": self._generate_economic_recommendations(threat_type, severity),
            "timestamp": datetime.now(),
        }

    def _assess_systemic_risk(self, z_score: float, volatility: float, baseline_std: float) -> str:
        """Assess systemic economic risk."""
        if abs(z_score) > 3 or volatility > baseline_std * 3:
            return "critical"
        elif abs(z_score) > 2 or volatility > baseline_std * 2:
            return "elevated"
        return "low"

    def _estimate_economic_impact(self, z_score: float) -> str:
        """Estimate economic impact in USD."""
        if z_score < -3:
            return "$10B+"
        elif z_score < -2:
            return "$1B-10B"
        return "<$1B"

    def _generate_economic_recommendations(
        self, threat_type: EconomicThreat, severity: str
    ) -> list[str]:
        """Generate economic resilience recommendations."""
        recommendations = []

        if threat_type == EconomicThreat.MARKET_CRASH:
            recommendations.append("Activate market stabilization measures")
            recommendations.append("Coordinate with central bank for liquidity support")
            if severity == "critical":
                recommendations.append("URGENT: Implement emergency circuit breakers")
        elif threat_type == EconomicThreat.UNEMPLOYMENT_SPIKE:
            recommendations.append("Deploy job creation programs")
            recommendations.append("Provide unemployment assistance")
            if severity == "critical":
                recommendations.append("CRITICAL: Emergency economic stimulus required")

        if not recommendations:
            recommendations.append("Continue routine economic monitoring")

        return recommendations
