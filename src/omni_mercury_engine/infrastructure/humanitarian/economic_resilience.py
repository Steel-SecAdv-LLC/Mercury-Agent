# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Economic Resilience Module - Market and financial anomaly detection.

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

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np


class EconomicThreat(Enum):
    """Economic threat."""

    NORMAL = "normal"
    MARKET_CRASH = "market_crash"
    UNEMPLOYMENT_SPIKE = "unemployment_spike"
    INFLATION_SURGE = "inflation_surge"
    RECESSION_RISK = "recession_risk"


class EconomicResilienceDetector:
    """Detect economic anomalies and systemic risks."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self.market_baseline = {"mean": 100.0, "std": 15.0}
        self.unemployment_baseline = {"mean": 5.0, "std": 2.0}

    def detect(
        self,
        data: np.ndarray[Any, Any],
        detection_type: str = "market",
        economic_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Detect economic anomalies.

        Args:
            data: Economic indicators (prices, employment, etc.)
            detection_type: 'market', 'employment', 'inflation', 'gdp'
            economic_context: Additional economic context

        Returns:
            Economic resilience anomaly detection results
        """
        if detection_type == "market":
            return self.detect_market_anomaly(data, economic_context)
        elif detection_type == "employment":
            return self.detect_employment_anomaly(data, economic_context)
        elif detection_type == "inflation":
            return self.detect_inflation_anomaly(data, economic_context)
        elif detection_type == "gdp":
            return self.detect_recession_risk(data, economic_context)
        else:
            return self.detect_market_anomaly(data, economic_context)

    def detect_market_anomaly(
        self, market_data: np.ndarray[Any, Any], economic_context: dict[str, Any] | None = None
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
            "anomaly_detected": bool(abs(z_score) > 2 or volatility > baseline_std * 2),
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
            "timestamp": datetime.now(UTC),
        }

    def detect_employment_anomaly(
        self, employment_data: np.ndarray[Any, Any], economic_context: dict[str, Any] | None = None
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
            "anomaly_detected": bool(z_score > 2),
            "threat_type": threat_type.value,
            "severity": severity,
            "metrics": {
                "unemployment_rate_pct": float(unemployment_rate),
                "z_score": float(z_score),
            },
            "labor_market_stress": "severe" if severity == "critical" else "moderate",
            "recommendations": self._generate_economic_recommendations(threat_type, severity),
            "timestamp": datetime.now(UTC),
        }

    #: Published inflation severity thresholds (annual CPI inflation, %):
    #: * 40: the inflation-crisis threshold of Bruno & Easterly (1998),
    #:   J Monet Econ 41:3-26 ("Inflation crises and long-run growth").
    #: * 15: the lower edge of the "moderate inflation" band of Dornbusch &
    #:   Fischer (1993), World Bank Econ Rev 7:1-44 (15-30%/yr).
    #: * 10: double-digit inflation, the conventional surge threshold the
    #:   two papers build on.
    INFLATION_CRISIS_PCT = 40.0
    INFLATION_MODERATE_BAND_PCT = 15.0
    INFLATION_SURGE_PCT = 10.0

    def detect_inflation_anomaly(
        self, inflation_data: np.ndarray[Any, Any], economic_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Price-index surge detection on published inflation thresholds.

        The latest annual CPI inflation rate is graded against the
        literature's thresholds (see the class constants): a surge at
        double-digit inflation (>= 10%), the Dornbusch-Fischer moderate-
        inflation band at >= 15%, and the Bruno-Easterly crisis threshold at
        >= 40%.

        Args:
            inflation_data: Annual (year-over-year) CPI inflation rates in
                percent, oldest first; the latest value is graded.
            economic_context: Unused; kept for signature parity.

        Returns:
            Detection result; ``threat_type`` is ``inflation_surge`` from
            the 10% threshold upward.
        """
        if len(inflation_data) == 0:
            return {"anomaly_detected": False}
        rates = np.asarray(inflation_data, dtype=float).ravel()
        if not np.all(np.isfinite(rates)):
            raise ValueError("inflation_data contains non-finite values")

        latest = float(rates[-1])
        acceleration = float(latest - rates[-2]) if rates.size > 1 else 0.0

        threat_type = EconomicThreat.NORMAL
        severity = "low"
        if latest >= self.INFLATION_CRISIS_PCT:
            threat_type = EconomicThreat.INFLATION_SURGE
            severity = "critical"
        elif latest >= self.INFLATION_MODERATE_BAND_PCT:
            threat_type = EconomicThreat.INFLATION_SURGE
            severity = "high"
        elif latest >= self.INFLATION_SURGE_PCT:
            threat_type = EconomicThreat.INFLATION_SURGE
            severity = "medium"

        return {
            "anomaly_detected": bool(threat_type is EconomicThreat.INFLATION_SURGE),
            "threat_type": threat_type.value,
            "severity": severity,
            "metrics": {
                "latest_inflation_pct": latest,
                "acceleration_pp": acceleration,
                "surge_threshold_pct": self.INFLATION_SURGE_PCT,
                "crisis_threshold_pct": self.INFLATION_CRISIS_PCT,
            },
            "recommendations": self._generate_economic_recommendations(threat_type, severity),
            "timestamp": datetime.now(UTC),
        }

    def detect_recession_risk(
        self, gdp_growth_data: np.ndarray[Any, Any], economic_context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Recession-risk detection by the technical-recession rule.

        Flags :attr:`EconomicThreat.RECESSION_RISK` on two consecutive
        quarters of negative real GDP growth — the published
        "technical recession" rule of thumb (Julius Shiskin, 1974; the NBER
        Business Cycle Dating Committee's definition differs, which this
        output names in its metrics). Four or more consecutive negative
        quarters grade critical.

        Args:
            gdp_growth_data: Quarterly real GDP growth rates in percent,
                oldest first.
            economic_context: Unused; kept for signature parity.

        Returns:
            Detection result with the current negative-quarter streak.
        """
        if len(gdp_growth_data) == 0:
            return {"anomaly_detected": False}
        growth = np.asarray(gdp_growth_data, dtype=float).ravel()
        if not np.all(np.isfinite(growth)):
            raise ValueError("gdp_growth_data contains non-finite values")

        streak = 0
        for value in growth[::-1]:
            if value < 0:
                streak += 1
            else:
                break

        technical_recession = streak >= 2
        threat_type = (
            EconomicThreat.RECESSION_RISK if technical_recession else (EconomicThreat.NORMAL)
        )
        if streak >= 4:
            severity = "critical"
        elif technical_recession:
            severity = "high"
        else:
            severity = "low"

        return {
            "anomaly_detected": bool(technical_recession),
            "threat_type": threat_type.value,
            "severity": severity,
            "metrics": {
                "latest_growth_pct": float(growth[-1]),
                "negative_quarter_streak": streak,
                "rule": "two consecutive negative quarters (Shiskin 1974); "
                "NBER dating uses broader indicators",
            },
            "recommendations": self._generate_economic_recommendations(threat_type, severity),
            "timestamp": datetime.now(UTC),
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
        elif threat_type == EconomicThreat.INFLATION_SURGE:
            recommendations.append("Tighten monetary policy coordination with the central bank")
            recommendations.append("Protect purchasing power of vulnerable households")
            if severity == "critical":
                recommendations.append(
                    "CRITICAL: Inflation above the crisis threshold — stabilization program needed"
                )
        elif threat_type == EconomicThreat.RECESSION_RISK:
            recommendations.append("Prepare counter-cyclical fiscal measures")
            recommendations.append("Expand social safety nets ahead of labor-market impact")
            if severity == "critical":
                recommendations.append("CRITICAL: Prolonged contraction — coordinate full response")

        if not recommendations:
            recommendations.append("Continue routine economic monitoring")

        return recommendations
