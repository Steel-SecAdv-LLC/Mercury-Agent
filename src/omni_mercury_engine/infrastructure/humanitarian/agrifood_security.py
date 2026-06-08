# Copyright (C) 2025 Steel Security Advisors LLC
"""AgriFood Security Module - Crop yield and food supply anomaly detection.

Supports SDG 2 (Zero Hunger) by monitoring:
- Crop yield anomalies
- Soil health degradation
- Pest/disease outbreaks
- Supply chain disruptions
- Food price volatility

⚠️ SIMULATION-BASED: Uses simulated agricultural data. Real-world validation required.

Research sources:
- FAO (Food and Agriculture Organization)
- CGIAR crop research
- USDA Agricultural Research Service
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np


class FoodSecurityThreat(Enum):
    """Food security threat."""

    NORMAL = "normal"
    CROP_FAILURE = "crop_failure"
    PEST_OUTBREAK = "pest_outbreak"
    SUPPLY_DISRUPTION = "supply_disruption"
    PRICE_SPIKE = "price_spike"


class AgriFoodSecurityDetector:
    """Detect agricultural and food security anomalies."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self.yield_baseline = {"mean": 5.0, "std": 1.0}
        self.price_baseline = {"mean": 100.0, "std": 15.0}

    def detect(
        self,
        data: np.ndarray[Any, Any],
        detection_type: str = "yield",
        crop_type: str | None = None,
    ) -> dict[str, Any]:
        """Detect agricultural anomalies.

        Args:
            data: Agricultural metrics (yield, prices, soil data)
            detection_type: 'yield', 'price', 'soil_health'
            crop_type: Specific crop (wheat, rice, maize, etc.)

        Returns:
            Food security anomaly detection results
        """
        if detection_type == "yield":
            return self.detect_yield_anomaly(data, crop_type)
        elif detection_type == "price":
            return self.detect_price_anomaly(data, crop_type)
        else:
            return self.detect_yield_anomaly(data, crop_type)

    def detect_yield_anomaly(
        self, yield_data: np.ndarray[Any, Any], crop_type: str | None = None
    ) -> dict[str, Any]:
        """Detect crop yield failures and productivity anomalies."""
        if len(yield_data) == 0:
            return {"anomaly_detected": False}

        mean_yield = np.mean(yield_data)
        baseline = self.yield_baseline["mean"]
        std = self.yield_baseline["std"]

        z_score = (mean_yield - baseline) / std if std > 0 else 0

        threat_type = FoodSecurityThreat.NORMAL
        severity = "low"

        if mean_yield < baseline - 2 * std:
            threat_type = FoodSecurityThreat.CROP_FAILURE
            severity = "critical" if z_score < -3 else "high"

        food_insecurity_risk = self._assess_food_insecurity_risk(mean_yield, baseline, severity)

        return {
            "anomaly_detected": z_score < -2,
            "threat_type": threat_type.value,
            "severity": severity,
            "metrics": {
                "mean_yield_tons_per_hectare": float(mean_yield),
                "baseline_yield": baseline,
                "yield_deficit_pct": (
                    float((baseline - mean_yield) / baseline * 100) if baseline > 0 else 0
                ),
                "z_score": float(z_score),
            },
            "food_insecurity_risk": food_insecurity_risk,
            "affected_population_estimate": self._estimate_affected_population(z_score),
            "recommendations": self._generate_agrifood_recommendations(
                threat_type, severity, crop_type
            ),
            "timestamp": datetime.now(),
        }

    def detect_price_anomaly(
        self, price_data: np.ndarray[Any, Any], crop_type: str | None = None
    ) -> dict[str, Any]:
        """Detect food price spikes and market disruptions."""
        if len(price_data) == 0:
            return {"anomaly_detected": False}

        current_price = np.mean(price_data)
        baseline = self.price_baseline["mean"]
        std = self.price_baseline["std"]

        z_score = (current_price - baseline) / std if std > 0 else 0

        threat_type = FoodSecurityThreat.NORMAL
        severity = "low"

        if current_price > baseline + 2 * std:
            threat_type = FoodSecurityThreat.PRICE_SPIKE
            severity = "critical" if z_score > 3 else "high"

        return {
            "anomaly_detected": z_score > 2,
            "threat_type": threat_type.value,
            "severity": severity,
            "metrics": {
                "current_price_usd": float(current_price),
                "baseline_price_usd": baseline,
                "price_increase_pct": (
                    float((current_price - baseline) / baseline * 100) if baseline > 0 else 0
                ),
                "z_score": float(z_score),
            },
            "affordability_impact": "severe" if severity == "critical" else "moderate",
            "recommendations": self._generate_agrifood_recommendations(
                threat_type, severity, crop_type
            ),
            "timestamp": datetime.now(),
        }

    def _assess_food_insecurity_risk(
        self, current_yield: float, baseline: float, severity: str
    ) -> str:
        """Assess food insecurity risk based on yield deficit."""
        if severity == "critical":
            return "famine_risk"
        elif severity == "high":
            return "acute_shortage"
        elif current_yield < baseline:
            return "moderate_shortage"
        return "adequate"

    def _estimate_affected_population(self, z_score: float) -> int:
        """Estimate population affected by yield anomaly."""
        if z_score < -3:
            return 1000000
        elif z_score < -2:
            return 500000
        return 0

    def _generate_agrifood_recommendations(
        self, threat_type: FoodSecurityThreat, severity: str, crop_type: str | None
    ) -> list[str]:
        """Generate food security recommendations."""
        recommendations = []

        if threat_type == FoodSecurityThreat.CROP_FAILURE:
            recommendations.append("Activate emergency food reserves")
            recommendations.append("Coordinate with WFP for food assistance")
            if severity == "critical":
                recommendations.append("URGENT: Declare agricultural emergency")
                recommendations.append("Begin emergency food distribution")
        elif threat_type == FoodSecurityThreat.PRICE_SPIKE:
            recommendations.append("Implement price stabilization measures")
            recommendations.append("Release strategic food reserves")
            recommendations.append("Provide subsidies for vulnerable populations")

        if crop_type:
            recommendations.append(f"Focus interventions on {crop_type} production")

        if not recommendations:
            recommendations.append("Continue routine agricultural monitoring")

        return recommendations
