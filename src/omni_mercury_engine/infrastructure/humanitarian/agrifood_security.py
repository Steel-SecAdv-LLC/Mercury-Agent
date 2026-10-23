# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
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

from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np

from omni_mercury_engine.detectors.provenance import DecisionSource


class FoodSecurityThreat(Enum):
    """Food security threat."""

    NORMAL = "normal"
    CROP_FAILURE = "crop_failure"
    PEST_OUTBREAK = "pest_outbreak"
    SUPPLY_DISRUPTION = "supply_disruption"
    PRICE_SPIKE = "price_spike"


class AgriFoodSecurityDetector:
    """Detect agricultural and food security anomalies."""

    decision_source = (
        DecisionSource(
            kind="published",
            description=(
                "Soil organic carbon below 2.0% marks potentially serious decline in soil quality."
            ),
            citation=(
                "Loveland & Webb 2003, Soil & Tillage Research 70:1-18 "
                "(doi:10.1016/S0167-1987(02)00139-3); bibliographic record checked on Crossref."
            ),
        ),
        DecisionSource(
            kind="experimental",
            description=(
                "The other limits are attributed in the code to FAO and USDA documents that were "
                "not re-read: soil pH outside 5.5-7.5, bulk density above 1.6 g/cm3 (USDA NRCS), "
                "saturation-extract salinity at or above 4 dS/m (USDA Handbook 60, Richards 1954), "
                "500 hoppers/ha locust control density (FAO), cereal stocks-to-use below 17% "
                "(FAO). Yield and price anomalies alarm at a z-score above 2, a hand-set cut."
            ),
        ),
    )

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
            detection_type: 'yield', 'price', 'soil_health', 'pest', 'supply'
            crop_type: Specific crop (wheat, rice, maize, etc.)

        Returns:
            Food security anomaly detection results
        """
        if detection_type == "yield":
            return self.detect_yield_anomaly(data, crop_type)
        elif detection_type == "price":
            return self.detect_price_anomaly(data, crop_type)
        elif detection_type == "soil_health":
            return self.detect_soil_health_anomaly(data, crop_type)
        elif detection_type == "pest":
            return self.detect_pest_outbreak(data, crop_type)
        elif detection_type == "supply":
            return self.detect_supply_disruption(data, crop_type)
        else:
            return self.detect_yield_anomaly(data, crop_type)

    def detect_yield_anomaly(
        self, yield_data: np.ndarray[Any, Any], crop_type: str | None = None
    ) -> dict[str, Any]:
        """Detect crop yield failures and productivity anomalies.

        Raises:
            ValueError: Non-finite or negative yields — a NaN window must
                never read as a normal harvest.
        """
        if len(yield_data) == 0:
            return {"anomaly_detected": False}
        yields = np.asarray(yield_data, dtype=float)
        if not np.all(np.isfinite(yields)) or np.any(yields < 0):
            raise ValueError("yield_data values must be finite and non-negative")

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
            "anomaly_detected": bool(z_score < -2),
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
            "timestamp": datetime.now(UTC),
        }

    def detect_price_anomaly(
        self, price_data: np.ndarray[Any, Any], crop_type: str | None = None
    ) -> dict[str, Any]:
        """Detect food price spikes and market disruptions.

        Raises:
            ValueError: Non-finite or negative prices — a NaN window must
                never read as a calm market.
        """
        if len(price_data) == 0:
            return {"anomaly_detected": False}
        prices = np.asarray(price_data, dtype=float)
        if not np.all(np.isfinite(prices)) or np.any(prices < 0):
            raise ValueError("price_data values must be finite and non-negative")

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
            "anomaly_detected": bool(z_score > 2),
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
            "timestamp": datetime.now(UTC),
        }

    #: Published soil-health thresholds, one per indicator of the
    #: ``[soc_pct, ph, bulk_density_g_cm3, salinity_ds_m]`` vector:
    #: * Soil organic carbon below 2.0% marks potentially serious decline in
    #:   soil quality (Loveland & Webb 2003, Soil Till Res 70:1-18, the
    #:   widely cited 2% SOC threshold; FAO/ITPS Status of the World's Soil
    #:   Resources treats SOC loss as the leading degradation process).
    #: * pH outside 5.5-7.5, the optimal range for most crops (FAO).
    #: * Bulk density above 1.6 g/cm3 restricts root growth for most
    #:   textures (USDA NRCS Soil Quality Indicators: Bulk Density).
    #: * Saturation-extract salinity at or above 4 dS/m defines a saline
    #:   soil (USDA Agriculture Handbook 60, Richards 1954; FAO).
    SOC_CRITICAL_PCT = 2.0
    PH_OPTIMAL_RANGE = (5.5, 7.5)
    BULK_DENSITY_RESTRICTIVE_G_CM3 = 1.6
    SALINITY_SALINE_DS_M = 4.0

    #: FAO Desert Locust Guidelines control threshold: hopper densities
    #: around 500 per hectare trigger control operations; a tenfold
    #: exceedance grades critical.
    LOCUST_CONTROL_DENSITY_PER_HA = 500.0

    #: FAO minimum safe level for the world cereal stocks-to-use ratio,
    #: 17-18% (established after the 1974 World Food Conference; tracked by
    #: FAO/AMIS). Below 17% supply is disrupted; the 17-18% band is tight.
    STOCKS_TO_USE_MIN_SAFE_PCT = 17.0
    STOCKS_TO_USE_TIGHT_PCT = 18.0

    def detect_soil_health_anomaly(
        self, soil_data: np.ndarray[Any, Any], crop_type: str | None = None
    ) -> dict[str, Any]:
        """Soil-degradation detection against published FAO/USDA thresholds.

        Takes the four-indicator vector ``[soc_pct, ph, bulk_density_g_cm3,
        salinity_ds_m]`` (or an ``(N, 4)`` sample array, judged by its
        means) and counts the published thresholds breached (see the class
        constants for the sources). One breach grades medium, two high,
        three or more critical; degraded soil is a crop-failure risk.

        Args:
            soil_data: ``[soc_pct, ph, bulk_density_g_cm3, salinity_ds_m]``
                or ``(N, 4)`` samples of the same.
            crop_type: Specific crop, echoed into recommendations.

        Returns:
            Detection result listing the breached indicators.

        Raises:
            ValueError: Wrong shape or non-finite/negative values.
        """
        if len(soil_data) == 0:
            return {"anomaly_detected": False}
        arr = np.asarray(soil_data, dtype=float)
        if arr.ndim == 1 and arr.size == 4:
            arr = arr.reshape(1, 4)
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError(
                "soil_data takes [soc_pct, ph, bulk_density_g_cm3, salinity_ds_m] "
                f"rows, got shape {np.asarray(soil_data).shape}"
            )
        if not np.all(np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError("soil indicators must be finite and non-negative")

        soc, ph, bulk_density, salinity = (float(v) for v in arr.mean(axis=0))

        breached: list[str] = []
        if soc < self.SOC_CRITICAL_PCT:
            breached.append("soil_organic_carbon_below_2pct")
        if not self.PH_OPTIMAL_RANGE[0] <= ph <= self.PH_OPTIMAL_RANGE[1]:
            breached.append("ph_outside_optimal_range")
        if bulk_density > self.BULK_DENSITY_RESTRICTIVE_G_CM3:
            breached.append("bulk_density_root_restrictive")
        if salinity >= self.SALINITY_SALINE_DS_M:
            breached.append("saline_soil")

        if len(breached) >= 3:
            severity = "critical"
        elif len(breached) == 2:
            severity = "high"
        elif breached:
            severity = "medium"
        else:
            severity = "low"
        threat_type = (
            FoodSecurityThreat.CROP_FAILURE if len(breached) >= 2 else FoodSecurityThreat.NORMAL
        )

        return {
            "anomaly_detected": bool(breached),
            "threat_type": threat_type.value,
            "severity": severity,
            "metrics": {
                "soil_organic_carbon_pct": soc,
                "ph": ph,
                "bulk_density_g_cm3": bulk_density,
                "salinity_ds_m": salinity,
                "thresholds_breached": breached,
            },
            "recommendations": self._generate_agrifood_recommendations(
                threat_type, severity, crop_type
            ),
            "timestamp": datetime.now(UTC),
        }

    def detect_pest_outbreak(
        self, pest_density_per_ha: np.ndarray[Any, Any], crop_type: str | None = None
    ) -> dict[str, Any]:
        """Pest-outbreak detection on the FAO desert-locust control threshold.

        The latest observed density is graded against the FAO Desert Locust
        Guidelines control trigger (:data:`LOCUST_CONTROL_DENSITY_PER_HA`,
        about 500 hoppers/ha); ten times that density grades critical. The
        calibration is the desert-locust one — for other pests, the
        economic-threshold concept applies (Stern et al. 1959, Hilgardia
        29:81-101) and callers supply their pest's threshold by scaling.

        Args:
            pest_density_per_ha: Observed pest densities per hectare, oldest
                first; the latest value is graded.
            crop_type: Specific crop, echoed into recommendations.

        Returns:
            Detection result; ``threat_type`` is ``pest_outbreak`` at or
            above the control threshold.
        """
        if len(pest_density_per_ha) == 0:
            return {"anomaly_detected": False}
        densities = np.asarray(pest_density_per_ha, dtype=float).ravel()
        if not np.all(np.isfinite(densities)) or np.any(densities < 0):
            raise ValueError("pest densities must be finite and non-negative")

        latest = float(densities[-1])
        outbreak = latest >= self.LOCUST_CONTROL_DENSITY_PER_HA
        threat_type = FoodSecurityThreat.PEST_OUTBREAK if outbreak else FoodSecurityThreat.NORMAL
        if latest >= 10 * self.LOCUST_CONTROL_DENSITY_PER_HA:
            severity = "critical"
        elif outbreak:
            severity = "high"
        else:
            severity = "low"

        return {
            "anomaly_detected": bool(outbreak),
            "threat_type": threat_type.value,
            "severity": severity,
            "metrics": {
                "pest_density_per_ha": latest,
                "control_threshold_per_ha": self.LOCUST_CONTROL_DENSITY_PER_HA,
            },
            "recommendations": self._generate_agrifood_recommendations(
                threat_type, severity, crop_type
            ),
            "timestamp": datetime.now(UTC),
        }

    def detect_supply_disruption(
        self, stocks_to_use_pct: np.ndarray[Any, Any], crop_type: str | None = None
    ) -> dict[str, Any]:
        """Supply-disruption detection on the FAO stocks-to-use safe level.

        The latest cereal stocks-to-use ratio is graded against FAO's
        minimum safe level (:data:`STOCKS_TO_USE_MIN_SAFE_PCT`, 17%, with
        the 17-18% band tight): below 17% supply is disrupted (high; below
        14% critical), inside the band it is tight (medium).

        Args:
            stocks_to_use_pct: Stocks-to-use ratios in percent, oldest
                first; the latest value is graded.
            crop_type: Specific crop, echoed into recommendations.

        Returns:
            Detection result; ``threat_type`` is ``supply_disruption`` below
            the minimum safe level.
        """
        if len(stocks_to_use_pct) == 0:
            return {"anomaly_detected": False}
        ratios = np.asarray(stocks_to_use_pct, dtype=float).ravel()
        if not np.all(np.isfinite(ratios)) or np.any(ratios < 0):
            raise ValueError("stocks_to_use_pct values must be finite and non-negative")

        latest = float(ratios[-1])
        disrupted = latest < self.STOCKS_TO_USE_MIN_SAFE_PCT
        tight = latest < self.STOCKS_TO_USE_TIGHT_PCT

        threat_type = (
            FoodSecurityThreat.SUPPLY_DISRUPTION if disrupted else FoodSecurityThreat.NORMAL
        )
        if latest < 14.0:
            severity = "critical"
        elif disrupted:
            severity = "high"
        elif tight:
            severity = "medium"
        else:
            severity = "low"

        return {
            "anomaly_detected": bool(disrupted),
            "threat_type": threat_type.value,
            "severity": severity,
            "metrics": {
                "stocks_to_use_pct": latest,
                "min_safe_pct": self.STOCKS_TO_USE_MIN_SAFE_PCT,
            },
            "recommendations": self._generate_agrifood_recommendations(
                threat_type, severity, crop_type
            ),
            "timestamp": datetime.now(UTC),
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
        elif threat_type == FoodSecurityThreat.PEST_OUTBREAK:
            recommendations.append("Launch coordinated pest control operations")
            recommendations.append("Survey adjacent areas for spread")
            if severity == "critical":
                recommendations.append("CRITICAL: Request regional pest-control assistance (FAO)")
        elif threat_type == FoodSecurityThreat.SUPPLY_DISRUPTION:
            recommendations.append("Release strategic reserves to stabilize supply")
            recommendations.append("Diversify import sources and reduce export restrictions")
            if severity == "critical":
                recommendations.append("CRITICAL: Coordinate international supply response (AMIS)")

        if crop_type:
            recommendations.append(f"Focus interventions on {crop_type} production")

        if not recommendations:
            recommendations.append("Continue routine agricultural monitoring")

        return recommendations
