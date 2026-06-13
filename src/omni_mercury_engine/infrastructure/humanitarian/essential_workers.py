# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Essential Critical Infrastructure Workers monitoring.

Monitors workforce continuity for 8 essential worker categories
with emphasis on survivor-first ethical principles.

Reference: CISA Essential Workers Advisory List
"""

from __future__ import annotations

from typing import Any


class EssentialWorkersMonitor:
    """Essential critical infrastructure workers anomaly detector.

    Monitors workforce continuity for 8 essential worker categories with emphasis on survivor-first
    ethical principles.
    """

    def __init__(self, ethical_config: dict[str, float] | None = None) -> None:
        """Initialize Essential Workers Monitor.

        Args:
            ethical_config: Ethical scalar configuration
        """
        self.worker_categories = {
            "health_social_care": {
                "roles": ["doctors", "nurses", "paramedics", "care_workers", "pharmacists"],
                "baseline_capacity": 1.0,
                "critical_threshold": 0.70,
            },
            "education_childcare": {
                "roles": ["teachers", "childcare_workers", "support_staff", "administrators"],
                "baseline_capacity": 1.0,
                "critical_threshold": 0.75,
            },
            "public_services": {
                "roles": [
                    "justice",
                    "religious",
                    "journalists",
                    "civil_servants",
                    "postal_workers",
                ],
                "baseline_capacity": 1.0,
                "critical_threshold": 0.80,
            },
            "government": {
                "roles": ["administrators", "emergency_coordinators", "public_officials"],
                "baseline_capacity": 1.0,
                "critical_threshold": 0.75,
            },
            "food_goods": {
                "roles": ["farmers", "food_processors", "grocery_workers", "delivery_drivers"],
                "baseline_capacity": 1.0,
                "critical_threshold": 0.65,
            },
            "safety_security": {
                "roles": ["police", "firefighters", "military", "border_patrol", "security"],
                "baseline_capacity": 1.0,
                "critical_threshold": 0.80,
            },
            "transport_border": {
                "roles": [
                    "drivers",
                    "pilots",
                    "air_traffic_controllers",
                    "port_workers",
                    "transit",
                ],
                "baseline_capacity": 1.0,
                "critical_threshold": 0.70,
            },
            "utilities_finance": {
                "roles": ["power_workers", "water_treatment", "telecom", "banking", "it_support"],
                "baseline_capacity": 1.0,
                "critical_threshold": 0.75,
            },
        }

        self.ethical_scalars = {
            "survivor_first": (
                ethical_config.get("survivor_first", 0.95) if ethical_config else 0.95
            ),
            "compassion": ethical_config.get("compassion", 0.90) if ethical_config else 0.90,
            "omnibenevolent": (
                ethical_config.get("omnibenevolent", 0.85) if ethical_config else 0.85
            ),
        }

    def detect(self, data: dict[str, Any], category: str) -> dict[str, Any]:
        """Detect workforce anomalies for a worker category.

        Args:
            data: Worker availability, absenteeism, skills data
            category: Worker category (e.g., 'health_social_care')

        Returns:
            Detection results with capacity status, recommendations
        """
        if category not in self.worker_categories:
            raise ValueError(f"Unknown worker category: {category}")

        current_capacity = data.get("current_capacity", 1.0)
        absenteeism_rate = data.get("absenteeism_rate", 0.03)
        skill_shortage_critical = data.get("skill_shortage", False)

        category_config = self.worker_categories[category]
        critical_threshold = category_config["critical_threshold"]

        capacity_anomaly = current_capacity < critical_threshold
        absenteeism_anomaly = absenteeism_rate > 0.15

        anomaly_detected = capacity_anomaly or absenteeism_anomaly or skill_shortage_critical

        severity = (
            "critical" if current_capacity < 0.60 else "high" if capacity_anomaly else "medium"
        )

        return {
            "category": category,
            "anomaly_detected": anomaly_detected,
            "current_capacity": current_capacity,
            "critical_threshold": critical_threshold,
            "absenteeism_rate": absenteeism_rate,
            "skill_shortage": skill_shortage_critical,
            "severity": severity,
            "ethical_priority": self._calculate_ethical_priority(category, current_capacity),
            "recommendations": self._generate_workforce_recommendations(
                category, current_capacity, absenteeism_anomaly, skill_shortage_critical
            ),
        }

    def model_crisis_scenario(self, scenario_type: str) -> dict[str, Any]:
        """Model workforce impacts under crisis scenarios.

        Args:
            scenario_type: 'pandemic', 'natural_disaster', 'cyber_attack', 'civil_unrest'

        Returns:
            Predicted capacity impacts across all worker categories
        """
        impact_multipliers = {
            "pandemic": {
                "health_social_care": 0.70,
                "education_childcare": 0.60,
                "public_services": 0.80,
                "government": 0.85,
                "food_goods": 0.85,
                "safety_security": 0.90,
                "transport_border": 0.75,
                "utilities_finance": 0.85,
            },
            "natural_disaster": {
                "health_social_care": 0.75,
                "education_childcare": 0.50,
                "public_services": 0.70,
                "government": 0.80,
                "food_goods": 0.70,
                "safety_security": 0.95,
                "transport_border": 0.60,
                "utilities_finance": 0.75,
            },
            "cyber_attack": {
                "health_social_care": 0.85,
                "education_childcare": 0.90,
                "public_services": 0.75,
                "government": 0.70,
                "food_goods": 0.80,
                "safety_security": 0.95,
                "transport_border": 0.80,
                "utilities_finance": 0.60,
            },
        }

        multipliers = impact_multipliers.get(
            scenario_type, dict.fromkeys(self.worker_categories, 0.85)
        )

        predictions = {}
        critical_categories = []

        for category, config in self.worker_categories.items():
            baseline_val = config["baseline_capacity"]
            baseline = float(baseline_val) if isinstance(baseline_val, (int, float)) else 1.0
            multiplier_val = multipliers.get(category, 0.85)
            multiplier = float(multiplier_val) if isinstance(multiplier_val, (int, float)) else 0.85
            predicted_capacity = baseline * multiplier
            threshold_val = config["critical_threshold"]
            threshold = float(threshold_val) if isinstance(threshold_val, (int, float)) else 0.75
            critical = predicted_capacity < threshold

            if critical:
                critical_categories.append(category)

            predictions[category] = {
                "predicted_capacity": predicted_capacity,
                "baseline": config["baseline_capacity"],
                "critical_threshold": config["critical_threshold"],
                "will_be_critical": critical,
            }

        return {
            "scenario_type": scenario_type,
            "predictions": predictions,
            "critical_categories": critical_categories,
            "overall_resilience_score": 1.0
            - (len(critical_categories) / len(self.worker_categories)),
        }

    def _calculate_ethical_priority(self, category: str, current_capacity: float) -> float:
        """Calculate ethical priority score based on survivor-first principles."""
        category_priorities = {
            "health_social_care": 0.95,
            "safety_security": 0.92,
            "food_goods": 0.90,
            "utilities_finance": 0.88,
            "government": 0.85,
            "transport_border": 0.82,
            "education_childcare": 0.80,
            "public_services": 0.78,
        }

        base_priority = category_priorities.get(category, 0.75)
        capacity_factor = 1.0 if current_capacity >= 0.80 else (1.0 + (0.80 - current_capacity))

        return min(1.0, base_priority * capacity_factor * self.ethical_scalars["survivor_first"])

    def _generate_workforce_recommendations(
        self, category: str, capacity: float, absenteeism_high: bool, skill_shortage: bool
    ) -> list[Any]:
        """Generate recommendations for workforce issues."""
        recommendations = []

        if capacity < 0.60:
            recommendations.append("CRITICAL: Activate emergency workforce mobilization")
            recommendations.append("Request mutual aid from neighboring jurisdictions")
            recommendations.append("Deploy reserve/auxiliary personnel immediately")
        elif capacity < 0.75:
            recommendations.append("Activate contingency staffing plans")
            recommendations.append("Prioritize essential functions, defer non-critical tasks")

        if absenteeism_high:
            recommendations.append("Investigate absenteeism causes (illness, childcare, fear)")
            recommendations.append(
                "Provide support services (childcare, mental health, hazard pay)"
            )

        if skill_shortage:
            recommendations.append("Expedite training for cross-functional staff")
            recommendations.append("Recruit retired professionals for temporary service")
            recommendations.append("Simplify procedures to reduce skill requirements where safe")

        if not recommendations:
            recommendations.append("Monitor workforce metrics, maintain situational awareness")

        return recommendations
