"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""World Bank economic sectors anomaly detection.

Monitors 21 ISIC Rev 4 economic sectors for anomalies with focus
on sustainable development and regenerative economic patterns.

Reference: UN Statistics Division - ISIC Rev 4
"""

from typing import Any

import numpy as np


class WorldBankSectorsMonitor:
    """
    World Bank economic sectors anomaly detector.

    Monitors 21 ISIC Rev 4 economic sectors for anomalies with focus on sustainable development and
    regenerative economic patterns.
    """

    def __init__(self, regenerative_config: dict[str, Any] | None = None) -> None:
        """
        Initialize World Bank Sectors Monitor.

        Args:
            regenerative_config: Regenerative architecture configuration
        """
        self.isic_sectors: dict[str, dict[str, Any]] = {
            "A": {"name": "Agriculture, forestry and fishing", "sdg_priority": 0.90},
            "B": {"name": "Mining and quarrying", "sdg_priority": 0.60},
            "C": {"name": "Manufacturing", "sdg_priority": 0.75},
            "D": {
                "name": "Electricity, gas, steam and air conditioning supply",
                "sdg_priority": 0.95,
            },
            "E": {"name": "Water supply; sewerage, waste management", "sdg_priority": 0.95},
            "F": {"name": "Construction", "sdg_priority": 0.70},
            "G": {"name": "Wholesale and retail trade", "sdg_priority": 0.70},
            "H": {"name": "Transportation and storage", "sdg_priority": 0.75},
            "I": {"name": "Accommodation and food service", "sdg_priority": 0.65},
            "J": {"name": "Information and communication", "sdg_priority": 0.85},
            "K": {"name": "Financial and insurance activities", "sdg_priority": 0.75},
            "L": {"name": "Real estate activities", "sdg_priority": 0.70},
            "M": {
                "name": "Professional, scientific and technical activities",
                "sdg_priority": 0.85,
            },
            "N": {"name": "Administrative and support service activities", "sdg_priority": 0.70},
            "O": {"name": "Public administration and defence", "sdg_priority": 0.90},
            "P": {"name": "Education", "sdg_priority": 0.95},
            "Q": {"name": "Human health and social work activities", "sdg_priority": 0.95},
            "R": {"name": "Arts, entertainment and recreation", "sdg_priority": 0.65},
            "S": {"name": "Other service activities", "sdg_priority": 0.70},
            "T": {"name": "Activities of households as employers", "sdg_priority": 0.75},
            "U": {"name": "Activities of extraterritorial organizations", "sdg_priority": 0.80},
        }

        self.sector_dependencies = {
            "C": ["B", "D", "E", "H"],
            "D": ["B"],
            "F": ["C", "D", "H"],
            "G": ["C", "H"],
            "H": ["D"],
            "J": ["D"],
            "M": ["J", "P"],
        }

        self.regenerative_config = regenerative_config or {}

    def detect(
        self, data: dict[str, Any], sector_code: str, region: str = "global"
    ) -> dict[str, Any]:
        """
        Detect economic anomalies in a specific ISIC sector.

        Args:
            data: Economic indicators (GDP contribution, employment, growth rate, etc.)
            sector_code: ISIC sector code ('A' through 'U')
            region: Geographic region ('global', 'sub_saharan_africa', 'southeast_asia', etc.)

        Returns:
            Detection results with sustainability assessment, recommendations
        """
        if sector_code not in self.isic_sectors:
            raise ValueError(f"Unknown ISIC sector code: {sector_code}")

        gdp_growth_rate = data.get("gdp_growth_percent", 0)
        employment_change = data.get("employment_change_percent", 0)
        sustainability_score = data.get("sustainability_score", 0.50)
        trade_disruption = data.get("trade_disruption_detected", False)

        gdp_anomaly = abs(gdp_growth_rate) > 10 or gdp_growth_rate < -5
        employment_anomaly = employment_change < -10
        sustainability_low = sustainability_score < 0.40

        anomaly_detected = (
            gdp_anomaly or employment_anomaly or trade_disruption or sustainability_low
        )

        sector_info = self.isic_sectors[sector_code]
        regenerative_score = self._calculate_regenerative_score(sector_code, sustainability_score)

        return {
            "sector_code": sector_code,
            "sector_name": sector_info["name"],
            "region": region,
            "anomaly_detected": anomaly_detected,
            "metrics": {
                "gdp_growth_percent": gdp_growth_rate,
                "employment_change_percent": employment_change,
                "sustainability_score": sustainability_score,
                "trade_disruption": trade_disruption,
            },
            "anomalies": {
                "gdp_shock": gdp_anomaly,
                "employment_decline": employment_anomaly,
                "low_sustainability": sustainability_low,
                "trade_disruption": trade_disruption,
            },
            "sdg_priority": sector_info["sdg_priority"],
            "regenerative_score": regenerative_score,
            "recommendations": self._generate_economic_recommendations(
                sector_code, gdp_anomaly, employment_anomaly, sustainability_low
            ),
        }

    def analyze_sector_interdependencies(self, affected_sectors: list[Any]) -> dict[str, Any]:
        """
        Analyze economic impact cascades across sector dependencies.

        Args:
            affected_sectors: List of ISIC sector codes with economic shocks

        Returns:
            Cascading impact analysis across dependent sectors
        """
        affected = set(affected_sectors)
        dependent_sectors = set()

        for sector in affected:
            dependents = [s for s, deps in self.sector_dependencies.items() if sector in deps]
            dependent_sectors.update(dependents)

        total_affected = len(affected | dependent_sectors)
        cascade_severity = min(100, total_affected * 4.5)

        return {
            "initial_affected_sectors": list(affected),
            "cascading_affected_sectors": list(dependent_sectors - affected),
            "total_sectors_at_risk": total_affected,
            "cascade_severity_score": cascade_severity,
            "recommendations": [
                "Implement targeted economic support for affected sectors",
                "Monitor supply chain disruptions in dependent sectors",
                "Coordinate cross-sector resilience planning",
            ],
        }

    def assess_regional_sustainability(self, regional_data: dict[str, float]) -> dict[str, Any]:
        """
        Assess overall economic sustainability for a region.

        Args:
            regional_data: Sector-wise sustainability scores for a region

        Returns:
            Regional sustainability assessment with SDG alignment
        """
        weighted_scores = []

        for sector_code, sustainability_score in regional_data.items():
            if sector_code in self.isic_sectors:
                sdg_priority = float(self.isic_sectors[sector_code]["sdg_priority"])  # type: ignore[arg-type, unused-ignore]
                weighted_scores.append(float(sustainability_score) * sdg_priority)

        overall_score = 0.5 if not weighted_scores else np.mean(weighted_scores)

        sdg_alignment = (
            "high" if overall_score > 0.75 else "medium" if overall_score > 0.60 else "low"
        )

        return {
            "overall_sustainability_score": overall_score,
            "sdg_alignment": sdg_alignment,
            "sectors_assessed": len(weighted_scores),
            "regenerative_potential": (
                "net_positive"
                if overall_score > 0.80
                else "sustainable" if overall_score > 0.65 else "needs_improvement"
            ),
        }

    def _calculate_regenerative_score(self, sector_code: str, sustainability_score: float) -> float:
        """Calculate regenerative economics score using permaculture principles."""
        sector_multipliers = {
            "A": 1.2,
            "D": 1.15,
            "E": 1.15,
            "P": 1.10,
            "Q": 1.10,
        }

        multiplier = sector_multipliers.get(sector_code, 1.0)
        regenerative_score = sustainability_score * multiplier

        return min(1.0, regenerative_score)

    def _generate_economic_recommendations(
        self,
        sector_code: str,
        gdp_anomaly: bool,
        employment_anomaly: bool,
        sustainability_low: bool,
    ) -> list[Any]:
        """Generate recommendations for economic sector issues."""
        recommendations = []

        if gdp_anomaly:
            recommendations.append("Investigate causes of economic shock")
            recommendations.append(
                "Implement countercyclical fiscal policies if recession detected"
            )

        if employment_anomaly:
            recommendations.append("Urgent: Deploy workforce support programs")
            recommendations.append("Consider sector-specific job retraining initiatives")

        if sustainability_low:
            recommendations.append("Prioritize green transition investments")
            recommendations.append("Align sector policies with SDG targets")
            recommendations.append("Implement circular economy principles")

        high_priority_sectors = {"D", "E", "P", "Q"}
        if sector_code in high_priority_sectors:
            recommendations.append(
                f"PRIORITY: Sector {sector_code} is critical for SDGs - expedite interventions"
            )

        if not recommendations:
            recommendations.append("Monitor sector performance, maintain sustainable practices")

        return recommendations
