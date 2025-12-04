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
Crisis Monitoring for Humanitarian CI

Integrates GEOINT (Geospatial Intelligence) for natural disaster,
humanitarian crisis, and essential worker protection monitoring.

Survivor-first prioritization using ethical scalars.
"""

import logging
from dataclasses import dataclass
from typing import Any


@dataclass
class CrisisAlert:
    """Alert from crisis monitoring"""

    crisis_detected: bool
    crisis_type: str
    severity: str
    affected_population: int

    vulnerable_groups: list[str]
    survivor_priorities: list[str]
    geoint_indicators: list[str]
    recommended_response: list[str]


class CrisisMonitor:
    """
    Humanitarian Crisis Monitor (Survivor-First CI)

    Features:
    - GEOINT fusion for disaster detection
    - Vulnerable population prioritization
    - Essential worker protection
    - Real-time survivor-first alerts
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        self.severity_threshold = self.config.get("severity_threshold", 0.7)

        self.crisis_types = [
            "natural_disaster",
            "pandemic",
            "humanitarian_emergency",
            "infrastructure_failure",
            "mass_casualty",
        ]

        self.logger.info("CrisisMonitor initialized")

    def monitor_crisis(
        self,
        geoint_data: dict[str, Any] | None = None,
        osint_data: dict[str, Any] | None = None,
    ) -> CrisisAlert:
        """
        Monitor for humanitarian crises using multi-INT fusion.

        Args:
            geoint_data: Geospatial intelligence data
            osint_data: Open-source intelligence data

        Returns:
            CrisisAlert with response recommendations
        """
        if geoint_data is None:
            geoint_data = {}

        crisis_score = self._compute_crisis_score(geoint_data, osint_data)

        crisis_detected = crisis_score > self.severity_threshold

        crisis_type = self._classify_crisis(geoint_data, osint_data)

        severity = self._determine_severity(crisis_score)

        affected_population = self._estimate_affected_population(geoint_data, crisis_score)

        vulnerable_groups = self._identify_vulnerable_groups(crisis_type, severity)

        survivor_priorities = self._prioritize_survivors(vulnerable_groups, crisis_type)

        geoint_indicators = self._extract_geoint_indicators(geoint_data)

        response = self._recommend_response(crisis_type, severity, affected_population)

        alert = CrisisAlert(
            crisis_detected=crisis_detected,
            crisis_type=crisis_type,
            severity=severity,
            affected_population=affected_population,
            vulnerable_groups=vulnerable_groups,
            survivor_priorities=survivor_priorities,
            geoint_indicators=geoint_indicators,
            recommended_response=response,
        )

        if crisis_detected:
            self.logger.warning(
                f"Crisis detected: {crisis_type} "
                f"(severity={severity}, affected={affected_population:,})"
            )

        return alert

    def _compute_crisis_score(
        self, geoint_data: dict[str, Any], osint_data: dict[str, Any] | None
    ) -> float:
        """Compute crisis severity score from multi-INT sources."""
        score = geoint_data.get("threat_score", 0.0)

        if osint_data:
            score = max(score, osint_data.get("threat_score", 0.0))

        return float(min(score, 1.0))

    def _classify_crisis(
        self, geoint_data: dict[str, Any], osint_data: dict[str, Any] | None
    ) -> str:
        """Classify crisis type."""
        indicators = geoint_data.get("indicators", [])

        if "earthquake" in str(indicators).lower() or "flood" in str(indicators).lower():
            return "natural_disaster"
        elif "pandemic" in str(indicators).lower() or "disease" in str(indicators).lower():
            return "pandemic"
        elif "infrastructure" in str(indicators).lower():
            return "infrastructure_failure"
        else:
            return "humanitarian_emergency"

    def _determine_severity(self, crisis_score: float) -> str:
        """Determine crisis severity level."""
        if crisis_score > 0.9:
            return "CATASTROPHIC"
        elif crisis_score > 0.7:
            return "SEVERE"
        elif crisis_score > 0.5:
            return "MODERATE"
        else:
            return "LOW"

    def _estimate_affected_population(
        self, geoint_data: dict[str, Any], crisis_score: float
    ) -> int:
        """Estimate affected population (simulated)."""
        base_population = geoint_data.get("population_density", 10000)
        affected = int(base_population * crisis_score * 10)
        return affected

    def _identify_vulnerable_groups(self, crisis_type: str, severity: str) -> list[str]:
        """Identify vulnerable population groups."""
        groups = ["Elderly", "Children", "Disabled individuals", "Low-income families"]

        if crisis_type == "pandemic":
            groups.extend(["Immunocompromised individuals", "Healthcare workers"])
        elif crisis_type == "natural_disaster":
            groups.extend(["Homeless population", "Rural communities"])

        if severity in ["SEVERE", "CATASTROPHIC"]:
            groups.append("General population")

        return groups

    def _prioritize_survivors(self, vulnerable_groups: list[str], crisis_type: str) -> list[str]:
        """Prioritize survivors for rescue/aid (survivor-first principle)."""
        priorities = []

        priorities.append("Immediate medical attention for critically injured")
        priorities.append("Evacuation of vulnerable groups: " + ", ".join(vulnerable_groups[:3]))
        priorities.append("Secure food, water, and shelter for displaced populations")
        priorities.append("Establish communication channels for family reunification")

        if crisis_type == "pandemic":
            priorities.insert(0, "Quarantine and treatment for infected individuals")

        return priorities

    def _extract_geoint_indicators(self, geoint_data: dict[str, Any]) -> list[str]:
        """Extract GEOINT indicators."""
        indicators = geoint_data.get("indicators", [])
        return [str(ind) for ind in indicators]

    def _recommend_response(
        self, crisis_type: str, severity: str, affected_population: int
    ) -> list[str]:
        """Recommend humanitarian response actions."""
        response = []

        if severity in ["SEVERE", "CATASTROPHIC"]:
            response.append("Activate national emergency response (FEMA, Red Cross)")
            response.append("Deploy search and rescue teams")
            response.append("Establish emergency medical facilities")

        if affected_population > 100000:
            response.append("Request international humanitarian aid")
            response.append("Coordinate with UN agencies (UNHCR, WHO, WFP)")

        if crisis_type == "natural_disaster":
            response.append("Distribute emergency supplies (food, water, blankets)")
            response.append("Establish temporary shelters")

        if crisis_type == "pandemic":
            response.append("Activate pandemic response protocols")
            response.append("Deploy mobile testing and vaccination units")

        response.append("Prioritize vulnerable populations in all response efforts")

        return response
