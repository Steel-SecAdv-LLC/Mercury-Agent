# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Crisis Monitoring for Humanitarian CI.

Fuses GEOINT (Geospatial Intelligence) and OSINT reports on natural disasters,
outbreaks, infrastructure failures and mass-casualty incidents.

Severity is the highest validated report score. The crisis type comes from the
indicators of both reports. The affected population is the exposure the GEOINT
report supplies (``affected_population``, or ``population_density`` per km²
times ``affected_area_km2``), the quantity GDACS-style impact estimates rest
on; without exposure data it is ``None`` rather than a guess.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any

# Word-boundary patterns per crisis type, checked in this order.
_CRISIS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "natural_disaster",
        re.compile(
            r"\b(earthquakes?|flood\w*|tsunamis?|hurricanes?|cyclones?|typhoons?|"
            r"tropical storms?|wildfires?|volcan\w*|eruptions?|droughts?|landslides?|"
            r"tornado\w*)\b"
        ),
    ),
    ("pandemic", re.compile(r"\b(pandemic|epidemic|outbreaks?|disease\w*)\b")),
    (
        "infrastructure_failure",
        re.compile(r"\b(infrastructure|power outages?|blackouts?|grid failure|dam failure)\b"),
    ),
    ("mass_casualty", re.compile(r"\b(mass casualt\w*|mci|multiple casualties)\b")),
)


@dataclass
class CrisisAlert:
    """Alert from crisis monitoring."""

    crisis_detected: bool
    crisis_type: str
    severity: str
    affected_population: int | None

    vulnerable_groups: list[str]
    survivor_priorities: list[str]
    geoint_indicators: list[str]
    recommended_response: list[str]


class CrisisMonitor:
    """Humanitarian Crisis Monitor (most-exposed-first counter-intelligence).

    Features:
    - GEOINT and OSINT report fusion for crisis detection and typing
    - Affected population from supplied exposure data, never invented
    - Vulnerable population prioritization
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the instance."""
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
        """Monitor for humanitarian crises using multi-INT fusion.

        Args:
            geoint_data: Geospatial intelligence data: ``threat_score`` in [0, 1],
                ``indicators``, and exposure as ``affected_population`` or
                ``population_density`` (per km²) with ``affected_area_km2``
            osint_data: Open-source intelligence data: ``threat_score`` and
                ``indicators``

        Returns:
            CrisisAlert with response recommendations
        """
        if geoint_data is None:
            geoint_data = {}

        crisis_score = self._compute_crisis_score(geoint_data, osint_data)

        crisis_detected = crisis_score > self.severity_threshold

        crisis_type = self._classify_crisis(geoint_data, osint_data)

        severity = self._determine_severity(crisis_score)

        affected_population = self._estimate_affected_population(geoint_data)

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
            affected = "unknown" if affected_population is None else f"{affected_population:,}"
            self.logger.warning(
                f"Crisis detected: {crisis_type} (severity={severity}, affected={affected})"
            )

        return alert

    def _compute_crisis_score(
        self, geoint_data: dict[str, Any], osint_data: dict[str, Any] | None
    ) -> float:
        """Highest report score, each clipped to [0, 1]; invalid scores count as 0.

        Severity takes the most severe credible report rather than accumulating
        reports, which would inflate the severity of one event.
        """
        score = _unit_score(geoint_data.get("threat_score"))

        if osint_data:
            score = max(score, _unit_score(osint_data.get("threat_score")))

        return score

    def _classify_crisis(
        self, geoint_data: dict[str, Any], osint_data: dict[str, Any] | None
    ) -> str:
        """Classify crisis type from GEOINT and OSINT indicators.

        Each indicator is matched on word boundaries (``_`` and ``-`` read as
        spaces). Cause-based types win over ``mass_casualty``; with no match the
        type is ``humanitarian_emergency``.
        """
        indicators = list(_as_list(geoint_data.get("indicators")))
        if osint_data:
            indicators.extend(_as_list(osint_data.get("indicators")))
        texts = [re.sub(r"[_\-]+", " ", str(ind)).lower() for ind in indicators]

        for crisis_type, pattern in _CRISIS_PATTERNS:
            if any(pattern.search(text) for text in texts):
                return crisis_type
        return "humanitarian_emergency"

    def _determine_severity(self, crisis_score: float) -> str:
        """Determine crisis severity level."""
        if crisis_score > 0.9:
            return "critical"
        elif crisis_score > 0.7:
            return "high"
        elif crisis_score > 0.5:
            return "medium"
        else:
            return "low"

    def _estimate_affected_population(self, geoint_data: dict[str, Any]) -> int | None:
        """Exposed population from the GEOINT report, or ``None`` if not supplied.

        Uses ``affected_population`` when given, else ``population_density``
        (people per km²) times ``affected_area_km2``. Negative or non-finite
        values are ignored. The crisis score does not scale the count.
        """
        direct = _non_negative(geoint_data.get("affected_population"))
        if direct is not None:
            return round(direct)
        density = _non_negative(geoint_data.get("population_density"))
        area = _non_negative(geoint_data.get("affected_area_km2"))
        if density is None or area is None:
            return None
        return round(density * area)

    def _identify_vulnerable_groups(self, crisis_type: str, severity: str) -> list[str]:
        """Identify vulnerable population groups."""
        groups = ["Elderly", "Children", "Disabled individuals", "Low-income families"]

        if crisis_type == "pandemic":
            groups.extend(["Immunocompromised individuals", "Healthcare workers"])
        elif crisis_type == "natural_disaster":
            groups.extend(["Homeless population", "Rural communities"])

        if severity in ["high", "critical"]:
            groups.append("General population")

        return groups

    def _prioritize_survivors(self, vulnerable_groups: list[str], crisis_type: str) -> list[str]:
        """Order survivors for rescue/aid by need, most exposed first."""
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
        return [str(ind) for ind in _as_list(geoint_data.get("indicators"))]

    def _recommend_response(
        self, crisis_type: str, severity: str, affected_population: int | None
    ) -> list[str]:
        """Recommend humanitarian response actions."""
        response = []

        if severity in ["high", "critical"]:
            response.append("Activate national emergency response (FEMA, Red Cross)")
            response.append("Deploy search and rescue teams")
            response.append("Establish emergency medical facilities")

        if affected_population is None:
            response.append(
                "Estimate the exposed population (gridded population over the hazard "
                "footprint) before scaling the response"
            )
        elif affected_population > 100000:
            response.append("Request international humanitarian aid")
            response.append("Coordinate with UN agencies (UNHCR, WHO, WFP)")

        if crisis_type == "mass_casualty":
            response.append("Activate mass-casualty incident triage (e.g., START)")
            response.append("Coordinate hospital surge capacity and patient distribution")

        if crisis_type == "natural_disaster":
            response.append("Distribute emergency supplies (food, water, blankets)")
            response.append("Establish temporary shelters")

        if crisis_type == "pandemic":
            response.append("Activate pandemic response protocols")
            response.append("Deploy mobile testing and vaccination units")

        response.append("Prioritize vulnerable populations in all response efforts")

        return response


def _unit_score(value: Any) -> float:
    """A report score clipped to [0, 1]; missing, non-numeric or non-finite is 0."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return min(max(score, 0.0), 1.0)


def _non_negative(value: Any) -> float | None:
    """A finite, non-negative number, else ``None``."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0.0 else None


def _as_list(value: Any) -> list[Any]:
    """Indicators as a list; a single string counts as one indicator."""
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]
