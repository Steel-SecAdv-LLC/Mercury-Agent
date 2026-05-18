"""OSHA compliance anomaly detection for industry-specific safety monitoring.

Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
details.

This module ports ``osha_compliance_anomaly.py`` from Omni-AXA-Engine and
upgrades the heat-index calculation to the National Weather Service Rothfusz
regression with the standard low-humidity and low-temperature adjustments.

The simplified ``T + 0.5*RH`` heuristic used in the original implementation
diverged from the NWS Rothfusz regression in opposite directions depending on
operating point.  At high humidity it materially **over-reported** apparent
temperature (T=95 F, RH=70% returned ~130 F under the heuristic vs. ~122 F
under Rothfusz, an 8 F over-report).  At low humidity (RH < 40%) it
**under-reported** because it did not apply the low-humidity adjustment.
Both directions cause OSHA-relevant misclassification; the Rothfusz regression
replaces the heuristic so the detector neither cries wolf nor sleeps through
real heat stress.

OSHA standard citations may optionally be validated against the live eCFR API
(https://www.ecfr.gov).  Validation is opt-in via the ``ecfr_client`` argument
so that the detector remains usable in air-gapped deployments.

OSHA Focus Areas
----------------
* Construction: falls, electrical hazards, struck-by, caught-in/between
* Agriculture: machinery, chemicals, heat stress, confined spaces
* Healthcare: violence, infections, ergonomics, hazardous drugs
* Manufacturing: ergonomics, noise, machine guarding, chemical exposure

References
----------
* OSHA Standards: https://www.osha.gov/laws-regs
* NWS Rothfusz regression:
  https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml
* eCFR API: https://www.ecfr.gov/developers/documentation/api/v1
"""

from __future__ import annotations

import json
import logging
import math
import threading
from collections.abc import Mapping  # noqa: TC003 - used in Final[Mapping[...]] at runtime
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Final
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class OSHASector(Enum):
    """OSHA-regulated industry sector."""

    CONSTRUCTION = "CONSTRUCTION"
    AGRICULTURE = "AGRICULTURE"
    HEALTHCARE = "HEALTHCARE"
    MANUFACTURING = "MANUFACTURING"
    GENERAL_INDUSTRY = "GENERAL_INDUSTRY"
    MARITIME = "MARITIME"


class HazardCategory(Enum):
    """OSHA hazard category."""

    FALL = "FALL"
    ELECTRICAL = "ELECTRICAL"
    STRUCK_BY = "STRUCK_BY"
    CAUGHT_IN_BETWEEN = "CAUGHT_IN_BETWEEN"
    CHEMICAL = "CHEMICAL"
    BIOLOGICAL = "BIOLOGICAL"
    ERGONOMIC = "ERGONOMIC"
    NOISE = "NOISE"
    HEAT_STRESS = "HEAT_STRESS"
    VIOLENCE = "VIOLENCE"
    MACHINERY = "MACHINERY"
    CONFINED_SPACE = "CONFINED_SPACE"


class ComplianceLevel(Enum):
    """OSHA compliance assessment severity ladder."""

    CRITICAL_VIOLATION = 1
    SERIOUS_VIOLATION = 2
    OTHER_THAN_SERIOUS = 3
    DE_MINIMIS = 4
    COMPLIANT = 5


@dataclass(frozen=True)
class OSHAStandard:
    """Reference to an OSHA standard citation."""

    standard: str
    title: str
    description: str


@dataclass
class OSHAHazard:
    """Detected OSHA hazard."""

    hazard_id: str
    category: HazardCategory
    sector: OSHASector
    severity: float
    description: str
    osha_standard: str
    recommendations: list[str]
    compliance_level: ComplianceLevel
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    citation_verified: bool | None = None


@dataclass(frozen=True)
class OSHATrainingRecommendation:
    """OSHA training program recommendation."""

    program_name: str
    duration_hours: int
    target_audience: tuple[str, ...]
    topics: tuple[str, ...]
    certification: bool


# ---------------------------------------------------------------------------
# Heat-index calculation (NWS Rothfusz regression)
# ---------------------------------------------------------------------------

_ROTHFUSZ_COEFFS: Final[tuple[float, ...]] = (
    -42.379,
    2.04901523,
    10.14333127,
    -0.22475541,
    -0.00683783,
    -0.05481717,
    0.00122874,
    0.00085282,
    -0.00000199,
)


def compute_heat_index_fahrenheit(temperature_f: float, relative_humidity: float) -> float:
    """Compute apparent temperature using the NWS Rothfusz regression.

    Implements the National Weather Service heat-index algorithm published by
    the Weather Prediction Center.  At low apparent temperatures (HI < 80°F)
    the simple Steadman formula is returned to avoid the Rothfusz polynomial's
    over-correction.  At higher apparent temperatures the full regression is
    used with the two standard adjustments:

    * **Low-humidity adjustment**: ``RH < 13%`` and ``80°F <= T <= 112°F``.
      ``HI -= ((13 - RH) / 4) * sqrt((17 - |T - 95|) / 17)``
    * **Low-temperature/high-humidity adjustment**: ``RH > 85%`` and
      ``80°F <= T <= 87°F``.  ``HI += ((RH - 85) / 10) * ((87 - T) / 5)``

    Args:
        temperature_f: Dry-bulb air temperature in degrees Fahrenheit.
        relative_humidity: Relative humidity as a percentage in ``[0, 100]``.

    Returns:
        Apparent temperature in degrees Fahrenheit.

    Raises:
        ValueError: If ``relative_humidity`` is outside ``[0, 100]`` or the
            temperature is not finite.
    """
    if not math.isfinite(temperature_f):
        raise ValueError(f"temperature_f must be finite, got {temperature_f!r}")
    if not 0.0 <= relative_humidity <= 100.0:
        raise ValueError(f"relative_humidity must be in [0, 100], got {relative_humidity!r}")

    t = float(temperature_f)
    rh = float(relative_humidity)

    simple_hi = 0.5 * (t + 61.0 + ((t - 68.0) * 1.2) + (rh * 0.094))
    if (simple_hi + t) / 2.0 < 80.0:
        return simple_hi

    c = _ROTHFUSZ_COEFFS
    hi = (
        c[0]
        + c[1] * t
        + c[2] * rh
        + c[3] * t * rh
        + c[4] * t * t
        + c[5] * rh * rh
        + c[6] * t * t * rh
        + c[7] * t * rh * rh
        + c[8] * t * t * rh * rh
    )

    if rh < 13.0 and 80.0 <= t <= 112.0:
        hi -= ((13.0 - rh) / 4.0) * math.sqrt((17.0 - abs(t - 95.0)) / 17.0)
    elif rh > 85.0 and 80.0 <= t <= 87.0:
        hi += ((rh - 85.0) / 10.0) * ((87.0 - t) / 5.0)

    return hi


# ---------------------------------------------------------------------------
# eCFR client (optional, live)
# ---------------------------------------------------------------------------


class ECFRClientError(RuntimeError):
    """Raised when the eCFR API returns an unrecoverable error."""


class ECFRClient:
    """Read-only client for the public Code of Federal Regulations API.

    The eCFR is a free public service operated by the U.S. National Archives
    and the Office of the Federal Register.  Authentication is not required.

    Rate limiting
    -------------
    The public eCFR API publishes a 60 req/min/IP guidance.  This client
    **does not enforce that limit programmatically** - operators running
    batch audits across many citations should cap concurrency at the call
    site (e.g. a thread / asyncio semaphore around
    :meth:`verify_citation`) or pace requests externally.  The in-process
    cache (:attr:`_cache`, protected by :attr:`_cache_lock`) reduces
    duplicate lookups during a single audit run and is the primary
    mechanism by which Mercury stays under the published limit.
    """

    DEFAULT_BASE_URL: Final[str] = "https://www.ecfr.gov"

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 10.0,
        user_agent: str = "Mercury-Agent/1.7 OSHA-Compliance",
    ) -> None:
        """Initialise the eCFR client.

        Args:
            base_url: Base URL for the eCFR HTTP API.
            timeout_seconds: Network timeout for each request.
            user_agent: HTTP ``User-Agent`` header value.
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = float(timeout_seconds)
        self._user_agent = user_agent
        self._cache: dict[tuple[str, str, str], bool] = {}
        self._cache_lock = threading.Lock()

    @staticmethod
    def parse_citation(citation: str) -> tuple[str, str, str] | None:
        """Parse a CFR citation string of the form ``29 CFR 1926.501``.

        Args:
            citation: Citation string.

        Returns:
            A ``(title, part, section)`` tuple, or ``None`` if the citation is
            not parseable as a numeric Title/Part/Section reference (e.g.
            "OSHA Guidelines").
        """
        text = citation.strip()
        marker = " CFR "
        if marker not in text:
            return None
        title_part, rest = text.split(marker, 1)
        title = title_part.strip()
        if not title.isdigit():
            return None
        section_text = rest.split()[0]
        if "." in section_text:
            part, section = section_text.split(".", 1)
        else:
            part, section = section_text, ""
        if not part.isdigit():
            return None
        return title, part, section

    def verify_citation(self, citation: str) -> bool:
        """Check whether the cited CFR section exists in current eCFR.

        Args:
            citation: Citation in the form ``29 CFR 1910.95``.

        Returns:
            ``True`` if the cited Title/Part is currently published in the
            eCFR, ``False`` otherwise.  Returns ``False`` for citations that
            cannot be parsed as numeric Title/Part references (e.g.
            non-binding OSHA guidance), so guidance citations are reported as
            unverified rather than raising.
        """
        parsed = self.parse_citation(citation)
        if parsed is None:
            return False
        title, part, section = parsed

        with self._cache_lock:
            cached = self._cache.get((title, part, section))
            if cached is not None:
                return cached

        url = f"{self._base_url}/api/versioner/v1/structure/current/title-{title}.json"
        request = Request(  # noqa: S310 - HTTPS URL constructed from validated digits
            url,
            headers={"User-Agent": self._user_agent, "Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:  # noqa: S310
                if response.status != 200:
                    raise ECFRClientError(f"Unexpected status {response.status} for {url}")
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 404:
                with self._cache_lock:
                    self._cache[(title, part, section)] = False
                return False
            raise ECFRClientError(f"eCFR HTTP error {exc.code}: {exc.reason}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ECFRClientError(f"eCFR request failed: {exc}") from exc

        verified = _ecfr_structure_contains_part(payload, part)
        with self._cache_lock:
            self._cache[(title, part, section)] = verified
        return verified


def _ecfr_structure_contains_part(structure: Any, part_number: str) -> bool:
    """Walk the eCFR JSON structure looking for a part matching ``part_number``."""
    if isinstance(structure, dict):
        if structure.get("type") == "part" and str(structure.get("identifier", "")) == part_number:
            return True
        for child in structure.get("children", []) or []:
            if _ecfr_structure_contains_part(child, part_number):
                return True
    elif isinstance(structure, list):
        for child in structure:
            if _ecfr_structure_contains_part(child, part_number):
                return True
    return False


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


_DEFAULT_HAZARD_THRESHOLDS: Final[Mapping[HazardCategory, float]] = {
    HazardCategory.FALL: 0.70,
    HazardCategory.ELECTRICAL: 0.75,
    HazardCategory.STRUCK_BY: 0.65,
    HazardCategory.CAUGHT_IN_BETWEEN: 0.65,
    HazardCategory.CHEMICAL: 0.80,
    HazardCategory.BIOLOGICAL: 0.85,
    HazardCategory.ERGONOMIC: 0.60,
    HazardCategory.NOISE: 0.70,
    HazardCategory.HEAT_STRESS: 0.75,
    HazardCategory.VIOLENCE: 0.80,
    HazardCategory.MACHINERY: 0.70,
    HazardCategory.CONFINED_SPACE: 0.85,
}


_DEFAULT_OSHA_STANDARDS: Final[Mapping[HazardCategory, OSHAStandard]] = {
    HazardCategory.FALL: OSHAStandard(
        standard="29 CFR 1926.501",
        title="Fall Protection in Construction",
        description=("Requirements for fall protection systems at 6 feet or higher"),
    ),
    HazardCategory.ELECTRICAL: OSHAStandard(
        standard="29 CFR 1910.303",
        title="Electrical Safety Standards",
        description="Design safety standards for electrical systems",
    ),
    HazardCategory.STRUCK_BY: OSHAStandard(
        standard="29 CFR 1926.602",
        title="Material Handling Equipment",
        description="Safety requirements for vehicles and equipment",
    ),
    HazardCategory.CAUGHT_IN_BETWEEN: OSHAStandard(
        standard="29 CFR 1926.651",
        title="Excavation Safety",
        description="Requirements for trenching and excavation",
    ),
    HazardCategory.CHEMICAL: OSHAStandard(
        standard="29 CFR 1910.1200",
        title="Hazard Communication Standard (HCS)",
        description="Chemical hazard communication and labeling",
    ),
    HazardCategory.BIOLOGICAL: OSHAStandard(
        standard="29 CFR 1910.1030",
        title="Bloodborne Pathogens",
        description="Protection from bloodborne pathogen exposure",
    ),
    HazardCategory.ERGONOMIC: OSHAStandard(
        standard="OSHA Ergonomics Guidelines",
        title="Ergonomics Program Guidelines",
        description="Prevention of musculoskeletal disorders",
    ),
    HazardCategory.NOISE: OSHAStandard(
        standard="29 CFR 1910.95",
        title="Occupational Noise Exposure",
        description="Hearing conservation program requirements",
    ),
    HazardCategory.HEAT_STRESS: OSHAStandard(
        standard="OSHA Heat Illness Prevention",
        title="Heat Illness Prevention Campaign",
        description="Water, rest, shade requirements",
    ),
    HazardCategory.VIOLENCE: OSHAStandard(
        standard="OSHA Workplace Violence Guidelines",
        title="Workplace Violence Prevention",
        description="Guidelines for preventing workplace violence",
    ),
    HazardCategory.MACHINERY: OSHAStandard(
        standard="29 CFR 1910.212",
        title="Machine Guarding",
        description="Requirements for machine safeguarding",
    ),
    HazardCategory.CONFINED_SPACE: OSHAStandard(
        standard="29 CFR 1910.146",
        title="Permit-Required Confined Spaces",
        description="Entry procedures for confined spaces",
    ),
}


_DEFAULT_TRAINING_PROGRAMS: Final[Mapping[str, OSHATrainingRecommendation]] = {
    "OSHA_10": OSHATrainingRecommendation(
        program_name="OSHA 10-Hour Outreach Training",
        duration_hours=10,
        target_audience=("Entry-level workers", "New hires"),
        topics=(
            "Hazard recognition",
            "Fall protection",
            "Electrical safety",
            "Personal protective equipment",
            "Health hazards",
        ),
        certification=True,
    ),
    "OSHA_30": OSHATrainingRecommendation(
        program_name="OSHA 30-Hour Outreach Training",
        duration_hours=30,
        target_audience=("Supervisors", "Safety personnel"),
        topics=(
            "Advanced hazard recognition",
            "OSHA standards and regulations",
            "Accident investigation",
            "Safety program development",
            "Emergency response planning",
        ),
        certification=True,
    ),
    "HAZWOPER": OSHATrainingRecommendation(
        program_name="HAZWOPER Training",
        duration_hours=40,
        target_audience=("Hazmat responders", "Cleanup workers"),
        topics=(
            "Hazardous waste operations",
            "Emergency response",
            "Chemical protective equipment",
            "Decontamination procedures",
            "Site safety plans",
        ),
        certification=True,
    ),
    "FALL_PROTECTION": OSHATrainingRecommendation(
        program_name="Fall Protection Training",
        duration_hours=4,
        target_audience=("Construction workers", "Roofers"),
        topics=(
            "Fall hazard identification",
            "Personal fall arrest systems",
            "Guardrail systems",
            "Safety net systems",
            "Rescue procedures",
        ),
        certification=False,
    ),
}


class OSHAComplianceDetector:
    """OSHA compliance anomaly detector for industry-specific safety monitoring.

    Detects workplace hazards, assesses compliance, and recommends corrective
    actions aligned with OSHA standards.  Heat-index hazards use the NWS
    Rothfusz regression with the standard low-humidity and low-temperature
    adjustments; numeric CFR citations may be verified against the live eCFR
    API by passing an :class:`ECFRClient` instance.
    """

    def __init__(
        self,
        sector: OSHASector = OSHASector.GENERAL_INDUSTRY,
        *,
        ecfr_client: ECFRClient | None = None,
        hazard_thresholds: Mapping[HazardCategory, float] | None = None,
    ) -> None:
        """Initialise the detector.

        Args:
            sector: Primary industry sector for compliance monitoring.
            ecfr_client: Optional eCFR client.  When supplied, every emitted
                hazard with a numeric ``29 CFR x.y`` citation will be checked
                against the live eCFR and the result stored on
                :attr:`OSHAHazard.citation_verified`.
            hazard_thresholds: Optional override for the per-category severity
                thresholds (values in ``[0, 1]``).
        """
        self.sector = sector
        self._ecfr_client = ecfr_client
        self.hazard_thresholds: dict[HazardCategory, float] = dict(
            hazard_thresholds or _DEFAULT_HAZARD_THRESHOLDS
        )
        self.osha_standards: dict[HazardCategory, OSHAStandard] = dict(_DEFAULT_OSHA_STANDARDS)
        self.training_programs: dict[str, OSHATrainingRecommendation] = dict(
            _DEFAULT_TRAINING_PROGRAMS
        )

    # -- public API ---------------------------------------------------------

    def detect_hazards(
        self,
        sensor_data: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> list[OSHAHazard]:
        """Detect OSHA compliance hazards from sensor and contextual data.

        Args:
            sensor_data: Sensor readings and monitoring data.
            context: Additional context (location, activity, personnel).

        Returns:
            List of detected OSHA hazards.
        """
        ctx: Mapping[str, Any] = context or {}
        hazards: list[OSHAHazard]
        if self.sector is OSHASector.CONSTRUCTION:
            hazards = self._detect_construction_hazards(sensor_data, ctx)
        elif self.sector is OSHASector.AGRICULTURE:
            hazards = self._detect_agriculture_hazards(sensor_data, ctx)
        elif self.sector is OSHASector.HEALTHCARE:
            hazards = self._detect_healthcare_hazards(sensor_data, ctx)
        elif self.sector is OSHASector.MANUFACTURING:
            hazards = self._detect_manufacturing_hazards(sensor_data, ctx)
        else:
            hazards = self._detect_general_hazards(sensor_data, ctx)

        for hazard in hazards:
            if self._ecfr_client is not None:
                try:
                    hazard.citation_verified = self._ecfr_client.verify_citation(
                        hazard.osha_standard
                    )
                except ECFRClientError as exc:
                    logger.warning(
                        "eCFR verification failed for %s: %s",
                        hazard.osha_standard,
                        exc,
                    )
                    hazard.citation_verified = None
        return hazards

    def recommend_training(self, hazards: list[OSHAHazard]) -> list[OSHATrainingRecommendation]:
        """Recommend OSHA training programs for the detected hazards.

        Args:
            hazards: Detected hazards.

        Returns:
            Recommended training programs (no duplicates).
        """
        recommendations: list[OSHATrainingRecommendation] = []
        seen: set[str] = set()

        def _add(name: str) -> None:
            if name in seen:
                return
            program = self.training_programs.get(name)
            if program is None:
                return
            recommendations.append(program)
            seen.add(name)

        if any(h.compliance_level.value <= 2 for h in hazards):
            _add("OSHA_30")
        else:
            _add("OSHA_10")

        categories = {h.category for h in hazards}
        if HazardCategory.FALL in categories:
            _add("FALL_PROTECTION")
        if HazardCategory.CHEMICAL in categories:
            _add("HAZWOPER")
        return recommendations

    def generate_compliance_report(
        self,
        hazards: list[OSHAHazard],
        facility_info: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Generate a structured OSHA compliance report.

        Args:
            hazards: Detected hazards.
            facility_info: Facility information and context.

        Returns:
            Compliance report dictionary suitable for serialisation.
        """
        critical = [h for h in hazards if h.compliance_level is ComplianceLevel.CRITICAL_VIOLATION]
        serious = [h for h in hazards if h.compliance_level is ComplianceLevel.SERIOUS_VIOLATION]
        if hazards:
            avg_severity = sum(h.severity for h in hazards) / len(hazards)
        else:
            avg_severity = 0.0
        overall = max(0.0, 1.0 - avg_severity)
        training = self.recommend_training(hazards)
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "facility": dict(facility_info),
            "sector": self.sector.value,
            "overall_compliance_score": overall,
            "total_hazards": len(hazards),
            "critical_violations": len(critical),
            "serious_violations": len(serious),
            "hazards": [
                {
                    "id": h.hazard_id,
                    "category": h.category.value,
                    "severity": h.severity,
                    "description": h.description,
                    "standard": h.osha_standard,
                    "citation_verified": h.citation_verified,
                    "compliance_level": h.compliance_level.name,
                    "recommendations": list(h.recommendations),
                }
                for h in hazards
            ],
            "training_recommendations": [
                {
                    "program": t.program_name,
                    "duration": t.duration_hours,
                    "audience": list(t.target_audience),
                    "topics": list(t.topics),
                }
                for t in training
            ],
            "next_steps": self._generate_next_steps(hazards),
            "osha_resources": {
                "consultation": "https://www.osha.gov/consultation",
                "training": "https://www.osha.gov/training",
                "standards": "https://www.osha.gov/laws-regs",
                "etools": "https://www.osha.gov/etools",
            },
        }

    # -- sector-specific detection ------------------------------------------

    def _detect_construction_hazards(
        self,
        sensor_data: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> list[OSHAHazard]:
        """Detect construction-specific hazards (OSHA Focus Four)."""
        del context
        hazards: list[OSHAHazard] = []
        height = float(sensor_data.get("height_above_ground", 0.0))
        if height >= 6.0 and not bool(sensor_data.get("fall_protection_active", False)):
            severity = min(0.70 + (height - 6.0) * 0.05, 1.0)
            hazards.append(
                self._build_hazard(
                    category=HazardCategory.FALL,
                    sector=OSHASector.CONSTRUCTION,
                    severity=severity,
                    description=(f"Fall hazard detected at {height:.1f} feet without protection"),
                    recommendations=[
                        "Install guardrail systems",
                        "Provide personal fall arrest systems",
                        "Implement safety net systems",
                        "Conduct fall protection training",
                    ],
                    compliance_level=ComplianceLevel.CRITICAL_VIOLATION,
                )
            )
        voltage = float(sensor_data.get("electrical_voltage", 0.0))
        if voltage > 50.0 and not bool(sensor_data.get("electrical_protection", False)):
            severity = min(0.75 + (voltage / 1000.0), 1.0)
            hazards.append(
                self._build_hazard(
                    category=HazardCategory.ELECTRICAL,
                    sector=OSHASector.CONSTRUCTION,
                    severity=severity,
                    description=f"Electrical hazard: {voltage:.0f}V without protection",
                    recommendations=[
                        "De-energize circuits before work",
                        "Use lockout/tagout procedures",
                        "Provide insulated tools and PPE",
                        "Maintain safe clearance distances",
                    ],
                    compliance_level=ComplianceLevel.CRITICAL_VIOLATION,
                )
            )
        return hazards

    def _detect_agriculture_hazards(
        self,
        sensor_data: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> list[OSHAHazard]:
        """Detect agriculture-specific hazards."""
        del context
        hazards: list[OSHAHazard] = []
        proximity = float(sensor_data.get("machinery_proximity_meters", 100.0))
        if proximity < 3.0 and bool(sensor_data.get("machinery_active", False)):
            severity = 1.0 - (proximity / 3.0)
            hazards.append(
                self._build_hazard(
                    category=HazardCategory.MACHINERY,
                    sector=OSHASector.AGRICULTURE,
                    severity=severity,
                    description=(f"Unsafe proximity to active machinery: {proximity:.1f}m"),
                    recommendations=[
                        "Maintain safe distance from equipment",
                        "Install machine guards",
                        "Implement lockout/tagout",
                        "Provide machinery safety training",
                    ],
                    compliance_level=ComplianceLevel.SERIOUS_VIOLATION,
                )
            )
        temperature = float(sensor_data.get("temperature_fahrenheit", 70.0))
        humidity = float(sensor_data.get("humidity_percent", 50.0))
        heat_index = compute_heat_index_fahrenheit(temperature, humidity)
        if heat_index > 103.0:
            severity = min(0.75 + (heat_index - 103.0) / 20.0, 1.0)
            hazards.append(
                self._build_hazard(
                    category=HazardCategory.HEAT_STRESS,
                    sector=OSHASector.AGRICULTURE,
                    severity=severity,
                    description=f"Heat stress risk: Heat index {heat_index:.1f}°F",
                    recommendations=[
                        "Provide water, rest, and shade",
                        "Implement acclimatization program",
                        "Monitor workers for heat illness",
                        "Adjust work schedules for extreme heat",
                    ],
                    compliance_level=ComplianceLevel.SERIOUS_VIOLATION,
                )
            )
        return hazards

    def _detect_healthcare_hazards(
        self,
        sensor_data: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> list[OSHAHazard]:
        """Detect healthcare-specific hazards."""
        del context
        hazards: list[OSHAHazard] = []
        violence_score = float(sensor_data.get("violence_risk_score", 0.0))
        if violence_score > 0.70:
            hazards.append(
                self._build_hazard(
                    category=HazardCategory.VIOLENCE,
                    sector=OSHASector.HEALTHCARE,
                    severity=violence_score,
                    description="Workplace violence risk detected",
                    recommendations=[
                        "Implement workplace violence prevention program",
                        "Provide de-escalation training",
                        "Install security systems and alarms",
                        "Establish emergency response procedures",
                    ],
                    compliance_level=ComplianceLevel.SERIOUS_VIOLATION,
                )
            )
        pathogen_exposure = float(sensor_data.get("pathogen_exposure_risk", 0.0))
        if pathogen_exposure > 0.80:
            hazards.append(
                self._build_hazard(
                    category=HazardCategory.BIOLOGICAL,
                    sector=OSHASector.HEALTHCARE,
                    severity=pathogen_exposure,
                    description="Bloodborne pathogen exposure risk",
                    recommendations=[
                        "Implement exposure control plan",
                        "Provide hepatitis B vaccination",
                        "Use engineering controls (sharps containers)",
                        "Ensure proper PPE availability and use",
                    ],
                    compliance_level=ComplianceLevel.CRITICAL_VIOLATION,
                )
            )
        return hazards

    def _detect_manufacturing_hazards(
        self,
        sensor_data: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> list[OSHAHazard]:
        """Detect manufacturing-specific hazards."""
        del context
        hazards: list[OSHAHazard] = []
        noise = float(sensor_data.get("noise_level_db", 0.0))
        if noise > 85.0:
            severity = min(0.70 + (noise - 85.0) / 30.0, 1.0)
            hazards.append(
                self._build_hazard(
                    category=HazardCategory.NOISE,
                    sector=OSHASector.MANUFACTURING,
                    severity=severity,
                    description=f"Excessive noise exposure: {noise:.1f} dB",
                    recommendations=[
                        "Implement hearing conservation program",
                        "Provide hearing protection devices",
                        "Conduct audiometric testing",
                        "Implement engineering controls to reduce noise",
                    ],
                    compliance_level=ComplianceLevel.SERIOUS_VIOLATION,
                )
            )
        ergo = float(sensor_data.get("repetitive_motion_score", 0.0))
        if ergo > 0.70:
            hazards.append(
                self._build_hazard(
                    category=HazardCategory.ERGONOMIC,
                    sector=OSHASector.MANUFACTURING,
                    severity=ergo,
                    description="Ergonomic hazard: Repetitive motion detected",
                    recommendations=[
                        "Conduct ergonomic job analysis",
                        "Implement job rotation",
                        "Provide ergonomic tools and equipment",
                        "Train workers on proper techniques",
                    ],
                    compliance_level=ComplianceLevel.OTHER_THAN_SERIOUS,
                )
            )
        return hazards

    def _detect_general_hazards(
        self,
        sensor_data: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> list[OSHAHazard]:
        """Detect general-industry hazards via score keys."""
        del context
        hazards: list[OSHAHazard] = []
        for category, threshold in self.hazard_thresholds.items():
            key = f"{category.value.lower()}_score"
            if key not in sensor_data:
                continue
            score = float(sensor_data[key])
            if score > threshold:
                hazards.append(self._build_hazard_from_score(category, score))
        return hazards

    # -- helpers ------------------------------------------------------------

    def _build_hazard(
        self,
        *,
        category: HazardCategory,
        sector: OSHASector,
        severity: float,
        description: str,
        recommendations: list[str],
        compliance_level: ComplianceLevel,
    ) -> OSHAHazard:
        """Construct an :class:`OSHAHazard` with the right citation."""
        standard = self.osha_standards[category]
        timestamp = datetime.now(UTC)
        return OSHAHazard(
            hazard_id=f"{category.value}_{timestamp.timestamp():.6f}",
            category=category,
            sector=sector,
            severity=float(severity),
            description=description,
            osha_standard=standard.standard,
            recommendations=list(recommendations),
            compliance_level=compliance_level,
            timestamp=timestamp,
        )

    def _build_hazard_from_score(self, category: HazardCategory, severity: float) -> OSHAHazard:
        """Construct a generic hazard from a raw severity score."""
        standard = self.osha_standards[category]
        level = self._determine_compliance_level(severity)
        return self._build_hazard(
            category=category,
            sector=self.sector,
            severity=severity,
            description=(f"{category.value} hazard detected (severity: {severity:.2f})"),
            recommendations=[
                f"Review {standard.title}",
                "Conduct hazard assessment",
                "Implement corrective actions",
                "Provide appropriate training",
            ],
            compliance_level=level,
        )

    @staticmethod
    def _determine_compliance_level(severity: float) -> ComplianceLevel:
        """Map a severity score to an :class:`ComplianceLevel`."""
        if severity >= 0.90:
            return ComplianceLevel.CRITICAL_VIOLATION
        if severity >= 0.75:
            return ComplianceLevel.SERIOUS_VIOLATION
        if severity >= 0.60:
            return ComplianceLevel.OTHER_THAN_SERIOUS
        if severity >= 0.40:
            return ComplianceLevel.DE_MINIMIS
        return ComplianceLevel.COMPLIANT

    @staticmethod
    def _generate_next_steps(hazards: list[OSHAHazard]) -> list[str]:
        """Generate prioritised next steps for compliance follow-up."""
        next_steps: list[str] = []
        critical = [h for h in hazards if h.compliance_level is ComplianceLevel.CRITICAL_VIOLATION]
        if critical:
            next_steps.append(
                f"IMMEDIATE ACTION REQUIRED: Address {len(critical)} critical violations"
            )
            next_steps.append("Consider requesting OSHA On-Site Consultation")
        serious = [h for h in hazards if h.compliance_level is ComplianceLevel.SERIOUS_VIOLATION]
        if serious:
            next_steps.append(
                f"HIGH PRIORITY: Remediate {len(serious)} serious violations within 30 days"
            )
        next_steps.extend(
            [
                "Develop written safety and health program",
                "Conduct regular workplace inspections",
                "Implement hazard reporting system",
                "Schedule OSHA training for supervisors and workers",
                "Review and update emergency action plans",
            ]
        )
        return next_steps


def get_osha_compliance_detector(
    sector: OSHASector = OSHASector.GENERAL_INDUSTRY,
    *,
    ecfr_client: ECFRClient | None = None,
) -> OSHAComplianceDetector:
    """Factory returning a configured :class:`OSHAComplianceDetector`.

    Args:
        sector: Primary industry sector to monitor.
        ecfr_client: Optional eCFR client for live citation verification.

    Returns:
        A new detector instance.
    """
    return OSHAComplianceDetector(sector=sector, ecfr_client=ecfr_client)


__all__ = [
    "ComplianceLevel",
    "ECFRClient",
    "ECFRClientError",
    "HazardCategory",
    "OSHAComplianceDetector",
    "OSHAHazard",
    "OSHASector",
    "OSHAStandard",
    "OSHATrainingRecommendation",
    "compute_heat_index_fahrenheit",
    "get_osha_compliance_detector",
]
