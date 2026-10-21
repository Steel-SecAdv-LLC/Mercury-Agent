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

When a GEOINT report carries the inputs of a published GDACS model (see
:mod:`.gdacs`), the alert also carries the GDACS level: earthquake scores from
the shakemap or "EQ parameters" model, flood levels from reported deaths and
displaced people, and tropical-cyclone wind levels from the track points' wind,
buffer populations and country vulnerability. An Orange or Red level counts as a
detected crisis and raises
severity to at least ``high`` or to ``critical``, this module's reading of
GDACS's descriptions (Orange: national capacity may need support; Red:
international intervention likely); the hazard makes the type
``natural_disaster``. Incomplete or invalid GDACS inputs are named in the
response instead of being guessed.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any, NamedTuple

import numpy as np

from omni_mercury_engine.infrastructure.humanitarian.crisis_monitoring import gdacs

# Earthquake scores are calibrated to casualties: below 1 under about 10, 1-2
# about 10-100, 2 and above more than 100 (gdacs.org/Knowledge/models_eq.aspx);
# flood levels are thresholds on reported deaths and displacement (models_fl).
_GDACS_RESPONSE = {
    ("earthquake", "red"): (
        "GDACS Red earthquake: humanitarian intervention likely needed "
        "(score calibrated to 100+ casualties)"
    ),
    ("earthquake", "orange"): (
        "GDACS Orange earthquake: national capacity may need support "
        "(score calibrated to 10-100 casualties)"
    ),
    ("earthquake", "green"): "GDACS Green earthquake: within national coping capacity",
    ("flood", "red"): "GDACS Red flood: more than 1,000 dead or 800,000 displaced",
    ("flood", "orange"): "GDACS Orange flood: more than 100 dead or 80,000 displaced",
    ("flood", "green"): "GDACS Green flood: below the Orange thresholds",
    # "All TCs that required an international humanitarian intervention or a major
    # national relief effort were flagged by GDACS at red level" (models_tc).
    ("tropical_cyclone", "red"): (
        "GDACS Red tropical cyclone (wind): the level GDACS has given every cyclone "
        "that needed international or major national relief"
    ),
    ("tropical_cyclone", "orange"): (
        "GDACS Orange tropical cyclone (wind): exposure and country vulnerability "
        "above the Orange thresholds"
    ),
    (
        "tropical_cyclone",
        "green",
    ): "GDACS Green tropical cyclone (wind): below the Orange thresholds",
}
_SEVERITIES = ("low", "medium", "high", "critical")
_GDACS_SEVERITY = {"orange": "high", "red": "critical"}
_EARTHQUAKE_INPUTS = (
    "shakemap_scaled_population",
    "population_by_mmi",
    "magnitude",
    "depth_km",
    "population_within_km",
)
_FLOOD_INPUTS = ("deaths", "displaced")
_CYCLONE_INPUTS = ("cyclone_points", "max_wind_kt", "max_storm_surge_m")
_CYCLONE_ALIASES = {"hurricane", "typhoon", "cyclone", "tc", "tropical cyclone", "tropical storm"}
_GDACS_INPUTS = (
    "hazard",
    *_EARTHQUAKE_INPUTS,
    *_FLOOD_INPUTS,
    *_CYCLONE_INPUTS,
    "inform_lcc",
    "coping_capacity_factor",
    "country_iso3",
    "gdacs_coefficients",
)


_GDACS_HAZARDS = ("earthquake", "flood", "tropical_cyclone")


class _GdacsAssessment(NamedTuple):
    hazard: str
    level: str | None
    score: float | None
    notes: list[str]


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
    (
        "pandemic",
        re.compile(
            r"\b(pandemic|epidemic|disease\w*|"
            r"outbreaks?(?!\s+of\s+(violence|fighting|hostilities|conflict|war|unrest|riots?)))\b"
        ),
    ),
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
    gdacs_level: str | None = None
    gdacs_score: float | None = None


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
                ``population_density`` (per km²) with ``affected_area_km2``.
                Optional GDACS inputs: ``hazard`` (``earthquake``, ``flood``
                or ``tropical_cyclone``, else inferred from the inputs, and from
                the indicators when they fit more than one hazard);
                ``shakemap_scaled_population`` (10**GDACS's ``ScaledPopulation``,
                a log10, or 0 where GDACS reports 0) or ``population_by_mmi``
                (``{"VII": n, "VIII": n, "IX": n}``) or
                ``magnitude``, ``depth_km`` and ``population_within_km``
                (``{20: n, 50: n, 75: n, 100: n}``); ``deaths`` and
                ``displaced``; ``inform_lcc`` or ``coping_capacity_factor``;
                ``country_iso3`` for GDACS's country terms; and
                ``gdacs_coefficients`` overriding them by
                :class:`~.gdacs.EarthquakeCoefficients` field name. Tropical
                cyclones: ``cyclone_points``, a list of track points with
                ``max_wind_kt``, ``population_34kt``, ``population_64kt``, the
                point's ``country_iso3`` (or ``tc_vulnerability``), optional
                ``country_population`` and ``lead_time_hours``; or one point
                given by those keys on the report itself; and
                ``max_storm_surge_m``
            osint_data: Open-source intelligence data: ``threat_score`` and
                ``indicators``

        Returns:
            CrisisAlert with response recommendations
        """
        if geoint_data is None:
            geoint_data = {}

        crisis_score = self._compute_crisis_score(geoint_data, osint_data)

        crisis_type = self._classify_crisis(geoint_data, osint_data)

        assessment = self._gdacs_assessment(geoint_data, osint_data)
        gdacs_level, gdacs_score = assessment.level, assessment.score
        if assessment.hazard in _GDACS_HAZARDS:
            crisis_type = "natural_disaster"

        crisis_detected = crisis_score > self.severity_threshold or gdacs_level in (
            "orange",
            "red",
        )

        severity = self._determine_severity(crisis_score)
        floor = _GDACS_SEVERITY.get(gdacs_level or "")
        if floor and _SEVERITIES.index(floor) > _SEVERITIES.index(severity):
            severity = floor

        affected_population = self._estimate_affected_population(geoint_data)

        vulnerable_groups = self._identify_vulnerable_groups(crisis_type, severity)

        survivor_priorities = self._prioritize_survivors(vulnerable_groups, crisis_type)

        geoint_indicators = self._extract_geoint_indicators(geoint_data)

        response = self._recommend_response(crisis_type, severity, affected_population)
        if gdacs_level is not None:
            response.insert(0, _GDACS_RESPONSE[(assessment.hazard, gdacs_level)])
        response.extend(assessment.notes)

        alert = CrisisAlert(
            crisis_detected=crisis_detected,
            crisis_type=crisis_type,
            severity=severity,
            affected_population=affected_population,
            vulnerable_groups=vulnerable_groups,
            survivor_priorities=survivor_priorities,
            geoint_indicators=geoint_indicators,
            recommended_response=response,
            gdacs_level=gdacs_level,
            gdacs_score=gdacs_score,
        )

        if crisis_detected:
            affected = "unknown" if affected_population is None else f"{affected_population:,}"
            self.logger.warning(
                f"Crisis detected: {crisis_type} (severity={severity}, affected={affected})"
            )

        return alert

    def _gdacs_assessment(
        self, geoint_data: dict[str, Any], osint_data: dict[str, Any] | None
    ) -> _GdacsAssessment:
        """GDACS hazard, level, score and notes from the report's inputs.

        The level is ``None`` when the report carries no GDACS inputs, or when
        they are incomplete or invalid, which a note then explains.
        """
        hazard, unresolved = self._gdacs_hazard(geoint_data, osint_data)
        assess = {
            "earthquake": self._earthquake_level,
            "flood": self._flood_level,
            "tropical_cyclone": self._cyclone_level,
        }.get(hazard)
        if assess is None:
            return _GdacsAssessment(hazard, None, None, [] if unresolved is None else [unresolved])
        stated = any(key in geoint_data for key in _GDACS_INPUTS)
        notes: list[str] = []
        try:
            level, score = assess(geoint_data, notes, stated)
        except (TypeError, ValueError) as error:
            return _GdacsAssessment(hazard, None, None, [f"GDACS level not computed: {error}"])
        return _GdacsAssessment(hazard, level, score, notes)

    @staticmethod
    def _flood_level(
        geoint_data: dict[str, Any], notes: list[str], stated: bool
    ) -> tuple[str | None, float | None]:
        """GDACS flood level from ``deaths`` and ``displaced``; floods have no score."""
        if not any(key in geoint_data for key in _FLOOD_INPUTS):
            if stated:
                notes.append("GDACS flood level needs deaths or displaced")
            return None, None
        level = gdacs.flood_alert_level(
            _number(geoint_data.get("deaths", 0)), _number(geoint_data.get("displaced", 0))
        )
        return level, None

    def _earthquake_level(
        self, geoint_data: dict[str, Any], notes: list[str], stated: bool
    ) -> tuple[str | None, float | None]:
        """GDACS earthquake level and final score, after coping capacity."""
        if geoint_data.get("shakemap_scaled_population") is not None:
            score = gdacs.shakemap_score(
                _number(geoint_data["shakemap_scaled_population"]),
                self._gdacs_coefficients(geoint_data, notes),
            )
        elif geoint_data.get("population_by_mmi") is not None:
            by_mmi = _mapping("population_by_mmi", geoint_data["population_by_mmi"])
            unknown = set(by_mmi) - {"VII", "VIII", "IX"}
            if unknown:
                raise ValueError(
                    f"population_by_mmi takes VII, VIII and IX, not {sorted(map(str, unknown))}"
                )
            score = gdacs.earthquake_shakemap_score(
                _number(by_mmi.get("VII", 0)),
                _number(by_mmi.get("VIII", 0)),
                _number(by_mmi.get("IX", 0)),
                self._gdacs_coefficients(geoint_data, notes),
            )
        elif "magnitude" in geoint_data and "population_within_km" in geoint_data:
            radii = _radii(geoint_data["population_within_km"])
            score = gdacs.earthquake_parameter_score(
                _number(geoint_data["magnitude"]),
                _number(geoint_data.get("depth_km")),
                radii,
                self._gdacs_coefficients(geoint_data, notes),
            )
        else:
            if stated:
                notes.append(
                    "GDACS earthquake level needs shakemap_scaled_population, "
                    "population_by_mmi, or magnitude, depth_km and population_within_km"
                )
            return None, None
        factor = geoint_data.get("coping_capacity_factor")
        if factor is None and geoint_data.get("inform_lcc") is not None:
            factor = gdacs.coping_capacity_factor(_number(geoint_data["inform_lcc"]))
        if factor is None:
            notes.append("GDACS score uses no coping-capacity factor (no INFORM LCC supplied)")
            return gdacs.alert_level(score), score
        final = gdacs.apply_coping_capacity(score, _number(factor))
        return gdacs.alert_level(final), final

    def _cyclone_level(
        self, geoint_data: dict[str, Any], notes: list[str], stated: bool
    ) -> tuple[str | None, float | None]:
        """GDACS tropical-cyclone wind level over the track points, and its score."""
        points = geoint_data.get("cyclone_points")
        if points is None:
            if "max_wind_kt" not in geoint_data:
                if stated:
                    notes.append(
                        "GDACS tropical cyclone level needs cyclone_points, or max_wind_kt "
                        "with population_34kt and population_64kt"
                    )
                return None, None
            points = [geoint_data]
        elif isinstance(points, (str, bytes, Mapping)) or not isinstance(points, Sequence):
            raise ValueError("cyclone_points must be a list of track points")
        track = [self._cyclone_point(_mapping("a cyclone point", p), geoint_data) for p in points]
        if any(p.population_64kt > 0 and p.country_population is None for p in track):
            notes.append(
                "GDACS's 10%-of-country thresholds not applied (no country_population supplied)"
            )
        level = gdacs.tropical_cyclone_level(track)
        surge = geoint_data.get("max_storm_surge_m")
        if surge is not None:
            try:  # a bad surge value must not discard the wind level
                metres = _number(surge)
                surge_level = gdacs.storm_surge_level(metres)
            except (TypeError, ValueError) as error:
                notes.append(f"GDACS storm surge level not computed: {error}")
            else:
                if surge_level is not None:
                    notes.append(
                        f"GDACS storm surge level {surge_level} ({metres:g} m), "
                        "reported beside the wind level and outside its score"
                    )
        return level, gdacs.tropical_cyclone_score(level)

    @staticmethod
    def _cyclone_point(point: Mapping[Any, Any], report: dict[str, Any]) -> gdacs.CyclonePoint:
        """A track point; the country may come from the point or the report.

        A key the point leaves out or sets to ``None`` falls back to the report's.
        """
        missing = [
            k for k in ("max_wind_kt", "population_34kt", "population_64kt") if k not in point
        ]
        if missing:
            raise ValueError(f"a cyclone point needs {', '.join(missing)}")
        p34, p64 = _number(point["population_34kt"]), _number(point["population_64kt"])
        vulnerability = point.get("tc_vulnerability")
        iso3 = _first_given("country_iso3", point, report)
        if vulnerability is None and iso3 is not None:
            vulnerability = gdacs.tropical_cyclone_vulnerability(str(iso3))
        if vulnerability is None:
            if p34 > 0 or p64 > 0:
                raise ValueError(
                    "a cyclone point with exposure needs country_iso3 or tc_vulnerability"
                )
            vulnerability = "low"  # nobody exposed: the class cannot change the level
        population = _first_given("country_population", point, report)
        return gdacs.CyclonePoint(
            max_wind_kt=_number(point["max_wind_kt"]),
            population_34kt=p34,
            population_64kt=p64,
            vulnerability=str(vulnerability).strip().lower(),
            country_population=None if population is None else _number(population),
            lead_time_hours=_number(point.get("lead_time_hours", 0.0)),
        )

    def _gdacs_hazard(
        self, geoint_data: dict[str, Any], osint_data: dict[str, Any] | None
    ) -> tuple[str, str | None]:
        """The stated ``hazard``, else the one whose model inputs the report carries.

        Returns:
            The hazard (``""`` for none), and a note when the report carries
            both earthquake and tropical-cyclone inputs and its indicators do
            not name exactly one of them.
        """
        stated = geoint_data.get("hazard")
        if stated is not None:
            hazard = re.sub(r"[_\-]+", " ", str(stated)).strip().lower()
            hazard = "tropical_cyclone" if hazard in _CYCLONE_ALIASES else hazard.replace(" ", "_")
            return hazard, None
        by_inputs = [
            hazard
            for hazard, keys in (
                ("tropical_cyclone", ("cyclone_points", "max_wind_kt")),
                ("earthquake", tuple(k for k in _EARTHQUAKE_INPUTS if k != "depth_km")),
            )
            if any(key in geoint_data for key in keys)
        ]
        if len(by_inputs) == 1:
            return by_inputs[0], None
        indicated = self._infer_hazards(geoint_data, osint_data)
        if by_inputs:
            named = [hazard for hazard in by_inputs if hazard in indicated]
            if len(named) == 1:
                return named[0], None
            return "", (
                "GDACS level not computed: the report carries earthquake and tropical "
                "cyclone inputs; state hazard"
            )
        if "flood" in indicated and any(key in geoint_data for key in _FLOOD_INPUTS):
            return "flood", None
        for hazard in ("earthquake", "tropical_cyclone", "flood"):
            if hazard in indicated:
                return hazard, None
        return "", None

    @staticmethod
    def _gdacs_coefficients(
        geoint_data: dict[str, Any], notes: list[str]
    ) -> gdacs.EarthquakeCoefficients:
        """Terms from GDACS's table for ``country_iso3``, overridden by ``gdacs_coefficients``."""
        iso3 = geoint_data.get("country_iso3")
        overrides = _mapping("gdacs_coefficients", geoint_data.get("gdacs_coefficients") or {})
        if iso3 is None:
            base = gdacs.GLOBAL_COEFFICIENTS
            if not overrides:
                notes.append("GDACS score uses the global model (no country_iso3 supplied)")
        else:
            base = gdacs.earthquake_coefficients(iso3)
        unknown = set(overrides) - {f.name for f in fields(base)}
        if unknown:
            raise ValueError(f"unknown gdacs_coefficients {sorted(map(str, unknown))}")
        return replace(base, **{name: _number(value) for name, value in overrides.items()})

    @staticmethod
    def _infer_hazards(geoint_data: dict[str, Any], osint_data: dict[str, Any] | None) -> set[str]:
        """The GDACS hazards (earthquake, flood, tropical cyclone) the indicators name."""
        indicators = list(_as_list(geoint_data.get("indicators")))
        if osint_data:
            indicators.extend(_as_list(osint_data.get("indicators")))
        text = " ".join(re.sub(r"[_\-]+", " ", str(i)).lower() for i in indicators)
        hazards = set()
        if re.search(r"\bearthquakes?\b", text):
            hazards.add("earthquake")
        if re.search(r"\b(hurricanes?|typhoons?|cyclones?|tropical storms?)\b", text):
            hazards.add("tropical_cyclone")
        if re.search(r"\bflood\w*", text):
            hazards.add("flood")
        return hazards

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
    if isinstance(value, bool):
        return 0.0
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


def _mapping(name: str, value: Any) -> Mapping[Any, Any]:
    """``value`` if it is a mapping, else ``ValueError`` naming ``name``."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def _first_given(key: str, *sources: Mapping[Any, Any]) -> Any:
    """The first non-``None`` value of ``key`` among ``sources``, else ``None``."""
    return next((source[key] for source in sources if source.get(key) is not None), None)


def _number(value: Any) -> float:
    """A finite float, else ``ValueError`` (booleans are not numbers here)."""
    if value is None or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"expected a number, got {value!r}")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(
            f"expected a finite number, got {type(value).__name__} too large for a float"
        ) from None
    if not math.isfinite(number):
        raise ValueError(f"expected a finite number, got {value!r}")
    return number


def _radii(value: Any) -> dict[int, float]:
    """``population_within_km`` keyed by whole-km radius, each radius once."""
    radii: dict[int, float] = {}
    for key, population in _mapping("population_within_km", value).items():
        radius = _number(key)
        if not radius.is_integer():
            raise ValueError(f"population_within_km radius {key!r} is not a whole number of km")
        if int(radius) in radii:
            raise ValueError(f"population_within_km gives the {int(radius)} km radius twice")
        radii[int(radius)] = _number(population)
    return radii
