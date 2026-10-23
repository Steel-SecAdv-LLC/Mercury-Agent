# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Marine Biodiversity & Ecosystem Detector.

Comprehensive marine ecosystem monitoring for conservation and early warning:
- Species abundance tracking (population decline detection)
- Coral bleaching detection and prediction
- Ocean acidification monitoring (pH anomalies)
- Marine heatwave detection
- Ecosystem collapse prediction
- Harmful algal bloom (HAB) detection
- Fishery collapse early warning
- Deep-sea ecosystem monitoring

Integrations:
- Oceanography pattern recognizer
- Biometric analysis (camera trap AI)
- Chemistry integration (pH, CO2, nutrients)
- Satellite ocean color monitoring
- Underwater acoustic monitoring

Research sources:
- NOAA Fisheries
- IUCN Red List
- Coral Reef Watch
- Ocean Biodiversity Information System (OBIS)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from omni_mercury_engine.core.exceptions import InputModalityError
from omni_mercury_engine.detectors._hazard_input import InputGrader
from omni_mercury_engine.detectors.provenance import DecisionSource

#: Every value ``predict_biodiversity_threat`` reads: section -> key -> (unit,
#: physical range).  Supplied values are checked first: a NaN failed every
#: threshold comparison and read as "healthy".
MARINE_INPUTS: dict[str, dict[str, tuple[str, float, float]]] = {
    "coral_data": {
        "sst_anomaly_c": ("degC", -15.0, 15.0),
        "degree_heating_weeks": ("degC-weeks", 0.0, 52.0),
    },
    "chemistry_data": {
        "ph": ("pH", 6.0, 9.0),
        "baseline_ph": ("pH", 6.0, 9.0),
    },
    "temperature_data": {
        "anomaly_c": ("degC", -15.0, 15.0),
        "p90_anomaly_c": ("degC", 0.01, 15.0),
        "days_at_or_above_p90": ("days", 0.0, 36500.0),
    },
}

#: Optional keys per section: a supplied section must carry every
#: non-optional key, and may add these.
MARINE_OPTIONAL_INPUTS: dict[str, frozenset[str]] = {
    "chemistry_data": frozenset({"baseline_ph"}),
    "temperature_data": frozenset({"p90_anomaly_c", "days_at_or_above_p90"}),
}

#: Global mean open-ocean surface pH(T) for year 2000, 60degN-60degS:
#: 8.07 +/- 0.02 (Jiang, Carter, Feely, Lauvset & Olsen 2019, "Surface ocean
#: pH and buffer capacity: past, present and future", Sci. Rep. 9:18624,
#: doi:10.1038/s41598-019-55039-4).  Used as the acidification baseline only
#: when the caller supplies no local ``baseline_ph``.
GLOBAL_MEAN_SURFACE_PH: float = 8.07

#: U.S. EPA marine pH criterion (Quality Criteria for Water, 1986, "Gold
#: Book", EPA 440/5-86-001): marine aquatic-life pH within 6.5-8.5 and not
#: varied more than 0.2 units outside the naturally occurring range.  The
#: acid-direction departure is the acidification event line.
EPA_MARINE_PH_DEPARTURE: float = 0.2

#: NOAA Coral Reef Watch Bleaching Alert Area levels (5 km methodology,
#: coralreefwatch.noaa.gov/product/5km/methodology.php, updated Dec 2023).
#: HotSpot = SST minus the maximum monthly mean (MMM) climatology, degC;
#: DHW = degree heating weeks (degC-weeks over the trailing 12 weeks).
#: Each row: (alert level name, minimum DHW) -- all rows above "watch"
#: additionally require HotSpot >= 1.
CRW_ALERT_LEVELS: tuple[tuple[str, float], ...] = (
    ("alert_level_5", 20.0),
    ("alert_level_4", 16.0),
    ("alert_level_3", 12.0),
    ("alert_level_2", 8.0),
    ("alert_level_1", 4.0),
    ("warning", 0.0),
)

#: Hobday et al. (2016, Prog. Oceanogr. 141:227-238) marine heatwave
#: definition: SST above the seasonally varying 90th-percentile climatology
#: for at least 5 consecutive days.
HOBDAY_MIN_DAYS: float = 5.0

#: Hobday et al. (2018, Oceanography 31(2):162-173,
#: doi:10.5670/oceanog.2018.205) categories: the anomaly expressed in
#: multiples of the climatology-to-90th-percentile difference.
HOBDAY_CATEGORIES: tuple[tuple[str, float], ...] = (
    ("extreme", 4.0),
    ("severe", 3.0),
    ("strong", 2.0),
    ("moderate", 1.0),
)


class EcosystemHealth(Enum):
    """Ecosystem health status."""

    THRIVING = "thriving"
    HEALTHY = "healthy"
    STRESSED = "stressed"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    COLLAPSED = "collapsed"


@dataclass
class BiodiversityPredictionResult:
    """Marine biodiversity prediction results."""

    ecosystem_threatened: bool
    confidence: float
    health_status: str

    species_decline_detected: bool = False
    coral_bleaching_detected: bool = False
    ocean_acidification: bool = False
    marine_heatwave: bool = False

    biodiversity_index: float | None = None
    species_richness: int | None = None
    population_trend: str | None = None

    ph_level: float | None = None
    ph_baseline: float | None = None
    ph_baseline_basis: str | None = None
    temperature_anomaly_c: float | None = None
    coral_alert_level: str | None = None
    heatwave_category: str | None = None
    heatwave_basis: str | None = None

    threatened_species: list[str] = field(default_factory=list)
    conservation_actions: list[str] = field(default_factory=list)

    # Partial-input contract: True when part of the record was set aside;
    # ``set_aside`` names each piece and why.
    partial: bool = False
    set_aside: list[dict[str, Any]] = field(default_factory=list)


class CoralBleachingDetector:
    """Coral bleaching detection per NOAA Coral Reef Watch alert levels.

    ``sst_anomaly_c`` is read as the CRW HotSpot (SST minus the maximum
    monthly mean climatology); with ``degree_heating_weeks`` the pair maps
    onto the published Bleaching Alert Area ladder (see
    :data:`CRW_ALERT_LEVELS`).  The retired rule fired on either signal
    alone; CRW requires both (HotSpot >= 1 degC *and* DHW >= 4) before
    reef-wide bleaching is expected.
    """

    decision_source = DecisionSource(
        kind="published",
        description=(
            "Bleaching fires at NOAA Coral Reef Watch Bleaching Alert Level 1 "
            "or above: HotSpot >= 1 degC (SST anomaly above the maximum "
            "monthly mean) AND degree heating weeks >= 4; severity follows "
            "the published alert ladder (warning < alert_level_1..5)."
        ),
        citation=(
            "NOAA Coral Reef Watch, Daily Global 5km Satellite Coral Bleaching "
            "Heat Stress Monitoring methodology (Bleaching Alert Area table, "
            "updated Dec 2023), coralreefwatch.noaa.gov/product/5km/methodology.php"
        ),
    )

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)

    def detect_coral_bleaching(self, coral_data: dict[str, Any]) -> dict[str, Any]:
        """Detect coral bleaching heat stress per the CRW alert ladder."""
        hotspot_c = float(coral_data.get("sst_anomaly_c", 0.0) or 0.0)
        degree_heating_weeks = float(coral_data.get("degree_heating_weeks", 0.0) or 0.0)

        if hotspot_c <= 0.0:
            alert_level = "no_stress"
        elif hotspot_c < 1.0:
            alert_level = "watch"
        else:
            alert_level = "warning"
            for name, min_dhw in CRW_ALERT_LEVELS:
                if degree_heating_weeks >= min_dhw:
                    alert_level = name
                    break

        # Alert Level 1 is CRW's "risk of reef-wide bleaching"; Level 2 and
        # above add mortality risk.
        bleaching_detected = alert_level.startswith("alert_level_")
        if alert_level in ("alert_level_2", "alert_level_3", "alert_level_4", "alert_level_5"):
            severity = "severe"
        elif alert_level == "alert_level_1":
            severity = "moderate"
        else:
            severity = "mild"

        return {
            "bleaching_detected": bleaching_detected,
            "severity": severity,
            "alert_level": alert_level,
            "hotspot_c": hotspot_c,
            "degree_heating_weeks": degree_heating_weeks,
        }


class MarineBiodiversityDetector:
    """Comprehensive marine biodiversity monitoring system."""

    decision_source = (
        CoralBleachingDetector.decision_source,
        DecisionSource(
            kind="published",
            description=(
                "Ocean acidification fires when measured pH is more than 0.2 "
                "units below the natural baseline (supplied baseline_ph, else "
                "the global mean surface pH of 8.07)."
            ),
            citation=(
                "U.S. EPA Quality Criteria for Water 1986 (Gold Book, EPA "
                "440/5-86-001), marine pH criterion (6.5-8.5, <= 0.2 units "
                "outside the natural range); global baseline from Jiang et al. "
                "(2019), Sci. Rep. 9:18624, doi:10.1038/s41598-019-55039-4"
            ),
        ),
        DecisionSource(
            kind="published",
            description=(
                "Marine heatwave per Hobday et al. (2016): anomaly at or above "
                "the local seasonally varying 90th percentile for >= 5 days, "
                "when p90_anomaly_c and days_at_or_above_p90 are supplied; "
                "categories per Hobday et al. (2018)."
            ),
            citation=(
                "Hobday et al. (2016), Prog. Oceanogr. 141:227-238, "
                "doi:10.1016/j.pocean.2015.12.014; Hobday et al. (2018), "
                "Oceanography 31(2):162-173, doi:10.5670/oceanog.2018.205"
            ),
        ),
        DecisionSource(
            kind="experimental",
            description=(
                "With only a point SST anomaly (no local percentile "
                "climatology) the published marine-heatwave definition cannot "
                "be evaluated; the legacy 2.0 degC screen is retained and the "
                "result is labelled heatwave_basis='screening_2c_unsourced'."
            ),
        ),
    )

    def __init__(self) -> None:
        """Initialize the instance."""
        self.coral_detector = CoralBleachingDetector()
        self.logger = logging.getLogger(__name__)

    def predict_biodiversity_threat(
        self, marine_data: dict[str, Any]
    ) -> BiodiversityPredictionResult:
        """Predict marine ecosystem threats.

        Three indicators are assessed from whatever is measured: coral
        bleaching (``coral_data``), ocean acidification (``chemistry_data.ph``)
        and a marine heatwave (``temperature_data.anomaly_c``).  The ecosystem
        is threatened when any assessed indicator fires, and ``confidence`` is
        the fraction of assessed indicators that fire.

        Unusable pieces are set aside and reported on the result
        (``partial``, ``set_aside``): an unknown key or section, a non-finite
        or non-physical value (see :data:`MARINE_INPUTS`), and a section
        missing a required field -- its indicator cannot be graded (the coral
        model read a missing value as 0, so ``{}`` or a misspelt key graded as
        a measured all-clear).  The other indicators are still assessed.

        Raises:
            InputModalityError: If the record is not a mapping.
            InvalidObservationError: When no indicator can be assessed.
        """
        if not isinstance(marine_data, dict):
            raise InputModalityError(
                f"predict_biodiversity_threat expects a mapping, got {type(marine_data).__name__}"
            )
        grader = InputGrader()
        graded = grader.grade_sections(marine_data, MARINE_INPUTS)
        for section in list(graded):
            optional = MARINE_OPTIONAL_INPUTS.get(section, frozenset())
            missing = sorted(
                key
                for key in MARINE_INPUTS[section]
                if key not in optional and key not in graded[section]
            )
            if missing:
                grader.discard(
                    section, f"missing required {missing}; this indicator cannot be graded"
                )
                del graded[section]
        grader.require_usable(
            graded,
            "no indicator measured: supply at least one of "
            f"{sorted(MARINE_INPUTS)}; an unassessed ecosystem is not a healthy one",
        )
        marine_data = graded

        result = BiodiversityPredictionResult(
            ecosystem_threatened=False,
            confidence=0.0,
            health_status="healthy",
            partial=grader.partial,
            set_aside=list(grader.set_aside),
        )
        assessed = 0

        if "coral_data" in marine_data:
            coral_result = self.coral_detector.detect_coral_bleaching(marine_data["coral_data"])
            result.coral_bleaching_detected = coral_result["bleaching_detected"]
            result.coral_alert_level = coral_result["alert_level"]
            assessed += 1

        # pH is assessed only when measured: a chemistry section without
        # ``ph`` used to report pH 8.1 that nobody measured.  The event line
        # is the EPA marine criterion (Gold Book, 1986): more than 0.2 pH
        # units below the natural baseline -- the supplied local
        # ``baseline_ph``, else the global mean surface pH of Jiang et al.
        # (2019).
        if marine_data.get("chemistry_data", {}).get("ph") is not None:
            chemistry = marine_data["chemistry_data"]
            ph_level = float(chemistry["ph"])
            baseline = chemistry.get("baseline_ph")
            if baseline is not None:
                result.ph_baseline = float(baseline)
                result.ph_baseline_basis = "supplied local baseline_ph"
            else:
                result.ph_baseline = GLOBAL_MEAN_SURFACE_PH
                result.ph_baseline_basis = (
                    "global mean surface pH 8.07 (Jiang et al. 2019, "
                    "doi:10.1038/s41598-019-55039-4); supply baseline_ph for "
                    "the local natural range"
                )
            result.ph_level = ph_level
            result.ocean_acidification = ph_level < result.ph_baseline - EPA_MARINE_PH_DEPARTURE
            assessed += 1

        if marine_data.get("temperature_data", {}).get("anomaly_c") is not None:
            temperature = marine_data["temperature_data"]
            temp_anomaly = float(temperature["anomaly_c"])
            result.temperature_anomaly_c = temp_anomaly
            p90 = temperature.get("p90_anomaly_c")
            days = temperature.get("days_at_or_above_p90")
            if p90 is not None and days is not None:
                # Hobday et al. (2016): >= 5 consecutive days above the local
                # seasonally varying 90th percentile; categories per Hobday
                # et al. (2018) as multiples of the climatology-to-p90 gap.
                p90 = float(p90)
                result.heatwave_basis = "hobday_2016"
                result.marine_heatwave = temp_anomaly >= p90 and float(days) >= HOBDAY_MIN_DAYS
                if result.marine_heatwave:
                    multiple = temp_anomaly / p90
                    for name, min_multiple in HOBDAY_CATEGORIES:
                        if multiple >= min_multiple:
                            result.heatwave_category = name
                            break
            else:
                # No published single-point threshold exists; the 2.0 degC
                # screen is retained, labelled, until the caller supplies the
                # local percentile climatology the published definition needs.
                result.heatwave_basis = "screening_2c_unsourced"
                result.marine_heatwave = temp_anomaly > 2.0
            assessed += 1

        threat_indicators = sum(
            [
                result.coral_bleaching_detected,
                result.ocean_acidification,
                result.marine_heatwave,
            ]
        )

        if threat_indicators >= 2:
            result.health_status = EcosystemHealth.CRITICAL.value
        elif threat_indicators >= 1:
            result.health_status = EcosystemHealth.STRESSED.value
        # One decision with the status: acidification plus a heatwave used to
        # report health "critical" yet ecosystem_threatened False (only coral
        # bleaching set it), with confidence a constant 0.8 or 0.
        result.ecosystem_threatened = threat_indicators >= 1
        result.confidence = threat_indicators / assessed if assessed else 0.0

        result.conservation_actions = self._generate_conservation_actions(result)

        return result

    def _generate_conservation_actions(self, result: BiodiversityPredictionResult) -> list[str]:
        """Generate conservation recommendations."""
        actions = []

        if result.coral_bleaching_detected:
            actions.append("Reduce local stressors on reef ecosystems")
            actions.append("Monitor coral health continuously")

        if result.ocean_acidification:
            actions.append("Support CO2 emission reduction efforts")

        return actions
