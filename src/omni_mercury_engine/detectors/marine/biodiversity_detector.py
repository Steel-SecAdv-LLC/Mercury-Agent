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

from omni_mercury_engine.core.exceptions import InputModalityError, InvalidObservationError
from omni_mercury_engine.detectors._hazard_input import finite_scalar

#: Every value ``predict_biodiversity_threat`` reads: section -> key -> (unit,
#: physical range).  Supplied values are checked first: a NaN failed every
#: threshold comparison and read as "healthy".
MARINE_INPUTS: dict[str, dict[str, tuple[str, float, float]]] = {
    "coral_data": {
        "sst_anomaly_c": ("degC", -15.0, 15.0),
        "degree_heating_weeks": ("degC-weeks", 0.0, 52.0),
    },
    "chemistry_data": {"ph": ("pH", 6.0, 9.0)},
    "temperature_data": {"anomaly_c": ("degC", -15.0, 15.0)},
}


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
    temperature_anomaly_c: float | None = None

    threatened_species: list[str] = field(default_factory=list)
    conservation_actions: list[str] = field(default_factory=list)


class CoralBleachingDetector:
    """Coral bleaching detection from temperature and stress indicators."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self.logger = logging.getLogger(__name__)

    def detect_coral_bleaching(self, coral_data: dict[str, Any]) -> dict[str, Any]:
        """Detect coral bleaching events."""
        sst_anomaly_c = coral_data.get("sst_anomaly_c", 0.0)
        degree_heating_weeks = coral_data.get("degree_heating_weeks", 0.0)

        bleaching_threshold = 1.0
        severe_threshold = 4.0

        bleaching_detected = sst_anomaly_c > bleaching_threshold or degree_heating_weeks > 4.0

        if degree_heating_weeks > severe_threshold:
            severity = "severe"
        elif degree_heating_weeks > 2.0:
            severity = "moderate"
        else:
            severity = "mild"

        return {
            "bleaching_detected": bleaching_detected,
            "severity": severity,
            "degree_heating_weeks": float(degree_heating_weeks),
        }


class MarineBiodiversityDetector:
    """Comprehensive marine biodiversity monitoring system."""

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

        Raises:
            InputModalityError: If the record or a section is not a mapping.
            InvalidObservationError: When no indicator is measured, a section
                has an unknown or missing field, or a value is non-finite or
                non-physical (see :data:`MARINE_INPUTS`).
        """
        if not isinstance(marine_data, dict):
            raise InputModalityError(
                f"predict_biodiversity_threat expects a mapping, got {type(marine_data).__name__}"
            )
        for section, fields in MARINE_INPUTS.items():
            if section not in marine_data:
                continue
            values = marine_data[section]
            if not isinstance(values, dict):
                raise InputModalityError(
                    f"{section} must be a mapping, got {type(values).__name__}"
                )
            # Every field of a supplied section is required: the coral model
            # read a missing value as 0, so ``{}`` or a misspelt key graded as
            # a measured all-clear.
            unknown = sorted(set(values) - set(fields))
            if unknown:
                raise InvalidObservationError(
                    f"{section} has unknown keys {unknown}; accepted: {sorted(fields)}"
                )
            missing = sorted(key for key in fields if values.get(key) is None)
            if missing:
                raise InvalidObservationError(f"{section} is missing {missing}")
            for key, (unit, lo, hi) in fields.items():
                finite_scalar(values[key], key, unit=unit, lo=lo, hi=hi)
        if not any(section in marine_data for section in MARINE_INPUTS):
            raise InvalidObservationError(
                "no indicator measured: supply at least one of "
                f"{sorted(MARINE_INPUTS)}; an unassessed ecosystem is not a healthy one"
            )

        result = BiodiversityPredictionResult(
            ecosystem_threatened=False, confidence=0.0, health_status="healthy"
        )
        assessed = 0

        if "coral_data" in marine_data:
            coral_result = self.coral_detector.detect_coral_bleaching(marine_data["coral_data"])
            result.coral_bleaching_detected = coral_result["bleaching_detected"]
            assessed += 1

        # pH is assessed only when measured: a chemistry section without
        # ``ph`` used to report pH 8.1 that nobody measured.
        if marine_data.get("chemistry_data", {}).get("ph") is not None:
            ph_level = float(marine_data["chemistry_data"]["ph"])
            result.ph_level = ph_level
            result.ocean_acidification = ph_level < 7.9
            assessed += 1

        if marine_data.get("temperature_data", {}).get("anomaly_c") is not None:
            temp_anomaly = float(marine_data["temperature_data"]["anomaly_c"])
            result.temperature_anomaly_c = temp_anomaly
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
