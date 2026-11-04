# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Meteorological, hydrological and severe-storm hazard detectors.

Physics/statistics cores for drought (SPI/SPEI), heatwave (percentile
climatology + Excess Heat Factor), atmospheric rivers (IVT + Ralph et al.
AR scale), lightning (Schultz et al. 2-sigma lightning jump), the
hurricane -> surge -> compound-flood cascade, the severe-storm
cluster (hail/SHIP, winter and ice storms, derecho criteria, dust
storms), and the EPA PM2.5 Air Quality Index. Those detectors work
untrained, carry literature citations, and fail loudly on inadequate input.

Also home to the three storm/hydrology detectors that historically sat in
``detectors/geological/`` (which keeps import shims): tornado (severe-storm
meteorology, NEXRAD radar morphology), hurricane (tropical meteorology, ERA5
wind-field physics) and flood (hydrology, precipitation-runoff physics) —
each with an optional trained checkpoint on top of its physics core.
"""

from __future__ import annotations

from omni_mercury_engine.detectors.meteorological.air_quality import (
    AirQualityAssessment,
    AirQualityDetector,
    pm25_aqi,
)
from omni_mercury_engine.detectors.meteorological.atmospheric_river_detector import (
    ARAssessmentResult,
    AREpisode,
    AtmosphericRiverDetector,
    IVTResult,
    compute_ivt,
)
from omni_mercury_engine.detectors.meteorological.derecho_detector import DerechoDetector
from omni_mercury_engine.detectors.meteorological.drought_detector import (
    DroughtAssessmentResult,
    DroughtCategory,
    DroughtDetector,
    classify_usdm,
    compute_spei,
    compute_spi,
    thornthwaite_pet,
)
from omni_mercury_engine.detectors.meteorological.dust_storm_detector import DustStormDetector
from omni_mercury_engine.detectors.meteorological.flood_detector import (
    FloodDetector,
    FloodPredictionResult,
    FloodSeverity,
    FloodType,
)
from omni_mercury_engine.detectors.meteorological.hail_detector import HailDetector
from omni_mercury_engine.detectors.meteorological.heatwave_detector import (
    HeatRiskCategory,
    HeatwaveAssessmentResult,
    HeatwaveDetector,
    HeatwaveEvent,
    HeatwaveSeverity,
    heat_alert_category,
    heat_index_f,
)
from omni_mercury_engine.detectors.meteorological.hurricane_detector import (
    CycloneType,
    HurricaneDetector,
    HurricanePredictionResult,
    SaffirSimpsonCategory,
)
from omni_mercury_engine.detectors.meteorological.lightning_detector import (
    LightningCell,
    LightningDetector,
    LightningJump,
    LightningJumpResult,
)
from omni_mercury_engine.detectors.meteorological.surge_flood_cascade import (
    CascadeAssessment,
    CascadeStage,
    EvidenceRecord,
    SurgeFloodCascade,
    SurgeSeries,
)
from omni_mercury_engine.detectors.meteorological.tornado_detector import (
    TornadoDetector,
    TornadoIntensity,
    TornadoPredictionResult,
    TornadoThreatLevel,
)
from omni_mercury_engine.detectors.meteorological.winter_storm_detector import WinterStormDetector

__all__ = [
    "ARAssessmentResult",
    "AREpisode",
    "AirQualityAssessment",
    "AirQualityDetector",
    "AtmosphericRiverDetector",
    "CascadeAssessment",
    "CascadeStage",
    "CycloneType",
    "DerechoDetector",
    "DroughtAssessmentResult",
    "DroughtCategory",
    "DroughtDetector",
    "DustStormDetector",
    "EvidenceRecord",
    "FloodDetector",
    "FloodPredictionResult",
    "FloodSeverity",
    "FloodType",
    "HailDetector",
    "HeatRiskCategory",
    "HeatwaveAssessmentResult",
    "HeatwaveDetector",
    "HeatwaveEvent",
    "HeatwaveSeverity",
    "HurricaneDetector",
    "HurricanePredictionResult",
    "IVTResult",
    "LightningCell",
    "LightningDetector",
    "LightningJump",
    "LightningJumpResult",
    "SaffirSimpsonCategory",
    "SurgeFloodCascade",
    "SurgeSeries",
    "TornadoDetector",
    "TornadoIntensity",
    "TornadoPredictionResult",
    "TornadoThreatLevel",
    "WinterStormDetector",
    "classify_usdm",
    "compute_ivt",
    "compute_spei",
    "compute_spi",
    "heat_alert_category",
    "heat_index_f",
    "pm25_aqi",
    "thornthwaite_pet",
]
