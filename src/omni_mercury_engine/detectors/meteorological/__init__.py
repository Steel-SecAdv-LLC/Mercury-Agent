# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Meteorological hazard detectors.

Physics/statistics cores for drought (SPI/SPEI), heatwave (percentile
climatology + Excess Heat Factor), atmospheric rivers (IVT + Ralph et al.
AR scale), lightning (Schultz et al. 2-sigma lightning jump), the
hurricane -> surge -> compound-flood cascade, and the severe-storm
cluster (hail/SHIP, winter and ice storms, derecho criteria, dust
storms).  All detectors work untrained, carry literature citations, and
fail loudly on inadequate input; none carries a neural network.
"""

from __future__ import annotations

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
from omni_mercury_engine.detectors.meteorological.winter_storm_detector import WinterStormDetector

__all__ = [
    "ARAssessmentResult",
    "AREpisode",
    "AtmosphericRiverDetector",
    "CascadeAssessment",
    "CascadeStage",
    "DerechoDetector",
    "DroughtAssessmentResult",
    "DroughtCategory",
    "DroughtDetector",
    "DustStormDetector",
    "EvidenceRecord",
    "HailDetector",
    "HeatRiskCategory",
    "HeatwaveAssessmentResult",
    "HeatwaveDetector",
    "HeatwaveEvent",
    "HeatwaveSeverity",
    "IVTResult",
    "LightningCell",
    "LightningDetector",
    "LightningJump",
    "LightningJumpResult",
    "SurgeFloodCascade",
    "SurgeSeries",
    "WinterStormDetector",
    "classify_usdm",
    "compute_ivt",
    "compute_spei",
    "compute_spi",
    "heat_alert_category",
    "heat_index_f",
    "thornthwaite_pet",
]
