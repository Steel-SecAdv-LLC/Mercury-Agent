# Copyright (C) 2025 Steel Security Advisors LLC
"""Pandemic Detection & Response Module.

Comprehensive pandemic detection integrating:
- Real-time case surveillance and mutation tracking
- SEIR epidemiological forecasting with chaos detection
- QBM-based pathogen detection with MASINT fusion

Part of Mercury Agent Medical Interdiction framework.
"""

from __future__ import annotations

from omni_mercury_engine.medical.pandemic.bio_threats import BioThreatResult, PathogenDetector
from omni_mercury_engine.medical.pandemic.forecasting import EpidemicForecaster, PandemicForecast
from omni_mercury_engine.medical.pandemic.pandemic_detector import (
    CaseSurgeDetector,
    MutationTracker,
    OutbreakSeverity,
    PandemicDetector,
    PandemicPredictionResult,
    TransmissionNetworkAnalyzer,
    VariantConcern,
)

__all__ = [
    "BioThreatResult",
    # Case surveillance
    "CaseSurgeDetector",
    # Forecasting
    "EpidemicForecaster",
    "MutationTracker",
    # Enums
    "OutbreakSeverity",
    # Main detector
    "PandemicDetector",
    "PandemicForecast",
    "PandemicPredictionResult",
    # Bio-threats
    "PathogenDetector",
    "TransmissionNetworkAnalyzer",
    "VariantConcern",
]
