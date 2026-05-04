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

"""
Pandemic Detection & Response Module

Comprehensive pandemic detection integrating:
- Real-time case surveillance and mutation tracking
- SEIR epidemiological forecasting with chaos detection
- QBM-based pathogen detection with MASINT fusion

Part of Mercury Agent Medical Interdiction framework.
"""

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
