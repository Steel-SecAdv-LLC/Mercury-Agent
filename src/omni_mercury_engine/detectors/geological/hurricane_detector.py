# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Import-compatibility shim: the hurricane detector moved to the meteorological package.

Tropical cyclones are meteorology (ERA5 wind-field physics, Saffir-Simpson
intensity, sea-surface-temperature coupling),
not geology; the module sat in ``detectors/geological/`` for historical
reasons only. The canonical module is
:mod:`omni_mercury_engine.detectors.meteorological.hurricane_detector`; every public
name is re-exported here (listed in ``__all__``, the re-export mechanism
pyflakes, ruff and mypy agree on), so existing imports keep working. See
DEPRECATION.md.

The module documentation below is kept verbatim for readers of this
historical path; the canonical module carries it too.

Hurricane/Cyclone/Typhoon Detector - Tropical Cyclone Monitoring System.

Comprehensive tropical cyclone detection for humanitarian early warning:
- Sea surface temperature (SST) monitoring
- Wind speed pattern analysis
- Barometric pressure tracking
- Saffir-Simpson scale classification
- Resonance engine for frequency amplification of storm signals
- Cross-domain fusion with flood/tornado detectors

Integrations:
- Satellite SST data processing (GOES, Himawari)
- Reconnaissance aircraft data (Hurricane Hunters)
- Buoy network observations
- 3R mechanism for adaptive storm tracking
- Storm surge prediction models

Research sources:
- NOAA National Hurricane Center (NHC)
- Joint Typhoon Warning Center (JTWC)
- World Meteorological Organization (WMO)
- Academic research on tropical cyclone intensification

⚠️ SIMULATION-BASED: For research/development. NOT a replacement for official
hurricane centers (NHC, JTWC). Always defer to official hurricane warnings.

Performance: Enhanced tracking via resonance frequency amplification + recursive analysis
"""

from __future__ import annotations

from omni_mercury_engine.detectors.meteorological.hurricane_detector import (
    CycloneType,
    HurricaneDetector,
    HurricanePredictionResult,
    PressureTracker,
    ResonanceFrequencyAmplifier,
    SaffirSimpsonCategory,
    SeaSurfaceTemperatureAnalyzer,
    WindPatternAnalyzer,
    grade_cyclone_data,
    intensification_rate_kt_24h,
)

__all__ = [
    "CycloneType",
    "HurricaneDetector",
    "HurricanePredictionResult",
    "PressureTracker",
    "ResonanceFrequencyAmplifier",
    "SaffirSimpsonCategory",
    "SeaSurfaceTemperatureAnalyzer",
    "WindPatternAnalyzer",
    "grade_cyclone_data",
    "intensification_rate_kt_24h",
]
