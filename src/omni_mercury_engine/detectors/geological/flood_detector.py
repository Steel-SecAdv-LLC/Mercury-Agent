# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Import-compatibility shim: the flood detector moved to the meteorological package.

Floods are hydrology and hydro-climate science (precipitation-runoff physics,
river-gauge and soil-saturation modelling),
not geology; the module sat in ``detectors/geological/`` for historical
reasons only. The canonical module is
:mod:`omni_mercury_engine.detectors.meteorological.flood_detector`; every public
name is re-exported here (listed in ``__all__``, the re-export mechanism
pyflakes, ruff and mypy agree on), so existing imports keep working. See
DEPRECATION.md.

The module documentation below is kept verbatim for readers of this
historical path; the canonical module carries it too.

Flood Detector - Multi-Parameter Flood Monitoring System.

Comprehensive flood detection for humanitarian early warning:
- Precipitation accumulation analysis
- River gauge monitoring
- Soil saturation models
- Topographic runoff prediction
- Refactoring engine for dynamic model optimization
- Cross-domain fusion with hurricane/tornado detectors

Integrations:
- USGS stream gauge network
- NWS precipitation data
- Satellite soil moisture observations
- 3R mechanism for adaptive flood modeling
- Dam/reservoir level monitoring

Research sources:
- NOAA National Weather Service
- USGS Water Resources
- FEMA flood mapping
- Academic research on hydrological modeling

⚠️ SIMULATION-BASED: For research/development. NOT a replacement for official
flood warnings (NWS, USGS). Always defer to official flood warnings.

Performance: Enhanced prediction via refactoring engine + dynamic optimization
"""

from __future__ import annotations

from omni_mercury_engine.detectors.meteorological.flood_detector import (
    FloodDetector,
    FloodPredictionOptimizer,
    FloodPredictionResult,
    FloodSeverity,
    FloodType,
    PrecipitationAnalyzer,
    RiverGaugeMonitor,
    SoilSaturationModel,
    grade_flood_data,
)

__all__ = [
    "FloodDetector",
    "FloodPredictionOptimizer",
    "FloodPredictionResult",
    "FloodSeverity",
    "FloodType",
    "PrecipitationAnalyzer",
    "RiverGaugeMonitor",
    "SoilSaturationModel",
    "grade_flood_data",
]
