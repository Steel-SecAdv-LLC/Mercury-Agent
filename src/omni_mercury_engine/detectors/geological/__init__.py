# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Geological hazard detectors.

Comprehensive geological anomaly detection for humanitarian early warning:
seismology (earthquake), seismogenic marine hazards (tsunami), volcanology,
slope stability (landslide, rockfall, avalanche), ground deformation
(subsidence) and the post-fire debris-flow cascade.

The tornado, hurricane and flood detectors are **meteorology and hydrology**
and now live in :mod:`omni_mercury_engine.detectors.meteorological`; the
meteor detector is **space science** and lives in
:mod:`omni_mercury_engine.space.meteor_detector`. Their names are re-exported
here so existing imports keep working (see DEPRECATION.md).
"""

from __future__ import annotations

from omni_mercury_engine.detectors.geological.disaster_detectors import (
    EarthquakeDetector,
    EarthquakeMagnitude,
    EarthquakePredictionResult,
    MeteorDetector,
    MeteorPredictionResult,
    MeteorThreatLevel,
    SolarFlareClass,
    SolarFlareDetector,
    SolarFlarePredictionResult,
    TsunamiDetector,
    TsunamiPredictionResult,
    TsunamiSeverity,
)
from omni_mercury_engine.detectors.meteorological.flood_detector import (
    FloodDetector,
    FloodPredictionResult,
    FloodSeverity,
    FloodType,
)
from omni_mercury_engine.detectors.meteorological.hurricane_detector import (
    CycloneType,
    HurricaneDetector,
    HurricanePredictionResult,
    SaffirSimpsonCategory,
)
from omni_mercury_engine.detectors.meteorological.tornado_detector import (
    TornadoDetector,
    TornadoIntensity,
    TornadoPredictionResult,
    TornadoThreatLevel,
)

__all__ = [
    "CycloneType",
    "EarthquakeDetector",
    "EarthquakeMagnitude",
    "EarthquakePredictionResult",
    "FloodDetector",
    "FloodPredictionResult",
    "FloodSeverity",
    "FloodType",
    "HurricaneDetector",
    "HurricanePredictionResult",
    "MeteorDetector",
    "MeteorPredictionResult",
    "MeteorThreatLevel",
    "SaffirSimpsonCategory",
    "SolarFlareClass",
    "SolarFlareDetector",
    "SolarFlarePredictionResult",
    "TornadoDetector",
    "TornadoIntensity",
    "TornadoPredictionResult",
    "TornadoThreatLevel",
    "TsunamiDetector",
    "TsunamiPredictionResult",
    "TsunamiSeverity",
]
