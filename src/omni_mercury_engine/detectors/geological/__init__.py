# Copyright (C) 2025 Steel Security Advisors LLC
"""Geological Hazard Detectors.

Comprehensive geological anomaly detection for humanitarian early warning.
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
from omni_mercury_engine.detectors.geological.flood_detector import (
    FloodDetector,
    FloodPredictionResult,
    FloodSeverity,
    FloodType,
)
from omni_mercury_engine.detectors.geological.hurricane_detector import (
    CycloneType,
    HurricaneDetector,
    HurricanePredictionResult,
    SaffirSimpsonCategory,
)
from omni_mercury_engine.detectors.geological.tornado_detector import (
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
