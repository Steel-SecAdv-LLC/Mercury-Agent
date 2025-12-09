"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""
from __future__ import annotations

"""
Geological Hazard Detectors

Comprehensive geological anomaly detection for humanitarian early warning.
"""

from omni_anomaly_engine.detectors.geological.disaster_detectors import (
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
from omni_anomaly_engine.detectors.geological.flood_detector import (
    FloodDetector,
    FloodPredictionResult,
    FloodSeverity,
    FloodType,
)
from omni_anomaly_engine.detectors.geological.hurricane_detector import (
    CycloneType,
    HurricaneDetector,
    HurricanePredictionResult,
    SaffirSimpsonCategory,
)
from omni_anomaly_engine.detectors.geological.tornado_detector import (
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
