# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Import-compatibility shim: the tornado detector moved to the meteorological package.

Tornadoes are severe-storm meteorology (NEXRAD radar morphology, mesocyclone
and instability physics),
not geology; the module sat in ``detectors/geological/`` for historical
reasons only. The canonical module is
:mod:`omni_mercury_engine.detectors.meteorological.tornado_detector`; every public
name is re-exported here (listed in ``__all__``, the re-export mechanism
pyflakes, ruff and mypy agree on), so existing imports keep working. See
DEPRECATION.md.
"""

from __future__ import annotations

from omni_mercury_engine.detectors.meteorological.tornado_detector import (
    AtmosphericInstabilityAnalyzer,
    DopplerRadarAnalyzer,
    PressureGradientMonitor,
    RecursiveFeatureExtractor,
    ResonancePatternAnalyzer,
    TornadoDetector,
    TornadoIntensity,
    TornadoPredictionResult,
    TornadoThreatLevel,
    grade_weather_data,
)

__all__ = [
    "AtmosphericInstabilityAnalyzer",
    "DopplerRadarAnalyzer",
    "PressureGradientMonitor",
    "RecursiveFeatureExtractor",
    "ResonancePatternAnalyzer",
    "TornadoDetector",
    "TornadoIntensity",
    "TornadoPredictionResult",
    "TornadoThreatLevel",
    "grade_weather_data",
]
