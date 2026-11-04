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

The module documentation below is kept verbatim for readers of this
historical path; the canonical module carries it too.

Tornado Detector - Multi-Modal Severe Weather Monitoring.

Comprehensive tornado detection for humanitarian early warning:
- Doppler radar pattern analysis (mesocyclone detection)
- Atmospheric pressure gradient monitoring
- Temperature/humidity anomaly detection
- Historical tornado alley correlation
- FFT-based resonance pattern analysis (3R integration)
- Recursive feature extraction for multi-scale patterns

Integrations:
- Doppler radar velocity data processing
- Surface observation networks (ASOS, AWOS)
- Upper-air soundings for atmospheric instability
- 3R mechanism for self-healing monitoring networks
- Cross-domain fusion with hurricane/flood detectors

Research sources:
- NOAA Storm Prediction Center
- National Weather Service
- NSSL (National Severe Storms Laboratory)
- Academic research on mesocyclone detection

⚠️ SIMULATION-BASED: For research/development. NOT a replacement for official
weather services (NWS, SPC). Always defer to official tornado warnings.

Performance: Enhanced detection via FFT resonance + recursive feature extraction
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
