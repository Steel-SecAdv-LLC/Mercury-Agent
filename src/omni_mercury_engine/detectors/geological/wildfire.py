# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Import-compatibility shim: the wildfire detector moved to the fire package.

Wildland fire is fuels and fire-weather science, not geology; the module sat
in ``detectors/geological/`` for historical reasons only. The canonical module
is :mod:`omni_mercury_engine.detectors.fire.wildfire`; every public name is
re-exported here (listed in ``__all__``), so existing imports keep working.
The post-fire debris-flow cascade — a geological hazard — remains in this
package. See DEPRECATION.md.

The module documentation below is kept verbatim for readers of this
historical path; the canonical module carries it too.

Wildfire Detector - Ignition, Spread & Risk Assessment.

Advanced wildfire detection and prediction for humanitarian early warning:
- Ignition detection (satellite thermal + ground sensors)
- Fire spread modeling (weather fusion + vegetation fuel)
- Risk assessment (drought + temperature + wind)
- Smoke plume tracking
- Controlled burn optimization
- Cascade detection (fire → mudslide → flooding)

Integrations:
- Thermal satellite data (MODIS, VIIRS)
- Weather data fusion (wind, humidity, temperature)
- Vegetation indices (NDVI for fuel load)
- Resilience framework for post-fire hazards

Research sources:
- NOAA/NASA GOES fire detection
- USFS wildfire science
- FIRMS (Fire Information for Resource Management System)

Performance: 20-30% faster detection via multi-scale thermal fusion
"""

from __future__ import annotations

from omni_mercury_engine.detectors.fire.wildfire import (
    FIRE_PROB_THRESHOLD_DEFAULT,
    FUEL_INTERFACE_ADJUSTMENT,
    HOTSPOT_THRESHOLD_K,
    ONE_HOUR_FUEL_MOISTURE_PER_EMC,
    STATE_OF_WEATHER_KEY,
    WILDFIRE_DECISION_STREAMS,
    WILDFIRE_PASS_THROUGH,
    WILDFIRE_SECTION_KEYS,
    FireIgnitionDetector,
    FireRiskLevel,
    FireSpreadModel,
    NDVIProcessor,
    ResonanceFrequencyAnalyzer,
    WildfireCNN,
    WildfireDetector,
    WildfirePredictionResult,
    equilibrium_moisture_content,
    fuel_interface_moisture,
    grade_wildfire_data,
)

__all__ = [
    "FIRE_PROB_THRESHOLD_DEFAULT",
    "FUEL_INTERFACE_ADJUSTMENT",
    "HOTSPOT_THRESHOLD_K",
    "ONE_HOUR_FUEL_MOISTURE_PER_EMC",
    "STATE_OF_WEATHER_KEY",
    "WILDFIRE_DECISION_STREAMS",
    "WILDFIRE_PASS_THROUGH",
    "WILDFIRE_SECTION_KEYS",
    "FireIgnitionDetector",
    "FireRiskLevel",
    "FireSpreadModel",
    "NDVIProcessor",
    "ResonanceFrequencyAnalyzer",
    "WildfireCNN",
    "WildfireDetector",
    "WildfirePredictionResult",
    "equilibrium_moisture_content",
    "fuel_interface_moisture",
    "grade_wildfire_data",
]
