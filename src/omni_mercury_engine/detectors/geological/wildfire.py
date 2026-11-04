# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Import-compatibility shim: the wildfire detector moved to the fire package.

Wildland fire is fuels and fire-weather science, not geology; the module sat
in ``detectors/geological/`` for historical reasons only. The canonical module
is :mod:`omni_mercury_engine.detectors.fire.wildfire`; every public name is
re-exported here (listed in ``__all__``), so existing imports keep working.
The post-fire debris-flow cascade — a geological hazard — remains in this
package. See DEPRECATION.md.
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
