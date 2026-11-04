# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Wildland fire science detectors.

Fuels, ignition and spread: NFDRS equilibrium fuel moisture, the wind-driven
spread model, NDVI fuel/drought/burn metrics, FIRMS brightness-temperature
physics and the optional trained ignition network. Wildland fire is its own
discipline — fuels and fire weather, neither geology nor meteorology — so the
detector lives here; ``detectors/geological/`` keeps an import shim (the
post-fire debris-flow cascade, a geological hazard, stays there). See
DEPRECATION.md.
"""

from __future__ import annotations

from omni_mercury_engine.detectors.fire.wildfire import (
    FireIgnitionDetector,
    FireRiskLevel,
    FireSpreadModel,
    NDVIProcessor,
    WildfireDetector,
    WildfirePredictionResult,
    equilibrium_moisture_content,
    fuel_interface_moisture,
    grade_wildfire_data,
)

__all__ = [
    "FireIgnitionDetector",
    "FireRiskLevel",
    "FireSpreadModel",
    "NDVIProcessor",
    "WildfireDetector",
    "WildfirePredictionResult",
    "equilibrium_moisture_content",
    "fuel_interface_moisture",
    "grade_wildfire_data",
]
