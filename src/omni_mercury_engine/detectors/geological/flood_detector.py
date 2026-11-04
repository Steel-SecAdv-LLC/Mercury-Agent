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
