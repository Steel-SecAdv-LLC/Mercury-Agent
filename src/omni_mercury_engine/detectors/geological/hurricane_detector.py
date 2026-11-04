# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Import-compatibility shim: the hurricane detector moved to the meteorological package.

Tropical cyclones are meteorology (ERA5 wind-field physics, Saffir-Simpson
intensity, sea-surface-temperature coupling),
not geology; the module sat in ``detectors/geological/`` for historical
reasons only. The canonical module is
:mod:`omni_mercury_engine.detectors.meteorological.hurricane_detector`; every public
name is re-exported here (listed in ``__all__``, the re-export mechanism
pyflakes, ruff and mypy agree on), so existing imports keep working. See
DEPRECATION.md.
"""

from __future__ import annotations

from omni_mercury_engine.detectors.meteorological.hurricane_detector import (
    CycloneType,
    HurricaneDetector,
    HurricanePredictionResult,
    PressureTracker,
    ResonanceFrequencyAmplifier,
    SaffirSimpsonCategory,
    SeaSurfaceTemperatureAnalyzer,
    WindPatternAnalyzer,
    grade_cyclone_data,
    intensification_rate_kt_24h,
)

__all__ = [
    "CycloneType",
    "HurricaneDetector",
    "HurricanePredictionResult",
    "PressureTracker",
    "ResonanceFrequencyAmplifier",
    "SaffirSimpsonCategory",
    "SeaSurfaceTemperatureAnalyzer",
    "WindPatternAnalyzer",
    "grade_cyclone_data",
    "intensification_rate_kt_24h",
]
