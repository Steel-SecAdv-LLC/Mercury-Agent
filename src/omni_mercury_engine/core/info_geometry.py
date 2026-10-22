# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compatibility shim: moved to :mod:`omni_mercury_engine.detectors.info_geometry`.

The module now lives with the other detectors; this shim keeps released
``omni_mercury_engine.core.info_geometry`` imports working unchanged. New code should
import from the new location.
"""

from __future__ import annotations

from omni_mercury_engine.detectors.info_geometry import (
    FisherInformationMatrix,
    FisherRaoAdaptiveThreshold,
    InformationGeometryDetector,
    NaturalGradient,
    StatisticalManifold,
)

__all__ = [
    "FisherInformationMatrix",
    "FisherRaoAdaptiveThreshold",
    "InformationGeometryDetector",
    "NaturalGradient",
    "StatisticalManifold",
]
