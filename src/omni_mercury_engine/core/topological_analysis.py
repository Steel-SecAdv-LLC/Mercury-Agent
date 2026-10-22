# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compatibility shim: moved to :mod:`omni_mercury_engine.detectors.topological_analysis`.

The module now lives with the other detectors; this shim keeps released
``omni_mercury_engine.core.topological_analysis`` imports working unchanged. New code should
import from the new location.
"""

from __future__ import annotations

from omni_mercury_engine.detectors.topological_analysis import (
    PersistenceDiagram,
    TopologicalAnomalyDetector,
    VietorisRipsFiltration,
    bottleneck_distance,
    wasserstein_distance_pd,
)

__all__ = [
    "PersistenceDiagram",
    "TopologicalAnomalyDetector",
    "VietorisRipsFiltration",
    "bottleneck_distance",
    "wasserstein_distance_pd",
]
