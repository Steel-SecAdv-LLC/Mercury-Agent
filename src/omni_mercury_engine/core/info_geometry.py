# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Information Geometry for Out-of-Distribution Detection -- Phase 4C. (moved; shim).

The implementation now lives in :mod:`omni_mercury_engine.detectors.info_geometry`; this shim re-exports it
so released ``omni_mercury_engine.core.info_geometry`` imports keep
working unchanged. New code should import from the new location.

Based on: IGEOOD - An Information Geometry Approach to Out-of-Distribution Detection
(ICLR 2022: https://openreview.net/pdf?id=mfwdY3U_9ea)

Phase 4C adds Fisher Information Metric Adaptive Thresholds:
  - FisherInformationMatrix: closed-form Gaussian and empirical score estimation
  - NaturalGradient: F-inverse times Euclidean gradient via Cholesky decomposition
  - FisherRaoAdaptiveThreshold: adaptive anomaly thresholds from the FIM
  - StatisticalManifold: manifold bookkeeping, geodesic distance, exponential map
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
