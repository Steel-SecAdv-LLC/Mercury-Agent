# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compatibility shim: moved to :mod:`omni_mercury_engine.detectors.drift`.

The module now lives with the other detectors; this shim keeps released
``omni_mercury_engine.ml.drift`` imports working unchanged. New code should
import from the new location.
"""

from __future__ import annotations

from omni_mercury_engine.detectors.drift import (
    ChiSquaredDriftDetector,
    CovarianceDriftDetector,
    DriftIncomparableError,
    DriftResult,
    DriftSeverity,
    DriftType,
    EnsembleDriftDetector,
    KolmogorovSmirnovDriftDetector,
    OnlineDriftDetector,
    PopulationStabilityIndexDetector,
    PrincipalAxisDriftDetector,
    create_drift_detector,
)

__all__ = [
    "ChiSquaredDriftDetector",
    "CovarianceDriftDetector",
    "DriftIncomparableError",
    "DriftResult",
    "DriftSeverity",
    "DriftType",
    "EnsembleDriftDetector",
    "KolmogorovSmirnovDriftDetector",
    "OnlineDriftDetector",
    "PopulationStabilityIndexDetector",
    "PrincipalAxisDriftDetector",
    "create_drift_detector",
]
