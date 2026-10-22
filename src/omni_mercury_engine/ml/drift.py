# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Data and Model Drift Detection Module. (moved; shim).

The implementation now lives in :mod:`omni_mercury_engine.detectors.drift`; this shim re-exports it
so released ``omni_mercury_engine.ml.drift`` imports keep
working unchanged. New code should import from the new location.

Provides comprehensive drift detection capabilities:
- Statistical drift detection (Kolmogorov-Smirnov, Chi-squared, PSI)
- Covariance-structure drift (affine-invariant SPD distance), which catches
  changes that leave every marginal intact (decoupled sensors, a broken
  physical relationship) and so are invisible to per-feature tests
- Principal-axis mean shift (a per-axis KS test in the reference's PCA basis),
  which catches shifts along low-variance directions of correlated data that
  every per-feature test dilutes
- Feature drift monitoring for anomaly detection models
- Concept drift detection for classification models
- Online drift detection with adaptive windowing

Inspired by Alibi-Detect and Evidently frameworks.
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
