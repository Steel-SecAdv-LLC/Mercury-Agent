# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Interpretable visual anomaly analysis: detect, localise, classify and explain.

A NumPy/SciPy-only engine (no deep backbone or model download required) that
learns the normal appearance of an image domain and scrutinises new images
against it, with conformally calibrated decisions, pixel-precise localisation,
named anomaly types backed by evidence, image forensics, and capture-condition
awareness. See :class:`VisualAnomalyAnalyzer` for the full method.
"""

from __future__ import annotations

from omni_mercury_engine.vision.analyzer import (
    ANOMALY_TYPES,
    AnalyzerConfig,
    AnomalyRegion,
    StreamEvent,
    StreamingVisualMonitor,
    VisualAnomalyAnalyzer,
    VisualAnomalyReport,
)
from omni_mercury_engine.vision.features import (
    FEATURE_GROUPS,
    PatchFeatures,
    extract_patch_features,
)
from omni_mercury_engine.vision.imaging import (
    DEGRADATIONS,
    CanonicalImage,
    EnvironmentReport,
    ImageProvenance,
    assess_environment,
    degrade,
    to_canonical,
)

__all__ = [
    "ANOMALY_TYPES",
    "DEGRADATIONS",
    "FEATURE_GROUPS",
    "AnalyzerConfig",
    "AnomalyRegion",
    "CanonicalImage",
    "EnvironmentReport",
    "ImageProvenance",
    "PatchFeatures",
    "StreamEvent",
    "StreamingVisualMonitor",
    "VisualAnomalyAnalyzer",
    "VisualAnomalyReport",
    "assess_environment",
    "degrade",
    "extract_patch_features",
    "to_canonical",
]
