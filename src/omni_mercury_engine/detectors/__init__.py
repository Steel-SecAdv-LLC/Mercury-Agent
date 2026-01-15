"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations


"""
Detector module for Mercury Agent ♱

Provides specialized anomaly detectors for different data types and domains.

Includes:
    - Base detectors (statistical, temporal, spatial, dimensional)
    - Visual anomaly detectors (PatchCore, PaDiM, STFPM, Reverse Distillation, CFlow)
    - Vision-Language Model detectors (AnyAnomaly, LAVAD)
"""

from omni_mercury_engine.detectors.dimensional import DimensionalAnalyzer
from omni_mercury_engine.detectors.directive import SigmaDirectiveDetector
from omni_mercury_engine.detectors.spatial import SpatialAnomalyDetector
from omni_mercury_engine.detectors.statistical import StatisticalAnomalyDetector
from omni_mercury_engine.detectors.temporal import TemporalAnomalyDetector

# SOTA Visual Anomaly Detection
from omni_mercury_engine.detectors.visual import (
    BaseVisualDetector,
    CFlowDetector,
    PaDiMDetector,
    PatchCoreDetector,
    ReverseDistillationDetector,
    STFPMDetector,
)

# Vision-Language Model Detectors
from omni_mercury_engine.detectors.vlm import AnyAnomalyDetector, BaseVLMDetector, LAVADDetector


__all__ = [
    # VLM detectors
    "AnyAnomalyDetector",
    "BaseVLMDetector",
    "BaseVisualDetector",
    "CFlowDetector",
    # Base detectors
    "DimensionalAnalyzer",
    "LAVADDetector",
    "PaDiMDetector",
    # Visual detectors
    "PatchCoreDetector",
    "ReverseDistillationDetector",
    "STFPMDetector",
    "SigmaDirectiveDetector",
    "SpatialAnomalyDetector",
    "StatisticalAnomalyDetector",
    "TemporalAnomalyDetector",
]
