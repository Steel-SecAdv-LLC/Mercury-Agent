"""
OMNI ♱ AVA (O♱A)
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

"""
Detector module for OMNI ♱ AVA

Provides specialized anomaly detectors for different data types and domains.

Includes:
    - Base detectors (statistical, temporal, spatial, dimensional)
    - Visual anomaly detectors (PatchCore, PaDiM, STFPM, Reverse Distillation, CFlow)
    - Vision-Language Model detectors (AnyAnomaly, LAVAD)
"""

from omni_anomaly_engine.detectors.dimensional import DimensionalAnalyzer
from omni_anomaly_engine.detectors.directive import SigmaDirectiveDetector
from omni_anomaly_engine.detectors.spatial import SpatialAnomalyDetector
from omni_anomaly_engine.detectors.statistical import (
    StatisticalAnomalyDetector,
)
from omni_anomaly_engine.detectors.temporal import TemporalAnomalyDetector

# SOTA Visual Anomaly Detection
from omni_anomaly_engine.detectors.visual import (
    PatchCoreDetector,
    PaDiMDetector,
    STFPMDetector,
    ReverseDistillationDetector,
    CFlowDetector,
    BaseVisualDetector,
)

# Vision-Language Model Detectors
from omni_anomaly_engine.detectors.vlm import (
    AnyAnomalyDetector,
    LAVADDetector,
    BaseVLMDetector,
)

__all__ = [
    # Base detectors
    "DimensionalAnalyzer",
    "SigmaDirectiveDetector",
    "SpatialAnomalyDetector",
    "StatisticalAnomalyDetector",
    "TemporalAnomalyDetector",
    # Visual detectors
    "PatchCoreDetector",
    "PaDiMDetector",
    "STFPMDetector",
    "ReverseDistillationDetector",
    "CFlowDetector",
    "BaseVisualDetector",
    # VLM detectors
    "AnyAnomalyDetector",
    "LAVADDetector",
    "BaseVLMDetector",
]
