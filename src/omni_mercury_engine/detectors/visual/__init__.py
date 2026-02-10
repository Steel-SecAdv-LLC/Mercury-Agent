"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisors LLC

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
Visual Anomaly Detection Module

State-of-the-art visual anomaly detection algorithms for industrial,
medical, and surveillance applications. Implements cutting-edge methods
from recent literature (2023-2025).

Algorithms:
    - PatchCore: Memory-efficient patch-based detection with coreset selection
    - PaDiM: Patch Distribution Modeling with Mahalanobis distance
    - STFPM: Student-Teacher Feature Pyramid Matching
    - ReverseDistillation: Reverse knowledge distillation architecture
    - CFlow: Conditional Normalizing Flow for precise localization

Research Sources:
    - PatchCore: "Towards Total Recall in Industrial Anomaly Detection" (CVPR 2022)
    - PaDiM: "PaDiM: a Patch Distribution Modeling Framework" (ICPR 2020)
    - STFPM: "Student-Teacher Feature Pyramid Matching" (arXiv 2021)
    - Reverse Distillation: "Anomaly Detection via Reverse Distillation" (CVPR 2022)
    - CFlow-AD: "Real-Time Unsupervised Anomaly Detection" (WACV 2022)

Example:
    Basic usage::

        from omni_mercury_engine.detectors.visual import PatchCoreDetector
        import torch

        detector = PatchCoreDetector(backbone='wide_resnet50_2')

        # Fit on normal images
        normal_images = torch.randn(100, 3, 224, 224)
        detector.fit(normal_images)

        # Detect anomalies
        test_images = torch.randn(10, 3, 224, 224)
        results = detector.detect(test_images)
        print(f"Anomaly scores: {results['scores']}")
        print(f"Anomaly maps: {results['anomaly_maps'].shape}")
"""

from omni_mercury_engine.detectors.visual.backbone import FeatureExtractor, get_backbone
from omni_mercury_engine.detectors.visual.base_visual import (
    BaseVisualDetector,
    VisualDetectorConfig,
)
from omni_mercury_engine.detectors.visual.cflow import CFlowDetector
from omni_mercury_engine.detectors.visual.padim import PaDiMDetector
from omni_mercury_engine.detectors.visual.patchcore import PatchCoreDetector
from omni_mercury_engine.detectors.visual.reverse_distillation import ReverseDistillationDetector
from omni_mercury_engine.detectors.visual.stfpm import STFPMDetector

__all__ = [
    # Base classes
    "BaseVisualDetector",
    "CFlowDetector",
    "FeatureExtractor",
    "PaDiMDetector",
    # Detectors
    "PatchCoreDetector",
    "ReverseDistillationDetector",
    "STFPMDetector",
    "VisualDetectorConfig",
    "get_backbone",
]
