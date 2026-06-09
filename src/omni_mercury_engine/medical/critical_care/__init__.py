# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Critical Care Module.

Specialized critical care detection for humanitarian healthcare:
- Sepsis detection and SOFA scoring
- Neurocritical care (stroke, seizure, TBI, ICP)

Part of Mercury Agent Medical framework.
"""

from __future__ import annotations

from omni_mercury_engine.medical.critical_care.neurocritical_care import (
    ICPMonitor,
    NeurocriticalCarePredictor,
    NeurocriticalPredictionResult,
    NIHSSCalculator,
    SeizurePredictor,
    SeizureType,
    StrokeDetector,
    StrokeType,
)
from omni_mercury_engine.medical.critical_care.sepsis_detector import (
    QuickSOFACalculator,
    SepsisDetector,
    SepsisPredictionResult,
    SepsisProgressionPredictor,
    SepsisStage,
    SOFACalculator,
)

__all__ = [
    "ICPMonitor",
    "NIHSSCalculator",
    # Neurocritical
    "NeurocriticalCarePredictor",
    "NeurocriticalPredictionResult",
    "QuickSOFACalculator",
    "SOFACalculator",
    "SeizurePredictor",
    "SeizureType",
    # Sepsis
    "SepsisDetector",
    "SepsisPredictionResult",
    "SepsisProgressionPredictor",
    "SepsisStage",
    "StrokeDetector",
    "StrokeType",
]
