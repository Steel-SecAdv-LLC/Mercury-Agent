"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Critical Care Module

Specialized critical care detection for humanitarian healthcare:
- Sepsis detection and SOFA scoring
- Neurocritical care (stroke, seizure, TBI, ICP)

Part of Mercury Agent Medical framework.
"""

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
