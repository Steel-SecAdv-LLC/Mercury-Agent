"""
Mercury Agent
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
Cardiology Module

Advanced cardiac anomaly detection for humanitarian healthcare:
- ECG rhythm analysis
- Arrhythmia classification
- Cardiac risk prediction

Part of Mercury Agent Medical framework.
"""

from omni_mercury_engine.medical.cardiology.cardiology_predictor import (
    ArrhythmiaType,
    CardiacBiomarkerAnalyzer,
    CardiologyPredictionResult,
    CardiologyPredictor,
    ECGRhythmAnalyzer,
    FraminghamRiskCalculator,
)

__all__ = [
    "ArrhythmiaType",
    "CardiacBiomarkerAnalyzer",
    "CardiologyPredictionResult",
    "CardiologyPredictor",
    "ECGRhythmAnalyzer",
    "FraminghamRiskCalculator",
]
