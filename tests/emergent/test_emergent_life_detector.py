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
from __future__ import annotations

"""Comprehensive tests for Emergent Life Detector module."""

import numpy as np

from omni_anomaly_engine.emergent.emergent_life_detector import (
    EmergentLifeDetector,
    LifeDetectionResult,
    SETICosmicSignalAnalyzer,
)


class TestSETIAnalyzer:
    def test_natural_signal(self):
        analyzer = SETICosmicSignalAnalyzer()
        signal = np.random.randn(1000) * 0.5
        result = analyzer.detect_seti_anomaly(signal)
        assert "seti_confidence" in result
        assert isinstance(result["technosignatures"], list)


class TestEmergentLifeDetector:
    def test_comprehensive_detection(self):
        detector = EmergentLifeDetector()
        signal = np.random.randn(1000)
        result = detector.detect_emergent_life(signal, "comprehensive")
        assert isinstance(result, LifeDetectionResult)
        assert 0.0 <= result.confidence <= 1.0
