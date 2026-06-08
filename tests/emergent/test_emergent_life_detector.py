# Copyright (C) 2025 Steel Security Advisors LLC
"""Comprehensive tests for Emergent Life Detector module."""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.emergent.emergent_life_detector import (
    EmergentLifeDetector,
    LifeDetectionResult,
    SETICosmicSignalAnalyzer,
)


class TestSETIAnalyzer:
    def test_natural_signal(self) -> None:
        analyzer = SETICosmicSignalAnalyzer()
        signal = np.random.randn(1000) * 0.5
        result = analyzer.detect_seti_anomaly(signal)
        assert "seti_confidence" in result
        assert isinstance(result["technosignatures"], list)


class TestEmergentLifeDetector:
    def test_comprehensive_detection(self) -> None:
        detector = EmergentLifeDetector()
        signal = np.random.randn(1000)
        result = detector.detect_emergent_life(signal, "comprehensive")
        assert isinstance(result, LifeDetectionResult)
        assert 0.0 <= result.confidence <= 1.0
