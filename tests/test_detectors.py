# Copyright (C) 2025 Steel Security Advisors LLC
"""Test detector modules."""

from __future__ import annotations

from typing import Any

from omni_mercury_engine.detectors.dimensional import DimensionalAnalyzer
from omni_mercury_engine.detectors.directive import SigmaDirectiveDetector
from omni_mercury_engine.detectors.spatial import SpatialAnomalyDetector
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
from omni_mercury_engine.detectors.temporal import TemporalAnomalyDetector


def test_statistical_detector(sample_data: Any) -> None:
    """Test statistical anomaly detection"""
    detector = MercuryAnomalyDetector()
    detector.fit(sample_data)
    result = detector.detect(sample_data)

    assert "scores" in result
    assert "is_anomaly" in result
    assert len(result["scores"]) == len(sample_data)


def test_temporal_detector(sample_data: Any) -> None:
    """Test temporal anomaly detection"""
    detector = TemporalAnomalyDetector()
    detector.fit(sample_data)
    result = detector.detect(sample_data)

    assert "scores" in result
    assert "is_anomaly" in result
    assert "trend_flags" in result


def test_spatial_detector(sample_data: Any) -> None:
    """Test spatial anomaly detection"""
    detector = SpatialAnomalyDetector()
    detector.fit(sample_data)
    result = detector.detect(sample_data)

    assert "scores" in result
    assert "is_anomaly" in result
    assert "distance_scores" in result


def test_dimensional_detector(sample_data: Any) -> None:
    """Test dimensional anomaly detection"""
    detector = DimensionalAnalyzer()
    detector.fit(sample_data)
    result = detector.detect(sample_data)

    assert "scores" in result
    assert "pca_errors" in result


def test_directive_detector(sample_data: Any) -> None:
    """Test directive-based detection with quantum enhancements"""
    detector = SigmaDirectiveDetector()
    detector.fit(sample_data)
    result = detector.detect(sample_data)

    assert "scores" in result
    assert "pcp_scores" in result
