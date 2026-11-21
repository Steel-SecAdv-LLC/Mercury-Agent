# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
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


def test_a_fitted_directive_detector_copies_and_pickles(sample_data: Any) -> None:
    """It holds a ``threading.local``; ``copy.deepcopy`` and ``pickle`` raised ``TypeError``."""
    import copy
    import pickle

    import numpy as np

    detector = SigmaDirectiveDetector()
    detector.fit(sample_data)
    detector.clear_memory()
    expected = np.asarray(detector.detect(sample_data)["scores"])

    for twin in (copy.deepcopy(detector), pickle.loads(pickle.dumps(detector))):
        assert twin._thread_local is not detector._thread_local
        np.testing.assert_array_equal(np.asarray(twin.detect(sample_data)["scores"]), expected)
