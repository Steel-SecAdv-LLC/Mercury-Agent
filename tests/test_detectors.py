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
Test detector modules
"""

from omni_mercury_engine.detectors.dimensional import DimensionalAnalyzer
from omni_mercury_engine.detectors.directive import SigmaDirectiveDetector
from omni_mercury_engine.detectors.spatial import SpatialAnomalyDetector
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
from omni_mercury_engine.detectors.temporal import TemporalAnomalyDetector


def test_statistical_detector(sample_data):
    """Test statistical anomaly detection"""
    detector = MercuryAnomalyDetector()
    detector.fit(sample_data)
    result = detector.detect(sample_data)

    assert "scores" in result
    assert "is_anomaly" in result
    assert len(result["scores"]) == len(sample_data)


def test_temporal_detector(sample_data):
    """Test temporal anomaly detection"""
    detector = TemporalAnomalyDetector()
    detector.fit(sample_data)
    result = detector.detect(sample_data)

    assert "scores" in result
    assert "is_anomaly" in result
    assert "trend_flags" in result


def test_spatial_detector(sample_data):
    """Test spatial anomaly detection"""
    detector = SpatialAnomalyDetector()
    detector.fit(sample_data)
    result = detector.detect(sample_data)

    assert "scores" in result
    assert "is_anomaly" in result
    assert "distance_scores" in result


def test_dimensional_detector(sample_data):
    """Test dimensional anomaly detection"""
    detector = DimensionalAnalyzer()
    detector.fit(sample_data)
    result = detector.detect(sample_data)

    assert "scores" in result
    assert "pca_errors" in result


def test_directive_detector(sample_data):
    """Test directive-based detection with quantum enhancements"""
    detector = SigmaDirectiveDetector()
    detector.fit(sample_data)
    result = detector.detect(sample_data)

    assert "scores" in result
    assert "pcp_scores" in result
