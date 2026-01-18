"""
Mercury Agent ♱
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


"""Tests for enhanced astrophysical anomaly detection model"""

import numpy as np

from omni_mercury_engine.models.astrophysical import AstrophysicalAnomalyModel


def test_astrophysical_black_hole_features():
    """Test black hole feature extraction"""
    model = AstrophysicalAnomalyModel()
    data = np.random.randn(8, 12)
    features = model.extract_features(data)

    assert features.shape[0] == 8
    assert features.shape[1] == 24


def test_event_horizon_detection():
    """Test event horizon state detection"""
    model = AstrophysicalAnomalyModel(mass_equivalent=1.0)

    close_data = np.random.randn(1, 10) * 0.01
    result = model.predict(close_data)

    assert result["anomaly_scores"][0] > 0.3


def test_gravitational_field_computation():
    """Test gravitational field strength computation"""
    model = AstrophysicalAnomalyModel()

    data = np.random.randn(5, 8)
    features = model.extract_features(data)

    grav_fields = features[:, 1]
    assert np.all(grav_fields >= 0)


def test_omni_code_integration():
    """Test Omni-Code integration in astrophysical model"""
    model = AstrophysicalAnomalyModel()
    data = np.random.randn(2, 6)

    result = model.predict(data)
    assert "anomaly_scores" in result
