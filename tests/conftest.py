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

"""
Pytest configuration and fixtures
"""

import pytest
import numpy as np
import torch


@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    return np.random.randn(100, 10)


@pytest.fixture
def sample_tensor():
    """Generate sample tensor for testing"""
    return torch.randn(100, 10)


@pytest.fixture
def anomaly_data():
    """Generate data with known anomalies"""
    normal = np.random.randn(90, 10)
    anomalies = np.random.randn(10, 10) * 5
    return np.vstack([normal, anomalies])


@pytest.fixture
def biometric_sample():
    """Generate sample biometric data"""
    return {
        "image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "face_mesh": np.random.randn(468, 3),
    }
