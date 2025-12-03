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

Uses DeterministicRNG for reproducible tests.
All test fixtures now use seeded random number generation
to ensure consistent test results across runs.
"""

import numpy as np
import pytest

from omni_anomaly_engine.utils.rng import (
    DeterministicRNG,
    get_global_rng,
    set_global_seed,
)

# Optional torch import for ML tests
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Default seed for reproducibility
DEFAULT_TEST_SEED = 42


@pytest.fixture(autouse=True)
def set_random_seed():
    """
    Set a deterministic seed before each test.

    This fixture runs automatically for all tests to ensure
    reproducibility and eliminate test flakiness from RNG.
    """
    set_global_seed(DEFAULT_TEST_SEED)
    np.random.seed(DEFAULT_TEST_SEED)
    if HAS_TORCH:
        torch.manual_seed(DEFAULT_TEST_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(DEFAULT_TEST_SEED)
    yield


@pytest.fixture
def deterministic_rng():
    """
    Provide a DeterministicRNG instance for tests.

    Use this fixture when you need explicit control over
    the RNG in your test.
    """
    return DeterministicRNG(seed=DEFAULT_TEST_SEED)


@pytest.fixture
def sample_data(deterministic_rng):
    """Generate sample data for testing using deterministic RNG."""
    return deterministic_rng.randn(100, 10)


@pytest.fixture
def sample_tensor(set_random_seed):
    """Generate sample tensor for testing (requires torch)"""
    if not HAS_TORCH:
        pytest.skip("torch not installed - skipping ML test")
    return torch.randn(100, 10)


@pytest.fixture
def anomaly_data(deterministic_rng):
    """Generate data with known anomalies using deterministic RNG."""
    normal = deterministic_rng.randn(90, 10)
    anomalies = deterministic_rng.randn(10, 10) * 5
    return np.vstack([normal, anomalies])


@pytest.fixture
def biometric_sample(deterministic_rng):
    """Generate sample biometric data using deterministic RNG."""
    return {
        "image": deterministic_rng.randint(0, 255, (224, 224, 3)).astype(np.uint8),
        "face_mesh": deterministic_rng.randn(468, 3),
    }


@pytest.fixture
def univariate_data(deterministic_rng):
    """Generate univariate time series data for testing."""
    return deterministic_rng.randn(1000)


@pytest.fixture
def multivariate_data(deterministic_rng):
    """Generate multivariate time series data for testing."""
    return deterministic_rng.randn(500, 20)


@pytest.fixture
def ecg_signal(deterministic_rng):
    """Generate synthetic ECG-like signal for medical tests."""
    t = np.linspace(0, 10, 5000)
    # Simple ECG-like waveform
    ecg = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t)
    ecg += deterministic_rng.randn(len(t)) * 0.1
    return ecg


@pytest.fixture
def threat_features(deterministic_rng):
    """Generate synthetic threat feature vectors for security tests."""
    return deterministic_rng.randn(256)


@pytest.fixture
def seismic_sequence(deterministic_rng):
    """Generate synthetic seismic sequence for geological tests."""
    return deterministic_rng.randn(100, 32)


@pytest.fixture
def thermal_data(deterministic_rng):
    """Generate synthetic thermal data for volcanic monitoring tests."""
    base_temp = 288.0  # 15°C in Kelvin
    return {
        "brightness_temperature_k": deterministic_rng.randn(100) * 10 + base_temp,
        "radiant_heat_mw": deterministic_rng.rand(1)[0] * 100,
    }


@pytest.fixture
def gas_emissions(deterministic_rng):
    """Generate synthetic gas emission data for volcanic tests."""
    return {
        "so2_tons_per_day": deterministic_rng.rand(1)[0] * 200 + 50,
        "co2_tons_per_day": deterministic_rng.rand(1)[0] * 1000 + 200,
    }


@pytest.fixture
def schumann_resonance(deterministic_rng):
    """Generate synthetic Schumann resonance data."""
    t = np.linspace(0, 1, 1000)
    # 7.83 Hz fundamental frequency with noise
    signal = np.sin(2 * np.pi * 7.83 * t) + deterministic_rng.randn(len(t)) * 0.1
    return signal


# Marker for slow tests
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "security: marks security tests")
    config.addinivalue_line("markers", "medical: marks medical domain tests")
    config.addinivalue_line("markers", "geological: marks geological domain tests")
