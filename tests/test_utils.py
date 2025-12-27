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

"""
Test utility functions
"""

import numpy as np

from omni_anomaly_engine.utils import (
    compress_information,
    compute_time_dilation,
    decompress_information,
    detect_singularity,
    gravitational_lensing,
    normalize_data,
)


def test_normalize_standard(sample_data):
    """Test standard normalization"""
    normalized = normalize_data(sample_data, method="standard")

    assert normalized.shape == sample_data.shape
    assert abs(np.mean(normalized)) < 0.1
    assert abs(np.std(normalized) - 1.0) < 0.1


def test_normalize_minmax(sample_data):
    """Test minmax normalization"""
    normalized = normalize_data(sample_data, method="minmax")

    assert normalized.shape == sample_data.shape
    assert np.min(normalized) >= 0
    assert np.max(normalized) <= 1


def test_compress_decompress(sample_data):
    """Test data compression and decompression"""
    compressed, metadata = compress_information(sample_data)

    assert len(compressed) < sample_data.nbytes
    assert metadata["compression_ratio"] > 1

    decompressed = decompress_information(compressed, metadata)
    np.testing.assert_array_equal(sample_data, decompressed)


def test_gravitational_lensing(sample_data):
    """Test signal amplification"""
    amplified = gravitational_lensing(sample_data, amplification_factor=2.0)

    assert amplified.shape == sample_data.shape


def test_detect_singularity(sample_data):
    """Test singularity detection"""
    result = detect_singularity(sample_data)

    assert "singularity_detected" in result
    assert "singularity_count" in result
    assert isinstance(result["singularity_detected"], bool)


def test_compute_time_dilation():
    """Test time dilation computation"""
    priority_scores = np.array([0.1, 0.5, 0.9])
    dilation = compute_time_dilation(priority_scores)

    assert len(dilation) == len(priority_scores)
    assert all(dilation >= 1.0)
