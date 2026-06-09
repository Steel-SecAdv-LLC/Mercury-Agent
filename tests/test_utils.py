# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test utility functions."""

from __future__ import annotations

from typing import Any

import numpy as np

from omni_mercury_engine.utils import (
    compress_information,
    compute_time_dilation,
    decompress_information,
    detect_singularity,
    gravitational_lensing,
    normalize_data,
)


def test_normalize_standard(sample_data: Any) -> None:
    """Test standard normalization"""
    normalized = normalize_data(sample_data, method="standard")

    assert normalized.shape == sample_data.shape
    assert abs(np.mean(normalized)) < 0.1
    assert abs(np.std(normalized) - 1.0) < 0.1


def test_normalize_minmax(sample_data: Any) -> None:
    """Test minmax normalization"""
    normalized = normalize_data(sample_data, method="minmax")

    assert normalized.shape == sample_data.shape
    assert np.min(normalized) >= 0
    assert np.max(normalized) <= 1


def test_compress_decompress(sample_data: Any) -> None:
    """Test data compression and decompression"""
    compressed, metadata = compress_information(sample_data)

    assert len(compressed) < sample_data.nbytes
    assert metadata["compression_ratio"] > 1

    decompressed = decompress_information(compressed, metadata)
    np.testing.assert_array_equal(sample_data, decompressed)


def test_gravitational_lensing(sample_data: Any) -> None:
    """Test signal amplification"""
    amplified = gravitational_lensing(sample_data, amplification_factor=2.0)

    assert amplified.shape == sample_data.shape


def test_detect_singularity(sample_data: Any) -> None:
    """Test singularity detection"""
    result = detect_singularity(sample_data)

    assert "singularity_detected" in result
    assert "singularity_count" in result
    assert isinstance(result["singularity_detected"], bool)


def test_compute_time_dilation() -> None:
    """Test time dilation computation"""
    priority_scores = np.array([0.1, 0.5, 0.9])
    dilation = compute_time_dilation(priority_scores)

    assert len(dilation) == len(priority_scores)
    assert all(dilation >= 1.0)
