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

"""
Test attention mechanism functionality
"""

import pytest

# Conditional torch import
try:
    import torch
    from torch import nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore
    nn = None  # type: ignore

# Skip all tests in this module if torch is not available
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")

# Conditional imports - only when torch is available
if HAS_TORCH:
    from omni_mercury_engine.ml.attention import (
        CrossModalAttention,
        MultiHeadDetectorAttention,
        SpatialAttention,
        TemporalAttention,
    )


def test_multihead_detector_attention_init():
    """Test multi-head detector attention initialization"""
    attention = MultiHeadDetectorAttention(embed_dim=64, num_heads=4)
    assert attention is not None


def test_multihead_detector_attention_forward():
    """Test multi-head detector attention forward pass"""
    attention = MultiHeadDetectorAttention(embed_dim=64, num_heads=4)

    query = torch.randn(2, 10, 64)
    key = torch.randn(2, 10, 64)
    value = torch.randn(2, 10, 64)

    output, weights = attention(query, key, value)

    assert output.shape == (2, 10, 64)
    assert weights.shape[0] == 2


def test_temporal_attention_init():
    """Test temporal attention initialization"""
    attention = TemporalAttention(hidden_dim=64, num_heads=4)
    assert attention.hidden_dim == 64
    assert attention.num_heads == 4


def test_temporal_attention_forward():
    """Test temporal attention forward pass"""
    attention = TemporalAttention(hidden_dim=64, num_heads=4)

    x = torch.randn(2, 10, 64)
    output, weights = attention(x)

    assert output.shape == (2, 10, 64)
    assert weights.shape[0] == 2


def test_spatial_attention_init():
    """Test spatial attention initialization"""
    attention = SpatialAttention(channels=32)
    assert attention is not None


def test_spatial_attention_forward():
    """Test spatial attention forward pass"""
    attention = SpatialAttention(channels=32)

    x = torch.randn(4, 32, 8, 8)
    output = attention(x)

    assert output.shape == (4, 32, 8, 8)


def test_crossmodal_attention_init():
    """Test cross-modal attention initialization"""
    attention = CrossModalAttention(dim1=32, dim2=64, hidden_dim=48)
    assert attention is not None


def test_crossmodal_attention_forward():
    """Test cross-modal attention forward pass"""
    attention = CrossModalAttention(dim1=32, dim2=64, hidden_dim=48)

    features1 = torch.randn(4, 1, 32)
    features2 = torch.randn(4, 1, 64)

    output, weights = attention(features1, features2)
    assert output.shape == (4, 1, 48)
