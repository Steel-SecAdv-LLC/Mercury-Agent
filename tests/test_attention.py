# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test attention mechanism functionality."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import pytest

# Probe for torch without binding it at module import — keeps the
# ``Module | None`` retypings off the file and lets ``pytestmark`` skip
# the suite cleanly when torch is absent.  ``TYPE_CHECKING`` makes mypy
# resolve the symbols regardless of runtime availability.
HAS_TORCH = importlib.util.find_spec("torch") is not None

if TYPE_CHECKING or HAS_TORCH:
    import torch

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


def test_multihead_detector_attention_init() -> None:
    """Test multi-head detector attention initialization"""
    attention = MultiHeadDetectorAttention(embed_dim=64, num_heads=4)
    assert attention is not None


def test_multihead_detector_attention_forward() -> None:
    """Test multi-head detector attention forward pass"""
    attention = MultiHeadDetectorAttention(embed_dim=64, num_heads=4)

    query = torch.randn(2, 10, 64)
    key = torch.randn(2, 10, 64)
    value = torch.randn(2, 10, 64)

    output, weights = attention(query, key, value)

    assert output.shape == (2, 10, 64)
    assert weights.shape[0] == 2


def test_temporal_attention_init() -> None:
    """Test temporal attention initialization"""
    attention = TemporalAttention(hidden_dim=64, num_heads=4)
    assert attention.hidden_dim == 64
    assert attention.num_heads == 4


def test_temporal_attention_forward() -> None:
    """Test temporal attention forward pass"""
    attention = TemporalAttention(hidden_dim=64, num_heads=4)

    x = torch.randn(2, 10, 64)
    output, weights = attention(x)

    assert output.shape == (2, 10, 64)
    assert weights.shape[0] == 2


def test_spatial_attention_init() -> None:
    """Test spatial attention initialization"""
    attention = SpatialAttention(channels=32)
    assert attention is not None


def test_spatial_attention_forward() -> None:
    """Test spatial attention forward pass"""
    attention = SpatialAttention(channels=32)

    x = torch.randn(4, 32, 8, 8)
    output = attention(x)

    assert output.shape == (4, 32, 8, 8)


def test_crossmodal_attention_init() -> None:
    """Test cross-modal attention initialization"""
    attention = CrossModalAttention(dim1=32, dim2=64, hidden_dim=48)
    assert attention is not None


def test_crossmodal_attention_forward() -> None:
    """Test cross-modal attention forward pass"""
    attention = CrossModalAttention(dim1=32, dim2=64, hidden_dim=48)

    features1 = torch.randn(4, 1, 32)
    features2 = torch.randn(4, 1, 64)

    output, weights = attention(features1, features2)
    assert output.shape == (4, 1, 48)
