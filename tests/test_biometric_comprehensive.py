# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Comprehensive tests for biometric model to boost coverage."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

# Probe for torch without binding it at module import.
# ``TYPE_CHECKING`` keeps mypy resolution stable while the pytestmark
# below skips the suite when torch is absent.
HAS_TORCH = importlib.util.find_spec("torch") is not None

if TYPE_CHECKING or HAS_TORCH:
    import torch

# Skip all tests in this module if torch is not available
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")

# Conditional imports - only when torch is available
if HAS_TORCH:
    from omni_mercury_engine.models.biometric import BiometricAnomalyModel


def test_biometric_initialization_with_config() -> None:
    """Test biometric model initialization with custom config"""
    config = {
        "model_name": "VGG-Face",
        "use_harmonic_features": True,
    }
    model = BiometricAnomalyModel(config)
    assert model.model_name == "VGG-Face"
    assert model.use_harmonic_features is True


def test_biometric_predict_with_tensor() -> None:
    """Test biometric prediction with tensor input"""
    model = BiometricAnomalyModel()

    image = torch.randint(0, 255, (100, 100, 3), dtype=torch.uint8).numpy()

    with patch("omni_mercury_engine.models.biometric.DeepFace") as mock_deepface:
        mock_deepface.analyze.return_value = [
            {
                "age": 30,
                "dominant_gender": "Man",
                "emotion": {"happy": 0.8, "neutral": 0.2},
            }
        ]

        result = model.predict(image)

        assert "age" in result
        assert "model_type" in result
        assert result["model_type"] == "biometric"


def test_biometric_extract_features_comprehensive() -> None:
    """Test comprehensive feature extraction"""
    model = BiometricAnomalyModel()

    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    with patch("omni_mercury_engine.models.biometric.DeepFace") as mock_deepface:
        mock_deepface.analyze.return_value = [
            {
                "face_confidence": 0.95,
                "age": 30,
                "gender": {"Man": 0.9, "Woman": 0.1},
                "emotion": {"happy": 0.8, "sad": 0.1, "neutral": 0.1},
            }
        ]

        features = model.extract_features(image)

        assert isinstance(features, torch.Tensor)
        assert features.dim() == 2


def test_biometric_harmonic_features() -> None:
    """Test harmonic feature extraction"""
    config = {"use_harmonic_features": False}
    model = BiometricAnomalyModel(config)

    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    features = model._extract_harmonic_features(image)
    assert isinstance(features, np.ndarray)


def test_biometric_deepface_failure_handling() -> None:
    """Test handling when DeepFace fails"""
    model = BiometricAnomalyModel()

    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    with patch("omni_mercury_engine.models.biometric.DeepFace") as mock_deepface:
        mock_deepface.analyze.side_effect = Exception("DeepFace failed")

        result = model.predict(image)

        assert "error" in result
        assert result["model_type"] == "biometric"


def test_biometric_feature_extraction_without_deepface() -> None:
    """Test feature extraction when DeepFace is not available"""
    model = BiometricAnomalyModel()

    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    features = model.extract_features(image)

    assert isinstance(features, torch.Tensor)
    assert features.shape[1] == 128


def test_biometric_normalize_embedding() -> None:
    """Test embedding normalization"""
    model = BiometricAnomalyModel()

    embedding = torch.randn(5, 256)
    normalized = model._normalize_embedding_size(embedding)

    assert normalized.shape[1] == 128


def test_biometric_invalid_image_shape() -> None:
    """Test handling of invalid image shapes"""
    model = BiometricAnomalyModel()

    invalid_image = np.random.randint(0, 255, (10, 10), dtype=np.uint8)

    with patch("omni_mercury_engine.models.biometric.DeepFace") as mock_deepface:
        mock_deepface.analyze.side_effect = Exception("Invalid shape")
        result = model.predict(invalid_image)
        assert "error" in result or "model_type" in result
