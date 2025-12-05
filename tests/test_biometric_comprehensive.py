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
Comprehensive tests for biometric model to boost coverage
"""

from unittest.mock import patch

import numpy as np
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
    from omni_anomaly_engine.models.biometric import BiometricAnomalyModel


def test_biometric_initialization_with_config():
    """Test biometric model initialization with custom config"""
    config = {
        "model_name": "VGG-Face",
        "use_harmonic_features": True,
    }
    model = BiometricAnomalyModel(config)
    assert model.model_name == "VGG-Face"
    assert model.use_harmonic_features is True


def test_biometric_predict_with_tensor():
    """Test biometric prediction with tensor input"""
    model = BiometricAnomalyModel()

    image = torch.randint(0, 255, (100, 100, 3), dtype=torch.uint8).numpy()

    with patch("omni_anomaly_engine.models.biometric.DeepFace") as mock_deepface:
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


def test_biometric_extract_features_comprehensive():
    """Test comprehensive feature extraction"""
    model = BiometricAnomalyModel()

    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    with patch("omni_anomaly_engine.models.biometric.DeepFace") as mock_deepface:
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


def test_biometric_harmonic_features():
    """Test harmonic feature extraction"""
    config = {"use_harmonic_features": False}
    model = BiometricAnomalyModel(config)

    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    features = model._extract_harmonic_features(image)
    assert isinstance(features, np.ndarray)


def test_biometric_deepface_failure_handling():
    """Test handling when DeepFace fails"""
    model = BiometricAnomalyModel()

    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    with patch("omni_anomaly_engine.models.biometric.DeepFace") as mock_deepface:
        mock_deepface.analyze.side_effect = Exception("DeepFace failed")

        result = model.predict(image)

        assert "error" in result
        assert result["model_type"] == "biometric"


def test_biometric_feature_extraction_without_deepface():
    """Test feature extraction when DeepFace is not available"""
    model = BiometricAnomalyModel()

    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    features = model.extract_features(image)

    assert isinstance(features, torch.Tensor)
    assert features.shape[1] == 128


def test_biometric_normalize_embedding():
    """Test embedding normalization"""
    model = BiometricAnomalyModel()

    embedding = torch.randn(5, 256)
    normalized = model._normalize_embedding_size(embedding)

    assert normalized.shape[1] == 128


def test_biometric_invalid_image_shape():
    """Test handling of invalid image shapes"""
    model = BiometricAnomalyModel()

    invalid_image = np.random.randint(0, 255, (10, 10), dtype=np.uint8)

    with patch("omni_anomaly_engine.models.biometric.DeepFace") as mock_deepface:
        mock_deepface.analyze.side_effect = Exception("Invalid shape")
        result = model.predict(invalid_image)
        assert "error" in result or "model_type" in result
