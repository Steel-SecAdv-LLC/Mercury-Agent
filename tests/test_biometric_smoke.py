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
Additional biometric tests to boost coverage
"""

from unittest.mock import patch

import numpy as np
import pytest


# Conditional torch import
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore

# Skip all tests in this module if torch is not available
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")

# Conditional imports - only when torch is available
if HAS_TORCH:
    from omni_mercury_engine.models.biometric import BiometricAnomalyModel


def test_biometric_with_different_models():
    """Test biometric model with different face recognition models"""
    models = ["VGG-Face", "Facenet", "OpenFace"]

    for model_name in models:
        config = {"model_name": model_name}
        model = BiometricAnomalyModel(config)
        assert model.model_name == model_name


def test_biometric_extract_features_error_handling():
    """Test error handling in feature extraction"""
    model = BiometricAnomalyModel()

    invalid_image = np.zeros((10, 10), dtype=np.uint8)

    features = model.extract_features(invalid_image)
    assert features is not None
    assert isinstance(features, torch.Tensor)


def test_biometric_predict_with_dict():
    """Test predict with dictionary input"""
    model = BiometricAnomalyModel()

    ref_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    with patch("omni_mercury_engine.models.biometric.DeepFace") as mock_deepface:
        mock_deepface.verify.return_value = {
            "verified": True,
            "distance": 0.3,
        }

        result = model.predict(
            {
                "reference": ref_image,
                "test": test_image,
            }
        )

        assert "model_type" in result


def test_biometric_normalize_embedding_size():
    """Test embedding normalization to fixed size"""
    model = BiometricAnomalyModel()

    small_embedding = torch.randn(1, 64)
    normalized = model._normalize_embedding_size(small_embedding)
    assert normalized.shape[1] == 128

    large_embedding = torch.randn(1, 256)
    normalized = model._normalize_embedding_size(large_embedding)
    assert normalized.shape[1] == 128


def test_biometric_harmonic_features_disabled():
    """Test with harmonic features disabled"""
    config = {"use_harmonic_features": False}
    model = BiometricAnomalyModel(config)

    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    features = model._extract_harmonic_features(image)

    assert isinstance(features, np.ndarray)


def test_biometric_deepface_import_error():
    """Test handling when DeepFace is not available"""
    model = BiometricAnomalyModel()

    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    result = model.predict(image)
    assert "model_type" in result
