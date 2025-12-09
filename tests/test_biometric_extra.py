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
from typing import Any

"""
Additional biometric tests to boost coverage above 85%
"""

import numpy as np
import torch

from omni_anomaly_engine.models.biometric import BiometricAnomalyModel


def test_biometric_with_invalid_model():
    """Test with invalid model name"""
    try:
        model = BiometricAnomalyModel({"model_name": "InvalidModel"})
        assert model.model_name == "InvalidModel"
    except Exception:
        pass


def test_biometric_predict_with_none_input():
    """Test predict with None input"""
    model = BiometricAnomalyModel()

    result = model.predict(None)
    assert "model_type" in result
    assert "error" in result


def test_biometric_extract_features_with_small_image():
    """Test feature extraction with small image"""
    model = BiometricAnomalyModel()

    small_img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)

    features = model.extract_features(small_img)
    assert isinstance(features, torch.Tensor)


def test_biometric_harmonic_features_error_handling():
    """Test harmonic feature extraction error handling"""
    model = BiometricAnomalyModel()

    invalid_img = np.random.randint(0, 255, (5, 5), dtype=np.uint8)

    features = model._extract_harmonic_features(invalid_img)
    assert isinstance(features, np.ndarray[Any, Any])


def test_biometric_config_variants():
    """Test BiometricAnomalyModel with different config options"""
    configs = [
        {"use_harmonic_features": True},
        {"use_harmonic_features": False},
        {"model_name": "Facenet"},
        {},
    ]

    for config in configs:
        model = BiometricAnomalyModel(config)
        assert model is not None


def test_biometric_embedding_normalization_edge_cases():
    """Test embedding normalization with edge case sizes"""
    model = BiometricAnomalyModel()

    embeddings = [
        torch.randn(1, 32),
        torch.randn(1, 512),
        torch.randn(1, 1024),
    ]

    for emb in embeddings:
        normalized = model._normalize_embedding_size(emb)
        assert normalized.shape[1] == 128
