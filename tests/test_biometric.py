"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisors LLC

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
Test biometric model functionality
"""

from unittest.mock import patch

import numpy as np

from omni_mercury_engine.models.biometric import BiometricAnomalyModel


def test_biometric_initialization():
    """Test biometric model initialization"""
    model = BiometricAnomalyModel()
    assert model is not None
    assert hasattr(model, "harmonic_decomposer")
    assert hasattr(model, "fourier_analyzer")


@patch("omni_mercury_engine.models.biometric.DeepFace")
def test_biometric_predict_valid_image(mock_deepface):
    """Test biometric prediction with valid image"""
    mock_deepface.analyze.return_value = [
        {
            "age": 25,
            "gender": {"Woman": 0.8, "Man": 0.2},
            "emotion": {
                "happy": 0.5,
                "sad": 0.1,
                "angry": 0.1,
                "neutral": 0.3,
            },
        }
    ]

    model = BiometricAnomalyModel()
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = model.predict(image)

    assert "model_type" in result
    assert result["model_type"] == "biometric"


@patch("omni_mercury_engine.models.biometric.DeepFace")
def test_biometric_extract_features(mock_deepface):
    """Test biometric feature extraction"""
    mock_deepface.represent.return_value = [{"embedding": np.random.randn(128).tolist()}]

    model = BiometricAnomalyModel()
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    features = model.extract_features(image)

    assert features is not None
    assert features.shape[-1] >= 128


@patch("omni_mercury_engine.models.biometric.DeepFace")
def test_biometric_predict_error_handling(mock_deepface):
    """Test biometric error handling"""
    mock_deepface.analyze.side_effect = Exception("Face not detected")

    model = BiometricAnomalyModel()
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    result = model.predict(image)

    assert "error" in result or "model_type" in result


def test_biometric_config():
    """Test biometric model with custom config"""
    config = {"use_harmonic": True, "detector_backend": "opencv"}
    model = BiometricAnomalyModel(config=config)
    assert model.config["use_harmonic"] is True
