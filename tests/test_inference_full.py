"""
Mercury Agent
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
Comprehensive tests for inference module to boost coverage
"""

import numpy as np
import pytest

# Conditional torch import
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]

# Skip all tests in this module if torch is not available
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")

# Conditional imports - only when torch is available
if HAS_TORCH:
    from omni_mercury_engine.ml.inference import FusionInference


def test_fusion_inference_initialization():
    """Test FusionInference initialization"""
    engine = FusionInference()
    assert engine is not None
    assert engine.model is not None


def test_fusion_inference_predict():
    """Test FusionInference predict method"""
    engine = FusionInference()

    features = {
        "statistical": torch.randn(5, 10),
        "temporal": torch.randn(5, 32),
    }

    result = engine.predict(features)

    assert "anomaly_probs" in result
    assert "class_predictions" in result
    assert "severity_scores" in result


def test_fusion_inference_with_device():
    """Test FusionInference with specific device"""
    engine = FusionInference(device="cpu")
    assert engine.device.type == "cpu"


def test_fusion_inference_predict_batch():
    """Test FusionInference predict_batch method"""
    engine = FusionInference()

    data = [{"statistical": torch.randn(10), "temporal": torch.randn(32)} for _ in range(5)]

    results = engine.predict_batch(data, batch_size=2)

    assert len(results) == len(data)
    assert "anomaly_prob" in results[0]
    assert "class_prediction" in results[0]


def test_fusion_inference_explain():
    """Test FusionInference explain method"""
    engine = FusionInference()

    features = {
        "statistical": torch.randn(1, 10),
        "temporal": torch.randn(1, 32),
    }

    explanation = engine.explain(features)

    assert "prediction" in explanation
    assert "is_anomaly" in explanation["prediction"]
    assert "confidence" in explanation["prediction"]


def test_fusion_inference_with_numpy():
    """Test FusionInference with numpy arrays"""
    engine = FusionInference()

    features = {
        "statistical": np.random.randn(3, 10),
        "temporal": np.random.randn(3, 32),
    }

    result = engine.predict(features)
    assert result is not None


def test_fusion_inference_return_attention():
    """Test FusionInference with attention weights"""
    engine = FusionInference()

    features = {
        "statistical": torch.randn(2, 10),
        "temporal": torch.randn(2, 32),
    }

    result = engine.predict(features, return_attention=True)
    assert "anomaly_probs" in result


def test_fusion_inference_multiple_features():
    """Test FusionInference with multiple feature types"""
    engine = FusionInference()

    features = {
        "statistical": torch.randn(4, 10),
        "temporal": torch.randn(4, 32),
        "biometric": torch.randn(4, 128),
        "quantum": torch.randn(4, 16, 2),
    }

    result = engine.predict(features)

    assert "anomaly_probs" in result
    assert result["anomaly_probs"].shape[0] == 4
