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
Tests for Knowledge Distillation modules.

Tests Dual-Student Knowledge Distillation for anomaly detection.
"""

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


pytestmark = [pytest.mark.slow, pytest.mark.visual]


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestDualStudentConfig:
    """Tests for DualStudentConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from omni_anomaly_engine.ml.distillation import DualStudentConfig

        config = DualStudentConfig()
        assert config.backbone == "resnet18"
        assert config.hidden_dim == 256
        assert config.temperature == 4.0
        assert config.alpha == 0.5

    def test_custom_config(self):
        """Test custom configuration."""
        from omni_anomaly_engine.ml.distillation import DualStudentConfig

        config = DualStudentConfig(
            backbone="resnet34",
            hidden_dim=128,
            learning_rate=1e-3,
            num_epochs=50,
        )
        assert config.backbone == "resnet34"
        assert config.hidden_dim == 128
        assert config.learning_rate == 1e-3


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestEncoderDecoderStudent:
    """Tests for Encoder-Decoder student network."""

    def test_initialization(self):
        """Test Encoder-Decoder student initialization."""
        from omni_anomaly_engine.ml.distillation.dual_student import EncoderDecoderStudent

        student = EncoderDecoderStudent(in_channels=256, hidden_dim=128)
        assert student is not None

    def test_forward_pass(self):
        """Test Encoder-Decoder forward pass."""
        from omni_anomaly_engine.ml.distillation.dual_student import EncoderDecoderStudent

        student = EncoderDecoderStudent(in_channels=256, hidden_dim=128)

        # Create input [B, C, H, W]
        x = torch.randn(2, 256, 14, 14)
        output = student(x)

        assert output.shape == x.shape


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestEncoderEncoderStudent:
    """Tests for Encoder-Encoder student network."""

    def test_initialization(self):
        """Test Encoder-Encoder student initialization."""
        from omni_anomaly_engine.ml.distillation.dual_student import EncoderEncoderStudent

        student = EncoderEncoderStudent(in_channels=256, hidden_dim=128)
        assert student is not None

    def test_forward_pass(self):
        """Test Encoder-Encoder forward pass."""
        from omni_anomaly_engine.ml.distillation.dual_student import EncoderEncoderStudent

        student = EncoderEncoderStudent(in_channels=256, hidden_dim=128)

        # Create input [B, C, H, W]
        x = torch.randn(2, 256, 14, 14)
        output = student(x)

        assert output.shape == x.shape

    def test_attention_mechanism(self):
        """Test attention mechanism in Encoder-Encoder student."""
        from omni_anomaly_engine.ml.distillation.dual_student import EncoderEncoderStudent

        student = EncoderEncoderStudent(in_channels=256, hidden_dim=128)

        # Verify attention layers exist
        assert hasattr(student, "attention")


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestDualStudentDistillation:
    """Tests for Dual-Student Distillation main class."""

    def test_initialization_default(self):
        """Test initialization with default config."""
        from omni_anomaly_engine.ml.distillation import DualStudentDistillation

        distiller = DualStudentDistillation()
        assert distiller is not None
        assert distiller.config.backbone == "resnet18"

    def test_initialization_with_dict_config(self):
        """Test initialization with dictionary config."""
        from omni_anomaly_engine.ml.distillation import DualStudentDistillation

        config = {
            "backbone": "resnet18",
            "hidden_dim": 128,
            "num_epochs": 5,
        }
        distiller = DualStudentDistillation(config=config)
        assert distiller.config.hidden_dim == 128

    def test_initialization_with_dataclass_config(self):
        """Test initialization with dataclass config."""
        from omni_anomaly_engine.ml.distillation import (
            DualStudentConfig,
            DualStudentDistillation,
        )

        config = DualStudentConfig(hidden_dim=512)
        distiller = DualStudentDistillation(config=config)
        assert distiller.config.hidden_dim == 512

    def test_not_fitted_error(self, sample_image):
        """Test error when calling detect before fit."""
        from omni_anomaly_engine.ml.distillation import DualStudentDistillation

        distiller = DualStudentDistillation()

        with pytest.raises(RuntimeError, match="Must call fit"):
            distiller.detect(sample_image)

    @pytest.mark.slow
    def test_fit_and_detect(self, sample_image_batch, sample_image):
        """Test fitting and detection workflow."""
        from omni_anomaly_engine.ml.distillation import (
            DualStudentConfig,
            DualStudentDistillation,
        )

        # Use minimal config for fast testing
        config = DualStudentConfig(
            num_epochs=2,
            batch_size=2,
        )
        distiller = DualStudentDistillation(config=config)

        # Fit on normal images
        distiller.fit(sample_image_batch)
        assert distiller._is_fitted

        # Detect anomalies
        result = distiller.detect(sample_image)

        assert "scores" in result
        assert "anomaly_maps" in result
        assert "is_anomaly" in result
        assert "student1_maps" in result
        assert "student2_maps" in result

    def test_distillation_loss(self):
        """Test distillation loss computation."""
        from omni_anomaly_engine.ml.distillation import DualStudentDistillation

        distiller = DualStudentDistillation()

        teacher_feat = torch.randn(2, 256, 14, 14)
        student_feat = torch.randn(2, 256, 14, 14)

        loss = distiller._distillation_loss(teacher_feat, student_feat)

        assert loss.ndim == 0  # Scalar
        assert loss >= 0

    def test_aggregate_features(self):
        """Test feature aggregation from multiple layers."""
        from omni_anomaly_engine.ml.distillation import DualStudentDistillation

        distiller = DualStudentDistillation()

        features = {
            "layer2": torch.randn(2, 128, 28, 28),
            "layer3": torch.randn(2, 256, 14, 14),
        }

        aggregated = distiller._aggregate_features(features)

        # Should concatenate and resize to smallest spatial size
        assert aggregated.shape[0] == 2
        assert aggregated.shape[2] == 14
        assert aggregated.shape[3] == 14


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestDistillationModule:
    """Tests for distillation module imports and structure."""

    def test_module_imports(self):
        """Test that all exports are available."""
        from omni_anomaly_engine.ml.distillation import (
            DualStudentConfig,
            DualStudentDistillation,
        )

        assert DualStudentDistillation is not None
        assert DualStudentConfig is not None

    def test_module_all_exports(self):
        """Test __all__ exports."""
        from omni_anomaly_engine.ml import distillation

        assert "DualStudentDistillation" in distillation.__all__
        assert "DualStudentConfig" in distillation.__all__
