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
Tests for configuration classes
"""

from omni_mercury_engine.core.config import (
    DetectorConfig,
    DeviceType,
    EngineConfig,
    FusionConfig,
    FusionMode,
    ModelConfig,
)


class TestDeviceType:
    """Test DeviceType enum."""

    def test_device_types(self):
        """Test all device types."""
        assert DeviceType.CPU.value == "cpu"
        assert DeviceType.CUDA.value == "cuda"
        assert DeviceType.MPS.value == "mps"


class TestFusionMode:
    """Test FusionMode enum."""

    def test_fusion_modes(self):
        """Test all fusion modes."""
        assert FusionMode.EARLY.value == "early"
        assert FusionMode.LATE.value == "late"
        assert FusionMode.HYBRID.value == "hybrid"


class TestDetectorConfig:
    """Test DetectorConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        config = DetectorConfig()

        assert config.enabled is True
        assert config.threshold == 0.5
        assert config.use_quantum_enhanced is True
        assert config.use_nano_detection is True
        assert config.use_harmonic_detection is True
        assert config.params == {}

    def test_custom_values(self):
        """Test custom values."""
        config = DetectorConfig(
            enabled=False,
            threshold=0.8,
            use_quantum_enhanced=False,
            params={"key": "value"},
        )

        assert config.enabled is False
        assert config.threshold == 0.8
        assert config.use_quantum_enhanced is False
        assert config.params == {"key": "value"}


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        config = ModelConfig()

        assert config.enabled is True
        assert config.use_harmonic_features is True
        assert config.use_black_hole_features is True
        assert config.params == {}

    def test_custom_values(self):
        """Test custom values."""
        config = ModelConfig(enabled=False, use_harmonic_features=False, params={"model": "custom"})

        assert config.enabled is False
        assert config.use_harmonic_features is False
        assert config.params == {"model": "custom"}


class TestFusionConfig:
    """Test FusionConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        config = FusionConfig()

        assert config.mode == FusionMode.HYBRID
        assert config.attention_heads == 4
        assert config.hidden_dim == 128
        assert config.dropout == 0.1
        assert config.learning_rate == 0.001
        assert config.weight_decay == 0.0001
        assert config.optimizer == "adamw"

    def test_custom_values(self):
        """Test custom values."""
        config = FusionConfig(mode=FusionMode.EARLY, attention_heads=8, hidden_dim=256, dropout=0.2)

        assert config.mode == FusionMode.EARLY
        assert config.attention_heads == 8
        assert config.hidden_dim == 256
        assert config.dropout == 0.2


class TestEngineConfig:
    """Test EngineConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        config = EngineConfig()

        assert config.device == DeviceType.CPU
        assert config.fusion_mode == FusionMode.HYBRID
        assert config.batch_size == 32
        assert config.num_workers == 4
        assert config.model_path is None
        assert config.cache_dir == "./cache"
        assert config.log_level == "INFO"

    def test_post_init_detectors(self):
        """Test __post_init__ creates default detectors."""
        config = EngineConfig()

        assert len(config.detectors) == 5
        assert "statistical" in config.detectors
        assert "temporal" in config.detectors
        assert "spatial" in config.detectors
        assert "dimensional" in config.detectors
        assert "directive" in config.detectors

    def test_post_init_models(self):
        """Test __post_init__ creates default models."""
        config = EngineConfig()

        assert len(config.models) == 6
        assert "quantum" in config.models
        assert "astrophysical" in config.models
        assert "biometric" in config.models
        assert "affective" in config.models
        assert "neural" in config.models
        assert "consciousness" in config.models

    def test_custom_detectors(self):
        """Test custom detectors don't get overwritten."""
        custom_detectors = {
            "custom": DetectorConfig(threshold=0.9),
        }
        config = EngineConfig(detectors=custom_detectors)

        assert len(config.detectors) == 1
        assert "custom" in config.detectors
        assert config.detectors["custom"].threshold == 0.9

    def test_custom_models(self):
        """Test custom models don't get overwritten."""
        custom_models = {
            "custom_model": ModelConfig(enabled=False),
        }
        config = EngineConfig(models=custom_models)

        assert len(config.models) == 1
        assert "custom_model" in config.models
        assert config.models["custom_model"].enabled is False

    def test_full_custom_config(self):
        """Test fully customized config."""
        config = EngineConfig(
            device=DeviceType.CUDA,
            fusion_mode=FusionMode.LATE,
            batch_size=64,
            num_workers=8,
            model_path="/path/to/model",
            cache_dir="/custom/cache",
            log_level="DEBUG",
        )

        assert config.device == DeviceType.CUDA
        assert config.fusion_mode == FusionMode.LATE
        assert config.batch_size == 64
        assert config.num_workers == 8
        assert config.model_path == "/path/to/model"
        assert config.cache_dir == "/custom/cache"
        assert config.log_level == "DEBUG"
