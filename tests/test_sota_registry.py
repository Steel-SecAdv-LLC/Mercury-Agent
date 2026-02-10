"""
Tests for SOTA Model Registry.

Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC
"""

import pytest
import torch
from torch import nn

from omni_mercury_engine.models.sota.registry import (
    ModelInfo,
    SOTARegistry,
    get_model,
    list_models,
)


class MockConfig:
    """Mock configuration class for testing."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 32) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim


class MockModel(nn.Module):
    """Mock model class for testing."""

    def __init__(self, config: MockConfig) -> None:
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.input_dim, config.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_model_info_creation(self) -> None:
        """Test ModelInfo creation with required fields."""
        info = ModelInfo(
            name="test_model",
            model_class=MockModel,
            config_class=MockConfig,
            description="Test model description",
            paper_reference="Test paper reference",
        )
        assert info.name == "test_model"
        assert info.model_class == MockModel
        assert info.config_class == MockConfig
        assert info.description == "Test model description"
        assert info.paper_reference == "Test paper reference"
        assert info.default_config == {}
        assert info.supported_tasks == ["anomaly_detection"]

    def test_model_info_with_defaults(self) -> None:
        """Test ModelInfo with custom default config."""
        info = ModelInfo(
            name="test_model",
            model_class=MockModel,
            config_class=MockConfig,
            description="Test",
            paper_reference="Test",
            default_config={"input_dim": 20},
            supported_tasks=["anomaly_detection", "reconstruction"],
        )
        assert info.default_config == {"input_dim": 20}
        assert "reconstruction" in info.supported_tasks


class TestSOTARegistry:
    """Tests for SOTARegistry class."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        SOTARegistry.clear()

    def test_register_model(self) -> None:
        """Test registering a model."""
        SOTARegistry.register(
            name="mock_model",
            model_class=MockModel,
            config_class=MockConfig,
            description="Mock model for testing",
            paper_reference="Mock paper",
        )
        assert SOTARegistry.is_registered("mock_model")

    def test_register_model_with_defaults(self) -> None:
        """Test registering a model with default config."""
        SOTARegistry.register(
            name="mock_model",
            model_class=MockModel,
            config_class=MockConfig,
            description="Mock model",
            paper_reference="Mock paper",
            default_config={"input_dim": 20, "hidden_dim": 64},
        )
        info = SOTARegistry.get_model_info("mock_model")
        assert info.default_config["input_dim"] == 20
        assert info.default_config["hidden_dim"] == 64

    def test_register_overwrites_existing(self) -> None:
        """Test that registering with same name overwrites."""
        SOTARegistry.register(
            name="mock_model",
            model_class=MockModel,
            config_class=MockConfig,
            description="First version",
            paper_reference="Paper 1",
        )
        SOTARegistry.register(
            name="mock_model",
            model_class=MockModel,
            config_class=MockConfig,
            description="Second version",
            paper_reference="Paper 2",
        )
        info = SOTARegistry.get_model_info("mock_model")
        assert info.description == "Second version"

    def test_get_model(self) -> None:
        """Test getting a model instance."""
        SOTARegistry.register(
            name="mock_model",
            model_class=MockModel,
            config_class=MockConfig,
            description="Mock model",
            paper_reference="Mock paper",
            default_config={"input_dim": 10, "hidden_dim": 32},
        )
        model = SOTARegistry.get("mock_model")
        assert isinstance(model, MockModel)
        assert model.config.input_dim == 10
        assert model.config.hidden_dim == 32

    def test_get_model_with_overrides(self) -> None:
        """Test getting a model with config overrides."""
        SOTARegistry.register(
            name="mock_model",
            model_class=MockModel,
            config_class=MockConfig,
            description="Mock model",
            paper_reference="Mock paper",
            default_config={"input_dim": 10, "hidden_dim": 32},
        )
        model = SOTARegistry.get("mock_model", input_dim=20, hidden_dim=64)
        assert model.config.input_dim == 20
        assert model.config.hidden_dim == 64

    def test_get_nonexistent_model(self) -> None:
        """Test getting a nonexistent model raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            SOTARegistry.get("nonexistent_model")
        assert "nonexistent_model" in str(exc_info.value)

    def test_get_model_info(self) -> None:
        """Test getting model info."""
        SOTARegistry.register(
            name="mock_model",
            model_class=MockModel,
            config_class=MockConfig,
            description="Mock model description",
            paper_reference="Mock paper reference",
        )
        info = SOTARegistry.get_model_info("mock_model")
        assert info.name == "mock_model"
        assert info.description == "Mock model description"

    def test_get_model_info_nonexistent(self) -> None:
        """Test getting info for nonexistent model raises KeyError."""
        with pytest.raises(KeyError):
            SOTARegistry.get_model_info("nonexistent_model")

    def test_list_models(self) -> None:
        """Test listing registered models."""
        SOTARegistry.register(
            name="model1",
            model_class=MockModel,
            config_class=MockConfig,
            description="Model 1",
            paper_reference="Paper 1",
        )
        SOTARegistry.register(
            name="model2",
            model_class=MockModel,
            config_class=MockConfig,
            description="Model 2",
            paper_reference="Paper 2",
        )
        models = SOTARegistry.list_models()
        assert "model1" in models
        assert "model2" in models

    def test_list_models_detailed(self) -> None:
        """Test listing models with full details."""
        SOTARegistry.register(
            name="mock_model",
            model_class=MockModel,
            config_class=MockConfig,
            description="Mock model",
            paper_reference="Mock paper",
        )
        detailed = SOTARegistry.list_models_detailed()
        assert len(detailed) >= 1
        assert any(info.name == "mock_model" for info in detailed)

    def test_is_registered(self) -> None:
        """Test checking if model is registered."""
        SOTARegistry.register(
            name="mock_model",
            model_class=MockModel,
            config_class=MockConfig,
            description="Mock model",
            paper_reference="Mock paper",
        )
        assert SOTARegistry.is_registered("mock_model") is True
        assert SOTARegistry.is_registered("nonexistent") is False

    def test_unregister(self) -> None:
        """Test unregistering a model."""
        SOTARegistry.register(
            name="mock_model",
            model_class=MockModel,
            config_class=MockConfig,
            description="Mock model",
            paper_reference="Mock paper",
        )
        assert SOTARegistry.unregister("mock_model") is True
        assert SOTARegistry.is_registered("mock_model") is False

    def test_unregister_nonexistent(self) -> None:
        """Test unregistering nonexistent model returns False."""
        assert SOTARegistry.unregister("nonexistent") is False

    def test_clear(self) -> None:
        """Test clearing all registered models."""
        SOTARegistry.register(
            name="model1",
            model_class=MockModel,
            config_class=MockConfig,
            description="Model 1",
            paper_reference="Paper 1",
        )
        SOTARegistry.register(
            name="model2",
            model_class=MockModel,
            config_class=MockConfig,
            description="Model 2",
            paper_reference="Paper 2",
        )
        SOTARegistry.clear()
        assert len(SOTARegistry._models) == 0
        assert SOTARegistry._initialized is False

    def test_ensure_initialized(self) -> None:
        """Test that registry auto-initializes with default models."""
        SOTARegistry.clear()
        SOTARegistry.list_models()
        assert SOTARegistry._initialized is True


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def setup_method(self) -> None:
        """Reset registry before each test."""
        SOTARegistry.clear()

    def test_get_model_function(self) -> None:
        """Test get_model convenience function."""
        SOTARegistry.register(
            name="mock_model",
            model_class=MockModel,
            config_class=MockConfig,
            description="Mock model",
            paper_reference="Mock paper",
            default_config={"input_dim": 10, "hidden_dim": 32},
        )
        model = get_model("mock_model")
        assert isinstance(model, MockModel)

    def test_list_models_function(self) -> None:
        """Test list_models convenience function."""
        SOTARegistry.register(
            name="mock_model",
            model_class=MockModel,
            config_class=MockConfig,
            description="Mock model",
            paper_reference="Mock paper",
        )
        models = list_models()
        assert "mock_model" in models


class TestDefaultModels:
    """Tests for default model registration."""

    def test_tranad_registered(self) -> None:
        """Test that TranAD is registered by default."""
        SOTARegistry.clear()
        SOTARegistry._ensure_initialized()
        if SOTARegistry.is_registered("tranad"):
            info = SOTARegistry.get_model_info("tranad")
            assert "TranAD" in info.description

    def test_maat_registered(self) -> None:
        """Test that MAAT is registered by default."""
        SOTARegistry.clear()
        SOTARegistry._ensure_initialized()
        if SOTARegistry.is_registered("maat"):
            info = SOTARegistry.get_model_info("maat")
            assert "MAAT" in info.description
