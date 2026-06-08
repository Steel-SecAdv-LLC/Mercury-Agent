# Copyright (C) 2025 Steel Security Advisors LLC
"""Mercury Agent SOTA Model Registry.

Provides a unified interface for accessing state-of-the-art anomaly detection
models. This registry enables:
- Centralized model discovery and instantiation
- Consistent configuration management
- Easy integration with benchmarking and production pipelines

Example usage:
    from omni_mercury_engine.models.sota.registry import SOTARegistry

    # List available models
    models = SOTARegistry.list_models()

    # Get a configured model
    tranad = SOTARegistry.get("tranad", input_dim=30)

    # Get model info
    info = SOTARegistry.get_model_info("tranad")
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a registered SOTA model.

    Attributes:
        name: Model identifier
        model_class: The model class
        config_class: The configuration dataclass
        description: Human-readable description
        paper_reference: Academic paper reference
        default_config: Default configuration values
        supported_tasks: List of supported tasks (e.g., "anomaly_detection", "reconstruction")
    """

    name: str
    model_class: type[nn.Module]
    config_class: type
    description: str
    paper_reference: str
    default_config: dict[str, Any] = field(default_factory=dict)
    supported_tasks: list[str] = field(default_factory=lambda: ["anomaly_detection"])


class SOTARegistry:
    """Registry for state-of-the-art anomaly detection models.

    This singleton registry provides centralized access to SOTA models
    with consistent configuration and instantiation patterns.

    Example:
        # Get a model with default config
        model = SOTARegistry.get("tranad")

        # Get a model with custom config
        model = SOTARegistry.get("tranad", input_dim=64, d_model=512)

        # List all available models
        models = SOTARegistry.list_models()
    """

    _models: dict[str, ModelInfo] = {}
    _initialized: bool = False

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Ensure the registry is initialized with default models."""
        if not cls._initialized:
            cls._register_default_models()
            cls._initialized = True

    @classmethod
    def _register_default_models(cls) -> None:
        """Register the default SOTA models."""
        try:
            from omni_mercury_engine.models.sota.tranad import TranADConfig, TranADModel

            cls.register(
                name="tranad",
                model_class=TranADModel,
                config_class=TranADConfig,
                description=(
                    "TranAD: Deep Transformer Networks for Anomaly Detection. "
                    "Uses dual-decoder architecture with focus score conditioning "
                    "and adversarial training for robust reconstruction-based detection."
                ),
                paper_reference=(
                    "Tuli et al., 'TranAD: Deep Transformer Networks for Anomaly "
                    "Detection in Multivariate Time Series Data', VLDB 2022. "
                    "https://arxiv.org/abs/2201.07284"
                ),
                default_config=asdict(TranADConfig()),
            )
            logger.info("Registered TranAD model")
        except ImportError as e:
            logger.warning(f"Failed to register TranAD: {e}")

        try:
            from omni_mercury_engine.models.sota.maat import MAATConfig, MAATModel

            cls.register(
                name="maat",
                model_class=MAATModel,
                config_class=MAATConfig,
                description=(
                    "MAAT: Mamba Adaptive Anomaly Transformer. "
                    "Combines sparse attention with Mamba-SSM for efficient "
                    "long-range dependency capture in noisy environments."
                ),
                paper_reference=(
                    "Benaissa et al., 'MAAT: Mamba Adaptive Anomaly Transformer', "
                    "arXiv 2025. https://arxiv.org/abs/2502.07858"
                ),
                default_config=asdict(MAATConfig()),
            )
            logger.info("Registered MAAT model")
        except ImportError as e:
            logger.warning(f"Failed to register MAAT: {e}")

    @classmethod
    def register(
        cls,
        name: str,
        model_class: type[nn.Module],
        config_class: type,
        description: str = "",
        paper_reference: str = "",
        default_config: dict[str, Any] | None = None,
        supported_tasks: list[str] | None = None,
    ) -> None:
        """Register a SOTA model in the registry.

        Args:
            name: Unique identifier for the model
            model_class: The model class (must be a nn.Module subclass)
            config_class: The configuration dataclass
            description: Human-readable description
            paper_reference: Academic paper reference
            default_config: Default configuration values
            supported_tasks: List of supported tasks
        """
        if name in cls._models:
            logger.warning(f"Overwriting existing model registration: {name}")

        cls._models[name] = ModelInfo(
            name=name,
            model_class=model_class,
            config_class=config_class,
            description=description,
            paper_reference=paper_reference,
            default_config=default_config or {},
            supported_tasks=supported_tasks or ["anomaly_detection"],
        )
        logger.debug(f"Registered model: {name}")

    @classmethod
    def get(cls, name: str, **config_overrides: Any) -> nn.Module:
        """Get a configured model instance.

        Args:
            name: Model identifier
            **config_overrides: Configuration overrides

        Returns:
            Configured model instance

        Raises:
            KeyError: If model is not registered
        """
        cls._ensure_initialized()

        if name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise KeyError(f"Model '{name}' not found. Available models: {available}")

        info = cls._models[name]

        # Merge default config with overrides
        config_dict = {**info.default_config, **config_overrides}

        # Create config instance
        config = info.config_class(**config_dict)

        # Create and return model
        return info.model_class(config)

    @classmethod
    def get_model_info(cls, name: str) -> ModelInfo:
        """Get information about a registered model.

        Args:
            name: Model identifier

        Returns:
            ModelInfo dataclass with model details

        Raises:
            KeyError: If model is not registered
        """
        cls._ensure_initialized()

        if name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise KeyError(f"Model '{name}' not found. Available models: {available}")

        return cls._models[name]

    @classmethod
    def list_models(cls) -> list[str]:
        """List all registered model names.

        Returns:
            List of model identifiers
        """
        cls._ensure_initialized()
        return list(cls._models.keys())

    @classmethod
    def list_models_detailed(cls) -> list[ModelInfo]:
        """List all registered models with full details.

        Returns:
            List of ModelInfo dataclasses
        """
        cls._ensure_initialized()
        return list(cls._models.values())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a model is registered.

        Args:
            name: Model identifier

        Returns:
            True if model is registered
        """
        cls._ensure_initialized()
        return name in cls._models

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister a model from the registry.

        Args:
            name: Model identifier

        Returns:
            True if model was unregistered, False if not found
        """
        if name in cls._models:
            del cls._models[name]
            logger.debug(f"Unregistered model: {name}")
            return True
        return False

    @classmethod
    def clear(cls) -> None:
        """Clear all registered models.

        Warning: This will remove all models including defaults.
        Call _register_default_models() to restore defaults.
        """
        cls._models.clear()
        cls._initialized = False
        logger.debug("Cleared all registered models")


def get_model(name: str, **config_overrides: Any) -> nn.Module:
    """Convenience function to get a model from the registry.

    Args:
        name: Model identifier
        **config_overrides: Configuration overrides

    Returns:
        Configured model instance

    Example:
        model = get_model("tranad", input_dim=30)
    """
    return SOTARegistry.get(name, **config_overrides)


def list_models() -> list[str]:
    """Convenience function to list available models.

    Returns:
        List of model identifiers
    """
    return SOTARegistry.list_models()
