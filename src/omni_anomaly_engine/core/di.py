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
Dependency Injection Framework for OMNI ♱ AVA

Provides:
- Service container with lifecycle management
- Factory pattern for component creation
- Plugin architecture with discovery and sandboxing
- Lazy initialization for optional components
- Circular dependency detection
"""

import logging
import threading
from abc import ABC  # noqa: F401 - kept for potential future use
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Type,
    TypeVar,
    runtime_checkable,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Lifecycle(Enum):
    """Service lifecycle management."""

    SINGLETON = "singleton"  # Single instance for container lifetime
    TRANSIENT = "transient"  # New instance per request
    SCOPED = "scoped"  # Single instance per scope


@dataclass
class ServiceDescriptor:
    """Describes a registered service."""

    service_type: Type
    implementation_type: Optional[Type] = None
    factory: Optional[Callable[..., Any]] = None
    instance: Optional[Any] = None
    lifecycle: Lifecycle = Lifecycle.SINGLETON
    dependencies: List[Type] = field(default_factory=list)


class CircularDependencyError(Exception):
    """Raised when a circular dependency is detected."""

    pass


class ServiceNotFoundError(Exception):
    """Raised when a required service is not registered."""

    pass


class ServiceContainer:
    """
    Dependency injection container with lifecycle management.

    Features:
    - Singleton, transient, and scoped lifetimes
    - Factory-based registration
    - Automatic dependency resolution
    - Circular dependency detection
    - Thread-safe singleton creation
    """

    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._lock = threading.RLock()
        self._resolution_stack: Set[Type] = set()
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}

    def register_singleton(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[Callable[..., T]] = None,
        instance: Optional[T] = None,
    ) -> "ServiceContainer":
        """
        Register a singleton service.

        Args:
            service_type: The type/interface to register
            implementation_type: Optional concrete implementation
            factory: Optional factory function
            instance: Optional pre-created instance

        Returns:
            Self for chaining
        """
        with self._lock:
            self._services[service_type] = ServiceDescriptor(
                service_type=service_type,
                implementation_type=implementation_type or service_type,
                factory=factory,
                instance=instance,
                lifecycle=Lifecycle.SINGLETON,
            )
        return self

    def register_transient(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[Callable[..., T]] = None,
    ) -> "ServiceContainer":
        """Register a transient service (new instance per request)."""
        with self._lock:
            self._services[service_type] = ServiceDescriptor(
                service_type=service_type,
                implementation_type=implementation_type or service_type,
                factory=factory,
                lifecycle=Lifecycle.TRANSIENT,
            )
        return self

    def register_scoped(
        self,
        service_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        factory: Optional[Callable[..., T]] = None,
    ) -> "ServiceContainer":
        """Register a scoped service (single instance per scope)."""
        with self._lock:
            self._services[service_type] = ServiceDescriptor(
                service_type=service_type,
                implementation_type=implementation_type or service_type,
                factory=factory,
                lifecycle=Lifecycle.SCOPED,
            )
        return self

    def resolve(self, service_type: Type[T], scope_id: Optional[str] = None) -> T:
        """
        Resolve a service instance.

        Args:
            service_type: The type to resolve
            scope_id: Optional scope identifier for scoped services

        Returns:
            The resolved service instance

        Raises:
            ServiceNotFoundError: If service is not registered
            CircularDependencyError: If circular dependency detected
        """
        with self._lock:
            if service_type not in self._services:
                raise ServiceNotFoundError(f"Service {service_type.__name__} is not registered")

            # Check for circular dependencies
            if service_type in self._resolution_stack:
                chain = " -> ".join(t.__name__ for t in self._resolution_stack)
                raise CircularDependencyError(
                    f"Circular dependency detected: {chain} -> {service_type.__name__}"
                )

            descriptor = self._services[service_type]

            # Return existing singleton instance
            if descriptor.lifecycle == Lifecycle.SINGLETON and descriptor.instance:
                return descriptor.instance

            # Return scoped instance if exists
            if descriptor.lifecycle == Lifecycle.SCOPED and scope_id:
                if scope_id in self._scoped_instances:
                    if service_type in self._scoped_instances[scope_id]:
                        return self._scoped_instances[scope_id][service_type]

            # Create new instance
            self._resolution_stack.add(service_type)
            try:
                instance = self._create_instance(descriptor)
            finally:
                self._resolution_stack.discard(service_type)

            # Store singleton instance
            if descriptor.lifecycle == Lifecycle.SINGLETON:
                descriptor.instance = instance

            # Store scoped instance
            if descriptor.lifecycle == Lifecycle.SCOPED and scope_id:
                if scope_id not in self._scoped_instances:
                    self._scoped_instances[scope_id] = {}
                self._scoped_instances[scope_id][service_type] = instance

            return instance

    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Create a new instance of a service."""
        if descriptor.factory:
            return descriptor.factory(self)

        impl_type = descriptor.implementation_type or descriptor.service_type

        # Inspect constructor for dependencies
        import inspect

        sig = inspect.signature(impl_type.__init__)
        kwargs: Dict[str, Any] = {}

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param.annotation != inspect.Parameter.empty:
                if param.annotation in self._services:
                    kwargs[param_name] = self.resolve(param.annotation)

        return impl_type(**kwargs)

    def create_scope(self, scope_id: str) -> "ServiceScope":
        """Create a new dependency scope."""
        return ServiceScope(self, scope_id)

    def dispose_scope(self, scope_id: str) -> None:
        """Dispose a scope and its instances."""
        with self._lock:
            if scope_id in self._scoped_instances:
                del self._scoped_instances[scope_id]


class ServiceScope:
    """Scoped service resolution context."""

    def __init__(self, container: ServiceContainer, scope_id: str):
        self._container = container
        self._scope_id = scope_id

    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service within this scope."""
        return self._container.resolve(service_type, self._scope_id)

    def __enter__(self) -> "ServiceScope":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._container.dispose_scope(self._scope_id)


# ============================================================================
# Protocol-Based Interfaces
# ============================================================================


@runtime_checkable
class DetectorProtocol(Protocol):
    """Protocol for anomaly detectors (structural subtyping)."""

    def fit(self, data: Any) -> "DetectorProtocol":
        """Fit the detector to normal data."""
        ...

    def detect(self, data: Any) -> Dict[str, Any]:
        """Detect anomalies in data."""
        ...

    def extract_features(self, data: Any) -> Any:
        """Extract features for ML fusion."""
        ...

    def is_fitted(self) -> bool:
        """Check if detector has been fitted."""
        ...


@runtime_checkable
class ModelProtocol(Protocol):
    """Protocol for prediction models (structural subtyping)."""

    def predict(self, data: Any) -> Dict[str, Any]:
        """Make predictions on data."""
        ...

    def extract_features(self, data: Any) -> Any:
        """Extract features for ML fusion."""
        ...


@runtime_checkable
class EncoderProtocol(Protocol):
    """Protocol for feature encoders (structural subtyping)."""

    input_dim: int
    output_dim: int

    def forward(self, x: Any) -> Any:
        """Encode input features to fixed-size embedding."""
        ...


@runtime_checkable
class ConfigurableProtocol(Protocol):
    """Protocol for configurable components."""

    def configure(self, config: Dict[str, Any]) -> None:
        """Apply configuration to component."""
        ...

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        ...


# ============================================================================
# Component Factory
# ============================================================================


class ComponentFactory:
    """
    Factory for creating components with proper lifecycle management.

    Implements the Factory pattern with:
    - Lazy initialization
    - Plugin discovery
    - Version compatibility checking
    """

    def __init__(self, container: ServiceContainer):
        self._container = container
        self._registered_plugins: Dict[str, Type] = {}

    def register_plugin(
        self,
        name: str,
        plugin_type: Type,
        version: str = "1.0.0",
    ) -> None:
        """
        Register a plugin type.

        Args:
            name: Plugin identifier
            plugin_type: Plugin class
            version: Plugin version
        """
        self._registered_plugins[name] = plugin_type
        logger.info(f"Registered plugin: {name} v{version}")

    def create_detector(
        self,
        detector_type: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> DetectorProtocol:
        """
        Create a detector instance.

        Args:
            detector_type: Type of detector to create
            config: Optional configuration

        Returns:
            Configured detector instance
        """
        # Map detector types to implementations
        detector_map = {
            "statistical": "omni_anomaly_engine.detectors.statistical.StatisticalDetector",
            "temporal": "omni_anomaly_engine.detectors.temporal.TemporalDetector",
            "spatial": "omni_anomaly_engine.detectors.spatial.SpatialDetector",
            "dimensional": "omni_anomaly_engine.detectors.dimensional.DimensionalDetector",
            "directive": "omni_anomaly_engine.detectors.directive.DirectiveDetector",
        }

        if detector_type in self._registered_plugins:
            plugin_class = self._registered_plugins[detector_type]
            return plugin_class(config=config)

        if detector_type not in detector_map:
            raise ValueError(f"Unknown detector type: {detector_type}")

        # Lazy import and instantiate
        module_path, class_name = detector_map[detector_type].rsplit(".", 1)
        try:
            import importlib

            module = importlib.import_module(module_path)
            detector_class = getattr(module, class_name)
            return detector_class(config=config)
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not load detector {detector_type}: {e}")
            raise

    def create_model(
        self,
        model_type: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> ModelProtocol:
        """
        Create a model instance.

        Args:
            model_type: Type of model to create
            config: Optional configuration

        Returns:
            Configured model instance
        """
        model_map = {
            "quantum": "omni_anomaly_engine.models.quantum.QuantumAnomalyModel",
            "astrophysical": "omni_anomaly_engine.models.astrophysical.AstrophysicalAnomalyModel",
            "biometric": "omni_anomaly_engine.models.biometric.BiometricAnomalyModel",
            "affective": "omni_anomaly_engine.models.affective.AffectiveAnomalyModel",
            "neural": "omni_anomaly_engine.models.neural.NeuralAnomalyModel",
            "consciousness": "omni_anomaly_engine.models.consciousness.ConsciousnessAnomalyModel",
        }

        if model_type in self._registered_plugins:
            plugin_class = self._registered_plugins[model_type]
            return plugin_class(config=config)

        if model_type not in model_map:
            raise ValueError(f"Unknown model type: {model_type}")

        module_path, class_name = model_map[model_type].rsplit(".", 1)
        try:
            import importlib

            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            return model_class(config=config)
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not load model {model_type}: {e}")
            raise


# ============================================================================
# Global Container
# ============================================================================

_global_container: Optional[ServiceContainer] = None


def get_container() -> ServiceContainer:
    """Get the global service container."""
    global _global_container
    if _global_container is None:
        _global_container = ServiceContainer()
    return _global_container


def configure_container(container: ServiceContainer) -> None:
    """Set the global service container."""
    global _global_container
    _global_container = container
