"""
Hyperparameter Search Space Definition.

Provides flexible definition of hyperparameter search spaces for AutoML.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


class HyperParameter(ABC):
    """Base class for hyperparameters."""

    def __init__(self, name: str) -> None:
        """Initialize hyperparameter."""
        self._name = name

    @property
    def name(self) -> str:
        """Parameter name."""
        return self._name

    @abstractmethod
    def sample(self, rng: np.random.Generator | None = None) -> Any:
        """Sample a value from the parameter space."""
        pass

    @abstractmethod
    def to_normalized(self, value: Any) -> float:
        """Convert value to normalized [0, 1] space."""
        pass

    @abstractmethod
    def from_normalized(self, normalized: float) -> Any:
        """Convert from normalized [0, 1] space to actual value."""
        pass

    @abstractmethod
    def get_bounds(self) -> tuple[Any, Any]:
        """Get parameter bounds."""
        pass


class UniformParameter(HyperParameter):
    """Uniform distribution parameter."""

    def __init__(
        self,
        name: str,
        low: float,
        high: float,
    ) -> None:
        """Initialize uniform parameter."""
        super().__init__(name)
        self._low = low
        self._high = high

    def sample(self, rng: np.random.Generator | None = None) -> float:
        """Sample uniformly."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(self._low, self._high)

    def to_normalized(self, value: float) -> float:
        """Convert to [0, 1]."""
        return (value - self._low) / (self._high - self._low)

    def from_normalized(self, normalized: float) -> float:
        """Convert from [0, 1]."""
        return self._low + normalized * (self._high - self._low)

    def get_bounds(self) -> tuple[float, float]:
        """Get bounds."""
        return (self._low, self._high)


class LogUniformParameter(HyperParameter):
    """Log-uniform distribution parameter."""

    def __init__(
        self,
        name: str,
        low: float,
        high: float,
    ) -> None:
        """Initialize log-uniform parameter."""
        super().__init__(name)
        self._low = low
        self._high = high
        self._log_low = np.log(low)
        self._log_high = np.log(high)

    def sample(self, rng: np.random.Generator | None = None) -> float:
        """Sample log-uniformly."""
        if rng is None:
            rng = np.random.default_rng()
        log_val = rng.uniform(self._log_low, self._log_high)
        return np.exp(log_val)

    def to_normalized(self, value: float) -> float:
        """Convert to [0, 1]."""
        log_val = np.log(value)
        return (log_val - self._log_low) / (self._log_high - self._log_low)

    def from_normalized(self, normalized: float) -> float:
        """Convert from [0, 1]."""
        log_val = self._log_low + normalized * (self._log_high - self._log_low)
        return np.exp(log_val)

    def get_bounds(self) -> tuple[float, float]:
        """Get bounds."""
        return (self._low, self._high)


class IntUniformParameter(HyperParameter):
    """Integer uniform distribution parameter."""

    def __init__(
        self,
        name: str,
        low: int,
        high: int,
    ) -> None:
        """Initialize integer parameter."""
        super().__init__(name)
        self._low = low
        self._high = high

    def sample(self, rng: np.random.Generator | None = None) -> int:
        """Sample integer uniformly."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.integers(self._low, self._high + 1)

    def to_normalized(self, value: int) -> float:
        """Convert to [0, 1]."""
        return (value - self._low) / (self._high - self._low)

    def from_normalized(self, normalized: float) -> int:
        """Convert from [0, 1]."""
        return round(self._low + normalized * (self._high - self._low))

    def get_bounds(self) -> tuple[int, int]:
        """Get bounds."""
        return (self._low, self._high)


class CategoricalParameter(HyperParameter):
    """Categorical parameter."""

    def __init__(
        self,
        name: str,
        choices: list[Any],
    ) -> None:
        """Initialize categorical parameter."""
        super().__init__(name)
        self._choices = choices

    def sample(self, rng: np.random.Generator | None = None) -> Any:
        """Sample from choices."""
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.integers(0, len(self._choices))
        return self._choices[idx]

    def to_normalized(self, value: Any) -> float:
        """Convert to [0, 1]."""
        idx = self._choices.index(value)
        return idx / (len(self._choices) - 1) if len(self._choices) > 1 else 0.0

    def from_normalized(self, normalized: float) -> Any:
        """Convert from [0, 1]."""
        idx = round(normalized * (len(self._choices) - 1))
        return self._choices[idx]

    def get_bounds(self) -> tuple[int, int]:
        """Get bounds as indices."""
        return (0, len(self._choices) - 1)

    @property
    def choices(self) -> list[Any]:
        """Get available choices."""
        return self._choices


class ConditionalParameter(HyperParameter):
    """Parameter conditional on another parameter's value."""

    def __init__(
        self,
        name: str,
        parent_name: str,
        parent_value: Any,
        parameter: HyperParameter,
    ) -> None:
        """Initialize conditional parameter."""
        super().__init__(name)
        self._parent_name = parent_name
        self._parent_value = parent_value
        self._parameter = parameter

    @property
    def parent_name(self) -> str:
        """Parent parameter name."""
        return self._parent_name

    @property
    def parent_value(self) -> Any:
        """Parent value that activates this parameter."""
        return self._parent_value

    def is_active(self, parent_actual_value: Any) -> bool:
        """Check if this parameter is active."""
        return parent_actual_value == self._parent_value

    def sample(self, rng: np.random.Generator | None = None) -> Any:
        """Sample from underlying parameter."""
        return self._parameter.sample(rng)

    def to_normalized(self, value: Any) -> float:
        """Convert to [0, 1]."""
        return self._parameter.to_normalized(value)

    def from_normalized(self, normalized: float) -> Any:
        """Convert from [0, 1]."""
        return self._parameter.from_normalized(normalized)

    def get_bounds(self) -> tuple[Any, Any]:
        """Get bounds."""
        return self._parameter.get_bounds()


@dataclass
class SearchSpace:
    """Complete hyperparameter search space."""

    parameters: dict[str, HyperParameter] = field(default_factory=dict)

    def add(self, parameter: HyperParameter) -> SearchSpace:
        """Add a parameter to the search space."""
        self.parameters[parameter.name] = parameter
        return self

    def sample(self, rng: np.random.Generator | None = None) -> dict[str, Any]:
        """Sample a configuration from the search space."""
        if rng is None:
            rng = np.random.default_rng()

        config = {}
        active_params = {}

        for name, param in self.parameters.items():
            if isinstance(param, ConditionalParameter):
                continue
            config[name] = param.sample(rng)
            active_params[name] = True

        for name, param in self.parameters.items():
            if not isinstance(param, ConditionalParameter):
                continue

            parent_value = config.get(param.parent_name)
            if param.is_active(parent_value):
                config[name] = param.sample(rng)
                active_params[name] = True

        return config

    def to_normalized(self, config: dict[str, Any]) -> np.ndarray:
        """Convert configuration to normalized array."""
        values = []
        for name, param in self.parameters.items():
            if name in config:
                values.append(param.to_normalized(config[name]))
            else:
                values.append(0.5)
        return np.array(values)

    def from_normalized(self, normalized: np.ndarray) -> dict[str, Any]:
        """Convert normalized array to configuration."""
        config = {}
        for i, (name, param) in enumerate(self.parameters.items()):
            if i < len(normalized):
                config[name] = param.from_normalized(normalized[i])
        return config

    @property
    def dimensions(self) -> int:
        """Number of dimensions in search space."""
        return len(self.parameters)

    @classmethod
    def from_dict(cls, spec: dict[str, tuple]) -> SearchSpace:
        """
        Create search space from dictionary specification.

        Format: {"param_name": ("type", arg1, arg2, ...)}
        Types: "uniform", "log_uniform", "int_uniform", "categorical"
        """
        space = cls()

        for name, spec_tuple in spec.items():
            param_type = spec_tuple[0]
            args = spec_tuple[1:]

            if param_type == "uniform":
                space.add(UniformParameter(name, args[0], args[1]))
            elif param_type == "log_uniform":
                space.add(LogUniformParameter(name, args[0], args[1]))
            elif param_type == "int_uniform":
                space.add(IntUniformParameter(name, args[0], args[1]))
            elif param_type == "categorical":
                space.add(CategoricalParameter(name, list(args[0])))

        return space
