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

"""Core modules for Mercury Agent ♱."""

from omni_mercury_engine.core.double_helix_engine import (
    DoubleHelixEvolutionEngine,
    HelixConfig,
    MercuryEquationEngine,
)
from omni_mercury_engine.core.global_omni_scalar_network import (
    EnhancementResult,
    EthicalGate,
    GlobalOmniScalarNetwork,
    MultiHeadAttentionFusion,
    ScalarGroup,
    ScalarRegistration,
    get_global_scalar_network,
    reset_global_network,
)

__all__ = [
    "DomainType",
    "DoubleHelixEvolutionEngine",
    "EnhancementResult",
    "EthicalConfig",
    "EthicalGate",
    "FusionWeightConfig",
    "GlobalOmniScalarNetwork",
    "HelixConfig",
    "MercuryEngineConfig",
    "MercuryEquationEngine",
    "MultiHeadAttentionFusion",
    "ScalarGroup",
    "ScalarRegistration",
    "ThreeRConfig",
    "get_default_config",
    "get_global_scalar_network",
    "reset_global_network",
]


# Lazy imports for engine configuration (requires pydantic)
def get_default_config():  # type: ignore[no-untyped-def]
    """Get the global default configuration. Lazy import to avoid pydantic at module load."""
    from omni_mercury_engine.core.engine_config import get_default_config as _get

    return _get()


class MercuryEngineConfig:
    """Lazy-loaded MercuryEngineConfig wrapper."""

    def __new__(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        from omni_mercury_engine.core.engine_config import MercuryEngineConfig as _Config

        return _Config(*args, **kwargs)


class EthicalConfig:
    """Lazy-loaded EthicalConfig wrapper."""

    def __new__(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        from omni_mercury_engine.core.engine_config import EthicalConfig as _Config

        return _Config(*args, **kwargs)


class FusionWeightConfig:
    """Lazy-loaded FusionWeightConfig wrapper."""

    def __new__(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        from omni_mercury_engine.core.engine_config import FusionWeightConfig as _Config

        return _Config(*args, **kwargs)


class ThreeRConfig:
    """Lazy-loaded ThreeRConfig wrapper."""

    def __new__(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        from omni_mercury_engine.core.engine_config import ThreeRConfig as _Config

        return _Config(*args, **kwargs)


class DomainType:
    """Lazy-loaded DomainType wrapper."""

    def __new__(cls, value):  # type: ignore[no-untyped-def]
        from omni_mercury_engine.core.engine_config import DomainType as _DomainType

        return _DomainType(value)
