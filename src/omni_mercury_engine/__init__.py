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
Mercury Agent ♱: ML-Centric Anomaly Detection Framework

A unified toolkit for anomaly detection across security, biometrics, temporal patterns,
and multi-dimensional data using neural network fusion of specialized detectors.
"""

from typing import TYPE_CHECKING


# Type-only imports for static analysis (CodeQL, mypy, etc.)
# These are not imported at runtime to support lazy loading
if TYPE_CHECKING:
    from omni_mercury_engine.core.config import EngineConfig as EngineConfig
    from omni_mercury_engine.core.exceptions import (
        DetectorException as DetectorException,
        FusionException as FusionException,
        ModelException as ModelException,
        OmniAnomalyException as OmniAnomalyException,
    )
    from omni_mercury_engine.engine import OmniMercuryEngine as OmniMercuryEngine

# Lazy imports to support running without ML dependencies (torch)
# The OmniMercuryEngine requires torch, but we defer the import to allow
# CLI help commands and other lightweight operations to work without it.


def __getattr__(name: str):
    """Lazy import for OmniMercuryEngine to defer torch dependency."""
    if name == "OmniMercuryEngine":
        from omni_mercury_engine.engine import OmniMercuryEngine

        return OmniMercuryEngine
    elif name == "EngineConfig":
        from omni_mercury_engine.core.config import EngineConfig

        return EngineConfig
    elif name in ("OmniAnomalyException", "DetectorException", "ModelException", "FusionException"):
        from omni_mercury_engine.core.exceptions import (
            DetectorException,
            FusionException,
            ModelException,
            OmniAnomalyException,
        )

        return {
            "OmniAnomalyException": OmniAnomalyException,
            "DetectorException": DetectorException,
            "ModelException": ModelException,
            "FusionException": FusionException,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "1.1.0"
__author__ = "Steel Security Advisors LLC"
__license__ = "GPL-3.0"

__all__ = [
    "DetectorException",
    "EngineConfig",
    "FusionException",
    "ModelException",
    "OmniAnomalyException",
    "OmniMercuryEngine",
]
