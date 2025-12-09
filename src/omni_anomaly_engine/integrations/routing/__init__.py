"""
Request routing and fallback handling components.
"""
from __future__ import annotations

from omni_anomaly_engine.integrations.routing.fallback import (
    FallbackChain,
    FallbackError,
    FallbackHandler,
    FallbackResult,
)
from omni_anomaly_engine.integrations.routing.router import (
    RequestRouter,
    Route,
    RouteMatch,
    RouteNotFoundError,
)

__all__ = [
    "FallbackChain",
    "FallbackError",
    "FallbackHandler",
    "FallbackResult",
    "RequestRouter",
    "Route",
    "RouteMatch",
    "RouteNotFoundError",
]
