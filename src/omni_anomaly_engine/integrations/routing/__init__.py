"""
Request routing and fallback handling components.
"""

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
    "RequestRouter",
    "Route",
    "RouteMatch",
    "RouteNotFoundError",
    "FallbackHandler",
    "FallbackChain",
    "FallbackResult",
    "FallbackError",
]
