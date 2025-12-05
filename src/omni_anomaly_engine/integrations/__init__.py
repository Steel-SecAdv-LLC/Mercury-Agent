"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Integration components for external services and routing.

This package provides:
- HTTP client with circuit breaker and retry logic
- External service stubs for testing and development
- Request routing and fallback handling
"""

from omni_anomaly_engine.integrations.http.client import HTTPClient, HTTPClientConfig, HTTPResponse
from omni_anomaly_engine.integrations.routing.fallback import (
    FallbackChain,
    FallbackHandler,
    FallbackResult,
)
from omni_anomaly_engine.integrations.routing.router import RequestRouter, Route, RouteMatch

__all__ = [
    "FallbackChain",
    # Fallback
    "FallbackHandler",
    "FallbackResult",
    # HTTP Client
    "HTTPClient",
    "HTTPClientConfig",
    "HTTPResponse",
    # Routing
    "RequestRouter",
    "Route",
    "RouteMatch",
]
