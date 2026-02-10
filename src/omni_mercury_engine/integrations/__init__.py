"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Integration components for external services and routing.

This package provides:
- HTTP client with circuit breaker and retry logic
- External service stubs for testing and development
- Request routing and fallback handling
- Ava-Guardian post-quantum cryptography adapter
"""

from __future__ import annotations

from omni_mercury_engine.integrations.http.client import HTTPClient, HTTPClientConfig, HTTPResponse
from omni_mercury_engine.integrations.mercury_guardian import (
    AVA_GUARDIAN_AVAILABLE,
    DILITHIUM_AVAILABLE,
    KYBER_AVAILABLE,
    CryptoAnomaly,
    CryptoAnomalyType,
    EWMATimingMonitor,
    MercuryGuardianAdapter,
    create_mercury_guardian_adapter,
)
from omni_mercury_engine.integrations.routing.fallback import (
    FallbackChain,
    FallbackHandler,
    FallbackResult,
)
from omni_mercury_engine.integrations.routing.router import RequestRouter, Route, RouteMatch

__all__ = [
    "AVA_GUARDIAN_AVAILABLE",
    "DILITHIUM_AVAILABLE",
    "KYBER_AVAILABLE",
    "CryptoAnomaly",
    "CryptoAnomalyType",
    "EWMATimingMonitor",
    "FallbackChain",
    "FallbackHandler",
    "FallbackResult",
    "HTTPClient",
    "HTTPClientConfig",
    "HTTPResponse",
    "MercuryGuardianAdapter",
    "RequestRouter",
    "Route",
    "RouteMatch",
    "create_mercury_guardian_adapter",
]
