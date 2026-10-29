# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Integration components for external services and routing.

This package provides:
- HTTP client with circuit breaker and retry logic
- External service doubles have MOVED OUT of the installed package to
  ``tools/service_doubles/`` (repo-only, not shipped in the wheel). They were
  143 KB of test scaffolding inside ``src/``: nothing in the engine imported
  them, the financial and weather doubles fabricated prices and readings with
  a seeded RNG, and the ``RedisCache`` they defined was a second class of that
  name shadowing the real one in ``core/feature_pipeline.py``.
- Request routing and fallback handling
- AMA Cryptography post-quantum cryptography adapter
"""

from __future__ import annotations

from omni_mercury_engine.integrations.http.client import HTTPClient, HTTPClientConfig, HTTPResponse
from omni_mercury_engine.integrations.mercury_amacrypto import (
    AMA_CRYPTOGRAPHY_AVAILABLE,
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
    "AMA_CRYPTOGRAPHY_AVAILABLE",
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
