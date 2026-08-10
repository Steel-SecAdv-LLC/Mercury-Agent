# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
r"""FastAPI server for real-time anomaly detection.

This module provides a REST API for multi-domain anomaly detection using
the Mercury Agent framework. It implements best practices from Azure AI Anomaly
Detector and provides comprehensive OpenAPI documentation.

API Reference:
    - Azure AI Anomaly Detector:
      https://azure.microsoft.com/en-us/products/ai-services/ai-anomaly-detector

Example:
    Start the server with uvicorn::

        uvicorn omni_mercury_engine.api.server:app --host 0.0.0.0 --port 8000

    Make a detection request::

        curl -X POST "http://localhost:8000/api/v1/detect/univariate" \\
            -H "Content-Type: application/json" \\
            -d '{"data": [1.0, 2.0, 1.5, 10.0, 1.8], "sensitivity": 0.5}'
"""

from __future__ import annotations

import contextvars
import logging
import os
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from enum import StrEnum
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

# Type alias for ASGI middleware call_next parameter
RequestResponseEndpoint = Callable[[Request], Awaitable[Response]]

# Context variable for request correlation ID - accessible throughout request lifecycle
correlation_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)

from omni_mercury_engine._version import __version__ as _DISTRIBUTION_VERSION
from omni_mercury_engine.validation.api_validators import (
    APIRequestValidator,
    ValidationConfig,
)

# Configure PII-masking logger
logger = logging.getLogger(__name__)

# Initialize API request validator with production configuration
_validation_config = ValidationConfig(
    max_data_points=int(os.getenv("OMNI_MAX_DATA_POINTS", "100000")),
    max_features=int(os.getenv("OMNI_MAX_FEATURES", "1000")),
    max_string_length=int(os.getenv("OMNI_MAX_STRING_LENGTH", "256")),
    max_nan_ratio=float(os.getenv("OMNI_MAX_NAN_RATIO", "0.1")),
    max_inf_ratio=float(os.getenv("OMNI_MAX_INF_RATIO", "0.01")),
    strict_mode=os.getenv("OMNI_STRICT_VALIDATION", "false").lower() == "true",
)
_api_validator = APIRequestValidator(_validation_config)


class PIIMaskingFilter(logging.Filter):
    """Filter to mask PII data in log messages for security compliance."""

    # Patterns for common PII data types
    PII_PATTERNS = [
        (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL_REDACTED]"),
        (re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"), "[PHONE_REDACTED]"),
        (re.compile(r"\b\d{3}[-]?\d{2}[-]?\d{4}\b"), "[SSN_REDACTED]"),
        (re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"), "[CARD_REDACTED]"),
        (
            re.compile(
                r'(?i)(api[_-]?key|apikey|secret|password|token|auth)["\']?\s*[:=]\s*["\']?[\w\-]+'
            ),
            r"\1=[REDACTED]",
        ),
        (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP_REDACTED]"),
        (re.compile(r"Bearer\s+[\w\-\.]+"), "Bearer [TOKEN_REDACTED]"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        """Mask PII in log record message."""
        if hasattr(record, "msg") and isinstance(record.msg, str):
            msg = record.msg
            for pattern, replacement in self.PII_PATTERNS:
                msg = pattern.sub(replacement, msg)
            record.msg = msg
        return True


# Apply PII masking filter to all security-related loggers
for logger_name in ["omni_mercury_engine.api", "omni_mercury_engine.security", "uvicorn.access"]:
    _logger = logging.getLogger(logger_name)
    _logger.addFilter(PIIMaskingFilter())

# API version information — tracks the installed distribution version via the
# single source of truth in omni_mercury_engine._version (no manual drift).
API_VERSION = _DISTRIBUTION_VERSION
API_TITLE = "Mercury Agent API"
API_DESCRIPTION = """
## Overview

The Mercury Agent API provides multi-domain anomaly detection capabilities through a REST interface.
This API is designed for real-time anomaly detection in time-series data.

## Features

- **Univariate Detection**: Analyze single-variable time series for anomalies
- **Multivariate Detection**: Analyze multi-dimensional time series with correlation awareness
- **Configurable Sensitivity**: Adjust detection threshold via sensitivity parameter
- **Z-Score Based Detection**: Uses statistical z-score method with configurable thresholds

## Authentication

Currently, no authentication is required. Future versions will implement API key authentication.

## Rate Limiting

- **Default**: 100 requests per minute per IP
- **Burst**: Up to 20 requests per second

## Versioning

This API follows semantic versioning. The current version is v1.
Breaking changes will be introduced in new major versions (v2, v3, etc.).

## Support

For issues and feature requests, please visit:
https://github.com/Steel-SecAdv-LLC/Mercury-Agent/issues
"""

# OpenAPI tags for endpoint grouping
tags_metadata = [
    {
        "name": "Health",
        "description": "Health check endpoints for monitoring service availability and status.",
    },
    {
        "name": "Detection",
        "description": "Anomaly detection endpoints for univariate "
        "and multivariate time series analysis.",
        "externalDocs": {
            "description": "Anomaly Detection Best Practices",
            "url": "https://learn.microsoft.com/en-us/azure/ai-services/anomaly-detector/",
        },
    },
]

# =============================================================================
# Lifespan warmup
# =============================================================================
# Production-mode warmup hook.  Mercury's detection endpoints sit on top
# of numpy SIMD, Pydantic v2 model compilation, and the
# ``APIRequestValidator`` field-level validation graph.  All three pay
# a measurable cold-start cost on the *first* request — Pydantic JIT-
# compiles the model on first validation, numpy lazily resolves BLAS
# dispatch on first ``np.mean``/``np.std``, and the validator's
# Hypothesis-trained heuristics load on first call.  Without warmup, a
# real client sees that cold-start tail on its first POST (regularly
# 800 - 1800 ms on GHA's ubuntu-latest), and k8s liveness probes
# observe an "API ready" /health response while the next request still
# takes a second to return.
#
# The lifespan runs once per worker process, AFTER uvicorn binds the
# socket and BEFORE the first external request is served.  By driving
# 3 in-process detection calls against synthetic data here, we
# guarantee /health returns 200 only after Pydantic + numpy + the
# validator are all hot.  Synthesised inputs are tiny (16 samples
# univariate, 8x3 multivariate) so the warmup itself is < 100 ms in
# every environment we've measured; the heavy work is the one-shot
# JIT compile that the first real request would have eaten anyway.
#
# This is intentionally NOT gated behind an env var or feature flag.
# A warmed API is the correct posture for every deployment, and the
# cost is negligible.  Tests that need a cold start can stub the
# function via ``omni_mercury_engine.api.server._warmup`` monkey-
# patching.


async def _warmup(app_instance: FastAPI) -> None:
    """Drive synthetic detection calls so the first real request is warm.

    Called from :func:`lifespan` exactly once per worker process with
    deterministic, validator-clean inputs (a tiny univariate series,
    an 8x3 multivariate matrix, a single /health hit).  Any failure
    here is propagated -- a hard fault during warmup is a real bug
    (Pydantic model regression, validator graph drift, numpy ABI
    mismatch) and must crash the worker so the orchestrator marks
    the deployment unhealthy.  Silent degradation would let a broken
    detection path serve traffic; the contract this module signs
    with the rest of the system is "if uvicorn is up, detection
    works."  Tests that want to simulate a warmup failure should
    monkey-patch one of ``detect_univariate`` / ``detect_multivariate``
    / ``health_check`` and assert that ``await _warmup(...)`` raises.
    """
    # Pydantic model + validator warmup via the univariate path.
    univariate_data = [float(i % 7) for i in range(16)]
    univariate_req = UnivariateRequest(data=univariate_data, sensitivity=0.5)
    await detect_univariate(univariate_req)

    # Multivariate path warms a different numpy code path (L2 norm,
    # axis-wise mean/std) plus a different validator branch.
    multivariate_data = [[float(i + j) for j in range(3)] for i in range(8)]
    multivariate_req = MultivariateRequest(data=multivariate_data, sensitivity=0.5)
    await detect_multivariate(multivariate_req)

    # Health path warms the middleware chain (correlation-ID,
    # rate-limit bypass, logger format).
    await health_check()

    logger.info("API warmup completed; first request will not pay cold-start cost")


@asynccontextmanager
async def lifespan(app_instance: FastAPI) -> AsyncIterator[None]:
    """ASGI lifespan handler: warmup + maintenance on startup, cleanup on stop.

    The maintenance task (see :mod:`omni_mercury_engine.api.maintenance`)
    prunes expired sessions/tokens, aged usage-ledger rows, and stale
    rate-limit state, and applies the TOTP sealing migration — once at
    startup, then periodically. It is failure-isolated: a maintenance error
    is logged and never blocks serving.
    """
    await _warmup(app_instance)
    from omni_mercury_engine.api.maintenance import start_maintenance_task

    maintenance_task = start_maintenance_task()
    try:
        yield
    finally:
        if maintenance_task is not None:
            maintenance_task.cancel()


# Initialize FastAPI application with comprehensive OpenAPI configuration
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    openapi_tags=tags_metadata,  # type: ignore[arg-type, unused-ignore]
    contact={
        "name": "Steel Security Advisors LLC",
        "url": "https://github.com/Steel-SecAdv-LLC/Mercury-Agent",
        "email": "steel.sa.llc@gmail.com",
    },
    license_info={
        "name": "GPL-3.0-or-later",
        "url": "https://www.gnu.org/licenses/gpl-3.0.html",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    servers=[
        {"url": "http://localhost:8000", "description": "Local development server"},
        {"url": "https://api.mercury-agent.org", "description": "Production server"},
    ],
    lifespan=lifespan,
)


# =============================================================================
# Rate Limiting Middleware
# =============================================================================
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token bucket rate limiting middleware.

    Uses the unified rate limiting module for consistent behavior across the API.
    Configurable via environment variables:
        - OMNI_RATE_LIMIT_ENABLED: Enable/disable rate limiting (default: true)
        - OMNI_RATE_LIMIT_REQUESTS_PER_MINUTE: Max requests per minute (default: 100)
        - OMNI_RATE_LIMIT_BURST: Burst size (default: 20)
        - MERCURY_TRUSTED_PROXY_HOPS: How many trailing X-Forwarded-For hops
          were appended by this deployment's own proxies (default 0 = the
          header is untrusted and the TCP peer address identifies the client)
        - MERCURY_KEYSTORE_PATH: When set, bucket state lives in the shared
          SQLite file — limits are global across workers and survive restarts
    """

    def __init__(
        self,
        app: FastAPI,
        requests_per_minute: int | None = None,
        burst_size: int | None = None,
    ):
        """Initialize the instance."""
        super().__init__(app)
        from omni_mercury_engine.api.rate_limit_store import build_shared_bucket_backend
        from omni_mercury_engine.security.rate_limiting import RateLimiter

        self.enabled = os.getenv("OMNI_RATE_LIMIT_ENABLED", "true").lower() == "true"
        self.requests_per_minute = requests_per_minute or int(
            os.getenv("OMNI_RATE_LIMIT_REQUESTS_PER_MINUTE", "100")
        )
        self.burst_size = burst_size or int(os.getenv("OMNI_RATE_LIMIT_BURST", "20"))

        # Unified rate limiter over the shared SQLite backend when
        # MERCURY_KEYSTORE_PATH is set (cross-worker, restart-persistent,
        # atomic consume) — otherwise the unchanged in-memory default.
        self._limiter = RateLimiter(
            requests_per_minute=self.requests_per_minute,
            burst_size=self.burst_size,
            backend=build_shared_bucket_backend(),
        )

    def _get_client_id(self, request: Request) -> str:
        """Extract the client identifier from the request.

        Resolution goes through :func:`~omni_mercury_engine.api.client_ip.
        resolve_client_ip`: X-Forwarded-For is only consulted when
        ``MERCURY_TRUSTED_PROXY_HOPS`` declares a proxy tier, and then only at
        the right-most trusted position — the left-most entry is
        client-writable and taking it (the previous behaviour) let any caller
        mint a fresh rate-limit bucket per request.
        """
        from omni_mercury_engine.api.client_ip import resolve_client_ip

        return resolve_client_ip(
            request.client.host if request.client else None,
            request.headers.get("X-Forwarded-For"),
        )

    def _check_rate_limit(self, client_id: str) -> tuple[bool, dict[str, int]]:
        """Check if request is within rate limit using unified rate limiter."""
        info = self._limiter.check(client_id)
        return info.allowed, {
            "limit": info.limit,
            "remaining": info.remaining,
            "reset": info.reset_at,
        }

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request with rate limiting."""
        # Skip rate limiting if disabled, for health checks, docs, and the
        # Prometheus scrape target (scrapers poll on a fixed interval and
        # must not drain the service budget of the scraper's IP).
        if not self.enabled or request.url.path in [
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]:
            return await call_next(request)

        client_id = self._get_client_id(request)
        allowed, info = self._check_rate_limit(client_id)

        if not allowed:
            return Response(
                content='{"error": "rate_limit_exceeded", "message": "Too many requests. Please retry later."}',
                status_code=429,
                media_type="application/json",
                headers={
                    "X-RateLimit-Limit": str(info["limit"]),
                    "X-RateLimit-Remaining": str(info["remaining"]),
                    "X-RateLimit-Reset": str(info["reset"]),
                    "Retry-After": "60",
                },
            )

        response = await call_next(request)

        # Add rate limit headers to all responses
        response.headers["X-RateLimit-Limit"] = str(info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(info["reset"])

        return response


# Register quota middleware FIRST so it runs INNERMOST (Starlette runs
# middleware in reverse registration order): a request must pass the global
# rate limiter before it can reserve account quota, and a 429 from either
# layer never double-charges the other.
from omni_mercury_engine.api.quota_middleware import QuotaMiddleware

app.add_middleware(QuotaMiddleware)  # type: ignore[arg-type, unused-ignore]

# Register rate limiting middleware
app.add_middleware(RateLimitMiddleware)  # type: ignore[arg-type, unused-ignore]


# =============================================================================
# Correlation ID Middleware (Observability & Distributed Tracing)
# =============================================================================
class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Middleware for request correlation ID tracking.

    Provides distributed tracing support by:
    - Accepting existing correlation IDs from upstream services (X-Correlation-ID header)
    - Generating new UUIDs for requests without correlation IDs
    - Propagating correlation IDs in responses for downstream tracing
    - Setting context variable for access throughout request lifecycle

    Headers:
        X-Correlation-ID: UUID for request tracing (in/out)
        X-Request-ID: Alias for X-Correlation-ID (in only)

    Usage in downstream code:
        from omni_mercury_engine.api.server import correlation_id_ctx
        correlation_id = correlation_id_ctx.get()
    """

    HEADER_NAME = "X-Correlation-ID"
    HEADER_ALIAS = "X-Request-ID"

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request with correlation ID tracking."""
        # Extract or generate correlation ID
        correlation_id = (
            request.headers.get(self.HEADER_NAME)
            or request.headers.get(self.HEADER_ALIAS)
            or str(uuid.uuid4())
        )

        # Set context variable for access in handlers
        token = correlation_id_ctx.set(correlation_id)

        # Add correlation ID to request state for easy access
        request.state.correlation_id = correlation_id

        # Track request timing
        start_time = time.perf_counter()

        try:
            response = await call_next(request)

            # Calculate request duration
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Add tracing headers to response
            response.headers[self.HEADER_NAME] = correlation_id
            response.headers["X-Request-Duration-Ms"] = f"{duration_ms:.2f}"

            # Record Prometheus HTTP metrics (http_requests_total /
            # http_request_duration_seconds) the monitoring stack + API HPA
            # consume. Best-effort and fully isolated: a metrics error must never
            # affect the response. Labels with the matched route template, and
            # collapses unmatched requests (404s, scanner/probe traffic on
            # arbitrary paths) to a single "__unmatched__" label — using the raw
            # URL path there would let external callers explode label cardinality.
            try:
                from omni_mercury_engine.core.metrics import record_http_request

                route = request.scope.get("route")
                endpoint = getattr(route, "path", None) or "__unmatched__"
                record_http_request(
                    request.method, endpoint, response.status_code, duration_ms / 1000.0
                )
            except Exception:  # pragma: no cover - metrics must never break a request
                pass

            # Log request completion with correlation ID
            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"status={response.status_code} duration={duration_ms:.2f}ms "
                f"correlation_id={correlation_id}"
            )

            return response
        finally:
            # Reset context variable
            correlation_id_ctx.reset(token)


# Register correlation ID middleware (runs before rate limiting)
app.add_middleware(CorrelationIDMiddleware)

# =============================================================================
# CORS Middleware Configuration (Security Hardening)
# =============================================================================
# Configure CORS based on environment
_cors_origins_env = os.getenv("MERCURY_CORS_ORIGINS", "")
_cors_allow_credentials = os.getenv("MERCURY_CORS_CREDENTIALS", "false").lower() == "true"

# In production, explicitly specify allowed origins
# In development, allow localhost origins
#
# Production-mode selection prefers the canonical ``MERCURY_ENV`` (see
# :mod:`omni_mercury_engine._env`); ``MERCURY_AGENT_ENV`` is retained
# as a backward-compatible alias because the v1.7.0 release shipped
# with the API server reading it directly, and operators may already
# have it baked into deployment manifests.  When both are set,
# ``MERCURY_ENV=production`` wins (matches the rest of the codebase);
# ``MERCURY_AGENT_ENV`` is honoured only when ``MERCURY_ENV`` is unset.
from omni_mercury_engine._env import is_production as _mercury_is_production

_mercury_env_raw = os.getenv("MERCURY_ENV", "").strip()
_agent_env_raw = os.getenv("MERCURY_AGENT_ENV", "").strip().lower()
if _mercury_env_raw:
    _is_production = _mercury_is_production()
else:
    _is_production = _agent_env_raw == "production"

if _is_production:
    # Production: require explicit origin configuration
    if _cors_origins_env:
        _raw_origins = [origin.strip() for origin in _cors_origins_env.split(",") if origin.strip()]
        _allowed_origins: list[str] = []
        for _origin in _raw_origins:
            from urllib.parse import urlparse as _urlparse

            _parsed = _urlparse(_origin)
            # Validate: must have scheme + netloc, no path beyond "/"
            if _parsed.scheme in ("http", "https") and _parsed.netloc and _parsed.path in ("", "/"):
                _allowed_origins.append(_origin)
            else:
                logger.warning("Ignoring invalid CORS origin: %s", _origin)
    else:
        # Default to same-origin only in production (no CORS)
        _allowed_origins = []
else:
    # Development: allow common local development origins
    _allowed_origins = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8080",
    ]
    # Add any custom origins from environment
    if _cors_origins_env:
        _allowed_origins.extend([origin.strip() for origin in _cors_origins_env.split(",")])

if _allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins,
        allow_credentials=_cors_allow_credentials,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
    )
    logger.info(f"CORS enabled for origins: {len(_allowed_origins)} configured")


# =============================================================================
# Security Response Headers
# =============================================================================
# Registered LAST, which makes it the OUTERMOST middleware (Starlette runs the
# stack in reverse registration order). Outermost is required for coverage: the
# rate limiter's 429 and the quota layer's 503 are produced by middleware and
# never reach a route handler, and a CORS preflight is answered by
# CORSMiddleware itself -- all three are browser-reachable responses that must
# still carry the header set.
from omni_mercury_engine.api.security_headers import SecurityHeadersMiddleware

app.add_middleware(SecurityHeadersMiddleware)  # type: ignore[arg-type, unused-ignore]


# Enums for API parameters
class DetectionMethod(StrEnum):
    """Available anomaly detection methods."""

    ZSCORE = "zscore"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"


class SeverityLevel(StrEnum):
    """Severity levels for detected anomalies."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Request Models with comprehensive validation and documentation
class UnivariateRequest(BaseModel):
    """Request model for univariate anomaly detection.

    This model accepts a single time series and optional parameters
    for configuring the detection sensitivity.

    Attributes:
        data: List of numerical values representing the time series.
            Minimum 3 data points required for statistical analysis.
        sensitivity: Detection sensitivity from 0.0 (least sensitive) to 1.0
            (most sensitive). Higher values detect more anomalies.
        method: Detection method to use (default: zscore).

    Example:
        ```json
        {
            "data": [1.0, 2.0, 1.5, 100.0, 1.8, 2.1],
            "sensitivity": 0.7
        }
        ```
    """

    data: list[float] = Field(
        ...,
        min_length=3,
        description="Time series data points. Minimum 3 data points required.",
        json_schema_extra={
            "example": [1.0, 2.0, 1.5, 10.0, 1.8, 2.1, 1.9, 50.0, 2.0],
        },
    )
    sensitivity: float | None = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Detection sensitivity (0.0-1.0). Higher = more sensitive.",
    )
    method: DetectionMethod | None = Field(
        default=DetectionMethod.ZSCORE,
        description="Detection method to use.",
    )

    @field_validator("data")
    @classmethod
    def validate_data(cls, v: list[float]) -> list[float]:
        """Validate that data contains finite values."""
        if not all(np.isfinite(x) for x in v):
            raise ValueError("Data contains non-finite values (NaN or Inf)")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "data": [1.0, 2.0, 1.5, 10.0, 1.8, 2.1, 1.9, 50.0, 2.0],
                    "sensitivity": 0.5,
                }
            ]
        }
    }


class MultivariateRequest(BaseModel):
    """Request model for multivariate anomaly detection.

    This model accepts multi-dimensional time series data where each row
    represents a time point and each column represents a feature/variable.

    Attributes:
        data: 2D array of numerical values. Each inner list represents
            a time point with multiple feature values.
        features: Optional list of feature names for result interpretation.
        sensitivity: Detection sensitivity from 0.0 to 1.0.
        correlate_features: Whether to consider cross-feature correlations.

    Example:
        ```json
        {
            "data": [[1.0, 2.0], [1.1, 2.1], [100.0, 2.0], [1.2, 2.2]],
            "features": ["temperature", "pressure"],
            "sensitivity": 0.6
        }
        ```
    """

    data: list[list[float]] = Field(
        ...,
        min_length=3,
        description="2D array of time series data. Shape: [n_samples, n_features]",
        json_schema_extra={
            "example": [[1.0, 2.0], [1.1, 2.1], [100.0, 2.0], [1.2, 2.2]],
        },
    )
    features: list[str] | None = Field(
        default=None,
        description="Feature names for result interpretation.",
    )
    sensitivity: float | None = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Detection sensitivity (0.0-1.0).",
    )
    correlate_features: bool | None = Field(
        default=True,
        description="Consider cross-feature correlations in detection.",
    )

    @field_validator("data")
    @classmethod
    def validate_multivariate_data(cls, v: list[list[float]]) -> list[list[float]]:
        """Validate multivariate data structure and values."""
        if not v:
            raise ValueError("Data cannot be empty")

        n_features = len(v[0])
        for i, row in enumerate(v):
            if len(row) != n_features:
                raise ValueError(f"Inconsistent feature count at row {i}")
            if not all(np.isfinite(x) for x in row):
                raise ValueError(f"Non-finite values at row {i}")

        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "data": [[1.0, 2.0], [1.1, 2.1], [100.0, 2.0], [1.2, 2.2]],
                    "features": ["temperature", "pressure"],
                    "sensitivity": 0.6,
                    "correlate_features": True,
                }
            ]
        }
    }


# Response Models
class HealthResponse(BaseModel):
    """Health check response model.

    Attributes:
        status: Service status ('healthy', 'degraded', 'unhealthy').
        version: API version string.
        uptime_seconds: Server uptime in seconds (if available).
    """

    status: str = Field(
        ..., description="Service health status", json_schema_extra={"example": "healthy"}
    )
    version: str = Field(..., description="API version", json_schema_extra={"example": API_VERSION})
    uptime_seconds: float | None = Field(default=None, description="Server uptime in seconds")

    model_config = {
        "json_schema_extra": {
            "examples": [{"status": "healthy", "version": API_VERSION, "uptime_seconds": 3600.5}]
        }
    }


class AnomalyPoint(BaseModel):
    """Individual anomaly point information.

    Attributes:
        index: Index of the anomaly in the input data.
        value: Original value at the anomaly point.
        score: Anomaly score (higher = more anomalous).
        severity: Severity level of the anomaly.
    """

    index: int = Field(..., description="Index position in input data")
    value: float = Field(..., description="Original data value")
    score: float = Field(..., description="Anomaly score (0.0-1.0+)")
    severity: SeverityLevel = Field(..., description="Severity classification")


class UnivariateResponse(BaseModel):
    """Response model for univariate anomaly detection.

    Attributes:
        anomalies: Boolean list indicating anomaly status per data point.
        scores: Numerical anomaly scores for each data point.
        anomaly_points: Detailed information about detected anomalies.
        method: Detection method used.
        threshold: Threshold value used for detection.
        summary: Summary statistics of the detection.
    """

    anomalies: list[bool] = Field(
        ..., description="Boolean flags for each data point (True = anomaly)"
    )
    scores: list[float] = Field(..., description="Anomaly scores for each data point")
    anomaly_points: list[AnomalyPoint] = Field(
        default_factory=list, description="Detailed anomaly information"
    )
    method: str = Field(..., description="Detection method used")
    threshold: float = Field(..., description="Detection threshold value")
    summary: dict[str, Any] = Field(
        default_factory=dict, description="Detection summary statistics"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "anomalies": [False, False, False, True, False],
                    "scores": [0.5, 0.6, 0.4, 3.5, 0.5],
                    "anomaly_points": [
                        {"index": 3, "value": 10.0, "score": 3.5, "severity": "high"}
                    ],
                    "method": "zscore",
                    "threshold": 2.5,
                    "summary": {"total_points": 5, "anomaly_count": 1, "anomaly_rate": 0.2},
                }
            ]
        }
    }


class MultivariateResponse(BaseModel):
    """Response model for multivariate anomaly detection.

    Attributes:
        anomalies: Boolean list indicating anomaly status per time point.
        scores: Combined anomaly scores for each time point.
        feature_contributions: Per-feature contribution to anomaly scores.
        method: Detection method used.
        threshold: Threshold value used for detection.
        features: Feature names used in analysis.
        summary: Summary statistics of the detection.
    """

    anomalies: list[bool] = Field(..., description="Boolean flags for each time point")
    scores: list[float] = Field(..., description="Combined anomaly scores")
    feature_contributions: dict[str, list[float]] | None = Field(
        default=None, description="Per-feature anomaly contributions"
    )
    method: str = Field(..., description="Detection method used")
    threshold: float = Field(..., description="Detection threshold value")
    features: list[str] = Field(..., description="Feature names")
    summary: dict[str, Any] = Field(
        default_factory=dict, description="Detection summary statistics"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "anomalies": [False, False, True, False],
                    "scores": [0.5, 0.6, 4.2, 0.5],
                    "feature_contributions": {
                        "temperature": [0.3, 0.4, 4.0, 0.3],
                        "pressure": [0.2, 0.2, 0.2, 0.2],
                    },
                    "method": "multivariate",
                    "threshold": 2.5,
                    "features": ["temperature", "pressure"],
                    "summary": {
                        "total_points": 4,
                        "anomaly_count": 1,
                        "anomaly_rate": 0.25,
                    },
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Standard error response model.

    Attributes:
        error: Error type/code.
        message: Human-readable error message.
        details: Additional error details (optional).
        request_id: Request ID for support reference.
    """

    error: str = Field(..., description="Error type/code")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] | None = Field(default=None, description="Additional error details")
    request_id: str | None = Field(default=None, description="Request ID for support")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": "ValidationError",
                    "message": "Data must contain at least 3 points",
                    "details": {"field": "data", "min_length": 3},
                    "request_id": "req_abc123",
                }
            ]
        }
    }


def _classify_severity(score: float, threshold: float) -> SeverityLevel:
    """Classify anomaly severity based on score.

    Args:
        score: The anomaly score.
        threshold: The detection threshold.

    Returns:
        SeverityLevel enum value.
    """
    ratio = score / threshold if threshold > 0 else score
    if ratio < 1.5:
        return SeverityLevel.LOW
    elif ratio < 2.5:
        return SeverityLevel.MEDIUM
    elif ratio < 4.0:
        return SeverityLevel.HIGH
    else:
        return SeverityLevel.CRITICAL


# API Endpoints
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health Check",
    description="Check the health status of the API service.",
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {"example": {"status": "healthy", "version": API_VERSION}}
            },
        },
        503: {
            "description": "Service is unhealthy",
            "model": ErrorResponse,
        },
    },
)
async def health_check() -> HealthResponse:
    """Check the health status of the API service.

    This endpoint can be used for load balancer health checks
    and monitoring service availability.

    Returns:
        HealthResponse: Service status and version information.

    Example:
        ```bash
        curl http://localhost:8000/health
        ```

        Response:
        ```json
        {"status": "healthy", "version": "x.y.z"}
        ```
    """
    return HealthResponse(status="healthy", version=API_VERSION)


@app.post(
    "/api/v1/detect/univariate",
    response_model=UnivariateResponse,
    tags=["Detection"],
    summary="Detect Univariate Anomalies",
    description="""
Detect anomalies in univariate (single-variable) time-series data.

## Algorithm

The `method` field selects the statistical algorithm (the algorithm actually
run is echoed in `summary.algorithm`; the response `method` field is the fixed
endpoint identity `"univariate"`):

- **`zscore`** (default): z = |x - μ| / σ; flag points where z > threshold, with
  `threshold = 2.0 + (1.0 - sensitivity) * 3.0` (sensitivity=1.0 → 2.0, most
  sensitive; sensitivity=0.0 → 5.0, least sensitive).
- **`iqr`**: score = distance outside the [Q1, Q3] box in IQR units; flag points
  above `threshold = 1.5 + (1.0 - sensitivity) * 3.0`.
- **`isolation_forest`**: model-based and **not** served by this lightweight
  endpoint — returns HTTP 400; use `POST /api/v1/detect/multivariate` instead.

A validation failure (bad input) returns HTTP 400 with structured detail.

## Use Cases

- Monitoring sensor readings for equipment failure detection
- Financial time series analysis for fraud detection
- Network traffic analysis for intrusion detection
- IoT device health monitoring
    """,
    responses={
        200: {
            "description": "Detection completed successfully",
            "model": UnivariateResponse,
        },
        400: {
            "description": "Invalid request data",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": {
                        "error": "ValidationError",
                        "message": "Data must contain at least 3 points",
                    }
                }
            },
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse,
        },
    },
)
async def detect_univariate(request: UnivariateRequest) -> UnivariateResponse:
    """Detect anomalies in univariate time-series data.

    This endpoint performs statistical anomaly detection on single-variable
    time series data using the z-score method with configurable sensitivity.

    Args:
        request: UnivariateRequest containing data and parameters.

    Returns:
        UnivariateResponse with anomaly flags, scores, and detailed analysis.

    Raises:
        HTTPException: 400 if data validation fails.
        HTTPException: 500 if detection processing fails.

    Example:
        Request:
        ```json
        {
            "data": [1.0, 2.0, 1.5, 10.0, 1.8],
            "sensitivity": 0.5
        }
        ```

        Response:
        ```json
        {
            "anomalies": [false, false, false, true, false],
            "scores": [0.73, 0.09, 0.32, 3.52, 0.12],
            "method": "zscore",
            "threshold": 3.5
        }
        ```
    """
    try:
        # P2: Comprehensive input validation
        validation_result = _api_validator.validate_univariate_request(
            data=request.data,
            sensitivity=request.sensitivity,
        )

        if not validation_result.is_valid:
            error_details = [e.to_dict() for e in validation_result.errors]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "ValidationError",
                    "message": "Input validation failed",
                    "errors": error_details,
                },
            )

        # Log warnings if any
        for warning in validation_result.warnings:
            logger.warning(f"Univariate request warning: {warning}")

        # Use validated/sanitized data
        data = validation_result.sanitized_data["data"]
        sensitivity = validation_result.sanitized_data["sensitivity"]

        # Dispatch on the requested algorithm. Previously the ``method`` field
        # was accepted and enum-validated but ignored -- every request ran
        # z-score regardless -- so ``method: iqr`` silently returned z-score
        # results. The response ``method`` stays "univariate" (the endpoint
        # identity that clients and the e2e contract depend on); the algorithm
        # actually run is recorded in ``summary["algorithm"]``.
        algorithm = request.method or DetectionMethod.ZSCORE
        data_arr = np.asarray(data, dtype=float)
        method_stats: dict[str, float]
        if algorithm == DetectionMethod.ZSCORE:
            threshold = 2.0 + (1.0 - sensitivity) * 3.0
            mean = float(np.mean(data_arr))
            std = float(np.std(data_arr))
            score_arr = np.abs((data_arr - mean) / (std + 1e-8))
            method_stats = {"mean": mean, "std": std}
        elif algorithm == DetectionMethod.IQR:
            # Score = distance outside the [Q1, Q3] box in IQR units; the
            # sensitivity-scaled threshold plays the role of the classic 1.5*IQR
            # fence (more sensitive -> lower fence).
            threshold = 1.5 + (1.0 - sensitivity) * 3.0
            q1 = float(np.percentile(data_arr, 25))
            q3 = float(np.percentile(data_arr, 75))
            iqr = q3 - q1
            outside = np.maximum.reduce([q1 - data_arr, data_arr - q3, np.zeros_like(data_arr)])
            score_arr = outside / (iqr + 1e-8)
            method_stats = {"q1": q1, "q3": q3, "iqr": iqr}
        else:  # ISOLATION_FOREST -- model-based, not served by this endpoint
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "method 'isolation_forest' is not available on the univariate "
                    "endpoint; use 'zscore' or 'iqr', or POST "
                    "/api/v1/detect/multivariate for model-based detection."
                ),
            )

        # Determine anomalies
        anomalies = (score_arr > threshold).tolist()
        scores = score_arr.tolist()
        mean = method_stats.get("mean", float(np.mean(data_arr)))
        std = method_stats.get("std", float(np.std(data_arr)))

        # Build detailed anomaly points
        anomaly_points = []
        for i, (is_anomaly, score, value) in enumerate(zip(anomalies, scores, data, strict=False)):
            if is_anomaly:
                anomaly_points.append(
                    AnomalyPoint(
                        index=i,
                        value=float(value),
                        score=score,
                        severity=_classify_severity(score, threshold),
                    )
                )

        # Build summary
        anomaly_count = sum(anomalies)
        summary = {
            "total_points": len(data),
            "anomaly_count": anomaly_count,
            "anomaly_rate": anomaly_count / len(data) if len(data) > 0 else 0,
            "algorithm": algorithm.value,
            "mean": float(mean),
            "std": float(std),
            "max_score": float(max(scores)) if scores else 0,
            **method_stats,
        }

        return UnivariateResponse(
            anomalies=anomalies,
            scores=scores,
            anomaly_points=anomaly_points,
            method="univariate",
            threshold=threshold,
            summary=summary,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except HTTPException:
        # Intentional HTTP errors -- notably the 400 raised above on validation
        # failure -- must propagate unchanged. Without this passthrough the broad
        # ``except Exception`` below re-wraps them as opaque 500s, hiding the
        # structured validation detail from the client (matches the ordering
        # already used by ``detect_multivariate``).
        raise
    except Exception as e:
        logger.error("Univariate detection failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during detection.",
        ) from e


@app.post(
    "/api/v1/detect/multivariate",
    response_model=MultivariateResponse,
    tags=["Detection"],
    summary="Detect Multivariate Anomalies",
    description="""
Detect anomalies in multivariate (multi-dimensional) time-series data.

## Algorithm

Uses multivariate z-score detection:
1. Calculate mean vector (μ) and standard deviation vector (σ) across features
2. Normalize each dimension: z_ij = (x_ij - μ_j) / σ_j
3. Compute combined score using L2 norm: score_i = ||z_i||₂
4. Flag points where score > threshold as anomalies

## Feature Correlations

When `correlate_features=True`, the algorithm considers cross-feature
correlations to detect anomalies that span multiple dimensions but might
not be detectable in individual features.

## Use Cases

- Multi-sensor industrial equipment monitoring
- Financial portfolio risk analysis
- Multi-metric system health monitoring
- Environmental monitoring with multiple sensors
    """,
    responses={
        200: {
            "description": "Detection completed successfully",
            "model": MultivariateResponse,
        },
        400: {
            "description": "Invalid request data",
            "model": ErrorResponse,
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse,
        },
    },
)
async def detect_multivariate(request: MultivariateRequest) -> MultivariateResponse:
    """Detect anomalies in multivariate time-series data.

    This endpoint performs statistical anomaly detection on multi-dimensional
    time series data, considering relationships between different features.

    Args:
        request: MultivariateRequest containing data matrix and parameters.

    Returns:
        MultivariateResponse with anomaly flags, scores, and feature analysis.

    Raises:
        HTTPException: 400 if data shape is invalid or validation fails.
        HTTPException: 500 if detection processing fails.

    Example:
        Request:
        ```json
        {
            "data": [[1.0, 2.0], [1.1, 2.1], [100.0, 2.0], [1.2, 2.2]],
            "features": ["temperature", "pressure"],
            "sensitivity": 0.6
        }
        ```

        Response:
        ```json
        {
            "anomalies": [false, false, true, false],
            "scores": [0.5, 0.6, 15.2, 0.5],
            "method": "multivariate",
            "threshold": 2.8
        }
        ```
    """
    try:
        # P2: Comprehensive input validation
        validation_result = _api_validator.validate_multivariate_request(
            data=request.data,
            features=request.features,
            sensitivity=request.sensitivity,
        )

        if not validation_result.is_valid:
            error_details = [e.to_dict() for e in validation_result.errors]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "ValidationError",
                    "message": "Input validation failed",
                    "errors": error_details,
                },
            )

        # Log warnings if any
        for warning in validation_result.warnings:
            logger.warning(f"Multivariate request warning: {warning}")

        # Use validated/sanitized data
        data = validation_result.sanitized_data["data"]
        sensitivity = validation_result.sanitized_data["sensitivity"]

        n_samples, n_features = data.shape

        # Calculate threshold based on sensitivity
        threshold = 2.0 + (1.0 - sensitivity) * 3.0

        # Compute multivariate z-scores
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        normalized = (data - mean) / std

        # Combined score using L2 norm
        z_scores = np.linalg.norm(normalized, axis=1)

        # Per-feature contributions - use sanitized feature names
        sanitized_features = validation_result.sanitized_data.get("features")
        feature_names = sanitized_features or [f"feature_{i}" for i in range(n_features)]
        feature_contributions = {
            name: np.abs(normalized[:, i]).tolist() for i, name in enumerate(feature_names)
        }

        # Determine anomalies
        anomalies = (z_scores > threshold).tolist()
        scores = z_scores.tolist()

        # Build summary
        anomaly_count = sum(anomalies)
        summary = {
            "total_points": n_samples,
            "n_features": n_features,
            "anomaly_count": anomaly_count,
            "anomaly_rate": anomaly_count / n_samples if n_samples > 0 else 0,
            "max_score": float(max(scores)) if scores else 0,
            "feature_importance": {
                name: float(np.mean(np.abs(normalized[:, i])))
                for i, name in enumerate(feature_names)
            },
        }

        return MultivariateResponse(
            anomalies=anomalies,
            scores=scores,
            feature_contributions=feature_contributions,
            method="multivariate",
            threshold=threshold,
            features=feature_names,
            summary=summary,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error("Multivariate detection failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during detection.",
        ) from e


# Custom OpenAPI schema
def custom_openapi() -> dict[str, Any]:
    """Generate custom OpenAPI schema with additional documentation.

    Returns:
        Dict containing the complete OpenAPI specification.
    """
    if app.openapi_schema:
        return dict(app.openapi_schema)

    openapi_schema: dict[str, Any] = get_openapi(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        routes=app.routes,
        tags=tags_metadata,  # type: ignore[arg-type, unused-ignore]
    )

    # Add security schemes for future use
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for authentication (future implementation)",
        },
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token for authentication (future implementation)",
        },
    }

    # Add rate limiting information
    openapi_schema["info"]["x-rate-limit"] = {
        "requests_per_minute": 100,
        "burst_limit": 20,
    }

    app.openapi_schema = openapi_schema
    return openapi_schema


app.openapi = custom_openapi  # type: ignore[method-assign, unused-ignore]

# =============================================================================
# Include Modular API Routes
# =============================================================================

# Include Batch Processing Routes
try:
    from omni_mercury_engine.api.routes.batch import router as batch_router

    app.include_router(batch_router)
    logger.info("Batch processing routes registered")
except ImportError as e:
    logger.warning(f"Batch processing routes not available: {e}")

# Include Model Management Routes
try:
    from omni_mercury_engine.api.routes.models import router as models_router

    app.include_router(models_router)
    logger.info("Model management routes registered")
except ImportError as e:
    logger.warning(f"Model management routes not available: {e}")

# Include Data Export Routes
try:
    from omni_mercury_engine.api.routes.export import router as export_router

    app.include_router(export_router)
    logger.info("Data export routes registered")
except ImportError as e:
    logger.warning(f"Data export routes not available: {e}")

# Include Advanced Detection Routes
try:
    from omni_mercury_engine.api.routes.detection import router as detection_router

    app.include_router(detection_router)
    logger.info("Advanced detection routes registered")
except ImportError as e:
    logger.warning(f"Advanced detection routes not available: {e}")

# Include Self-Service Account Routes (opt-in: inert unless the flows are used;
# store/mailer default to in-memory/console without MERCURY_KEYSTORE_PATH/SMTP)
try:
    from omni_mercury_engine.api.routes.accounts import router as accounts_router

    app.include_router(accounts_router)
    logger.info("Account routes registered")
except ImportError as e:
    logger.warning(f"Account routes not available: {e}")

# Include Hazard Visualization Routes
try:
    from omni_mercury_engine.api.routes.hazard import router as hazard_router

    app.include_router(hazard_router)
    logger.info("Hazard visualization routes registered")
except ImportError as e:
    logger.warning(f"Hazard visualization routes not available: {e}")

# Serve the Account Frontend (opt-in via MERCURY_FRONTEND_ENABLED=true; left
# off, every existing route — including the 404 at / — stays byte-identical)
try:
    from omni_mercury_engine.api.frontend import frontend_enabled, register_frontend

    if frontend_enabled():
        register_frontend(app)
        logger.info("Account frontend registered")
except ImportError as e:
    logger.warning(f"Account frontend not available: {e}")

# Include Voice Interface Routes
try:
    from omni_mercury_engine.api.voice import router as voice_router

    app.include_router(voice_router)
    logger.info("Voice interface routes registered")
except ImportError as e:
    logger.warning(f"Voice interface routes not available: {e}")


# Prometheus metrics at the conventional root path (/metrics) — the target scraped
# by monitoring/prometheus/prometheus.yml, the k8s ``prometheus.io/path``
# annotations, and the Helm chart. The Prometheus exposition handler lives on
# health_router (api/health.py); register it directly on the app at root so the
# scrape target resolves instead of returning 404.
try:
    from omni_mercury_engine.api.health import health_metrics as _health_metrics

    app.add_api_route(
        "/metrics",
        _health_metrics,
        methods=["GET"],
        tags=["Health"],
        include_in_schema=False,
    )
    logger.info("Prometheus /metrics endpoint registered at root")
except ImportError as e:
    logger.warning(f"/metrics endpoint not available: {e}")


# =============================================================================
# Server Startup Function
# =============================================================================
def run_server(
    host: str | None = None,
    port: int = 8000,
    workers: int = 1,
    reload: bool = False,
    log_level: str = "info",
) -> None:
    """Run the Mercury Agent API server.

    Args:
        host: Host address to bind to. Defaults to MERCURY_HOST env var or 127.0.0.1.
              Set MERCURY_HOST=0.0.0.0 for production deployments requiring external access.
        port: Port number to listen on
        workers: Number of worker processes
        reload: Enable auto-reload for development
        log_level: Logging level (debug, info, warning, error)
    """
    # Security: Default to localhost (127.0.0.1) for safety
    # Use MERCURY_HOST environment variable for production deployments
    if host is None:
        host = os.environ.get("MERCURY_HOST", "127.0.0.1")
    import uvicorn

    uvicorn.run(
        "omni_mercury_engine.api.server:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level=log_level,
    )


if __name__ == "__main__":
    run_server()
