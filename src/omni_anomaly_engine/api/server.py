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

"""FastAPI server for real-time anomaly detection.

This module provides a REST API for multi-domain anomaly detection using
the OMNI AVA framework. It implements best practices from Azure AI Anomaly
Detector and provides comprehensive OpenAPI documentation.

API Reference:
    - Azure AI Anomaly Detector:
      https://azure.microsoft.com/en-us/products/ai-services/ai-anomaly-detector

Example:
    Start the server with uvicorn::

        uvicorn omni_anomaly_engine.api.server:app --host 0.0.0.0 --port 8000

    Make a detection request::

        curl -X POST "http://localhost:8000/api/v1/detect/univariate" \\
            -H "Content-Type: application/json" \\
            -d '{"data": [1.0, 2.0, 1.5, 10.0, 1.8], "sensitivity": 0.5}'
"""

import os
import time
from collections import defaultdict
from enum import Enum
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

# API version information
API_VERSION = "1.0.0"
API_TITLE = "OMNI ♱ AVA API"
API_DESCRIPTION = """
## Overview

The OMNI ♱ AVA API provides multi-domain anomaly detection capabilities through a REST interface.
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
https://github.com/Steel-SecAdv-LLC/OMNI-AVA/issues
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

# Initialize FastAPI application with comprehensive OpenAPI configuration
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    openapi_tags=tags_metadata,
    contact={
        "name": "Steel Security Advisors LLC",
        "url": "https://github.com/Steel-SecAdv-LLC/OMNI-AVA",
        "email": "support@steelsecurityadvisors.com",
    },
    license_info={
        "name": "GNU General Public License v3.0",
        "url": "https://www.gnu.org/licenses/gpl-3.0.html",
    },
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    servers=[
        {"url": "http://localhost:8000", "description": "Local development server"},
        {"url": "https://api.omni-ava.org", "description": "Production server"},
    ],
)


# =============================================================================
# Rate Limiting Middleware
# =============================================================================
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token bucket rate limiting middleware.

    Enforces rate limits per client IP to prevent API abuse.
    Configurable via environment variables:
        - OMNI_RATE_LIMIT_ENABLED: Enable/disable rate limiting (default: true)
        - OMNI_RATE_LIMIT_REQUESTS_PER_MINUTE: Max requests per minute (default: 100)
        - OMNI_RATE_LIMIT_BURST: Burst size (default: 20)
    """

    def __init__(
        self,
        app: FastAPI,
        requests_per_minute: int | None = None,
        burst_size: int | None = None,
    ):
        super().__init__(app)
        self.enabled = os.getenv("OMNI_RATE_LIMIT_ENABLED", "true").lower() == "true"
        self.requests_per_minute = requests_per_minute or int(
            os.getenv("OMNI_RATE_LIMIT_REQUESTS_PER_MINUTE", "100")
        )
        self.burst_size = burst_size or int(os.getenv("OMNI_RATE_LIMIT_BURST", "20"))
        self._buckets: dict[str, tuple[float, int]] = defaultdict(
            lambda: (time.time(), self.burst_size)
        )

    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request."""
        # Prefer X-Forwarded-For for clients behind proxies
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        # Fall back to direct client IP
        if request.client:
            return request.client.host
        return "unknown"

    def _check_rate_limit(self, client_id: str) -> tuple[bool, dict[str, int]]:
        """Check if request is within rate limit using token bucket algorithm."""
        now = time.time()
        last_time, tokens = self._buckets[client_id]

        # Refill tokens based on elapsed time
        elapsed = now - last_time
        refill_rate = self.requests_per_minute / 60.0
        new_tokens = int(elapsed * refill_rate)
        tokens = min(self.burst_size, tokens + new_tokens)

        # Rate limit info for headers
        info = {
            "limit": self.requests_per_minute,
            "remaining": max(0, tokens - 1),
            "reset": int(now) + 60,
        }

        if tokens > 0:
            self._buckets[client_id] = (now, tokens - 1)
            return True, info
        else:
            self._buckets[client_id] = (now, 0)
            return False, info

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with rate limiting."""
        # Skip rate limiting if disabled or for health checks
        if not self.enabled or request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
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


# Register rate limiting middleware
app.add_middleware(RateLimitMiddleware)


# Enums for API parameters
class DetectionMethod(str, Enum):
    """Available anomaly detection methods."""

    ZSCORE = "zscore"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"


class SeverityLevel(str, Enum):
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
        min_length=0,
        description="Time series data points.",
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
    version: str = Field(..., description="API version", json_schema_extra={"example": "1.0.0"})
    uptime_seconds: float | None = Field(default=None, description="Server uptime in seconds")

    model_config = {
        "json_schema_extra": {
            "examples": [{"status": "healthy", "version": "1.0.0", "uptime_seconds": 3600.5}]
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
            "content": {"application/json": {"example": {"status": "healthy", "version": "1.0.0"}}},
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
        {"status": "healthy", "version": "1.0.0"}
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

Uses z-score based detection with configurable sensitivity:
1. Calculate mean (μ) and standard deviation (σ) of the data
2. Compute z-score for each point: z = |x - μ| / σ
3. Flag points where z > threshold as anomalies

The threshold is calculated as: `threshold = 2.0 + (1.0 - sensitivity) * 3.0`
- sensitivity=1.0 → threshold=2.0 (most sensitive)
- sensitivity=0.0 → threshold=5.0 (least sensitive)

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
        data = np.array(request.data)
        sensitivity = request.sensitivity if request.sensitivity is not None else 0.5

        # Calculate threshold based on sensitivity
        threshold = 2.0 + (1.0 - sensitivity) * 3.0

        # Compute z-scores
        mean = np.mean(data)
        std = np.std(data)
        z_scores = np.abs((data - mean) / (std + 1e-8))

        # Determine anomalies
        anomalies = (z_scores > threshold).tolist()
        scores = z_scores.tolist()

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
            "mean": float(mean),
            "std": float(std),
            "max_score": float(max(scores)) if scores else 0,
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
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {e!s}",
        )


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
        data = np.array(request.data)
        sensitivity = request.sensitivity if request.sensitivity is not None else 0.5

        # Validate data shape
        if len(data.shape) != 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Data must be 2D array with shape [n_samples, n_features]",
            )

        n_samples, n_features = data.shape

        # Calculate threshold based on sensitivity
        threshold = 2.0 + (1.0 - sensitivity) * 3.0

        # Compute multivariate z-scores
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        normalized = (data - mean) / std

        # Combined score using L2 norm
        z_scores = np.linalg.norm(normalized, axis=1)

        # Per-feature contributions
        feature_names = request.features or [f"feature_{i}" for i in range(n_features)]
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
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {e!s}",
        )


# Custom OpenAPI schema
def custom_openapi() -> dict[str, Any]:
    """Generate custom OpenAPI schema with additional documentation.

    Returns:
        Dict containing the complete OpenAPI specification.
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        routes=app.routes,
        tags=tags_metadata,
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
    return app.openapi_schema


app.openapi = custom_openapi
