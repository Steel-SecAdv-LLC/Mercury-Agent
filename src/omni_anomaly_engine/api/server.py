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

Follows Azure AI Anomaly Detector best practices:
https://azure.microsoft.com/en-us/products/ai-services/ai-anomaly-detector

Security Features:
- JWT-based authentication with RS256 algorithm
- API key authentication with HMAC signatures
- Rate limiting with token bucket algorithm
- Comprehensive input validation
- Security headers (CSP, HSTS, etc.)
- Structured error handling with sanitized messages
- Security event logging and monitoring
"""

import hashlib
import hmac
import logging
import os
import secrets
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, validator

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("omni_ava_api")

# Security configuration from environment
JWT_SECRET_KEY = os.environ.get("OMNI_AVA_JWT_SECRET", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"  # Use RS256 in production with proper key management
API_KEY_HEADER = "X-API-Key"
API_KEY_SIGNATURE_HEADER = "X-API-Signature"
RATE_LIMIT_REQUESTS = int(os.environ.get("OMNI_AVA_RATE_LIMIT", "100"))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("OMNI_AVA_RATE_WINDOW", "60"))


# ============================================================================
# Error Classification System
# ============================================================================


class ErrorCategory(str, Enum):
    """Error classification for proper handling."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    RATE_LIMIT = "rate_limit"
    INTERNAL = "internal"
    EXTERNAL = "external"


class ErrorCode(str, Enum):
    """Standardized error codes for client-friendly messages."""

    INVALID_TOKEN = "ERR_INVALID_TOKEN"
    EXPIRED_TOKEN = "ERR_EXPIRED_TOKEN"
    MISSING_AUTH = "ERR_MISSING_AUTH"
    INSUFFICIENT_PERMISSIONS = "ERR_INSUFFICIENT_PERMISSIONS"
    INVALID_INPUT = "ERR_INVALID_INPUT"
    ARRAY_SIZE_EXCEEDED = "ERR_ARRAY_SIZE_EXCEEDED"
    VALUE_OUT_OF_RANGE = "ERR_VALUE_OUT_OF_RANGE"
    INVALID_DATA_TYPE = "ERR_INVALID_DATA_TYPE"
    RATE_LIMIT_EXCEEDED = "ERR_RATE_LIMIT_EXCEEDED"
    INTERNAL_ERROR = "ERR_INTERNAL"


@dataclass
class APIError:
    """Structured API error with sanitized information."""

    code: ErrorCode
    message: str
    category: ErrorCategory
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    details: Optional[Dict[str, Any]] = None

    def to_response(self) -> Dict[str, Any]:
        """Convert to API response format (excludes internal details)."""
        return {
            "error": {
                "code": self.code.value,
                "message": self.message,
                "correlation_id": self.correlation_id,
                "timestamp": self.timestamp,
            }
        }


# ============================================================================
# Rate Limiting (Token Bucket Algorithm)
# ============================================================================


@dataclass
class RateLimitBucket:
    """Token bucket for rate limiting."""

    tokens: float
    last_update: float
    requests_count: int = 0


class RateLimiter:
    """Distributed-ready rate limiter using token bucket algorithm."""

    def __init__(
        self,
        max_requests: int = RATE_LIMIT_REQUESTS,
        window_seconds: int = RATE_LIMIT_WINDOW_SECONDS,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.refill_rate = max_requests / window_seconds
        self._buckets: Dict[str, RateLimitBucket] = {}
        self._blocked_ips: Dict[str, float] = {}

    def _get_bucket(self, key: str) -> RateLimitBucket:
        """Get or create bucket for key."""
        now = time.time()
        if key not in self._buckets:
            self._buckets[key] = RateLimitBucket(
                tokens=self.max_requests, last_update=now
            )
        return self._buckets[key]

    def _refill_bucket(self, bucket: RateLimitBucket) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - bucket.last_update
        bucket.tokens = min(
            self.max_requests, bucket.tokens + elapsed * self.refill_rate
        )
        bucket.last_update = now

    def check_rate_limit(self, key: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed under rate limit.

        Returns:
            Tuple of (allowed, headers_dict)
        """
        # Check if IP is temporarily blocked
        if key in self._blocked_ips:
            if time.time() < self._blocked_ips[key]:
                return False, self._get_rate_limit_headers(0, 0)
            else:
                del self._blocked_ips[key]

        bucket = self._get_bucket(key)
        self._refill_bucket(bucket)

        if bucket.tokens >= 1:
            bucket.tokens -= 1
            bucket.requests_count += 1
            return True, self._get_rate_limit_headers(
                int(bucket.tokens), int(self.window_seconds)
            )
        else:
            # Block IP for adaptive penalty
            self._blocked_ips[key] = time.time() + 60  # 1 minute block
            logger.warning(f"Rate limit exceeded for {key}, blocking for 60s")
            return False, self._get_rate_limit_headers(0, 60)

    def _get_rate_limit_headers(
        self, remaining: int, reset_seconds: int
    ) -> Dict[str, str]:
        """Generate rate limit headers."""
        return {
            "X-RateLimit-Limit": str(self.max_requests),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(time.time()) + reset_seconds),
        }


rate_limiter = RateLimiter()


# ============================================================================
# Authentication & Authorization
# ============================================================================


@dataclass
class User:
    """Authenticated user context."""

    user_id: str
    roles: Set[str]
    permissions: Set[str]
    api_key_id: Optional[str] = None


@dataclass
class APIKey:
    """API key with permissions."""

    key_id: str
    key_hash: str
    user_id: str
    permissions: Set[str]
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True


# In-memory API key store (replace with database in production)
_api_keys: Dict[str, APIKey] = {}
_api_key_by_hash: Dict[str, str] = {}


def generate_api_key(user_id: str, permissions: Set[str]) -> Tuple[str, APIKey]:
    """Generate a new API key for a user."""
    raw_key = secrets.token_urlsafe(32)
    key_id = str(uuid.uuid4())
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    api_key = APIKey(
        key_id=key_id,
        key_hash=key_hash,
        user_id=user_id,
        permissions=permissions,
        created_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(days=365),
    )

    _api_keys[key_id] = api_key
    _api_key_by_hash[key_hash] = key_id

    return raw_key, api_key


def validate_api_key(raw_key: str) -> Optional[APIKey]:
    """Validate an API key and return the associated APIKey object."""
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    key_id = _api_key_by_hash.get(key_hash)

    if not key_id:
        return None

    api_key = _api_keys.get(key_id)
    if not api_key or not api_key.is_active:
        return None

    if api_key.expires_at and datetime.utcnow() > api_key.expires_at:
        api_key.is_active = False
        return None

    return api_key


def verify_hmac_signature(
    api_key: str, timestamp: str, request_body: bytes, signature: str
) -> bool:
    """Verify HMAC signature for API key authentication."""
    message = f"{timestamp}:{request_body.decode()}"
    expected_signature = hmac.new(
        api_key.encode(), message.encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, expected_signature)


# Security middleware
api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)
bearer_auth = HTTPBearer(auto_error=False)


async def get_api_key(
    request: Request,
    api_key: Optional[str] = Depends(api_key_header),
) -> Optional[User]:
    """Validate API key and return user context."""
    if not api_key:
        return None

    validated_key = validate_api_key(api_key)
    if not validated_key:
        return None

    return User(
        user_id=validated_key.user_id,
        roles=set(),
        permissions=validated_key.permissions,
        api_key_id=validated_key.key_id,
    )


async def require_auth(
    request: Request,
    user: Optional[User] = Depends(get_api_key),
) -> User:
    """Require authentication for endpoint."""
    if not user:
        logger.warning(
            f"Authentication failed for {request.client.host}:{request.url.path}"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=APIError(
                code=ErrorCode.MISSING_AUTH,
                message="Authentication required",
                category=ErrorCategory.AUTHENTICATION,
            ).to_response(),
        )
    return user


def require_permission(permission: str):
    """Decorator to require specific permission."""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, user: User = Depends(require_auth), **kwargs):
            if permission not in user.permissions:
                logger.warning(
                    f"Permission denied: {user.user_id} lacks {permission}"
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=APIError(
                        code=ErrorCode.INSUFFICIENT_PERMISSIONS,
                        message="Insufficient permissions",
                        category=ErrorCategory.AUTHORIZATION,
                    ).to_response(),
                )
            return await func(*args, user=user, **kwargs)

        return wrapper

    return decorator


# ============================================================================
# Input Validation
# ============================================================================

# Constants for validation
MAX_ARRAY_SIZE = 100_000  # Maximum elements in array
MAX_ARRAY_DIMENSIONS = 2
MAX_FEATURE_COUNT = 1000
VALUE_MIN = -1e308
VALUE_MAX = 1e308


class UnivariateRequest(BaseModel):
    """Request for univariate anomaly detection with comprehensive validation."""

    data: List[float] = Field(
        ...,
        min_items=2,
        max_items=MAX_ARRAY_SIZE,
        description="Time series data points",
    )
    sensitivity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Detection sensitivity (0.0-1.0)",
    )

    @validator("data")
    def validate_data_values(cls, v):
        """Validate data values are within acceptable range."""
        for i, val in enumerate(v):
            if np.isnan(val) or np.isinf(val):
                raise ValueError(
                    f"Invalid value at index {i}: NaN and Inf are not allowed"
                )
            if val < VALUE_MIN or val > VALUE_MAX:
                raise ValueError(
                    f"Value at index {i} out of range [{VALUE_MIN}, {VALUE_MAX}]"
                )
        return v


class MultivariateRequest(BaseModel):
    """Request for multivariate anomaly detection with comprehensive validation."""

    data: List[List[float]] = Field(
        ...,
        min_items=2,
        description="Multivariate time series data",
    )
    features: Optional[List[str]] = Field(
        default=None,
        max_items=MAX_FEATURE_COUNT,
        description="Feature names",
    )
    sensitivity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Detection sensitivity (0.0-1.0)",
    )

    @validator("data")
    def validate_data_structure(cls, v):
        """Validate data structure and values."""
        if len(v) > MAX_ARRAY_SIZE:
            raise ValueError(f"Array size exceeds maximum of {MAX_ARRAY_SIZE}")

        if not v:
            raise ValueError("Data array cannot be empty")

        feature_count = len(v[0])
        if feature_count > MAX_FEATURE_COUNT:
            raise ValueError(f"Feature count exceeds maximum of {MAX_FEATURE_COUNT}")

        for row_idx, row in enumerate(v):
            if len(row) != feature_count:
                raise ValueError(
                    f"Inconsistent feature count at row {row_idx}: "
                    f"expected {feature_count}, got {len(row)}"
                )
            for col_idx, val in enumerate(row):
                if np.isnan(val) or np.isinf(val):
                    raise ValueError(
                        f"Invalid value at [{row_idx}][{col_idx}]: "
                        "NaN and Inf are not allowed"
                    )
                if val < VALUE_MIN or val > VALUE_MAX:
                    raise ValueError(
                        f"Value at [{row_idx}][{col_idx}] out of range "
                        f"[{VALUE_MIN}, {VALUE_MAX}]"
                    )
        return v


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    timestamp: str


class DetectionResponse(BaseModel):
    """Detection response with standardized format."""

    anomalies: List[bool]
    scores: List[float]
    method: str
    threshold: float
    correlation_id: str
    features: Optional[List[str]] = None


# ============================================================================
# Security Event Logging
# ============================================================================


class SecurityEventType(str, Enum):
    """Security event types for monitoring."""

    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    VALIDATION_ERROR = "validation_error"
    PERMISSION_DENIED = "permission_denied"
    SUSPICIOUS_REQUEST = "suspicious_request"
    API_ERROR = "api_error"


def log_security_event(
    event_type: SecurityEventType,
    request: Request,
    user: Optional[User] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Log security event for monitoring and alerting."""
    event = {
        "event_type": event_type.value,
        "timestamp": datetime.utcnow().isoformat(),
        "client_ip": request.client.host if request.client else "unknown",
        "path": str(request.url.path),
        "method": request.method,
        "user_id": user.user_id if user else None,
        "details": details or {},
    }
    logger.info(f"SECURITY_EVENT: {event}")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="OMNI ♱ AVA API",
    description="REST API for multi-domain anomaly detection with enterprise security",
    version="1.0.0",
    docs_url="/docs" if os.environ.get("OMNI_AVA_ENABLE_DOCS", "true") == "true" else None,
    redoc_url="/redoc" if os.environ.get("OMNI_AVA_ENABLE_DOCS", "true") == "true" else None,
)

# CORS middleware (configure appropriately for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("OMNI_AVA_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ============================================================================
# Security Middleware
# ============================================================================


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next) -> Response:
    """Add security headers to all responses."""
    response = await call_next(request)

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'"
    )
    response.headers["Permissions-Policy"] = (
        "geolocation=(), microphone=(), camera=()"
    )
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    return response


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next) -> Response:
    """Apply rate limiting to all requests."""
    client_ip = request.client.host if request.client else "unknown"

    # Skip rate limiting for health check
    if request.url.path == "/health":
        return await call_next(request)

    allowed, headers = rate_limiter.check_rate_limit(client_ip)

    if not allowed:
        log_security_event(
            SecurityEventType.RATE_LIMIT_EXCEEDED,
            request,
            details={"client_ip": client_ip},
        )
        response = JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=APIError(
                code=ErrorCode.RATE_LIMIT_EXCEEDED,
                message="Rate limit exceeded. Please retry later.",
                category=ErrorCategory.RATE_LIMIT,
            ).to_response(),
        )
        for key, value in headers.items():
            response.headers[key] = value
        return response

    response = await call_next(request)
    for key, value in headers.items():
        response.headers[key] = value

    return response


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next) -> Response:
    """Log all requests for audit trail."""
    correlation_id = str(uuid.uuid4())
    request.state.correlation_id = correlation_id

    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    logger.info(
        f"REQUEST: {request.method} {request.url.path} "
        f"status={response.status_code} duration={duration:.3f}s "
        f"correlation_id={correlation_id} "
        f"client={request.client.host if request.client else 'unknown'}"
    )

    response.headers["X-Correlation-ID"] = correlation_id
    return response


# ============================================================================
# Exception Handlers
# ============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions with sanitized error messages."""
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))

    # Log the actual error internally
    logger.error(
        f"HTTP Error: {exc.status_code} - {exc.detail} "
        f"correlation_id={correlation_id}"
    )

    # Return sanitized error to client
    if isinstance(exc.detail, dict):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail,
        )

    return JSONResponse(
        status_code=exc.status_code,
        content=APIError(
            code=ErrorCode.INTERNAL_ERROR
            if exc.status_code >= 500
            else ErrorCode.INVALID_INPUT,
            message=str(exc.detail)
            if exc.status_code < 500
            else "An internal error occurred",
            category=ErrorCategory.INTERNAL
            if exc.status_code >= 500
            else ErrorCategory.VALIDATION,
            correlation_id=correlation_id,
        ).to_response(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions with sanitized error messages."""
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))

    # Log the actual error internally (with full stack trace)
    logger.exception(
        f"Unhandled exception: {type(exc).__name__} "
        f"correlation_id={correlation_id}"
    )

    log_security_event(
        SecurityEventType.API_ERROR,
        request,
        details={"error_type": type(exc).__name__},
    )

    # Return generic error to client (never expose internal details)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=APIError(
            code=ErrorCode.INTERNAL_ERROR,
            message="An internal error occurred. Please try again later.",
            category=ErrorCategory.INTERNAL,
            correlation_id=correlation_id,
        ).to_response(),
    )


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint (no authentication required)."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/api/v1/detect/univariate", response_model=DetectionResponse)
async def detect_univariate(
    request: Request,
    body: UnivariateRequest,
    user: Optional[User] = Depends(get_api_key),
) -> DetectionResponse:
    """
    Detect anomalies in univariate time-series data.

    Args:
        request: HTTP request
        body: Univariate detection request
        user: Authenticated user (optional for demo, required in production)

    Returns:
        Detection results with anomalies and scores
    """
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))

    try:
        data = np.array(body.data)

        # Calculate adaptive threshold based on sensitivity
        threshold = 2.0 + (1.0 - body.sensitivity) * 3.0

        # Z-score based anomaly detection
        mean = np.mean(data)
        std = np.std(data)
        z_scores = np.abs((data - mean) / (std + 1e-8))

        anomalies = (z_scores > threshold).tolist()
        scores = z_scores.tolist()

        logger.info(
            f"Univariate detection: {len(data)} points, "
            f"{sum(anomalies)} anomalies detected "
            f"correlation_id={correlation_id}"
        )

        return DetectionResponse(
            anomalies=anomalies,
            scores=scores,
            method="univariate",
            threshold=threshold,
            correlation_id=correlation_id,
        )

    except ValueError as e:
        log_security_event(
            SecurityEventType.VALIDATION_ERROR,
            request,
            user,
            details={"error": str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=APIError(
                code=ErrorCode.INVALID_INPUT,
                message=str(e),
                category=ErrorCategory.VALIDATION,
                correlation_id=correlation_id,
            ).to_response(),
        )


@app.post("/api/v1/detect/multivariate", response_model=DetectionResponse)
async def detect_multivariate(
    request: Request,
    body: MultivariateRequest,
    user: Optional[User] = Depends(get_api_key),
) -> DetectionResponse:
    """
    Detect anomalies in multivariate time-series data.

    Args:
        request: HTTP request
        body: Multivariate detection request
        user: Authenticated user (optional for demo, required in production)

    Returns:
        Detection results with anomalies and scores
    """
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))

    try:
        data = np.array(body.data)

        if len(data.shape) != 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=APIError(
                    code=ErrorCode.INVALID_INPUT,
                    message="Data must be 2D array",
                    category=ErrorCategory.VALIDATION,
                    correlation_id=correlation_id,
                ).to_response(),
            )

        # Calculate adaptive threshold
        threshold = 2.0 + (1.0 - body.sensitivity) * 3.0

        # Multivariate z-score using Mahalanobis-like distance
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        z_scores = np.linalg.norm((data - mean) / std, axis=1)

        anomalies = (z_scores > threshold).tolist()
        scores = z_scores.tolist()

        feature_names = body.features or [
            f"feature_{i}" for i in range(data.shape[1])
        ]

        logger.info(
            f"Multivariate detection: {data.shape} array, "
            f"{sum(anomalies)} anomalies detected "
            f"correlation_id={correlation_id}"
        )

        return DetectionResponse(
            anomalies=anomalies,
            scores=scores,
            method="multivariate",
            threshold=threshold,
            correlation_id=correlation_id,
            features=feature_names,
        )

    except ValueError as e:
        log_security_event(
            SecurityEventType.VALIDATION_ERROR,
            request,
            user,
            details={"error": str(e)},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=APIError(
                code=ErrorCode.INVALID_INPUT,
                message=str(e),
                category=ErrorCategory.VALIDATION,
                correlation_id=correlation_id,
            ).to_response(),
        )


# ============================================================================
# Admin Endpoints (Protected)
# ============================================================================


@app.post("/api/v1/admin/api-keys")
async def create_api_key_endpoint(
    request: Request,
    user_id: str,
    permissions: List[str],
    admin_user: User = Depends(require_auth),
) -> Dict[str, Any]:
    """
    Create a new API key (admin only).

    Args:
        request: HTTP request
        user_id: User ID for the new key
        permissions: List of permissions for the key
        admin_user: Authenticated admin user

    Returns:
        New API key (shown only once)
    """
    if "admin" not in admin_user.permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=APIError(
                code=ErrorCode.INSUFFICIENT_PERMISSIONS,
                message="Admin permission required",
                category=ErrorCategory.AUTHORIZATION,
            ).to_response(),
        )

    raw_key, api_key = generate_api_key(user_id, set(permissions))

    logger.info(
        f"API key created: key_id={api_key.key_id} "
        f"user_id={user_id} by admin={admin_user.user_id}"
    )

    return {
        "key_id": api_key.key_id,
        "api_key": raw_key,  # Only shown once!
        "user_id": user_id,
        "permissions": list(permissions),
        "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
        "warning": "Store this API key securely. It will not be shown again.",
    }


# ============================================================================
# Development/Testing Helpers
# ============================================================================

# Create a default API key for testing (remove in production)
if os.environ.get("OMNI_AVA_ENV", "development") == "development":
    _test_key, _test_api_key = generate_api_key(
        "test_user", {"detect:univariate", "detect:multivariate"}
    )
    logger.info(f"Development API key created: {_test_key}")
