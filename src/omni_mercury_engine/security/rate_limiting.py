"""
Mercury Agent - Unified Rate Limiting

Copyright (C) 2025 Steel Security Advisors LLC

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

Unified rate limiting module consolidating:
- Token bucket algorithm with burst support
- Sliding window rate limiting
- Memory management with TTL cleanup
- Thread-safe operations
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

logger = logging.getLogger(__name__)


class RateLimitAlgorithm(Enum):
    """Rate limiting algorithm selection."""

    TOKEN_BUCKET = "token_bucket"  # noqa: S105 - not a password
    SLIDING_WINDOW = "sliding_window"


@dataclass
class RateLimitInfo:
    """Rate limit status information for response headers."""

    allowed: bool
    limit: int
    remaining: int
    reset_at: int
    retry_after: float | None = None

    def to_headers(self) -> dict[str, str]:
        """Convert to HTTP response headers."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(self.reset_at),
        }
        if self.retry_after is not None:
            headers["Retry-After"] = str(int(self.retry_after))
        return headers


class RateLimitBackend(Protocol):
    """Protocol for rate limit storage backends (e.g., Redis)."""

    def get(self, key: str) -> tuple[float, int] | None:
        """Get bucket state: (last_update_time, tokens)."""
        ...

    def set(self, key: str, last_time: float, tokens: int, ttl: int) -> None:
        """Set bucket state with TTL."""
        ...

    def delete(self, key: str) -> None:
        """Delete bucket state."""
        ...


class InMemoryBackend:
    """Thread-safe in-memory rate limit backend."""

    def __init__(self, max_entries: int = 10000, ttl_seconds: int = 300) -> None:
        self._buckets: dict[str, tuple[float, int]] = {}
        self._lock = threading.RLock()
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._last_cleanup = time.time()

    def get(self, key: str) -> tuple[float, int] | None:
        """Get."""
        with self._lock:
            return self._buckets.get(key)

    def set(self, key: str, last_time: float, tokens: int, ttl: int) -> None:
        """Set."""
        with self._lock:
            self._cleanup_if_needed()
            self._buckets[key] = (last_time, tokens)

    def delete(self, key: str) -> None:
        """Delete."""
        with self._lock:
            self._buckets.pop(key, None)

    def _cleanup_if_needed(self) -> None:
        """Remove stale entries to prevent memory exhaustion."""
        now = time.time()
        if now - self._last_cleanup < 60:
            return

        self._last_cleanup = now
        stale_threshold = now - self._ttl_seconds

        # Remove stale entries
        stale_keys = [
            key for key, (last_time, _) in self._buckets.items() if last_time < stale_threshold
        ]
        for key in stale_keys:
            del self._buckets[key]

        # If still over limit, remove oldest entries
        if len(self._buckets) > self._max_entries:
            sorted_entries = sorted(self._buckets.items(), key=lambda x: x[1][0])
            excess = len(self._buckets) - self._max_entries
            for key, _ in sorted_entries[:excess]:
                del self._buckets[key]

        if stale_keys:
            logger.debug(f"Rate limiter cleanup: removed {len(stale_keys)} stale entries")


class RateLimiter:
    """
    Unified rate limiter with token bucket and sliding window algorithms.

    Features:
    - Token bucket algorithm with configurable burst
    - Sliding window algorithm for strict rate limiting
    - Memory management with automatic cleanup
    - Thread-safe operations
    - Pluggable backends (in-memory, Redis, etc.)

    Example:
        >>> limiter = RateLimiter(requests_per_minute=100, burst_size=20)
        >>> info = limiter.check("user:123")
        >>> if info.allowed:
        ...     # Process request
        ...     pass
        >>> else:
        ...     # Return 429 with headers
        ...     headers = info.to_headers()
    """

    # Default configuration
    DEFAULT_REQUESTS_PER_MINUTE = 100
    DEFAULT_BURST_SIZE = 20
    DEFAULT_MAX_ENTRIES = 10000
    DEFAULT_TTL_SECONDS = 300

    def __init__(
        self,
        requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
        burst_size: int | None = None,
        algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET,
        backend: RateLimitBackend | None = None,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        # Legacy parameter aliases for backward compatibility
        max_requests: int | None = None,
        window_seconds: int | None = None,
    ) -> None:
        """
        Initialize unified rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
            burst_size: Maximum burst size (defaults to requests_per_minute // 5)
            algorithm: Rate limiting algorithm to use
            backend: Storage backend (defaults to in-memory)
            max_entries: Maximum number of tracked clients
            ttl_seconds: TTL for bucket entries in seconds

        Legacy Args (for backward compatibility):
            max_requests: Alias for requests_per_minute
            window_seconds: Ignored (always uses per-minute rate)
        """
        # Handle legacy parameters
        is_legacy_mode = max_requests is not None
        if is_legacy_mode and max_requests is not None:
            requests_per_minute = max_requests
        _ = window_seconds  # Ignored, kept for backward compatibility

        self.requests_per_minute = requests_per_minute
        # In legacy mode, burst_size defaults to max_requests for backward compatibility
        # In new mode, burst_size defaults to requests_per_minute // 5
        if burst_size is not None:
            self.burst_size = burst_size
        elif is_legacy_mode:
            self.burst_size = requests_per_minute  # Legacy: burst = max_requests
        else:
            self.burst_size = max(1, requests_per_minute // 5)  # New: burst = rpm / 5
        self.algorithm = algorithm
        self.backend = backend or InMemoryBackend(max_entries=max_entries, ttl_seconds=ttl_seconds)

        # For sliding window algorithm
        self._sliding_requests: dict[str, list[float]] = {}
        self._sliding_lock = threading.RLock()

        logger.info(
            f"RateLimiter initialized: {requests_per_minute}/min, "
            f"burst={self.burst_size}, algorithm={algorithm.value}"
        )

    def check(self, identifier: str) -> RateLimitInfo:
        """
        Check if request is allowed and update rate limit state.

        Args:
            identifier: Unique identifier for the client (IP, user ID, API key)

        Returns:
            RateLimitInfo with allowed status and header information
        """
        if self.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return self._check_token_bucket(identifier)
        else:
            return self._check_sliding_window(identifier)

    def _check_token_bucket(self, identifier: str) -> RateLimitInfo:
        """Token bucket algorithm with burst support."""
        now = time.time()

        # Get current bucket state
        bucket = self.backend.get(identifier)
        if bucket is not None:
            last_time, tokens = bucket
            # Refill tokens based on elapsed time
            elapsed = now - last_time
            refill_rate = self.requests_per_minute / 60.0
            new_tokens = int(elapsed * refill_rate)
            tokens = min(self.burst_size, tokens + new_tokens)
        else:
            last_time = now
            tokens = self.burst_size

        # Check if request is allowed
        if tokens > 0:
            self.backend.set(identifier, now, tokens - 1, self.DEFAULT_TTL_SECONDS)
            return RateLimitInfo(
                allowed=True,
                limit=self.requests_per_minute,
                remaining=tokens - 1,
                reset_at=int(now) + 60,
            )
        else:
            # Calculate retry time
            time_to_next_token = 60.0 / self.requests_per_minute
            self.backend.set(identifier, now, 0, self.DEFAULT_TTL_SECONDS)
            return RateLimitInfo(
                allowed=False,
                limit=self.requests_per_minute,
                remaining=0,
                reset_at=int(now) + 60,
                retry_after=time_to_next_token,
            )

    def _check_sliding_window(self, identifier: str) -> RateLimitInfo:
        """Sliding window algorithm for strict rate limiting."""
        now = time.time()
        window_start = now - 60.0  # 60-second window

        with self._sliding_lock:
            # Get and clean up old requests
            requests = self._sliding_requests.get(identifier, [])
            requests = [req_time for req_time in requests if req_time > window_start]

            if len(requests) >= self.requests_per_minute:
                # Rate limit exceeded
                oldest = min(requests) if requests else now
                retry_after = oldest + 60.0 - now
                self._sliding_requests[identifier] = requests
                return RateLimitInfo(
                    allowed=False,
                    limit=self.requests_per_minute,
                    remaining=0,
                    reset_at=int(oldest + 60),
                    retry_after=max(0, retry_after),
                )

            # Allow request
            requests.append(now)
            self._sliding_requests[identifier] = requests
            return RateLimitInfo(
                allowed=True,
                limit=self.requests_per_minute,
                remaining=self.requests_per_minute - len(requests),
                reset_at=int(now) + 60,
            )

    def is_allowed(self, identifier: str) -> bool:
        """
        Simple check if request is allowed (backward compatible).

        Args:
            identifier: Unique identifier for the client

        Returns:
            True if allowed, False if rate limited
        """
        return self.check(identifier).allowed

    def reset(self, identifier: str) -> None:
        """
        Reset rate limit for identifier.

        Args:
            identifier: Unique identifier to reset
        """
        self.backend.delete(identifier)
        with self._sliding_lock:
            self._sliding_requests.pop(identifier, None)

    def get_status(self, identifier: str) -> RateLimitInfo:
        """
        Get current rate limit status without consuming a token.

        Args:
            identifier: Unique identifier to check

        Returns:
            Current rate limit status
        """
        now = time.time()

        if self.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            bucket = self.backend.get(identifier)
            if bucket is not None:
                last_time, tokens = bucket
                elapsed = now - last_time
                refill_rate = self.requests_per_minute / 60.0
                new_tokens = int(elapsed * refill_rate)
                tokens = min(self.burst_size, tokens + new_tokens)
            else:
                tokens = self.burst_size

            return RateLimitInfo(
                allowed=tokens > 0,
                limit=self.requests_per_minute,
                remaining=tokens,
                reset_at=int(now) + 60,
            )
        else:
            with self._sliding_lock:
                window_start = now - 60.0
                requests = self._sliding_requests.get(identifier, [])
                requests = [req_time for req_time in requests if req_time > window_start]
                remaining = self.requests_per_minute - len(requests)

                return RateLimitInfo(
                    allowed=remaining > 0,
                    limit=self.requests_per_minute,
                    remaining=max(0, remaining),
                    reset_at=int(now) + 60,
                )


# Global singleton for convenience
_default_limiter: RateLimiter | None = None


def get_rate_limiter(
    requests_per_minute: int = RateLimiter.DEFAULT_REQUESTS_PER_MINUTE,
    burst_size: int | None = None,
) -> RateLimiter:
    """
    Get or create the default rate limiter singleton.

    Args:
        requests_per_minute: Requests per minute (only used on first call)
        burst_size: Burst size (only used on first call)

    Returns:
        Singleton RateLimiter instance
    """
    global _default_limiter
    if _default_limiter is None:
        _default_limiter = RateLimiter(
            requests_per_minute=requests_per_minute,
            burst_size=burst_size,
        )
    return _default_limiter


def reset_default_limiter() -> None:
    """Reset the default rate limiter singleton (useful for testing)."""
    global _default_limiter
    _default_limiter = None
