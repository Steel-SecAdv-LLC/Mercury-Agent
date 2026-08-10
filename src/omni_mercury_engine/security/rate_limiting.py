# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unified Rate Limiting.

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
    """Protocol for rate limit storage backends (e.g., Redis).

    Token balances are **fractional** (``float``).  A token bucket refills
    continuously at ``requests_per_minute / 60`` tokens per second, so a
    backend that rounds the balance to whole tokens on every store loses the
    sub-token remainder each time the bucket is touched.  At any polling rate
    faster than one request per refill period, that remainder is the entire
    refill -- see :meth:`RateLimiter._check_token_bucket` for the arithmetic
    this protocol exists to keep exact.
    """

    def get(self, key: str) -> tuple[float, float] | None:
        """Get bucket state: ``(last_update_time, tokens)``."""
        ...

    def set(self, key: str, last_time: float, tokens: float, ttl: int) -> None:
        """Set bucket state with TTL."""
        ...

    def delete(self, key: str) -> None:
        """Delete bucket state."""
        ...


class InMemoryBackend:
    """Thread-safe in-memory rate limit backend.

    Implements the optional atomic :meth:`consume_token` extension so the
    process-local path spends tokens under exactly the same arithmetic as
    :class:`~omni_mercury_engine.api.rate_limit_store.SqliteRateLimitBackend`.
    Both backends therefore deliver the configured rate; swapping one for the
    other changes durability and cross-process sharing, never the rate.
    """

    def __init__(self, max_entries: int = 10000, ttl_seconds: int = 300) -> None:
        """Initialize the instance."""
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.RLock()
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._last_cleanup = time.time()

    def get(self, key: str) -> tuple[float, float] | None:
        """Get."""
        with self._lock:
            return self._buckets.get(key)

    def set(self, key: str, last_time: float, tokens: float, ttl: int) -> None:
        """Set."""
        with self._lock:
            self._cleanup_if_needed()
            self._buckets[key] = (last_time, float(tokens))

    def delete(self, key: str) -> None:
        """Delete."""
        with self._lock:
            self._buckets.pop(key, None)

    def consume_token(
        self,
        key: str,
        *,
        refill_rate: float,
        burst: int,
        now: float,
    ) -> tuple[bool, float]:
        """Atomically refill ``key``'s bucket and spend one token if available.

        Mirrors ``SqliteRateLimitBackend.consume_token`` exactly -- fractional
        refill, capacity clamp, spend-on-success -- with the instance ``RLock``
        standing in for the SQLite ``BEGIN IMMEDIATE`` transaction.

        Args:
            key: Bucket identifier.
            refill_rate: Tokens added per second.
            burst: Bucket capacity.
            now: Current UNIX time (injected for deterministic tests).

        Returns:
            ``(allowed, tokens_remaining)`` -- ``tokens_remaining`` is the
            balance *after* the spend (or the unspendable balance on deny).
        """
        with self._lock:
            self._cleanup_if_needed()
            state = self._buckets.get(key)
            if state is None:
                tokens = float(burst)
            else:
                last_time, stored = state
                elapsed = max(0.0, now - last_time)
                tokens = min(float(burst), stored + elapsed * refill_rate)
            allowed = tokens >= 1.0
            if allowed:
                tokens -= 1.0
            self._buckets[key] = (now, tokens)
        return allowed, tokens

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
    """Unified rate limiter with token bucket and sliding window algorithms.

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
        """Initialize unified rate limiter.

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
        """Check if request is allowed and update rate limit state.

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
        """Token bucket algorithm with burst support.

        When the backend exposes an atomic ``consume_token`` (both shipped
        backends do), the whole refill-and-spend happens inside the backend's
        own critical section — the only race-free option once buckets are
        shared across worker processes. The plain ``get``/``set`` path below
        is the compatibility route for third-party backends that implement
        only the two-method protocol.

        Both routes refill **fractionally**: ``tokens += elapsed *
        requests_per_minute / 60`` with no truncation, and the balance is
        stored as a float. Truncating the refill to whole tokens while also
        advancing ``last_time`` to ``now`` — the behaviour this method used
        to have on the ``get``/``set`` route — discards the elapsed time that
        produced the truncated remainder. A client polling faster than
        ``requests_per_minute / 60`` then never accumulates a whole token
        between calls and is starved permanently after its initial burst,
        delivering a small fraction of the configured rate rather than the
        configured rate.
        """
        now = time.time()
        refill_rate = self.requests_per_minute / 60.0

        consume = getattr(self.backend, "consume_token", None)
        if callable(consume):
            allowed, remaining = consume(
                identifier,
                refill_rate=refill_rate,
                burst=self.burst_size,
                now=now,
            )
            if allowed:
                return RateLimitInfo(
                    allowed=True,
                    limit=self.requests_per_minute,
                    remaining=int(remaining),
                    reset_at=int(now) + 60,
                )
            return RateLimitInfo(
                allowed=False,
                limit=self.requests_per_minute,
                remaining=0,
                reset_at=int(now) + 60,
                retry_after=self._retry_after(remaining, refill_rate),
            )

        # Compatibility route: refill-then-spend over a plain get/set backend.
        bucket = self.backend.get(identifier)
        if bucket is not None:
            last_time, stored = bucket
            # max(0.0, ...) guards a backwards clock jump (NTP step, or a
            # backend shared between hosts): elapsed time can never be
            # negative, and a negative refill would drain the bucket.
            elapsed = max(0.0, now - last_time)
            tokens = min(float(self.burst_size), float(stored) + elapsed * refill_rate)
        else:
            tokens = float(self.burst_size)

        if tokens >= 1.0:
            tokens -= 1.0
            self.backend.set(identifier, now, tokens, self.DEFAULT_TTL_SECONDS)
            return RateLimitInfo(
                allowed=True,
                limit=self.requests_per_minute,
                remaining=int(tokens),
                reset_at=int(now) + 60,
            )

        # Denied: still write back the refilled (fractional) balance and the
        # new timestamp. Storing the fraction is what makes advancing
        # ``last_time`` lossless — the elapsed time is not discarded, it has
        # been converted into the stored remainder.
        self.backend.set(identifier, now, tokens, self.DEFAULT_TTL_SECONDS)
        return RateLimitInfo(
            allowed=False,
            limit=self.requests_per_minute,
            remaining=0,
            reset_at=int(now) + 60,
            retry_after=self._retry_after(tokens, refill_rate),
        )

    @staticmethod
    def _retry_after(tokens: float, refill_rate: float) -> float:
        """Seconds until the bucket holds one whole token again.

        ``tokens`` is the (sub-1.0) balance left after a denial, so the wait
        is the time to refill the shortfall — not a flat ``60 /
        requests_per_minute``, which over-reports whenever the bucket is
        already part-way to the next token.
        """
        if refill_rate <= 0:
            return float("inf")
        return max(0.0, (1.0 - tokens) / refill_rate)

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
        """Simple check if request is allowed (backward compatible).

        Args:
            identifier: Unique identifier for the client

        Returns:
            True if allowed, False if rate limited
        """
        return self.check(identifier).allowed

    def reset(self, identifier: str) -> None:
        """Reset rate limit for identifier.

        Args:
            identifier: Unique identifier to reset
        """
        self.backend.delete(identifier)
        with self._sliding_lock:
            self._sliding_requests.pop(identifier, None)

    def get_status(self, identifier: str) -> RateLimitInfo:
        """Get current rate limit status without consuming a token.

        Args:
            identifier: Unique identifier to check

        Returns:
            Current rate limit status. Read-only: no token is consumed and no
            bucket state is written, so the projection here uses exactly the
            same fractional refill arithmetic as
            :meth:`_check_token_bucket`. Truncating the refill (as this used
            to) would report a client as exhausted while ``check`` would have
            granted it, and vice versa.
        """
        now = time.time()

        if self.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            bucket = self.backend.get(identifier)
            if bucket is not None:
                last_time, stored = bucket
                elapsed = max(0.0, now - last_time)
                refill_rate = self.requests_per_minute / 60.0
                tokens = min(float(self.burst_size), float(stored) + elapsed * refill_rate)
            else:
                tokens = float(self.burst_size)

            return RateLimitInfo(
                allowed=tokens >= 1.0,
                limit=self.requests_per_minute,
                remaining=int(tokens),
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


# Global singleton for convenience. Construction is lock-guarded
# (double-checked) so two threads racing the first call cannot each build a
# limiter and leave callers split across two independent token pools.
_default_limiter: RateLimiter | None = None
_default_limiter_lock = threading.Lock()


def get_rate_limiter(
    requests_per_minute: int = RateLimiter.DEFAULT_REQUESTS_PER_MINUTE,
    burst_size: int | None = None,
) -> RateLimiter:
    """Get or create the default rate limiter singleton (thread-safe).

    Args:
        requests_per_minute: Requests per minute (only used on first call)
        burst_size: Burst size (only used on first call)

    Returns:
        Singleton RateLimiter instance
    """
    global _default_limiter
    if _default_limiter is None:
        with _default_limiter_lock:
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
