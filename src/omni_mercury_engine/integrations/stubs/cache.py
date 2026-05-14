"""
Mercury Agent

Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Cache service stub for testing and development.

Example:
    >>> cache = CacheStub()
    >>> await cache.set("user:123", {"name": "Alice"}, ttl=300)
    >>> data = await cache.get("user:123")
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Domain-specific TTL policy (seconds).  Used by get_domain_ttl()
# to apply appropriate cache lifetimes per data domain.
DOMAIN_TTL: dict[str, int] = {
    "environmental": 300,  # 5 min - data refreshes frequently
    "security": 60,  # 1 min - threat data must be fresh
    "climate": 3600,  # 1 hour - climate data is slow-changing
    "medical": 600,  # 10 min
    "space": 1800,  # 30 min
    "financial": 120,  # 2 min - markets move fast
    "industrial": 600,  # 10 min
    "default": 600,  # 10 min fallback
}


def get_domain_ttl(domain: str) -> int:
    """
    Return cache TTL in seconds for a given data domain.

    Args:
        domain: Data domain name (e.g., "environmental", "security").

    Returns:
        TTL in seconds.
    """
    return DOMAIN_TTL.get(domain, DOMAIN_TTL["default"])


@dataclass
class CacheEntry:
    """
    Cache entry with metadata.

    Attributes:
        key: Cache key.
        value: Cached value.
        created_at: Creation timestamp.
        expires_at: Expiration timestamp.
        ttl: Time-to-live in seconds.
        hits: Number of times accessed.
    """

    key: str
    value: Any
    created_at: float
    expires_at: float | None
    ttl: int | None
    hits: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "value": self.value,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "expires_at": (
                datetime.fromtimestamp(self.expires_at).isoformat() if self.expires_at else None
            ),
            "ttl": self.ttl,
            "hits": self.hits,
            "is_expired": self.is_expired,
        }


class CacheStub:
    """Stub implementation of cache service.

    Provides mock caching functionality for testing.
    Simulates Redis-like operations.

    Example:
        >>> cache = CacheStub()
        >>> await cache.set("key", "value", ttl=60)
        >>> value = await cache.get("key")
        >>> await cache.delete("key")
    """

    def __init__(
        self,
        seed: int | None = None,
        latency_ms: tuple[int, int] = (1, 10),
        failure_rate: float = 0.0,
        max_size: int = 10000,
    ):
        """
        Initialize cache stub.

        Args:
            seed: Random seed for reproducibility.
            latency_ms: Min/max simulated latency.
            failure_rate: Probability of simulated failure.
            max_size: Maximum number of entries.
        """
        self._rng = random.Random(seed)
        self._latency_ms = latency_ms
        self._failure_rate = failure_rate
        self._max_size = max_size

        # Cache storage
        self._cache: dict[str, CacheEntry] = {}

        # Metrics
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._deletes = 0

    async def _simulate_latency(self) -> None:
        """Simulate cache latency."""
        latency = self._rng.randint(*self._latency_ms) / 1000.0
        await asyncio.sleep(latency)

    def _maybe_fail(self) -> None:
        """Potentially raise exception to simulate failure."""
        if self._rng.random() < self._failure_rate:
            raise ConnectionError("Simulated cache failure")

    def _evict_if_needed(self) -> None:
        """Evict entries if cache is full."""
        if len(self._cache) >= self._max_size:
            # Remove expired entries first
            expired_keys = [k for k, v in self._cache.items() if v.is_expired]
            for key in expired_keys:
                del self._cache[key]

            # If still full, remove LRU entries
            if len(self._cache) >= self._max_size:
                # Sort by hits (least used first)
                sorted_entries = sorted(
                    self._cache.items(),
                    key=lambda x: (x[1].hits, x[1].created_at),
                )
                # Remove 10% of entries
                remove_count = max(1, len(self._cache) // 10)
                for key, _ in sorted_entries[:remove_count]:
                    del self._cache[key]

    async def get(self, key: str) -> Any | None:
        """
        Get value from cache.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found/expired.
        """
        await self._simulate_latency()
        self._maybe_fail()

        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None

        if entry.is_expired:
            del self._cache[key]
            self._misses += 1
            return None

        entry.hits += 1
        self._hits += 1
        return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time-to-live in seconds.
            nx: Only set if not exists.
            xx: Only set if exists.

        Returns:
            True if set, False if conditions not met.
        """
        await self._simulate_latency()
        self._maybe_fail()

        exists = key in self._cache and not self._cache[key].is_expired

        if nx and exists:
            return False
        if xx and not exists:
            return False

        self._evict_if_needed()

        now = time.time()
        # TTL of 0 means expire immediately
        expires_at = now + ttl if ttl and ttl > 0 else (now - 1 if ttl == 0 else None)
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=now,
            expires_at=expires_at,
            ttl=ttl,
        )
        self._cache[key] = entry
        self._sets += 1
        return True

    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key.

        Returns:
            True if key existed and was deleted.
        """
        await self._simulate_latency()
        self._maybe_fail()

        if key in self._cache:
            del self._cache[key]
            self._deletes += 1
            return True
        return False

    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key.

        Returns:
            True if key exists and not expired.
        """
        await self._simulate_latency()
        self._maybe_fail()

        entry = self._cache.get(key)
        if entry is None:
            return False
        if entry.is_expired:
            del self._cache[key]
            return False
        return True

    async def mget(self, keys: list[str]) -> dict[str, Any | None]:
        """
        Get multiple values.

        Args:
            keys: List of cache keys.

        Returns:
            Dictionary mapping keys to values (None for missing keys).
        """
        await self._simulate_latency()
        self._maybe_fail()

        results: dict[str, Any | None] = {}
        for key in keys:
            entry = self._cache.get(key)
            if entry and not entry.is_expired:
                entry.hits += 1
                self._hits += 1
                results[key] = entry.value
            else:
                self._misses += 1
                results[key] = None
        return results

    async def mset(self, mapping: dict[str, Any], ttl: int | None = None) -> bool:
        """
        Set multiple values.

        Args:
            mapping: Dictionary of key-value pairs.
            ttl: Time-to-live for all entries.

        Returns:
            True if successful.
        """
        await self._simulate_latency()
        self._maybe_fail()

        now = time.time()
        for key, value in mapping.items():
            self._evict_if_needed()
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                expires_at=now + ttl if ttl else None,
                ttl=ttl,
            )
            self._cache[key] = entry
            self._sets += 1
        return True

    async def incr(self, key: str, amount: int = 1) -> int:
        """
        Increment integer value.

        Args:
            key: Cache key.
            amount: Amount to increment.

        Returns:
            New value.
        """
        await self._simulate_latency()
        self._maybe_fail()

        entry = self._cache.get(key)
        if entry is None or entry.is_expired:
            await self.set(key, amount)
            return amount

        if not isinstance(entry.value, int):
            raise ValueError(f"Value for key '{key}' is not an integer")

        new_value = entry.value + amount
        entry.value = new_value
        return new_value

    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration time on key.

        Args:
            key: Cache key.
            ttl: Time-to-live in seconds.

        Returns:
            True if key exists and expiration was set.
        """
        await self._simulate_latency()
        self._maybe_fail()

        entry = self._cache.get(key)
        if entry is None or entry.is_expired:
            return False

        entry.expires_at = time.time() + ttl
        entry.ttl = ttl
        return True

    async def ttl(self, key: str) -> int | None:
        """
        Get remaining TTL for key.

        Args:
            key: Cache key.

        Returns:
            Remaining TTL in seconds, None if no TTL or key doesn't exist.
        """
        await self._simulate_latency()
        self._maybe_fail()

        entry = self._cache.get(key)
        if entry is None or entry.is_expired:
            return None

        if entry.expires_at is None:
            return None

        remaining = int(entry.expires_at - time.time())
        return max(0, remaining)

    async def keys(self, pattern: str = "*") -> list[str]:
        """Get keys matching pattern.

        Args:
            pattern: Glob-style pattern (* = any).

        Returns:
            List of matching keys.
        """
        await self._simulate_latency()
        self._maybe_fail()

        # Clean expired entries
        expired = [k for k, v in self._cache.items() if v.is_expired]
        for key in expired:
            del self._cache[key]

        return [k for k in self._cache if fnmatch.fnmatch(k, pattern)]

    async def flush(self) -> int:
        """
        Clear all entries from cache.

        Returns:
            Number of entries cleared.
        """
        await self._simulate_latency()
        self._maybe_fail()

        count = len(self._cache)
        self._cache.clear()
        return count

    async def clear(self) -> None:
        """Clear all entries from cache (alias for flush)."""
        await self.flush()

    async def mdelete(self, keys: list[str]) -> int:
        """
        Delete multiple keys from cache.

        Args:
            keys: List of cache keys to delete.

        Returns:
            Number of keys deleted.
        """
        await self._simulate_latency()
        self._maybe_fail()

        deleted = 0
        for key in keys:
            if key in self._cache:
                del self._cache[key]
                self._deletes += 1
                deleted += 1
        return deleted

    async def decr(self, key: str, amount: int = 1) -> int:
        """
        Decrement integer value.

        Args:
            key: Cache key.
            amount: Amount to decrement.

        Returns:
            New value.
        """
        return await self.incr(key, -amount)

    async def persist(self, key: str) -> bool:
        """
        Remove expiration from key.

        Args:
            key: Cache key.

        Returns:
            True if key exists and expiration was removed.
        """
        await self._simulate_latency()
        self._maybe_fail()

        entry = self._cache.get(key)
        if entry is None or entry.is_expired:
            return False

        entry.expires_at = None
        entry.ttl = None
        return True

    async def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        await self._simulate_latency()
        self._maybe_fail()

        return {
            "total_entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (
                self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0
            ),
            "sets": self._sets,
            "deletes": self._deletes,
            "max_size": self._max_size,
        }

    async def size(self) -> int:
        """
        Get number of entries in cache.

        Returns:
            Number of entries.
        """
        await self._simulate_latency()
        self._maybe_fail()

        # Clean expired entries first
        expired = [k for k, v in self._cache.items() if v.is_expired]
        for key in expired:
            del self._cache[key]

        return len(self._cache)

    def get_metrics(self) -> dict[str, Any]:
        """Get cache metrics."""
        total_requests = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total_requests if total_requests > 0 else 0,
            "sets": self._sets,
            "deletes": self._deletes,
            "size": len(self._cache),
            "max_size": self._max_size,
        }

    async def health_check(self) -> dict[str, Any]:
        """
        Check cache health.

        Returns:
            Health status.
        """
        try:
            await self._simulate_latency()
            return {
                "healthy": True,
                "size": len(self._cache),
                "memory_usage_pct": len(self._cache) / self._max_size * 100,
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
            }


class CacheBackend(Enum):
    """Supported cache backends."""

    REDIS = "redis"
    MEMORY = "memory"
    STUB = "stub"


class RedisCache:
    """Production-ready Redis cache client.

    Supports both redis-py (sync) and aioredis/redis.asyncio (async).
    Falls back to in-memory stub when Redis is unavailable.

    Example:
        >>> # Using Redis
        >>> cache = RedisCache(
        ...     host="localhost",
        ...     port=6379,
        ...     password=os.getenv("REDIS_PASSWORD")
        ... )
        >>> await cache.set("key", {"data": "value"}, ttl=300)
        >>> data = await cache.get("key")

        >>> # Using environment variables
        >>> cache = RedisCache.from_env()

        >>> # With automatic fallback to stub
        >>> cache = RedisCache(fallback_to_stub=True)
    """

    def __init__(
        self,
        host: str | None = None,
        port: int = 6379,
        password: str | None = None,
        db: int = 0,
        ssl: bool = False,
        timeout: int = 5,
        max_connections: int = 10,
        prefix: str = "mercury:",
        fallback_to_stub: bool = True,
        serializer: str = "json",
    ):
        """
        Initialize Redis cache.

        Args:
            host: Redis host (default: REDIS_HOST env or localhost).
            port: Redis port (default: 6379).
            password: Redis password (default: REDIS_PASSWORD env).
            db: Redis database number.
            ssl: Use SSL connection.
            timeout: Connection timeout in seconds.
            max_connections: Maximum pool connections.
            prefix: Key prefix for namespacing.
            fallback_to_stub: Fall back to in-memory stub on connection failure.
            serializer: Serialization format. Only "json" is supported;
                callers that need to cache a non-JSON-serializable value
                must convert it before set(). The legacy "pickle"
                serializer has been removed -- pickle has no role in
                the Mercury Agent runtime outside the one-shot
                ``tools/migrate_pkl.py`` migration tool.
        """
        if serializer != "json":
            raise ValueError(
                f"RedisCache: serializer={serializer!r} is no longer supported. "
                "Use serializer='json' and convert non-JSON values before caching. "
                "Pickle deserialisation has been removed from the Mercury Agent "
                "runtime (see tools/migrate_pkl.py for offline conversion)."
            )
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port
        self.password = password or os.getenv("REDIS_PASSWORD")
        self.db = db
        self.ssl = ssl
        self.timeout = timeout
        self.max_connections = max_connections
        self.prefix = prefix
        self.fallback_to_stub = fallback_to_stub
        self.serializer = serializer

        self._client: Any = None
        self._stub = CacheStub()
        self._connected = False
        self._connection_error: str | None = None

        # Metrics
        self._call_count = 0
        self._errors = 0
        self._fallback_count = 0

    @classmethod
    def from_env(cls) -> RedisCache:
        """
        Create Redis cache from environment variables.

        Environment variables:
            REDIS_HOST: Redis host (default: localhost)
            REDIS_PORT: Redis port (default: 6379)
            REDIS_PASSWORD: Redis password
            REDIS_DB: Redis database number (default: 0)
            REDIS_SSL: Use SSL (default: false)
            REDIS_PREFIX: Key prefix (default: mercury:)
        """
        return cls(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD"),
            db=int(os.getenv("REDIS_DB", "0")),
            ssl=os.getenv("REDIS_SSL", "").lower() in ("true", "1", "yes"),
            prefix=os.getenv("REDIS_PREFIX", "mercury:"),
        )

    async def _ensure_connected(self) -> bool:
        """Ensure Redis connection is established."""
        if self._connected and self._client:
            return True

        try:
            import redis.asyncio as aioredis

            self._client = aioredis.Redis(
                host=self.host,  # type: ignore[arg-type, unused-ignore]
                port=self.port,
                password=self.password,
                db=self.db,
                ssl=self.ssl,
                socket_timeout=self.timeout,
                socket_connect_timeout=self.timeout,
                decode_responses=True,
            )

            # Test connection
            await self._client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.host}:{self.port}")
            return True

        except ImportError:
            self._connection_error = "redis package not installed (pip install redis)"
            logger.warning(self._connection_error)
            return False

        except Exception as e:
            self._connection_error = str(e)
            logger.warning(f"Redis connection failed: {e}")
            return False

    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.prefix}{key}"

    def _serialize(self, value: Any) -> str:
        """Serialize value for storage as JSON.

        The pickle serializer was removed; callers that need to cache
        non-JSON-serialisable values must convert them first.
        """
        return json.dumps(value)

    def _deserialize(self, data: str | None) -> Any:
        """Deserialize JSON value from storage."""
        if data is None:
            return None
        return json.loads(data)

    async def get(self, key: str) -> Any | None:
        """
        Get value from cache.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found.

        Raises:
            json.JSONDecodeError: The stored payload is not valid JSON.
                Corrupted Redis data is a contract violation, not a
                transient connectivity error -- swallowing it into a
                stub-fallback would silently hide cache poisoning or a
                pickle-era payload that has not been migrated. The
                error surfaces so the operator can rebuild the affected
                key (or rotate ``MERCURY_CACHE_*`` if cross-process
                contamination is suspected).
        """
        self._call_count += 1

        if not await self._ensure_connected():
            if self.fallback_to_stub:
                self._fallback_count += 1
                return await self._stub.get(key)
            return None

        try:
            data = await self._client.get(self._make_key(key))
        except Exception as e:
            # Connectivity / Redis-side failure: stay quiet and fall
            # back. The JSON-deserialise step is intentionally NOT
            # inside this try block -- a malformed payload is a
            # contract violation we want surfaced, not a transient
            # error worth masking.
            self._errors += 1
            logger.warning(f"Redis get error: {e}")
            if self.fallback_to_stub:
                self._fallback_count += 1
                return await self._stub.get(key)
            return None
        return self._deserialize(data)

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time-to-live in seconds.
            nx: Only set if not exists.
            xx: Only set if exists.

        Returns:
            True if set successfully.

        Raises:
            TypeError: ``value`` is not JSON-serialisable. The cache
                contract is JSON-only, and that contract MUST hold
                regardless of whether the Redis client is reachable.
                Validating up front (before any fallback) prevents the
                in-memory stub from silently accepting a value that
                Redis would have refused -- otherwise a developer
                running offline could write code that worked in the
                stub and crashed in production the first time the
                Redis path was exercised.
        """
        self._call_count += 1

        # Validate JSON-serialisability BEFORE the fallback decision.
        # The same contract applies on every code path; ``_serialize``
        # raises TypeError for sets / bytes / custom objects, and that
        # error must surface to the caller rather than be papered over
        # by storing the raw object in the stub.
        serialized = self._serialize(value)

        if not await self._ensure_connected():
            if self.fallback_to_stub:
                self._fallback_count += 1
                return await self._stub.set(key, value, ttl, nx, xx)
            return False

        try:
            full_key = self._make_key(key)

            # Build set options
            set_kwargs: dict[str, Any] = {}
            if ttl:
                set_kwargs["ex"] = ttl
            if nx:
                set_kwargs["nx"] = True
            if xx:
                set_kwargs["xx"] = True

            result = await self._client.set(full_key, serialized, **set_kwargs)
            return result is not None and result is not False
        except Exception as e:
            self._errors += 1
            logger.warning(f"Redis set error: {e}")
            if self.fallback_to_stub:
                self._fallback_count += 1
                return await self._stub.set(key, value, ttl, nx, xx)
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key.

        Returns:
            True if key existed and was deleted.
        """
        self._call_count += 1

        if not await self._ensure_connected():
            if self.fallback_to_stub:
                self._fallback_count += 1
                return await self._stub.delete(key)
            return False

        try:
            result = await self._client.delete(self._make_key(key))
            return bool(result > 0)
        except Exception as e:
            self._errors += 1
            logger.warning(f"Redis delete error: {e}")
            if self.fallback_to_stub:
                self._fallback_count += 1
                return await self._stub.delete(key)
            return False

    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key.

        Returns:
            True if key exists.
        """
        self._call_count += 1

        if not await self._ensure_connected():
            if self.fallback_to_stub:
                self._fallback_count += 1
                return await self._stub.exists(key)
            return False

        try:
            result = await self._client.exists(self._make_key(key))
            return bool(result > 0)
        except Exception as e:
            self._errors += 1
            logger.warning(f"Redis exists error: {e}")
            if self.fallback_to_stub:
                self._fallback_count += 1
                return await self._stub.exists(key)
            return False

    async def mget(self, keys: list[str]) -> dict[str, Any | None]:
        """
        Get multiple values.

        Args:
            keys: List of cache keys.

        Returns:
            Dictionary mapping keys to values.

        Raises:
            json.JSONDecodeError: Any stored payload is not valid JSON.
                Matches the ``get`` contract so bulk reads cannot
                silently swallow a corrupted entry that the single-key
                path surfaces loudly.
        """
        self._call_count += 1

        if not await self._ensure_connected():
            if self.fallback_to_stub:
                self._fallback_count += 1
                return await self._stub.mget(keys)
            return dict.fromkeys(keys)

        try:
            full_keys = [self._make_key(k) for k in keys]
            values = await self._client.mget(full_keys)
        except Exception as e:
            # Connectivity / Redis-side failure: stay quiet and fall
            # back, mirroring ``get``. JSON deserialisation runs after
            # this block so a corrupted payload (``JSONDecodeError``)
            # surfaces as the same contract violation it does for
            # ``get`` instead of being silently masked by the stub.
            self._errors += 1
            logger.warning(f"Redis mget error: {e}")
            if self.fallback_to_stub:
                self._fallback_count += 1
                return await self._stub.mget(keys)
            return dict.fromkeys(keys)
        return {keys[i]: self._deserialize(v) for i, v in enumerate(values)}

    async def mset(self, mapping: dict[str, Any], ttl: int | None = None) -> bool:
        """
        Set multiple values.

        Args:
            mapping: Dictionary of key-value pairs.
            ttl: Time-to-live for all entries.

        Returns:
            True if successful.

        Raises:
            TypeError: any ``mapping`` value is not JSON-serialisable.
                Validated up front so the contract holds whether the
                call ends up in Redis or in the in-memory stub
                fallback. See ``RedisCache.set`` for the rationale.
        """
        self._call_count += 1

        # Serialise all values up front so the JSON contract is
        # enforced regardless of whether the call ends up in Redis or
        # in the in-memory stub fallback. A single TypeError surfaces
        # the offending value to the caller; we never silently route
        # a non-JSON value to the stub.
        serialized_mapping = {self._make_key(k): self._serialize(v) for k, v in mapping.items()}

        if not await self._ensure_connected():
            if self.fallback_to_stub:
                self._fallback_count += 1
                return await self._stub.mset(mapping, ttl)
            return False

        try:
            # Use pipeline for efficiency
            async with self._client.pipeline(transaction=True) as pipe:
                await pipe.mset(serialized_mapping)
                if ttl:
                    for key in serialized_mapping:
                        await pipe.expire(key, ttl)
                await pipe.execute()
            return True
        except Exception as e:
            self._errors += 1
            logger.warning(f"Redis mset error: {e}")
            if self.fallback_to_stub:
                self._fallback_count += 1
                return await self._stub.mset(mapping, ttl)
            return False

    async def incr(self, key: str, amount: int = 1) -> int:
        """
        Increment integer value.

        Args:
            key: Cache key.
            amount: Amount to increment.

        Returns:
            New value.
        """
        self._call_count += 1

        if not await self._ensure_connected():
            if self.fallback_to_stub:
                self._fallback_count += 1
                return await self._stub.incr(key, amount)
            return 0

        try:
            return int(await self._client.incrby(self._make_key(key), amount))
        except Exception as e:
            self._errors += 1
            logger.warning(f"Redis incr error: {e}")
            if self.fallback_to_stub:
                self._fallback_count += 1
                return await self._stub.incr(key, amount)
            return 0

    async def decr(self, key: str, amount: int = 1) -> int:
        """
        Decrement integer value.

        Args:
            key: Cache key.
            amount: Amount to decrement.

        Returns:
            New value.
        """
        return await self.incr(key, -amount)

    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration time on key.

        Args:
            key: Cache key.
            ttl: Time-to-live in seconds.

        Returns:
            True if key exists and expiration was set.
        """
        self._call_count += 1

        if not await self._ensure_connected():
            if self.fallback_to_stub:
                self._fallback_count += 1
                return await self._stub.expire(key, ttl)
            return False

        try:
            return bool(await self._client.expire(self._make_key(key), ttl))
        except Exception as e:
            self._errors += 1
            logger.warning(f"Redis expire error: {e}")
            if self.fallback_to_stub:
                self._fallback_count += 1
                return await self._stub.expire(key, ttl)
            return False

    async def ttl(self, key: str) -> int | None:
        """
        Get remaining TTL for key.

        Args:
            key: Cache key.

        Returns:
            Remaining TTL in seconds, None if no TTL or key doesn't exist.
        """
        self._call_count += 1

        if not await self._ensure_connected():
            if self.fallback_to_stub:
                self._fallback_count += 1
                return await self._stub.ttl(key)
            return None

        try:
            result = await self._client.ttl(self._make_key(key))
            return result if result > 0 else None
        except Exception as e:
            self._errors += 1
            logger.warning(f"Redis ttl error: {e}")
            if self.fallback_to_stub:
                self._fallback_count += 1
                return await self._stub.ttl(key)
            return None

    async def keys(self, pattern: str = "*") -> list[str]:
        """
        Get keys matching pattern.

        Args:
            pattern: Glob-style pattern.

        Returns:
            List of matching keys (without prefix).
        """
        self._call_count += 1

        if not await self._ensure_connected():
            if self.fallback_to_stub:
                self._fallback_count += 1
                return await self._stub.keys(pattern)
            return []

        try:
            full_pattern = self._make_key(pattern)
            matched_keys = await self._client.keys(full_pattern)
            # Remove prefix from keys
            prefix_len = len(self.prefix)
            return [k[prefix_len:] if k.startswith(self.prefix) else k for k in matched_keys]
        except Exception as e:
            self._errors += 1
            logger.warning(f"Redis keys error: {e}")
            if self.fallback_to_stub:
                self._fallback_count += 1
                return await self._stub.keys(pattern)
            return []

    async def flush(self) -> int:
        """
        Clear all entries with our prefix.

        Returns:
            Number of entries cleared.
        """
        self._call_count += 1

        if not await self._ensure_connected():
            if self.fallback_to_stub:
                self._fallback_count += 1
                return await self._stub.flush()
            return 0

        try:
            keys = await self._client.keys(f"{self.prefix}*")
            if keys:
                return int(await self._client.delete(*keys))
            return 0
        except Exception as e:
            self._errors += 1
            logger.warning(f"Redis flush error: {e}")
            if self.fallback_to_stub:
                self._fallback_count += 1
                return await self._stub.flush()
            return 0

    async def clear(self) -> None:
        """Clear all entries (alias for flush)."""
        await self.flush()

    async def health_check(self) -> dict[str, Any]:
        """
        Check cache health.

        Returns:
            Health status.
        """
        if not await self._ensure_connected():
            return {
                "healthy": False,
                "backend": "stub" if self.fallback_to_stub else "none",
                "error": self._connection_error,
            }

        try:
            await self._client.ping()
            info = await self._client.info("memory")
            return {
                "healthy": True,
                "backend": "redis",
                "host": self.host,
                "port": self.port,
                "used_memory": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", "unknown"),
            }
        except Exception as e:
            return {
                "healthy": False,
                "backend": "redis",
                "error": str(e),
            }

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            try:
                await self._client.close()
                self._connected = False
                logger.info("Redis connection closed")
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")

    def get_metrics(self) -> dict[str, Any]:
        """Get cache metrics."""
        return {
            "backend": "redis" if self._connected else "stub",
            "host": self.host,
            "port": self.port,
            "connected": self._connected,
            "call_count": self._call_count,
            "errors": self._errors,
            "fallback_count": self._fallback_count,
            "error_rate": self._errors / self._call_count if self._call_count > 0 else 0,
        }


# Factory function for cache creation
def create_cache(
    backend: str = "stub",
    host: str | None = None,
    port: int = 6379,
    password: str | None = None,
    **kwargs: Any,
) -> RedisCache | CacheStub:
    """Create cache with appropriate backend.

    Args:
        backend: Cache backend ("redis", "memory", "stub").
        host: Redis host.
        port: Redis port.
        password: Redis password.
        **kwargs: Additional backend-specific options.

    Returns:
        Configured cache instance.

    Example:
        >>> # For testing
        >>> cache = create_cache(backend="stub")

        >>> # For production with Redis
        >>> cache = create_cache(
        ...     backend="redis",
        ...     host="localhost",
        ...     port=6379
        ... )

        >>> # Auto-detect from environment
        >>> cache = create_cache(backend="redis")  # Uses REDIS_* env vars
    """
    if backend.lower() in ("memory", "stub"):
        return CacheStub(**kwargs)

    if backend.lower() == "redis":
        return RedisCache(
            host=host,
            port=port,
            password=password,
            **kwargs,
        )

    logger.warning(f"Unknown cache backend: {backend}, using stub")
    return CacheStub()
