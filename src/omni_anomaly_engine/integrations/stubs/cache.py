"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

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

import asyncio
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class CacheEntry:
    """Cache entry with metadata.

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
        """Initialize cache stub.

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
        """Get value from cache.

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
        """Set value in cache.

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

    async def delete(self, key: str) -> bool:
        """Delete key from cache.

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
        """Check if key exists in cache.

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

    async def mget(self, keys: list[str]) -> list[Any | None]:
        """Get multiple values.

        Args:
            keys: List of cache keys.

        Returns:
            List of values (None for missing keys).
        """
        await self._simulate_latency()
        self._maybe_fail()

        results = []
        for key in keys:
            entry = self._cache.get(key)
            if entry and not entry.is_expired:
                entry.hits += 1
                self._hits += 1
                results.append(entry.value)
            else:
                self._misses += 1
                results.append(None)
        return results

    async def mset(self, mapping: dict[str, Any], ttl: int | None = None) -> bool:
        """Set multiple values.

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
        """Increment integer value.

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
        """Set expiration time on key.

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

    async def ttl(self, key: str) -> int:
        """Get remaining TTL for key.

        Args:
            key: Cache key.

        Returns:
            Remaining TTL in seconds, -1 if no TTL, -2 if not exists.
        """
        await self._simulate_latency()
        self._maybe_fail()

        entry = self._cache.get(key)
        if entry is None or entry.is_expired:
            return -2

        if entry.expires_at is None:
            return -1

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

        import fnmatch

        # Clean expired entries
        expired = [k for k, v in self._cache.items() if v.is_expired]
        for key in expired:
            del self._cache[key]

        return [k for k in self._cache.keys() if fnmatch.fnmatch(k, pattern)]

    async def flush(self) -> int:
        """Clear all entries from cache.

        Returns:
            Number of entries cleared.
        """
        await self._simulate_latency()
        self._maybe_fail()

        count = len(self._cache)
        self._cache.clear()
        return count

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
        """Check cache health.

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
