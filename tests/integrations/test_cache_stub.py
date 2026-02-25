"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests for integrations/stubs/cache.py module.
Comprehensive test coverage for cache stub functionality.
"""

from __future__ import annotations

import asyncio
import time

import pytest

# TODO: install pytest-asyncio in CI for full test coverage
pytest.importorskip("pytest_asyncio")

from omni_mercury_engine.integrations.stubs.cache import CacheEntry, CacheStub


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_basic_entry(self):
        """Test basic cache entry creation."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=time.time(),
            expires_at=None,
            ttl=None,
        )
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.hits == 0

    def test_entry_with_ttl(self):
        """Test cache entry with TTL."""
        now = time.time()
        entry = CacheEntry(
            key="key",
            value="value",
            created_at=now,
            expires_at=now + 300,
            ttl=300,
        )
        assert entry.ttl == 300
        assert entry.expires_at == now + 300

    def test_is_expired_false(self):
        """Test non-expired entry."""
        entry = CacheEntry(
            key="key",
            value="value",
            created_at=time.time(),
            expires_at=time.time() + 3600,
            ttl=3600,
        )
        assert entry.is_expired is False

    def test_is_expired_true(self):
        """Test expired entry."""
        entry = CacheEntry(
            key="key",
            value="value",
            created_at=time.time() - 100,
            expires_at=time.time() - 50,
            ttl=50,
        )
        assert entry.is_expired is True

    def test_is_expired_no_expiry(self):
        """Test entry without expiry is never expired."""
        entry = CacheEntry(
            key="key",
            value="value",
            created_at=time.time(),
            expires_at=None,
            ttl=None,
        )
        assert entry.is_expired is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        entry = CacheEntry(
            key="test_key",
            value={"data": "test"},
            created_at=time.time(),
            expires_at=None,
            ttl=None,
            hits=5,
        )
        d = entry.to_dict()
        assert d["key"] == "test_key"
        assert d["value"] == {"data": "test"}
        assert d["hits"] == 5
        assert "created_at" in d
        assert d["ttl"] is None

    def test_to_dict_with_expiry(self):
        """Test serialization with expiry time."""
        now = time.time()
        entry = CacheEntry(
            key="key",
            value="value",
            created_at=now,
            expires_at=now + 300,
            ttl=300,
        )
        d = entry.to_dict()
        assert d["expires_at"] is not None
        assert d["ttl"] == 300
        assert "is_expired" in d


class TestCacheStubInitialization:
    """Tests for CacheStub initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        cache = CacheStub()
        assert cache._max_size == 10000
        assert cache._failure_rate == 0.0

    def test_custom_seed(self):
        """Test initialization with custom seed."""
        cache = CacheStub(seed=42)
        assert cache._rng is not None

    def test_custom_latency(self):
        """Test initialization with custom latency."""
        cache = CacheStub(latency_ms=(5, 20))
        assert cache._latency_ms == (5, 20)

    def test_custom_failure_rate(self):
        """Test initialization with custom failure rate."""
        cache = CacheStub(failure_rate=0.1)
        assert cache._failure_rate == 0.1

    def test_custom_max_size(self):
        """Test initialization with custom max size."""
        cache = CacheStub(max_size=100)
        assert cache._max_size == 100


class TestCacheStubBasicOperations:
    """Tests for basic cache operations."""

    @pytest.fixture
    def cache(self):
        """Create cache fixture."""
        return CacheStub(seed=42, latency_ms=(0, 1))

    @pytest.mark.asyncio
    async def test_set_and_get(self, cache):
        """Test basic set and get operations."""
        await cache.set("key1", "value1")
        result = await cache.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, cache):
        """Test getting nonexistent key returns None."""
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_with_ttl(self, cache):
        """Test set with TTL."""
        await cache.set("key", "value", ttl=300)
        result = await cache.get("key")
        assert result == "value"

    @pytest.mark.asyncio
    async def test_delete_key(self, cache):
        """Test key deletion."""
        await cache.set("key", "value")
        deleted = await cache.delete("key")
        assert deleted is True
        result = await cache.get("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_key(self, cache):
        """Test deleting nonexistent key."""
        deleted = await cache.delete("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_exists_true(self, cache):
        """Test exists returns True for existing key."""
        await cache.set("key", "value")
        exists = await cache.exists("key")
        assert exists is True

    @pytest.mark.asyncio
    async def test_exists_false(self, cache):
        """Test exists returns False for nonexistent key."""
        exists = await cache.exists("nonexistent")
        assert exists is False

    @pytest.mark.asyncio
    async def test_clear_cache(self, cache):
        """Test clearing all cache entries."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.clear()
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None


class TestCacheStubDataTypes:
    """Tests for caching different data types."""

    @pytest.fixture
    def cache(self):
        """Create cache fixture."""
        return CacheStub(seed=42, latency_ms=(0, 1))

    @pytest.mark.asyncio
    async def test_cache_string(self, cache):
        """Test caching string value."""
        await cache.set("string", "hello world")
        assert await cache.get("string") == "hello world"

    @pytest.mark.asyncio
    async def test_cache_integer(self, cache):
        """Test caching integer value."""
        await cache.set("int", 42)
        assert await cache.get("int") == 42

    @pytest.mark.asyncio
    async def test_cache_float(self, cache):
        """Test caching float value."""
        await cache.set("float", 3.14159)
        assert await cache.get("float") == 3.14159

    @pytest.mark.asyncio
    async def test_cache_dict(self, cache):
        """Test caching dictionary value."""
        data = {"name": "Alice", "age": 30}
        await cache.set("dict", data)
        assert await cache.get("dict") == data

    @pytest.mark.asyncio
    async def test_cache_list(self, cache):
        """Test caching list value."""
        data = [1, 2, 3, 4, 5]
        await cache.set("list", data)
        assert await cache.get("list") == data

    @pytest.mark.asyncio
    async def test_cache_none(self, cache):
        """Test caching None value."""
        await cache.set("none", None)
        # Note: This might return None which could be ambiguous
        _ = await cache.get("none")
        # Implementation dependent - just verify no exception


class TestCacheStubTTL:
    """Tests for TTL (Time To Live) functionality."""

    @pytest.fixture
    def cache(self):
        """Create cache fixture."""
        return CacheStub(seed=42, latency_ms=(0, 1))

    @pytest.mark.asyncio
    async def test_expired_entry_not_returned(self, cache):
        """Test expired entry returns None."""
        # Set with very short TTL
        await cache.set("key", "value", ttl=0)
        # Wait briefly
        await asyncio.sleep(0.01)
        result = await cache.get("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_non_expired_entry_returned(self, cache):
        """Test non-expired entry is returned."""
        await cache.set("key", "value", ttl=3600)
        result = await cache.get("key")
        assert result == "value"

    @pytest.mark.asyncio
    async def test_get_ttl(self, cache):
        """Test getting remaining TTL."""
        await cache.set("key", "value", ttl=300)
        ttl = await cache.ttl("key")
        assert ttl is not None
        assert ttl > 0
        assert ttl <= 300

    @pytest.mark.asyncio
    async def test_get_ttl_nonexistent(self, cache):
        """Test getting TTL for nonexistent key."""
        ttl = await cache.ttl("nonexistent")
        assert ttl is None


class TestCacheStubBulkOperations:
    """Tests for bulk cache operations."""

    @pytest.fixture
    def cache(self):
        """Create cache fixture."""
        return CacheStub(seed=42, latency_ms=(0, 1))

    @pytest.mark.asyncio
    async def test_mset(self, cache):
        """Test setting multiple keys."""
        data = {"key1": "value1", "key2": "value2", "key3": "value3"}
        await cache.mset(data)
        assert await cache.get("key1") == "value1"
        assert await cache.get("key2") == "value2"
        assert await cache.get("key3") == "value3"

    @pytest.mark.asyncio
    async def test_mget(self, cache):
        """Test getting multiple keys."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        results = await cache.mget(["key1", "key2", "nonexistent"])
        assert results["key1"] == "value1"
        assert results["key2"] == "value2"
        assert results["nonexistent"] is None

    @pytest.mark.asyncio
    async def test_mdelete(self, cache):
        """Test deleting multiple keys."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        count = await cache.mdelete(["key1", "key2", "nonexistent"])
        assert count == 2
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None


class TestCacheStubStatistics:
    """Tests for cache statistics."""

    @pytest.fixture
    def cache(self):
        """Create cache fixture."""
        return CacheStub(seed=42, latency_ms=(0, 1))

    @pytest.mark.asyncio
    async def test_get_stats(self, cache):
        """Test getting cache statistics."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.get("key1")
        await cache.get("key1")
        await cache.get("nonexistent")

        stats = await cache.get_stats()
        assert "total_entries" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert stats["total_entries"] == 2

    @pytest.mark.asyncio
    async def test_hit_count(self, cache):
        """Test hit count tracking."""
        await cache.set("key", "value")
        await cache.get("key")
        await cache.get("key")
        await cache.get("key")

        stats = await cache.get_stats()
        assert stats["hits"] >= 3

    @pytest.mark.asyncio
    async def test_miss_count(self, cache):
        """Test miss count tracking."""
        await cache.get("nonexistent1")
        await cache.get("nonexistent2")

        stats = await cache.get_stats()
        assert stats["misses"] >= 2

    @pytest.mark.asyncio
    async def test_size(self, cache):
        """Test cache size reporting."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        size = await cache.size()
        assert size == 2


class TestCacheStubIncrementDecrement:
    """Tests for increment/decrement operations."""

    @pytest.fixture
    def cache(self):
        """Create cache fixture."""
        return CacheStub(seed=42, latency_ms=(0, 1))

    @pytest.mark.asyncio
    async def test_incr(self, cache):
        """Test increment operation."""
        await cache.set("counter", 10)
        result = await cache.incr("counter")
        assert result == 11

    @pytest.mark.asyncio
    async def test_incr_by_amount(self, cache):
        """Test increment by specific amount."""
        await cache.set("counter", 10)
        result = await cache.incr("counter", amount=5)
        assert result == 15

    @pytest.mark.asyncio
    async def test_decr(self, cache):
        """Test decrement operation."""
        await cache.set("counter", 10)
        result = await cache.decr("counter")
        assert result == 9

    @pytest.mark.asyncio
    async def test_decr_by_amount(self, cache):
        """Test decrement by specific amount."""
        await cache.set("counter", 10)
        result = await cache.decr("counter", amount=3)
        assert result == 7

    @pytest.mark.asyncio
    async def test_incr_nonexistent_key(self, cache):
        """Test increment on nonexistent key."""
        result = await cache.incr("new_counter")
        assert result == 1


class TestCacheStubKeyOperations:
    """Tests for key listing and pattern operations."""

    @pytest.fixture
    def cache(self):
        """Create cache fixture."""
        return CacheStub(seed=42, latency_ms=(0, 1))

    @pytest.mark.asyncio
    async def test_keys(self, cache):
        """Test listing all keys."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        keys = await cache.keys()
        assert "key1" in keys
        assert "key2" in keys

    @pytest.mark.asyncio
    async def test_keys_pattern(self, cache):
        """Test listing keys with pattern."""
        await cache.set("user:1", "Alice")
        await cache.set("user:2", "Bob")
        await cache.set("item:1", "Widget")

        keys = await cache.keys(pattern="user:*")
        assert len(keys) == 2
        assert all(k.startswith("user:") for k in keys)


class TestCacheStubExpireAndPersist:
    """Tests for expire and persist operations."""

    @pytest.fixture
    def cache(self):
        """Create cache fixture."""
        return CacheStub(seed=42, latency_ms=(0, 1))

    @pytest.mark.asyncio
    async def test_expire(self, cache):
        """Test setting expiry on existing key."""
        await cache.set("key", "value")
        result = await cache.expire("key", 300)
        assert result is True

        ttl = await cache.ttl("key")
        assert ttl is not None
        assert ttl <= 300

    @pytest.mark.asyncio
    async def test_expire_nonexistent(self, cache):
        """Test setting expiry on nonexistent key."""
        result = await cache.expire("nonexistent", 300)
        assert result is False

    @pytest.mark.asyncio
    async def test_persist(self, cache):
        """Test removing expiry from key."""
        await cache.set("key", "value", ttl=300)
        result = await cache.persist("key")
        assert result is True

        ttl = await cache.ttl("key")
        assert ttl is None


class TestCacheStubFailureSimulation:
    """Tests for failure simulation."""

    @pytest.mark.asyncio
    async def test_no_failures_with_zero_rate(self):
        """Test no failures with zero failure rate."""
        cache = CacheStub(failure_rate=0.0, latency_ms=(0, 1))
        # Should not raise
        await cache.set("key", "value")
        result = await cache.get("key")
        assert result == "value"

    @pytest.mark.asyncio
    async def test_high_failure_rate(self):
        """Test high failure rate causes some failures."""
        cache = CacheStub(failure_rate=1.0, seed=42, latency_ms=(0, 1))
        # With 100% failure rate, operations should fail
        with pytest.raises((Exception, RuntimeError)):
            await cache.set("key", "value")


class TestCacheStubMaxSize:
    """Tests for max size enforcement."""

    @pytest.mark.asyncio
    async def test_max_size_enforced(self):
        """Test max size is enforced."""
        cache = CacheStub(max_size=3, latency_ms=(0, 1))
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")
        await cache.set("key4", "value4")

        # Should have at most 3 entries
        size = await cache.size()
        assert size <= 3
