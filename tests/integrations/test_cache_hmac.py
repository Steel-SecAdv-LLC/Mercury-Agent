"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests for cache serialisation behaviour.

The pickle serialiser was removed from the runtime; only JSON is
supported. These tests cover:

- JSON serialisation round-trip
- Refusal of legacy ``serializer="pickle"`` constructor argument
- Domain TTL policies
- Cache factory
- Key prefixing
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from omni_mercury_engine.integrations.stubs.cache import (
    DOMAIN_TTL,
    CacheStub,
    RedisCache,
    create_cache,
    get_domain_ttl,
)

# =============================================================================
# Legacy pickle serialiser refusal
# =============================================================================


class TestLegacyPickleRefused:
    """The 'pickle' serialiser is gone; the constructor must refuse it."""

    def test_pickle_serializer_rejected(self):
        with pytest.raises(ValueError, match="serializer"):
            RedisCache(serializer="pickle", fallback_to_stub=True)

    def test_unknown_serializer_rejected(self):
        with pytest.raises(ValueError, match="serializer"):
            RedisCache(serializer="msgpack", fallback_to_stub=True)

    def test_json_serializer_accepted(self):
        # No exception
        cache = RedisCache(serializer="json", fallback_to_stub=True)
        assert cache.serializer == "json"


# =============================================================================
# JSON Serialization Tests
# =============================================================================


class TestJSONSerialization:
    """Tests for JSON serialization (safe mode)."""

    @pytest.fixture
    def json_cache(self):
        """Create RedisCache with JSON serializer."""
        return RedisCache(serializer="json", fallback_to_stub=True)

    def test_serialize_json(self, json_cache):
        """Test JSON serialization."""
        serialized = json_cache._serialize({"key": "value", "num": 42})
        assert isinstance(serialized, str)
        import json

        parsed = json.loads(serialized)
        assert parsed == {"key": "value", "num": 42}

    def test_deserialize_json(self, json_cache):
        """Test JSON deserialization."""
        import json

        data = json.dumps({"key": "value"})
        result = json_cache._deserialize(data)
        assert result == {"key": "value"}

    def test_json_roundtrip(self, json_cache):
        """Test JSON serialize/deserialize round-trip."""
        original = {"list": [1, 2, 3], "str": "hello", "null": None}
        serialized = json_cache._serialize(original)
        deserialized = json_cache._deserialize(serialized)
        assert deserialized == original


# =============================================================================
# Domain TTL Tests
# =============================================================================


class TestDomainTTL:
    """Tests for domain-specific TTL policies."""

    def test_known_domains(self):
        """Test TTL values for known domains."""
        assert get_domain_ttl("environmental") == 300
        assert get_domain_ttl("security") == 60
        assert get_domain_ttl("climate") == 3600
        assert get_domain_ttl("medical") == 600
        assert get_domain_ttl("financial") == 120

    def test_unknown_domain_uses_default(self):
        """Test unknown domain falls back to default TTL."""
        assert get_domain_ttl("unknown_domain") == DOMAIN_TTL["default"]
        assert get_domain_ttl("") == DOMAIN_TTL["default"]

    def test_all_domains_have_positive_ttl(self):
        """Test all domain TTLs are positive integers."""
        for domain, ttl in DOMAIN_TTL.items():
            assert isinstance(ttl, int), f"TTL for {domain} is not an integer"
            assert ttl > 0, f"TTL for {domain} is not positive"


# =============================================================================
# Cache Factory Tests
# =============================================================================


class TestCacheFactory:
    """Tests for create_cache factory function."""

    def test_create_stub_cache(self):
        """Test creating stub cache."""
        cache = create_cache(backend="stub")
        assert isinstance(cache, CacheStub)

    def test_create_memory_cache(self):
        """Test creating memory cache (alias for stub)."""
        cache = create_cache(backend="memory")
        assert isinstance(cache, CacheStub)

    def test_create_redis_cache(self):
        """Test creating Redis cache instance."""
        cache = create_cache(backend="redis")
        assert isinstance(cache, RedisCache)

    def test_create_unknown_backend_falls_back(self):
        """Test unknown backend falls back to stub."""
        cache = create_cache(backend="unknown_backend")
        assert isinstance(cache, CacheStub)

    def test_redis_from_env(self):
        """Test Redis cache creation from environment."""
        with patch.dict(
            os.environ,
            {
                "REDIS_HOST": "test-host",
                "REDIS_PORT": "6380",
            },
        ):
            cache = RedisCache.from_env()
            assert cache.host == "test-host"
            assert cache.port == 6380


# =============================================================================
# RedisCache Key Prefixing Tests
# =============================================================================


class TestRedisCacheKeyPrefixing:
    """Tests for Redis key prefixing."""

    def test_make_key_adds_prefix(self):
        """Test that _make_key adds the configured prefix."""
        cache = RedisCache(prefix="test:")
        assert cache._make_key("mykey") == "test:mykey"

    def test_default_prefix(self):
        """Test default key prefix is 'mercury:'."""
        cache = RedisCache()
        assert cache._make_key("key") == "mercury:key"
