"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests for cache HMAC signing and security hardening.

Covers:
- HMAC-signed pickle serialization
- HMAC signature verification (tamper detection)
- Production enforcement for MERCURY_CACHE_SECRET
- Domain TTL policies
- RedisCache serialization/deserialization
- Cache factory
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
# HMAC Signing Key Tests
# =============================================================================


class TestHMACSigningKey:
    """Tests for HMAC signing key management."""

    def test_get_signing_key_with_env_var(self):
        """Test signing key from environment variable."""
        with patch.dict(os.environ, {"MERCURY_CACHE_SECRET": "my-strong-secret"}):
            key = RedisCache._get_signing_key()
            assert isinstance(key, bytes)
            assert len(key) == 32  # SHA-256 produces 32 bytes

    def test_get_signing_key_default_dev(self):
        """Test default signing key in development mode."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MERCURY_CACHE_SECRET", None)
            os.environ.pop("MERCURY_AGENT_ENV", None)
            # Reset warning flag
            CacheStub._CACHE_SECRET_WARNED = False
            key = RedisCache._get_signing_key()
            assert isinstance(key, bytes)
            assert len(key) == 32

    def test_get_signing_key_production_requires_secret(self):
        """Test production mode requires MERCURY_CACHE_SECRET."""
        with patch.dict(
            os.environ,
            {"MERCURY_AGENT_ENV": "production"},
            clear=False,
        ):
            os.environ.pop("MERCURY_CACHE_SECRET", None)
            with pytest.raises(ValueError, match="MERCURY_CACHE_SECRET"):
                RedisCache._get_signing_key()


# =============================================================================
# Pickle Serialization with HMAC Tests
# =============================================================================


class TestPickleHMACSerialization:
    """Tests for HMAC-signed pickle serialization/deserialization."""

    @pytest.fixture
    def redis_cache(self):
        """Create RedisCache with pickle serializer."""
        with patch.dict(os.environ, {"MERCURY_CACHE_SECRET": "test-secret-key"}):
            cache = RedisCache(serializer="pickle", fallback_to_stub=True)
            return cache

    def test_serialize_pickle_includes_hmac(self, redis_cache):
        """Test that pickle serialization prepends HMAC signature."""
        serialized = redis_cache._serialize({"key": "value"})
        assert isinstance(serialized, str)
        assert "." in serialized  # HMAC.base64_payload format
        parts = serialized.split(".", 1)
        assert len(parts) == 2
        # First part should be hex-encoded HMAC (64 chars for SHA-256)
        assert len(parts[0]) == 64

    def test_deserialize_pickle_valid_hmac(self, redis_cache):
        """Test deserialization succeeds with valid HMAC."""
        original = {"data": [1, 2, 3], "nested": {"a": True}}
        serialized = redis_cache._serialize(original)
        deserialized = redis_cache._deserialize(serialized)
        assert deserialized == original

    def test_deserialize_pickle_tampered_data(self, redis_cache):
        """Test deserialization rejects tampered data."""
        original = {"secret": "classified"}
        serialized = redis_cache._serialize(original)

        # Tamper with the base64 payload
        sig, _, payload = serialized.partition(".")
        tampered = sig + "." + payload[:-4] + "XXXX"

        result = redis_cache._deserialize(tampered)
        assert result is None  # Should reject tampered data

    def test_deserialize_pickle_tampered_signature(self, redis_cache):
        """Test deserialization rejects tampered signature."""
        original = {"data": "test"}
        serialized = redis_cache._serialize(original)

        # Replace signature with wrong value
        _, _, payload = serialized.partition(".")
        tampered = "a" * 64 + "." + payload

        result = redis_cache._deserialize(tampered)
        assert result is None

    def test_deserialize_pickle_missing_signature(self, redis_cache):
        """Test deserialization rejects data without HMAC separator."""
        result = redis_cache._deserialize("no_dot_separator_here")
        assert result is None

    def test_deserialize_none_returns_none(self, redis_cache):
        """Test deserialization of None returns None."""
        result = redis_cache._deserialize(None)
        assert result is None


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
        with patch.dict(os.environ, {
            "REDIS_HOST": "test-host",
            "REDIS_PORT": "6380",
        }):
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
