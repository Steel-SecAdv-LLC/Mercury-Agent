"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Tests for cache serialisation behaviour.

The pickle serialiser was removed from the runtime; only JSON is
supported. These tests cover:

- Refusal of legacy ``serializer="pickle"`` constructor argument
- Refusal of any non-JSON serializer (case-sensitive, exact match)
- Operator-actionable error messages that point at the migration tool
- JSON serialisation round-trip across the full json type spectrum
  (None, bool, int, float, str, list, dict, nested, unicode)
- Non-JSON-serialisable values surface as loud TypeError -- never
  silently dropped or coerced
- Domain TTL policies (lookup, defaults, completeness)
- Cache factory across all supported backends
- Key prefixing, including empty / custom / colon-suffix variants
- ``from_env`` constructor across all REDIS_* env vars, including
  the SSL flag parsing variants
"""

from __future__ import annotations

import json
import os
from typing import Any
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

    def test_pickle_serializer_rejected(self) -> None:
        with pytest.raises(ValueError, match="serializer"):
            RedisCache(serializer="pickle", fallback_to_stub=True)

    def test_unknown_serializer_rejected(self) -> None:
        with pytest.raises(ValueError, match="serializer"):
            RedisCache(serializer="msgpack", fallback_to_stub=True)

    def test_json_serializer_accepted(self) -> None:
        # No exception
        cache = RedisCache(serializer="json", fallback_to_stub=True)
        assert cache.serializer == "json"

    def test_pickle_error_message_points_to_migrate_pkl(self) -> None:
        """Operator-actionable message: the error must name the migration tool."""
        with pytest.raises(ValueError) as exc_info:
            RedisCache(serializer="pickle", fallback_to_stub=True)
        msg = str(exc_info.value)
        assert (
            "migrate_pkl" in msg
        ), "The pickle refusal should point operators at the offline migration tool"

    def test_empty_serializer_rejected(self) -> None:
        """The constructor must not silently coerce '' to 'json'."""
        with pytest.raises(ValueError, match="serializer"):
            RedisCache(serializer="", fallback_to_stub=True)

    def test_case_sensitive_serializer(self) -> None:
        """'JSON' is not 'json' -- exact match only, no normalisation."""
        with pytest.raises(ValueError, match="serializer"):
            RedisCache(serializer="JSON", fallback_to_stub=True)

    def test_default_serializer_is_json(self) -> None:
        """Default-constructed cache is JSON, never pickle."""
        cache = RedisCache(fallback_to_stub=True)
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

    def test_serialize_json(self, json_cache: Any) -> None:
        """Test JSON serialization."""
        serialized = json_cache._serialize({"key": "value", "num": 42})
        assert isinstance(serialized, str)
        parsed = json.loads(serialized)
        assert parsed == {"key": "value", "num": 42}

    def test_deserialize_json(self, json_cache: Any) -> None:
        """Test JSON deserialization."""
        data = json.dumps({"key": "value"})
        result = json_cache._deserialize(data)
        assert result == {"key": "value"}

    def test_json_roundtrip(self, json_cache: Any) -> None:
        """Test JSON serialize/deserialize round-trip."""
        original = {"list": [1, 2, 3], "str": "hello", "null": None}
        serialized = json_cache._serialize(original)
        deserialized = json_cache._deserialize(serialized)
        assert deserialized == original

    def test_deserialize_none_returns_none(self, json_cache: Any) -> None:
        """Missing keys come back from Redis as None; the cache must pass that through."""
        assert json_cache._deserialize(None) is None

    @pytest.mark.parametrize(
        "value",
        [
            None,
            True,
            False,
            0,
            1,
            -1,
            42,
            2**53 - 1,
            -(2**53 - 1),
            0.0,
            3.14159,
            -2.718,
            "",
            "ascii",
            "unicode: ✓ ☠ é",
            'embedded "quotes" and \\backslashes',
            [],
            [1, 2, 3],
            {},
            {"a": 1},
        ],
    )
    def test_json_roundtrip_scalar_and_container_types(self, json_cache: Any, value: Any) -> None:
        """Every JSON-spec type must round-trip without loss."""
        serialized = json_cache._serialize(value)
        deserialized = json_cache._deserialize(serialized)
        assert deserialized == value

    def test_json_roundtrip_deeply_nested(self, json_cache: Any) -> None:
        """Nested containers (dict-of-dict-of-list-of-dict) must round-trip."""
        original = {
            "level1": {
                "level2": {
                    "items": [
                        {"id": 1, "tags": ["a", "b"]},
                        {"id": 2, "tags": []},
                    ],
                    "meta": None,
                }
            }
        }
        serialized = json_cache._serialize(original)
        assert isinstance(serialized, str)
        assert json_cache._deserialize(serialized) == original

    def test_non_json_value_raises_typeerror(self, json_cache: Any) -> None:
        """Sets, bytes, datetimes -- the cache MUST refuse loudly, not coerce.

        Pickle would have happily serialised any of these; JSON has no
        cross-language way to represent them, so the contract for JSON
        cache is: caller converts first, or the call fails. No silent
        dropping or stringification.
        """

        class _NotJSONable:
            pass

        for hostile in ({1, 2, 3}, b"bytes", _NotJSONable()):
            with pytest.raises(TypeError):
                json_cache._serialize(hostile)

    def test_deserialize_malformed_json_raises(self, json_cache: Any) -> None:
        """Corrupted payloads must raise JSONDecodeError, not return garbage."""
        with pytest.raises(json.JSONDecodeError):
            json_cache._deserialize("{not valid json")

    def test_string_with_dot_safely_roundtrips(self, json_cache: Any) -> None:
        """The old pickle path used '.' as the HMAC/payload separator.

        Make sure strings that contain '.' are not split or otherwise
        mangled by the JSON path -- a real regression risk if the old
        pickle parsing code is ever re-introduced by accident.
        """
        original = "hello.world.contains.dots"
        assert json_cache._deserialize(json_cache._serialize(original)) == original


# =============================================================================
# Domain TTL Tests
# =============================================================================


class TestDomainTTL:
    """Tests for domain-specific TTL policies."""

    def test_known_domains(self) -> None:
        """Test TTL values for known domains."""
        assert get_domain_ttl("environmental") == 300
        assert get_domain_ttl("security") == 60
        assert get_domain_ttl("climate") == 3600
        assert get_domain_ttl("medical") == 600
        assert get_domain_ttl("financial") == 120

    def test_unknown_domain_uses_default(self) -> None:
        """Test unknown domain falls back to default TTL."""
        assert get_domain_ttl("unknown_domain") == DOMAIN_TTL["default"]
        assert get_domain_ttl("") == DOMAIN_TTL["default"]

    def test_all_domains_have_positive_ttl(self) -> None:
        """Test all domain TTLs are positive integers."""
        for domain, ttl in DOMAIN_TTL.items():
            assert isinstance(ttl, int), f"TTL for {domain} is not an integer"
            assert ttl > 0, f"TTL for {domain} is not positive"

    def test_default_key_present(self) -> None:
        """The 'default' bucket must exist; ``get_domain_ttl`` relies on it."""
        assert "default" in DOMAIN_TTL

    def test_domain_lookup_is_case_sensitive(self) -> None:
        """No silent case-folding: 'Security' is not 'security'."""
        assert get_domain_ttl("Security") == DOMAIN_TTL["default"]
        assert get_domain_ttl("SECURITY") == DOMAIN_TTL["default"]


# =============================================================================
# Cache Factory Tests
# =============================================================================


class TestCacheFactory:
    """Tests for create_cache factory function."""

    def test_create_stub_cache(self) -> None:
        """Test creating stub cache."""
        cache = create_cache(backend="stub")
        assert isinstance(cache, CacheStub)

    def test_create_memory_cache(self) -> None:
        """Test creating memory cache (alias for stub)."""
        cache = create_cache(backend="memory")
        assert isinstance(cache, CacheStub)

    def test_create_redis_cache(self) -> None:
        """Test creating Redis cache instance."""
        cache = create_cache(backend="redis")
        assert isinstance(cache, RedisCache)

    def test_create_unknown_backend_falls_back(self) -> None:
        """Test unknown backend falls back to stub."""
        cache = create_cache(backend="unknown_backend")
        assert isinstance(cache, CacheStub)

    def test_create_redis_propagates_serializer_refusal(self) -> None:
        """A factory caller must not be able to sneak ``serializer='pickle'`` past us."""
        with pytest.raises(ValueError, match="serializer"):
            create_cache(backend="redis", serializer="pickle")

    def test_redis_from_env(self) -> None:
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

    def test_redis_from_env_full_surface(self) -> None:
        """``from_env`` must honour every documented REDIS_* variable."""
        with patch.dict(
            os.environ,
            {
                "REDIS_HOST": "host.example",
                "REDIS_PORT": "9999",
                "REDIS_PASSWORD": "s3cret",
                "REDIS_DB": "7",
                "REDIS_SSL": "true",
                "REDIS_PREFIX": "myapp:",
            },
            clear=False,
        ):
            cache = RedisCache.from_env()
            assert cache.host == "host.example"
            assert cache.port == 9999
            assert cache.password == "s3cret"
            assert cache.db == 7
            assert cache.ssl is True
            assert cache.prefix == "myapp:"

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("Yes", True),
            ("false", False),
            ("0", False),
            ("no", False),
            ("", False),
            ("nope", False),
        ],
    )
    def test_redis_from_env_ssl_parsing(self, raw: Any, expected: Any) -> None:
        """REDIS_SSL must parse as truthy only for documented affirmative values."""
        with patch.dict(os.environ, {"REDIS_SSL": raw}, clear=False):
            cache = RedisCache.from_env()
            assert cache.ssl is expected

    def test_redis_from_env_invalid_port_raises(self) -> None:
        """REDIS_PORT='not-a-number' must raise ValueError, never silently default."""
        with (
            patch.dict(os.environ, {"REDIS_PORT": "not-a-number"}, clear=False),
            pytest.raises(ValueError),
        ):
            RedisCache.from_env()


# =============================================================================
# RedisCache Key Prefixing Tests
# =============================================================================


class TestRedisCacheKeyPrefixing:
    """Tests for Redis key prefixing."""

    def test_make_key_adds_prefix(self) -> None:
        """Test that _make_key adds the configured prefix."""
        cache = RedisCache(prefix="test:")
        assert cache._make_key("mykey") == "test:mykey"

    def test_default_prefix(self) -> None:
        """Test default key prefix is 'mercury:'."""
        cache = RedisCache()
        assert cache._make_key("key") == "mercury:key"

    def test_empty_prefix(self) -> None:
        """An explicit empty prefix must produce the unprefixed key."""
        cache = RedisCache(prefix="")
        assert cache._make_key("key") == "key"

    def test_prefix_without_colon(self) -> None:
        """No automatic colon insertion -- prefix is concatenated verbatim."""
        cache = RedisCache(prefix="raw")
        assert cache._make_key("key") == "rawkey"

    def test_prefix_preserves_key_unicode(self) -> None:
        """Keys with unicode characters must not be mangled by prefixing."""
        cache = RedisCache(prefix="✓:")
        assert cache._make_key("kéy") == "✓:kéy"


# =============================================================================
# RedisCache constructor surface
# =============================================================================


class TestRedisCacheConstructor:
    """Tests for the RedisCache __init__ argument surface.

    The bandit-sensitive defaults (no pickle, no implicit insecure fallbacks)
    are policy and must be enforced at construction time, not lazily.
    """

    def test_default_metrics_zero(self) -> None:
        """Fresh cache has zero call/error/fallback counters."""
        cache = RedisCache()
        assert cache._call_count == 0
        assert cache._errors == 0
        assert cache._fallback_count == 0

    def test_default_not_connected(self) -> None:
        """Fresh cache is disconnected until ``_ensure_connected`` succeeds."""
        cache = RedisCache()
        assert cache._connected is False
        assert cache._client is None

    def test_fallback_stub_is_isolated_instance(self) -> None:
        """Each RedisCache owns its own CacheStub fallback; they do not share state."""
        a = RedisCache(fallback_to_stub=True)
        b = RedisCache(fallback_to_stub=True)
        assert a._stub is not b._stub

    def test_custom_host_port_overrides_env(self) -> None:
        """Explicit constructor args win over REDIS_* env vars."""
        with patch.dict(
            os.environ,
            {"REDIS_HOST": "env-host", "REDIS_PORT": "1111"},
            clear=False,
        ):
            cache = RedisCache(host="explicit-host", port=2222)
            assert cache.host == "explicit-host"
            assert cache.port == 2222


# =============================================================================
# CacheStub fallback behaviour
# =============================================================================


class TestFallbackToStubPreservesJSONContract:
    """JSON-only contract holds even when the call falls back to the in-memory stub.

    The bug this regression test pins: the previous code routed to
    ``_stub.set(value)`` BEFORE calling ``_serialize(value)``, so a
    non-JSON-serialisable value (set, bytes, custom object) was
    silently stored in the stub when Redis was unavailable. Developers
    running offline would write code that worked locally and crashed
    in production the first time the Redis path was exercised.
    """

    @pytest.mark.asyncio
    async def test_set_rejects_non_json_in_fallback_mode(self) -> None:
        """``fallback_to_stub=True`` must not bypass the JSON contract."""
        cache = RedisCache(fallback_to_stub=True)
        # Redis is unreachable in the test sandbox; _ensure_connected
        # returns False and the fallback path runs. The TypeError must
        # surface BEFORE the stub is touched.
        with pytest.raises(TypeError):
            await cache.set("k", {1, 2, 3})  # set() is not JSON-able

    @pytest.mark.asyncio
    async def test_set_rejects_bytes_in_fallback_mode(self) -> None:
        cache = RedisCache(fallback_to_stub=True)
        with pytest.raises(TypeError):
            await cache.set("k", b"raw bytes")

    @pytest.mark.asyncio
    async def test_mset_rejects_non_json_in_fallback_mode(self) -> None:
        cache = RedisCache(fallback_to_stub=True)
        with pytest.raises(TypeError):
            await cache.mset({"a": 1, "b": {7, 8}})  # 'b' is a set


class TestCacheStub:
    """The in-memory CacheStub is the always-available fallback.

    It must behave like a real cache for the public surface that
    callers depend on, so that fallback-to-stub is invisible to the
    caller other than via the metrics counters.
    """

    @pytest.mark.asyncio
    async def test_set_then_get(self) -> None:
        cache = CacheStub()
        await cache.set("key", {"v": 1})
        assert await cache.get("key") == {"v": 1}

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self) -> None:
        cache = CacheStub()
        assert await cache.get("never-set") is None

    @pytest.mark.asyncio
    async def test_delete_removes_key(self) -> None:
        cache = CacheStub()
        await cache.set("k", 1)
        assert await cache.delete("k") is True
        assert await cache.get("k") is None

    @pytest.mark.asyncio
    async def test_exists_reports_membership(self) -> None:
        cache = CacheStub()
        assert await cache.exists("missing") is False
        await cache.set("present", "v")
        assert await cache.exists("present") is True

    @pytest.mark.asyncio
    async def test_set_nx_blocks_when_key_exists(self) -> None:
        """nx=True (set-if-not-exists) must NOT overwrite an existing key."""
        cache = CacheStub()
        await cache.set("k", "first")
        await cache.set("k", "second", nx=True)
        assert await cache.get("k") == "first"
