# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for JWT authentication edge cases and fail-closed production behavior.

Covers:
- Missing JWT_SECRET_KEY environment variable
- Expired JWT tokens
- Malformed JWT tokens
- Invalid signatures
- Missing required claims
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest
from ama_cryptography.crypto_api import HMAC_HKDF_AVAILABLE

# ``api.auth`` imports fastapi at module level; skip cleanly when the
# optional ``api`` extra is absent so the rest of the suite is still
# discoverable.
pytest.importorskip("fastapi")


class TestJWTAuthMissingKey:
    """Tests for JWT authentication with missing secret key."""

    def test_jwt_auth_missing_key_dev_fallback(self) -> None:
        """Test JWT auth uses dev fallback when key missing in non-production."""
        # Clear any existing JWT_SECRET_KEY
        with patch.dict(os.environ, {}, clear=True):
            # Remove production indicators
            os.environ.pop("MERCURY_AGENT_ENV", None)
            os.environ.pop("ENV", None)
            os.environ.pop("ENVIRONMENT", None)
            os.environ.pop("JWT_SECRET_KEY", None)

            from omni_mercury_engine.api.auth import JWTAuth

            # Reset the warning flag for testing
            JWTAuth._warned_about_fallback = False

            auth = JWTAuth(allow_dev_fallback=True)

            # Should use the ephemeral per-process fallback key: not the old
            # published constant, a proper 32-byte random hex, and stable within
            # the process so two instances can sign/verify each other's tokens.
            assert auth.using_fallback is True
            assert auth.secret_key == JWTAuth._get_dev_fallback_key()
            assert auth.secret_key != "MERCURY_AGENT_DEV_FALLBACK_KEY_DO_NOT_USE_IN_PRODUCTION"
            assert len(auth.secret_key) == 64  # secrets.token_hex(32)
            assert JWTAuth(allow_dev_fallback=True).secret_key == auth.secret_key

    def test_jwt_auth_missing_key_no_fallback_raises(self) -> None:
        """Test JWT auth raises error when key missing and fallback disabled."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("JWT_SECRET_KEY", None)
            os.environ.pop("MERCURY_AGENT_ENV", None)

            from omni_mercury_engine.api.auth import JWTAuth

            with pytest.raises(ValueError, match="JWT_SECRET_KEY"):
                JWTAuth(allow_dev_fallback=False)

    def test_jwt_auth_missing_key_production_derives_from_ama_hd(self) -> None:
        """In production w/ AMA available, JWT key is derived from HD key management.

        FIPS 204/205 substrate is online: ``get_auth_key_manager()`` delegates
        to AMA's HD key management which yields a deterministic ``jwt_sign``
        key material. The constructor must therefore *succeed* (no ValueError)
        and ``self.secret_key`` must be populated. Pre-PR-#167 the test
        asserted ValueError unconditionally, which silently masked the AMA
        success path the moment the upstream library landed; splitting the
        contract into two paths surfaces both branches honestly.
        """
        with patch.dict(os.environ, {"MERCURY_AGENT_ENV": "production"}, clear=True):
            os.environ.pop("JWT_SECRET_KEY", None)

            from omni_mercury_engine.api.auth import JWTAuth

            assert HMAC_HKDF_AVAILABLE, "AMA native HMAC/HKDF backend is mandatory."
            auth = JWTAuth()
            assert (
                auth.secret_key is not None and len(auth.secret_key) > 0
            ), "AMA HD-derived JWT key must populate self.secret_key in production."
            assert auth.using_fallback is False

    def test_jwt_auth_missing_key_production_raises_when_ama_unavailable(self) -> None:
        """In production w/o AMA, JWTAuth raises ValueError pinning the HD failure.

        Mocks ``get_auth_key_manager`` to raise (the same surface signature
        the constructor sees when AMA is import-broken or the native backend
        is missing) and asserts the constructor raises ``ValueError`` whose
        message names ``production`` so operators know to set
        ``JWT_SECRET_KEY`` rather than relying on HD derivation.
        """
        from unittest.mock import patch as _patch

        with patch.dict(os.environ, {"MERCURY_AGENT_ENV": "production"}, clear=True):
            os.environ.pop("JWT_SECRET_KEY", None)

            from omni_mercury_engine.api.auth import JWTAuth

            with (
                _patch(
                    "omni_mercury_engine.api.auth.get_auth_key_manager",
                    side_effect=RuntimeError("AMA HD key management unavailable in test"),
                ),
                pytest.raises(ValueError, match="production"),
            ):
                JWTAuth()


class TestJWTAuthExpiredToken:
    """Tests for JWT authentication with expired tokens."""

    @pytest.fixture
    def jwt_auth(self):
        """Create JWTAuth instance with test key."""
        with patch.dict(os.environ, {"JWT_SECRET_KEY": "test_secret_key_for_testing"}):
            from omni_mercury_engine.api.auth import JWTAuth

            return JWTAuth()

    @pytest.fixture
    def expired_token(self):
        """Create an expired JWT token via Mercury's native JWT library."""
        from omni_mercury_engine.security import native_jwt as jwt

        payload = {
            "sub": "test_user",
            "username": "testuser",
            "roles": ["user"],
            "permissions": ["read"],
            "exp": datetime.now() - timedelta(hours=1),  # Expired 1 hour ago
            "iat": datetime.now() - timedelta(hours=2),
        }
        return jwt.encode(payload, "test_secret_key_for_testing", algorithm="HS256")

    @pytest.mark.asyncio
    async def test_expired_token_returns_none(self, jwt_auth: Any, expired_token: Any) -> None:
        """Test that expired tokens return None."""
        result = await jwt_auth._validate_jwt(expired_token)
        assert result is None


class TestJWTAuthMalformedToken:
    """Tests for JWT authentication with malformed tokens."""

    @pytest.fixture
    def jwt_auth(self):
        """Create JWTAuth instance with test key."""
        with patch.dict(os.environ, {"JWT_SECRET_KEY": "test_secret_key_for_testing"}):
            from omni_mercury_engine.api.auth import JWTAuth

            return JWTAuth()

    @pytest.mark.asyncio
    async def test_malformed_token_returns_none(self, jwt_auth: Any) -> None:
        """Test that malformed tokens return None."""
        malformed_tokens = [
            "not.a.valid.jwt",
            "completely_invalid",
            "",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",  # Only header
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0",  # Missing signature
            "a.b.c",  # Invalid base64
        ]

        for token in malformed_tokens:
            result = await jwt_auth._validate_jwt(token)
            assert result is None, f"Expected None for malformed token: {token}"

    @pytest.mark.asyncio
    async def test_invalid_signature_returns_none(self, jwt_auth: Any) -> None:
        """Test that tokens with invalid signatures return None."""
        from omni_mercury_engine.security import native_jwt as jwt

        # Create token with different key
        payload = {
            "sub": "test_user",
            "username": "testuser",
            "roles": ["user"],
            "permissions": ["read"],
            "exp": datetime.now() + timedelta(hours=1),
            "iat": datetime.now(),
        }
        wrong_key_token = jwt.encode(payload, "wrong_secret_key", algorithm="HS256")

        result = await jwt_auth._validate_jwt(wrong_key_token)
        assert result is None


class TestJWTAuthMissingClaims:
    """Tests for JWT authentication with missing required claims."""

    @pytest.fixture
    def jwt_auth(self):
        """Create JWTAuth instance with test key."""
        with patch.dict(os.environ, {"JWT_SECRET_KEY": "test_secret_key_for_testing"}):
            from omni_mercury_engine.api.auth import JWTAuth

            return JWTAuth()

    @pytest.mark.asyncio
    async def test_missing_sub_claim_returns_none(self, jwt_auth: Any) -> None:
        """Test that tokens missing 'sub' claim return None."""
        from omni_mercury_engine.security import native_jwt as jwt

        # Token without 'sub' claim
        payload = {
            "username": "testuser",
            "roles": ["user"],
            "permissions": ["read"],
            "exp": datetime.now() + timedelta(hours=1),
            "iat": datetime.now(),
        }
        token = jwt.encode(payload, "test_secret_key_for_testing", algorithm="HS256")

        result = await jwt_auth._validate_jwt(token)
        assert result is None

    @pytest.mark.asyncio
    async def test_missing_exp_claim_returns_none(self, jwt_auth: Any) -> None:
        """Test that tokens missing 'exp' claim return None."""
        from omni_mercury_engine.security import native_jwt as jwt

        # Token without 'exp' claim
        payload = {
            "sub": "test_user",
            "username": "testuser",
            "roles": ["user"],
            "permissions": ["read"],
            "iat": datetime.now(),
        }
        token = jwt.encode(payload, "test_secret_key_for_testing", algorithm="HS256")

        result = await jwt_auth._validate_jwt(token)
        assert result is None


class TestJWTAuthValidToken:
    """Tests for JWT authentication with valid tokens."""

    @pytest.fixture
    def jwt_auth(self):
        """Create JWTAuth instance with test key."""
        with patch.dict(os.environ, {"JWT_SECRET_KEY": "test_secret_key_for_testing"}):
            from omni_mercury_engine.api.auth import JWTAuth

            return JWTAuth()

    @pytest.mark.asyncio
    async def test_valid_token_returns_user(self, jwt_auth: Any) -> None:
        """Test that valid tokens return User object."""
        from omni_mercury_engine.security import native_jwt as jwt

        payload = {
            "sub": "test_user_123",
            "username": "testuser",
            "email": "test@example.com",
            "roles": ["user", "analyst"],
            "permissions": ["read", "detect"],
            "exp": datetime.now() + timedelta(hours=1),
            "iat": datetime.now(),
        }
        token = jwt.encode(payload, "test_secret_key_for_testing", algorithm="HS256")

        result = await jwt_auth._validate_jwt(token)

        assert result is not None
        assert result.id == "test_user_123"
        assert result.username == "testuser"
        assert result.email == "test@example.com"
        assert "user" in result.roles
        assert "analyst" in result.roles

    @pytest.mark.asyncio
    async def test_create_and_validate_token_roundtrip(self, jwt_auth: Any) -> None:
        """Test creating and validating a token works correctly.

        Backed by Mercury's native JWT library — no third-party JWT
        package required.  Previously gated on ``importorskip('jwt')``
        which would silently skip in any environment lacking pyjwt;
        the native back-end makes the test deterministically runnable
        in every Mercury install.
        """
        from omni_mercury_engine.api.auth import JWTAuth

        # Create token
        token = JWTAuth.create_token(
            user_id="roundtrip_user",
            username="roundtrip_test",
            secret_key="test_secret_key_for_testing",  # noqa: S106  # nosec B106 - test only
            email="roundtrip@example.com",
            roles=["user"],
            permissions=["read", "detect"],
        )

        # Validate token
        result = await jwt_auth._validate_jwt(token)

        assert result is not None
        assert result.id == "roundtrip_user"
        assert result.username == "roundtrip_test"


class TestAPIKeyAuth:
    """Tests for API key authentication."""

    def test_api_key_store_create_and_retrieve(self) -> None:
        """Test creating and retrieving API keys."""
        from omni_mercury_engine.api.auth import APIKeyStore, Permission

        store = APIKeyStore()

        # Create key
        raw_key, api_key = store.create_key(
            name="test_key",
            user_id="test_user",
            permissions={Permission.READ, Permission.DETECT},
        )

        # Retrieve by raw key
        retrieved = store.get_by_key(raw_key)
        assert retrieved is not None
        assert retrieved.name == "test_key"
        assert retrieved.user_id == "test_user"

        # Retrieve by ID
        retrieved_by_id = store.get_by_id(api_key.key_id)
        assert retrieved_by_id is not None
        assert retrieved_by_id.key_id == api_key.key_id

    def test_api_key_store_revoke(self) -> None:
        """Test revoking API keys."""
        from omni_mercury_engine.api.auth import APIKeyStore

        store = APIKeyStore()
        raw_key, api_key = store.create_key(name="revoke_test", user_id="test_user")

        # Key should be active
        assert api_key.is_active is True

        # Revoke key
        result = store.revoke(api_key.key_id)
        assert result is True

        # Key should be inactive
        retrieved = store.get_by_id(api_key.key_id)
        assert retrieved is not None
        assert retrieved.is_active is False

    def test_api_key_expiration(self) -> None:
        """Test API key expiration check."""
        from omni_mercury_engine.api.auth import APIKeyStore

        store = APIKeyStore()

        # Create key that expires in 1 day
        _, api_key = store.create_key(name="expiring_key", user_id="test_user", expires_in_days=1)

        # Key should not be expired
        assert api_key.is_expired is False

        # Create key with no expiration
        _, no_expire_key = store.create_key(
            name="no_expire_key", user_id="test_user", expires_in_days=None
        )

        # Key should not be expired
        assert no_expire_key.is_expired is False


# Run with: pytest tests/security/test_jwt_auth.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestAMAMasterSeed:
    """Tests for shared HD master-seed sourcing via ``AMA_MASTER_SEED``.

    The seed makes AMA HD key derivation deterministic fleet-wide: every
    process with the same seed derives the same ``jwt_sign`` material, so
    HD-derived JWT keys verify across workers, replicas, and restarts.
    Without it the seed is per-process and the production path must warn.
    """

    _SEED_HEX = "ab" * 64  # 64 bytes, valid hex

    @staticmethod
    def _reset_singleton() -> None:
        import omni_mercury_engine.api.auth as auth_module

        auth_module._auth_key_manager = None

    def test_env_seed_makes_derivation_deterministic(self) -> None:
        """Two managers built from the same env seed derive identical material."""
        with patch.dict(os.environ, {"AMA_MASTER_SEED": self._SEED_HEX}, clear=False):
            from omni_mercury_engine.api.auth import (
                AuthKeyManager,
                _load_master_seed_from_env,
            )

            km1 = AuthKeyManager(master_seed=_load_master_seed_from_env())
            km2 = AuthKeyManager(master_seed=_load_master_seed_from_env())

            assert km1.seed_is_ephemeral is False
            assert km1.get_active_key_material("jwt_sign") == km2.get_active_key_material(
                "jwt_sign"
            )

    def test_ephemeral_managers_diverge(self) -> None:
        """Seedless managers derive different material (the documented hazard)."""
        from omni_mercury_engine.api.auth import AuthKeyManager

        km1 = AuthKeyManager()
        km2 = AuthKeyManager()

        assert km1.seed_is_ephemeral is True
        assert km1.get_active_key_material("jwt_sign") != km2.get_active_key_material("jwt_sign")

    def test_invalid_hex_seed_raises(self) -> None:
        """A malformed seed fails loudly instead of degrading to ephemeral."""
        with patch.dict(os.environ, {"AMA_MASTER_SEED": "not-hex!"}, clear=False):
            from omni_mercury_engine.api.auth import _load_master_seed_from_env

            with pytest.raises(ValueError, match="hex"):
                _load_master_seed_from_env()

    def test_short_seed_raises(self) -> None:
        """A seed below 32 decoded bytes is rejected."""
        with patch.dict(os.environ, {"AMA_MASTER_SEED": "ab" * 16}, clear=False):
            from omni_mercury_engine.api.auth import _load_master_seed_from_env

            with pytest.raises(ValueError, match="32 bytes"):
                _load_master_seed_from_env()

    def test_unset_seed_returns_none(self) -> None:
        """Unset/empty AMA_MASTER_SEED yields None (ephemeral fallback)."""
        with patch.dict(os.environ, {}, clear=True):
            from omni_mercury_engine.api.auth import _load_master_seed_from_env

            assert _load_master_seed_from_env() is None

    def test_whitespace_only_seed_raises(self) -> None:
        """Whitespace-only is malformed, not unset — never a silent fallback.

        A whitespace-only value means an operator intended to set a seed
        (empty mounted secret, templating failure); treating it as unset
        would silently downgrade production to per-process keys. A trailing
        newline on a *valid* seed is harmless (``bytes.fromhex`` ignores
        ASCII whitespace), so only the garbage case fails.
        """
        with patch.dict(os.environ, {"AMA_MASTER_SEED": " \n\t "}, clear=False):
            from omni_mercury_engine.api.auth import _load_master_seed_from_env

            with pytest.raises(ValueError, match="32 bytes"):
                _load_master_seed_from_env()

    def test_production_jwt_key_deterministic_across_processes(self) -> None:
        """With the env seed set, two fresh singletons yield the same JWT key.

        Resetting the module-level ``_auth_key_manager`` between
        constructions simulates two separate worker processes.
        """
        env = {"MERCURY_AGENT_ENV": "production", "AMA_MASTER_SEED": self._SEED_HEX}
        with patch.dict(os.environ, env, clear=True):
            from omni_mercury_engine.api.auth import JWTAuth

            try:
                self._reset_singleton()
                key_a = JWTAuth().secret_key
                self._reset_singleton()
                key_b = JWTAuth().secret_key
            finally:
                self._reset_singleton()

            assert key_a == key_b
            assert key_a is not None and len(key_a) > 0

    def test_production_seedless_derivation_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Seedless production HD derivation logs the per-process-key hazard."""
        with patch.dict(os.environ, {"MERCURY_AGENT_ENV": "production"}, clear=True):
            from omni_mercury_engine.api.auth import JWTAuth

            try:
                self._reset_singleton()
                with caplog.at_level("WARNING", logger="omni_mercury_engine.api.auth"):
                    auth = JWTAuth()
            finally:
                self._reset_singleton()

            assert auth.secret_key is not None
            assert "EPHEMERAL" in caplog.text
            assert "AMA_MASTER_SEED" in caplog.text


class TestProductionFlagAlignment:
    """The auth layer's production check honours the canonical MERCURY_ENV.

    ``_is_production_env`` mirrors ``api/server.py``: ``MERCURY_ENV`` wins
    whenever set (unknown values raise loudly); the legacy
    ``MERCURY_AGENT_ENV`` / ``ENV`` / ``ENVIRONMENT`` aliases apply only
    when it is unset. Pre-fix, ``MERCURY_ENV=production`` alone left
    ``JWTAuth`` on the dev fallback key — the hazard this class pins shut.
    """

    @staticmethod
    def _reset_singleton() -> None:
        import omni_mercury_engine.api.auth as auth_module

        auth_module._auth_key_manager = None

    def test_canonical_mercury_env_production_engages_hd_derivation(self) -> None:
        """MERCURY_ENV=production alone must NOT fall back to the dev key."""
        with patch.dict(os.environ, {"MERCURY_ENV": "production"}, clear=True):
            from omni_mercury_engine.api.auth import JWTAuth

            try:
                self._reset_singleton()
                auth = JWTAuth()
            finally:
                self._reset_singleton()

            assert auth.using_fallback is False
            assert auth.secret_key != JWTAuth._get_dev_fallback_key()
            assert auth.secret_key is not None and len(auth.secret_key) > 0

    def test_canonical_flag_wins_over_legacy_alias(self) -> None:
        """MERCURY_ENV=development overrides a legacy production alias."""
        env = {"MERCURY_ENV": "development", "MERCURY_AGENT_ENV": "production"}
        with patch.dict(os.environ, env, clear=True):
            from omni_mercury_engine.api.auth import JWTAuth

            JWTAuth._warned_about_fallback = True  # silence repeat warning
            auth = JWTAuth(allow_dev_fallback=True)

            assert auth.using_fallback is True

    def test_unknown_canonical_value_raises_loudly(self) -> None:
        """A typo'd MERCURY_ENV fails JWTAuth construction, not silently."""
        from omni_mercury_engine._env import MercuryProductionConfigError

        with patch.dict(os.environ, {"MERCURY_ENV": "prod"}, clear=True):
            from omni_mercury_engine.api.auth import JWTAuth

            with pytest.raises(MercuryProductionConfigError):
                JWTAuth()

    def test_api_key_salt_enforced_under_canonical_flag(self) -> None:
        """APIKeyStore.hash_key enforces the salt under MERCURY_ENV=production."""
        from omni_mercury_engine.api.auth import APIKeyStore

        if not APIKeyStore._HASH_SALT_IS_DEFAULT:
            pytest.skip("API_KEY_HASH_SALT configured in this environment")

        with (
            patch.dict(os.environ, {"MERCURY_ENV": "production"}, clear=True),
            pytest.raises(ValueError, match="API_KEY_HASH_SALT"),
        ):
            APIKeyStore.hash_key("any-key")
