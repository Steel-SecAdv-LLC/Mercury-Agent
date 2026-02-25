"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Comprehensive tests for api/auth.py module.

Covers all error paths, security hardening, and authentication flows:
- API key creation, retrieval, revocation, expiration
- PBKDF2 hashing with 260k iterations
- Production salt enforcement
- JWT max token age enforcement
- User permissions and roles
- Rate limiter integration
- Decorator-based permission/role enforcement
- AuthConfig validation
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

# TODO: install fastapi in CI for full test coverage
pytest.importorskip("fastapi")

from omni_mercury_engine.api.auth import (
    APIKey,
    APIKeyAuth,
    APIKeyStore,
    AuthConfig,
    AuthenticationError,
    AuthMethod,
    AuthorizationError,
    JWTAuth,
    Permission,
    User,
    get_api_key_store,
    get_rate_limiter,
    require_permission,
    require_role,
)

# =============================================================================
# User Model Tests
# =============================================================================


class TestUser:
    """Tests for User dataclass."""

    def test_user_creation_defaults(self):
        """Test User creation with default values."""
        user = User(id="u1", username="alice")
        assert user.id == "u1"
        assert user.username == "alice"
        assert user.email is None
        assert user.roles == []
        assert user.permissions == set()
        assert isinstance(user.authenticated_at, datetime)

    def test_user_has_permission_direct(self):
        """Test direct permission check."""
        user = User(id="u1", username="alice", permissions={Permission.READ})
        assert user.has_permission(Permission.READ) is True
        assert user.has_permission(Permission.WRITE) is False

    def test_user_admin_has_all_permissions(self):
        """Test admin user implicitly has all permissions."""
        user = User(id="u1", username="admin", permissions={Permission.ADMIN})
        assert user.has_permission(Permission.READ) is True
        assert user.has_permission(Permission.WRITE) is True
        assert user.has_permission(Permission.DELETE) is True
        assert user.has_permission(Permission.DETECT) is True
        assert user.has_permission(Permission.EXPORT) is True

    def test_user_has_role_direct(self):
        """Test direct role check."""
        user = User(id="u1", username="alice", roles=["analyst"])
        assert user.has_role("analyst") is True
        assert user.has_role("admin") is False

    def test_user_admin_role_grants_all_roles(self):
        """Test admin role implicitly grants all roles."""
        user = User(id="u1", username="admin", roles=["admin"])
        assert user.has_role("analyst") is True
        assert user.has_role("viewer") is True


# =============================================================================
# APIKey Model Tests
# =============================================================================


class TestAPIKey:
    """Tests for APIKey dataclass."""

    def test_api_key_not_expired_no_expiry(self):
        """Test key without expiry is never expired."""
        key = APIKey(
            key_id="k1",
            key_hash="hash",
            name="test",
            user_id="u1",
            expires_at=None,
        )
        assert key.is_expired is False

    def test_api_key_not_expired_future(self):
        """Test key with future expiry is not expired."""
        key = APIKey(
            key_id="k1",
            key_hash="hash",
            name="test",
            user_id="u1",
            expires_at=datetime.now() + timedelta(days=30),
        )
        assert key.is_expired is False

    def test_api_key_expired_past(self):
        """Test key with past expiry is expired."""
        key = APIKey(
            key_id="k1",
            key_hash="hash",
            name="test",
            user_id="u1",
            expires_at=datetime.now() - timedelta(hours=1),
        )
        assert key.is_expired is True


# =============================================================================
# Exception Tests
# =============================================================================


class TestAuthExceptions:
    """Tests for authentication/authorization exceptions."""

    def test_authentication_error(self):
        """Test AuthenticationError attributes."""
        err = AuthenticationError("bad creds", code="invalid_key")
        assert str(err) == "bad creds"
        assert err.code == "invalid_key"

    def test_authentication_error_default_code(self):
        """Test AuthenticationError default code."""
        err = AuthenticationError("failed")
        assert err.code == "auth_failed"

    def test_authorization_error(self):
        """Test AuthorizationError attributes."""
        err = AuthorizationError("forbidden", required="admin")
        assert str(err) == "forbidden"
        assert err.required == "admin"


# =============================================================================
# APIKeyStore Tests
# =============================================================================


class TestAPIKeyStore:
    """Tests for APIKeyStore with PBKDF2 hashing."""

    def test_hash_key_deterministic(self):
        """Test that hashing the same key produces the same result."""
        h1 = APIKeyStore.hash_key("test-key-123")
        h2 = APIKeyStore.hash_key("test-key-123")
        assert h1 == h2

    def test_hash_key_different_keys_differ(self):
        """Test that different keys produce different hashes."""
        h1 = APIKeyStore.hash_key("key-a")
        h2 = APIKeyStore.hash_key("key-b")
        assert h1 != h2

    def test_hash_key_uses_260k_iterations(self):
        """Test that default hash iterations is 260000 (OWASP 2024)."""
        assert APIKeyStore._HASH_ITERATIONS == 260000

    def test_hash_key_production_requires_salt(self):
        """Test that production mode requires API_KEY_HASH_SALT."""
        with patch.dict(
            os.environ,
            {"MERCURY_AGENT_ENV": "production"},
            clear=False,
        ):
            # Ensure the default salt flag is True (no custom salt set)
            original = APIKeyStore._HASH_SALT_IS_DEFAULT
            APIKeyStore._HASH_SALT_IS_DEFAULT = True
            try:
                with pytest.raises(ValueError, match="API_KEY_HASH_SALT"):
                    APIKeyStore.hash_key("any-key")
            finally:
                APIKeyStore._HASH_SALT_IS_DEFAULT = original

    def test_create_key_returns_raw_and_object(self):
        """Test create_key returns raw key string and APIKey object."""
        store = APIKeyStore()
        raw_key, api_key = store.create_key(name="test", user_id="u1")
        assert isinstance(raw_key, str)
        assert len(raw_key) > 20  # token_urlsafe(32) is ~43 chars
        assert isinstance(api_key, APIKey)
        assert api_key.name == "test"
        assert api_key.user_id == "u1"
        assert api_key.is_active is True

    def test_create_key_default_permissions(self):
        """Test that default permissions are READ and DETECT."""
        store = APIKeyStore()
        _, api_key = store.create_key(name="test", user_id="u1")
        assert Permission.READ in api_key.permissions
        assert Permission.DETECT in api_key.permissions

    def test_create_key_custom_permissions(self):
        """Test custom permission assignment."""
        store = APIKeyStore()
        _, api_key = store.create_key(
            name="test",
            user_id="u1",
            permissions={Permission.ADMIN},
        )
        assert Permission.ADMIN in api_key.permissions

    def test_create_key_with_expiry(self):
        """Test key creation with expiry."""
        store = APIKeyStore()
        _, api_key = store.create_key(name="test", user_id="u1", expires_in_days=7)
        assert api_key.expires_at is not None
        assert api_key.is_expired is False

    def test_get_by_key_found(self):
        """Test retrieving key by raw key value."""
        store = APIKeyStore()
        raw_key, _ = store.create_key(name="test", user_id="u1")
        retrieved = store.get_by_key(raw_key)
        assert retrieved is not None
        assert retrieved.name == "test"

    def test_get_by_key_not_found(self):
        """Test retrieving non-existent key returns None."""
        store = APIKeyStore()
        assert store.get_by_key("nonexistent-key") is None

    def test_get_by_id_found(self):
        """Test retrieving key by ID."""
        store = APIKeyStore()
        _, api_key = store.create_key(name="test", user_id="u1")
        retrieved = store.get_by_id(api_key.key_id)
        assert retrieved is not None
        assert retrieved.key_id == api_key.key_id

    def test_get_by_id_not_found(self):
        """Test retrieving non-existent ID returns None."""
        store = APIKeyStore()
        assert store.get_by_id("nonexistent") is None

    def test_revoke_existing_key(self):
        """Test revoking an existing key."""
        store = APIKeyStore()
        _, api_key = store.create_key(name="test", user_id="u1")
        assert store.revoke(api_key.key_id) is True
        assert store.get_by_id(api_key.key_id).is_active is False

    def test_revoke_nonexistent_key(self):
        """Test revoking non-existent key returns False."""
        store = APIKeyStore()
        assert store.revoke("nonexistent") is False

    def test_update_last_used(self):
        """Test updating last used timestamp."""
        store = APIKeyStore()
        _, api_key = store.create_key(name="test", user_id="u1")
        assert api_key.last_used_at is None
        store.update_last_used(api_key.key_id)
        assert api_key.last_used_at is not None

    def test_update_last_used_nonexistent(self):
        """Test updating non-existent key is a no-op."""
        store = APIKeyStore()
        store.update_last_used("nonexistent")  # Should not raise


# =============================================================================
# APIKeyAuth Dependency Tests
# =============================================================================


class TestAPIKeyAuth:
    """Tests for API key authentication FastAPI dependency."""

    def _make_request(self, api_key_header: str | None = None):
        """Create a mock FastAPI Request."""
        request = MagicMock()
        request.headers = {}
        if api_key_header:
            request.headers["X-API-Key"] = api_key_header
        request.state = MagicMock()
        return request

    @pytest.mark.asyncio
    async def test_missing_api_key_auto_error(self):
        """Test HTTPException raised when key missing and auto_error=True."""
        from fastapi import HTTPException

        auth = APIKeyAuth(auto_error=True)
        request = self._make_request()
        with pytest.raises(HTTPException) as exc_info:
            await auth(request)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_api_key_no_auto_error(self):
        """Test None returned when key missing and auto_error=False."""
        auth = APIKeyAuth(auto_error=False)
        request = self._make_request()
        result = await auth(request)
        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_api_key_auto_error(self):
        """Test HTTPException raised for invalid key."""
        from fastapi import HTTPException

        auth = APIKeyAuth(auto_error=True)
        request = self._make_request(api_key_header="invalid-key-value")
        with pytest.raises(HTTPException) as exc_info:
            await auth(request)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_api_key_returns_user(self):
        """Test valid key returns User and sets request.state.user."""
        store = get_api_key_store()
        raw_key, api_key = store.create_key(name="valid_test", user_id="test_user_1")
        auth = APIKeyAuth(auto_error=True)
        request = self._make_request(api_key_header=raw_key)

        user = await auth(request, api_key=raw_key)
        assert user is not None
        assert user.id == "test_user_1"
        assert "api_key" in user.metadata.get("auth_method", "")

    @pytest.mark.asyncio
    async def test_revoked_api_key_raises(self):
        """Test HTTPException raised for revoked key."""
        from fastapi import HTTPException

        store = get_api_key_store()
        raw_key, api_key = store.create_key(name="revoked_test", user_id="test_user_2")
        store.revoke(api_key.key_id)

        auth = APIKeyAuth(auto_error=True)
        request = self._make_request()
        with pytest.raises(HTTPException) as exc_info:
            await auth(request, api_key=raw_key)
        assert exc_info.value.status_code == 401
        assert "revoked" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_expired_api_key_raises(self):
        """Test HTTPException raised for expired key."""
        from fastapi import HTTPException

        store = get_api_key_store()
        raw_key, api_key = store.create_key(
            name="expired_test", user_id="test_user_3", expires_in_days=0
        )
        # Force expiry to the past
        api_key.expires_at = datetime.now() - timedelta(hours=1)

        auth = APIKeyAuth(auto_error=True)
        request = self._make_request()
        with pytest.raises(HTTPException) as exc_info:
            await auth(request, api_key=raw_key)
        assert exc_info.value.status_code == 401
        assert "expired" in str(exc_info.value.detail).lower()


# =============================================================================
# JWT Max Token Age Tests
# =============================================================================


class TestJWTMaxTokenAge:
    """Tests for JWT maximum token age enforcement."""

    @pytest.fixture
    def jwt_auth(self):
        """Create JWTAuth instance with test key."""
        with patch.dict(os.environ, {"JWT_SECRET_KEY": "test-jwt-key-for-max-age"}):
            return JWTAuth()

    @pytest.mark.asyncio
    async def test_old_token_rejected(self, jwt_auth):
        """Test that tokens exceeding max age are rejected."""
        jwt = pytest.importorskip("jwt")

        payload = {
            "sub": "old_user",
            "username": "old_test",
            "roles": ["user"],
            "permissions": ["read"],
            "exp": datetime.now() + timedelta(hours=1),
            "iat": datetime.now() - timedelta(hours=100),  # Issued 100h ago
        }
        token = jwt.encode(payload, "test-jwt-key-for-max-age", algorithm="HS256")

        with patch.dict(os.environ, {"JWT_MAX_TOKEN_AGE_HOURS": "72"}):
            result = await jwt_auth._validate_jwt(token)
        assert result is None

    @pytest.mark.asyncio
    async def test_recent_token_accepted(self, jwt_auth):
        """Test that recently issued tokens are accepted."""
        jwt = pytest.importorskip("jwt")

        payload = {
            "sub": "recent_user",
            "username": "recent_test",
            "roles": ["user"],
            "permissions": ["read"],
            "exp": datetime.now() + timedelta(hours=24),
            "iat": datetime.now() - timedelta(hours=1),  # Issued 1h ago
        }
        token = jwt.encode(payload, "test-jwt-key-for-max-age", algorithm="HS256")

        with patch.dict(os.environ, {"JWT_MAX_TOKEN_AGE_HOURS": "72"}):
            result = await jwt_auth._validate_jwt(token)
        assert result is not None
        assert result.id == "recent_user"

    @pytest.mark.asyncio
    async def test_token_without_iat_accepted(self, jwt_auth):
        """Test that tokens without iat claim skip age check."""
        jwt = pytest.importorskip("jwt")

        payload = {
            "sub": "no_iat_user",
            "username": "no_iat_test",
            "roles": ["user"],
            "permissions": ["read"],
            "exp": datetime.now() + timedelta(hours=1),
            # No iat claim
        }
        token = jwt.encode(payload, "test-jwt-key-for-max-age", algorithm="HS256")

        result = await jwt_auth._validate_jwt(token)
        assert result is not None


# =============================================================================
# JWT Token Creation Tests
# =============================================================================


class TestJWTTokenCreation:
    """Tests for JWT token creation."""

    def test_create_token_success(self):
        """Test creating a JWT token."""
        jwt = pytest.importorskip("jwt")
        test_key = "test-secret"

        token = JWTAuth.create_token(
            user_id="u1",
            username="alice",
            secret_key=test_key,
            email="alice@example.com",
            roles=["user", "analyst"],
            permissions=["read", "detect"],
        )

        assert isinstance(token, str)
        # Decode to verify structure
        decoded = jwt.decode(token, test_key, algorithms=["HS256"])
        assert decoded["sub"] == "u1"
        assert decoded["username"] == "alice"
        assert decoded["email"] == "alice@example.com"
        assert decoded["roles"] == ["user", "analyst"]
        assert decoded["permissions"] == ["read", "detect"]
        assert "jti" in decoded
        assert "iat" in decoded
        assert "exp" in decoded

    def test_create_token_default_values(self):
        """Test token creation with default roles/permissions."""
        jwt = pytest.importorskip("jwt")
        test_key = "test-secret"

        token = JWTAuth.create_token(
            user_id="u2",
            username="bob",
            secret_key=test_key,
        )

        decoded = jwt.decode(token, test_key, algorithms=["HS256"])
        assert decoded["roles"] == ["user"]
        assert decoded["permissions"] == ["read"]
        assert "email" not in decoded

    def test_create_token_requires_pyjwt(self):
        """Test that missing PyJWT raises ImportError."""
        test_key = "test-secret"
        with patch.dict("sys.modules", {"jwt": None}), pytest.raises(ImportError, match="PyJWT"):
            JWTAuth.create_token(
                user_id="u3",
                username="carol",
                secret_key=test_key,
            )


# =============================================================================
# Permission and Role Decorator Tests
# =============================================================================


class TestRequirePermission:
    """Tests for require_permission decorator."""

    @pytest.mark.asyncio
    async def test_permission_granted(self):
        """Test that user with correct permission passes."""

        @require_permission(Permission.READ)
        async def protected_endpoint(request=None):
            return {"ok": True}

        request = MagicMock()
        user = User(
            id="u1",
            username="alice",
            permissions={Permission.READ},
        )
        request.state.user = user

        result = await protected_endpoint(request=request)
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_permission_denied(self):
        """Test that user without permission gets 403."""
        from fastapi import HTTPException

        @require_permission(Permission.ADMIN)
        async def admin_endpoint(request=None):
            return {"ok": True}

        request = MagicMock()
        user = User(
            id="u1",
            username="alice",
            permissions={Permission.READ},
        )
        request.state.user = user

        with pytest.raises(HTTPException) as exc_info:
            await admin_endpoint(request=request)
        assert exc_info.value.status_code == 403


class TestRequireRole:
    """Tests for require_role decorator."""

    @pytest.mark.asyncio
    async def test_role_granted(self):
        """Test that user with correct role passes."""

        @require_role("analyst")
        async def analyst_endpoint(request=None):
            return {"ok": True}

        request = MagicMock()
        user = User(id="u1", username="alice", roles=["analyst"])
        request.state.user = user

        result = await analyst_endpoint(request=request)
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_role_denied(self):
        """Test that user without role gets 403."""
        from fastapi import HTTPException

        @require_role("admin")
        async def admin_endpoint(request=None):
            return {"ok": True}

        request = MagicMock()
        user = User(id="u1", username="alice", roles=["user"])
        request.state.user = user

        with pytest.raises(HTTPException) as exc_info:
            await admin_endpoint(request=request)
        assert exc_info.value.status_code == 403


# =============================================================================
# AuthConfig Tests
# =============================================================================


class TestAuthConfig:
    """Tests for AuthConfig Pydantic model."""

    def test_default_config(self):
        """Test AuthConfig default values."""
        config = AuthConfig()
        assert config.enabled is False
        assert config.methods == [AuthMethod.API_KEY]
        assert config.jwt_secret is None
        assert config.jwt_algorithm == "HS256"
        assert config.jwt_expiration_hours == 24
        assert config.rate_limit_enabled is True
        assert config.rate_limit_requests_per_minute == 100
        assert config.rate_limit_burst == 20

    def test_custom_config(self):
        """Test AuthConfig with custom values."""
        test_jwt_secret = "my-secret"
        config = AuthConfig(
            enabled=True,
            methods=[AuthMethod.JWT, AuthMethod.API_KEY],
            jwt_secret=test_jwt_secret,
            jwt_expiration_hours=48,
        )
        assert config.enabled is True
        assert len(config.methods) == 2
        assert config.jwt_secret == test_jwt_secret
        assert config.jwt_expiration_hours == 48


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for authentication enums."""

    def test_auth_method_values(self):
        """Test all AuthMethod values exist."""
        assert AuthMethod.API_KEY.value == "api_key"
        assert AuthMethod.JWT.value == "jwt"
        assert AuthMethod.BASIC.value == "basic"
        assert AuthMethod.OAUTH2.value == "oauth2"

    def test_permission_values(self):
        """Test all Permission values exist."""
        assert Permission.READ.value == "read"
        assert Permission.WRITE.value == "write"
        assert Permission.DELETE.value == "delete"
        assert Permission.ADMIN.value == "admin"
        assert Permission.DETECT.value == "detect"
        assert Permission.EXPORT.value == "export"


# =============================================================================
# Global Store / Limiter Accessor Tests
# =============================================================================


class TestGlobalAccessors:
    """Tests for global singleton accessors."""

    def test_get_api_key_store_returns_same_instance(self):
        """Test global API key store is a singleton."""
        store1 = get_api_key_store()
        store2 = get_api_key_store()
        assert store1 is store2

    def test_get_rate_limiter_returns_same_instance(self):
        """Test global rate limiter is a singleton."""
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()
        assert limiter1 is limiter2
