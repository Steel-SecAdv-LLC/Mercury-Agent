# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Authentication and authorization middleware for the API.

Example:
    Using API key authentication::

        from omni_mercury_engine.api.auth import require_api_key, APIKeyAuth

        @app.get("/protected")
        @require_api_key
        async def protected_endpoint(api_key: str = Depends(APIKeyAuth())):
            return {"message": "authenticated"}

    Using JWT authentication::

        from omni_mercury_engine.api.auth import JWTAuth, require_role

        @app.get("/admin")
        @require_role("admin")
        async def admin_endpoint(user: User = Depends(JWTAuth())):
            return {"user": user.username}
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import TYPE_CHECKING, Any

try:
    from ama_cryptography.key_management import (
        HDKeyDerivation,
        KeyRotationManager,
    )

    _AMA_KEY_MGMT_AVAILABLE = True
except ImportError:
    _AMA_KEY_MGMT_AVAILABLE = False

from fastapi import HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class AuthMethod(Enum):
    """Authentication methods."""

    API_KEY = "api_key"
    JWT = "jwt"
    BASIC = "basic"
    OAUTH2 = "oauth2"


class Permission(Enum):
    """Available permissions."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    DETECT = "detect"
    EXPORT = "export"


@dataclass
class User:
    """Authenticated user information.

    Attributes:
        id: Unique user identifier.
        username: Username.
        email: User email.
        roles: List of assigned roles.
        permissions: Set of permissions.
        metadata: Additional user metadata.
    """

    id: str
    username: str
    email: str | None = None
    roles: list[str] = field(default_factory=list)
    permissions: set[Permission] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    authenticated_at: datetime = field(default_factory=datetime.now)

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions or Permission.ADMIN in self.permissions

    def has_role(self, role: str) -> bool:
        """Check if user has specific role."""
        return role in self.roles or "admin" in self.roles


@dataclass
class APIKey:
    """API key information.

    Attributes:
        key_id: Unique key identifier.
        key_hash: Hashed key value.
        name: Human-readable key name.
        user_id: Associated user ID.
        permissions: Key permissions.
        created_at: Creation timestamp.
        expires_at: Expiration timestamp.
        last_used_at: Last usage timestamp.
        rate_limit: Requests per minute limit.
        is_active: Whether key is active.
    """

    key_id: str
    key_hash: str
    name: str
    user_id: str
    permissions: set[Permission] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    last_used_at: datetime | None = None
    rate_limit: int = 100
    is_active: bool = True

    @property
    def is_expired(self) -> bool:
        """Check if key has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


class AuthenticationError(Exception):
    """Authentication failed."""

    def __init__(self, message: str, code: str = "auth_failed") -> None:
        """Initialize the instance."""
        super().__init__(message)
        self.code = code


class AuthorizationError(Exception):
    """Authorization failed."""

    def __init__(self, message: str, required: str | None = None) -> None:
        """Initialize the instance."""
        super().__init__(message)
        self.required = required


class AuthProvider(ABC):
    """Abstract base class for authentication providers."""

    @abstractmethod
    async def authenticate(self, credentials: Any) -> User:
        """Authenticate credentials and return user.

        Args:
            credentials: Authentication credentials.

        Returns:
            Authenticated user.

        Raises:
            AuthenticationError: If authentication fails.
        """
        pass

    @abstractmethod
    async def validate_token(self, token: str) -> User | None:
        """Validate a token and return user if valid.

        Args:
            token: Token to validate.

        Returns:
            User if valid, None otherwise.
        """
        pass


class APIKeyStore:
    """In-memory API key store.

    In production, this should be backed by a database.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self._keys: dict[str, APIKey] = {}
        self._key_lookup: dict[str, str] = {}  # hash -> key_id

    # Salt for API key hashing - in production, MUST be set via environment.
    # Generate with: openssl rand -hex 32
    _HASH_SALT = os.getenv("API_KEY_HASH_SALT", "mercury-agent-api-key-salt-v1")
    _HASH_SALT_IS_DEFAULT = not os.getenv("API_KEY_HASH_SALT")
    # PBKDF2 iterations — 260 000 meets OWASP 2024 recommendation for SHA-256.
    _HASH_ITERATIONS = int(os.getenv("API_KEY_HASH_ITERATIONS", "260000"))

    @staticmethod
    def hash_key(key: str) -> str:
        """Hash an API key for storage using PBKDF2-HMAC-SHA256.

        Uses PBKDF2 (Password-Based Key Derivation Function 2) which is a
        computationally expensive hash function suitable for credential storage.
        This provides protection against brute-force and rainbow table attacks.

        Security Note:
            In production, set API_KEY_HASH_SALT environment variable to a
            secure random value. Generate with: `openssl rand -hex 32`
            Optionally adjust API_KEY_HASH_ITERATIONS (default: 260000).
        """
        if APIKeyStore._HASH_SALT_IS_DEFAULT:
            is_prod = os.getenv("MERCURY_AGENT_ENV", "").lower() == "production"
            if is_prod:
                raise ValueError(
                    "API_KEY_HASH_SALT environment variable is required in production. "
                    "Generate with: openssl rand -hex 32"
                )
            logger.warning(
                "Using default API key hash salt. Set API_KEY_HASH_SALT for production deployments."
            )
        return hashlib.pbkdf2_hmac(
            hash_name="sha256",
            password=key.encode(),
            salt=APIKeyStore._HASH_SALT.encode(),
            iterations=APIKeyStore._HASH_ITERATIONS,
        ).hex()

    def create_key(
        self,
        name: str,
        user_id: str,
        permissions: set[Permission] | None = None,
        expires_in_days: int | None = None,
        rate_limit: int = 100,
    ) -> tuple[str, APIKey]:
        """Create a new API key.

        Args:
            name: Key name.
            user_id: Associated user ID.
            permissions: Key permissions.
            expires_in_days: Days until expiration.
            rate_limit: Requests per minute limit.

        Returns:
            Tuple of (raw_key, APIKey object).
        """
        raw_key = secrets.token_urlsafe(32)
        key_hash = self.hash_key(raw_key)
        key_id = secrets.token_hex(8)

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            user_id=user_id,
            permissions=permissions or {Permission.READ, Permission.DETECT},
            expires_at=(
                datetime.now() + timedelta(days=expires_in_days) if expires_in_days else None
            ),
            rate_limit=rate_limit,
        )

        self._keys[key_id] = api_key
        self._key_lookup[key_hash] = key_id

        return raw_key, api_key

    def get_by_key(self, raw_key: str) -> APIKey | None:
        """Get API key by raw key value."""
        key_hash = self.hash_key(raw_key)
        key_id = self._key_lookup.get(key_hash)
        if key_id:
            return self._keys.get(key_id)
        return None

    def get_by_id(self, key_id: str) -> APIKey | None:
        """Get API key by ID."""
        return self._keys.get(key_id)

    def revoke(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id in self._keys:
            self._keys[key_id].is_active = False
            return True
        return False

    def update_last_used(self, key_id: str) -> None:
        """Update last used timestamp."""
        if key_id in self._keys:
            self._keys[key_id].last_used_at = datetime.now()


class AuthKeyManager:
    """AMA Key Management integration for Mercury's auth layer.

    Provides HD key derivation, key rotation, and lifecycle management
    for API keys, JWT signing keys, and audit trail signing keys via
    AMA Cryptography's ``HDKeyDerivation`` and ``KeyRotationManager``.

    Key Purposes:
        - ``api_key``:    API key derivation and rotation
        - ``jwt_sign``:   JWT signing key rotation
        - ``audit_sign``: Audit trail signing key rotation
    """

    # HD derivation purpose constants (BIP44-style)
    PURPOSE_API_KEY = 100
    PURPOSE_JWT_SIGN = 101
    PURPOSE_AUDIT_SIGN = 102

    def __init__(
        self,
        master_seed: bytes | None = None,
        rotation_period_days: int = 90,
    ) -> None:
        """Initialize the auth key manager.

        Args:
            master_seed: HD derivation master seed (generated if None)
            rotation_period_days: Default key rotation period in days
        """
        if not _AMA_KEY_MGMT_AVAILABLE:
            raise RuntimeError(
                "AuthKeyManager requires AMA Cryptography key management. "
                "Install with: pip install 'ama-cryptography @ "
                "git+https://github.com/Steel-SecAdv-LLC/AMA-Cryptography.git'"
            )
        self._hd = HDKeyDerivation(seed=master_seed)
        self._rotation = KeyRotationManager(
            rotation_period=timedelta(days=rotation_period_days),
        )
        self._key_index: dict[str, int] = {
            "api_key": 0,
            "jwt_sign": 0,
            "audit_sign": 0,
        }

        # Register initial keys for each purpose
        for purpose, purpose_id in [
            ("api_key", self.PURPOSE_API_KEY),
            ("jwt_sign", self.PURPOSE_JWT_SIGN),
            ("audit_sign", self.PURPOSE_AUDIT_SIGN),
        ]:
            key_id = f"{purpose}-0"
            path = f"m/{purpose_id}'/0'/0'/0'"
            self._rotation.register_key(
                key_id=key_id,
                purpose=purpose,
                derivation_path=path,
                expires_in=timedelta(days=rotation_period_days),
            )

    def derive_key(self, purpose: str, index: int | None = None) -> bytes:
        """Derive a key for the given purpose using HD derivation.

        Args:
            purpose: One of ``api_key``, ``jwt_sign``, ``audit_sign``
            index: Key index (uses current index if None)

        Returns:
            32-byte derived key
        """
        purpose_map = {
            "api_key": self.PURPOSE_API_KEY,
            "jwt_sign": self.PURPOSE_JWT_SIGN,
            "audit_sign": self.PURPOSE_AUDIT_SIGN,
        }
        purpose_id = purpose_map.get(purpose)
        if purpose_id is None:
            raise ValueError(f"Unknown key purpose: {purpose}")

        if index is None:
            index = self._key_index.get(purpose, 0)

        result: bytes = self._hd.derive_key(
            purpose=purpose_id,
            account=0,
            change=0,
            index=index,
        )
        return result

    def rotate_key(self, purpose: str) -> tuple[str, str]:
        """Rotate the key for the given purpose.

        Derives a new key via HD derivation, registers it with the
        rotation manager, and initiates the rotation. The old key
        remains in ROTATING state for a grace period.

        Args:
            purpose: One of ``api_key``, ``jwt_sign``, ``audit_sign``

        Returns:
            Tuple of (old_key_id, new_key_id)
        """
        current_index = self._key_index.get(purpose, 0)
        new_index = current_index + 1
        self._key_index[purpose] = new_index

        old_key_id = f"{purpose}-{current_index}"
        new_key_id = f"{purpose}-{new_index}"

        purpose_map = {
            "api_key": self.PURPOSE_API_KEY,
            "jwt_sign": self.PURPOSE_JWT_SIGN,
            "audit_sign": self.PURPOSE_AUDIT_SIGN,
        }
        purpose_id = purpose_map.get(purpose, 0)
        path = f"m/{purpose_id}'/0'/0'/{new_index}'"

        self._rotation.register_key(
            key_id=new_key_id,
            purpose=purpose,
            parent_id=old_key_id,
            derivation_path=path,
            expires_in=timedelta(days=self._rotation.rotation_period.days),
        )

        if old_key_id in self._rotation.keys:
            self._rotation.initiate_rotation(old_key_id, new_key_id)
            logger.info(f"Key rotation initiated: {old_key_id} → {new_key_id} (purpose={purpose})")

        return old_key_id, new_key_id

    def should_rotate(self, purpose: str) -> bool:
        """Check if the active key for a purpose needs rotation."""
        current_index = self._key_index.get(purpose, 0)
        key_id = f"{purpose}-{current_index}"
        result: bool = self._rotation.should_rotate(key_id)
        return result

    def get_active_key_material(self, purpose: str) -> bytes:
        """Get the current active key material for a purpose."""
        current_index = self._key_index.get(purpose, 0)
        return self.derive_key(purpose, current_index)

    def complete_rotation(self, purpose: str) -> None:
        """Complete rotation by deprecating the previous key."""
        current_index = self._key_index.get(purpose, 0)
        if current_index > 0:
            old_key_id = f"{purpose}-{current_index - 1}"
            self._rotation.complete_rotation(old_key_id)

    def revoke_key(self, purpose: str, index: int, reason: str = "compromised") -> None:
        """Revoke a specific key version."""
        key_id = f"{purpose}-{index}"
        self._rotation.revoke_key(key_id, reason=reason)
        logger.warning(f"Key revoked: {key_id} (reason={reason})")

    def get_rotation_status(self) -> dict[str, Any]:
        """Get status of all managed keys."""
        result: dict[str, Any] = self._rotation.export_metadata()
        return result


# Global API key store (in production, use dependency injection)
_api_key_store = APIKeyStore()

# Global AMA key manager instance
_auth_key_manager: AuthKeyManager | None = None


def get_api_key_store() -> APIKeyStore:
    """Get the API key store instance."""
    return _api_key_store


def get_auth_key_manager() -> AuthKeyManager:
    """Get or create the global AMA auth key manager instance."""
    global _auth_key_manager
    if _auth_key_manager is None:
        _auth_key_manager = AuthKeyManager()
    return _auth_key_manager


class APIKeyAuth:
    """API Key authentication dependency.

    Usage:
        @app.get("/protected")
        async def endpoint(user: User = Depends(APIKeyAuth())):
            return {"user": user.username}
    """

    def __init__(
        self,
        header_name: str = "X-API-Key",
        auto_error: bool = True,
    ):
        """Initialize the instance."""
        self.header_name = header_name
        self.auto_error = auto_error
        self.api_key_header = APIKeyHeader(
            name=header_name,
            auto_error=auto_error,
        )

    async def __call__(
        self,
        request: Request,
        api_key: str | None = None,
    ) -> User | None:
        """Authenticate request with API key.

        Args:
            request: FastAPI request.
            api_key: API key from header.

        Returns:
            Authenticated user.

        Raises:
            HTTPException: If authentication fails.
        """
        # Get key from header
        if api_key is None:
            api_key = request.headers.get(self.header_name)

        if not api_key:
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key required",
                    headers={"WWW-Authenticate": "ApiKey"},
                )
            return None

        # Validate key
        store = get_api_key_store()
        key_obj = store.get_by_key(api_key)

        if not key_obj:
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key",
                )
            return None

        if not key_obj.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key has been revoked",
            )

        if key_obj.is_expired:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key has expired",
            )

        # Update last used
        store.update_last_used(key_obj.key_id)

        # Create user object
        user = User(
            id=key_obj.user_id,
            username=f"api_key:{key_obj.name}",
            permissions=key_obj.permissions,
            metadata={"key_id": key_obj.key_id, "auth_method": "api_key"},
        )

        # Store user in request state
        request.state.user = user

        return user


class JWTAuth:
    """JWT Bearer token authentication dependency.

    Production-ready implementation backed by
    :mod:`omni_mercury_engine.security.native_jwt` — Mercury's pure-stdlib
    HS256 JWT library.  The native back-end retires the prior ``pyjwt``
    dependency and the upstream-disputed ``CVE-2025-45768`` /
    ``PYSEC-2025-183`` advisory from Mercury's audited supply chain
    without adding any new third-party library.  The encode/decode
    contract, exception types, and call sites are preserved unchanged.

    Usage:
        @app.get("/protected")
        async def endpoint(user: User = Depends(JWTAuth())):
            return {"user": user.username}
    """

    # Development fallback key - NEVER use in production
    _DEV_FALLBACK_KEY = "MERCURY_AGENT_DEV_FALLBACK_KEY_DO_NOT_USE_IN_PRODUCTION"
    _warned_about_fallback = False

    def __init__(
        self,
        secret_key: str | None = None,
        algorithm: str = "HS256",
        auto_error: bool = True,
        allow_dev_fallback: bool = True,
    ):
        """Initialize JWT authentication.

        Args:
            secret_key: JWT signing key (overrides environment variable)
            algorithm: JWT signing algorithm (default: HS256)
            auto_error: Raise HTTPException on auth failure
            allow_dev_fallback: Allow insecure fallback key for development

        Security Note:
            In production, always set JWT_SECRET_KEY environment variable.
            Generate a secure key with: `openssl rand -hex 32`

        Migration Note (v1.0 -> v2.0):
            JWT_SECRET_KEY is now required in production. If migrating from
            an older version, ensure you set this environment variable before
            deploying. See CHANGELOG.md for migration instructions.
        """
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY")
        self.using_fallback = False

        if self.secret_key is None:
            # Check if we're in a production environment
            is_production = os.getenv("MERCURY_AGENT_ENV", "").lower() == "production"
            is_production = is_production or os.getenv("ENV", "").lower() == "production"
            is_production = is_production or os.getenv("ENVIRONMENT", "").lower() == "production"

            if is_production:
                # In production, derive JWT signing key from AMA Key Management
                try:
                    km = get_auth_key_manager()
                    derived = km.get_active_key_material("jwt_sign")
                    self.secret_key = derived.hex()
                    logger.info(
                        "JWT signing key derived from AMA HD Key Management (purpose=jwt_sign)"
                    )
                except Exception as e:
                    raise ValueError(
                        "JWT_SECRET_KEY environment variable is required in production "
                        "and AMA HD key derivation failed. "
                        "Generate a secure random key (e.g., `openssl rand -hex 32`) and set "
                        "JWT_SECRET_KEY in your environment or .env file. "
                        f"HD derivation error: {e}"
                    ) from e
            elif allow_dev_fallback:
                # Use fallback key for development only
                self.secret_key = self._DEV_FALLBACK_KEY
                self.using_fallback = True

                # Log warning only once per class (not per instance)
                if not JWTAuth._warned_about_fallback:
                    logger.warning(
                        "JWT_SECRET_KEY not set — using insecure dev fallback key "
                        "(dev only; set JWT_SECRET_KEY before production)."
                    )
                    JWTAuth._warned_about_fallback = True
            else:
                raise ValueError(
                    "JWT_SECRET_KEY environment variable is required for JWT authentication. "
                    "Generate a secure random key (e.g., `openssl rand -hex 32`) and set "
                    "JWT_SECRET_KEY in your environment or .env file."
                )

        self.algorithm = algorithm
        self.auto_error = auto_error
        self.bearer = HTTPBearer(auto_error=auto_error)

    async def __call__(
        self,
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = None,
    ) -> User | None:
        """Authenticate request with JWT token.

        Args:
            request: FastAPI request.
            credentials: Bearer token credentials.

        Returns:
            Authenticated user.

        Raises:
            HTTPException: If authentication fails.
        """
        if credentials is None:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header[7:]
            else:
                if self.auto_error:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Bearer token required",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                return None
        else:
            token = credentials.credentials

        # Validate JWT using PyJWT with cryptographic signature verification
        user = await self._validate_jwt(token)
        if user is None:
            if self.auto_error:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token",
                )
            return None

        request.state.user = user
        return user

    async def _validate_jwt(self, token: str) -> User | None:
        """Validate JWT token using Mercury's native HS256 JWT module.

        The library is part of Mercury's own ``security`` package
        (``omni_mercury_engine.security.native_jwt``), so no
        third-party dependency is required.  See the module docstring
        for the security properties enforced (alg whitelisting,
        constant-time signature verification, temporal-claim coercion,
        ``alg: none`` rejection by construction).

        Token payload expected format::

            {
                "sub": "user_id",
                "username": "user_name",
                "email": "user@example.com",
                "roles": ["user"],
                "permissions": ["read", "detect"],
                "exp": 1234567890
            }
        """
        from omni_mercury_engine.security import native_jwt as jwt

        try:
            if self.secret_key is None:
                logger.error("JWT secret key is not configured")
                return None

            # Decode and validate the JWT
            payload = jwt.decode(
                token,
                self.secret_key or "",
                algorithms=[self.algorithm],
                options={
                    "require": ["exp", "sub"],  # Require expiration and subject
                    "verify_exp": True,
                    "verify_signature": True,
                },
            )

            # Extract user information from payload
            user_id = payload.get("sub")
            if not user_id:
                logger.warning("JWT missing subject claim")
                return None

            # Enforce maximum token lifetime (default 72 h) to limit
            # window of exposure for long-lived tokens.
            max_token_age_s = int(os.getenv("JWT_MAX_TOKEN_AGE_HOURS", "72")) * 3600
            iat = payload.get("iat")
            if iat is not None:
                issued_at = datetime.fromtimestamp(float(iat))
                age = (datetime.now() - issued_at).total_seconds()
                if age > max_token_age_s:
                    logger.warning(
                        "JWT token exceeds maximum age (%d s > %d s)", int(age), max_token_age_s
                    )
                    return None

            # Parse permissions from payload
            permission_names = payload.get("permissions", ["read"])
            permissions = set()
            for perm_name in permission_names:
                try:
                    permissions.add(Permission(perm_name))
                except ValueError:
                    logger.warning(f"Unknown permission in JWT: {perm_name}")

            return User(
                id=user_id,
                username=payload.get("username", f"user_{user_id}"),
                email=payload.get("email"),
                roles=payload.get("roles", ["user"]),
                permissions=permissions or {Permission.READ},
                metadata={"auth_method": "jwt", "token_id": payload.get("jti")},
            )

        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            # Sanitize error message to prevent log injection via crafted tokens
            safe_msg = str(e).replace("\n", " ").replace("\r", " ")
            logger.warning("Invalid JWT token: %s", safe_msg)
            return None
        except (KeyError, TypeError, ValueError) as e:
            # Malformed payload - missing required fields or wrong types
            safe_msg = str(e).replace("\n", " ").replace("\r", " ")
            logger.warning("JWT payload malformed: %s: %s", type(e).__name__, safe_msg)
            return None
        except Exception as e:
            # Unexpected errors - log at error level for investigation
            safe_msg = str(e).replace("\n", " ").replace("\r", " ")
            logger.error("Unexpected JWT validation error: %s: %s", type(e).__name__, safe_msg)
            return None

    @staticmethod
    def create_token(
        user_id: str,
        username: str,
        secret_key: str,
        email: str | None = None,
        roles: list[str] | None = None,
        permissions: list[str] | None = None,
        algorithm: str = "HS256",
        expires_in_hours: int = 24,
    ) -> str:
        """Create a new JWT token.

        Args:
            user_id: Unique user identifier
            username: User's username
            secret_key: Secret key for signing
            email: User's email (optional)
            roles: User's roles (default: ["user"])
            permissions: User's permissions (default: ["read"])
            algorithm: Signing algorithm
            expires_in_hours: Token validity duration

        Returns:
            Signed JWT token string
        """
        from omni_mercury_engine.security import native_jwt as jwt

        payload = {
            "sub": user_id,
            "username": username,
            "roles": roles or ["user"],
            "permissions": permissions or ["read"],
            "iat": datetime.now(),
            "exp": datetime.now() + timedelta(hours=expires_in_hours),
            "jti": secrets.token_hex(16),  # Unique token ID
        }

        if email:
            payload["email"] = email

        result: str = jwt.encode(payload, secret_key, algorithm=algorithm)
        return result


class RequestRateLimiter:
    """Request-aware rate limiter wrapper.

    Wraps the unified RateLimiter with FastAPI Request support. Uses the consolidated rate limiting
    module for actual implementation.
    """

    def __init__(
        self,
        requests_per_minute: int = 100,
        burst_size: int = 20,
    ):
        """Initialize the instance."""
        from omni_mercury_engine.security.rate_limiting import RateLimiter as UnifiedRateLimiter

        self._limiter = UnifiedRateLimiter(
            requests_per_minute=requests_per_minute,
            burst_size=burst_size,
        )
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size

    def _get_key(self, request: Request, user: User | None = None) -> str:
        """Get rate limit key for request."""
        if user:
            return f"user:{user.id}"
        # Fall back to IP
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"

    def is_allowed(
        self,
        request: Request,
        user: User | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Check if request is allowed under rate limit.

        Args:
            request: FastAPI request.
            user: Authenticated user (optional).

        Returns:
            Tuple of (allowed, rate_limit_info).
        """
        key = self._get_key(request, user)
        info = self._limiter.check(key)

        return info.allowed, {
            "limit": info.limit,
            "remaining": info.remaining,
            "reset": info.reset_at,
        }


# Backward-compatible alias
RateLimiter = RequestRateLimiter

# Global rate limiter
_rate_limiter = RequestRateLimiter()


def get_rate_limiter() -> RequestRateLimiter:
    """Get the rate limiter instance."""
    return _rate_limiter


def require_permission(permission: Permission) -> Callable[..., Any]:
    """Decorator to require specific permission.

    Args:
        permission: Required permission.

    Returns:
        Decorator function.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            request = kwargs.get("request")
            if request and hasattr(request.state, "user"):
                user = request.state.user
                if not user.has_permission(permission):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Permission required: {permission.value}",
                    )
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_role(role: str) -> Callable[..., Any]:
    """Decorator to require specific role.

    Args:
        role: Required role.

    Returns:
        Decorator function.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            request = kwargs.get("request")
            if request and hasattr(request.state, "user"):
                user = request.state.user
                if not user.has_role(role):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Role required: {role}",
                    )
            return await func(*args, **kwargs)

        return wrapper

    return decorator


async def rate_limit_middleware(
    request: Request,
    call_next: Callable[..., Any],
) -> Any:
    """Rate limiting middleware.

    Args:
        request: FastAPI request.
        call_next: Next middleware/handler.

    Returns:
        Response.
    """
    limiter = get_rate_limiter()
    user = getattr(request.state, "user", None) if hasattr(request, "state") else None

    allowed, info = limiter.is_allowed(request, user)

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(info["limit"]),
                "X-RateLimit-Remaining": str(info["remaining"]),
                "X-RateLimit-Reset": str(info["reset"]),
            },
        )

    response = await call_next(request)

    # Add rate limit headers
    response.headers["X-RateLimit-Limit"] = str(info["limit"])
    response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
    response.headers["X-RateLimit-Reset"] = str(info["reset"])

    return response


class AuthConfig(BaseModel):
    """Authentication configuration."""

    enabled: bool = False
    methods: list[AuthMethod] = [AuthMethod.API_KEY]
    jwt_secret: str | None = None
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst: int = 20
