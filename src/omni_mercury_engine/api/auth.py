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
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import TYPE_CHECKING, Any

from omni_mercury_engine._env import is_production as _env_is_production

try:
    from ama_cryptography.key_management import (
        HDKeyDerivation,
        KeyRotationManager,
    )

    _AMA_KEY_MGMT_AVAILABLE = True
except ImportError:
    _AMA_KEY_MGMT_AVAILABLE = False

# ``pydantic`` is a core dependency (always installed); ``AuthConfig`` stays a
# real model regardless of the optional ``[api]`` extra.
from pydantic import BaseModel

# FastAPI is the optional ``[api]`` extra.  The framework-independent surface of
# this module — native JWT mint/verify (``JWTAuth.create_token`` +
# ``omni_mercury_engine.security.native_jwt``), the ``User`` / ``APIKey`` /
# ``Permission`` models, ``APIKeyStore`` and ``AuthKeyManager`` — must import and
# run with **no** web framework present.  Concretely, the in-process
# ``Eos_XVIII`` onboarding coordinator mints and validates a session token with
# no server.  The HTTP dependency-injection surface (``APIKeyAuth`` / ``JWTAuth``
# as FastAPI ``Depends`` objects, the ``require_*`` decorators, and the
# rate-limit middleware) genuinely needs FastAPI; when the extra is absent those
# names resolve to fail-closed placeholders that raise an actionable error only
# if they are actually constructed/called (never merely imported).
#
# The ``TYPE_CHECKING`` branch always imports the real FastAPI symbols so static
# analysis types every annotation correctly; the runtime ``else`` branch (which
# mypy skips) provides the placeholders, so no ``# type: ignore`` is needed.
if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi import HTTPException, Request, status
    from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

    from omni_mercury_engine.api.key_store import KeyStore
else:
    try:
        from fastapi import HTTPException, Request, status
        from fastapi.security import (
            APIKeyHeader,
            HTTPAuthorizationCredentials,
            HTTPBearer,
        )
    except ImportError:
        _FASTAPI_HINT = (
            "FastAPI is required for Mercury's HTTP auth surface; install the "
            "API extra:  pip install 'mercury-agent[api]'"
        )

        # Annotation-only names (this module uses ``from __future__ import
        # annotations``, so these are never evaluated at runtime).
        Request = Any
        HTTPAuthorizationCredentials = Any

        class _RequiresFastAPI:
            """Fail-closed stand-in for a FastAPI symbol used as a runtime value.

            Importing this module without FastAPI keeps the framework-independent
            auth primitives usable; constructing a FastAPI-only object (a
            ``Depends`` dependency, an ``HTTPException``) without FastAPI
            installed raises a clear, actionable error instead of an opaque
            ``NameError`` deep inside a request handler.
            """

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError(_FASTAPI_HINT)

        class _FastAPIStatusUnavailable:
            """Stand-in for ``fastapi.status`` — attribute access fails closed."""

            def __getattr__(self, _name: str) -> Any:
                raise RuntimeError(_FASTAPI_HINT)

        HTTPException = _RequiresFastAPI
        APIKeyHeader = _RequiresFastAPI
        HTTPBearer = _RequiresFastAPI
        status = _FastAPIStatusUnavailable()

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
            if _is_production_env():
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

    def list_by_user(self, user_id: str) -> list[APIKey]:
        """Return every key owned by ``user_id``, newest first.

        Mirrors :meth:`SqliteKeyStore.list_by_user`: metadata only (the raw
        key is never retained), ordered newest-first so the two backends are
        interchangeable behind the :class:`KeyStore` contract.
        """
        owned = [key for key in self._keys.values() if key.user_id == user_id]
        owned.sort(key=lambda k: k.created_at, reverse=True)
        return owned

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
            master_seed: HD derivation master seed. When None, a random
                per-process seed is generated — derived keys then differ
                across processes and restarts (``seed_is_ephemeral`` is
                set so callers can surface that hazard).
            rotation_period_days: Default key rotation period in days
        """
        if not _AMA_KEY_MGMT_AVAILABLE:
            raise RuntimeError(
                "AuthKeyManager requires AMA Cryptography key management. "
                "Install with: pip install 'ama-cryptography @ "
                "git+https://github.com/Steel-SecAdv-LLC/AMA-Cryptography.git'"
            )
        self.seed_is_ephemeral = master_seed is None
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

    @property
    def rotation_manager(self) -> KeyRotationManager:
        """The underlying AMA ``KeyRotationManager`` (shared rotation state).

        Exposed so components like the adaptive-posture controller can drive
        real key rotation through the same manager Mercury's purposes use,
        rather than being handed ``None`` and silently no-op'ing.
        """
        return self._rotation

    @property
    def hd_derivation(self) -> HDKeyDerivation:
        """The underlying AMA ``HDKeyDerivation`` (BIP32 key material)."""
        return self._hd

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


# Global API key store. The concrete backend is selected once, lazily, by
# ``key_store.build_key_store()``: in-memory by default (unchanged dev/test
# behaviour), durable SQLite when ``MERCURY_KEYSTORE_PATH`` is set. Lazy
# construction keeps the same-instance contract callers rely on while letting
# the environment pick the backend without importing ``key_store`` eagerly
# (which would create an import cycle: key_store imports the models from here).
_api_key_store: KeyStore | None = None
_api_key_store_lock = threading.Lock()

# Global AMA key manager instance
_auth_key_manager: AuthKeyManager | None = None
_auth_key_manager_lock = threading.Lock()


def get_api_key_store() -> KeyStore:
    """Get the process-wide API key store, constructing it on first use.

    The backend is chosen by ``key_store.build_key_store()`` from the
    environment; the constructed instance is cached so every caller shares one
    store for the process lifetime. Construction is lock-guarded
    (double-checked): two request threads racing the first build must never
    each construct a store — with the in-memory backend the loser's keys
    would silently vanish when the winner's instance is later returned.
    """
    global _api_key_store
    if _api_key_store is None:
        with _api_key_store_lock:
            if _api_key_store is None:
                from omni_mercury_engine.api.key_store import build_key_store

                _api_key_store = build_key_store()
    return _api_key_store


def _is_production_env() -> bool:
    """Decide production mode for the auth layer.

    Mirrors ``api/server.py``'s precedence exactly: the canonical
    ``MERCURY_ENV`` flag (``omni_mercury_engine._env``) wins whenever it
    is set — including raising :class:`MercuryProductionConfigError` on
    unknown values, so typos stay loud. Only when ``MERCURY_ENV`` is
    unset do the legacy aliases this module has honoured since v1.x
    (``MERCURY_AGENT_ENV``, ``ENV``, ``ENVIRONMENT``) apply.
    """
    if os.getenv("MERCURY_ENV", "").strip():
        return _env_is_production()
    return any(
        os.getenv(var, "").strip().lower() == "production"
        for var in ("MERCURY_AGENT_ENV", "ENV", "ENVIRONMENT")
    )


def _load_master_seed_from_env() -> bytes | None:
    """Load the AMA HD master seed from ``AMA_MASTER_SEED`` (hex-encoded).

    Returns ``None`` when the variable is unset or empty (callers then fall
    back to an ephemeral per-process seed). A set-but-malformed value raises
    ``ValueError`` instead of silently degrading to an ephemeral seed —
    a typo here would otherwise invalidate every token fleet-wide without
    any visible error. A whitespace-only value counts as malformed, not
    empty: ``bytes.fromhex`` ignores ASCII whitespace (so a trailing
    newline on a valid seed is harmless), leaving zero decoded bytes to
    fail the length check loudly rather than masking an operator's
    intent to configure a seed.

    Returns:
        Decoded seed bytes (>= 32 bytes; 64 recommended), or None.

    Raises:
        ValueError: The value is not valid hex or decodes to fewer than
            32 bytes (including whitespace-only values, which decode to
            zero bytes).
    """
    raw = os.getenv("AMA_MASTER_SEED")
    if not raw:
        return None
    try:
        seed = bytes.fromhex(raw)
    except ValueError as exc:
        raise ValueError(
            "AMA_MASTER_SEED must be a hex string (generate with: "
            "`openssl rand -hex 64`). Refusing to fall back to an ephemeral "
            "per-process seed on a malformed value — that would silently "
            "break token verification across workers and restarts."
        ) from exc
    if len(seed) < 32:
        raise ValueError(
            "AMA_MASTER_SEED must decode to at least 32 bytes "
            f"(64 recommended); got {len(seed)} bytes."
        )
    return seed


def get_auth_key_manager() -> AuthKeyManager:
    """Get or create the global AMA auth key manager instance.

    The HD master seed is sourced from the ``AMA_MASTER_SEED`` environment
    variable (hex; see :func:`_load_master_seed_from_env`). When it is set,
    every process derives identical key material, so HD-derived JWT signing
    keys verify across workers, replicas, and restarts. When unset, the
    seed is generated per process and derived keys are ephemeral.
    """
    global _auth_key_manager
    if _auth_key_manager is None:
        with _auth_key_manager_lock:
            if _auth_key_manager is None:
                _auth_key_manager = AuthKeyManager(master_seed=_load_master_seed_from_env())
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

    # Development fallback signing key. Generated ONCE per process (lazily) —
    # deliberately NOT a published constant. A hard-coded key in source control
    # (the old behavior) is CWE-798: any deployment that reached this dev path
    # by misconfiguration (forgot both MERCURY_ENV and JWT_SECRET_KEY) could
    # have its admin tokens minted by anyone reading the repo. A per-process
    # random key removes that: tokens are valid within one process (dev works),
    # cannot be forged from a known value, and do NOT verify across workers /
    # replicas / restarts — so a multi-replica production deployment that forgot
    # to set a key fails VISIBLY (auth breaks) instead of silently accepting
    # forged tokens. NEVER rely on this in production; set JWT_SECRET_KEY or
    # AMA_MASTER_SEED.
    _dev_fallback_key: str | None = None
    _dev_fallback_key_lock = threading.Lock()
    _warned_about_fallback = False

    @classmethod
    def _get_dev_fallback_key(cls) -> str:
        """Return this process's ephemeral dev signing key, creating it once.

        Creation is lock-guarded (double-checked): two threads racing the
        lazy init — e.g. a threaded ASGI server constructing two ``JWTAuth``
        instances concurrently — must never observe different keys, or a
        token signed by one instance would not verify with the other in the
        same process.
        """
        if cls._dev_fallback_key is None:
            with cls._dev_fallback_key_lock:
                if cls._dev_fallback_key is None:
                    cls._dev_fallback_key = secrets.token_hex(32)
        return cls._dev_fallback_key

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
            Key resolution order: explicit ``secret_key`` argument, then the
            ``JWT_SECRET_KEY`` environment variable. Production mode is
            decided by :func:`_is_production_env`: the canonical
            ``MERCURY_ENV`` flag wins when set (unknown values raise);
            the legacy ``MERCURY_AGENT_ENV`` / ``ENV`` / ``ENVIRONMENT``
            aliases apply only when ``MERCURY_ENV`` is unset. In
            production with no key set, the signing key is derived via AMA HD Key
            Management (``get_auth_key_manager()``, purpose ``jwt_sign``);
            a failed derivation raises ``ValueError``. The HD master seed
            is sourced from ``AMA_MASTER_SEED`` (hex, ``openssl rand -hex
            64``) — set it and derivation is deterministic fleet-wide.
            Without it the seed is generated per process, derived keys
            differ across workers/replicas/restarts, and a warning is
            logged; in that case set ``AMA_MASTER_SEED`` or
            ``JWT_SECRET_KEY`` (``openssl rand -hex 32``).
        """
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY")
        self.using_fallback = False

        if self.secret_key is None:
            # Canonical MERCURY_ENV first, legacy aliases second — the
            # same precedence api/server.py applies (see _is_production_env).
            if _is_production_env():
                # In production, derive JWT signing key from AMA Key Management
                try:
                    km = get_auth_key_manager()
                    derived = km.get_active_key_material("jwt_sign")
                    self.secret_key = derived.hex()
                    logger.info(
                        "JWT signing key derived from AMA HD Key Management (purpose=jwt_sign)"
                    )
                    if km.seed_is_ephemeral:
                        logger.warning(
                            "JWT signing key was derived from an EPHEMERAL per-process "
                            "HD master seed: tokens will not verify across workers, "
                            "replicas, or restarts. Set AMA_MASTER_SEED "
                            "(`openssl rand -hex 64`) for deterministic fleet-wide "
                            "derivation, or set JWT_SECRET_KEY directly."
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
                # Use an ephemeral per-process key for development only. Not a
                # published constant (see _get_dev_fallback_key): unforgeable
                # from source, and non-portable so a misconfigured multi-replica
                # deployment fails visibly rather than silently.
                self.secret_key = self._get_dev_fallback_key()
                self.using_fallback = True

                # Log warning only once per class (not per instance)
                if not JWTAuth._warned_about_fallback:
                    logger.warning(
                        "JWT_SECRET_KEY not set — using an ephemeral per-process dev "
                        "signing key. Tokens will not verify across workers/replicas/"
                        "restarts. Set JWT_SECRET_KEY (`openssl rand -hex 32`) or "
                        "AMA_MASTER_SEED before production."
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
        """Get rate limit key for request.

        Anonymous callers are keyed by their trusted-proxy-resolved client IP
        (see :mod:`omni_mercury_engine.api.client_ip`), never by a raw
        client-writable header.
        """
        if user:
            return f"user:{user.id}"
        from omni_mercury_engine.api.client_ip import resolve_client_ip

        client_ip = resolve_client_ip(
            request.client.host if request.client else None,
            request.headers.get("X-Forwarded-For"),
        )
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
