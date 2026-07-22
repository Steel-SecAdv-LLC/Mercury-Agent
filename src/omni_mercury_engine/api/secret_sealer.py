# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""At-rest sealing for account secrets (TOTP seeds) over SecureDataHandler.

A TOTP seed is a *shared symmetric secret*: anyone who reads it can mint valid
second-factor codes forever, so storing it plaintext makes the identity
database a single point of 2FA compromise (backup copies included). This
module seals those seeds through the existing quantum-resistant
:class:`~omni_mercury_engine.security.encryption.SecureDataHandler` API
(AES-256-GCM via the AMA backend — a 256-bit key keeps a ≥128-bit
post-quantum margin) with the owning account id bound in as AAD, so a sealed
value can neither be read nor swapped onto another account's row without
failing authentication.

**Key material must be stable** for a durable store — sealing under a random
per-process key would orphan every seed on restart, which is *worse* than
plaintext. Resolution order:

1. ``MERCURY_DATA_ENC_KEY`` — 64 hex chars (``openssl rand -hex 32``).
   Explicit operator key; malformed values raise instead of degrading.
2. ``AMA_MASTER_SEED`` — the fleet HD seed already used for JWT signing keys;
   the sealing key is derived from it via HKDF-SHA256 with a fixed,
   purpose-scoped info string, so every worker derives the same key and the
   TOTP sealing key is domain-separated from every other derived key.
3. Neither set — an ephemeral process key. Safe for the in-memory identity
   store (its rows die with the process anyway); for a durable store the
   sealer instead reports itself unavailable so callers keep secrets in
   whatever form they already have rather than bricking 2FA on the next
   restart. The builder logs exactly which mode was chosen.

The sealed wire format is ``enc$v1$<base64(nonce || tag || ciphertext)>``;
:meth:`SecretSealer.unseal` passes legacy plaintext values through unchanged
so existing rows keep working and are upgraded opportunistically (see
:func:`migrate_plaintext_totp_secrets` and the auth-service write paths).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
from typing import TYPE_CHECKING

from omni_mercury_engine.security.encryption import SecureDataHandler

if TYPE_CHECKING:
    from omni_mercury_engine.api.identity_store import IdentityStore

logger = logging.getLogger(__name__)

__all__ = [
    "DATA_ENC_KEY_ENV",
    "SealedSecretError",
    "SecretSealer",
    "build_secret_sealer",
    "migrate_plaintext_totp_secrets",
]

DATA_ENC_KEY_ENV = "MERCURY_DATA_ENC_KEY"
_PREFIX = "enc$v1$"
_HKDF_INFO = b"mercury-agent/totp-at-rest/v1"


class SealedSecretError(ValueError):
    """A sealed value failed to open (tampered, wrong key, or wrong row)."""


def _hkdf_sha256(seed: bytes, info: bytes, length: int = 32) -> bytes:
    """RFC 5869 HKDF-SHA256 (extract with a fixed salt, then expand).

    Args:
        seed: Input keying material (the fleet master seed).
        info: Context string providing domain separation.
        length: Output length in bytes (≤ 8160).

    Returns:
        Derived key bytes.
    """
    prk = hmac.new(b"mercury-hkdf-salt-v1", seed, hashlib.sha256).digest()
    output = b""
    block = b""
    counter = 1
    while len(output) < length:
        block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
        output += block
        counter += 1
    return output[:length]


class SecretSealer:
    """Seals/opens short account secrets with AAD-bound at-rest encryption."""

    def __init__(self, key: bytes, *, key_is_stable: bool) -> None:
        """Wire the sealer to its key material.

        Args:
            key: 32-byte AES-256 key.
            key_is_stable: Whether ``key`` survives process restarts. Callers
                with durable storage must only *write* sealed values when this
                is ``True`` (reading is always safe).
        """
        self._handler = SecureDataHandler(enable_quantum_resistant=False, at_rest_key=key)
        self.key_is_stable = key_is_stable

    @staticmethod
    def is_sealed(value: str) -> bool:
        """Whether ``value`` carries the sealed-envelope prefix."""
        return value.startswith(_PREFIX)

    def seal(self, plaintext: str, *, aad: str) -> str:
        """Seal ``plaintext`` bound to ``aad`` (the owning account id).

        Args:
            plaintext: The secret to protect.
            aad: Context string authenticated with the ciphertext.

        Returns:
            The ``enc$v1$...`` sealed form.
        """
        envelope = self._handler.encrypt_at_rest(plaintext, aad=aad.encode("utf-8"))
        return _PREFIX + base64.b64encode(envelope).decode("ascii")

    def unseal(self, stored: str, *, aad: str) -> str:
        """Open a stored value; legacy plaintext passes through unchanged.

        Args:
            stored: Either an ``enc$v1$...`` sealed value or a legacy
                plaintext secret from before sealing shipped.
            aad: The same context string used at :meth:`seal` time.

        Returns:
            The plaintext secret.

        Raises:
            SealedSecretError: If a sealed value fails authentication (any
                tampering, the wrong key, or a swapped row) — never silently
                returns garbage.
        """
        if not self.is_sealed(stored):
            return stored
        try:
            envelope = base64.b64decode(stored[len(_PREFIX) :], validate=True)
            plaintext = self._handler.decrypt_at_rest(envelope, aad=aad.encode("utf-8"))
        except SealedSecretError:
            raise
        except Exception as exc:
            raise SealedSecretError("sealed secret failed to open (tampered or wrong key)") from exc
        return plaintext.decode("utf-8")


def build_secret_sealer(*, store_is_durable: bool) -> SecretSealer:
    """Resolve key material from the environment and build the sealer.

    Args:
        store_is_durable: Whether sealed values will outlive this process.
            With no stable key configured and a durable store, the returned
            sealer is marked non-stable so write paths keep secrets unsealed
            (readable forever) instead of sealing them under a key that dies
            with the process.

    Returns:
        A ready :class:`SecretSealer`.

    Raises:
        ValueError: If ``MERCURY_DATA_ENC_KEY`` is set but not 64 hex chars —
            a typo must fail loudly, not silently downgrade protection.
    """
    raw = os.getenv(DATA_ENC_KEY_ENV, "").strip()
    if raw:
        try:
            key = bytes.fromhex(raw)
        except ValueError as exc:
            raise ValueError(
                f"{DATA_ENC_KEY_ENV} must be hex (generate with: openssl rand -hex 32)"
            ) from exc
        if len(key) != 32:
            raise ValueError(f"{DATA_ENC_KEY_ENV} must decode to exactly 32 bytes")
        return SecretSealer(key, key_is_stable=True)

    master = os.getenv("AMA_MASTER_SEED", "").strip()
    if master:
        try:
            seed = bytes.fromhex(master)
        except ValueError:
            seed = b""
        if len(seed) >= 32:
            return SecretSealer(_hkdf_sha256(seed, _HKDF_INFO), key_is_stable=True)
        # A malformed AMA_MASTER_SEED is auth.py's contract to reject loudly;
        # here we just refuse to derive from it.
        logger.warning(
            "AMA_MASTER_SEED is set but unusable for at-rest key derivation; "
            "TOTP sealing falls back to the no-stable-key path"
        )

    if store_is_durable:
        logger.warning(
            "No stable at-rest key configured (%s or AMA_MASTER_SEED) with a durable "
            "identity store: NEW TOTP secrets will be stored unsealed. Set %s "
            "(openssl rand -hex 32) and re-run the migration sweep to seal them.",
            DATA_ENC_KEY_ENV,
            DATA_ENC_KEY_ENV,
        )
        return SecretSealer(secrets.token_bytes(32), key_is_stable=False)
    # In-memory store: rows die with the process, so an ephemeral key loses
    # nothing and still keeps secrets unreadable in heap dumps.
    return SecretSealer(secrets.token_bytes(32), key_is_stable=True)


def migrate_plaintext_totp_secrets(store: IdentityStore, sealer: SecretSealer) -> int:
    """Seal every plaintext TOTP secret currently in ``store``.

    Safe to run repeatedly (already-sealed rows are skipped) and a no-op when
    the sealer's key is not stable — migrating onto a process-lifetime key
    would break every enrolled account at the next restart.

    Args:
        store: An identity store (uses ``iter_accounts`` / ``update_account``).
        sealer: The active sealer.

    Returns:
        How many accounts were migrated.
    """
    if not sealer.key_is_stable:
        return 0
    migrated = 0
    for account in store.iter_accounts():
        secret = account.totp_secret
        if secret and not sealer.is_sealed(secret):
            account.totp_secret = sealer.seal(secret, aad=account.id)
            store.update_account(account)
            migrated += 1
    if migrated:
        logger.info("sealed %d previously plaintext TOTP secret(s)", migrated)
    return migrated
