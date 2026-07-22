# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Password hashing for user accounts.

Distinct from API-key hashing (``APIKeyStore.hash_key``), which hashes a
high-entropy machine-generated secret with a *shared* salt. User passwords are
low-entropy and human-chosen, so each one gets its own random per-password salt,
and the salt + parameters travel *with* the hash in a single self-describing
string (PHC-style ``algorithm$iterations$salt$hash``). That means a stored hash
can always be verified without any external configuration, and the work factor
can be raised over time without invalidating existing hashes.

PBKDF2-HMAC-SHA256 is used (stdlib-only, no extra dependency) at the OWASP-2024
iteration floor. The raw password is never stored or logged.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

__all__ = [
    "DEFAULT_ITERATIONS",
    "hash_password",
    "needs_rehash",
    "verify_password",
]

#: PBKDF2-HMAC-SHA256 iteration count. 600_000 meets the OWASP 2024 guidance for
#: SHA-256 password storage. Raising this only affects newly created hashes;
#: existing hashes carry their own iteration count and still verify.
DEFAULT_ITERATIONS = 600_000

_ALGORITHM = "pbkdf2_sha256"
_SALT_BYTES = 16


def hash_password(password: str, *, iterations: int = DEFAULT_ITERATIONS) -> str:
    """Hash a plaintext password into a self-describing storage string.

    Args:
        password: The plaintext password. Must be non-empty.
        iterations: PBKDF2 iteration count to use for this hash.

    Returns:
        A ``pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>`` string safe to
        persist. The salt is freshly random per call.

    Raises:
        ValueError: If ``password`` is empty.
    """
    if not password:
        raise ValueError("password must not be empty")
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{_ALGORITHM}${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Check a plaintext password against a stored hash in constant time.

    Args:
        password: The plaintext password to check.
        stored: A hash previously produced by :func:`hash_password`.

    Returns:
        ``True`` if the password matches, ``False`` otherwise (including for a
        malformed or unrecognised ``stored`` value — verification fails closed
        rather than raising).
    """
    try:
        algorithm, iterations_s, salt_hex, expected_hex = stored.split("$")
    except (ValueError, AttributeError):
        return False
    if algorithm != _ALGORITHM:
        return False
    try:
        iterations = int(iterations_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(expected_hex)
    except ValueError:
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(candidate, expected)


def needs_rehash(stored: str, *, iterations: int = DEFAULT_ITERATIONS) -> bool:
    """Report whether a stored hash should be upgraded on next successful login.

    A hash needs rehashing when its algorithm is unrecognised or its iteration
    count is below the current target, so the work factor can be raised
    transparently as ``DEFAULT_ITERATIONS`` grows.

    Args:
        stored: A hash previously produced by :func:`hash_password`.
        iterations: The current target iteration count.

    Returns:
        ``True`` if the caller should re-hash the password after verifying it.
    """
    try:
        algorithm, iterations_s, _salt_hex, _hash_hex = stored.split("$")
    except (ValueError, AttributeError):
        return True
    if algorithm != _ALGORITHM:
        return True
    try:
        return int(iterations_s) < iterations
    except ValueError:
        return True
