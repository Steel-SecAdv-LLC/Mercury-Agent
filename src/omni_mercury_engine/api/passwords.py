# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Password hashing for user accounts.

Distinct from API-key hashing (``APIKeyStore.hash_key``), which hashes a
high-entropy machine-generated secret with a *shared* salt. User passwords are
low-entropy and human-chosen, so each one gets its own random per-password salt,
and the salt + parameters travel *with* the hash in a single self-describing
string (PHC-style). That means a stored hash can always be verified without any
external configuration, and the work factor can be raised over time without
invalidating existing hashes.

The current algorithm is **scrypt** (stdlib ``hashlib.scrypt``, RFC 7914) — a
*memory-hard* KDF: each guess costs the attacker ~``128 * n * r`` bytes of RAM
(32 MiB at the default ``n=2**15, r=8``), which is what defeats GPU/ASIC
password-cracking rigs that shrug off pure-CPU PBKDF2. Parameters follow the
OWASP Password Storage Cheat Sheet's accepted scrypt settings
(``n=2**15, r=8, p=3``) and are tunable per deployment via
``MERCURY_SCRYPT_N`` / ``MERCURY_SCRYPT_R`` / ``MERCURY_SCRYPT_P``; run
``python scripts/calibrate_password_kdf.py`` on the target hardware to measure
and pick (see that script for the calibration method and reference numbers).

Legacy ``pbkdf2_sha256$...`` hashes (the previous algorithm) still verify, and
:func:`needs_rehash` reports them as upgradable so the auth service
transparently re-hashes to scrypt on the next successful login. The raw
password is never stored or logged.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets

__all__ = [
    "DEFAULT_ITERATIONS",
    "SCRYPT_N",
    "SCRYPT_P",
    "SCRYPT_R",
    "hash_password",
    "needs_rehash",
    "verify_password",
]


def _scrypt_param(name: str, default: int, *, minimum: int, maximum: int) -> int:
    """Read one scrypt parameter from the environment, bounded and validated.

    Out-of-range or malformed values keep the known-good default — a typo in
    a deployment manifest must weaken nothing.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if value < minimum or value > maximum:
        return default
    return value


#: scrypt cost parameters (OWASP-accepted setting: n=2^15, r=8, p=3 → 32 MiB
#: per guess, ~90 ms on a 2023-class x86 core; see
#: scripts/calibrate_password_kdf.py for measurement on *your* hardware).
#: ``n`` must be a power of two; the env override is validated as one.
SCRYPT_N = _scrypt_param("MERCURY_SCRYPT_N", 2**15, minimum=2**13, maximum=2**22)
if SCRYPT_N & (SCRYPT_N - 1):  # not a power of two — fall back to the default
    SCRYPT_N = 2**15
SCRYPT_R = _scrypt_param("MERCURY_SCRYPT_R", 8, minimum=4, maximum=32)
SCRYPT_P = _scrypt_param("MERCURY_SCRYPT_P", 3, minimum=1, maximum=16)

#: PBKDF2-HMAC-SHA256 iteration count for *legacy* hash verification and for
#: any caller that explicitly asks for the old algorithm. 600_000 meets the
#: OWASP 2024 guidance for SHA-256 password storage.
DEFAULT_ITERATIONS = 600_000

_SCRYPT_ALGORITHM = "scrypt"
_PBKDF2_ALGORITHM = "pbkdf2_sha256"
_SALT_BYTES = 16
_DERIVED_KEY_BYTES = 32


def _scrypt_maxmem(n: int, r: int) -> int:
    """Memory ceiling for ``hashlib.scrypt`` (the algorithm's need + headroom)."""
    return 128 * n * r + (2 * 1024 * 1024)


def hash_password(password: str) -> str:
    """Hash a plaintext password into a self-describing storage string.

    Args:
        password: The plaintext password. Must be non-empty.

    Returns:
        A ``scrypt$<n>$<r>$<p>$<salt_hex>$<hash_hex>`` string safe to persist.
        The salt is freshly random per call.

    Raises:
        ValueError: If ``password`` is empty.
    """
    if not password:
        raise ValueError("password must not be empty")
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        maxmem=_scrypt_maxmem(SCRYPT_N, SCRYPT_R),
        dklen=_DERIVED_KEY_BYTES,
    )
    return f"{_SCRYPT_ALGORITHM}${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${digest.hex()}"


def _verify_scrypt(password: str, fields: list[str]) -> bool:
    """Verify against a ``scrypt$n$r$p$salt$hash`` record (fail closed)."""
    try:
        n, r, p = int(fields[1]), int(fields[2]), int(fields[3])
        salt = bytes.fromhex(fields[4])
        expected = bytes.fromhex(fields[5])
    except ValueError:
        return False
    # Bound stored parameters before running the KDF: a tampered row must not
    # be able to turn verification into a memory bomb.
    if not (2**10 <= n <= 2**24) or n & (n - 1) or not (1 <= r <= 64) or not (1 <= p <= 64):
        return False
    try:
        candidate = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=n,
            r=r,
            p=p,
            maxmem=_scrypt_maxmem(n, r),
            dklen=len(expected),
        )
    except (ValueError, MemoryError):
        return False
    return hmac.compare_digest(candidate, expected)


def _verify_pbkdf2(password: str, fields: list[str]) -> bool:
    """Verify against a legacy ``pbkdf2_sha256$iter$salt$hash`` record."""
    try:
        iterations = int(fields[1])
        salt = bytes.fromhex(fields[2])
        expected = bytes.fromhex(fields[3])
    except ValueError:
        return False
    if not (1 <= iterations <= 10_000_000):
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(candidate, expected)


def verify_password(password: str, stored: str) -> bool:
    """Check a plaintext password against a stored hash in constant time.

    Args:
        password: The plaintext password to check.
        stored: A hash previously produced by :func:`hash_password` — either
            the current ``scrypt`` form or the legacy ``pbkdf2_sha256`` form.

    Returns:
        ``True`` if the password matches, ``False`` otherwise (including for a
        malformed or unrecognised ``stored`` value — verification fails closed
        rather than raising).
    """
    if not isinstance(stored, str):
        return False
    fields = stored.split("$")
    if fields[0] == _SCRYPT_ALGORITHM and len(fields) == 6:
        return _verify_scrypt(password, fields)
    if fields[0] == _PBKDF2_ALGORITHM and len(fields) == 4:
        return _verify_pbkdf2(password, fields)
    return False


def needs_rehash(stored: str) -> bool:
    """Report whether a stored hash should be upgraded on next successful login.

    A hash needs rehashing when its algorithm is not the current one (legacy
    PBKDF2, or anything unrecognised) or when its scrypt cost parameters are
    below the currently configured targets, so the work factor rises
    transparently as parameters are tuned up.

    Args:
        stored: A hash previously produced by :func:`hash_password`.

    Returns:
        ``True`` if the caller should re-hash the password after verifying it.
    """
    if not isinstance(stored, str):
        return True
    fields = stored.split("$")
    if fields[0] != _SCRYPT_ALGORITHM or len(fields) != 6:
        return True
    try:
        n, r, p = int(fields[1]), int(fields[2]), int(fields[3])
    except ValueError:
        return True
    return n < SCRYPT_N or r < SCRYPT_R or p < SCRYPT_P
