# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Time-based one-time passwords (TOTP) for optional two-factor auth.

A dependency-free RFC 6238 (TOTP) / RFC 4226 (HOTP) implementation over the
stdlib ``hmac``/``hashlib`` primitives, so 2FA needs no third-party package.
The secret is a base32 string compatible with standard authenticator apps
(Google Authenticator, Aegis, 1Password, …); :func:`provisioning_uri` renders
the ``otpauth://`` URI those apps consume as a QR code.

Verification allows a small step window (default ±1) to tolerate clock skew,
and compares in constant time.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import struct
import time
from urllib.parse import quote, urlencode

__all__ = [
    "generate_secret",
    "generate_totp",
    "provisioning_uri",
    "verify_totp",
]

_DEFAULT_DIGITS = 6
_DEFAULT_PERIOD = 30
_SECRET_BYTES = 20  # 160-bit secret, the RFC 4226 recommendation.


def generate_secret() -> str:
    """Generate a fresh base32 TOTP secret for a new enrollment.

    Returns:
        An un-padded uppercase base32 string suitable for storage and for
        embedding in a :func:`provisioning_uri`.
    """
    return base64.b32encode(secrets.token_bytes(_SECRET_BYTES)).decode("ascii").rstrip("=")


def _hotp(secret: str, counter: int, digits: int) -> str:
    """Compute the RFC 4226 HOTP value for a secret and counter."""
    # Re-pad the base32 secret to a multiple of 8 chars before decoding.
    padding = "=" * (-len(secret) % 8)
    key = base64.b32decode(secret.upper() + padding)
    message = struct.pack(">Q", counter)
    digest = hmac.new(key, message, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    truncated = struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF
    return str(truncated % (10**digits)).zfill(digits)


def generate_totp(
    secret: str,
    *,
    at: float | None = None,
    digits: int = _DEFAULT_DIGITS,
    period: int = _DEFAULT_PERIOD,
) -> str:
    """Compute the current TOTP code for a secret.

    Args:
        secret: The base32 secret from :func:`generate_secret`.
        at: Unix time to compute the code for; defaults to now. Injected in
            tests for determinism.
        digits: Number of digits in the code.
        period: Time step in seconds.

    Returns:
        The zero-padded numeric code.
    """
    now = time.time() if at is None else at
    counter = int(now // period)
    return _hotp(secret, counter, digits)


def verify_totp(
    secret: str,
    code: str,
    *,
    at: float | None = None,
    digits: int = _DEFAULT_DIGITS,
    period: int = _DEFAULT_PERIOD,
    window: int = 1,
) -> bool:
    """Verify a submitted TOTP code, tolerating a small clock-skew window.

    Args:
        secret: The base32 secret the code should match.
        code: The user-submitted code.
        at: Unix time to verify against; defaults to now.
        digits: Number of digits expected.
        period: Time step in seconds.
        window: Number of steps before/after to also accept (clock skew).

    Returns:
        ``True`` if ``code`` matches any accepted step, ``False`` otherwise
        (including for a non-numeric or wrong-length code).
    """
    candidate = code.strip()
    if not candidate.isdigit() or len(candidate) != digits:
        return False
    now = time.time() if at is None else at
    counter = int(now // period)
    for step in range(-window, window + 1):
        expected = _hotp(secret, counter + step, digits)
        if hmac.compare_digest(expected, candidate):
            return True
    return False


def provisioning_uri(secret: str, account_name: str, issuer: str) -> str:
    """Build the ``otpauth://`` URI an authenticator app renders as a QR code.

    Args:
        secret: The base32 secret being enrolled.
        account_name: Label for the account (typically the user's email).
        issuer: Human-readable service name shown in the authenticator app.

    Returns:
        A standard ``otpauth://totp/...`` URI.
    """
    label = quote(f"{issuer}:{account_name}")
    params = urlencode(
        {
            "secret": secret,
            "issuer": issuer,
            "algorithm": "SHA1",
            "digits": _DEFAULT_DIGITS,
            "period": _DEFAULT_PERIOD,
        }
    )
    return f"otpauth://totp/{label}?{params}"
