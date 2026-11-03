# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AMA-Cryptography HMAC routing for Mercury Agent.

Thin Mercury-side adapter that surfaces AMA Cryptography's
ACVP-validated, constant-time, zero-third-party-dep HMAC primitives
to Mercury's JWT signing path so that ``HS256``, ``HS384``, and
``HS512`` JOSE algorithms run on the same C backend that already serves AMA's PQC
and HKDF stack in this deployment.

Routed primitives (AMA Cryptography, pinned v5.0.0)
--------------------------------------------
* ``ama_cryptography.pqc_backends.native_hmac_sha256(key, msg)``
  HMAC-SHA-256 (FIPS 198-1 / RFC 2104), ACVP-validated 150/150 vectors.
* ``ama_cryptography.pqc_backends.native_hmac_sha256_2(key, msg1, msg2)``
  Two-segment HMAC-SHA-256 — byte-identical to
  ``native_hmac_sha256(key, msg1 + msg2)`` but avoids materialising
  the concat in Python.  This is the JWT signing fast path
  (``b64(header) + "."`` and ``b64(payload)`` go in as two segments).
* ``ama_cryptography.pqc_backends.native_hmac_sha384(key, msg)``
  HMAC-SHA-384 (FIPS 198-1 / RFC 2104) — added in AMA v3.3.0.
* ``ama_cryptography.pqc_backends.native_hmac_sha512(key, msg)``
  HMAC-SHA-512 (FIPS 198-1 / RFC 2104).
* ``ama_cryptography.pqc_backends.native_hkdf_sha256(ikm, length, salt, info)``
  HKDF-SHA-256 (RFC 5869) -- the at-rest secret sealer's key derivation.
* ``ama_cryptography.pqc_backends.native_hmac_sha3_256(key, msg)`` and
  ``native_sha3_256(data)`` -- the tamper-evident audit chain's keyed hash
  and its torn-fragment digest.
* ``ama_cryptography.pqc_backends.native_pbkdf2_hmac_sha256(pw, salt, iters, n)``
  PBKDF2-HMAC-SHA-256 (SP 800-132) -- verification of legacy password rows.
* ``ama_cryptography.secure_memory.constant_time_compare(a, b)`` --
  ``ama_consttime_memcmp``, the constant-time equality every verifier uses.

Every one of these used to run on ``hashlib``/``hmac``, which on an
OpenSSL-linked CPython (the shipped ``python:3.14-slim`` image included)
dispatch to ``openssl_sha256`` / ``openssl_sha3_256`` / OpenSSL's HMAC and
``CRYPTO_memcmp``. The outputs are byte-identical by construction (the
algorithms are fixed by RFC 5869 / 2104 / FIPS 202 / SP 800-132 and the
adapter's tests pin equality against the stdlib on fixed vectors), so sealed
secrets, audit chains and stored password rows written before the routing
open and verify unchanged.

Fail-closed contract
--------------------
When AMA Cryptography is not installed, the C library is not loaded, or the
specific HMAC variant is unavailable, module import fails.  Mercury's JOSE
signing path must not silently fall through to stdlib HMAC while claiming the
AMA/PQC security posture.

References
----------
* AMA Cryptography ``ama_cryptography/pqc_backends.py`` (pinned v5.0.0) —
  upstream HMAC bindings.
* FIPS 198-1 — Keyed-Hash Message Authentication Code.
* RFC 2104 — HMAC.
* RFC 4231 — HMAC-SHA-2 test vectors.
* RFC 7518 §3.2 — JWS HS256 / HS384 / HS512.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from ama_cryptography.pqc_backends import (
    _HKDF_SHA2_NATIVE_AVAILABLE,
    _HMAC_SHA3_256_NATIVE_AVAILABLE,
    _HMAC_SHA256_NATIVE_AVAILABLE,
    _HMAC_SHA384_NATIVE_AVAILABLE,
    _HMAC_SHA512_NATIVE_AVAILABLE,
    _PBKDF2_NATIVE_AVAILABLE,
    _SHA3_256_NATIVE_AVAILABLE,
    native_hkdf_sha256,
    native_hmac_sha3_256,
    native_hmac_sha256,
    native_hmac_sha256_2,
    native_hmac_sha384,
    native_hmac_sha512,
    native_pbkdf2_hmac_sha256,
    native_sha3_256,
)
from ama_cryptography.secure_memory import constant_time_compare as _ama_constant_time_compare

HAS_AMA_HMAC_SHA256: bool = bool(_HMAC_SHA256_NATIVE_AVAILABLE)
HAS_AMA_HMAC_SHA384: bool = bool(_HMAC_SHA384_NATIVE_AVAILABLE)
HAS_AMA_HMAC_SHA512: bool = bool(_HMAC_SHA512_NATIVE_AVAILABLE)
HAS_AMA_HKDF_SHA256: bool = bool(_HKDF_SHA2_NATIVE_AVAILABLE)
HAS_AMA_HMAC_SHA3_256: bool = bool(_HMAC_SHA3_256_NATIVE_AVAILABLE)
HAS_AMA_SHA3_256: bool = bool(_SHA3_256_NATIVE_AVAILABLE)
HAS_AMA_PBKDF2_SHA256: bool = bool(_PBKDF2_NATIVE_AVAILABLE)

if not (HAS_AMA_HMAC_SHA256 and HAS_AMA_HMAC_SHA384 and HAS_AMA_HMAC_SHA512):
    raise ImportError(
        "AMA Cryptography native HMAC backend is mandatory for Mercury; "
        "build AMA v5.0.0 with native HMAC/PQC enabled."
    )

# Cached references to the AMA primitives.
_native_hmac_sha256: Callable[[bytes, bytes], bytes] = native_hmac_sha256
_native_hmac_sha256_2: Callable[[bytes, bytes, bytes], bytes] = native_hmac_sha256_2
_native_hmac_sha384: Callable[[bytes, bytes], bytes] = native_hmac_sha384
_native_hmac_sha512: Callable[[bytes, bytes], bytes] = native_hmac_sha512
_native_hkdf_sha256: Callable[..., bytearray] = native_hkdf_sha256
_native_hmac_sha3_256: Callable[[bytes, bytes], bytes] = native_hmac_sha3_256
_native_sha3_256: Callable[[bytes], bytes] = native_sha3_256
_native_pbkdf2_hmac_sha256: Callable[[bytes, bytes, int, int], bytearray] = (
    native_pbkdf2_hmac_sha256
)


def ama_hmac_sha256(key: bytes, msg: bytes) -> bytes:
    """Compute HMAC-SHA-256 via AMA's native C backend.

    Args:
        key: HMAC key (any length; keys >64 bytes are SHA-256-hashed
            first per RFC 2104 §2).
        msg: Message bytes to authenticate.

    Returns:
        32-byte HMAC-SHA-256 tag.

    Raises:
        RuntimeError: AMA Cryptography's native HMAC-SHA-256 binding was
            invalidated after import.
    """
    if not HAS_AMA_HMAC_SHA256:
        raise RuntimeError("AMA HMAC-SHA-256 unavailable")
    return _native_hmac_sha256(key, msg)


def ama_hmac_sha256_2(key: bytes, msg1: bytes, msg2: bytes) -> bytes:
    """Two-segment HMAC-SHA-256 — byte-identical to ``HMAC(key, msg1 || msg2)``.

    Avoids materialising the ``msg1 + msg2`` concat in Python.  This
    is the JWT signing fast path: pass ``header_segment + b"."`` and
    ``payload_segment``.

    Raises:
        RuntimeError: AMA Cryptography's two-segment HMAC-SHA-256 binding was
            invalidated after import.
    """
    if not HAS_AMA_HMAC_SHA256:
        raise RuntimeError("AMA HMAC-SHA-256 unavailable")
    return _native_hmac_sha256_2(key, msg1, msg2)


def ama_hmac_sha384(key: bytes, msg: bytes) -> bytes:
    """Compute HMAC-SHA-384 via AMA's native C backend (AMA v3.3.0+).

    Raises:
        RuntimeError: AMA Cryptography's native HMAC-SHA-384 binding was
            invalidated after import.
    """
    if not HAS_AMA_HMAC_SHA384:
        raise RuntimeError("AMA HMAC-SHA-384 unavailable")
    return _native_hmac_sha384(key, msg)


def ama_hmac_sha512(key: bytes, msg: bytes) -> bytes:
    """Compute HMAC-SHA-512 via AMA's native C backend.

    Raises:
        RuntimeError: AMA Cryptography's native HMAC-SHA-512 binding was
            invalidated after import.
    """
    if not HAS_AMA_HMAC_SHA512:
        raise RuntimeError("AMA HMAC-SHA-512 unavailable")
    return _native_hmac_sha512(key, msg)


def ama_hkdf_sha256(
    ikm: bytes, length: int, *, salt: bytes | None = None, info: bytes = b""
) -> bytes:
    """HKDF-SHA-256 (RFC 5869) via AMA's native C backend.

    ``salt=None`` is the RFC's "salt not provided" case (a zero-filled block
    of the hash length). Output is byte-identical to the stdlib formulation
    ``HMAC(salt, ikm)`` then ``T(i) = HMAC(PRK, T(i-1) || info || i)``.

    Raises:
        RuntimeError: AMA Cryptography's native HKDF-SHA-256 binding is
            unavailable. There is no stdlib fallback by design.
    """
    if not HAS_AMA_HKDF_SHA256:
        raise RuntimeError("AMA HKDF-SHA-256 unavailable")
    return bytes(_native_hkdf_sha256(ikm, length, salt=salt, info=info))


def ama_hmac_sha3_256(key: bytes, msg: bytes) -> bytes:
    """HMAC-SHA3-256 via AMA's native C backend.

    Raises:
        RuntimeError: AMA Cryptography's native HMAC-SHA3-256 binding is
            unavailable. There is no stdlib fallback by design.
    """
    if not HAS_AMA_HMAC_SHA3_256:
        raise RuntimeError("AMA HMAC-SHA3-256 unavailable")
    return _native_hmac_sha3_256(key, msg)


def ama_sha3_256(data: bytes) -> bytes:
    """SHA3-256 (FIPS 202) via AMA's native C backend, one shot.

    Raises:
        RuntimeError: AMA Cryptography's native SHA3-256 binding is
            unavailable. There is no stdlib fallback by design.
    """
    if not HAS_AMA_SHA3_256:
        raise RuntimeError("AMA SHA3-256 unavailable")
    return _native_sha3_256(data)


def ama_pbkdf2_hmac_sha256(password: bytes, salt: bytes, iterations: int, length: int) -> bytes:
    """PBKDF2-HMAC-SHA-256 (SP 800-132) via AMA's native C backend.

    Raises:
        RuntimeError: AMA Cryptography's native PBKDF2 binding is
            unavailable. There is no stdlib fallback by design.
    """
    if not HAS_AMA_PBKDF2_SHA256:
        raise RuntimeError("AMA PBKDF2-HMAC-SHA-256 unavailable")
    return bytes(_native_pbkdf2_hmac_sha256(password, salt, iterations, length))


def ama_constant_time_compare(a: bytes, b: bytes) -> bool:
    """Constant-time equality via AMA's native ``ama_consttime_memcmp``.

    Replaces ``hmac.compare_digest``, which is OpenSSL's ``CRYPTO_memcmp``
    on an OpenSSL-linked interpreter. Unequal lengths compare unequal, as
    with ``compare_digest``; AMA itself refuses to run without the native
    routine (its INVARIANT-7: no cryptographic fallbacks).
    """
    if len(a) != len(b):
        # Compare anyway so the call's cost does not reveal the length check.
        _ama_constant_time_compare(a, a)
        return False
    return bool(_ama_constant_time_compare(a, b))


def available() -> dict[str, bool | str]:
    """Return a diagnostic snapshot of AMA HMAC routing state.

    Useful for ``/health`` endpoints and audit logs.
    """
    flags = {
        "ama_hmac_sha256": HAS_AMA_HMAC_SHA256,
        "ama_hmac_sha384": HAS_AMA_HMAC_SHA384,
        "ama_hmac_sha512": HAS_AMA_HMAC_SHA512,
        "ama_hkdf_sha256": HAS_AMA_HKDF_SHA256,
        "ama_hmac_sha3_256": HAS_AMA_HMAC_SHA3_256,
        "ama_sha3_256": HAS_AMA_SHA3_256,
        "ama_pbkdf2_sha256": HAS_AMA_PBKDF2_SHA256,
    }
    missing = [name for name, ok in flags.items() if not ok]
    return {
        **flags,
        "reason": (
            "" if not missing else "AMA native primitive unavailable: " + ", ".join(missing)
        ),
    }


def _initialize() -> None:
    """Compatibility no-op: AMA HMAC is bound and validated at import."""


def _reinitialize_for_tests() -> None:
    """Re-run the import-time setup.  Test-only escape hatch."""
    _initialize()


__all__ = [
    "HAS_AMA_HKDF_SHA256",
    "HAS_AMA_HMAC_SHA3_256",
    "HAS_AMA_HMAC_SHA256",
    "HAS_AMA_HMAC_SHA384",
    "HAS_AMA_HMAC_SHA512",
    "HAS_AMA_PBKDF2_SHA256",
    "HAS_AMA_SHA3_256",
    "ama_constant_time_compare",
    "ama_hkdf_sha256",
    "ama_hmac_sha3_256",
    "ama_hmac_sha256",
    "ama_hmac_sha256_2",
    "ama_hmac_sha384",
    "ama_hmac_sha512",
    "ama_pbkdf2_hmac_sha256",
    "ama_sha3_256",
    "available",
]
