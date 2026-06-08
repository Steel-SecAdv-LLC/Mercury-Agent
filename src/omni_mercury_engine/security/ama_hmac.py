# Copyright (C) 2025 Steel Security Advisors LLC
"""AMA-Cryptography HMAC routing for Mercury Agent.

Thin Mercury-side adapter that surfaces AMA Cryptography's
ACVP-validated, constant-time, zero-third-party-dep HMAC primitives
to Mercury's JWT signing path so that ``HS256`` and ``HS512`` JOSE
algorithms run on the same C backend that already serves AMA's PQC
and HKDF stack in this deployment.

Routed primitives (AMA Cryptography v3.2.0+)
--------------------------------------------
* ``ama_cryptography.pqc_backends.native_hmac_sha256(key, msg)``
  HMAC-SHA-256 (FIPS 198-1 / RFC 2104), ACVP-validated 150/150 vectors.
* ``ama_cryptography.pqc_backends.native_hmac_sha256_2(key, msg1, msg2)``
  Two-segment HMAC-SHA-256 — byte-identical to
  ``native_hmac_sha256(key, msg1 + msg2)`` but avoids materialising
  the concat in Python.  This is the JWT signing fast path
  (``b64(header) + "."`` and ``b64(payload)`` go in as two segments).
* ``ama_cryptography.pqc_backends.native_hmac_sha512(key, msg)``
  HMAC-SHA-512 (FIPS 198-1 / RFC 2104).

Fail-closed contract
--------------------
When AMA Cryptography is not installed, the C library is not loaded, or the
specific HMAC variant is unavailable, module import fails.  Mercury's JOSE
signing path must not silently fall through to stdlib HMAC while claiming the
AMA/PQC security posture.

References
----------
* AMA Cryptography ``ama_cryptography/pqc_backends.py`` (v3.2.0+) —
  upstream HMAC bindings.
* FIPS 198-1 — Keyed-Hash Message Authentication Code.
* RFC 2104 — HMAC.
* RFC 4231 — HMAC-SHA-2 test vectors.
* RFC 7518 §3.2 — JWS HS256 / HS512.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from ama_cryptography.pqc_backends import (
    _HMAC_SHA256_NATIVE_AVAILABLE,
    _HMAC_SHA512_NATIVE_AVAILABLE,
    native_hmac_sha256,
    native_hmac_sha256_2,
    native_hmac_sha512,
)

HAS_AMA_HMAC_SHA256: bool = bool(_HMAC_SHA256_NATIVE_AVAILABLE)
HAS_AMA_HMAC_SHA512: bool = bool(_HMAC_SHA512_NATIVE_AVAILABLE)

if not (HAS_AMA_HMAC_SHA256 and HAS_AMA_HMAC_SHA512):
    raise ImportError(
        "AMA Cryptography native HMAC backend is mandatory for Mercury; "
        "build AMA v3.2.0 with native HMAC/PQC enabled."
    )

# Cached references to the AMA primitives.
_native_hmac_sha256: Callable[[bytes, bytes], bytes] = native_hmac_sha256
_native_hmac_sha256_2: Callable[[bytes, bytes, bytes], bytes] = native_hmac_sha256_2
_native_hmac_sha512: Callable[[bytes, bytes], bytes] = native_hmac_sha512


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


def ama_hmac_sha512(key: bytes, msg: bytes) -> bytes:
    """Compute HMAC-SHA-512 via AMA's native C backend.

    Raises:
        RuntimeError: AMA Cryptography's native HMAC-SHA-512 binding was
            invalidated after import.
    """
    if not HAS_AMA_HMAC_SHA512:
        raise RuntimeError("AMA HMAC-SHA-512 unavailable")
    return _native_hmac_sha512(key, msg)


def available() -> dict[str, bool | str]:
    """Return a diagnostic snapshot of AMA HMAC routing state.

    Useful for ``/health`` endpoints and audit logs.
    """
    return {
        "ama_hmac_sha256": HAS_AMA_HMAC_SHA256,
        "ama_hmac_sha512": HAS_AMA_HMAC_SHA512,
        "reason": (
            "" if (HAS_AMA_HMAC_SHA256 and HAS_AMA_HMAC_SHA512) else "AMA native HMAC unavailable"
        ),
    }


def _initialize() -> None:
    """Compatibility no-op: AMA HMAC is bound and validated at import."""


def _reinitialize_for_tests() -> None:
    """Re-run the import-time setup.  Test-only escape hatch."""
    _initialize()


__all__ = [
    "HAS_AMA_HMAC_SHA256",
    "HAS_AMA_HMAC_SHA512",
    "ama_hmac_sha256",
    "ama_hmac_sha256_2",
    "ama_hmac_sha512",
    "available",
]
