"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
AMA-Cryptography HMAC routing for Mercury Agent.

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

Fallback
--------
When AMA Cryptography is not installed, the C library is not loaded,
or the specific HMAC variant is unavailable, the corresponding
availability flag is ``False`` and callers (e.g.
:mod:`omni_mercury_engine.security.native_jwt`) fall through to
stdlib :mod:`hmac` over :mod:`hashlib`.  HMAC-SHA-2 is wire-format
defined by FIPS 198-1 / RFC 2104, so the AMA-routed and stdlib-routed
digests are byte-identical for the same ``(key, message)``; this
equivalence is locked by RFC 4231 known-answer vectors in
``tests/security/test_native_jwt_ama_routing.py``.

References
----------
* AMA Cryptography ``ama_cryptography/pqc_backends.py`` (v3.2.0+) —
  upstream HMAC bindings.
* FIPS 198-1 — Keyed-Hash Message Authentication Code.
* RFC 2104 — HMAC.
* RFC 4231 — HMAC-SHA-2 test vectors.
* RFC 7518 §3.2 — JWS HS256 / HS512.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

# Public availability flags — set at import time by ``_initialize``
# and re-evaluable via ``_reinitialize_for_tests`` to honour
# ``unittest.mock.patch`` swaps of the upstream AMA module.
HAS_AMA_HMAC_SHA256: bool = False
HAS_AMA_HMAC_SHA512: bool = False

# Reason string captured when AMA is unavailable; surfaced by
# :func:`available` for diagnostic logging.
_AMA_UNAVAILABLE_REASON: str = "ama_cryptography is not installed"

# Cached references to the AMA primitives — populated by
# :func:`_initialize` when AMA is importable.  ``None`` indicates the
# caller MUST fall back to stdlib.
_native_hmac_sha256: Callable[[bytes, bytes], bytes] | None = None
_native_hmac_sha256_2: Callable[[bytes, bytes, bytes], bytes] | None = None
_native_hmac_sha512: Callable[[bytes, bytes], bytes] | None = None


def ama_hmac_sha256(key: bytes, msg: bytes) -> bytes:
    """Compute HMAC-SHA-256 via AMA's native C backend.

    Args:
        key: HMAC key (any length; keys >64 bytes are SHA-256-hashed
            first per RFC 2104 §2).
        msg: Message bytes to authenticate.

    Returns:
        32-byte HMAC-SHA-256 tag.

    Raises:
        RuntimeError: AMA Cryptography is not installed or its native
            HMAC-SHA-256 binding is unavailable.  Callers that want a
            graceful stdlib fallback should check
            :data:`HAS_AMA_HMAC_SHA256` first.
    """
    if not HAS_AMA_HMAC_SHA256 or _native_hmac_sha256 is None:
        raise RuntimeError(f"AMA HMAC-SHA-256 unavailable: {_AMA_UNAVAILABLE_REASON}")
    return _native_hmac_sha256(key, msg)


def ama_hmac_sha256_2(key: bytes, msg1: bytes, msg2: bytes) -> bytes:
    """Two-segment HMAC-SHA-256 — byte-identical to ``HMAC(key, msg1 || msg2)``.

    Avoids materialising the ``msg1 + msg2`` concat in Python.  This
    is the JWT signing fast path: pass ``header_segment + b"."`` and
    ``payload_segment``.

    Raises:
        RuntimeError: AMA Cryptography is not installed or its
            two-segment HMAC-SHA-256 binding is unavailable.
    """
    if not HAS_AMA_HMAC_SHA256 or _native_hmac_sha256_2 is None:
        raise RuntimeError(f"AMA HMAC-SHA-256 unavailable: {_AMA_UNAVAILABLE_REASON}")
    return _native_hmac_sha256_2(key, msg1, msg2)


def ama_hmac_sha512(key: bytes, msg: bytes) -> bytes:
    """Compute HMAC-SHA-512 via AMA's native C backend.

    Raises:
        RuntimeError: AMA Cryptography is not installed or its native
            HMAC-SHA-512 binding is unavailable.
    """
    if not HAS_AMA_HMAC_SHA512 or _native_hmac_sha512 is None:
        raise RuntimeError(f"AMA HMAC-SHA-512 unavailable: {_AMA_UNAVAILABLE_REASON}")
    return _native_hmac_sha512(key, msg)


def available() -> dict[str, bool | str]:
    """Return a diagnostic snapshot of AMA HMAC routing state.

    Useful for ``/health`` endpoints and audit logs.
    """
    return {
        "ama_hmac_sha256": HAS_AMA_HMAC_SHA256,
        "ama_hmac_sha512": HAS_AMA_HMAC_SHA512,
        "reason": (
            "" if (HAS_AMA_HMAC_SHA256 and HAS_AMA_HMAC_SHA512) else _AMA_UNAVAILABLE_REASON
        ),
    }


def _initialize() -> None:
    """One-time AMA HMAC binding setup at module import.

    Side-effects ``HAS_AMA_HMAC_SHA256``, ``HAS_AMA_HMAC_SHA512`` and
    the cached primitive references.  Safe to re-run via
    :func:`_reinitialize_for_tests`.
    """
    global HAS_AMA_HMAC_SHA256, HAS_AMA_HMAC_SHA512
    global _native_hmac_sha256, _native_hmac_sha256_2, _native_hmac_sha512
    global _AMA_UNAVAILABLE_REASON

    HAS_AMA_HMAC_SHA256 = False
    HAS_AMA_HMAC_SHA512 = False
    _native_hmac_sha256 = None
    _native_hmac_sha256_2 = None
    _native_hmac_sha512 = None

    try:
        from ama_cryptography.pqc_backends import (
            _HMAC_SHA256_NATIVE_AVAILABLE,
            _HMAC_SHA512_NATIVE_AVAILABLE,
            native_hmac_sha256,
            native_hmac_sha256_2,
            native_hmac_sha512,
        )
    except ImportError as exc:
        # Catches both "ama_cryptography is not installed" AND
        # "ama_cryptography < v3.2.0 (missing native_hmac_sha256 binding)".
        _AMA_UNAVAILABLE_REASON = f"ama_cryptography v3.2.0+ not available: {exc}"
        return

    if bool(_HMAC_SHA256_NATIVE_AVAILABLE):
        _native_hmac_sha256 = native_hmac_sha256
        _native_hmac_sha256_2 = native_hmac_sha256_2
        HAS_AMA_HMAC_SHA256 = True
    if bool(_HMAC_SHA512_NATIVE_AVAILABLE):
        _native_hmac_sha512 = native_hmac_sha512
        HAS_AMA_HMAC_SHA512 = True

    if not (HAS_AMA_HMAC_SHA256 and HAS_AMA_HMAC_SHA512):
        _AMA_UNAVAILABLE_REASON = (
            "AMA Cryptography is installed but the native C HMAC backend "
            "is not fully loaded; build with "
            "`cmake -B build -DAMA_USE_NATIVE_PQC=ON && cmake --build build`"
        )
    else:
        _AMA_UNAVAILABLE_REASON = ""


def _reinitialize_for_tests() -> None:
    """Re-run the import-time setup.  Test-only escape hatch."""
    _initialize()


_initialize()


__all__ = [
    "HAS_AMA_HMAC_SHA256",
    "HAS_AMA_HMAC_SHA512",
    "ama_hmac_sha256",
    "ama_hmac_sha256_2",
    "ama_hmac_sha512",
    "available",
]
