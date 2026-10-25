# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mercury's sovereign emergency crypto tier — repository-owned, stdlib-only.

AMA Cryptography is Mercury's authoritative backend and is mandatory at
engine import (``omni_mercury_engine._pqc_gate``). This module exists for the
state that "will never happen": AMA present at import but failing at
*runtime* — a FIPS error-state lockout (AMA v5 inhibits output on every
surface once its power-on self-tests trip), a native-library fault, a
corrupted install discovered mid-process. When that happens Mercury must not
be forced to choose between "no crypto at all" and "silently borrow a
third-party library": external crypto libraries are benchmark-only in this
repository. This tier is the third option — handwritten, repository-owned
implementations of exactly the primitives an emergency needs, with an
explicit, fail-closed boundary around everything an emergency must NOT do.

Scope — deliberately asymmetric:

* **AEAD**: ChaCha20-Poly1305 (RFC 8439) only. ChaCha20 is add-rotate-xor
  over 32-bit words and Poly1305 is modular arithmetic — the construction
  standard software implementations use precisely because it carries no
  table lookups or data-dependent branches, unlike AES, whose S-box makes a
  pure-Python implementation a cache-timing hazard. AES-GCM therefore has
  **no** sovereign fallback: a caller that demands AES-GCM while AMA and the
  Rust backend are both down gets a loud error, never a silent algorithm
  swap. Output format is ``ciphertext || 16-byte tag`` — byte-compatible
  with the Rust and AMA tiers, so any tier decrypts any tier's output.
* **KDF**: HKDF-SHA256 (RFC 5869) over stdlib ``hmac``/``hashlib``
  (the repository's accepted hash base — e.g. the corpus SHA3 digests).
* **Signatures**: Ed25519 **verification only** (RFC 8032). Verification
  handles public inputs, so Python's variable-time arithmetic leaks nothing
  secret; it keeps Mercury's *read* paths alive (corpus-signature checks,
  Raft envelope verification, certificate audits) in a degraded state.
  **Signing is deliberately absent**: pure-Python scalar multiplication over
  a secret leaks through timing, so a degraded Mercury refuses to mint new
  signatures rather than mint side-channel-afflicted ones. Fail-safe reads,
  fail-closed writes.

Honest limitations, stated rather than hidden:

* Poly1305 runs on Python big integers; its timing varies with operand
  values. The MAC key is single-use per nonce (RFC 8439 §2.6) and the tag
  comparison is ``hmac.compare_digest``, which bounds the exposure, but this
  tier is an emergency measure, not a constant-time implementation.
* Throughput is pure-Python throughput (measured ~0.1 MB/s-class, versus
  hundreds of MB/s native). The emergency tier trades speed for existence.

Every primitive is pinned to its RFC's own test vectors in
``tests/crypto/test_sovereign_backup.py`` and cross-checked there against an
independent implementation.
"""

from __future__ import annotations

import hashlib
import hmac
import struct

__all__ = [
    "chacha20poly1305_decrypt",
    "chacha20poly1305_encrypt",
    "ed25519_verify",
    "hkdf_sha256",
]

# =============================================================================
# ChaCha20 stream cipher (RFC 8439 §2.1-2.4)
# =============================================================================

_MASK32 = 0xFFFFFFFF
#: "expand 32-byte k" — the RFC 8439 §2.3 constant words.
_CHACHA_CONSTANTS = (0x61707865, 0x3320646E, 0x79622D32, 0x6B206574)


def _rotl32(value: int, count: int) -> int:
    value &= _MASK32
    return ((value << count) | (value >> (32 - count))) & _MASK32


def _quarter_round(state: list[int], a: int, b: int, c: int, d: int) -> None:
    """RFC 8439 §2.1 quarter round, in place on four state indices."""
    state[a] = (state[a] + state[b]) & _MASK32
    state[d] = _rotl32(state[d] ^ state[a], 16)
    state[c] = (state[c] + state[d]) & _MASK32
    state[b] = _rotl32(state[b] ^ state[c], 12)
    state[a] = (state[a] + state[b]) & _MASK32
    state[d] = _rotl32(state[d] ^ state[a], 8)
    state[c] = (state[c] + state[d]) & _MASK32
    state[b] = _rotl32(state[b] ^ state[c], 7)


def _chacha20_block(key: bytes, counter: int, nonce: bytes) -> bytes:
    """One 64-byte ChaCha20 keystream block (RFC 8439 §2.3)."""
    state = list(_CHACHA_CONSTANTS)
    state.extend(struct.unpack("<8L", key))
    state.append(counter & _MASK32)
    state.extend(struct.unpack("<3L", nonce))
    working = state.copy()
    for _ in range(10):  # 20 rounds = 10 column+diagonal double-rounds
        _quarter_round(working, 0, 4, 8, 12)
        _quarter_round(working, 1, 5, 9, 13)
        _quarter_round(working, 2, 6, 10, 14)
        _quarter_round(working, 3, 7, 11, 15)
        _quarter_round(working, 0, 5, 10, 15)
        _quarter_round(working, 1, 6, 11, 12)
        _quarter_round(working, 2, 7, 8, 13)
        _quarter_round(working, 3, 4, 9, 14)
    return struct.pack("<16L", *((w + s) & _MASK32 for w, s in zip(working, state)))


def _chacha20_xor(key: bytes, nonce: bytes, data: bytes, initial_counter: int) -> bytes:
    """XOR ``data`` with the ChaCha20 keystream (RFC 8439 §2.4)."""
    out = bytearray(len(data))
    for block_index in range(0, len(data), 64):
        keystream = _chacha20_block(key, initial_counter + block_index // 64, nonce)
        chunk = data[block_index : block_index + 64]
        out[block_index : block_index + len(chunk)] = bytes(x ^ y for x, y in zip(chunk, keystream))
    return bytes(out)


# =============================================================================
# Poly1305 one-time authenticator (RFC 8439 §2.5)
# =============================================================================

_POLY1305_PRIME = (1 << 130) - 5
_CLAMP = 0x0FFFFFFC0FFFFFFC0FFFFFFC0FFFFFFF


def _poly1305_tag(one_time_key: bytes, message: bytes) -> bytes:
    """Poly1305 MAC of ``message`` under a 32-byte one-time key."""
    r = int.from_bytes(one_time_key[:16], "little") & _CLAMP
    s = int.from_bytes(one_time_key[16:32], "little")
    accumulator = 0
    for i in range(0, len(message), 16):
        block = message[i : i + 16]
        n = int.from_bytes(block + b"\x01", "little")
        accumulator = ((accumulator + n) * r) % _POLY1305_PRIME
    return ((accumulator + s) & ((1 << 128) - 1)).to_bytes(16, "little")


# =============================================================================
# ChaCha20-Poly1305 AEAD (RFC 8439 §2.8)
# =============================================================================


def _pad16(data: bytes) -> bytes:
    remainder = len(data) % 16
    return b"" if remainder == 0 else b"\x00" * (16 - remainder)


def _aead_mac_data(aad: bytes, ciphertext: bytes) -> bytes:
    return b"".join(
        (
            aad,
            _pad16(aad),
            ciphertext,
            _pad16(ciphertext),
            struct.pack("<Q", len(aad)),
            struct.pack("<Q", len(ciphertext)),
        )
    )


def chacha20poly1305_encrypt(key: bytes, nonce: bytes, plaintext: bytes, aad: bytes = b"") -> bytes:
    """RFC 8439 AEAD encrypt; returns ``ciphertext || 16-byte tag``.

    The output layout matches the Rust (`mercury_crypto`) and AMA tiers, so
    data sealed by any tier opens under any other.
    """
    if len(key) != 32:
        raise ValueError("ChaCha20-Poly1305 key must be 32 bytes")
    if len(nonce) != 12:
        raise ValueError("ChaCha20-Poly1305 nonce must be 12 bytes")
    one_time_key = _chacha20_block(key, 0, nonce)[:32]
    ciphertext = _chacha20_xor(key, nonce, plaintext, 1)
    tag = _poly1305_tag(one_time_key, _aead_mac_data(aad, ciphertext))
    return ciphertext + tag


def chacha20poly1305_decrypt(
    key: bytes, nonce: bytes, ciphertext_and_tag: bytes, aad: bytes = b""
) -> bytes:
    """RFC 8439 AEAD decrypt of ``ciphertext || tag``.

    Raises:
        ValueError: On a short input, or when the authentication tag does
            not verify (checked with ``hmac.compare_digest`` **before** any
            plaintext is produced).
    """
    if len(key) != 32:
        raise ValueError("ChaCha20-Poly1305 key must be 32 bytes")
    if len(nonce) != 12:
        raise ValueError("ChaCha20-Poly1305 nonce must be 12 bytes")
    if len(ciphertext_and_tag) < 16:
        raise ValueError("ciphertext shorter than the 16-byte tag")
    ciphertext = ciphertext_and_tag[:-16]
    tag = ciphertext_and_tag[-16:]
    one_time_key = _chacha20_block(key, 0, nonce)[:32]
    expected = _poly1305_tag(one_time_key, _aead_mac_data(aad, ciphertext))
    if not hmac.compare_digest(expected, tag):
        raise ValueError("Decryption failed: authentication tag mismatch")
    return _chacha20_xor(key, nonce, ciphertext, 1)


# =============================================================================
# HKDF-SHA256 (RFC 5869)
# =============================================================================


def hkdf_sha256(ikm: bytes, length: int, salt: bytes | None = None, info: bytes = b"") -> bytes:
    """RFC 5869 extract-then-expand over HMAC-SHA256.

    Deterministic by specification, so output is byte-identical to every
    conformant implementation (pinned against RFC 5869 appendix A vectors).
    """
    if length < 1 or length > 255 * 32:
        raise ValueError("HKDF-SHA256 output length must be in [1, 8160]")
    if salt is None:
        salt = b"\x00" * 32
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    okm = b""
    previous = b""
    counter = 1
    while len(okm) < length:
        previous = hmac.new(prk, previous + info + bytes([counter]), hashlib.sha256).digest()
        okm += previous
        counter += 1
    return okm[:length]


# =============================================================================
# Ed25519 verification (RFC 8032 §5.1) — verify-only, by design
# =============================================================================

_P = 2**255 - 19
_L = 2**252 + 27742317777372353535851937790883648493
_D = (-121665 * pow(121666, _P - 2, _P)) % _P
_SQRT_M1 = pow(2, (_P - 1) // 4, _P)

#: Base point B (RFC 8032 §5.1) in extended homogeneous coordinates (X,Y,Z,T).
_BASE_Y = (4 * pow(5, _P - 2, _P)) % _P
_BASE_X = 15112221349535400772501151409588531511454012693041857206046113283949847762202
_BASE = (_BASE_X, _BASE_Y, 1, (_BASE_X * _BASE_Y) % _P)
_IDENTITY = (0, 1, 1, 0)

_Point = tuple[int, int, int, int]


def _edwards_add(p: _Point, q: _Point) -> _Point:
    """Point addition, extended homogeneous coordinates (RFC 8032 §5.1.4)."""
    x1, y1, z1, t1 = p
    x2, y2, z2, t2 = q
    a = ((y1 - x1) * (y2 - x2)) % _P
    b = ((y1 + x1) * (y2 + x2)) % _P
    c = (t1 * 2 * _D * t2) % _P
    d = (z1 * 2 * z2) % _P
    e = b - a
    f = d - c
    g = d + c
    h = b + a
    return ((e * f) % _P, (g * h) % _P, (f * g) % _P, (e * h) % _P)


def _scalar_mult(scalar: int, point: _Point) -> _Point:
    """Double-and-add scalar multiplication.

    Variable-time, which is safe here: every scalar this module handles
    (``s`` and ``k`` in verification) is public once a signature is
    published. This is exactly why signing lives only in AMA.
    """
    result = _IDENTITY
    addend = point
    while scalar:
        if scalar & 1:
            result = _edwards_add(result, addend)
        addend = _edwards_add(addend, addend)
        scalar >>= 1
    return result


def _point_encode(point: _Point) -> bytes:
    x, y, z, _ = point
    z_inv = pow(z, _P - 2, _P)
    x, y = (x * z_inv) % _P, (y * z_inv) % _P
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


def _point_decode(encoded: bytes) -> _Point | None:
    """Canonical point decompression (RFC 8032 §5.1.3); ``None`` on rejection.

    Strictness matches AMA v5's verifier: a non-canonical ``y`` (>= p) is
    rejected, and so is the ``x = 0`` encoding with the sign bit set — the
    two encodings AMA v5.0.0 closed in both of its backends.
    """
    if len(encoded) != 32:
        return None
    y = int.from_bytes(encoded, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    if y >= _P:
        return None  # non-canonical field element
    y2 = (y * y) % _P
    u = (y2 - 1) % _P
    v = (_D * y2 + 1) % _P
    # x = (u/v)^((p+3)/8) via the combined exponentiation trick.
    x = (u * pow(v, 3, _P) * pow(u * pow(v, 7, _P), (_P - 5) // 8, _P)) % _P
    vxx = (v * x * x) % _P
    if vxx == u % _P:
        pass
    elif vxx == (-u) % _P:
        x = (x * _SQRT_M1) % _P
    else:
        return None  # not a point on the curve
    if x == 0 and sign == 1:
        return None  # RFC 8032 §5.1.3: x = 0 admits no sign bit
    if x & 1 != sign:
        x = _P - x
    return (x, y, 1, (x * y) % _P)


def ed25519_verify(message: bytes, signature: bytes, public_key: bytes) -> bool:
    """RFC 8032 §5.1.7 Ed25519 verification — strict, cofactorless.

    Accepts exactly what AMA's native verifier accepts: canonical point
    encodings for both ``A`` and ``R``, and ``s < L`` (rejecting the
    signature-malleability range). Returns ``False`` for any malformed
    input rather than raising, preserving Mercury's boolean verifier
    contract. There is deliberately no ``ed25519_sign`` counterpart here —
    see the module docstring.
    """
    if len(signature) != 64 or len(public_key) != 32:
        return False
    a_point = _point_decode(public_key)
    if a_point is None:
        return False
    r_point = _point_decode(signature[:32])
    if r_point is None:
        return False
    s = int.from_bytes(signature[32:], "little")
    if s >= _L:
        return False
    k = (
        int.from_bytes(hashlib.sha512(signature[:32] + public_key + message).digest(), "little")
        % _L
    )
    left = _scalar_mult(s, _BASE)
    right = _edwards_add(r_point, _scalar_mult(k, a_point))
    return _point_encode(left) == _point_encode(right)
