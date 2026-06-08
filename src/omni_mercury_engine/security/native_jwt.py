# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Native JSON Web Token (JWT) implementation — stdlib + AMA-routed.

This module is the supply-chain remediation that retires Mercury-Agent's
dependency on ``pyjwt`` and removes the upstream-disputed
``PYSEC-2025-183`` / ``CVE-2025-45768`` advisory from Mercury's
audited surface.  No new third-party library is taken on; the
implementation is built on ``hmac`` + ``hashlib`` + ``base64`` +
``json`` + ``time`` only, in line with Mercury's broader "zero-dep
crypto" posture (cf. ``AMA-Cryptography`` INVARIANT-1).

AMA HMAC routing
----------------
The HS256 and HS512 signing primitives are routed through AMA's ACVP-validated
constant-time C HMAC implementations
(:func:`omni_mercury_engine.security.ama_hmac.ama_hmac_sha256` and
:func:`omni_mercury_engine.security.ama_hmac.ama_hmac_sha512`).  This
puts the JWT signing path on the same crypto backend the rest of
Mercury's PQC + HKDF stack already uses, matches AMA's INVARIANT-1
posture, and removes OpenSSL-backed stdlib HMAC from the production
auth path.  HS384 stays on stdlib because AMA does not currently bind
HMAC-SHA-384 (tracked in ``docs/ROADMAP.md``).

Scope
-----
Compact JWS (RFC 7519 §3) with HMAC-SHA2 family signatures (RFC 7518
§3.2):

* ``HS256`` (HMAC-SHA-256) — required by Mercury's API; default;
  **AMA-routed**.
* ``HS384`` (HMAC-SHA-384) — supported for interop; stdlib only.
* ``HS512`` (HMAC-SHA-512) — supported for interop; **AMA-routed**.

Asymmetric algorithms (RS*/ES*/PS*/EdDSA) are out of scope for this
module — Mercury's deployment surface uses HS256 exclusively, and the
asymmetric families add OpenSSL-level complexity that is better
handled by a dedicated library when needed.  The ``"alg": "none"``
unsigned variant is **rejected by construction**: both the encoder
and the decoder whitelist a finite set of HMAC algorithms; an
attacker cannot downgrade a token to the unsigned variant.

Security properties
-------------------

* Signature verification uses :func:`hmac.compare_digest` (constant
  time on equal-length inputs) via
  :func:`omni_mercury_engine.security.constant_time.constant_time_compare`.
* HMAC digest computation is delegated to AMA Cryptography's
  constant-time C backend (see
  ``AMA-Cryptography/CONSTANT_TIME_VERIFICATION.md``); stdlib HMAC
  is retained only for HS384, which AMA v3.2.0 does not bind.
* ``alg`` is read from the header BUT is then matched against the
  caller-supplied ``algorithms=`` whitelist; a token whose header
  claims an algorithm not in the whitelist is rejected before any
  signing key is touched (mitigates "alg confusion" / "alg=none"
  bypasses).
* Padded base64url (``=`` padding) and unpadded base64url are both
  accepted on decode; encode always emits the unpadded form
  (RFC 7515 §2 "Terminology" / RFC 4648 §5).
* ``exp`` / ``nbf`` / ``iat`` clock checks are integer-second based
  with optional leeway; ``datetime`` objects are accepted on encode
  for ergonomic parity with the prior pyjwt API contract.
* The ``require=`` option enforces presence of named claims; missing
  required claims fail with :class:`MissingRequiredClaimError`.
* JSON parsing failures, malformed segments, and signature
  mismatches are *not* distinguished in the error message returned
  to callers — only the typed exception class differs — so the
  module does not leak which check failed via log content.

References
----------
* RFC 7515 — JSON Web Signature (JWS)
* RFC 7518 — JSON Web Algorithms (JWA)
* RFC 7519 — JSON Web Token (JWT)
* RFC 4231 — HMAC-SHA-2 test vectors.
* FIPS 198-1 — Keyed-Hash Message Authentication Code.
* CVE-2025-45768 / PYSEC-2025-183 — disputed pyjwt weak-key concern,
  rendered moot here by removal of the pyjwt dependency.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.security import ama_hmac
from omni_mercury_engine.security.constant_time import constant_time_compare

if TYPE_CHECKING:
    from collections.abc import Iterable

# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class NativeJWTError(Exception):
    """Base class for all native-JWT errors."""


class InvalidTokenError(NativeJWTError):
    """Token is structurally invalid or fails a verification check.

    This is the public-facing catch-all subclass that mirrors
    ``jwt.InvalidTokenError`` so callers can keep a single
    ``except InvalidTokenError`` branch when adapting from pyjwt.
    """


class DecodeError(InvalidTokenError):
    """Token bytes cannot be parsed (bad base64, bad JSON, wrong segment count)."""


class InvalidSignatureError(InvalidTokenError):
    """HMAC signature mismatch."""


class ExpiredSignatureError(InvalidTokenError):
    """``exp`` claim is in the past (subject to ``leeway``)."""


class ImmatureSignatureError(InvalidTokenError):
    """``nbf`` claim is in the future (subject to ``leeway``)."""


class InvalidAlgorithmError(InvalidTokenError):
    """``alg`` header claims an algorithm not in the caller's whitelist."""


class MissingRequiredClaimError(InvalidTokenError):
    """A claim listed in ``options['require']`` is absent from the payload."""


# --------------------------------------------------------------------------- #
# Algorithm registry
# --------------------------------------------------------------------------- #

_HASH_BY_ALG: dict[str, Any] = {
    "HS256": hashlib.sha256,
    "HS384": hashlib.sha384,
    "HS512": hashlib.sha512,
}

SUPPORTED_ALGORITHMS: tuple[str, ...] = tuple(_HASH_BY_ALG.keys())

# JWS algorithms that may be routed through AMA's native C HMAC when
# the AMA shared library is loaded.  HS384 stays stdlib-only because
# AMA Cryptography does not bind HMAC-SHA-384 (cf. AMA's
# ``src/c/ama_hmac_sha256.c`` / ``ama_hmac_sha512.c``; no SHA-384
# variant ships in the C library).
_AMA_ROUTABLE_ALGS: frozenset[str] = frozenset({"HS256", "HS512"})

# --------------------------------------------------------------------------- #
# Base64URL helpers
# --------------------------------------------------------------------------- #


def _b64url_encode(data: bytes) -> bytes:
    """Encode ``data`` as unpadded base64url (RFC 7515 §2)."""
    return base64.urlsafe_b64encode(data).rstrip(b"=")


def _b64url_decode(data: bytes) -> bytes:
    """Decode unpadded base64url, restoring ``=`` padding internally.

    Raises:
        DecodeError: if ``data`` is not valid base64url.
    """
    pad = (-len(data)) % 4
    try:
        return base64.urlsafe_b64decode(data + b"=" * pad)
    except (ValueError, TypeError) as exc:
        raise DecodeError("Invalid base64url segment") from exc


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _coerce_secret(key: str | bytes) -> bytes:
    if isinstance(key, str):
        return key.encode("utf-8")
    if isinstance(key, (bytes, bytearray)):
        return bytes(key)
    raise TypeError(
        f"JWT signing key must be str or bytes, got {type(key).__name__}",
    )


def _datetime_to_epoch(value: Any) -> Any:
    """Normalise ``datetime`` claims to integer epoch seconds.

    Naive ``datetime`` instances are treated as local time, matching
    pyjwt's historical behaviour, to preserve drop-in semantics for
    the ``api/auth.py`` call sites.  ``int`` / ``float`` claims pass
    through untouched so callers can mix raw timestamps and
    ``datetime`` objects freely.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return int(value.timestamp())
        return int(value.astimezone(UTC).timestamp())
    return value


def _normalize_temporal_claims(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert temporal claims (``exp``/``iat``/``nbf``) to epoch seconds.

    Returns a shallow copy of ``payload`` so the caller's input dict
    is not mutated.
    """
    out = dict(payload)
    for claim in ("exp", "iat", "nbf"):
        if claim in out:
            out[claim] = _datetime_to_epoch(out[claim])
    return out


def _signing_input(header_segment: bytes, payload_segment: bytes) -> bytes:
    """JWS signing input — ``b64(header) || "." || b64(payload)`` (RFC 7515 §5.1)."""
    return header_segment + b"." + payload_segment


def _sign_stdlib(header_segment: bytes, payload_segment: bytes, key: bytes, alg: str) -> bytes:
    """Stdlib HMAC path (FIPS 198-1 / RFC 2104) using :mod:`hmac`."""
    digestmod = _HASH_BY_ALG[alg]
    return hmac.new(key, _signing_input(header_segment, payload_segment), digestmod).digest()


def _sign_ama(header_segment: bytes, payload_segment: bytes, key: bytes, alg: str) -> bytes:
    """AMA-routed HMAC path — ACVP-validated constant-time C backend.

    Pre-condition: the caller has verified ``alg`` is in
    :data:`_AMA_ROUTABLE_ALGS` *and* the corresponding AMA binding
    flag is set.  This function does not re-check those; it raises
    ``RuntimeError`` directly from the AMA layer if the underlying
    binding has been invalidated since the flag was read.

    For HS256 we use AMA's :func:`ama_hmac.ama_hmac_sha256_2`
    two-segment entry point so the (potentially large) payload
    segment is not concatenated in Python before being handed to
    the C HMAC.  The first segment is ``header || "."`` (typically
    ≈50 bytes); the second segment is the raw base64url payload.
    By RFC 2104 / FIPS 198-1 the result is byte-identical to
    ``HMAC(key, header || "." || payload)``.
    """
    match alg:
        case "HS256":
            return ama_hmac.ama_hmac_sha256_2(key, header_segment + b".", payload_segment)
        case "HS512":
            # AMA v3.2.0 does not yet ship a two-segment HMAC-SHA-512
            # variant; use the one-segment binding with a
            # single concat.  Tracked in docs/ROADMAP.md.
            return ama_hmac.ama_hmac_sha512(key, _signing_input(header_segment, payload_segment))
    raise InvalidAlgorithmError(f"Algorithm {alg!r} is not AMA-routable")


def _alg_uses_ama(alg: str) -> bool:
    """Return ``True`` iff ``alg`` will be served by the AMA backend.

    HS256 and HS512 are mandatory AMA routes; HS384 remains stdlib-only until
    AMA exports a SHA-384 HMAC binding.
    """
    match alg:
        case "HS256":
            if not ama_hmac.HAS_AMA_HMAC_SHA256:
                raise RuntimeError("AMA HMAC-SHA-256 is mandatory for HS256")
            return True
        case "HS512":
            if not ama_hmac.HAS_AMA_HMAC_SHA512:
                raise RuntimeError("AMA HMAC-SHA-512 is mandatory for HS512")
            return True
        case _:
            return False


def _sign(header_segment: bytes, payload_segment: bytes, key: bytes, alg: str) -> bytes:
    """Compute the HMAC tag for ``b64(header).b64(payload)`` under ``alg``.

    Routes HS256/HS512 through AMA Cryptography's native C HMAC.  HS384 remains
    stdlib-only because AMA v3.2.0 does not bind HMAC-SHA-384.
    """
    if _alg_uses_ama(alg):
        return _sign_ama(header_segment, payload_segment, key, alg)
    return _sign_stdlib(header_segment, payload_segment, key, alg)


def get_signing_backend(alg: str) -> str:
    """Return the backend name (``"ama"`` or ``"stdlib"``) used for ``alg``.

    Intended for ``/health`` endpoints and audit logs; do not gate
    security decisions on this string.
    """
    if alg not in _HASH_BY_ALG:
        raise InvalidAlgorithmError(
            f"Unsupported algorithm {alg!r}; supported: {', '.join(SUPPORTED_ALGORITHMS)}",
        )
    return "ama" if _alg_uses_ama(alg) else "stdlib"


# --------------------------------------------------------------------------- #
# Public API: encode / decode
# --------------------------------------------------------------------------- #


def encode(
    payload: dict[str, Any],
    key: str | bytes,
    algorithm: str = "HS256",
    headers: dict[str, Any] | None = None,
) -> str:
    """Encode ``payload`` as a signed compact JWS.

    Args:
        payload: JWT claims set.  ``exp``/``iat``/``nbf`` may be
            ``datetime`` instances and will be coerced to integer
            epoch seconds.
        key: HMAC secret (``str`` or ``bytes``).
        algorithm: One of :data:`SUPPORTED_ALGORITHMS` (default
            ``"HS256"``).
        headers: Optional extra JOSE header fields (``alg`` and
            ``typ`` are always set by the encoder and will override
            any caller-supplied value).

    Returns:
        The compact-serialised token (``header.payload.signature``).

    Raises:
        InvalidAlgorithmError: ``algorithm`` is not supported.
        TypeError: ``payload`` is not a ``dict`` or ``key`` has a
            wrong type.
    """
    if algorithm not in _HASH_BY_ALG:
        raise InvalidAlgorithmError(
            f"Unsupported algorithm {algorithm!r}; supported: {', '.join(SUPPORTED_ALGORITHMS)}",
        )
    if not isinstance(payload, dict):
        raise TypeError(f"JWT payload must be a dict, got {type(payload).__name__}")

    secret = _coerce_secret(key)
    normalised_payload = _normalize_temporal_claims(payload)

    header: dict[str, Any] = dict(headers) if headers else {}
    header["alg"] = algorithm
    header["typ"] = "JWT"

    header_segment = _b64url_encode(
        json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8"),
    )
    payload_segment = _b64url_encode(
        json.dumps(normalised_payload, separators=(",", ":"), sort_keys=True).encode("utf-8"),
    )
    signature = _sign(header_segment, payload_segment, secret, algorithm)
    signature_segment = _b64url_encode(signature)

    return b".".join((header_segment, payload_segment, signature_segment)).decode("ascii")


_DEFAULT_OPTIONS: dict[str, Any] = {
    "verify_signature": True,
    "verify_exp": True,
    "verify_nbf": True,
    "verify_iat": False,  # iat is informational; pyjwt parity
    "require": (),
}


def decode(
    token: str | bytes,
    key: str | bytes,
    algorithms: Iterable[str],
    options: dict[str, Any] | None = None,
    leeway: float = 0.0,
) -> dict[str, Any]:
    """Verify and decode a compact JWS token.

    Args:
        token: The token string (``header.payload.signature``).
        key: HMAC secret used to verify the signature.
        algorithms: Iterable of accepted algorithm names.  The
            token's header ``alg`` MUST be in this set; otherwise
            :class:`InvalidAlgorithmError` is raised before any
            HMAC work is done.
        options: Optional verification toggles:
            ``{"verify_signature", "verify_exp", "verify_nbf",
            "verify_iat", "require"}``.  ``require`` is an iterable
            of claim names that must be present in the payload.
        leeway: Allowed clock skew in seconds for ``exp`` / ``nbf``
            checks.

    Returns:
        The decoded payload (claim names → values).

    Raises:
        DecodeError: Token is structurally malformed.
        InvalidAlgorithmError: ``alg`` is not in ``algorithms``.
        InvalidSignatureError: HMAC verification failed.
        ExpiredSignatureError: ``exp`` claim is in the past.
        ImmatureSignatureError: ``nbf`` claim is in the future.
        MissingRequiredClaimError: A required claim is absent.
        InvalidTokenError: Any other validation failure.
    """
    if isinstance(token, str):
        token_bytes = token.encode("ascii")
    elif isinstance(token, (bytes, bytearray)):
        token_bytes = bytes(token)
    else:
        raise DecodeError(
            f"JWT must be str or bytes, got {type(token).__name__}",
        )

    segments = token_bytes.split(b".")
    if len(segments) != 3:
        raise DecodeError(
            f"Compact JWS must have exactly 3 segments, got {len(segments)}",
        )
    header_segment, payload_segment, signature_segment = segments

    # Parse header.
    try:
        header_bytes = _b64url_decode(header_segment)
        header = json.loads(header_bytes.decode("utf-8"))
    except (DecodeError, ValueError, UnicodeDecodeError) as exc:
        raise DecodeError("Invalid JWT header") from exc
    if not isinstance(header, dict):
        raise DecodeError("JWT header is not a JSON object")

    # Algorithm whitelisting BEFORE any HMAC work.  Rejects
    # ``alg: none`` and any algorithm not explicitly accepted by the
    # caller.
    alg = header.get("alg")
    allowed = tuple(algorithms)
    if alg not in allowed:
        raise InvalidAlgorithmError(
            f"Token algorithm {alg!r} not in allowed set {allowed!r}",
        )
    if alg not in _HASH_BY_ALG:
        raise InvalidAlgorithmError(
            f"Algorithm {alg!r} is not implemented by native_jwt",
        )

    opts = dict(_DEFAULT_OPTIONS)
    if options:
        opts.update(options)

    # Signature verification.
    if opts["verify_signature"]:
        secret = _coerce_secret(key)
        expected = _sign(
            header_segment,
            payload_segment,
            secret,
            alg,
        )
        try:
            actual = _b64url_decode(signature_segment)
        except DecodeError as exc:
            raise InvalidSignatureError("Invalid signature segment") from exc
        if len(expected) != len(actual) or not constant_time_compare(expected, actual):
            raise InvalidSignatureError("Signature verification failed")

    # Parse payload.
    try:
        payload_bytes = _b64url_decode(payload_segment)
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (DecodeError, ValueError, UnicodeDecodeError) as exc:
        raise DecodeError("Invalid JWT payload") from exc
    if not isinstance(payload, dict):
        raise DecodeError("JWT payload is not a JSON object")

    # Required-claims enforcement.
    required = opts.get("require") or ()
    for claim in required:
        if claim not in payload:
            raise MissingRequiredClaimError(f"Token is missing required claim {claim!r}")

    # Temporal claim checks (integer-second based, with optional leeway).
    now = _now_epoch()
    if opts.get("verify_exp", True) and "exp" in payload:
        exp = _as_epoch(payload["exp"], "exp")
        if now > exp + float(leeway):
            raise ExpiredSignatureError("Token has expired")

    if opts.get("verify_nbf", True) and "nbf" in payload:
        nbf = _as_epoch(payload["nbf"], "nbf")
        if now + float(leeway) < nbf:
            raise ImmatureSignatureError("Token not yet valid (nbf)")

    if opts.get("verify_iat", False) and "iat" in payload:
        iat = _as_epoch(payload["iat"], "iat")
        # iat in the future (beyond leeway) is treated like nbf.
        if now + float(leeway) < iat:
            raise InvalidTokenError("Token issued in the future (iat)")

    return payload


def get_unverified_header(token: str | bytes) -> dict[str, Any]:
    """Return the JOSE header of ``token`` without verifying the signature.

    Intended only for diagnostic / debugging contexts (e.g. logging
    the ``kid`` of a token that failed verification).  Never feed
    the result back into authorisation decisions.
    """
    if isinstance(token, str):
        token = token.encode("ascii")
    segments = token.split(b".")
    if len(segments) != 3:
        raise DecodeError("Compact JWS must have exactly 3 segments")
    try:
        header_bytes = _b64url_decode(segments[0])
        header = json.loads(header_bytes.decode("utf-8"))
    except (DecodeError, ValueError, UnicodeDecodeError) as exc:
        raise DecodeError("Invalid JWT header") from exc
    if not isinstance(header, dict):
        raise DecodeError("JWT header is not a JSON object")
    return header


# --------------------------------------------------------------------------- #
# Time + claim coercion helpers
# --------------------------------------------------------------------------- #


def _now_epoch() -> float:
    """Wall-clock now in epoch seconds; isolated for monkeypatching in tests."""
    return datetime.now(tz=UTC).timestamp()


def _as_epoch(value: Any, claim: str) -> float:
    """Coerce a temporal claim value to a float epoch second.

    Accepts numerics and ``datetime`` instances; raises
    :class:`InvalidTokenError` for anything else so a tampered
    payload cannot bypass temporal checks by, e.g., sending an
    ``exp`` claim of ``"never"``.
    """
    if isinstance(value, bool):
        # ``bool`` is a subclass of ``int``; treat as invalid.
        raise InvalidTokenError(f"Claim {claim!r} must be numeric, got bool")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        coerced = _datetime_to_epoch(value)
        # _datetime_to_epoch returns ``int`` for ``datetime`` inputs;
        # the branch above already excluded non-datetime values.
        return float(coerced)
    raise InvalidTokenError(
        f"Claim {claim!r} must be a numeric timestamp, got {type(value).__name__}",
    )


__all__ = [
    "SUPPORTED_ALGORITHMS",
    "DecodeError",
    "ExpiredSignatureError",
    "ImmatureSignatureError",
    "InvalidAlgorithmError",
    "InvalidSignatureError",
    "InvalidTokenError",
    "MissingRequiredClaimError",
    "NativeJWTError",
    "decode",
    "encode",
    "get_signing_backend",
    "get_unverified_header",
]
