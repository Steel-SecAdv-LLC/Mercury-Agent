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
Regression and property tests for ``security.native_jwt``.

These tests lock the contract that ``api/auth.py`` (and any future
caller) relies on:

* Round-trip ``encode → decode`` is identity-preserving.
* The whitelist of algorithms is HMAC-only.
* ``alg: none`` and unknown algorithms are rejected BEFORE any HMAC
  work touches the signing key.
* Signature tampering is detected and produces
  :class:`InvalidSignatureError`.
* ``exp`` / ``nbf`` enforcement is integer-second precise and
  respects ``leeway``.
* ``options['require']`` enforces presence of named claims.
* ``datetime`` claim values are accepted by the encoder and stored
  as integer epoch seconds.

Property-style coverage is implemented with parametrize + direct
edge-case construction rather than Hypothesis to keep this file
self-contained and avoid pulling Hypothesis into the security
test-tier import surface.
"""

import base64
import json
from datetime import UTC, datetime, timedelta

import pytest

from omni_mercury_engine.security import native_jwt
from omni_mercury_engine.security.native_jwt import (
    SUPPORTED_ALGORITHMS,
    DecodeError,
    ExpiredSignatureError,
    ImmatureSignatureError,
    InvalidAlgorithmError,
    InvalidSignatureError,
    InvalidTokenError,
    MissingRequiredClaimError,
)

SECRET = "test-secret-key-which-is-clearly-not-for-production-use"


# --------------------------------------------------------------------------- #
# Round-trip
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("alg", SUPPORTED_ALGORITHMS)
def test_round_trip_preserves_payload(alg: str) -> None:
    payload = {"sub": "user-1", "scope": ["read", "write"], "n": 42}
    token = native_jwt.encode(payload, SECRET, algorithm=alg)
    decoded = native_jwt.decode(token, SECRET, algorithms=[alg])
    assert decoded == payload


def test_round_trip_with_bytes_key() -> None:
    payload = {"sub": "abc"}
    token = native_jwt.encode(payload, SECRET.encode("utf-8"), algorithm="HS256")
    decoded = native_jwt.decode(token, SECRET.encode("utf-8"), algorithms=["HS256"])
    assert decoded == payload


def test_round_trip_accepts_datetime_temporal_claims() -> None:
    now = datetime.now(tz=UTC).replace(microsecond=0)
    payload = {
        "sub": "x",
        "iat": now,
        "exp": now + timedelta(hours=1),
    }
    token = native_jwt.encode(payload, SECRET)
    decoded = native_jwt.decode(token, SECRET, algorithms=["HS256"])
    # datetime claims are stored as integer epoch seconds.
    assert decoded["iat"] == int(now.timestamp())
    assert decoded["exp"] == int((now + timedelta(hours=1)).timestamp())


# --------------------------------------------------------------------------- #
# Algorithm guard rails — including ``alg: none`` rejection
# --------------------------------------------------------------------------- #


def test_encode_rejects_unknown_algorithm() -> None:
    with pytest.raises(InvalidAlgorithmError):
        native_jwt.encode({"sub": "x"}, SECRET, algorithm="ES256")


def test_decode_rejects_alg_not_in_caller_allowlist() -> None:
    token = native_jwt.encode({"sub": "x"}, SECRET, algorithm="HS256")
    with pytest.raises(InvalidAlgorithmError):
        native_jwt.decode(token, SECRET, algorithms=["HS384"])


def _assemble_unsafe_token(header: dict[str, str], payload: dict[str, str]) -> str:
    """Build a token bypassing the safe encoder; used to test attack vectors."""

    def b64(obj: dict[str, str]) -> bytes:
        return base64.urlsafe_b64encode(
            json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8"),
        ).rstrip(b"=")

    header_seg = b64(header)
    payload_seg = b64(payload)
    return f"{header_seg.decode()}.{payload_seg.decode()}."


def test_decode_rejects_alg_none_unsigned_token() -> None:
    """The ``alg: none`` downgrade attack must fail before any HMAC work."""
    unsafe_token = _assemble_unsafe_token({"alg": "none", "typ": "JWT"}, {"sub": "x"})
    with pytest.raises(InvalidAlgorithmError):
        native_jwt.decode(unsafe_token, SECRET, algorithms=["HS256"])


def test_decode_rejects_alg_none_even_if_caller_lists_it() -> None:
    """Even if a caller mistakenly lists 'none', the decoder refuses it."""
    unsafe_token = _assemble_unsafe_token({"alg": "none", "typ": "JWT"}, {"sub": "x"})
    with pytest.raises(InvalidAlgorithmError):
        native_jwt.decode(unsafe_token, SECRET, algorithms=["none"])


# --------------------------------------------------------------------------- #
# Signature integrity
# --------------------------------------------------------------------------- #


def test_decode_rejects_tampered_signature() -> None:
    token = native_jwt.encode({"sub": "x"}, SECRET)
    header, payload, sig = token.split(".")
    # Flip a byte in the signature.
    sig_bytes = bytearray(base64.urlsafe_b64decode(sig + "=" * ((-len(sig)) % 4)))
    sig_bytes[0] ^= 0xFF
    tampered_sig = base64.urlsafe_b64encode(bytes(sig_bytes)).rstrip(b"=").decode()
    bad_token = f"{header}.{payload}.{tampered_sig}"
    with pytest.raises(InvalidSignatureError):
        native_jwt.decode(bad_token, SECRET, algorithms=["HS256"])


def test_decode_rejects_tampered_payload() -> None:
    token = native_jwt.encode({"sub": "x", "admin": False}, SECRET)
    header, payload, sig = token.split(".")
    # Re-encode the payload claiming admin=True; signature is now stale.
    new_payload = (
        base64.urlsafe_b64encode(
            json.dumps({"sub": "x", "admin": True}, separators=(",", ":"), sort_keys=True).encode(),
        )
        .rstrip(b"=")
        .decode()
    )
    bad_token = f"{header}.{new_payload}.{sig}"
    with pytest.raises(InvalidSignatureError):
        native_jwt.decode(bad_token, SECRET, algorithms=["HS256"])


def test_decode_rejects_wrong_secret() -> None:
    token = native_jwt.encode({"sub": "x"}, SECRET)
    with pytest.raises(InvalidSignatureError):
        native_jwt.decode(token, "different-secret", algorithms=["HS256"])


def test_decode_rejects_truncated_signature() -> None:
    token = native_jwt.encode({"sub": "x"}, SECRET)
    header, payload, _sig = token.split(".")
    bad_token = f"{header}.{payload}."
    with pytest.raises(InvalidSignatureError):
        native_jwt.decode(bad_token, SECRET, algorithms=["HS256"])


# --------------------------------------------------------------------------- #
# Structural integrity — must not depend on signature checks
# --------------------------------------------------------------------------- #


def test_decode_rejects_wrong_segment_count() -> None:
    with pytest.raises(DecodeError):
        native_jwt.decode("foo.bar", SECRET, algorithms=["HS256"])
    with pytest.raises(DecodeError):
        native_jwt.decode("a.b.c.d", SECRET, algorithms=["HS256"])


def test_decode_rejects_bad_base64() -> None:
    bad = "not_b64!!!.also_bad.too"
    with pytest.raises(DecodeError):
        native_jwt.decode(bad, SECRET, algorithms=["HS256"])


def test_decode_rejects_non_object_header() -> None:
    header_seg = base64.urlsafe_b64encode(b'"a string"').rstrip(b"=").decode()
    payload_seg = base64.urlsafe_b64encode(b"{}").rstrip(b"=").decode()
    token = f"{header_seg}.{payload_seg}.AAAA"
    with pytest.raises(DecodeError):
        native_jwt.decode(token, SECRET, algorithms=["HS256"])


def test_decode_rejects_non_object_payload() -> None:
    # Build a token with a JSON array payload and a real HMAC sig so
    # the failure mode is the post-signature payload-parse step.
    header_seg = base64.urlsafe_b64encode(
        json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":"), sort_keys=True).encode(),
    ).rstrip(b"=")
    payload_seg = base64.urlsafe_b64encode(b"[1,2,3]").rstrip(b"=")
    import hmac as _hmac

    sig_bytes = _hmac.new(
        SECRET.encode("utf-8"),
        header_seg + b"." + payload_seg,
        "sha256",
    ).digest()
    sig_seg = base64.urlsafe_b64encode(sig_bytes).rstrip(b"=")
    bad_token = b".".join((header_seg, payload_seg, sig_seg)).decode()
    with pytest.raises(DecodeError):
        native_jwt.decode(bad_token, SECRET, algorithms=["HS256"])


# --------------------------------------------------------------------------- #
# Temporal claims
# --------------------------------------------------------------------------- #


def test_expired_token_is_rejected() -> None:
    past = datetime.now(tz=UTC) - timedelta(hours=1)
    token = native_jwt.encode({"sub": "x", "exp": past}, SECRET)
    with pytest.raises(ExpiredSignatureError):
        native_jwt.decode(token, SECRET, algorithms=["HS256"])


def test_expired_token_passes_with_sufficient_leeway() -> None:
    just_now = datetime.now(tz=UTC) - timedelta(seconds=5)
    token = native_jwt.encode({"sub": "x", "exp": just_now}, SECRET)
    # 60s leeway covers the 5s lag.
    decoded = native_jwt.decode(token, SECRET, algorithms=["HS256"], leeway=60.0)
    assert decoded["sub"] == "x"


def test_nbf_in_future_is_rejected() -> None:
    future = datetime.now(tz=UTC) + timedelta(hours=1)
    token = native_jwt.encode({"sub": "x", "nbf": future}, SECRET)
    with pytest.raises(ImmatureSignatureError):
        native_jwt.decode(token, SECRET, algorithms=["HS256"])


def test_verify_exp_disabled_allows_expired_token() -> None:
    past = datetime.now(tz=UTC) - timedelta(hours=1)
    token = native_jwt.encode({"sub": "x", "exp": past}, SECRET)
    decoded = native_jwt.decode(
        token,
        SECRET,
        algorithms=["HS256"],
        options={"verify_exp": False},
    )
    assert decoded["sub"] == "x"


def test_non_numeric_exp_is_rejected() -> None:
    """``"exp": "never"`` must NOT be coerced to "valid forever"."""
    # Build a token whose payload has a string ``exp`` claim, using
    # the real encoder so signature is valid.
    payload = {"sub": "x"}
    token = native_jwt.encode(payload, SECRET)
    # Surgically replace the payload with one carrying a bogus exp.
    header, _payload_seg, sig = token.split(".")
    forged_payload = (
        base64.urlsafe_b64encode(
            json.dumps(
                {"sub": "x", "exp": "never"}, separators=(",", ":"), sort_keys=True
            ).encode(),
        )
        .rstrip(b"=")
        .decode()
    )
    # The signature will not match, so we should hit
    # InvalidSignatureError first — confirming the structural-check
    # ordering keeps signature verification ahead of payload
    # interpretation.
    bad = f"{header}.{forged_payload}.{sig}"
    with pytest.raises(InvalidSignatureError):
        native_jwt.decode(bad, SECRET, algorithms=["HS256"])

    # If signature verification is bypassed, the decoder must still
    # refuse to treat a string exp as valid.
    decoded_payload_alone = native_jwt.encode({"sub": "x", "exp": "never"}, SECRET)
    # encode treats the literal string as opaque (no coercion),
    # but decode must reject it because exp is not numeric.
    with pytest.raises(InvalidTokenError):
        native_jwt.decode(
            decoded_payload_alone,
            SECRET,
            algorithms=["HS256"],
        )


# --------------------------------------------------------------------------- #
# Required claims
# --------------------------------------------------------------------------- #


def test_required_claim_missing_raises() -> None:
    token = native_jwt.encode({"sub": "x"}, SECRET)
    with pytest.raises(MissingRequiredClaimError):
        native_jwt.decode(
            token,
            SECRET,
            algorithms=["HS256"],
            options={"require": ["sub", "exp"]},
        )


def test_required_claim_present_passes() -> None:
    exp = datetime.now(tz=UTC) + timedelta(hours=1)
    token = native_jwt.encode({"sub": "x", "exp": exp}, SECRET)
    decoded = native_jwt.decode(
        token,
        SECRET,
        algorithms=["HS256"],
        options={"require": ["sub", "exp"]},
    )
    assert decoded["sub"] == "x"


# --------------------------------------------------------------------------- #
# Header introspection
# --------------------------------------------------------------------------- #


def test_get_unverified_header_returns_alg_and_typ() -> None:
    token = native_jwt.encode({"sub": "x"}, SECRET, headers={"kid": "k1"})
    header = native_jwt.get_unverified_header(token)
    assert header["alg"] == "HS256"
    assert header["typ"] == "JWT"
    assert header["kid"] == "k1"


def test_get_unverified_header_rejects_malformed_token() -> None:
    with pytest.raises(DecodeError):
        native_jwt.get_unverified_header("not-a-jwt")


# --------------------------------------------------------------------------- #
# Defensive type handling
# --------------------------------------------------------------------------- #


def test_encode_rejects_non_dict_payload() -> None:
    with pytest.raises(TypeError):
        native_jwt.encode("not a dict", SECRET)  # type: ignore[arg-type]


def test_encode_rejects_non_str_non_bytes_key() -> None:
    with pytest.raises(TypeError):
        native_jwt.encode({"sub": "x"}, 12345)  # type: ignore[arg-type]


def test_decode_rejects_non_str_non_bytes_token() -> None:
    with pytest.raises(DecodeError):
        native_jwt.decode(12345, SECRET, algorithms=["HS256"])  # type: ignore[arg-type]
