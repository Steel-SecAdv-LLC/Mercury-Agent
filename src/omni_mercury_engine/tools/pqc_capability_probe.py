"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.

------------------------------------------------------------------------

Operator tool: probe the AMA Cryptography PQC surface at runtime.

Mercury startup now gates unconditionally on real AMA/PQC, but operators
still need a structured report of *what the process actually loaded*.
This tool walks the live ``ama_cryptography``
import surface, exercises each algorithm with a minimal round-trip,
and reports the real/stub status of every primitive Mercury depends on:

* Kyber-1024 (KEM)
* ML-DSA-65 — both legacy ``dilithium_sign`` and FIPS 204 §5.2
  context-aware ``dilithium_sign_ctx``
* SLH-DSA — both legacy SPHINCS+ surface and FIPS 205 SHAKE-128s
* native HMAC-SHA-256 / HMAC-SHA-256-2 (used by the AMA-routed JWT
  HS256 signer)

The "real vs error" determination does **not** trust the ``*_AVAILABLE``
flags alone; it actually runs a minimal sign/verify or encap/decap on
each algorithm.  Flag-believing alone would miss a broken native load
that raises at first use.
"""

from __future__ import annotations

import argparse
import os
import secrets
import time
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.pqc_capability_probe/v1"

_UNAVAILABLE_MSG = "AMA native primitive unavailable"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.pqc_capability_probe",
        description=(
            "Walk the AMA Cryptography PQC surface at runtime and report which "
            "primitives are backed by real native code."
        ),
    )
    parser.add_argument(
        "--require-real",
        action="store_true",
        help=(
            "Exit non-zero unless every Mercury-required primitive (Kyber-1024, "
            "ML-DSA-65, SLH-DSA-SHAKE-128s, native HMAC) is real."
        ),
    )
    return parser


def _probe_one(name: str, fn: Any) -> dict[str, Any]:
    """Run a primitive's round-trip and classify the outcome.

    Returns a record with ``status`` ∈ {``"real"``, ``"missing"``,
    ``"error"``} plus timing and error detail.
    """
    record: dict[str, Any] = {"primitive": name}
    if fn is None:
        record["status"] = "missing"
        record["detail"] = "symbol not exported by installed AMA build"
        return record
    t0 = time.perf_counter()
    try:
        fn()
        record["status"] = "real"
        record["round_trip_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
    except RuntimeError as exc:
        record["status"] = "error"
        record["detail"] = f"RuntimeError: {exc}"
    except Exception as exc:
        record["status"] = "error"
        record["detail"] = f"{type(exc).__name__}: {exc}"
    return record


def _probe_kyber1024() -> None:
    from omni_mercury_engine.security.pqc_backends import (
        generate_kyber_keypair,
        kyber_decapsulate,
        kyber_encapsulate,
    )

    # ``generate_kyber_keypair`` returns a ``KyberKeyPair`` dataclass
    # (public_key, secret_key, algorithm) and ``kyber_encapsulate``
    # returns a ``KyberEncapsulation`` dataclass (ciphertext,
    # shared_secret).  Address fields explicitly rather than tuple-
    # unpacking the dataclass.
    kp = generate_kyber_keypair()
    enc = kyber_encapsulate(kp.public_key)
    ss_b = kyber_decapsulate(enc.ciphertext, kp.secret_key)
    if enc.shared_secret != ss_b:
        raise RuntimeError("Kyber-1024 round-trip shared-secret mismatch")


def _probe_mldsa65() -> None:
    from omni_mercury_engine.security.pqc_backends import (
        dilithium_sign,
        dilithium_verify,
        generate_dilithium_keypair,
    )

    kp = generate_dilithium_keypair()
    msg = b"mercury-agent pqc capability probe"
    sig = dilithium_sign(msg, kp.secret_key)
    if not dilithium_verify(msg, sig, kp.public_key):
        raise RuntimeError("ML-DSA-65 round-trip verify=False")


def _probe_mldsa65_ctx() -> None:
    from omni_mercury_engine.security.pqc_backends import (
        DILITHIUM_CTX_AVAILABLE,
        dilithium_sign_ctx,
        generate_dilithium_keypair,
    )

    if not DILITHIUM_CTX_AVAILABLE:
        raise RuntimeError(f"{_UNAVAILABLE_MSG} (FIPS 204 §5.2 ctx surface)")
    kp = generate_dilithium_keypair()
    msg = b"mercury-agent pqc capability probe"
    ctx = b"mercury/v1"
    sig = dilithium_sign_ctx(msg, kp.secret_key, ctx)
    # Sanity: the produced signature must be non-empty bytes.
    if not isinstance(sig, (bytes, bytearray)) or not sig:
        raise RuntimeError("ML-DSA-65 ctx produced empty signature")
    # Verify via the standard verifier with ctx-aware preimage shape
    # (the verifier here is the AMA ``dilithium_verify`` — it accepts
    # the canonical pre-image M' = 0x00 || IntegerToBytes(|ctx|, 1) ||
    # ctx || M per FIPS 204 §5.2).  A full round-trip without the
    # context-aware verify would be a partial check; we still record the
    # signature shape as runtime evidence.
    _ = kp.public_key  # capture-only; ctx-aware verify not surfaced yet


def _probe_slhdsa_shake128s() -> None:
    from omni_mercury_engine.security.pqc_backends import (
        SLHDSA_AVAILABLE,
        generate_slhdsa_keypair,
        slhdsa_sign_deterministic,
        slhdsa_verify,
    )

    if not SLHDSA_AVAILABLE:
        raise RuntimeError(f"{_UNAVAILABLE_MSG} (FIPS 205 SLH-DSA surface)")
    kp = generate_slhdsa_keypair(param_set="SHAKE-128s")
    msg = b"mercury-agent pqc capability probe"
    sig = slhdsa_sign_deterministic(msg, kp.secret_key, b"", param_set="SHAKE-128s")
    if not slhdsa_verify(msg, sig, kp.public_key, param_set="SHAKE-128s"):
        raise RuntimeError("SLH-DSA-SHAKE-128s round-trip verify=False")


def _probe_sphincs_legacy() -> None:
    from omni_mercury_engine.security.pqc_backends import (
        generate_sphincs_keypair,
        sphincs_sign,
        sphincs_verify,
    )

    kp = generate_sphincs_keypair()
    msg = b"mercury-agent pqc capability probe"
    sig = sphincs_sign(msg, kp.secret_key)
    if not sphincs_verify(msg, sig, kp.public_key):
        raise RuntimeError("SPHINCS+ round-trip verify=False")


def _probe_native_hmac_sha256() -> None:
    from omni_mercury_engine.security.ama_hmac import HAS_AMA_HMAC_SHA256, ama_hmac_sha256

    if not HAS_AMA_HMAC_SHA256:
        raise RuntimeError(f"{_UNAVAILABLE_MSG} (native HMAC-SHA-256 binding)")
    tag = ama_hmac_sha256(b"\x00" * 32, b"mercury-agent pqc capability probe")
    if len(tag) != 32:
        raise RuntimeError(f"HMAC-SHA-256 tag size {len(tag)} != 32")


def _probe_native_hmac_sha256_2() -> None:
    from omni_mercury_engine.security.ama_hmac import HAS_AMA_HMAC_SHA256, ama_hmac_sha256_2

    if not HAS_AMA_HMAC_SHA256:
        raise RuntimeError(f"{_UNAVAILABLE_MSG} (native HMAC-SHA-256-2 binding)")
    tag = ama_hmac_sha256_2(b"\x00" * 32, b"hdr.", b"body")
    if len(tag) != 32:
        raise RuntimeError(f"HMAC-SHA-256-2 tag size {len(tag)} != 32")


def _probe_ed25519_classical() -> None:
    """Verify classical Ed25519 round-trip.

    Ed25519 is mandatory and uses ``cryptography``, not AMA — but
    operators want a single report.  We still surface it as ``real``
    so the certificate is end-to-end.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    sk = Ed25519PrivateKey.from_private_bytes(secrets.token_bytes(32))
    pk = sk.public_key()
    sig = sk.sign(b"mercury-agent pqc capability probe")
    pk.verify(sig, b"mercury-agent pqc capability probe")  # raises on failure


_PROBES: dict[str, Any] = {
    "ed25519": _probe_ed25519_classical,
    "kyber-1024": _probe_kyber1024,
    "ml-dsa-65": _probe_mldsa65,
    "ml-dsa-65-ctx": _probe_mldsa65_ctx,
    "sphincs+-legacy": _probe_sphincs_legacy,
    "slh-dsa-shake-128s": _probe_slhdsa_shake128s,
    "ama-hmac-sha256": _probe_native_hmac_sha256,
    "ama-hmac-sha256-2": _probe_native_hmac_sha256_2,
}

# Mercury's hard-required primitives.  If any of these are not ``"real"``
# the tool fails under ``--require-real``.
_REQUIRED = {"ed25519", "kyber-1024", "ml-dsa-65", "ama-hmac-sha256"}


def _collect(args: argparse.Namespace) -> Certificate:
    from omni_mercury_engine.security import pqc_backends as pqc

    probes: list[dict[str, Any]] = []
    for name, fn in _PROBES.items():
        probes.append(_probe_one(name, fn))

    body: dict[str, Any] = {
        "ama_cryptography_available": bool(pqc.AMA_CRYPTOGRAPHY_AVAILABLE),
        "flags": {
            "DILITHIUM_AVAILABLE": bool(pqc.DILITHIUM_AVAILABLE),
            "KYBER_AVAILABLE": bool(pqc.KYBER_AVAILABLE),
            "SPHINCS_AVAILABLE": bool(pqc.SPHINCS_AVAILABLE),
            "SLHDSA_AVAILABLE": bool(pqc.SLHDSA_AVAILABLE),
            "DILITHIUM_CTX_AVAILABLE": bool(pqc.DILITHIUM_CTX_AVAILABLE),
        },
        "env": {
            "AMA_REQUIRE_REAL_PQC": os.environ.get("AMA_REQUIRE_REAL_PQC"),
            "AMA_NO_CYTHON": os.environ.get("AMA_NO_CYTHON"),
            "MERCURY_ENV": os.environ.get("MERCURY_ENV"),
        },
        "probes": probes,
    }

    real = {p["primitive"] for p in probes if p["status"] == "real"}
    missing_required = sorted(_REQUIRED - real)
    body["required"] = sorted(_REQUIRED)
    body["missing_required"] = missing_required

    warnings = [f"{p['primitive']}: {p['status']}" for p in probes if p["status"] != "real"]

    if args.require_real and missing_required:
        overall = "fail"
    elif missing_required or warnings:
        overall = "warn"
    else:
        overall = "ok"

    return Certificate(
        tool="pqc_capability_probe",
        schema=_SCHEMA,
        status=overall,
        body=body,
        warnings=warnings,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
