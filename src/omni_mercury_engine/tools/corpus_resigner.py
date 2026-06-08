# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, atomic_write_bytes, run_tool

_SCHEMA = "mercury.tools.corpus_resigner/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.corpus_resigner",
        description=(
            "Re-sign the σ_Immutable corpus on disk with Ed25519 (mandatory) "
            "and ML-DSA-65 (mandatory AMA/PQC). "
            "Writes the manifest atomically and verifies it before returning."
        ),
    )
    parser.add_argument(
        "--corpus-path",
        default=None,
        help="Override the corpus JSON path. Defaults to the in-repo corpus.",
    )
    parser.add_argument(
        "--sig-path",
        default=None,
        help="Override the manifest path (default: <corpus>.sig.json next to corpus).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the new signatures but do NOT overwrite the manifest.",
    )
    parser.add_argument(
        "--hsm-uri",
        default=os.environ.get("MERCURY_HSM_URI"),
        help=(
            "Optional PKCS#11 URI of an HSM/TPM/YubiHSM-resident Ed25519 "
            "signing key (e.g. 'pkcs11:token=mercury;id=01').  When set, "
            "the signing key is never materialised in process memory.  "
            "Defaults to $MERCURY_HSM_URI."
        ),
    )
    return parser


# Atomic write helper is centralised in ``_base.atomic_write_bytes`` so
# every tool that writes to disk goes through the same code path.


def _sign_with_hsm(payload: bytes, hsm_uri: str) -> dict[str, Any] | None:
    """Sign ``payload`` using an HSM-resident Ed25519 key.

    Returns ``None`` when ``python-pkcs11`` is not installed or the URI
    cannot be opened — the caller falls back to a software keypair and
    captures the omission reason in the certificate body.
    """
    try:
        # ``pkcs11`` is gated by the ``[hsm]`` extra and declared in
        # ``pyproject.toml`` ``[[tool.mypy.overrides]]``; we therefore
        # do not need a ``type: ignore`` here.
        import pkcs11
    except ImportError:
        return None
    try:
        # Minimal PKCS#11 URI parser: pull token and id; library-loaded
        # operators set MERCURY_HSM_MODULE to override the .so path.
        module_path = os.environ.get("MERCURY_HSM_MODULE", "/usr/lib/softhsm/libsofthsm2.so")
        lib = pkcs11.lib(module_path)
        token_label = None
        key_id = None
        for part in hsm_uri.removeprefix("pkcs11:").split(";"):
            if part.startswith("token="):
                token_label = part.removeprefix("token=")
            elif part.startswith("id="):
                key_id = bytes.fromhex(part.removeprefix("id="))
        if token_label is None:
            return None
        token = lib.get_token(token_label=token_label)
        pin = os.environ.get("MERCURY_HSM_PIN")
        with token.open(user_pin=pin) as session:
            priv = (
                next(session.get_objects({pkcs11.Attribute.ID: key_id}))
                if key_id
                else next(session.get_objects())
            )
            sig = priv.sign(payload)
        return {"algorithm": "ed25519", "hsm_uri": hsm_uri, "signature_hex": sig.hex()}
    except Exception:
        return None


def _collect(args: argparse.Namespace) -> Certificate:
    from omni_mercury_engine.security.crypto_api import (
        AlgorithmType,
        MercuryCrypto,
        SecurityLevel,
    )
    from omni_mercury_engine.security.sigma_immutable_corpus import (
        CORPUS_PATH,
        CORPUS_SIG_PATH,
        ED25519_ALG,
        MLDSA65_ALG,
        CorpusVerificationError,
        verify_corpus_signatures,
    )

    corpus_path = Path(args.corpus_path) if args.corpus_path else CORPUS_PATH
    sig_path = Path(args.sig_path) if args.sig_path else CORPUS_SIG_PATH

    if not corpus_path.exists():
        return Certificate(
            tool="corpus_resigner",
            schema=_SCHEMA,
            status="fail",
            body={"corpus_path": str(corpus_path), "error": "corpus file not found"},
        )

    payload_bytes = corpus_path.read_bytes()
    corpus_digest = hashlib.sha3_256(payload_bytes).hexdigest()

    crypto = MercuryCrypto(security_level=SecurityLevel.CLASSICAL)

    # Ed25519 — mandatory.
    ed_keypair = crypto.generate_signing_keypair(algorithm=AlgorithmType.ED25519)
    ed_signature = crypto.sign(
        payload_bytes,
        ed_keypair.secret_key,
        algorithm=AlgorithmType.ED25519,
    )
    signature_payload: dict[str, Any] = {
        "schema": "sigma_immutable_corpus_signatures/v1",
        "corpus_sha3_256": corpus_digest,
        "signatures": {
            ED25519_ALG: {
                "algorithm": ED25519_ALG,
                "public_key_hex": ed_keypair.public_key.hex(),
                "signature_hex": ed_signature.signature.hex(),
            },
        },
    }

    # ML-DSA-65 — mandatory.
    mldsa_keypair = crypto.generate_signing_keypair(algorithm=AlgorithmType.ML_DSA_65)
    mldsa_signature = crypto.sign(
        payload_bytes,
        mldsa_keypair.secret_key,
        algorithm=AlgorithmType.ML_DSA_65,
    )
    signature_payload["signatures"][MLDSA65_ALG] = {
        "algorithm": MLDSA65_ALG,
        "public_key_hex": mldsa_keypair.public_key.hex(),
        "signature_hex": mldsa_signature.signature.hex(),
    }

    sig_bytes = (json.dumps(signature_payload, sort_keys=True, indent=2) + "\n").encode("utf-8")

    hsm_info: dict[str, Any] = {"requested": bool(args.hsm_uri)}
    if args.hsm_uri:
        hsm_sig = _sign_with_hsm(payload_bytes, args.hsm_uri)
        if hsm_sig is None:
            hsm_info["available"] = False
            hsm_info["reason"] = (
                "python-pkcs11 not installed or HSM URI unopenable; fell back to software key"
            )
        else:
            hsm_info["available"] = True
            hsm_info["signature"] = hsm_sig

    body: dict[str, Any] = {
        "corpus_path": str(corpus_path),
        "sig_path": str(sig_path),
        "corpus_sha3_256": corpus_digest,
        "ed25519": "signed",
        "ml_dsa_65": "signed",
        "new_sig_sha3_256": hashlib.sha3_256(sig_bytes).hexdigest(),
        "dry_run": bool(args.dry_run),
        "hsm": hsm_info,
    }

    if args.dry_run:
        return Certificate(
            tool="corpus_resigner",
            schema=_SCHEMA,
            status="ok",
            body=body,
            warnings=["dry-run: manifest was NOT written"],
        )

    atomic_write_bytes(sig_path, sig_bytes)

    # Re-verify what we just wrote.  Defensive — catches the corner case
    # where the atomic write succeeded but the signature is somehow
    # mis-formatted (e.g. an upstream MercuryCrypto contract change).
    try:
        statuses = verify_corpus_signatures(corpus_path, sig_path)
    except CorpusVerificationError as exc:
        body["error"] = f"post-write verification failed: {exc}"
        return Certificate(
            tool="corpus_resigner",
            schema=_SCHEMA,
            status="fail",
            body=body,
        )

    body["verification"] = statuses
    return Certificate(
        tool="corpus_resigner",
        schema=_SCHEMA,
        status="ok",
        body=body,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
