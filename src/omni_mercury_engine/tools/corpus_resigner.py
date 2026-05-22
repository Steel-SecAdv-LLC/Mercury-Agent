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

Operator tool: re-sign ``sigma_immutable_corpus.json`` atomically.

When operators rotate the σ_Immutable corpus (new positive/negative
samples, threshold adjustment, etc.) the
``sigma_immutable_corpus.sig.json`` manifest must be re-signed under
both Ed25519 and ML-DSA-65.  This was previously implicit operator
knowledge — wrap the official primitives behind a single CLI that:

* recomputes the SHA3-256 of the current corpus on disk;
* generates fresh Ed25519 + ML-DSA-65 keypairs and signs the corpus;
* writes the manifest *atomically* (temp-file + os.replace) so a
  crashed run leaves the previous signature intact rather than a
  half-written file;
* verifies the freshly written manifest before declaring success.

The tool intentionally does NOT regenerate the corpus from a seed —
that workflow is owned by ``scripts/train_sigma_immutable.py`` and
mixes in the trained-weights step.  This tool re-signs the corpus
that is *already on disk*; if the corpus needs regeneration use
the trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.corpus_resigner/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.corpus_resigner",
        description=(
            "Re-sign the σ_Immutable corpus on disk with Ed25519 (mandatory) "
            "and ML-DSA-65 (when the AMA native PQC backend is built). "
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
    return parser


def _atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically.

    Uses ``tempfile.NamedTemporaryFile`` in the same directory followed by
    ``os.replace`` — POSIX guarantees same-directory rename is atomic on
    every supported filesystem (ext4, xfs, btrfs, apfs).  A crash mid-write
    leaves either the old file untouched or the new file in place; never
    a half-written manifest.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        delete=False,
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp.name, path)
    except Exception:
        # Best-effort cleanup of the temp file on any failure path.
        try:
            tmp.close()
        except Exception:
            pass
        try:
            os.unlink(tmp.name)
        except FileNotFoundError:
            pass
        raise


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

    # ML-DSA-65 — best-effort; honest omission on absence.
    mldsa_status: str
    try:
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
        mldsa_status = "signed"
    except RuntimeError as exc:
        signature_payload["signatures"][MLDSA65_ALG] = {
            "algorithm": MLDSA65_ALG,
            "omitted": True,
            "omission_reason": str(exc),
        }
        mldsa_status = "omitted"

    sig_bytes = (json.dumps(signature_payload, sort_keys=True, indent=2) + "\n").encode("utf-8")

    body: dict[str, Any] = {
        "corpus_path": str(corpus_path),
        "sig_path": str(sig_path),
        "corpus_sha3_256": corpus_digest,
        "ed25519": "signed",
        "ml_dsa_65": mldsa_status,
        "new_sig_sha3_256": hashlib.sha3_256(sig_bytes).hexdigest(),
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        return Certificate(
            tool="corpus_resigner",
            schema=_SCHEMA,
            status="ok",
            body=body,
            warnings=["dry-run: manifest was NOT written"],
        )

    _atomic_write(sig_path, sig_bytes)

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
