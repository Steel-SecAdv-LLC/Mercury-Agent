"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

------------------------------------------------------------------------

Operator tool: assemble a signed release bundle (tarball) of the
existing per-tool evidence artefacts.

The bundle pins:

* ``release_manifest.json`` — the :mod:`release_manifest_builder` cert,
* ``sbom.json`` — the :mod:`sbom_emitter` CycloneDX output,
* ``slsa.json`` — the :mod:`slsa_provenance_emitter` attestation,
* ``corpus.sig.json`` — the σ_Immutable corpus signature manifest,
* ``kat_cert.json`` — the :mod:`kat_runner_standalone` certificate.

The bundle is sealed with **both** Ed25519 and ML-DSA-65 — mirroring
the σ_Immutable corpus signature scheme.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import (
    Certificate,
    atomic_write_bytes,
    run_tool,
    sign_certificate_ed25519,
)

_SCHEMA = "mercury.tools.signed_release_bundle/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.signed_release_bundle",
        description=("Assemble a signed release bundle from per-tool evidence artefacts."),
    )
    parser.add_argument(
        "--bundle-path",
        required=True,
        help="Output tarball path (sealed atomically).",
    )
    parser.add_argument(
        "--evidence",
        action="append",
        default=[],
        help=(
            "Repeatable path to include in the bundle, in `name=path` form "
            "(e.g. --evidence sbom.json=release/sbom.json)."
        ),
    )
    parser.add_argument(
        "--bundle-sign-key-hex",
        default=None,
        help=(
            "Ed25519 secret-seed hex used to sign the bundle.  Generated "
            "ephemerally if omitted; the public key is captured in the cert."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the bundle but do NOT write to disk.",
    )
    return parser


def _ml_dsa_sign(payload: bytes) -> dict[str, Any]:
    from omni_mercury_engine.security.crypto_api import (
        AlgorithmType,
        MercuryCrypto,
        SecurityLevel,
    )

    # ``SecurityLevel.HYBRID`` enables both classical and PQC signing paths
    # through the AMA-Cryptography providers.
    crypto = MercuryCrypto(security_level=SecurityLevel.HYBRID)
    kp = crypto.generate_signing_keypair(algorithm=AlgorithmType.ML_DSA_65)
    sig = crypto.sign(payload, kp.secret_key, algorithm=AlgorithmType.ML_DSA_65)
    return {
        "algorithm": "ml-dsa-65",
        "public_key_hex": kp.public_key.hex(),
        "signature_hex": sig.signature.hex(),
        "payload_sha3_256": hashlib.sha3_256(payload).hexdigest(),
    }


def _collect(args: argparse.Namespace) -> Certificate:
    entries: list[tuple[str, Path]] = []
    missing: list[str] = []
    for spec in args.evidence:
        if "=" not in spec:
            return Certificate(
                tool="signed_release_bundle",
                schema=_SCHEMA,
                status="fail",
                body={"error": f"invalid --evidence spec {spec!r}: missing '='"},
            )
        name, path = spec.split("=", 1)
        p = Path(path)
        if not p.is_file():
            missing.append(spec)
            continue
        entries.append((name, p))

    if missing:
        return Certificate(
            tool="signed_release_bundle",
            schema=_SCHEMA,
            status="fail",
            body={"missing": missing},
            warnings=[f"missing evidence: {m}" for m in missing],
        )

    # Build the tar in-memory so we can compute the digest before
    # committing to disk.  Deterministic: sorted entries, fixed mtime.
    buf = BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.PAX_FORMAT) as tar:
        for name, p in sorted(entries):
            data = p.read_bytes()
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = "mercury"
            info.gname = "mercury"
            info.mode = 0o644
            tar.addfile(info, BytesIO(data))
    bundle_bytes = buf.getvalue()
    bundle_digest = hashlib.sha256(bundle_bytes).hexdigest()
    payload_sha3 = hashlib.sha3_256(bundle_bytes).hexdigest()

    bundle_sign_key = args.bundle_sign_key_hex or secrets.token_hex(32)
    ed_sig = sign_certificate_ed25519(bundle_bytes, bundle_sign_key)
    mldsa_sig = _ml_dsa_sign(bundle_bytes)

    body: dict[str, Any] = {
        "bundle_path": args.bundle_path,
        "entries": [name for name, _ in sorted(entries)],
        "bundle_sha256": bundle_digest,
        "payload_sha3_256": payload_sha3,
        "signatures": {
            "ed25519": ed_sig,
            "ml-dsa-65": mldsa_sig,
        },
        "dry_run": bool(args.dry_run),
    }

    if not args.dry_run:
        atomic_write_bytes(Path(args.bundle_path), bundle_bytes)
        sig_path = Path(args.bundle_path + ".sig.json")
        atomic_write_bytes(
            sig_path,
            (
                json.dumps(
                    {
                        "schema": "mercury.signed_release_bundle.signatures/v1",
                        "bundle_sha256": bundle_digest,
                        "payload_sha3_256": payload_sha3,
                        "signatures": body["signatures"],
                    },
                    sort_keys=True,
                    indent=2,
                )
                + "\n"
            ).encode("utf-8"),
        )
        body["sig_path"] = str(sig_path)

    return Certificate(
        tool="signed_release_bundle",
        schema=_SCHEMA,
        status="ok",
        body=body,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
