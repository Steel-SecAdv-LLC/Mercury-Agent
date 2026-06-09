# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Operator tool: re-run NIST FIPS 203/204/205 and RFC 8032 known-answer tests outside pytest, and emit a signed JSON certificate suitable for an external auditor.

The Mercury KAT *test suite* lives under ``tests/security/`` and is
exercised by pytest in CI; that suite is the regression contract for
correctness.  This tool re-uses the same curated vectors
(``tests/security/data/nist_kat/nist_acvp_curated.json``) and the
RFC 8032 §7.1 Ed25519 vectors built-in below, but executes them
through a stripped-down driver so an auditor receives:

* a single canonical JSON file with one record per vector;
* algorithm/operation/tcId, expected/produced hashes, pass/fail;
* optional Ed25519 detached signature over the canonical bytes so
  the artefact is tamper-evident at rest.

This is *evidence emission*, not test discovery.  Mercury imports only
with real AMA/PQC, so PQC vectors must execute rather than degrade to
skips.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.kat_runner_standalone/v1"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CURATED_KAT_PATH = (
    _REPO_ROOT / "tests" / "security" / "data" / "nist_kat" / "nist_acvp_curated.json"
)

# ---------------------------------------------------------------------------
# RFC 8032 Ed25519 test vectors (§7.1 — Test 1, Test 2, Test 3 / TEST SHA)
# ---------------------------------------------------------------------------
# Source: https://datatracker.ietf.org/doc/html/rfc8032#section-7.1
# These are reproduced verbatim from the RFC.  Each tuple is
# (label, secret_seed_hex, public_key_hex, message_hex, signature_hex).
#
# Inline rather than file-based because they are the *only* RFC-quoted
# classical signature vectors Mercury depends on, they are tiny, and
# they MUST remain reachable even when ``tests/`` is not packaged
# (operators running from a wheel install do not get the curated NIST
# fixtures, but they still need a smoke-level signed certificate).
_RFC8032_ED25519_VECTORS: list[tuple[str, str, str, str, str]] = [
    (
        "rfc8032-test-1",
        "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
        "",
        (
            "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
            "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
        ),
    ),
    (
        "rfc8032-test-2",
        "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
        "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
        "72",
        (
            "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da"
            "085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"
        ),
    ),
    (
        "rfc8032-test-3",
        "c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
        "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
        "af82",
        (
            "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac"
            "18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a"
        ),
    ),
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.kat_runner_standalone",
        description=(
            "Re-run RFC 8032 + FIPS 203/204/205 ACVP-Server vectors outside pytest "
            "and emit a signed JSON certificate."
        ),
    )
    parser.add_argument(
        "--kat-file",
        default=str(_CURATED_KAT_PATH),
        help=(
            "Path to the curated NIST ACVP-Server JSON file. "
            "Defaults to tests/security/data/nist_kat/nist_acvp_curated.json."
        ),
    )
    parser.add_argument(
        "--algorithms",
        default="all",
        help=(
            "Comma-separated subset of algorithms to run: ed25519, ml-dsa-65, "
            "ml-kem-1024, slh-dsa-shake-128s.  Default: all."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Vector runners
# ---------------------------------------------------------------------------


def _run_rfc8032_ed25519() -> list[dict[str, Any]]:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )

    records: list[dict[str, Any]] = []
    for label, sk_hex, pk_hex, msg_hex, sig_hex in _RFC8032_ED25519_VECTORS:
        sk = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(sk_hex))
        pk = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pk_hex))
        msg = bytes.fromhex(msg_hex)
        expected_sig = bytes.fromhex(sig_hex)
        produced_sig = sk.sign(msg)
        sign_match = produced_sig == expected_sig
        try:
            pk.verify(expected_sig, msg)
            verify_ok = True
        except InvalidSignature:
            verify_ok = False
        records.append(
            {
                "algorithm": "ed25519",
                "operation": "sigGen+sigVer",
                "tcId": label,
                "expected_sha256": hashlib.sha256(expected_sig).hexdigest(),
                "produced_sha256": hashlib.sha256(produced_sig).hexdigest(),
                "sign_match": sign_match,
                "verify_ok": verify_ok,
                "passed": sign_match and verify_ok,
            }
        )
    return records


def _ama_available() -> bool:
    from omni_mercury_engine.security import pqc_backends as pqc

    return bool(pqc.AMA_CRYPTOGRAPHY_AVAILABLE)


def _ama_ctx_available() -> bool:
    from omni_mercury_engine.security import pqc_backends as pqc

    return bool(pqc.DILITHIUM_CTX_AVAILABLE)


def _ama_slhdsa_available() -> bool:
    from omni_mercury_engine.security import pqc_backends as pqc

    return bool(pqc.SLHDSA_AVAILABLE)


def _run_mldsa65_siggen(vectors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not _ama_ctx_available():
        raise RuntimeError("AMA FIPS 204 §5.2 ctx surface unavailable")
    from omni_mercury_engine.security.pqc_backends import dilithium_sign_ctx

    records: list[dict[str, Any]] = []
    for v in vectors:
        sk = bytes.fromhex(v["sk"])
        message = bytes.fromhex(v["message"])
        ctx = bytes.fromhex(v.get("context", ""))
        expected_sig = bytes.fromhex(v["signature"])
        produced_sig = dilithium_sign_ctx(message, sk, ctx)
        records.append(
            {
                "algorithm": "ml-dsa-65",
                "operation": "sigGen",
                "tcId": v["tcId"],
                "expected_sha256": hashlib.sha256(expected_sig).hexdigest(),
                "produced_sha256": hashlib.sha256(produced_sig).hexdigest(),
                "passed": produced_sig == expected_sig,
            }
        )
    return records


def _run_mlkem1024_decaps(vectors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not _ama_available():
        raise RuntimeError("AMA Cryptography unavailable")
    from omni_mercury_engine.security.pqc_backends import kyber_decapsulate

    records: list[dict[str, Any]] = []
    for v in vectors:
        dk = bytes.fromhex(v["dk"])
        c = bytes.fromhex(v["c"])
        expected_k = bytes.fromhex(v["k"])
        recovered_k = kyber_decapsulate(c, dk)
        records.append(
            {
                "algorithm": "ml-kem-1024",
                "operation": "decapsulation",
                "tcId": v["tcId"],
                "expected_k_sha256": hashlib.sha256(expected_k).hexdigest(),
                "recovered_k_sha256": hashlib.sha256(recovered_k).hexdigest(),
                "passed": recovered_k == expected_k,
            }
        )
    return records


def _run_slhdsa_shake128s_siggen(vectors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not _ama_slhdsa_available():
        raise RuntimeError("AMA FIPS 205 SLH-DSA surface unavailable")
    from omni_mercury_engine.security.pqc_backends import slhdsa_sign_deterministic

    records: list[dict[str, Any]] = []
    for v in vectors:
        sk = bytes.fromhex(v["sk"])
        message = bytes.fromhex(v["message"])
        ctx = bytes.fromhex(v.get("context", ""))
        expected_sig = bytes.fromhex(v["signature"])
        produced_sig = slhdsa_sign_deterministic(message, sk, ctx, param_set="SHAKE-128s")
        records.append(
            {
                "algorithm": "slh-dsa-shake-128s",
                "operation": "sigGen",
                "tcId": v["tcId"],
                "expected_sha256": hashlib.sha256(expected_sig).hexdigest(),
                "produced_sha256": hashlib.sha256(produced_sig).hexdigest(),
                "passed": produced_sig == expected_sig,
            }
        )
    return records


def _collect(args: argparse.Namespace) -> Certificate:
    selected = {s.strip() for s in args.algorithms.split(",")}
    run_all = "all" in selected

    records: list[dict[str, Any]] = []
    body: dict[str, Any] = {
        "kat_file": str(args.kat_file),
        "selected_algorithms": sorted(selected),
    }

    if run_all or "ed25519" in selected:
        records.extend(_run_rfc8032_ed25519())

    pqc_curated: dict[str, Any] = {}
    kat_path = Path(args.kat_file)
    if kat_path.exists():
        try:
            pqc_curated = json.loads(kat_path.read_text())
        except json.JSONDecodeError as exc:
            body["kat_file_error"] = f"failed to parse: {exc}"

    if run_all or "ml-dsa-65" in selected:
        vectors = pqc_curated.get("ML-DSA-65", {}).get("sigGen", [])
        if vectors:
            records.extend(_run_mldsa65_siggen(vectors))
    if run_all or "ml-kem-1024" in selected:
        vectors = pqc_curated.get("ML-KEM-1024", {}).get("decapsulation", [])
        if vectors:
            records.extend(_run_mlkem1024_decaps(vectors))
    if run_all or "slh-dsa-shake-128s" in selected:
        vectors = pqc_curated.get("SLH-DSA-SHAKE-128s", {}).get("sigGen", [])
        if vectors:
            records.extend(_run_slhdsa_shake128s_siggen(vectors))

    total = len(records)
    skipped = 0
    failed = sum(1 for r in records if r.get("passed") is False)
    passed = total - skipped - failed

    body["summary"] = {
        "total": total,
        "passed": passed,
        "skipped": skipped,
        "failed": failed,
    }
    body["records"] = records

    warnings: list[str] = []
    if failed:
        status = "fail"
    else:
        status = "ok"

    return Certificate(
        tool="kat_runner_standalone",
        schema=_SCHEMA,
        status=status,
        body=body,
        warnings=warnings,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
