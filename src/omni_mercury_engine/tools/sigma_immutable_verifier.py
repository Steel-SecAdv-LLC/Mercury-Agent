# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Operator tool: verify the σ_Immutable training corpus signatures.

Wraps :func:`omni_mercury_engine.security.sigma_immutable_corpus.verify_corpus_signatures`
in the standard operator-tool envelope.  Installable as
``mercury-agent verify-corpus`` (see ``cli.py``) and runnable directly
via ``python -m omni_mercury_engine.tools.sigma_immutable_verifier``.

Exit codes (in addition to the package-wide set in :mod:`._base`):

* ``EXIT_OK`` — every mandatory signature verified, SHA3-256 matches.
* ``EXIT_FAIL`` — corpus / manifest tampered with, Ed25519 invalid,
  or ML-DSA-65 missing/omitted/invalid.  This is a hard install-time gate.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.sigma_immutable_verifier/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.sigma_immutable_verifier",
        description=(
            "Verify the σ_Immutable signed corpus bundle (Ed25519 + ML-DSA-65). "
            "Emits a JSON evidence certificate suitable for install-time gating."
        ),
    )
    parser.add_argument(
        "--corpus-path",
        default=None,
        help=(
            "Override the corpus JSON path. Defaults to the in-repo "
            "src/omni_mercury_engine/security/sigma_immutable_corpus.json."
        ),
    )
    parser.add_argument(
        "--sig-path",
        default=None,
        help="Override the signature JSON path (defaults to ``<corpus>.sig.json`` next to corpus).",
    )
    return parser


def _collect(args: argparse.Namespace) -> Certificate:
    from omni_mercury_engine.security.sigma_immutable_corpus import (
        CORPUS_PATH,
        CORPUS_SIG_PATH,
        CorpusVerificationError,
        verify_corpus_signatures,
    )

    corpus_path = Path(args.corpus_path) if args.corpus_path else CORPUS_PATH
    sig_path = Path(args.sig_path) if args.sig_path else CORPUS_SIG_PATH

    if not corpus_path.exists():
        return Certificate(
            tool="sigma_immutable_verifier",
            schema=_SCHEMA,
            status="fail",
            body={
                "corpus_path": str(corpus_path),
                "sig_path": str(sig_path),
                "error": "corpus file not found",
            },
        )
    if not sig_path.exists():
        return Certificate(
            tool="sigma_immutable_verifier",
            schema=_SCHEMA,
            status="fail",
            body={
                "corpus_path": str(corpus_path),
                "sig_path": str(sig_path),
                "error": "signature file not found",
            },
        )

    corpus_bytes = corpus_path.read_bytes()
    sig_bytes = sig_path.read_bytes()
    body: dict[str, Any] = {
        "corpus_path": str(corpus_path),
        "sig_path": str(sig_path),
        "corpus_size_bytes": len(corpus_bytes),
        "corpus_sha3_256": hashlib.sha3_256(corpus_bytes).hexdigest(),
        "sig_sha3_256": hashlib.sha3_256(sig_bytes).hexdigest(),
    }

    try:
        statuses = verify_corpus_signatures(corpus_path, sig_path)
    except CorpusVerificationError as exc:
        body["error"] = str(exc)
        return Certificate(
            tool="sigma_immutable_verifier",
            schema=_SCHEMA,
            status="fail",
            body=body,
        )

    body["signatures"] = statuses

    overall = "ok"
    for alg, status in statuses.items():
        if status == "verified":
            continue
        overall = "fail"

    return Certificate(
        tool="sigma_immutable_verifier",
        schema=_SCHEMA,
        status=overall,
        body=body,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
