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

Operator tool: verify the σ_Immutable training corpus signatures.

Wraps :func:`omni_mercury_engine.security.sigma_immutable_corpus.verify_corpus_signatures`
in the standard operator-tool envelope.  Installable as
``mercury-agent verify-corpus`` (see ``cli.py``) and runnable directly
via ``python -m omni_mercury_engine.tools.sigma_immutable_verifier``.

Exit codes (in addition to the package-wide set in :mod:`._base`):

* ``EXIT_OK`` — every present signature verified, SHA3-256 matches.
* ``EXIT_FAIL`` — corpus / manifest tampered with, Ed25519 invalid,
  or ML-DSA-65 present-but-invalid.  This is a hard install-time gate.

The ML-DSA-65 ``omitted`` and ``skipped_no_backend`` paths produce a
``"warn"`` status (because the σ_Immutable runtime gate itself still
fires under the AMA-not-built branch — failing the operator tool would
be a false negative).  Operators who need the strong PQC posture run
with ``--require-pqc`` to escalate those branches to ``EXIT_FAIL``.
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
    parser.add_argument(
        "--require-pqc",
        action="store_true",
        help=(
            "Escalate ML-DSA-65 'omitted' or 'skipped_no_backend' to a hard failure. "
            "Use in PQC-required deployments where the AMA native backend MUST be built."
        ),
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

    warnings: list[str] = []
    overall = "ok"
    for alg, status in statuses.items():
        if status == "verified":
            continue
        msg = f"{alg}: {status}"
        warnings.append(msg)
        if args.require_pqc:
            overall = "fail"

    if overall == "ok" and warnings:
        overall = "warn"

    return Certificate(
        tool="sigma_immutable_verifier",
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
