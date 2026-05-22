"""Verify the σ_Immutable corpus signatures from the operator surface.

The σ_Immutable corpus
(``src/omni_mercury_engine/security/sigma_immutable_corpus.json``) is
the immutable training-data bundle whose features are baked into the
:class:`omni_mercury_engine.security.sigma_immutable_gate` decision
boundary.  It is signed at corpus-update time with both Ed25519
(mandatory classical signature) and ML-DSA-65 (PQC signature, gated
on the AMA Cryptography native build being available); the
signatures live in ``sigma_immutable_corpus.sig.json``.

This tool re-verifies the signatures against the on-disk corpus and
emits a machine-readable JSON report suitable for inclusion in an
audit trail.  It exists because the runtime code path
(:func:`omni_mercury_engine.security.sigma_immutable_corpus.verify_corpus_signatures`)
is only triggered during gate initialisation -- there was previously no
way for an operator to ask *"is the corpus on this machine signed and
intact?"* without standing up the full engine.

Exit codes
----------
* ``0`` -- corpus SHA3-256 matches the manifest AND every present
  signature verified.  ``ml-dsa-65`` may be reported as ``omitted`` or
  ``skipped_no_backend``; those are not failures because they document
  the platform's PQC posture rather than a corruption.
* ``1`` -- corpus integrity failure: SHA3 mismatch, missing mandatory
  signature, or a present signature failed to verify.
* ``2`` -- usage / configuration error (corpus or signature file not
  found, malformed signature payload, import-time failure of the
  cryptographic backend).

Usage
-----
::

    python -m tools.sigma_immutable_verifier
    python -m tools.sigma_immutable_verifier --strict-pqc
    python -m tools.sigma_immutable_verifier --out artifacts/sigma_verify.json

``--strict-pqc`` upgrades ``omitted`` / ``skipped_no_backend`` to a
failure (exit 1), for production environments where the ML-DSA-65
signature is required to be present and verified.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _candidate in (_REPO_ROOT, _REPO_ROOT / "src"):
    _candidate_str = str(_candidate)
    if _candidate_str not in sys.path:  # pragma: no cover - import bootstrap
        sys.path.insert(0, _candidate_str)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.sigma_immutable_verifier",
        description=(
            "Verify σ_Immutable corpus signatures (Ed25519 mandatory, "
            "ML-DSA-65 PQC if AMA Cryptography native lib is available)."
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help=(
            "Path to sigma_immutable_corpus.json.  Defaults to the shipped "
            "corpus under src/omni_mercury_engine/security/."
        ),
    )
    parser.add_argument(
        "--sig",
        type=Path,
        default=None,
        help="Path to sigma_immutable_corpus.sig.json.  Defaults to the shipped sig.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path for the audit report.",
    )
    parser.add_argument(
        "--strict-pqc",
        action="store_true",
        help=(
            "Treat an omitted or backend-unavailable ML-DSA-65 signature as a "
            "failure (exit 1).  Use in production where the PQC signature "
            "must be present and verified."
        ),
    )
    return parser


def _verify(corpus: Path | None, sig: Path | None) -> tuple[bool, dict[str, object]]:
    """Run the verification and return ``(ok, report)``.

    ``ok`` is True iff verification succeeded and no signature failed.
    The ``--strict-pqc`` interpretation is applied by the caller.
    """
    try:
        from omni_mercury_engine.security.sigma_immutable_corpus import (
            CORPUS_PATH,
            CORPUS_SIG_PATH,
            CorpusVerificationError,
            verify_corpus_signatures,
        )
    except ImportError as exc:
        return False, {"error": f"cannot import corpus module: {exc}"}

    cpath = corpus or CORPUS_PATH
    spath = sig or CORPUS_SIG_PATH
    if not cpath.exists():
        return False, {"error": f"corpus file not found: {cpath}"}
    if not spath.exists():
        return False, {"error": f"signature file not found: {spath}"}

    try:
        statuses = verify_corpus_signatures(corpus_path=cpath, sig_path=spath)
    except CorpusVerificationError as exc:
        return False, {
            "error": str(exc),
            "corpus_path": str(cpath),
            "signature_path": str(spath),
        }
    return True, {
        "corpus_path": str(cpath),
        "signature_path": str(spath),
        "statuses": statuses,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    ok, report = _verify(args.corpus, args.sig)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True))

    if not ok:
        if "error" in report and "statuses" not in report:
            # Usage / IO / import error: exit 2.
            err = str(report.get("error", ""))
            io_markers = ("not found", "cannot import")
            if any(marker in err for marker in io_markers):
                print(f"ERROR: {err}", file=sys.stderr)
                print(json.dumps(report, indent=2, sort_keys=True))
                return 2
        # Otherwise it's a verification failure: exit 1.
        print(f"FAIL: {report.get('error', 'verification failed')}", file=sys.stderr)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    statuses = report.get("statuses", {})
    if args.strict_pqc:
        weak = [
            f"{alg}={status}"
            for alg, status in statuses.items()
            if status in ("omitted", "skipped_no_backend")
        ]
        if weak:
            print(
                f"FAIL: --strict-pqc rejects weakened posture: {weak}",
                file=sys.stderr,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry-point
    raise SystemExit(main())
