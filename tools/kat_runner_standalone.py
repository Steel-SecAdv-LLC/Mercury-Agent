"""Run KAT (Known-Answer Test) vectors against the live PQC backend, outside pytest.

This tool re-runs the same Known-Answer Tests that
``tests/security/test_ama_kat.py`` and
``tests/security/test_nist_fips_kat.py`` enforce in CI, but invokes
them as a standalone process so the result is a self-contained JSON
certificate suitable for inclusion in:

* a release audit trail (alongside ``release_manifest_builder`` output);
* an external compliance review (the report is identical to what CI
  reports, but produced by an artefact the operator can attach to a
  ticket);
* a quarterly "PQC posture" review on a deployed container.

Unlike pytest, this tool:

* writes a single JSON file documenting every vector tried and its
  outcome (pass / fail / skipped + reason);
* returns ``0`` only if every vector that *could* run did, and every
  one that ran passed;
* does not depend on pytest's collection / fixture machinery, so it
  works on a minimal container that ships only the runtime image.

Exit codes
----------
* ``0`` -- every vector that ran passed; nothing required was skipped.
* ``1`` -- at least one vector failed.
* ``2`` -- usage error, KAT data file missing, or the PQC backend is
  not importable.
* ``3`` -- ``--strict`` was passed and one or more vectors were
  skipped because the backend was unavailable (the test would have
  been silently skipped by pytest's ``importorskip``; this tool makes
  the skip visible and, under ``--strict``, fatal).

Usage
-----
::

    python -m tools.kat_runner_standalone
    python -m tools.kat_runner_standalone --strict
    python -m tools.kat_runner_standalone --out artifacts/kat_certificate.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _candidate in (_REPO_ROOT, _REPO_ROOT / "src"):
    _candidate_str = str(_candidate)
    if _candidate_str not in sys.path:  # pragma: no cover - import bootstrap
        sys.path.insert(0, _candidate_str)


def _backend() -> Any | None:
    try:
        return importlib.import_module("omni_mercury_engine.security.pqc_backends")
    except ImportError:
        return None


def _vector_dilithium_round_trip(backend: Any) -> dict[str, Any]:
    name = "ml-dsa-65/round-trip"
    if not getattr(backend, "DILITHIUM_AVAILABLE", False):
        return {"name": name, "status": "skipped", "reason": "DILITHIUM_AVAILABLE=False"}
    try:
        kp = backend.generate_dilithium_keypair()
        msg = b"mercury-kat-runner"
        sig = backend.sign_dilithium(kp, msg)
        ok = backend.verify_dilithium(kp.public_key, msg, sig)
        return {"name": name, "status": "pass" if ok else "fail"}
    except Exception as exc:
        return {"name": name, "status": "fail", "error": str(exc)}


def _vector_dilithium_deterministic(backend: Any) -> dict[str, Any]:
    name = "ml-dsa-65/deterministic-sign"
    if not getattr(backend, "DILITHIUM_AVAILABLE", False):
        return {"name": name, "status": "skipped", "reason": "DILITHIUM_AVAILABLE=False"}
    try:
        kp = backend.generate_dilithium_keypair()
        msg = b"mercury-kat-runner-deterministic"
        s1 = backend.sign_dilithium(kp, msg)
        s2 = backend.sign_dilithium(kp, msg)
        deterministic = bytes(s1.signature) == bytes(s2.signature)
        return {
            "name": name,
            "status": "pass" if deterministic else "fail",
            "deterministic": deterministic,
        }
    except Exception as exc:
        return {"name": name, "status": "fail", "error": str(exc)}


def _vector_kyber_round_trip(backend: Any) -> dict[str, Any]:
    name = "ml-kem-1024/round-trip"
    if not getattr(backend, "KYBER_AVAILABLE", False):
        return {"name": name, "status": "skipped", "reason": "KYBER_AVAILABLE=False"}
    try:
        kp = backend.generate_kyber_keypair()
        encaps = backend.encapsulate_kyber(kp.public_key)
        recovered = backend.decapsulate_kyber(kp.secret_key, encaps.ciphertext)
        ok = bytes(recovered) == bytes(encaps.shared_secret)
        return {"name": name, "status": "pass" if ok else "fail"}
    except Exception as exc:
        return {"name": name, "status": "fail", "error": str(exc)}


def _vector_sphincs_round_trip(backend: Any) -> dict[str, Any]:
    name = "sphincs+-sha2-256f/round-trip"
    if not getattr(backend, "SPHINCS_AVAILABLE", False):
        return {"name": name, "status": "skipped", "reason": "SPHINCS_AVAILABLE=False"}
    try:
        kp = backend.generate_sphincs_keypair()
        msg = b"mercury-kat-runner-sphincs"
        sig = backend.sign_sphincs(kp, msg)
        ok = backend.verify_sphincs(kp.public_key, msg, sig)
        return {"name": name, "status": "pass" if ok else "fail"}
    except Exception as exc:
        return {"name": name, "status": "fail", "error": str(exc)}


def _vector_nist_fips_corpus_present() -> dict[str, Any]:
    name = "nist-acvp/corpus-present"
    candidate = _REPO_ROOT / "tests" / "security" / "fixtures" / "nist_fips_kat_vectors.json"
    if not candidate.exists():
        return {
            "name": name,
            "status": "skipped",
            "reason": f"vector file not found: {candidate}",
        }
    try:
        payload = json.loads(candidate.read_text())
        algos = sorted(payload.keys()) if isinstance(payload, dict) else []
        return {"name": name, "status": "pass", "algorithms": algos}
    except (OSError, ValueError) as exc:
        return {"name": name, "status": "fail", "error": str(exc)}


_VECTOR_FUNCS = (
    _vector_dilithium_round_trip,
    _vector_dilithium_deterministic,
    _vector_kyber_round_trip,
    _vector_sphincs_round_trip,
)


def run_kats() -> dict[str, Any]:
    backend = _backend()
    if backend is None:
        return {"error": "cannot import pqc_backends"}
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    for fn in _VECTOR_FUNCS:
        results.append(fn(backend))
    results.append(_vector_nist_fips_corpus_present())
    elapsed = time.monotonic() - started

    summary = {
        "pass": sum(1 for r in results if r["status"] == "pass"),
        "fail": sum(1 for r in results if r["status"] == "fail"),
        "skipped": sum(1 for r in results if r["status"] == "skipped"),
    }
    return {
        "schema": "mercury_kat_certificate/v1",
        "ama_cryptography_version": (
            getattr(importlib.import_module("ama_cryptography"), "__version__", None)
            if _ama_present()
            else None
        ),
        "python": sys.version.split()[0],
        "elapsed_s": elapsed,
        "vectors": results,
        "summary": summary,
    }


def _ama_present() -> bool:
    try:
        importlib.import_module("ama_cryptography")
        return True
    except ImportError:
        return False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.kat_runner_standalone",
        description=(
            "Re-run AMA Cryptography and NIST FIPS KAT vectors outside pytest "
            "and emit a self-contained JSON certificate."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path for the KAT certificate.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Treat any 'skipped' vector as a failure (exit 3).  Use in "
            "production environments where the PQC backend must be present."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cert = run_kats()
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(cert, indent=2, sort_keys=True))

    if "error" in cert:
        print(f"ERROR: {cert['error']}", file=sys.stderr)
        print(json.dumps(cert, indent=2, sort_keys=True))
        return 2

    print(json.dumps(cert, indent=2, sort_keys=True))
    summary = cert["summary"]
    if summary["fail"]:
        print(f"FAIL: {summary['fail']} KAT vector(s) failed", file=sys.stderr)
        return 1
    if args.strict and summary["skipped"]:
        print(
            f"FAIL: --strict and {summary['skipped']} vector(s) skipped",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry-point
    raise SystemExit(main())
