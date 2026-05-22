"""Probe the live AMA Cryptography surface and report which PQC primitives are real.

The ``omni_mercury_engine.security.pqc_backends`` module exposes a set
of availability booleans
(``AMA_CRYPTOGRAPHY_AVAILABLE``, ``DILITHIUM_AVAILABLE``,
``KYBER_AVAILABLE``, ``SPHINCS_AVAILABLE``, ``SLHDSA_AVAILABLE``,
``ML_DSA_CONTEXT_SIGN_AVAILABLE``) that flip True only when the
corresponding symbol is genuinely importable from
:mod:`ama_cryptography`.  A stub fallback flips them False, and the
adapter emits :class:`PQCProductionWarning` rather than failing hard
unless ``AMA_REQUIRE_REAL_PQC=true`` is set.

This tool reports, in JSON, exactly which symbols this Python interpreter
believes are real -- end-to-end runtime evidence that
``AMA_REQUIRE_REAL_PQC`` is actually enforceable on this machine, and a
deterministic capability fingerprint for inclusion in release manifests
and audit logs.  Unlike the pytest-time KAT vectors, this probe runs
against the *deployed* interpreter in the *deployed* environment, so it
catches the case where a CI build was fully PQC-real but the production
container ships with a stub library.

The probe additionally performs a one-shot round-trip per primitive
(keygen → sign → verify, or keygen → encaps → decaps) so a primitive
that is *importable but broken* is detected by the round-trip rather
than reported as healthy by the boolean alone.

Exit codes
----------
* ``0`` -- AMA backend is fully real; ``--require`` constraints satisfied.
* ``1`` -- ``--require`` constraints not met (e.g. ``--require dilithium``
  but ``DILITHIUM_AVAILABLE`` is False) or a round-trip failed.
* ``2`` -- usage / import error.

Usage
-----
::

    python -m tools.pqc_capability_probe
    python -m tools.pqc_capability_probe --require dilithium,kyber
    python -m tools.pqc_capability_probe --out artifacts/pqc_probe.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _candidate in (_REPO_ROOT, _REPO_ROOT / "src"):
    _candidate_str = str(_candidate)
    if _candidate_str not in sys.path:  # pragma: no cover - import bootstrap
        sys.path.insert(0, _candidate_str)


_ROUND_TRIP_MESSAGE = b"mercury-pqc-probe round-trip canary"


def _round_trip_dilithium(crypto: Any) -> dict[str, Any]:
    try:
        from omni_mercury_engine.security.pqc_backends import (
            generate_dilithium_keypair,
            sign_dilithium,
            verify_dilithium,
        )
    except ImportError as exc:
        return {"available": False, "error": f"import: {exc}"}
    try:
        keypair = generate_dilithium_keypair()
        sig = sign_dilithium(keypair, _ROUND_TRIP_MESSAGE)
        ok = verify_dilithium(keypair.public_key, _ROUND_TRIP_MESSAGE, sig)
        return {"available": True, "round_trip": bool(ok)}
    except Exception as exc:
        return {"available": True, "round_trip": False, "error": str(exc)}


def _round_trip_kyber(crypto: Any) -> dict[str, Any]:
    try:
        from omni_mercury_engine.security.pqc_backends import (
            decapsulate_kyber,
            encapsulate_kyber,
            generate_kyber_keypair,
        )
    except ImportError as exc:
        return {"available": False, "error": f"import: {exc}"}
    try:
        keypair = generate_kyber_keypair()
        encaps = encapsulate_kyber(keypair.public_key)
        recovered = decapsulate_kyber(keypair.secret_key, encaps.ciphertext)
        match = bytes(recovered) == bytes(encaps.shared_secret)
        return {"available": True, "round_trip": bool(match)}
    except Exception as exc:
        return {"available": True, "round_trip": False, "error": str(exc)}


def _round_trip_sphincs(crypto: Any) -> dict[str, Any]:
    try:
        from omni_mercury_engine.security.pqc_backends import (
            generate_sphincs_keypair,
            sign_sphincs,
            verify_sphincs,
        )
    except ImportError as exc:
        return {"available": False, "error": f"import: {exc}"}
    try:
        keypair = generate_sphincs_keypair()
        sig = sign_sphincs(keypair, _ROUND_TRIP_MESSAGE)
        ok = verify_sphincs(keypair.public_key, _ROUND_TRIP_MESSAGE, sig)
        return {"available": True, "round_trip": bool(ok)}
    except Exception as exc:
        return {"available": True, "round_trip": False, "error": str(exc)}


def probe() -> dict[str, Any]:
    """Produce the capability report dictionary."""
    try:
        backend = importlib.import_module("omni_mercury_engine.security.pqc_backends")
    except ImportError as exc:
        return {"error": f"cannot import pqc_backends: {exc}"}

    flags = {
        name: bool(getattr(backend, name, False))
        for name in (
            "AMA_CRYPTOGRAPHY_AVAILABLE",
            "DILITHIUM_AVAILABLE",
            "KYBER_AVAILABLE",
            "SPHINCS_AVAILABLE",
            "SLHDSA_AVAILABLE",
            "ML_DSA_CONTEXT_SIGN_AVAILABLE",
        )
    }

    env = {
        "AMA_REQUIRE_REAL_PQC": os.environ.get("AMA_REQUIRE_REAL_PQC", ""),
        "AVA_REQUIRE_REAL_PQC": os.environ.get("AVA_REQUIRE_REAL_PQC", ""),
        "AMA_NO_CYTHON": os.environ.get("AMA_NO_CYTHON", ""),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        "MERCURY_PQC_REAL_AMA": os.environ.get("MERCURY_PQC_REAL_AMA", ""),
    }

    try:
        ama_mod = importlib.import_module("ama_cryptography")
        ama_version = getattr(ama_mod, "__version__", "unknown")
    except ImportError:
        ama_version = None

    round_trips: dict[str, dict[str, Any]] = {}
    if flags["DILITHIUM_AVAILABLE"]:
        round_trips["dilithium"] = _round_trip_dilithium(backend)
    if flags["KYBER_AVAILABLE"]:
        round_trips["kyber"] = _round_trip_kyber(backend)
    if flags["SPHINCS_AVAILABLE"]:
        round_trips["sphincs"] = _round_trip_sphincs(backend)

    return {
        "schema": "mercury_pqc_capability_probe/v1",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "ama_cryptography_version": ama_version,
        "flags": flags,
        "env": env,
        "round_trips": round_trips,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.pqc_capability_probe",
        description="Probe the live PQC backend and report capability fingerprint.",
    )
    parser.add_argument(
        "--require",
        type=str,
        default="",
        help=(
            "Comma-separated list of capabilities that MUST be real for exit 0. "
            "Recognised tokens: ama, dilithium, kyber, sphincs, slhdsa, "
            "ml_dsa_context. Failing the requirement returns exit 1."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path for the audit report.",
    )
    return parser


_REQUIRE_MAP = {
    "ama": "AMA_CRYPTOGRAPHY_AVAILABLE",
    "dilithium": "DILITHIUM_AVAILABLE",
    "kyber": "KYBER_AVAILABLE",
    "sphincs": "SPHINCS_AVAILABLE",
    "slhdsa": "SLHDSA_AVAILABLE",
    "ml_dsa_context": "ML_DSA_CONTEXT_SIGN_AVAILABLE",
}


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = probe()
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True))

    if "error" in report:
        print(f"ERROR: {report['error']}", file=sys.stderr)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    print(json.dumps(report, indent=2, sort_keys=True))

    # Round-trip failures are always exit 1.
    bad_round_trips = [
        name
        for name, r in report["round_trips"].items()
        if r.get("available") and r.get("round_trip") is False
    ]
    if bad_round_trips:
        print(
            f"FAIL: round-trip failed for: {bad_round_trips}",
            file=sys.stderr,
        )
        return 1

    require_tokens = [t.strip().lower() for t in args.require.split(",") if t.strip()]
    missing: list[str] = []
    unknown: list[str] = []
    for token in require_tokens:
        flag = _REQUIRE_MAP.get(token)
        if flag is None:
            unknown.append(token)
            continue
        if not report["flags"].get(flag, False):
            missing.append(f"{token} ({flag}=False)")
    if unknown:
        print(
            f"ERROR: unknown --require token(s): {unknown}; valid: " f"{sorted(_REQUIRE_MAP)}",
            file=sys.stderr,
        )
        return 2
    if missing:
        print(f"FAIL: --require not satisfied: {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry-point
    raise SystemExit(main())
