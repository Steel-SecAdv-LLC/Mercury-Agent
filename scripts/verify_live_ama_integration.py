# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end verifier for Mercury ⇄ live AMA Cryptography integration.

This is the single reproducible command that proves the engine's import-time
PQC gate resolves against the **real** AMA native C backend (not a mock, not a
stand-in), that every algorithm surface the engine consumes round-trips
correctly, and — critically — that the fail-closed guarantee is preserved: the
gate still raises on a partial/absent backend.

It exercises the exact contract ``omni_mercury_engine.security.pqc_backends``
imports from ``ama_cryptography.pqc_backends`` (see that module's import block):
ML-DSA-65 sign/verify, Kyber-1024 encapsulate/decapsulate, SPHINCS+ sign/verify,
and FIPS 205 SLH-DSA for both parameter sets (SHAKE-128s / SHA2-256f).

Prerequisite — the AMA native library must be built and importable. Mirror
``.github/actions/build-ama-cryptography`` (or docs/INSTALLATION.md
'Post-Quantum Cryptography backend'):

    python -m pip install --upgrade "setuptools>=78.1.1" "wheel>=0.47.0" "cmake>=4.3.2"
    git clone --branch v3.2.0 --depth 1 \
        https://github.com/Steel-SecAdv-LLC/AMA-Cryptography.git /tmp/ama-cryptography
    cmake -S /tmp/ama-cryptography -B /tmp/ama-cryptography/build \
        -DCMAKE_BUILD_TYPE=Release -DAMA_USE_NATIVE_PQC=ON
    cmake --build /tmp/ama-cryptography/build -j"$(nproc)"
    AMA_NO_CYTHON=1 pip install --no-build-isolation /tmp/ama-cryptography

Then, from the Mercury-Agent checkout:

    PYTHONPATH=src python scripts/verify_live_ama_integration.py [--json]

Exit code is 0 only if every check passes; any failure (including the gate
declining to fail closed on a simulated partial backend) exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Runnable from a fresh checkout without an editable install: put the repo's
# src/ on sys.path before importing the engine (mirrors benchmarks/*.py).
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _check(results: list[dict], name: str, passed: bool, detail: str = "") -> None:
    results.append({"check": name, "passed": bool(passed), "detail": detail})


def _verify_round_trips(results: list[dict]) -> None:
    """Exercise every algorithm surface the engine imports from AMA."""
    from omni_mercury_engine.security import pqc_backends as mpb

    # Availability flags must reflect the live native backend.
    _check(
        results,
        "availability_flags",
        mpb.DILITHIUM_AVAILABLE and mpb.KYBER_AVAILABLE and mpb.SPHINCS_AVAILABLE,
        f"dilithium={mpb.DILITHIUM_AVAILABLE} kyber={mpb.KYBER_AVAILABLE} "
        f"sphincs={mpb.SPHINCS_AVAILABLE}",
    )

    # ML-DSA-65 (FIPS 204): keygen -> sign -> verify + tamper rejection.
    kp = mpb.generate_dilithium_keypair()
    sig = mpb.dilithium_sign(b"mercury-live", kp.secret_key)
    _check(
        results,
        "ml_dsa_65_sign_verify",
        mpb.dilithium_verify(b"mercury-live", sig, kp.public_key)
        and not mpb.dilithium_verify(b"tampered", sig, kp.public_key),
    )

    # Kyber-1024 (FIPS 203 ML-KEM): encapsulate -> decapsulate, secret equality.
    kk = mpb.generate_kyber_keypair()
    enc = mpb.kyber_encapsulate(kk.public_key)
    ss = mpb.kyber_decapsulate(enc.ciphertext, kk.secret_key)
    _check(
        results,
        "kyber_1024_kem",
        bytes(enc.shared_secret) == bytes(ss),
    )

    # SPHINCS+: keygen -> sign -> verify + tamper rejection.
    sp = mpb.generate_sphincs_keypair()
    ssig = mpb.sphincs_sign(b"mercury-live", sp.secret_key)
    _check(
        results,
        "sphincs_sign_verify",
        mpb.sphincs_verify(b"mercury-live", ssig, sp.public_key)
        and not mpb.sphincs_verify(b"tampered", ssig, sp.public_key),
    )

    # FIPS 205 SLH-DSA — both parameter sets the engine declares.
    for param_set in ("SHAKE-128s", "SHA2-256f"):
        lp = mpb.generate_slhdsa_keypair(param_set)
        lsig = mpb.slhdsa_sign(b"mercury-live", lp.secret_key, b"", param_set)
        det = mpb.slhdsa_sign_deterministic(b"mercury-live", lp.secret_key, b"", param_set)
        ok = (
            mpb.slhdsa_verify(b"mercury-live", lsig, lp.public_key, b"", param_set)
            and mpb.slhdsa_verify(b"mercury-live", det, lp.public_key, b"", param_set)
            and not mpb.slhdsa_verify(b"tampered", lsig, lp.public_key, b"", param_set)
        )
        _check(results, f"slh_dsa_{param_set}_sign_verify", ok)

    # The production-readiness validator (independent of the import gate).
    info = mpb.validate_pqc_environment()
    _check(results, "validate_pqc_environment", info.get("production_ready") is True)


def _verify_gate_passes(results: list[dict]) -> None:
    """The import-time gate must resolve silently against the live backend."""
    from omni_mercury_engine._pqc_gate import _enforce_pqc_production_gate

    try:
        _enforce_pqc_production_gate()
        _check(results, "import_gate_resolves_live", True, "gate passed against live AMA")
    except RuntimeError as exc:  # pragma: no cover - failure path
        _check(results, "import_gate_resolves_live", False, str(exc).splitlines()[0])


def _verify_fail_closed(results: list[dict]) -> None:
    """The gate MUST still fail closed on a partial backend.

    Force one algorithm's availability flag False to simulate a partial native
    build and assert the gate raises rather than silently degrading. The flag
    is restored in ``finally`` so the simulated failure cannot leak into the
    verifier's other checks; the gate reads the flag live on every call, so an
    in-process monkeypatch + restore is sufficient and faithful.
    """
    import ama_cryptography.pqc_backends as ap

    from omni_mercury_engine._pqc_gate import _enforce_pqc_production_gate

    original = ap.KYBER_AVAILABLE
    closed = False
    try:
        ap.KYBER_AVAILABLE = False
        _enforce_pqc_production_gate()
    except RuntimeError:
        closed = True
    finally:
        ap.KYBER_AVAILABLE = original

    _check(
        results,
        "fail_closed_on_partial_backend",
        closed,
        "gate raised on simulated missing Kyber"
        if closed
        else "gate did NOT raise on a partial backend",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    results: list[dict] = []
    _verify_gate_passes(results)
    _verify_round_trips(results)
    _verify_fail_closed(results)

    all_passed = all(r["passed"] for r in results)

    if args.json:
        print(json.dumps({"all_passed": all_passed, "checks": results}, indent=2))
    else:
        print("==== Mercury ⇄ live AMA Cryptography integration ====")
        for r in results:
            status = "PASS" if r["passed"] else "FAIL"
            line = f"  [{status}] {r['check']}"
            if r["detail"]:
                line += f"  ({r['detail']})"
            print(line)
        print("=" * 52)
        print("RESULT:", "ALL CHECKS PASSED" if all_passed else "FAILURES PRESENT")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
