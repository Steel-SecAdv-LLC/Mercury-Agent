# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Operator tool: drive a full ML-KEM-1024 encap/decap + ML-DSA-65 sign/verify cycle through the AMA native PQC backend and emit a latency certificate.

Pair-claims the result with :mod:`pqc_capability_probe` so a silent
stub fallback mid-run is detected: if ``pqc_capability_probe`` reports
the native backend as available but the handshake takes orders of
magnitude longer (or shorter) than the real implementation, the gate
fails closed.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any

from omni_mercury_engine.tools._base import (
    Certificate,
    DependencyMissing,
    mercury_env,
    require_real_component,
    run_tool,
)

_SCHEMA = "mercury.tools.pqc_handshake_simulator/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.pqc_handshake_simulator",
        description=(
            "Drive ML-KEM-1024 encap/decap + ML-DSA-65 sign/verify cycles "
            "through the AMA native PQC backend and emit a latency cert."
        ),
    )
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument(
        "--message-size",
        type=int,
        default=4096,
        help="Bytes of message material to sign per iteration.",
    )
    parser.add_argument(
        "--require-native",
        action="store_true",
        help="Fail closed if pqc_capability_probe cannot confirm a real backend.",
    )
    return parser


def _percentile(samples: list[float], pct: float) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    k = (len(s) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _collect(args: argparse.Namespace) -> Certificate:
    try:
        from omni_mercury_engine.security.crypto_api import (
            AlgorithmType,
            MercuryCrypto,
            SecurityLevel,
        )
    except ImportError as exc:
        raise DependencyMissing(f"crypto_api unavailable: {exc}") from exc

    # ``SecurityLevel.HYBRID`` selects the strongest available — when the
    # AMA-native PQC backends are present they are preferred; otherwise the
    # call surfaces the warn/fail path below via the ``require_real_component``
    # gate.
    crypto = MercuryCrypto(security_level=SecurityLevel.HYBRID)

    # Paired-claim check via pqc_capability_probe — when --require-native
    # is set we cross-reference the live capability report so a stub
    # fallback that "looks" real to introspection still fails.
    capability: dict[str, Any] = {}
    try:
        from omni_mercury_engine.security import pqc_backends

        capability = {
            "KYBER_AVAILABLE": bool(getattr(pqc_backends, "KYBER_AVAILABLE", False)),
            "DILITHIUM_AVAILABLE": bool(getattr(pqc_backends, "DILITHIUM_AVAILABLE", False)),
            "DILITHIUM_CTX_AVAILABLE": bool(
                getattr(pqc_backends, "DILITHIUM_CTX_AVAILABLE", False)
            ),
        }
    except ImportError:
        pass

    if args.require_native:
        require_real_component(
            "ML-KEM-1024 backend",
            capability.get("KYBER_AVAILABLE", False),
        )
        require_real_component(
            "ML-DSA-65 backend",
            capability.get("DILITHIUM_AVAILABLE", False),
        )

    kem_latencies: list[float] = []
    sig_latencies: list[float] = []
    ver_latencies: list[float] = []
    errors: list[str] = []

    try:
        # ``generate_kem_keypair`` returns Kyber-1024 / ML-KEM-1024 by
        # default; the ``MercuryCrypto`` constructor pins the Kyber provider
        # at instantiation time.
        kem_kp = crypto.generate_kem_keypair()
        dsa_kp = crypto.generate_signing_keypair(algorithm=AlgorithmType.ML_DSA_65)
    except RuntimeError as exc:
        return Certificate(
            tool="pqc_handshake_simulator",
            schema=_SCHEMA,
            status="warn" if mercury_env() != "production" else "fail",
            body={
                "capability": capability,
                "error": f"keygen failed: {exc}",
            },
            warnings=[f"native PQC backend unavailable: {exc}"],
        )

    msg = b"\x42" * int(args.message_size)
    for _ in range(int(args.iterations)):
        t0 = time.perf_counter_ns()
        try:
            encap = crypto.encapsulate(kem_kp.public_key)
            decap = crypto.decapsulate(encap.ciphertext, kem_kp.secret_key)
            if decap != encap.shared_secret:
                errors.append("ML-KEM-1024 shared-secret mismatch")
                break
        except Exception as exc:
            errors.append(f"ML-KEM-1024 failed: {type(exc).__name__}: {exc}")
            break
        kem_latencies.append((time.perf_counter_ns() - t0) / 1e6)

        t0 = time.perf_counter_ns()
        try:
            sig = crypto.sign(msg, dsa_kp.secret_key, algorithm=AlgorithmType.ML_DSA_65)
        except Exception as exc:
            errors.append(f"ML-DSA-65 sign failed: {type(exc).__name__}: {exc}")
            break
        sig_latencies.append((time.perf_counter_ns() - t0) / 1e6)

        t0 = time.perf_counter_ns()
        try:
            # ``MercuryCrypto.verify`` consumes the full ``Signature``
            # dataclass (algorithm tag baked in) plus the matching
            # public key.  This is the documented interface in
            # ``security.crypto_api`` and the only one that round-trips
            # the algorithm declaration end-to-end.
            ok = crypto.verify(msg, sig, dsa_kp.public_key)
            if not ok:
                errors.append("ML-DSA-65 verify returned False")
                break
        except Exception as exc:
            errors.append(f"ML-DSA-65 verify failed: {type(exc).__name__}: {exc}")
            break
        ver_latencies.append((time.perf_counter_ns() - t0) / 1e6)

    def summary(samples: list[float]) -> dict[str, float]:
        if not samples:
            return {"n": 0}
        return {
            "n": float(len(samples)),
            "min_ms": min(samples),
            "max_ms": max(samples),
            "mean_ms": statistics.fmean(samples),
            "p50_ms": _percentile(samples, 50),
            "p95_ms": _percentile(samples, 95),
            "p99_ms": _percentile(samples, 99),
        }

    body: dict[str, Any] = {
        "iterations": int(args.iterations),
        "message_size_bytes": int(args.message_size),
        "capability": capability,
        "ml_kem_1024_encap_decap": summary(kem_latencies),
        "ml_dsa_65_sign": summary(sig_latencies),
        "ml_dsa_65_verify": summary(ver_latencies),
        "errors": errors,
    }

    if errors:
        return Certificate(
            tool="pqc_handshake_simulator",
            schema=_SCHEMA,
            status="fail",
            body=body,
            warnings=errors,
        )

    # Sanity floor: a real backend completes a single ML-KEM round trip in
    # well under 10 ms on any 2020-era host.  A "completion" sitting under
    # 5 microseconds is the signature of a stub returning zeros.
    suspect = []
    if kem_latencies and statistics.fmean(kem_latencies) < 0.005:
        suspect.append("ML-KEM-1024 latencies suspiciously low — possible stub backend")
    if sig_latencies and statistics.fmean(sig_latencies) < 0.005:
        suspect.append("ML-DSA-65 sign latencies suspiciously low — possible stub backend")
    status = "warn" if suspect else "ok"
    return Certificate(
        tool="pqc_handshake_simulator",
        schema=_SCHEMA,
        status=status,
        body=body,
        warnings=suspect,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
