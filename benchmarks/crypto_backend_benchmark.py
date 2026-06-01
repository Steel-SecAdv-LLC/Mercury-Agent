"""
Mercury Agent - Copyright (C) 2025 Steel Security Advisors LLC
Licensed under GNU GPL v3

Measured Rust-vs-Python crypto benchmark — replaces the unbenchmarked
"BLAKE3 6.5x faster" README claim with a number you can reproduce.

The Rust PyO3 module (`rust_crypto/`) is opt-in and not built by default, so
`omni_mercury_engine.crypto` falls back to the `cryptography` package / `hashlib`.
This script times the *active* backend against a pure-Python reference for the
same primitives and reports the observed speedup. If the Rust backend is not
built, it says so and emits **no** speedup figure rather than fabricating one.

Usage::

    # Python-only environment (default): records baseline, notes rust absent
    python -m benchmarks.crypto_backend_benchmark

    # After `cd rust_crypto && maturin develop`: measures the real speedup
    python -m benchmarks.crypto_backend_benchmark --mb 1 --iters 200 \\
        --out artifacts/crypto_backend_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _time_call(fn: Callable[[], Any], iters: int, warmup: int = 5) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return {
        "median_ms": statistics.median(samples) * 1e3,
        "mean_ms": statistics.fmean(samples) * 1e3,
        "min_ms": min(samples) * 1e3,
    }


def _python_blake3_ref() -> tuple[Callable[[bytes], bytes], str]:
    """Pure-Python BLAKE3 reference, or the documented hashlib.sha256 fallback."""
    try:
        import blake3  # type: ignore

        return (lambda d: blake3.blake3(d).digest()), "blake3-wheel"
    except Exception:
        import hashlib

        return (lambda d: hashlib.sha256(d).digest()), "hashlib-sha256"


def run(mb: float, iters: int) -> dict[str, Any]:
    from omni_mercury_engine import crypto

    backend = crypto.get_crypto_backend()
    rust = crypto.is_rust_available()
    payload = b"\xa5" * int(mb * 1024 * 1024)

    active = _time_call(lambda: crypto.hash_data(payload, "blake3"), iters)
    py_ref_fn, py_ref_name = _python_blake3_ref()
    py_ref = _time_call(lambda: py_ref_fn(payload), iters)

    result: dict[str, Any] = {
        "provenance": {
            "active_backend": backend,
            "rust_available": rust,
            "python_reference": py_ref_name,
            "payload_mb": mb,
            "iterations": iters,
            "commit": _git_commit(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
        },
        "blake3_active_backend_ms": active,
        "blake3_python_reference_ms": py_ref,
    }

    if rust and backend == "rust":
        speedup = py_ref["median_ms"] / active["median_ms"] if active["median_ms"] else 0.0
        result["measured_speedup_rust_vs_python"] = round(speedup, 2)
        result["claim"] = (
            f"BLAKE3 Rust backend measured {speedup:.2f}x vs Python "
            f"{py_ref_name} on {platform.platform()} (commit {result['provenance']['commit']})."
        )
    else:
        result["measured_speedup_rust_vs_python"] = None
        result["claim"] = (
            "Rust backend NOT built; active backend is "
            f"'{backend}'. No Rust-vs-Python speedup measured. "
            "Build it with `cd rust_crypto && maturin develop` to benchmark the acceleration."
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mb", type=float, default=1.0, help="Payload size in MiB.")
    parser.add_argument("--iters", type=int, default=200, help="Timed iterations.")
    parser.add_argument("--out", default="artifacts/crypto_backend_benchmark.json")
    args = parser.parse_args(argv)

    result = run(args.mb, args.iters)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))

    p = result["provenance"]
    print("Crypto backend benchmark")
    print("-" * 72)
    print(f"  active backend:     {p['active_backend']} (rust_available={p['rust_available']})")
    print(f"  python reference:   {p['python_reference']}")
    print(f"  payload / iters:    {p['payload_mb']} MiB x {p['iterations']}")
    print(f"  BLAKE3 active:      {result['blake3_active_backend_ms']['median_ms']:.3f} ms (median)")
    print(f"  BLAKE3 python ref:  {result['blake3_python_reference_ms']['median_ms']:.3f} ms (median)")
    print(f"  speedup (rust/py):  {result['measured_speedup_rust_vs_python']}")
    print(f"  -> {result['claim']}")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
