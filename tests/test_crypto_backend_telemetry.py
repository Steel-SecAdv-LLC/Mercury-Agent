"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

Tests for the crypto backend telemetry contract
(`omni_mercury_engine.crypto.get_crypto_backend` / `is_rust_available`) and
the honest Rust-vs-Python benchmark.  These pin the behaviour that replaces the
previously unbenchmarked "6.5x faster" README claim: the active backend is
always observable, and the Python fallback is explicit.
"""

from __future__ import annotations

import hashlib

from omni_mercury_engine import crypto

VALID_BACKENDS = {"rust", "python-cryptography", "hashlib-only"}


def test_backend_is_reported_and_valid() -> None:
    backend = crypto.get_crypto_backend()
    assert backend in VALID_BACKENDS


def test_rust_flag_is_consistent_with_backend() -> None:
    rust = crypto.is_rust_available()
    assert isinstance(rust, bool)
    if rust:
        assert crypto.get_crypto_backend() == "rust"
    else:
        # Fallback must be explicit, never silently "rust" when unavailable.
        assert crypto.get_crypto_backend() in {"python-cryptography", "hashlib-only"}


def test_hash_data_is_deterministic_and_correct_length() -> None:
    digest1 = crypto.hash_data(b"mercury", "blake3")
    digest2 = crypto.hash_data(b"mercury", "blake3")
    assert digest1 == digest2
    assert isinstance(digest1, bytes)
    assert len(digest1) == 32  # 256-bit digest regardless of backend
    # sha256 path is exact and backend-independent
    assert crypto.hash_data(b"mercury", "sha256") == hashlib.sha256(b"mercury").digest()


def test_benchmark_runs_and_is_honest_about_rust() -> None:
    from benchmarks import crypto_backend_benchmark as cbb

    result = cbb.run(mb=0.05, iters=8)  # tiny payload/iters to keep the test fast
    prov = result["provenance"]
    assert prov["active_backend"] in VALID_BACKENDS
    assert "blake3_active_backend_ms" in result
    assert "blake3_python_reference_ms" in result
    if (
        prov["rust_available"]
        and prov["active_backend"] == "rust"
        and prov["python_reference"] == "blake3-wheel"
    ):
        assert result["measured_speedup_rust_vs_python"] is not None
    else:
        # No Rust or no like-for-like Python BLAKE3 reference => no fabricated number.
        assert result["measured_speedup_rust_vs_python"] is None
        assert (
            "No Rust-vs-Python speedup measured" in result["claim"]
            or "like-for-like" in result["claim"]
        )
