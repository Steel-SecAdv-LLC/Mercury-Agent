"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

End-to-end tests of the import-time PQC production gate against the
**real** AMA Cryptography native C library — no mock, no stub, no
``importorskip``, no ``skipif``.  These tests REQUIRE the real
``ama_cryptography`` package with its native ``libama_cryptography.so``
loadable via ``LD_LIBRARY_PATH``; they fail loudly if the dependency
isn't available, which is the correct behaviour for a contract that
should not silently degrade.

The CI workflows that collect this file
(``.github/workflows/pqc-production-check.yml::verify-real-pqc`` and
``.github/workflows/ci.yml::ml-tests``) both build AMA Cryptography
v3.1.0 from source before invoking pytest, so the tests below exercise
the real native cryptographic primitives every run.

What this file proves on every CI run
--------------------------------------
1. The Mercury PQC gate (``omni_mercury_engine._pqc_gate``) and the
   helper (``omni_mercury_engine.security.pqc_guards``) both accept a
   real, fully-built AMA install — no false-positive partial-install
   rejections.
2. Each of the three AMA primitives is actually exercised end-to-end
   against the native lib:

   - ML-DSA-65 (Dilithium): key-pair generation → sign → verify
     round-trip + signature-rejection on a tampered message.
   - Kyber-1024: key-pair generation → encapsulate → decapsulate
     round-trip + shared-secret equality.
   - SPHINCS+ (SLH-DSA): key-pair generation → sign → verify
     round-trip + signature-rejection on a tampered message.

3. Mercury's mandatory ``security/pqc_backends.py`` import is backed by the
   real ``ama_cryptography.pqc_backends`` submodule.
"""

from __future__ import annotations

import os

# These imports are deliberately at module scope and unguarded:
# if AMA Cryptography is not installed, this file fails to collect,
# which is exactly the signal we want — no skip, no silence.
import ama_cryptography.pqc_backends as ama_pqc_backends  # noqa: F401
from ama_cryptography.pqc_backends import (
    DILITHIUM_AVAILABLE,
    KYBER_AVAILABLE,
    SPHINCS_AVAILABLE,
    dilithium_sign,
    dilithium_verify,
    generate_dilithium_keypair,
    generate_kyber_keypair,
    generate_sphincs_keypair,
    kyber_decapsulate,
    kyber_encapsulate,
    sphincs_sign,
    sphincs_verify,
)

from omni_mercury_engine._pqc_gate import _enforce_pqc_production_gate
from omni_mercury_engine.security.pqc_guards import check_pqc_production_readiness


class TestRealAmaFlagsAreTrue:
    """The whole point of building AMA from source in CI is so these
    flags are True at runtime.  If any of them is False, the build
    has regressed (either AMA upstream or our ``-DAMA_USE_NATIVE_PQC=ON``
    invocation) and every other test in this file would silently
    test the wrong thing — pin them up front."""

    def test_dilithium_available_is_true(self) -> None:
        assert DILITHIUM_AVAILABLE is True, (
            "ama_cryptography.pqc_backends.DILITHIUM_AVAILABLE is False; "
            "the AMA native library was not built with -DAMA_USE_NATIVE_PQC=ON, "
            "or LD_LIBRARY_PATH does not point at the build/ directory."
        )

    def test_kyber_available_is_true(self) -> None:
        assert KYBER_AVAILABLE is True, (
            "ama_cryptography.pqc_backends.KYBER_AVAILABLE is False; "
            "rebuild AMA Cryptography with -DAMA_USE_NATIVE_PQC=ON."
        )

    def test_sphincs_available_is_true(self) -> None:
        assert SPHINCS_AVAILABLE is True, (
            "ama_cryptography.pqc_backends.SPHINCS_AVAILABLE is False; "
            "rebuild AMA Cryptography with -DAMA_USE_NATIVE_PQC=ON."
        )


class TestDilithiumRoundTripAgainstNativeLib:
    """ML-DSA-65 (FIPS 204) — exercise keygen + sign + verify end-to-end
    against the native AMA implementation."""

    def test_sign_and_verify_round_trip_succeeds(self) -> None:
        kp = generate_dilithium_keypair()
        message = b"Mercury Agent v1.7.0 PQC gate verification message."
        signature = dilithium_sign(message, kp.secret_key)
        assert dilithium_verify(message, signature, kp.public_key) is True

    def test_signature_rejects_tampered_message(self) -> None:
        """Sign a message, verify against a *tampered* copy."""
        kp = generate_dilithium_keypair()
        message = b"Original payload."
        tampered = b"Original payload!"  # one byte different
        signature = dilithium_sign(message, kp.secret_key)
        assert dilithium_verify(tampered, signature, kp.public_key) is False


class TestKyberRoundTripAgainstNativeLib:
    """Kyber-1024 (FIPS 203 ML-KEM) — exercise keygen + encapsulate +
    decapsulate end-to-end against the native AMA implementation."""

    def test_encapsulate_and_decapsulate_yield_matching_shared_secret(
        self,
    ) -> None:
        kp = generate_kyber_keypair()
        encap = kyber_encapsulate(kp.public_key)
        # The encapsulated shared secret must be reproducible by the
        # holder of the secret key — this is the entire correctness
        # property of a KEM.
        decapsulated_ss = kyber_decapsulate(encap.ciphertext, kp.secret_key)
        assert decapsulated_ss == encap.shared_secret
        # Sanity-check the shape: ML-KEM-1024 yields a 32-byte SS.
        assert isinstance(encap.shared_secret, bytes)
        assert len(encap.shared_secret) == 32


class TestSphincsRoundTripAgainstNativeLib:
    """SPHINCS+ / SLH-DSA-SHAKE-128s (FIPS 205 NIST L1) — exercise keygen
    + sign + verify end-to-end against the native AMA implementation.
    This also confirms ``SPHINCS_AVAILABLE`` is more than just a flag —
    the actual signing API is callable."""

    def test_sign_and_verify_round_trip_succeeds(self) -> None:
        kp = generate_sphincs_keypair()
        message = b"Hash-based-signature smoke under real AMA Cryptography."
        signature = sphincs_sign(message, kp.secret_key)
        assert sphincs_verify(message, signature, kp.public_key) is True

    def test_signature_rejects_tampered_message(self) -> None:
        kp = generate_sphincs_keypair()
        message = b"Original SPHINCS payload."
        tampered = b"Original SPHINCS payload!"
        signature = sphincs_sign(message, kp.secret_key)
        assert sphincs_verify(tampered, signature, kp.public_key) is False


class TestImportTimeGateAcceptsRealAma:
    """The production gate must NOT raise when the env var is set and
    AMA is fully built.  This is the success path that round-6 / round-7
    were chasing through fake-AMA mocking; it's now exercised against
    the real dependency."""

    def test_gate_returns_silently_with_env_set_and_real_ama_loaded(
        self,
    ) -> None:
        # Re-assert the AMA flags up front so a regression in the
        # AMA build is attributed correctly rather than blamed on
        # the gate.
        assert DILITHIUM_AVAILABLE and KYBER_AVAILABLE and SPHINCS_AVAILABLE

        previous = os.environ.get("AMA_REQUIRE_REAL_PQC")
        os.environ["AMA_REQUIRE_REAL_PQC"] = "true"
        try:
            _enforce_pqc_production_gate()  # must not raise
        finally:
            if previous is None:
                os.environ.pop("AMA_REQUIRE_REAL_PQC", None)
            else:
                os.environ["AMA_REQUIRE_REAL_PQC"] = previous


class TestCheckPqcProductionReadinessAcceptsRealAma:
    """``security.pqc_guards.check_pqc_production_readiness`` is the
    finer-boundary helper that mirrors the import-time gate's contract.
    It must return a populated result dict (and NOT raise) on a real,
    fully-built AMA install, even with ``AMA_REQUIRE_REAL_PQC=true``."""

    def test_helper_returns_full_dict_with_env_set_and_real_ama_loaded(
        self,
    ) -> None:
        assert DILITHIUM_AVAILABLE and KYBER_AVAILABLE and SPHINCS_AVAILABLE

        previous = os.environ.get("AMA_REQUIRE_REAL_PQC")
        os.environ["AMA_REQUIRE_REAL_PQC"] = "true"
        try:
            result = check_pqc_production_readiness()
        finally:
            if previous is None:
                os.environ.pop("AMA_REQUIRE_REAL_PQC", None)
            else:
                os.environ["AMA_REQUIRE_REAL_PQC"] = previous

        assert isinstance(result, dict)
        assert result["dilithium"] is True
        assert result["kyber"] is True
        assert result["sphincs"] is True
        # ``backend`` is set by ``get_active_backend()`` in the helper —
        # the value comes from ``security.pqc_backends.PQCBackend`` and
        # MUST resolve to the AMA Cryptography backend on this lane.
        assert "ama" in str(result["backend"]).lower()


class TestEnvVarsDoNotDisableAma:
    """AMA is mandatory regardless of production/dev compatibility flags."""

    def _clear(self) -> dict[str, str]:
        saved = {}
        for name in ("AMA_REQUIRE_REAL_PQC", "AVA_REQUIRE_REAL_PQC", "MERCURY_ENV"):
            if name in os.environ:
                saved[name] = os.environ.pop(name)
        return saved

    def _restore(self, saved: dict[str, str]) -> None:
        for name in ("AMA_REQUIRE_REAL_PQC", "AVA_REQUIRE_REAL_PQC", "MERCURY_ENV"):
            os.environ.pop(name, None)
        os.environ.update(saved)

    def test_production_without_env_var_passes_with_real_ama(self) -> None:
        assert DILITHIUM_AVAILABLE and KYBER_AVAILABLE and SPHINCS_AVAILABLE
        saved = self._clear()
        os.environ["MERCURY_ENV"] = "production"
        try:
            _enforce_pqc_production_gate()
        finally:
            self._restore(saved)

    def test_explicit_false_does_not_disable_gate(self) -> None:
        saved = self._clear()
        os.environ["MERCURY_ENV"] = "production"
        os.environ["AMA_REQUIRE_REAL_PQC"] = "false"
        try:
            _enforce_pqc_production_gate()
        finally:
            self._restore(saved)
