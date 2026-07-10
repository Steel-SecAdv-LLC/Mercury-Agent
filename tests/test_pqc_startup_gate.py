# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Smoke tests for the import-time PQC gate in ``src/omni_mercury_engine/_pqc_gate.py::_enforce_pqc_production_gate``.

Scope
-----
This file now pins the universal fail-closed contract: the gate always
requires a real AMA Cryptography v3.3.0 native build.  The env vars
``AMA_REQUIRE_REAL_PQC`` / ``AVA_REQUIRE_REAL_PQC`` are compatibility
diagnostics only; unset or false values must not disable the gate.
"""

from __future__ import annotations

import os

import ama_cryptography
import ama_cryptography.pqc_backends as ama_pqc_backends
import pytest

from omni_mercury_engine._pqc_gate import (
    _AMA_REQUIRED_VERSION,
    AMA_CRYPTO_VERSION_ENV,
    _enforce_ama_version,
    _enforce_pqc_production_gate,
    _release_matches,
)


def _scrub_real_pqc_env() -> dict[str, str]:
    """Save and clear the AMA_REQUIRE_REAL_PQC env vars.

    Returns the saved values so callers can restore them.  We avoid
    ``monkeypatch`` here because the cleanup ordering between
    ``monkeypatch`` and module-level imports of ``omni_mercury_engine``
    can produce false positives when the suite is run with the env
    var set globally.
    """
    saved: dict[str, str] = {}
    for name in ("AMA_REQUIRE_REAL_PQC", "AVA_REQUIRE_REAL_PQC"):
        if name in os.environ:
            saved[name] = os.environ.pop(name)
    return saved


def _restore_env(saved: dict[str, str]) -> None:
    for name, value in saved.items():
        os.environ[name] = value


class TestGateRequiresAmaRegardlessOfEnv:
    """Unset/false compatibility env vars must not create an AMA-less mode."""

    def test_returns_silently_when_env_var_is_missing(self) -> None:
        saved = _scrub_real_pqc_env()
        try:
            assert ama_pqc_backends.DILITHIUM_AVAILABLE
            assert ama_pqc_backends.KYBER_AVAILABLE
            assert ama_pqc_backends.SPHINCS_AVAILABLE
            _enforce_pqc_production_gate()  # must not raise
        finally:
            _restore_env(saved)

    def test_explicit_false_does_not_disable_gate(self) -> None:
        saved = _scrub_real_pqc_env()
        os.environ["AMA_REQUIRE_REAL_PQC"] = "false"
        try:
            _enforce_pqc_production_gate()
        finally:
            os.environ.pop("AMA_REQUIRE_REAL_PQC", None)
            _restore_env(saved)

    def test_legacy_false_does_not_disable_gate(self) -> None:
        saved = _scrub_real_pqc_env()
        os.environ["AVA_REQUIRE_REAL_PQC"] = "false"
        try:
            _enforce_pqc_production_gate()
        finally:
            os.environ.pop("AVA_REQUIRE_REAL_PQC", None)
            _restore_env(saved)


class TestGateFailsClosedWhenBackendIncomplete:
    """A missing algorithm flag must fail the gate closed (refuse to start)."""

    @pytest.mark.parametrize(
        ("flag", "friendly"),
        [
            ("DILITHIUM_AVAILABLE", "ML-DSA-65"),
            ("KYBER_AVAILABLE", "Kyber-1024"),
            ("SPHINCS_AVAILABLE", "SPHINCS+"),
        ],
    )
    def test_missing_flag_raises(
        self, flag: str, friendly: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ama_pqc_backends, flag, False, raising=False)
        with pytest.raises(RuntimeError, match="incomplete") as exc_info:
            _enforce_pqc_production_gate()
        # Assert the specific missing algorithm is named, not just the generic failure.
        assert friendly in str(exc_info.value)


class TestFipsPostEnforcement:
    """The gate consults AMA's FIPS 140-3 POST verdict, not just the flags."""

    def test_operational_backend_passes(self) -> None:
        from omni_mercury_engine._pqc_gate import _enforce_fips_post

        # The build under test is OPERATIONAL; the check is silent.
        assert ama_cryptography.module_status() == "OPERATIONAL"
        _enforce_fips_post()

    def test_post_failure_fails_gate_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A non-OPERATIONAL module (POST locked out) must refuse to start.

        Regression: the gate checked only the *_AVAILABLE flags + version, so a
        build whose known-answer self-tests failed — but still exposed the flags
        — passed Mercury's import gate and ran on a non-validated backend.
        """
        from ama_cryptography.exceptions import CryptoModuleError

        def _locked_out() -> None:
            raise CryptoModuleError("Module locked out by FIPS POST failure (test)")

        monkeypatch.setattr(ama_cryptography, "check_operational", _locked_out)
        monkeypatch.setattr(ama_cryptography, "module_status", lambda: "ERROR")

        with pytest.raises(RuntimeError, match=r"FIPS 140-3.*not OPERATIONAL"):
            _enforce_pqc_production_gate()


class TestAmaVersionEnforcement:
    """The gate pins AMA Cryptography to exactly ``_AMA_REQUIRED_VERSION`` (3.3.0)."""

    def test_pinned_version_is_3_3_0(self) -> None:
        assert _AMA_REQUIRED_VERSION == "3.3.0"

    def test_real_installed_version_passes(self) -> None:
        # The build under test installs the pinned version; the check is silent.
        _enforce_ama_version()

    def test_declared_env_mismatch_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(AMA_CRYPTO_VERSION_ENV, "3.1.0")
        with pytest.raises(RuntimeError, match="version mismatch"):
            _enforce_ama_version()

    def test_declared_env_match_passes_with_v_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(AMA_CRYPTO_VERSION_ENV, "v3.3.0")
        _enforce_ama_version()

    def test_installed_version_mismatch_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ama_cryptography, "__version__", "9.9.9", raising=False)
        with pytest.raises(RuntimeError, match="version mismatch"):
            _enforce_ama_version()

    @pytest.mark.parametrize(
        "version",
        # Trailing-zero equivalents of the pin are the same release and accepted.
        ["3.3.0", "v3.3.0", "3.3.0.post1", "3.3.0rc1", "3.3.0+cpu", "3.3", "3.3.0.0", "  3.3.0 "],
    )
    def test_release_matches_accepts_pinned_release_variants(self, version: str) -> None:
        assert _release_matches(version) is True

    @pytest.mark.parametrize(
        "version",
        # A *longer* release that merely shares the pinned prefix (3.3.0.1) is a
        # DIFFERENT release and must be refused, not truncated to the pin -- else
        # the Tier-0 version gate would silently accept an unpinned build.
        ["3.1.0", "3.2.0", "9.9.9", "2.0.0", "3", "3.3.0.1", "3.3.0.0.1", "3.3.1", "", "garbage"],
    )
    def test_release_matches_rejects_other_releases(self, version: str) -> None:
        assert _release_matches(version) is False

    def test_post_release_build_is_accepted_end_to_end(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A post/local build of the pinned release must NOT be refused.
        monkeypatch.setattr(ama_cryptography, "__version__", "3.3.0.post1", raising=False)
        _enforce_ama_version()  # must not raise
