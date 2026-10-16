# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Smoke tests for the import-time PQC gate in ``src/omni_mercury_engine/_pqc_gate.py::_enforce_pqc_production_gate``.

Scope
-----
This file now pins the universal fail-closed contract: the gate always
requires a real AMA Cryptography v5.0.0 native build.  The env vars
``AMA_REQUIRE_REAL_PQC`` / ``AVA_REQUIRE_REAL_PQC`` are compatibility
diagnostics only; unset or false values must not disable the gate.
``TestScopedGateContract`` pins *where* the gate is enforced: every agent
surface refuses without AMA while the analytic quickstart leaves import
AMA-free.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING

import ama_cryptography
import ama_cryptography.pqc_backends as ama_pqc_backends
import pytest

if TYPE_CHECKING:
    from pathlib import Path

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
    """The gate pins AMA Cryptography to exactly ``_AMA_REQUIRED_VERSION`` (5.0.0)."""

    def test_pinned_version_is_5_0_0(self) -> None:
        assert _AMA_REQUIRED_VERSION == "5.0.0"

    def test_real_installed_version_passes(self) -> None:
        # The build under test installs the pinned version; the check is silent.
        _enforce_ama_version()

    def test_declared_env_mismatch_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(AMA_CRYPTO_VERSION_ENV, "3.1.0")
        with pytest.raises(RuntimeError, match="version mismatch"):
            _enforce_ama_version()

    def test_declared_env_match_passes_with_v_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(AMA_CRYPTO_VERSION_ENV, "v5.0.0")
        _enforce_ama_version()

    def test_installed_version_mismatch_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ama_cryptography, "__version__", "9.9.9", raising=False)
        with pytest.raises(RuntimeError, match="version mismatch"):
            _enforce_ama_version()

    @pytest.mark.parametrize(
        "version",
        # Trailing-zero equivalents of the pin are the same release and accepted.
        # ``"5"`` is included: zero-padding makes it the same release as the pin.
        [
            "5.0.0",
            "v5.0.0",
            "5.0.0.post1",
            "5.0.0rc1",
            "5.0.0+cpu",
            "5.0",
            "5.0.0.0",
            "  5.0.0 ",
            "5",
        ],
    )
    def test_release_matches_accepts_pinned_release_variants(self, version: str) -> None:
        assert _release_matches(version) is True

    @pytest.mark.parametrize(
        "version",
        # A *longer* release that merely shares the pinned prefix (5.0.0.1) is a
        # DIFFERENT release and must be refused, not truncated to the pin -- else
        # the Tier-0 version gate would silently accept an unpinned build.
        # ``"5"`` is NOT here: it zero-pads to the pinned release and is accepted
        # above, unlike the old pin where ``"3"`` padded to (3, 0, 0) != (3, 3, 0).
        ["3.1.0", "4.0.0", "9.9.9", "2.0.0", "5.0.0.1", "5.0.0.0.1", "5.0.1", "", "garbage"],
    )
    def test_release_matches_rejects_other_releases(self, version: str) -> None:
        assert _release_matches(version) is False

    def test_post_release_build_is_accepted_end_to_end(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A post/local build of the pinned release must NOT be refused.
        monkeypatch.setattr(ama_cryptography, "__version__", "5.0.0.post1", raising=False)
        _enforce_ama_version()  # must not raise


class TestScopedGateContract:
    """Pins the maintainer-directed scoped contract.

    The standalone analytic leaves (what ``examples/mercury_quickstart.py``
    imports) must work without AMA so consumers reach a measured result fast;
    every agent surface (``engine``, ``crypto`` — the CLI/API serve the
    engine) must refuse without it. Each case runs in a subprocess with an
    ``ama_cryptography`` stub whose algorithms are unavailable, so the result
    is independent of the host's real AMA install.
    """

    @staticmethod
    def _run_with_unavailable_ama(tmp_path: Path, code: str) -> subprocess.CompletedProcess[str]:
        stub = tmp_path / "ama_stub" / "ama_cryptography"
        stub.mkdir(parents=True, exist_ok=True)
        (stub / "__init__.py").write_text(
            '__version__ = "5.0.0"\n'
            "DILITHIUM_AVAILABLE = False\n"
            "KYBER_AVAILABLE = False\n"
            "SPHINCS_AVAILABLE = False\n"
        )
        env = dict(os.environ)
        # The stub shadows the real AMA; any inherited entry (an uninstalled
        # checkout's src/) stays after it so the package itself still imports.
        inherited = os.environ.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(stub.parent), inherited) if p)
        # check=False: each test asserts on returncode explicitly — a refusal
        # (nonzero exit) is the expected outcome for the gated surfaces.
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=240,
            check=False,
        )

    def test_quickstart_analytic_chain_imports_without_ama(self, tmp_path: Path) -> None:
        result = self._run_with_unavailable_ama(
            tmp_path,
            "from omni_mercury_engine.datasets.maritime import DTUAISLoader\n"
            "from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector\n"
            "from omni_mercury_engine.ml.mercury_ml import roc_auc_score\n"
            "print('LEAVES_OK')\n",
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "LEAVES_OK" in result.stdout

    def test_agent_engine_refuses_without_ama(self, tmp_path: Path) -> None:
        result = self._run_with_unavailable_ama(
            tmp_path, "import omni_mercury_engine.engine\nprint('MUST_NOT_PRINT')\n"
        )
        assert result.returncode != 0
        assert "MUST_NOT_PRINT" not in result.stdout
        assert "AMA" in result.stderr  # the refusal names the missing backend

    def test_crypto_surface_refuses_without_ama(self, tmp_path: Path) -> None:
        result = self._run_with_unavailable_ama(
            tmp_path, "import omni_mercury_engine.crypto\nprint('MUST_NOT_PRINT')\n"
        )
        assert result.returncode != 0
        assert "MUST_NOT_PRINT" not in result.stdout
        assert "AMA" in result.stderr

    def test_engine_root_attribute_refuses_without_ama(self, tmp_path: Path) -> None:
        # Reaching the agent through the lazy root export is gated identically.
        result = self._run_with_unavailable_ama(
            tmp_path,
            "import omni_mercury_engine\n"
            "omni_mercury_engine.OmniMercuryEngine\n"
            "print('MUST_NOT_PRINT')\n",
        )
        assert result.returncode != 0
        assert "MUST_NOT_PRINT" not in result.stdout

    def test_security_primitives_import_without_ama(self, tmp_path: Path) -> None:
        # The AMA-free security primitives the analytic path leans on (safe
        # HTTP, endpoint validation, rate limiting, audit logging) resolve
        # through the lazy package without a PQC build.
        result = self._run_with_unavailable_ama(
            tmp_path,
            "from omni_mercury_engine.security import RateLimiter, SafeHTTPClient\n"
            "from omni_mercury_engine.security import SecureAuditLogger\n"
            "from omni_mercury_engine.security.input_validation import TrustedEndpoints\n"
            "print('PRIMITIVES_OK')\n",
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "PRIMITIVES_OK" in result.stdout

    def test_security_crypto_access_refuses_without_ama(self, tmp_path: Path) -> None:
        # First *access* to an AMA-backed name fails closed with the gate's
        # named refusal — lazy resolution defers the gate, never removes it.
        result = self._run_with_unavailable_ama(
            tmp_path,
            "from omni_mercury_engine.security import MercuryCrypto\n" "print('MUST_NOT_PRINT')\n",
        )
        assert result.returncode != 0
        assert "MUST_NOT_PRINT" not in result.stdout
        assert "AMA" in result.stderr
