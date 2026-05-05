"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Smoke tests for the import-time PQC production gate in
``src/omni_mercury_engine/_pqc_gate.py::_enforce_pqc_production_gate``.

The gate is a defensive fail-closed: when ``AMA_REQUIRE_REAL_PQC=true``
is in the environment, ``import omni_mercury_engine`` (which calls the
gate exactly once at package-load time) must raise ``RuntimeError`` if
any of the three AMA algorithms (Dilithium, Kyber, SPHINCS) is missing
or unavailable, so a process cannot start in a cryptographically
incomplete state.  Without the env var, the gate is a no-op and the
package imports against the soft PQC stubs in ``security/pqc_backends.py``
for development convenience.

These tests inject a fake ``ama_cryptography`` module into
``sys.modules`` with the three ``*_AVAILABLE`` flags set to chosen
values, then invoke the gate directly.  This matches the gate's
actual contract (it reads top-level package attributes, mirroring
how ``security/pqc_backends.py`` consumes them) and avoids any
dependence on whether ``ama_cryptography`` is installed in the test
environment.
"""

from __future__ import annotations

import sys
import types

import pytest

from omni_mercury_engine._pqc_gate import _enforce_pqc_production_gate


def _install_fake_ama(
    monkeypatch: pytest.MonkeyPatch,
    *,
    dilithium: bool | None,
    kyber: bool | None,
    sphincs: bool | None,
) -> None:
    """Inject a fake ``ama_cryptography`` package into ``sys.modules``.

    Each flag may be ``True``, ``False``, or ``None``.  ``None`` means
    the attribute is omitted entirely (so ``getattr(..., default=False)``
    falls through to ``False``).
    """
    fake = types.ModuleType("ama_cryptography")
    if dilithium is not None:
        fake.DILITHIUM_AVAILABLE = dilithium  # type: ignore[attr-defined]
    if kyber is not None:
        fake.KYBER_AVAILABLE = kyber  # type: ignore[attr-defined]
    if sphincs is not None:
        fake.SPHINCS_AVAILABLE = sphincs  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ama_cryptography", fake)


def _uninstall_ama(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import ama_cryptography`` raise ``ImportError``."""
    monkeypatch.setitem(sys.modules, "ama_cryptography", None)  # type: ignore[arg-type]


class TestNoOpWhenEnvUnset:
    def test_returns_silently_when_env_var_is_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AMA_REQUIRE_REAL_PQC", raising=False)
        monkeypatch.delenv("AVA_REQUIRE_REAL_PQC", raising=False)
        _enforce_pqc_production_gate()  # must not raise

    def test_returns_silently_when_env_var_is_explicitly_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AMA_REQUIRE_REAL_PQC", "false")
        _enforce_pqc_production_gate()


class TestFailClosedWhenLibMissing:
    def test_raises_when_ama_cryptography_not_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AMA_REQUIRE_REAL_PQC", "true")
        _uninstall_ama(monkeypatch)

        with pytest.raises(RuntimeError) as excinfo:
            _enforce_pqc_production_gate()

        msg = str(excinfo.value)
        assert "AMA_REQUIRE_REAL_PQC=true" in msg
        assert "import ama_cryptography failed" in msg
        # Recovery hint must point at the verified clone-and-build path.
        assert "git clone" in msg
        assert "AMA-Cryptography.git" in msg
        assert "AMA_NO_CYTHON=1" in msg


class TestFailClosedOnPartialBuild:
    """When a partially built install has Dilithium but not Kyber/SPHINCS,
    the gate must reject it — Mercury exposes those algorithms elsewhere
    and a partial build is a cryptographically incomplete state."""

    def test_raises_when_only_dilithium_is_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AMA_REQUIRE_REAL_PQC", "true")
        _install_fake_ama(monkeypatch, dilithium=True, kyber=False, sphincs=False)

        with pytest.raises(RuntimeError) as excinfo:
            _enforce_pqc_production_gate()

        msg = str(excinfo.value)
        assert "Kyber" in msg
        assert "SPHINCS" in msg
        # Dilithium WAS available, so it must NOT be listed.
        assert "Dilithium" not in msg and "ML-DSA-65" not in msg

    def test_raises_when_attribute_is_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If the attribute is absent on ``ama_cryptography``, treat as
        unavailable.  Mirrors what ``getattr(..., default=False)`` does
        when an older AMA version doesn't define one of the flags."""
        monkeypatch.setenv("AMA_REQUIRE_REAL_PQC", "true")
        _install_fake_ama(monkeypatch, dilithium=True, kyber=True, sphincs=None)

        with pytest.raises(RuntimeError) as excinfo:
            _enforce_pqc_production_gate()

        msg = str(excinfo.value)
        assert "SPHINCS" in msg


class TestPassesOnCompleteInstall:
    def test_returns_silently_when_all_three_flags_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mirrors the ``verify-real-pqc`` workflow's expectation:
        when AMA is built and the top-level package exposes all three
        ``*_AVAILABLE`` flags as True, the gate is a no-op."""
        monkeypatch.setenv("AMA_REQUIRE_REAL_PQC", "true")
        _install_fake_ama(monkeypatch, dilithium=True, kyber=True, sphincs=True)

        _enforce_pqc_production_gate()  # must not raise


class TestErrorMessageContents:
    def test_message_lists_only_missing_algos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An operator hitting the gate must see the *full* list of
        missing algorithms, not just the first one — so they don't
        rebuild, hit the next missing one, and have to build again."""
        monkeypatch.setenv("AMA_REQUIRE_REAL_PQC", "true")
        _install_fake_ama(monkeypatch, dilithium=False, kyber=False, sphincs=False)

        with pytest.raises(RuntimeError) as excinfo:
            _enforce_pqc_production_gate()

        msg = str(excinfo.value)
        assert "Dilithium" in msg or "ML-DSA-65" in msg
        assert "Kyber" in msg
        assert "SPHINCS" in msg
