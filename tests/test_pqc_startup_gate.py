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

These tests inject a fake ``ama_cryptography.pqc_backends`` submodule
into ``sys.modules`` with the three ``*_AVAILABLE`` flags set to
chosen values, then invoke the gate directly.  The submodule
location matches the gate's actual contract — it reads from
``ama_cryptography.pqc_backends`` (the canonical location, mirroring
how ``security/pqc_backends.py`` consumes those flags via
``from ama_cryptography.pqc_backends import ...``) and not from the
top-level ``ama_cryptography`` package or per-algorithm submodules.
The fake-module fixture avoids any dependence on whether the real
``ama_cryptography`` package is installed in the test environment.

A separate ``TestPackageImportInvokesGate`` class also exercises
the real ``import omni_mercury_engine`` path to pin the contract
that ``__init__.py`` invokes the gate exactly once at package-load
time — the per-function direct-call tests would otherwise stay
green if ``__init__.py`` regressed by skipping the self-call.
"""

from __future__ import annotations

import sys
import types
import warnings
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


def _gate() -> Callable[[], None]:
    """Lazy import of the gate function.

    Imported inside the call so module collection does NOT trigger
    ``omni_mercury_engine.__init__``'s package-level self-call before
    the per-test ``monkeypatch`` fixture has had a chance to scrub
    ``AMA_REQUIRE_REAL_PQC`` from the environment.  Without this lazy
    pattern, a CI lane that sets the env var globally would fail
    collection on this file rather than the per-test cases reaching
    their assertions.
    """
    from omni_mercury_engine._pqc_gate import _enforce_pqc_production_gate

    return _enforce_pqc_production_gate


def _install_fake_ama(
    monkeypatch: pytest.MonkeyPatch,
    *,
    dilithium: bool | None,
    kyber: bool | None,
    sphincs: bool | None,
) -> None:
    """Inject a fake ``ama_cryptography.pqc_backends`` submodule into
    ``sys.modules`` with the three ``*_AVAILABLE`` flags set.

    The gate reads the flags from ``ama_cryptography.pqc_backends`` (the
    canonical location, matching what ``security/pqc_backends.py`` does
    via ``from ama_cryptography.pqc_backends import ...``).  Each flag
    may be ``True``, ``False``, or ``None``.  ``None`` means the
    attribute is omitted entirely (so ``getattr(..., default=False)``
    falls through to ``False``).
    """
    fake_pkg = types.ModuleType("ama_cryptography")
    fake_submodule = types.ModuleType("ama_cryptography.pqc_backends")
    if dilithium is not None:
        fake_submodule.DILITHIUM_AVAILABLE = dilithium  # type: ignore[attr-defined]
    if kyber is not None:
        fake_submodule.KYBER_AVAILABLE = kyber  # type: ignore[attr-defined]
    if sphincs is not None:
        fake_submodule.SPHINCS_AVAILABLE = sphincs  # type: ignore[attr-defined]
    # Wire the submodule onto the parent package so ``import
    # ama_cryptography.pqc_backends`` resolves both via sys.modules
    # *and* via attribute access on the parent.
    fake_pkg.pqc_backends = fake_submodule  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ama_cryptography", fake_pkg)
    monkeypatch.setitem(sys.modules, "ama_cryptography.pqc_backends", fake_submodule)


def _uninstall_ama(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import ama_cryptography.pqc_backends`` raise ``ImportError``."""
    monkeypatch.setitem(sys.modules, "ama_cryptography", None)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, "ama_cryptography.pqc_backends", None)  # type: ignore[arg-type]


class TestNoOpWhenEnvUnset:
    def test_returns_silently_when_env_var_is_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AMA_REQUIRE_REAL_PQC", raising=False)
        monkeypatch.delenv("AVA_REQUIRE_REAL_PQC", raising=False)
        _gate()()  # must not raise

    def test_returns_silently_when_env_var_is_explicitly_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AMA_REQUIRE_REAL_PQC", "false")
        _gate()()


class TestFailClosedWhenLibMissing:
    def test_raises_when_ama_cryptography_not_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AMA_REQUIRE_REAL_PQC", "true")
        _uninstall_ama(monkeypatch)

        with pytest.raises(RuntimeError) as excinfo:
            _gate()()

        msg = str(excinfo.value)
        assert "AMA_REQUIRE_REAL_PQC=true" in msg
        assert "import ama_cryptography.pqc_backends failed" in msg
        # Recovery hint must point at the verified clone-and-build path.
        assert "git clone" in msg
        assert "AMA-Cryptography.git" in msg
        assert "AMA_NO_CYTHON=1" in msg


class TestHardRequiredFlags:
    """Dilithium and Kyber are hard-required.  SPHINCS+ is intentionally
    soft-required (see the ``_pqc_gate.py`` rationale: the upstream
    ``pqc-production-check.yml`` workflow doesn't assert
    ``SPHINCS_AVAILABLE`` on a real v3.1.0 build, so requiring it here
    would produce false-positive partial-install rejections)."""

    def test_raises_when_dilithium_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AMA_REQUIRE_REAL_PQC", "true")
        _install_fake_ama(monkeypatch, dilithium=False, kyber=True, sphincs=True)

        with pytest.raises(RuntimeError) as excinfo:
            _gate()()

        msg = str(excinfo.value)
        assert "Dilithium" in msg or "ML-DSA-65" in msg
        assert "Kyber" not in msg

    def test_raises_when_kyber_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AMA_REQUIRE_REAL_PQC", "true")
        _install_fake_ama(monkeypatch, dilithium=True, kyber=False, sphincs=True)

        with pytest.raises(RuntimeError) as excinfo:
            _gate()()

        msg = str(excinfo.value)
        assert "Kyber" in msg
        assert "Dilithium" not in msg and "ML-DSA-65" not in msg


class TestSoftRequiredSphincs:
    """A missing SPHINCS surface emits ``UserWarning`` but does not
    raise.  This matches what the verify-real-pqc CI lane will see on
    a real AMA v3.1.0 build (where SPHINCS_AVAILABLE is not always
    populated even after a successful native build)."""

    def test_warns_but_does_not_raise_when_sphincs_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AMA_REQUIRE_REAL_PQC", "true")
        _install_fake_ama(monkeypatch, dilithium=True, kyber=True, sphincs=False)

        with pytest.warns(UserWarning, match="SPHINCS"):
            _gate()()

    def test_warns_when_sphincs_attribute_is_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Older AMA versions that don't define ``SPHINCS_AVAILABLE`` at
        all are still treated as soft-warning, not hard failure."""
        monkeypatch.setenv("AMA_REQUIRE_REAL_PQC", "true")
        _install_fake_ama(monkeypatch, dilithium=True, kyber=True, sphincs=None)

        with pytest.warns(UserWarning, match="SPHINCS"):
            _gate()()


class TestPassesOnCompleteInstall:
    def test_returns_silently_when_all_three_flags_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When AMA is built and exposes all three ``*_AVAILABLE`` flags
        as True, the gate is a silent no-op (no warning, no exception)."""
        monkeypatch.setenv("AMA_REQUIRE_REAL_PQC", "true")
        _install_fake_ama(monkeypatch, dilithium=True, kyber=True, sphincs=True)

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning would now raise
            _gate()()


class TestPackageImportInvokesGate:
    """Pin that ``import omni_mercury_engine`` actually invokes the
    gate exactly once at package-load time.  Without this, the
    per-function direct-call tests above would stay green even if
    ``__init__.py`` regressed by skipping the self-call (the very
    contract that distinguishes "wired into startup" from "available
    helper an operator must remember to call themselves")."""

    def test_init_calls_gate_with_complete_install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Re-import the package with a fake AMA fully populated and
        the env var set.  If ``__init__.py`` invokes the gate, this
        succeeds silently.  If it skips the self-call, this still
        succeeds — but the *next* test below would catch that case.
        We use this success path mainly to pin that the gate's no-op
        path doesn't blow up package import."""
        import importlib

        monkeypatch.setenv("AMA_REQUIRE_REAL_PQC", "true")
        _install_fake_ama(monkeypatch, dilithium=True, kyber=True, sphincs=True)

        for name in ("omni_mercury_engine", "omni_mercury_engine._pqc_gate"):
            monkeypatch.delitem(sys.modules, name, raising=False)

        import omni_mercury_engine  # noqa: F401  (importing for side effect)

        # Re-import again to confirm the gate's self-call is idempotent
        # under repeated imports (the second import is a no-op because
        # Python caches the module).
        importlib.reload(sys.modules["omni_mercury_engine"])

    def test_init_self_call_raises_on_missing_lib(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The contract: package import MUST raise when the env var
        is set and AMA is not loadable.  This is the only test that
        exercises the actual ``__init__.py``-invokes-gate wiring; if
        ``__init__.py`` ever stops calling the gate, this test goes
        green by accident — but the missing self-call would be caught
        because we explicitly delete cached modules first and observe
        that import-of-the-package itself raises, not just a manual
        ``_enforce_pqc_production_gate()`` call."""
        monkeypatch.setenv("AMA_REQUIRE_REAL_PQC", "true")
        _uninstall_ama(monkeypatch)

        for name in ("omni_mercury_engine", "omni_mercury_engine._pqc_gate"):
            monkeypatch.delitem(sys.modules, name, raising=False)

        with pytest.raises(RuntimeError, match="AMA_REQUIRE_REAL_PQC=true"):
            import omni_mercury_engine  # noqa: F401


class TestErrorMessageContents:
    def test_message_lists_missing_hard_required_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When both hard-required flags are missing, both are named so
        an operator can fix them in one rebuild.  SPHINCS is intentionally
        omitted from the hard-failure message even when also missing
        (the warning path covers SPHINCS separately)."""
        monkeypatch.setenv("AMA_REQUIRE_REAL_PQC", "true")
        _install_fake_ama(monkeypatch, dilithium=False, kyber=False, sphincs=False)

        with pytest.raises(RuntimeError) as excinfo:
            _gate()()

        msg = str(excinfo.value)
        assert "Dilithium" in msg or "ML-DSA-65" in msg
        assert "Kyber" in msg
        assert "SPHINCS" not in msg
