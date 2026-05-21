"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Smoke tests for the import-time PQC production gate in
``src/omni_mercury_engine/_pqc_gate.py::_enforce_pqc_production_gate``.

Scope
-----
This file pins **only** the no-op contract: when
``AMA_REQUIRE_REAL_PQC`` is unset (the development-mode default),
the gate must return silently and ``import omni_mercury_engine``
must succeed regardless of whether the AMA Cryptography native
library is installed.

The gate's real-AMA fail-closed behaviour is exercised end-to-end
by ``.github/workflows/pqc-production-check.yml`` against the
actual AMA Cryptography v3.2.0 native build — that's the
authoritative test for the AMA-required path.  We deliberately
do NOT inject a fake ``ama_cryptography`` module here.  Mocking
the dependency would test our mock, not the real production
contract; the verify-real-pqc CI lane is the canonical place to
exercise that contract because it builds the actual library and
runs the gate against it.
"""

from __future__ import annotations

import os

from omni_mercury_engine._pqc_gate import _enforce_pqc_production_gate


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


class TestNoOpWhenEnvUnset:
    """The dev-mode default: env unset, gate is a silent no-op.

    These cases do not need ``ama_cryptography`` to be installed —
    the gate returns before touching the package when the env var
    is not set."""

    def test_returns_silently_when_env_var_is_missing(self) -> None:
        saved = _scrub_real_pqc_env()
        try:
            _enforce_pqc_production_gate()  # must not raise
        finally:
            _restore_env(saved)

    def test_returns_silently_when_env_var_is_explicitly_false(self) -> None:
        saved = _scrub_real_pqc_env()
        os.environ["AMA_REQUIRE_REAL_PQC"] = "false"
        try:
            _enforce_pqc_production_gate()
        finally:
            os.environ.pop("AMA_REQUIRE_REAL_PQC", None)
            _restore_env(saved)

    def test_returns_silently_when_legacy_env_var_is_unset(self) -> None:
        """``AVA_REQUIRE_REAL_PQC`` is the legacy compat name for the
        same flag.  Either being unset (or set to a non-truthy value)
        keeps the gate in no-op mode."""
        saved = _scrub_real_pqc_env()
        try:
            _enforce_pqc_production_gate()
        finally:
            _restore_env(saved)
