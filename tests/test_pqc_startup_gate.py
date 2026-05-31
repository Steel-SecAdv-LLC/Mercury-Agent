"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Smoke tests for the import-time PQC gate in
``src/omni_mercury_engine/_pqc_gate.py::_enforce_pqc_production_gate``.

Scope
-----
This file now pins the universal fail-closed contract: the gate always
requires a real AMA Cryptography v3.2.0 native build.  The env vars
``AMA_REQUIRE_REAL_PQC`` / ``AVA_REQUIRE_REAL_PQC`` are compatibility
diagnostics only; unset or false values must not disable the gate.
"""

from __future__ import annotations

import os

import ama_cryptography.pqc_backends as ama_pqc_backends

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
