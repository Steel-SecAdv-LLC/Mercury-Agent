"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Smoke tests for the import-time PQC production gate in
``src/omni_mercury_engine/__init__.py::_enforce_pqc_production_gate``.

The gate is a defensive fail-closed: when ``AMA_REQUIRE_REAL_PQC=true``
is in the environment, ``import omni_mercury_engine`` must raise
``RuntimeError`` if any of the three AMA algorithms (Dilithium, Kyber,
SPHINCS) is missing or unavailable, so a process cannot start in a
cryptographically incomplete state.  Without the env var, the gate is
a no-op and the package imports against the soft PQC stubs in
``security/pqc_backends.py`` for development convenience.

These tests exercise the gate function in isolation (rather than
re-importing the package, which the import system caches) so each
case is hermetic.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_INIT_PATH = _REPO_ROOT / "src" / "omni_mercury_engine" / "__init__.py"


def _load_gate() -> Any:
    """Re-load the gate function from a fresh module copy.

    We can't simply ``import omni_mercury_engine._enforce_pqc_production_gate``
    because ``__init__.py`` immediately ``del``s the symbol after running
    it once at package-import time.  Loading the source as an isolated
    module gives us back a callable handle.
    """
    spec = importlib.util.spec_from_file_location("_pqc_gate_test_copy", _INIT_PATH)
    assert spec is not None and spec.loader is not None
    module = types.ModuleType(spec.name)
    # Re-exec the module body but capture the function before the
    # ``_enforce_pqc_production_gate()`` self-call at the bottom is
    # reached, by stubbing the call.
    src = _INIT_PATH.read_text(encoding="utf-8")
    # Strip the self-call + del so we keep the function definition.
    src = src.replace(
        "_enforce_pqc_production_gate()\ndel _enforce_pqc_production_gate",
        "",
    )
    exec(compile(src, str(_INIT_PATH), "exec"), module.__dict__)
    gate = module.__dict__.get("_enforce_pqc_production_gate")
    assert gate is not None, "gate function not found in __init__.py"
    return gate


@pytest.fixture
def gate() -> Any:
    return _load_gate()


class TestNoOpWhenEnvUnset:
    def test_returns_silently_when_env_var_is_missing(
        self, gate: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AMA_REQUIRE_REAL_PQC", raising=False)
        monkeypatch.delenv("AVA_REQUIRE_REAL_PQC", raising=False)
        gate()  # must not raise

    def test_returns_silently_when_env_var_is_explicitly_false(
        self, gate: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AMA_REQUIRE_REAL_PQC", "false")
        gate()


class TestFailClosedWhenLibMissing:
    def test_raises_when_ama_cryptography_not_installed(
        self, gate: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AMA_REQUIRE_REAL_PQC", "true")
        # Force every ``ama_cryptography.*`` import to fail.
        original_import = importlib.import_module

        def fail_for_ama(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.startswith("ama_cryptography"):
                raise ImportError(f"stubbed: {name} not available")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", fail_for_ama)

        with pytest.raises(RuntimeError) as excinfo:
            gate()

        msg = str(excinfo.value)
        assert "AMA_REQUIRE_REAL_PQC=true" in msg
        # Every algorithm is missing here — confirm the message names
        # all three so an operator hitting this gate sees the full list.
        assert "Dilithium" in msg or "ML-DSA-65" in msg
        assert "Kyber" in msg
        assert "SPHINCS" in msg
        # Recovery hint must point at the verified clone-and-build path,
        # not the broken `cmake -B build` from the Mercury checkout.
        assert "git clone" in msg
        assert "AMA-Cryptography.git" in msg
        assert "AMA_NO_CYTHON=1" in msg


class TestFailClosedOnPartialBuild:
    """When a partially built install has Dilithium but not Kyber/SPHINCS,
    the gate must reject it — Mercury exposes those algorithms elsewhere
    and a partial build is a cryptographically incomplete state, which
    matches the contract on ``security.pqc_guards.check_pqc_production_readiness``."""

    def test_raises_when_only_dilithium_is_available(
        self, gate: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AMA_REQUIRE_REAL_PQC", "true")

        fake_dilithium = types.ModuleType("ama_cryptography.dilithium")
        fake_dilithium.DILITHIUM_AVAILABLE = True  # type: ignore[attr-defined]

        original_import = importlib.import_module

        def selective_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "ama_cryptography.dilithium":
                return fake_dilithium
            if name in ("ama_cryptography.kyber", "ama_cryptography.sphincs"):
                raise ImportError(f"stubbed: {name} not available")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", selective_import)

        with pytest.raises(RuntimeError) as excinfo:
            gate()

        msg = str(excinfo.value)
        assert "Kyber" in msg
        assert "SPHINCS" in msg
        # Dilithium WAS available, so it must NOT be listed.
        assert "Dilithium" not in msg and "ML-DSA-65" not in msg

    def test_raises_when_flag_is_false_even_if_module_imports(
        self, gate: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Module loadable but `*_AVAILABLE` flag is False (the post-import
        runtime probe failed) — gate must still reject."""
        monkeypatch.setenv("AMA_REQUIRE_REAL_PQC", "true")

        all_loaded_with_flags_false: dict[str, types.ModuleType] = {}
        for name, flag in [
            ("ama_cryptography.dilithium", "DILITHIUM_AVAILABLE"),
            ("ama_cryptography.kyber", "KYBER_AVAILABLE"),
            ("ama_cryptography.sphincs", "SPHINCS_AVAILABLE"),
        ]:
            mod = types.ModuleType(name)
            setattr(mod, flag, False)
            all_loaded_with_flags_false[name] = mod

        original_import = importlib.import_module

        def serve_fakes(name: str, *args: Any, **kwargs: Any) -> Any:
            if name in all_loaded_with_flags_false:
                return all_loaded_with_flags_false[name]
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", serve_fakes)

        with pytest.raises(RuntimeError):
            gate()


class TestPassesOnCompleteInstall:
    def test_returns_silently_when_all_three_algos_available(
        self, gate: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AMA_REQUIRE_REAL_PQC", "true")

        complete: dict[str, types.ModuleType] = {}
        for name, flag in [
            ("ama_cryptography.dilithium", "DILITHIUM_AVAILABLE"),
            ("ama_cryptography.kyber", "KYBER_AVAILABLE"),
            ("ama_cryptography.sphincs", "SPHINCS_AVAILABLE"),
        ]:
            mod = types.ModuleType(name)
            setattr(mod, flag, True)
            complete[name] = mod

        original_import = importlib.import_module

        def serve_fakes(name: str, *args: Any, **kwargs: Any) -> Any:
            if name in complete:
                return complete[name]
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(importlib, "import_module", serve_fakes)

        gate()  # must not raise
