# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Integration tests for cross-repo enum compatibility.

Verifies that Mercury Agent's posture enums and their scalar mappings agree
with the real ``ama_cryptography.adaptive_posture`` definitions. These tests
exercise the *production* symbols exported from
``omni_mercury_engine.integrations.mercury_amacrypto`` so they fail loudly if
module-level maps drift out of sync with the upstream ``ama_cryptography``
package.
"""

from __future__ import annotations

import importlib
import math
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from types import ModuleType

_MODULE_NAME = "omni_mercury_engine.integrations.mercury_amacrypto"


@pytest.fixture(scope="module")
def mercury_amacrypto() -> ModuleType:
    """Import ``mercury_amacrypto``; missing AMA fails collection."""
    return importlib.import_module(_MODULE_NAME)


EXPECTED_THREAT_LEVEL_MEMBERS = {"NOMINAL", "ELEVATED", "HIGH", "CRITICAL"}
EXPECTED_POSTURE_ACTION_MEMBERS = {
    "NONE",
    "INCREASE_MONITORING",
    "ROTATE_KEYS",
    "SWITCH_ALGORITHM",
    "ROTATE_AND_SWITCH",
}
# Legacy enum members that must never reappear.
REMOVED_THREAT_LEVEL_MEMBERS = {"LOW", "MEDIUM"}
REMOVED_POSTURE_ACTION_MEMBERS = {"ALERT"}


class TestEnumCompatibility:
    """Verify Mercury Agent posture enums match AMA Cryptography real enums."""

    def test_threat_level_members(self, mercury_amacrypto: ModuleType) -> None:
        """``ThreatLevel`` must expose exactly the expected members."""
        threat_level = mercury_amacrypto.ThreatLevel
        assert set(threat_level.__members__) == EXPECTED_THREAT_LEVEL_MEMBERS

    def test_posture_action_members(self, mercury_amacrypto: ModuleType) -> None:
        """``PostureAction`` must expose exactly the expected members."""
        posture_action = mercury_amacrypto.PostureAction
        assert set(posture_action.__members__) == EXPECTED_POSTURE_ACTION_MEMBERS

    def test_no_removed_threat_level_members(self, mercury_amacrypto: ModuleType) -> None:
        """Legacy ``ThreatLevel`` members (LOW/MEDIUM) must not reappear."""
        threat_level = mercury_amacrypto.ThreatLevel
        leaked = REMOVED_THREAT_LEVEL_MEMBERS & set(threat_level.__members__)
        assert not leaked, f"Removed ThreatLevel members resurfaced: {sorted(leaked)}"

    def test_no_removed_posture_action_members(self, mercury_amacrypto: ModuleType) -> None:
        """Legacy ``PostureAction`` members (ALERT) must not reappear."""
        posture_action = mercury_amacrypto.PostureAction
        leaked = REMOVED_POSTURE_ACTION_MEMBERS & set(posture_action.__members__)
        assert not leaked, f"Removed PostureAction members resurfaced: {sorted(leaked)}"

    def test_threat_level_map_covers_all_members(self, mercury_amacrypto: ModuleType) -> None:
        """``THREAT_LEVEL_MAP`` must cover every ``ThreatLevel`` member."""
        threat_level = mercury_amacrypto.ThreatLevel
        threat_level_map = mercury_amacrypto.THREAT_LEVEL_MAP
        assert set(threat_level_map.keys()) == set(threat_level)
        # Values should be finite floats bounded by [0.0, 1.0] so the scalar
        # can be registered directly into GOSNN without further normalisation.
        for member, value in threat_level_map.items():
            assert isinstance(value, float), f"{member} -> {value!r} is not float"
            assert math.isfinite(value)
            assert 0.0 <= value <= 1.0, f"{member} -> {value} outside [0, 1]"

    def test_action_map_covers_all_members(self, mercury_amacrypto: ModuleType) -> None:
        """``ACTION_MAP`` must cover every ``PostureAction`` member."""
        posture_action = mercury_amacrypto.PostureAction
        action_map = mercury_amacrypto.ACTION_MAP
        assert set(action_map.keys()) == set(posture_action)
        for member, value in action_map.items():
            assert isinstance(value, float), f"{member} -> {value!r} is not float"
            assert math.isfinite(value)
            assert value >= 0.0, f"{member} -> {value} must be non-negative"

    def test_threat_level_map_is_monotonic(self, mercury_amacrypto: ModuleType) -> None:
        """Severity ordering NOMINAL < ELEVATED < HIGH < CRITICAL must hold."""
        threat_level = mercury_amacrypto.ThreatLevel
        threat_level_map = mercury_amacrypto.THREAT_LEVEL_MAP
        ordered = [
            threat_level.NOMINAL,
            threat_level.ELEVATED,
            threat_level.HIGH,
            threat_level.CRITICAL,
        ]
        values = [threat_level_map[level] for level in ordered]
        assert values == sorted(values), (
            "THREAT_LEVEL_MAP values are not monotonically non-decreasing: "
            f"{list(zip([level.name for level in ordered], values))}"
        )

    def test_action_map_is_monotonic(self, mercury_amacrypto: ModuleType) -> None:
        """Action severity NONE < INCREASE_MONITORING < ... < ROTATE_AND_SWITCH."""
        posture_action = mercury_amacrypto.PostureAction
        action_map = mercury_amacrypto.ACTION_MAP
        ordered = [
            posture_action.NONE,
            posture_action.INCREASE_MONITORING,
            posture_action.ROTATE_KEYS,
            posture_action.SWITCH_ALGORITHM,
            posture_action.ROTATE_AND_SWITCH,
        ]
        values = [action_map[action] for action in ordered]
        assert values == sorted(values), (
            "ACTION_MAP values are not monotonically non-decreasing: "
            f"{list(zip([action.name for action in ordered], values))}"
        )

    def test_pqc_backend_source_value(self, mercury_amacrypto: ModuleType) -> None:
        """``_PQC_BACKEND_SOURCE`` must report the mandatory AMA backend."""
        source = mercury_amacrypto._PQC_BACKEND_SOURCE
        assert source == "ama_cryptography"

    def test_get_pqc_status_includes_backend_source(self, mercury_amacrypto: ModuleType) -> None:
        """``get_pqc_status()`` must surface ``pqc_backend_source``."""
        adapter = mercury_amacrypto.create_ama_cryptography_adapter(
            enable_timing_monitor=False, gosnn_synapse_enabled=False
        )
        status = adapter.get_pqc_status()
        assert "pqc_backend_source" in status
        assert status["pqc_backend_source"] == mercury_amacrypto._PQC_BACKEND_SOURCE


class TestSanitizeScalars:
    """Verify ``MercuryGuardianAdapter._sanitize_scalars`` defends GOSNN."""

    def _adapter(self, mercury_amacrypto: ModuleType) -> Any:
        return mercury_amacrypto.create_ama_cryptography_adapter(
            enable_timing_monitor=False, gosnn_synapse_enabled=False
        )

    def test_passes_through_finite_floats(self, mercury_amacrypto: ModuleType) -> None:
        adapter = self._adapter(mercury_amacrypto)
        clean = adapter._sanitize_scalars({"a": 0.0, "b": 1.5, "c": -2.25})
        assert clean == {"a": 0.0, "b": 1.5, "c": -2.25}

    def test_drops_nan_and_inf_never_clamps(self, mercury_amacrypto: ModuleType) -> None:
        """Non-finite scalars are EXCLUDED, never clamped to 0.0.

        Regression: these used to be sanitized to 0.0 and registered into the
        SECURITY scalar group, where a fabricated zero is indistinguishable
        from a measured quiet value (F10 never-clamp principle).
        """
        adapter = self._adapter(mercury_amacrypto)
        clean = adapter._sanitize_scalars(
            {
                "nan": float("nan"),
                "pos_inf": float("inf"),
                "neg_inf": float("-inf"),
                "ok": 0.42,
            }
        )
        assert clean == {"ok": 0.42}

    def test_coerces_int_and_bool_to_float(self, mercury_amacrypto: ModuleType) -> None:
        adapter = self._adapter(mercury_amacrypto)
        clean = adapter._sanitize_scalars({"i": 7, "b_true": True, "b_false": False})
        assert clean == {"i": 7.0, "b_true": 1.0, "b_false": 0.0}
        for value in clean.values():
            assert isinstance(value, float)

    def test_drops_non_numeric_values(self, mercury_amacrypto: ModuleType) -> None:
        adapter = self._adapter(mercury_amacrypto)
        clean = adapter._sanitize_scalars(
            {"keep": 1.0, "drop_str": "not-a-number", "drop_none": None, "drop_obj": object()}
        )
        assert clean == {"keep": 1.0}

    def test_drops_overflowing_value_without_raising(self, mercury_amacrypto: ModuleType) -> None:
        """``float(huge_int)`` raises ``OverflowError`` — that key must be dropped,
        not allowed to propagate and break the entire scalar registration."""
        adapter = self._adapter(mercury_amacrypto)
        clean = adapter._sanitize_scalars({"keep": 1.0, "huge": 10**10000})
        assert clean == {"keep": 1.0}

    def test_empty_input_returns_empty_dict(self, mercury_amacrypto: ModuleType) -> None:
        adapter = self._adapter(mercury_amacrypto)
        assert adapter._sanitize_scalars({}) == {}

    def test_output_is_a_new_dict(self, mercury_amacrypto: ModuleType) -> None:
        """Sanitization must not mutate the caller's dict."""
        adapter = self._adapter(mercury_amacrypto)
        source = {"a": float("nan"), "b": 1.0}
        clean = adapter._sanitize_scalars(source)
        # caller's dict left intact
        assert math.isnan(source["a"])
        assert source["b"] == 1.0
        # returned dict excludes the non-finite value (dropped loudly)
        assert clean == {"b": 1.0}
        assert clean is not source
