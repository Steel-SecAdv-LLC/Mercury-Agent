#!/usr/bin/env python3
"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Integration tests for cross-repo enum compatibility (Priority 3).

Verifies that Mercury-Agent's posture enums and their scalar mappings agree
with the real ``ama_cryptography.adaptive_posture`` definitions. These tests
exercise the *production* symbols exported from
``omni_mercury_engine.integrations.mercury_amacrypto`` so they fail loudly if
the stub enums or module-level maps drift out of sync with the upstream
``ama_cryptography`` package.
"""

from __future__ import annotations

import importlib

import pytest

_MODULE_NAME = "omni_mercury_engine.integrations.mercury_amacrypto"


@pytest.fixture(scope="module")
def mercury_amacrypto():
    """Import ``mercury_amacrypto`` or skip the test suite cleanly."""
    try:
        return importlib.import_module(_MODULE_NAME)
    except ImportError as exc:  # pragma: no cover - environment safeguard
        pytest.skip(f"{_MODULE_NAME} not importable: {exc}")


EXPECTED_THREAT_LEVEL_MEMBERS = {"NOMINAL", "ELEVATED", "HIGH", "CRITICAL"}
EXPECTED_POSTURE_ACTION_MEMBERS = {
    "NONE",
    "INCREASE_MONITORING",
    "ROTATE_KEYS",
    "SWITCH_ALGORITHM",
    "ROTATE_AND_SWITCH",
}
# Enum members that were removed when the stubs were aligned with
# ``ama_cryptography.adaptive_posture`` and must never reappear.
REMOVED_THREAT_LEVEL_MEMBERS = {"LOW", "MEDIUM"}
REMOVED_POSTURE_ACTION_MEMBERS = {"ALERT"}


class TestEnumCompatibility:
    """Verify Mercury-Agent posture enums match AMA Cryptography real enums."""

    def test_threat_level_members(self, mercury_amacrypto) -> None:
        """``ThreatLevel`` must expose exactly the expected members."""
        threat_level = mercury_amacrypto.ThreatLevel
        assert set(threat_level.__members__) == EXPECTED_THREAT_LEVEL_MEMBERS

    def test_posture_action_members(self, mercury_amacrypto) -> None:
        """``PostureAction`` must expose exactly the expected members."""
        posture_action = mercury_amacrypto.PostureAction
        assert set(posture_action.__members__) == EXPECTED_POSTURE_ACTION_MEMBERS

    def test_no_removed_threat_level_members(self, mercury_amacrypto) -> None:
        """Legacy ``ThreatLevel`` members (LOW/MEDIUM) must not reappear."""
        threat_level = mercury_amacrypto.ThreatLevel
        leaked = REMOVED_THREAT_LEVEL_MEMBERS & set(threat_level.__members__)
        assert not leaked, f"Removed ThreatLevel members resurfaced: {sorted(leaked)}"

    def test_no_removed_posture_action_members(self, mercury_amacrypto) -> None:
        """Legacy ``PostureAction`` members (ALERT) must not reappear."""
        posture_action = mercury_amacrypto.PostureAction
        leaked = REMOVED_POSTURE_ACTION_MEMBERS & set(posture_action.__members__)
        assert not leaked, f"Removed PostureAction members resurfaced: {sorted(leaked)}"

    def test_threat_level_map_covers_all_members(self, mercury_amacrypto) -> None:
        """``THREAT_LEVEL_MAP`` must cover every ``ThreatLevel`` member."""
        threat_level = mercury_amacrypto.ThreatLevel
        threat_level_map = mercury_amacrypto.THREAT_LEVEL_MAP
        assert set(threat_level_map.keys()) == set(threat_level)
        # Values should be finite floats bounded by [0.0, 1.0] so the scalar
        # can be registered directly into GOSNN without further normalisation.
        for member, value in threat_level_map.items():
            assert isinstance(value, float), f"{member} -> {value!r} is not float"
            assert 0.0 <= value <= 1.0, f"{member} -> {value} outside [0, 1]"

    def test_action_map_covers_all_members(self, mercury_amacrypto) -> None:
        """``ACTION_MAP`` must cover every ``PostureAction`` member."""
        posture_action = mercury_amacrypto.PostureAction
        action_map = mercury_amacrypto.ACTION_MAP
        assert set(action_map.keys()) == set(posture_action)
        for member, value in action_map.items():
            assert isinstance(value, float), f"{member} -> {value!r} is not float"
            assert value >= 0.0, f"{member} -> {value} must be non-negative"

    def test_threat_level_map_is_monotonic(self, mercury_amacrypto) -> None:
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

    def test_action_map_is_monotonic(self, mercury_amacrypto) -> None:
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
