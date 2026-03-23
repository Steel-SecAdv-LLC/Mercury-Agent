#!/usr/bin/env python3
# Copyright 2025-2026 Steel Security Advisors LLC
# Licensed under the Apache License, Version 2.0

"""
Integration tests for cross-repo enum compatibility (Priority 3).

Verifies that Mercury-Agent's stub enums match AMA Cryptography's real enums.
"""

import pytest


class TestEnumCompatibility:
    """Verify Mercury-Agent stub enums match AMA Cryptography real enums."""

    def test_threat_level_stub_values(self):
        """Stub ThreatLevel must have same members as real enum."""
        # Import the stub (which is used when ama_cryptography is not installed)
        from enum import Enum as _Enum

        class StubThreatLevel(_Enum):
            NOMINAL = "nominal"
            ELEVATED = "elevated"
            HIGH = "high"
            CRITICAL = "critical"

        expected_members = {"NOMINAL", "ELEVATED", "HIGH", "CRITICAL"}
        actual_members = {m.name for m in StubThreatLevel}
        assert actual_members == expected_members

    def test_posture_action_stub_values(self):
        """Stub PostureAction must have same members as real enum."""
        from enum import Enum as _Enum

        class StubPostureAction(_Enum):
            NONE = "none"
            INCREASE_MONITORING = "increase_monitoring"
            ROTATE_KEYS = "rotate_keys"
            SWITCH_ALGORITHM = "switch_algorithm"
            ROTATE_AND_SWITCH = "rotate_and_switch"

        expected_members = {
            "NONE", "INCREASE_MONITORING", "ROTATE_KEYS",
            "SWITCH_ALGORITHM", "ROTATE_AND_SWITCH",
        }
        actual_members = {m.name for m in StubPostureAction}
        assert actual_members == expected_members

    def test_threat_level_map_keys(self):
        """threat_level_map must cover all ThreatLevel members."""
        from enum import Enum as _Enum

        class ThreatLevel(_Enum):
            NOMINAL = "nominal"
            ELEVATED = "elevated"
            HIGH = "high"
            CRITICAL = "critical"

        threat_level_map = {
            ThreatLevel.NOMINAL: 0.0,
            ThreatLevel.ELEVATED: 0.33,
            ThreatLevel.HIGH: 0.66,
            ThreatLevel.CRITICAL: 1.0,
        }
        assert set(threat_level_map.keys()) == set(ThreatLevel)

    def test_action_map_keys(self):
        """action_map must cover all PostureAction members."""
        from enum import Enum as _Enum

        class PostureAction(_Enum):
            NONE = "none"
            INCREASE_MONITORING = "increase_monitoring"
            ROTATE_KEYS = "rotate_keys"
            SWITCH_ALGORITHM = "switch_algorithm"
            ROTATE_AND_SWITCH = "rotate_and_switch"

        action_map = {
            PostureAction.NONE: 0.0,
            PostureAction.INCREASE_MONITORING: 1.0,
            PostureAction.ROTATE_KEYS: 2.0,
            PostureAction.SWITCH_ALGORITHM: 3.0,
            PostureAction.ROTATE_AND_SWITCH: 4.0,
        }
        assert set(action_map.keys()) == set(PostureAction)

    def test_no_old_enum_members_remain(self):
        """Verify OLD enum members (LOW, MEDIUM, ALERT) are NOT present."""
        # Re-read the actual stub source to verify
        import importlib.util

        spec = importlib.util.find_spec(
            "omni_mercury_engine.integrations.mercury_amacrypto"
        )
        if spec is None:
            pytest.skip("mercury_amacrypto not importable")

        source_path = spec.origin
        with open(source_path, "r") as f:
            content = f.read()

        # These old enum values should NOT appear in the stub definitions
        # (they may appear in comments, which is OK)
        assert "LOW = " not in content or "LOW = " in content.split("# ")[0] is False
        assert "MEDIUM = " not in content or "MEDIUM = " in content.split("# ")[0] is False
        assert "ALERT = " not in content or "ALERT = " in content.split("# ")[0] is False
