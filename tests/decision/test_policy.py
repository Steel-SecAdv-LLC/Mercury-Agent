# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""DecisionPolicy validates its knobs and composes cleanly."""

from __future__ import annotations

import pytest

from omni_mercury_engine.decision import DecisionPolicy


class TestValidation:
    @pytest.mark.parametrize("bad", [-0.01, 0.5, 0.9])
    def test_indecision_margin_out_of_range_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="indecision_margin"):
            DecisionPolicy(indecision_margin=bad)

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_symbolic_floor_out_of_range_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="symbolic_agreement_floor"):
            DecisionPolicy(symbolic_agreement_floor=bad)

    def test_defaults_are_valid_and_conservative(self) -> None:
        p = DecisionPolicy()
        assert p.fail_closed_on_atypical is True
        assert p.fail_closed_on_ethical_block is True
        assert p.require_calibrated_for_act is False


class TestComposition:
    def test_with_overrides_returns_validated_copy(self) -> None:
        base = DecisionPolicy()
        tuned = base.with_overrides(indecision_margin=0.1, require_calibrated_for_act=True)
        assert tuned.indecision_margin == 0.1
        assert tuned.require_calibrated_for_act is True
        # Original is untouched (frozen dataclass).
        assert base.indecision_margin == 0.05
        # Overrides are validated too.
        with pytest.raises(ValueError, match="indecision_margin"):
            base.with_overrides(indecision_margin=1.0)

    def test_drift_severities_are_case_insensitive(self) -> None:
        p = DecisionPolicy(drift_defer_severities=frozenset({"high"}))
        assert p.drift_is_deferring("HIGH") is True
        assert p.drift_is_deferring("high") is True
        assert p.drift_is_deferring("low") is False

    def test_drift_disabled_never_defers(self) -> None:
        p = DecisionPolicy(defer_on_drift=False)
        assert p.drift_is_deferring("CRITICAL") is False

    def test_drift_none_severity_never_defers(self) -> None:
        assert DecisionPolicy().drift_is_deferring(None) is False

    def test_to_dict_is_json_safe(self) -> None:
        import json

        d = DecisionPolicy().to_dict()
        json.dumps(d)  # must not raise
        assert d["fail_closed_on_ethical_block"] is True
        assert isinstance(d["drift_defer_severities"], list)
