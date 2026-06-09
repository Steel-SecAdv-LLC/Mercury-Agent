# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unified MysteryRegistry: routing, provenance ledger, and σ_Immutable band protection."""

from __future__ import annotations

from typing import Any

import pytest

from omni_mercury_engine.core.global_omni_scalar_network import (
    GlobalOmniScalarNetwork,
    ScalarGroup,
    reset_global_network,
)
from omni_mercury_engine.verifiers import physics, registry
from omni_mercury_engine.verifiers.registry import MysteryRegistry


@pytest.fixture(autouse=True)
def _reset_gosnn() -> Any:
    reset_global_network()
    yield
    reset_global_network()


def _registry() -> MysteryRegistry:
    return MysteryRegistry(GlobalOmniScalarNetwork())


class TestRoutingAndStatuses:
    def test_each_tier_routes_to_the_right_verdict(self) -> None:
        reg = _registry()
        assert reg.submit_goldbach(100).status == "confirmed"
        assert reg.submit_twin_prime(11).status == "confirmed"
        assert reg.submit_twin_prime(7).status == "refuted"
        assert reg.submit_collatz(27).status == "confirmed"
        assert reg.submit_physics(physics.mass_energy_equivalence()).status == "confirmed"
        assert reg.submit_physics(physics.dimensionally_wrong_mass_energy()).status == "refuted"

    def test_inconclusive_collatz_registers_no_scalar(self) -> None:
        reg = _registry()
        entry = reg.submit_collatz(27, max_steps=3)
        assert entry.status == "inconclusive"
        assert entry.registered is False
        assert entry.value is None


class TestProvenanceLedger:
    def test_ledger_records_every_claim(self) -> None:
        reg = _registry()
        reg.submit_goldbach(100)
        reg.submit_twin_prime(7)
        reg.submit_collatz(27, max_steps=3)
        assert len(reg.ledger) == 3
        summary = reg.summary()
        assert summary["total_claims"] == 3
        assert isinstance(summary["by_status"], dict)


class TestBoundedScalarFootprint:
    def test_repeated_claims_share_one_scalar_key(self) -> None:
        reg = _registry()
        before = reg.summary()["operational_scalar_count"]
        for n in (100, 102, 104, 106):
            reg.submit_goldbach(n)
        after = reg.summary()["operational_scalar_count"]
        assert isinstance(before, int) and isinstance(after, int)
        assert after - before == 1  # one stable key regardless of how many claims

    def test_band_guard_refuses_to_overflow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reg = _registry()
        current = reg.summary()["operational_scalar_count"]
        assert isinstance(current, int)
        # Force the cap to the current count: a brand-new scalar key must not be registered.
        monkeypatch.setattr(registry, "SAFE_OPERATIONAL_CAP", current)
        entry = reg.submit_physics(physics.mass_energy_equivalence())
        assert entry.status == "confirmed"
        assert entry.registered is False
        assert "band budget reached" in entry.reason


class TestHonestyContract:
    def test_theorem_unavailable_or_decided_but_never_faked(self) -> None:
        reg = _registry()
        entry = reg.submit_theorem("theorem t : 2 + 2 = 4 := rfl\n", name="two_plus_two")
        assert entry.status in {"confirmed", "unavailable"}
        if entry.status == "unavailable":
            assert entry.registered is False
            assert (
                "omni_mystery_theorem_two_plus_two_verified"
                not in reg.gosnn.scalar_groups[ScalarGroup.MATHEMATICAL_MYSTERIES]
            )
