"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""The verifier vocabulary maps onto the unified three-state contract.

Pure-Python tier (no torch / no Lean required): the GOSNN constructs
without torch and the number-theory oracles are deterministic, so this
must-run tier validates the GROUNDED / UNAVAILABLE / UNDECIDABLE invariant
in every environment.
"""

import pytest

from omni_mercury_engine.core.global_omni_scalar_network import (
    GlobalOmniScalarNetwork,
    ScalarGroup,
    reset_global_network,
)
from omni_mercury_engine.verifiers import physics
from omni_mercury_engine.verifiers.lean_theorem import lean_available
from omni_mercury_engine.verifiers.registry import MysteryRegistry
from omni_mercury_engine.verifiers.three_state import (
    KNOWN_UNDECIDABLE_IN_GENERAL,
    ThreeState,
    three_state_of,
)


@pytest.fixture(autouse=True)
def _reset_gosnn():
    reset_global_network()
    yield
    reset_global_network()


@pytest.fixture
def registry() -> MysteryRegistry:
    return MysteryRegistry(GlobalOmniScalarNetwork())


class TestStatusMapping:
    """The literal reconciliation table is correct and total."""

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            ("confirmed", ThreeState.GROUNDED),
            ("refuted", ThreeState.GROUNDED),
            ("inconclusive", ThreeState.UNAVAILABLE),
            ("unavailable", ThreeState.UNAVAILABLE),
            ("undecidable", ThreeState.UNDECIDABLE),
        ],
    )
    def test_each_status_maps(self, status: str, expected: ThreeState) -> None:
        assert three_state_of(status) is expected

    def test_unknown_status_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown verifier status"):
            three_state_of("maybe")

    def test_wire_values_are_stable(self) -> None:
        # The string values are the cross-repo wire format; the companion
        # repo mirrors these exactly.  Pin them so a rename is deliberate.
        assert ThreeState.GROUNDED.value == "grounded"
        assert ThreeState.UNAVAILABLE.value == "unavailable"
        assert ThreeState.UNDECIDABLE.value == "undecidable"


class TestGroundedRegistersValue:
    """confirmed / refuted are both GROUNDED and carry the value."""

    def test_confirmed_is_grounded_and_registers(self, registry: MysteryRegistry) -> None:
        entry = registry.submit_goldbach(100)  # 100 = 3 + 97
        assert entry.status == "confirmed"
        assert entry.state is ThreeState.GROUNDED
        assert entry.value == 1.0
        assert entry.registered is True

    def test_refuted_is_grounded(self, registry: MysteryRegistry) -> None:
        entry = registry.submit_physics(physics.dimensionally_wrong_mass_energy())
        assert entry.status == "refuted"
        assert entry.state is ThreeState.GROUNDED
        assert entry.value == 0.0


class TestUnavailableAbstains:
    """Decidable-but-not-produced-this-run -> UNAVAILABLE, registers nothing."""

    def test_collatz_budget_exhausted_is_unavailable(self, registry: MysteryRegistry) -> None:
        # A bounded search that exhausts its budget: n=27 reaches 1 in 111
        # steps, so max_steps=5 cannot reach it.  Decidable with a larger
        # budget -> UNAVAILABLE, not UNDECIDABLE.
        entry = registry.submit_collatz(27, max_steps=5)
        assert entry.status == "inconclusive"
        assert entry.state is ThreeState.UNAVAILABLE
        assert entry.registered is False
        assert entry.value is None

    def test_goldbach_no_partition_is_unavailable(self, registry: MysteryRegistry) -> None:
        # Odd input: the candidate generator proposes no partition; the
        # instance is still decidable in principle -> UNAVAILABLE.
        entry = registry.submit_goldbach(7)
        assert entry.status == "inconclusive"
        assert entry.state is ThreeState.UNAVAILABLE
        assert entry.registered is False

    def test_theorem_without_lean_is_unavailable(self, registry: MysteryRegistry) -> None:
        if lean_available():
            pytest.skip("Lean is installed; the no-toolchain UNAVAILABLE path needs it absent")
        entry = registry.submit_theorem("theorem t : 2 + 2 = 4 := rfl\n", name="two_plus_two")
        assert entry.status == "unavailable"
        assert entry.state is ThreeState.UNAVAILABLE
        assert entry.registered is False
        assert entry.value is None


class TestUndecidableRegistersNothingEver:
    """No decision procedure in principle -> UNDECIDABLE, registers nothing."""

    def test_known_undecidable_problem(self, registry: MysteryRegistry) -> None:
        gosnn = registry.gosnn
        before = dict(gosnn.scalar_groups[ScalarGroup.MATHEMATICAL_MYSTERIES])

        entry = registry.submit_undecidable("collatz_general")

        assert entry.status == "undecidable"
        assert entry.state is ThreeState.UNDECIDABLE
        assert entry.registered is False
        assert entry.value is None
        # Registers nothing, EVER: the GOSNN group is untouched.
        assert gosnn.scalar_groups[ScalarGroup.MATHEMATICAL_MYSTERIES] == before
        assert "collatz_general" in KNOWN_UNDECIDABLE_IN_GENERAL

    def test_undecidable_distinct_from_unavailable(self, registry: MysteryRegistry) -> None:
        # The crux of the split: a decidable instance that abstained this
        # run (UNAVAILABLE) and a problem with no decision procedure
        # (UNDECIDABLE) must NOT collapse to the same state.
        unavailable = registry.submit_collatz(27, max_steps=5).state
        undecidable = registry.submit_undecidable("twin_prime_infinitude").state
        assert unavailable is not undecidable
        assert unavailable is ThreeState.UNAVAILABLE
        assert undecidable is ThreeState.UNDECIDABLE


class TestSummarySurfacesThreeState:
    """The aggregate summary reports the three-state breakdown."""

    def test_by_state_counts(self, registry: MysteryRegistry) -> None:
        registry.submit_goldbach(100)  # GROUNDED
        registry.submit_collatz(27, max_steps=5)  # UNAVAILABLE
        registry.submit_undecidable("collatz_general")  # UNDECIDABLE
        summary = registry.summary()
        by_state = summary["by_state"]
        assert isinstance(by_state, dict)
        assert by_state[ThreeState.GROUNDED.value] == 1
        assert by_state[ThreeState.UNAVAILABLE.value] == 1
        assert by_state[ThreeState.UNDECIDABLE.value] == 1
