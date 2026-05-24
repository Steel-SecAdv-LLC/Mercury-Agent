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

"""Physics verifier: dimensional + numeric consistency and oracle-grounded scalar."""

from typing import Any

import pytest

from omni_mercury_engine.core.global_omni_scalar_network import (
    GlobalOmniScalarNetwork,
    ScalarGroup,
    reset_global_network,
)
from omni_mercury_engine.verifiers import physics
from omni_mercury_engine.verifiers.physics import (
    PhysicsRelation,
    register_verified_scalar,
    verify_relation,
)


@pytest.fixture(autouse=True)
def _reset_gosnn() -> Any:
    reset_global_network()
    yield
    reset_global_network()


class TestVerifierHasTeeth:
    def test_mass_energy_is_confirmed(self) -> None:
        assert verify_relation(physics.mass_energy_equivalence()).valid

    def test_newtons_second_law_is_confirmed(self) -> None:
        assert verify_relation(physics.newtons_second_law()).valid

    def test_dimensional_mismatch_is_refuted(self) -> None:
        verdict = verify_relation(physics.dimensionally_wrong_mass_energy())
        assert not verdict.valid
        assert "dimensional mismatch" in verdict.reason

    def test_numeric_mismatch_is_refuted(self) -> None:
        rel = physics.mass_energy_equivalence()
        wrong = PhysicsRelation(
            name=rel.name, lhs=rel.lhs, rhs=rel.rhs, lhs_value=1.0, rhs_value=2.0
        )
        verdict = verify_relation(wrong)
        assert not verdict.valid
        assert "numeric mismatch" in verdict.reason


class TestScalarIsGroundedInTheVerdict:
    def test_confirmed_registers_one(self) -> None:
        gosnn = GlobalOmniScalarNetwork()
        value, verdict = register_verified_scalar(gosnn, physics.mass_energy_equivalence())
        assert verdict.valid and value == 1.0
        assert (
            gosnn.scalar_groups[ScalarGroup.PHYSICS_THEORIES][
                "omni_physics_mass_energy_equivalence_consistent"
            ]
            == 1.0
        )

    def test_refuted_registers_zero(self) -> None:
        gosnn = GlobalOmniScalarNetwork()
        value, verdict = register_verified_scalar(gosnn, physics.dimensionally_wrong_mass_energy())
        assert not verdict.valid and value == 0.0
        assert (
            gosnn.scalar_groups[ScalarGroup.PHYSICS_THEORIES][
                "omni_physics_wrong_mass_energy_consistent"
            ]
            == 0.0
        )
