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

"""Lean theorem verifier: the formal-proof tier, with honest behaviour when Lean is absent."""

from typing import Any

import pytest

from omni_mercury_engine.core.global_omni_scalar_network import (
    GlobalOmniScalarNetwork,
    ScalarGroup,
    reset_global_network,
)
from omni_mercury_engine.verifiers.lean_theorem import (
    FALSE_THEOREM,
    KNOWN_THEOREM,
    lean_available,
    register_verified_theorem,
    verify_lean_proof,
)


@pytest.fixture(autouse=True)
def _reset_gosnn() -> Any:
    reset_global_network()
    yield
    reset_global_network()


class TestNeverFakesAVerdict:
    """Runs regardless of Lean's presence -- the no-fake contract must always hold."""

    def test_validity_implies_availability(self) -> None:
        verdict = verify_lean_proof(KNOWN_THEOREM)
        # A valid verdict is impossible without a real oracle behind it.
        assert not (verdict.valid and not verdict.available)

    def test_unavailable_registers_no_scalar(self) -> None:
        if lean_available():
            pytest.skip("Lean is installed; the unavailable path is exercised only without it")
        gosnn = GlobalOmniScalarNetwork()
        value, verdict = register_verified_theorem(
            gosnn, KNOWN_THEOREM, theorem_name="two_plus_two"
        )
        assert verdict.available is False
        assert value is None
        assert gosnn.scalar_groups[ScalarGroup.MATHEMATICAL_MYSTERIES] == {}


@pytest.mark.skipif(not lean_available(), reason="requires a Lean 4 toolchain on PATH")
class TestLiveLeanKernel:
    """Runs only where Lean is installed -- the theorem path, end to end."""

    def test_known_theorem_is_accepted(self) -> None:
        verdict = verify_lean_proof(KNOWN_THEOREM)
        assert verdict.available and verdict.valid

    def test_false_theorem_is_rejected(self) -> None:
        verdict = verify_lean_proof(FALSE_THEOREM)
        assert verdict.available and not verdict.valid

    def test_accepted_theorem_grounds_scalar_to_one(self) -> None:
        gosnn = GlobalOmniScalarNetwork()
        value, verdict = register_verified_theorem(
            gosnn, KNOWN_THEOREM, theorem_name="two_plus_two"
        )
        assert verdict.valid
        assert value == 1.0
        assert (
            gosnn.scalar_groups[ScalarGroup.MATHEMATICAL_MYSTERIES][
                "omni_mystery_theorem_two_plus_two_verified"
            ]
            == 1.0
        )
