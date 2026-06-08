# Copyright (C) 2025 Steel Security Advisors LLC
"""Paradox verifier: DPLL consistency adjudication and oracle-grounded scalar."""

from __future__ import annotations

from typing import Any

import pytest

from omni_mercury_engine.core.global_omni_scalar_network import (
    GlobalOmniScalarNetwork,
    ScalarGroup,
    reset_global_network,
)
from omni_mercury_engine.verifiers import paradox
from omni_mercury_engine.verifiers.paradox import (
    ParadoxDefenseCertificate,
    register_verified_scalar,
    verify_defense,
)
from omni_mercury_engine.verifiers.propositional import iff, var


@pytest.fixture(autouse=True)
def _reset_gosnn() -> Any:
    reset_global_network()
    yield
    reset_global_network()


class TestVerifierHasTeeth:
    def test_liar_defense_is_confirmed(self) -> None:
        assert verify_defense(paradox.liar_paradox()).valid

    def test_russell_defense_is_confirmed(self) -> None:
        assert verify_defense(paradox.russell_paradox()).valid

    def test_inconsistent_defense_is_refuted(self) -> None:
        cert = ParadoxDefenseCertificate(
            name="bogus",
            naive=iff(var("X"), ~var("X")),
            defense=(frozenset({var("A")}), frozenset({~var("A")})),
        )
        verdict = verify_defense(cert)
        assert not verdict.valid
        assert "inconsistent" in verdict.reason

    def test_non_paradox_is_refuted(self) -> None:
        # A "paradox" whose naive framing is actually satisfiable is no paradox at all.
        cert = ParadoxDefenseCertificate(
            name="not_a_paradox",
            naive=(frozenset({var("P")}),),
            defense=(frozenset({var("Q")}),),
        )
        verdict = verify_defense(cert)
        assert not verdict.valid
        assert "no genuine contradiction" in verdict.reason


class TestScalarIsGroundedInTheVerdict:
    def test_confirmed_registers_one(self) -> None:
        gosnn = GlobalOmniScalarNetwork()
        value, verdict = register_verified_scalar(gosnn, paradox.liar_paradox())
        assert verdict.valid and value == 1.0
        assert gosnn.scalar_groups[ScalarGroup.PARADOX_DEFENSE]["omni_paradox_liar_defended"] == 1.0
