# Copyright (C) 2025 Steel Security Advisors LLC
"""Direct tests for the shared dimensional-analysis and propositional (DPLL) oracles."""

from __future__ import annotations

from fractions import Fraction

from omni_mercury_engine.verifiers.dimensional import (
    ACCELERATION,
    ENERGY,
    FORCE,
    MASS,
    MOMENTUM,
    TIME,
    VELOCITY,
)
from omni_mercury_engine.verifiers.propositional import clause, iff, is_satisfiable, solve, var


class TestDimensionalAlgebra:
    def test_energy_decomposition(self) -> None:
        assert ENERGY == MASS * VELOCITY**2

    def test_force_decomposition(self) -> None:
        assert FORCE == MASS * ACCELERATION

    def test_energy_is_not_momentum(self) -> None:
        assert ENERGY != MOMENTUM

    def test_rational_root(self) -> None:
        assert (TIME**2) ** Fraction(1, 2) == TIME


class TestDpllSatSolver:
    def test_simple_satisfiable(self) -> None:
        a, b = var("a"), var("b")
        cnf = (clause(a, b), clause(~a, b))
        model = solve(cnf)
        assert model is not None
        assert model["b"] is True

    def test_contradiction_is_unsat(self) -> None:
        a = var("a")
        assert not is_satisfiable((clause(a), clause(~a)))

    def test_liar_biconditional_is_unsat(self) -> None:
        # L <-> ~L is the canonical self-referential contradiction.
        assert not is_satisfiable(iff(var("L"), ~var("L")))

    def test_three_variable_chain(self) -> None:
        a, b, c = var("a"), var("b"), var("c")
        cnf = (*iff(a, b), *iff(b, c), clause(a))
        model = solve(cnf)
        assert model is not None
        assert model["a"] and model["b"] and model["c"]
