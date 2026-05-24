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

"""
Propositional consistency oracle shared by the paradox verifiers.

Propositional logic is decidable, so the consistency of a finite theory is a question an oracle
can settle outright.  The solver here is DPLL with unit propagation and pure-literal elimination
-- a real decision procedure that prunes the search, not an exponential truth-table sweep.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Literal:
    """A propositional literal: a variable name with a polarity."""

    name: str
    positive: bool = True

    def __invert__(self) -> Literal:
        return Literal(self.name, not self.positive)

    def __str__(self) -> str:
        return self.name if self.positive else f"~{self.name}"


Clause = frozenset[Literal]
CNF = tuple[Clause, ...]


def var(name: str) -> Literal:
    """A positive literal for ``name``."""
    return Literal(name, True)


def clause(*literals: Literal) -> Clause:
    """A disjunctive clause from the given literals."""
    return frozenset(literals)


def iff(a: Literal, b: Literal) -> CNF:
    """CNF for the biconditional ``a <-> b`` == (~a v b) and (~b v a)."""
    return (clause(~a, b), clause(~b, a))


def implies(a: Literal, b: Literal) -> CNF:
    """CNF for ``a -> b`` == (~a v b)."""
    return (clause(~a, b),)


def _simplify(clauses: tuple[Clause, ...], chosen: Literal) -> tuple[Clause, ...] | None:
    """Assign ``chosen`` true: drop satisfied clauses, remove its negation from the rest.

    Returns the reduced clause set, or ``None`` if an empty clause (conflict) results.
    """
    reduced: list[Clause] = []
    negation = ~chosen
    for c in clauses:
        if chosen in c:
            continue  # clause satisfied
        if negation in c:
            shrunk = c - {negation}
            if not shrunk:
                return None  # empty clause -> conflict
            reduced.append(shrunk)
        else:
            reduced.append(c)
    return tuple(reduced)


def _unit_and_pure(
    clauses: tuple[Clause, ...], assignment: dict[str, bool]
) -> tuple[tuple[Clause, ...] | None, dict[str, bool]]:
    """Apply unit propagation and pure-literal elimination to a fixpoint."""
    changed = True
    while changed and clauses:
        changed = False
        unit = next((next(iter(c)) for c in clauses if len(c) == 1), None)
        if unit is not None:
            assignment[unit.name] = unit.positive
            simplified = _simplify(clauses, unit)
            if simplified is None:
                return None, assignment
            clauses, changed = simplified, True
            continue
        literals = {lit for c in clauses for lit in c}
        pure = next(
            (lit for lit in literals if ~lit not in literals and lit.name not in assignment),
            None,
        )
        if pure is not None:
            assignment[pure.name] = pure.positive
            simplified = _simplify(clauses, pure)
            if simplified is None:
                return None, assignment
            clauses, changed = simplified, True
    return clauses, assignment


def solve(cnf: CNF, assignment: dict[str, bool] | None = None) -> dict[str, bool] | None:
    """Return a satisfying assignment for ``cnf`` (DPLL), or ``None`` if unsatisfiable."""
    assignment = dict(assignment or {})
    clauses, assignment = _unit_and_pure(cnf, assignment)
    if clauses is None:
        return None
    if not clauses:
        return assignment
    branch = next(iter(next(iter(clauses)))).name
    for value in (True, False):
        chosen = Literal(branch, value)
        simplified = _simplify(clauses, chosen)
        if simplified is None:
            continue
        extended = dict(assignment)
        extended[branch] = value
        result = solve(simplified, extended)
        if result is not None:
            return result
    return None


def is_satisfiable(cnf: CNF) -> bool:
    """Whether ``cnf`` has a satisfying assignment (i.e. the theory is consistent)."""
    return solve(cnf) is not None
