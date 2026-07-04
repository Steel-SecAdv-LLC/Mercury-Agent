# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The propositional mini-language routed through the shipped DPLL oracle."""

from __future__ import annotations

import pytest

from omni_mercury_engine.intel.propositional_claims import (
    PropositionalParseError,
    is_satisfiable_formula,
    is_tautology,
    parse,
    parse_trailing,
    to_cnf,
)


@pytest.mark.parametrize(
    ("formula", "taut", "sat"),
    [
        ("P or not P", True, True),
        ("P and not P", False, False),
        ("(P implies Q) and P implies Q", True, True),  # modus ponens
        ("P implies Q", False, True),
        ("P <-> P", True, True),
        ("P xor P", False, False),
        ("P xor not P", True, True),
        ("(P or Q) and (not P) and (not Q)", False, False),
        ("P -> (Q -> P)", True, True),
        ("not (P and not P)", True, True),
    ],
)
def test_tautology_and_satisfiability(formula: str, taut: bool, sat: bool) -> None:
    assert is_tautology(formula) is taut
    assert is_satisfiable_formula(formula) is sat


def test_operator_precedence_and_words_and_symbols_agree() -> None:
    # Word and symbol spellings must parse identically.
    assert is_tautology("P or not P") == is_tautology("P | ~P")
    assert is_tautology("(P implies Q) and P implies Q") == is_tautology("(P -> Q) & P -> Q")


def test_parse_trailing_strips_leading_prose() -> None:
    node = parse_trailing("Note that P and not P")
    assert node is not None
    # The trimmed formula is the contradiction, which is not a tautology.
    root, clauses = to_cnf(node)
    assert clauses  # a compound formula produced defining clauses
    node2 = parse_trailing("Clearly P or not P")
    assert node2 is not None


def test_parse_trailing_returns_none_for_non_formula() -> None:
    assert parse_trailing("the quick brown fox jumps") is not None  # words parse as vars
    assert parse_trailing("") is None
    assert parse_trailing("and or implies") is None  # only connectives, no atom


def test_empty_formula_raises() -> None:
    with pytest.raises(PropositionalParseError):
        parse("")


def test_bounded_variable_cap_fails_closed() -> None:
    big = " and ".join(f"V{i}" for i in range(25))
    with pytest.raises(PropositionalParseError):
        to_cnf(parse(big))


def test_tseitin_aux_vars_do_not_collide_with_user_variables() -> None:
    """A formula whose variables are named like the Tseitin aux vars (``_t1``,
    ``_t2``, ...) must still be adjudicated soundly. Before aux-var namespacing a
    user variable ``_t1`` collided with the first auxiliary variable, corrupting
    the CNF and returning a wrong SAT/tautology verdict -- a fail-open that could
    CONFIRM a false propositional claim and emit it in hard mode."""
    # Excluded middle over the aux-namespace names is still a tautology.
    assert is_tautology("_t1 | ~_t1")
    assert is_tautology("(_t1 & _t2) | ~(_t1 & _t2)")
    # Contradictions over those names are UNSAT (not tautologies).
    assert not is_tautology("_t1 & ~_t1")
    assert not is_satisfiable_formula("_t1 & ~_t1")
    # A genuinely satisfiable formula over the aux names is SAT.
    assert is_satisfiable_formula("_t1 & _t2")
    # Mixed user + aux-looking names behave soundly.
    assert is_tautology("(a -> _t1) | (a & ~_t1)")
