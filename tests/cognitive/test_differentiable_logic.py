# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test differentiable logic."""

from __future__ import annotations

from omni_mercury_engine.cognitive.differentiable_logic import (
    DifferentiableLogicEngine,
    GodelTNorm,
    LukasiewiczTNorm,
    ProductTNorm,
)


class TestProductTNorm:
    def test_conjunction_boundary_zero(self) -> None:
        t = ProductTNorm()
        assert t.conjunction(0.0, 0.5) == 0.0

    def test_conjunction_boundary_one(self) -> None:
        t = ProductTNorm()
        assert t.conjunction(1.0, 1.0) == 1.0

    def test_conjunction_product(self) -> None:
        t = ProductTNorm()
        result = t.conjunction(0.5, 0.8)
        assert abs(result - 0.4) < 1e-9

    def test_disjunction_boundary_zero(self) -> None:
        t = ProductTNorm()
        assert t.disjunction(0.0, 0.0) == 0.0

    def test_disjunction_boundary_one(self) -> None:
        t = ProductTNorm()
        assert t.disjunction(1.0, 0.0) == 1.0

    def test_disjunction_product(self) -> None:
        t = ProductTNorm()
        result = t.disjunction(0.5, 0.8)
        assert abs(result - 0.9) < 1e-9

    def test_negation_zero(self) -> None:
        t = ProductTNorm()
        assert t.negation(0.0) == 1.0

    def test_negation_one(self) -> None:
        t = ProductTNorm()
        assert t.negation(1.0) == 0.0


class TestLukasiewiczTNorm:
    def test_conjunction_boundary_zero(self) -> None:
        t = LukasiewiczTNorm()
        assert t.conjunction(0.0, 0.5) == 0.0

    def test_conjunction_boundary_one(self) -> None:
        t = LukasiewiczTNorm()
        assert t.conjunction(1.0, 1.0) == 1.0

    def test_disjunction_boundary_zero(self) -> None:
        t = LukasiewiczTNorm()
        assert t.disjunction(0.0, 0.0) == 0.0

    def test_disjunction_boundary_one(self) -> None:
        t = LukasiewiczTNorm()
        assert t.disjunction(1.0, 0.5) == 1.0

    def test_negation_zero(self) -> None:
        t = LukasiewiczTNorm()
        assert t.negation(0.0) == 1.0

    def test_negation_one(self) -> None:
        t = LukasiewiczTNorm()
        assert t.negation(1.0) == 0.0


class TestGodelTNorm:
    def test_conjunction_boundary_zero(self) -> None:
        t = GodelTNorm()
        assert t.conjunction(0.0, 0.5) == 0.0

    def test_conjunction_boundary_one(self) -> None:
        t = GodelTNorm()
        assert t.conjunction(1.0, 1.0) == 1.0

    def test_conjunction_min(self) -> None:
        t = GodelTNorm()
        assert t.conjunction(0.3, 0.7) == 0.3

    def test_disjunction_boundary_zero(self) -> None:
        t = GodelTNorm()
        assert t.disjunction(0.0, 0.0) == 0.0

    def test_disjunction_boundary_one(self) -> None:
        t = GodelTNorm()
        assert t.disjunction(1.0, 0.5) == 1.0

    def test_disjunction_max(self) -> None:
        t = GodelTNorm()
        assert t.disjunction(0.3, 0.7) == 0.7

    def test_negation_zero(self) -> None:
        t = GodelTNorm()
        assert t.negation(0.0) == 1.0

    def test_negation_one(self) -> None:
        t = GodelTNorm()
        assert t.negation(1.0) == 0.0


class TestDifferentiableLogicEngine:
    def test_instantiation(self) -> None:
        engine = DifferentiableLogicEngine()
        assert engine is not None

    def test_get_statistics_keys(self) -> None:
        engine = DifferentiableLogicEngine()
        stats = engine.get_statistics()
        assert "n_predicates" in stats
        assert "n_rules" in stats
        assert "n_facts" in stats
        assert "predicate_dim" in stats
        assert "max_proof_depth" in stats
        assert "t_norm" in stats

    def test_register_predicate_and_add_fact(self) -> None:
        engine = DifferentiableLogicEngine()
        engine.register_predicate("is_anomaly", arity=1)
        engine.add_fact("is_anomaly", arguments=("x",), truth_value=0.9)
        stats = engine.get_statistics()
        assert stats["n_predicates"] >= 1
        assert stats["n_facts"] >= 1
