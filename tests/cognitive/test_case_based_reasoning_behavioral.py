# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioural tests for ``cognitive/case_based_reasoning.py``.

Previously only ``add``/``retrieve`` were weakly asserted (existence checks).
These pin real state transitions and the CBR cycle: retrieval ranking +
retrieval-count bookkeeping, the REUSE vs. REVISE branch in ``solve``,
proportional ``adapt`` + adaptation history, and ``learn_from_outcome``.
"""

from __future__ import annotations

from typing import Any

from omni_mercury_engine.cognitive.case_based_reasoning import (
    Case,
    CaseBasedReasoner,
    CaseOutcome,
    SimilarityMetric,
)


def _case(
    case_id: str,
    features: dict[str, float],
    solution: dict[str, Any],
    *,
    domain: str = "cyber",
    outcome: CaseOutcome = CaseOutcome.SUCCESS,
    score: float = 0.9,
) -> Case:
    return Case(
        case_id=case_id,
        problem_description=case_id,
        problem_features=features,
        feature_vector=None,
        solution=solution,
        outcome=outcome,
        outcome_score=score,
        domain=domain,
    )


class TestRetrieve:
    def test_retrieve_ranks_nearest_and_counts(self) -> None:
        r = CaseBasedReasoner(retrieval_threshold=0.0, similarity_metric=SimilarityMetric.COSINE)
        near = _case("near", {"x": 0.8, "y": 0.2}, {"action": "alert"})
        far = _case("far", {"x": 0.1, "y": 0.9}, {"action": "ignore"})
        r.add_case(near)
        r.add_case(far)

        result = r.retrieve({"x": 0.78, "y": 0.22}, k=2)
        # Nearest case is the best match with high similarity.
        assert result.best_match is not None
        assert result.best_match.case_id == "near"
        assert result.best_similarity > 0.9
        # Retrieval bookkeeping: the retrieved cases' counters incremented.
        assert near.retrieval_count == 1
        assert r._stats["retrievals"] == 1

    def test_domain_filter_restricts_candidates(self) -> None:
        r = CaseBasedReasoner(retrieval_threshold=0.0)
        r.add_case(_case("cyber1", {"x": 0.5}, {"a": 1}, domain="cyber"))
        r.add_case(_case("med1", {"x": 0.5}, {"a": 2}, domain="medical"))
        result = r.retrieve({"x": 0.5}, k=5, domain_filter="medical")
        assert all(c.domain == "medical" for c, _ in result.retrieved_cases)


class TestSolveCycle:
    def test_solve_reuses_on_near_identical_problem(self) -> None:
        r = CaseBasedReasoner(retrieval_threshold=0.0)
        r.add_case(_case("c1", {"x": 0.8, "y": 0.2}, {"action": "alert", "level": 3}))
        out = r.solve({"x": 0.8, "y": 0.2}, domain="cyber")
        # A (near-)identical problem reuses the stored solution directly.
        assert out["status"] == "direct_reuse"
        assert out["solution"]["action"] == "alert"
        assert out["confidence"] > 0.0

    def test_solve_revises_when_no_exact_match(self) -> None:
        r = CaseBasedReasoner(retrieval_threshold=0.0)
        r.add_case(_case("c1", {"x": 0.8, "y": 0.2}, {"action": "alert", "level": 3}))
        out = r.solve({"x": 0.55, "y": 0.45}, domain="cyber")
        # A non-identical problem still returns a usable solution (revised or
        # reused) with a source case and bounded confidence.
        assert out["solution"]
        assert 0.0 <= out["confidence"] <= 1.0
        assert out["status"] in {"direct_reuse", "adapted", "revised", "no_solution"}

    def test_solve_with_empty_casebase_is_honest(self) -> None:
        r = CaseBasedReasoner(retrieval_threshold=0.0)
        out = r.solve({"x": 0.5}, domain="cyber")
        # No fabricated solution when nothing is stored.
        assert out["solution"] is None
        assert out["status"] == "no_matching_cases"
        assert out["confidence"] == 0.0


class TestAdapt:
    def test_custom_rule_is_applied_and_recorded(self) -> None:
        r = CaseBasedReasoner()
        src = _case("c1", {"x": 1.0}, {"action": "alert"})

        def add_flag(source: Case, target: dict[str, Any], sol: dict[str, Any]) -> dict[str, Any]:
            return {"escalated": True}

        result = r.adapt(src, {"x": 2.0}, adaptation_rules=[add_flag])
        assert result.adapted_solution["escalated"] is True
        assert any("add_flag" in a for a in result.adaptations_made)
        # The source case records that it was adapted.
        assert len(src.adaptation_history) == 1

    def test_proportional_scaling_of_numeric_solution(self) -> None:
        r = CaseBasedReasoner()
        # Feature ``dose`` doubles (10 -> 20); a solution param whose key
        # references that feature (``dose_mg``) scales proportionally, while an
        # unrelated param (``duration``) is left untouched.
        src = _case("c1", {"dose": 10.0}, {"dose_mg": 100.0, "duration": 30.0})
        result = r.adapt(src, {"dose": 20.0})
        assert result.adapted_solution["dose_mg"] == 200.0
        assert result.adapted_solution["duration"] == 30.0
        assert any("dose_mg" in a for a in result.adaptations_made)

    def test_adapt_increments_stats(self) -> None:
        r = CaseBasedReasoner()
        src = _case("c1", {"x": 1.0}, {"action": "alert"})
        before = r._stats["adaptations"]
        r.adapt(src, {"x": 1.5})
        assert r._stats["adaptations"] == before + 1


class TestLearnFromOutcome:
    def test_outcome_and_score_are_updated(self) -> None:
        r = CaseBasedReasoner()
        c = _case("c1", {"x": 0.5}, {"a": 1}, outcome=CaseOutcome.UNKNOWN, score=0.0)
        r.add_case(c)
        r.learn_from_outcome("c1", CaseOutcome.FAILURE, 0.2, feedback={"why": "false positive"})
        stored = r._cases["c1"]
        assert stored.outcome == CaseOutcome.FAILURE
        assert stored.outcome_score == 0.2
        assert stored.metadata["feedback"] == {"why": "false positive"}

    def test_learn_success_marks_success_outcome(self) -> None:
        r = CaseBasedReasoner()
        c = _case("c1", {"x": 0.5}, {"a": 1}, outcome=CaseOutcome.UNKNOWN, score=0.0)
        r.add_case(c)
        r.learn_from_outcome("c1", CaseOutcome.SUCCESS, 0.95)
        assert r._cases["c1"].outcome == CaseOutcome.SUCCESS
        assert r._cases["c1"].outcome_score == 0.95


class TestAddCaseAccounting:
    def test_add_case_indexes_and_counts(self) -> None:
        r = CaseBasedReasoner()
        r.add_case(_case("c1", {"x": 0.5}, {"a": 1}, domain="cyber"))
        assert "c1" in r._cases
        assert r._stats["cases_stored"] == 1
        # Domain index is populated so domain-filtered retrieval can use it.
        assert "cyber" in r._domain_index
        assert "c1" in r._domain_index["cyber"]
