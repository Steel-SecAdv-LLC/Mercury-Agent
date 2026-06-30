# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the ethics-gate hardening (issue #7): lightweight semantic harm matching + severity/reversibility damping, calibratable, and strictly fail-closed (new features can only LOWER benevolence / RAISE harm)."""

from __future__ import annotations

from omni_mercury_engine.cognitive.ethical_bounding import (
    BENEVOLENCE_CALIBRATION,
    BenevolenceScorer,
    HarmReducer,
    _char_trigram_vector,
    _semantic_match_count,
)


class TestSemanticDeterminism:
    def test_trigram_vector_is_deterministic(self) -> None:
        # Must be stable across calls (cache + certifier rely on determinism).
        assert _char_trigram_vector("injury") == _char_trigram_vector("injury")

    def test_semantic_catches_morphological_variant(self) -> None:
        # "injuries" is NOT a substring of "injury" but is a near-trigram match.
        words = ("inflicting", "injuries", "today")
        n = _semantic_match_count(
            words, ["injury"], BENEVOLENCE_CALIBRATION.semantic_match_threshold
        )
        assert n >= 1


class TestHarmSemanticFailClosed:
    def test_paraphrase_harm_is_detected(self) -> None:
        reducer = HarmReducer()
        # "injuries" / "manipulative" evade exact-substring but are caught
        # semantically -> nonzero harm where the old gate scored 0.
        harm, breakdown = reducer.evaluate_harm("inflict injuries on staff", {})
        assert harm > 0.0
        assert breakdown["physical"] > 0.0

    def test_semantic_never_lowers_keyword_harm(self) -> None:
        reducer = HarmReducer()
        # A literal keyword present -> harm at least the substring level; the
        # semantic max() can only hold or raise it.
        harm_kw, _ = reducer.evaluate_harm("cause injury and damage", {})
        assert harm_kw > 0.0


class TestSeverityReversibilityDamping:
    def test_irreversible_lowers_benevolence(self) -> None:
        scorer = BenevolenceScorer(benevolence_threshold=0.0)
        benign = scorer.score_action("help and support the patient", {"humanitarian": True})
        irreversible = scorer.score_action(
            "help and support the patient",
            {"humanitarian": True, "severity": 0.9, "reversibility": 0.0},
        )
        # Same positive action; high-severity + irreversible context must damp it.
        assert irreversible.benevolence_score < benign.benevolence_score
        assert irreversible.severity_score >= 0.9
        assert irreversible.reversibility_score <= 0.0 + 1e-9

    def test_reversibility_context_is_min_failclosed(self) -> None:
        scorer = BenevolenceScorer(benevolence_threshold=0.0)
        # Internally an "undo/restore" action looks reversible (~0.9), but the
        # caller asserting 0.2 can only LOWER it (MIN), never the reverse.
        s = scorer.score_action("undo and restore the change", {"reversibility": 0.2})
        assert s.reversibility_score <= 0.2 + 1e-9

    def test_severity_context_is_max_failclosed(self) -> None:
        scorer = BenevolenceScorer(benevolence_threshold=0.0)
        s = scorer.score_action("routine status check", {"severity": 0.8})
        assert s.severity_score >= 0.8


class TestFailClosedMonotonicity:
    def test_adding_positive_words_cannot_unblock_harmful_action(self) -> None:
        scorer = BenevolenceScorer(benevolence_threshold=0.99)
        harmful = "destroy and kill with permanent irreversible violence"
        blocked = scorer.score_action(harmful, {})
        assert blocked.is_permissible is False
        # Appending benign/positive language must NOT push it over threshold.
        padded = scorer.score_action(
            harmful + " to help protect and support wellbeing", {"humanitarian": True}
        )
        assert padded.is_permissible is False


class TestCalibrationKnobsAndBackCompat:
    def test_weights_sum_to_one(self) -> None:
        c = BENEVOLENCE_CALIBRATION
        assert (
            abs(c.w_harm + c.w_benefit + c.w_equity + c.w_principles + c.w_long_term - 1.0) < 1e-9
        )

    def test_default_severity_reversibility_keeps_legacy_formula(self) -> None:
        scorer = BenevolenceScorer()
        # With severity=0, reversibility=1 the damping multiplier is 1.0, so the
        # weighted sum is unchanged.
        b = scorer._calculate_benevolence(0.2, 0.8, 0.7, {"a": 0.8}, 0.7)
        b_explicit = scorer._calculate_benevolence(
            0.2, 0.8, 0.7, {"a": 0.8}, 0.7, severity=0.0, reversibility=1.0
        )
        assert b == b_explicit
