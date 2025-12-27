"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

Tests for Ethical Bounding and Benevolence Scoring module.
"""

from __future__ import annotations

import pytest

from omni_anomaly_engine.cognitive.ethical_bounding import (
    AlignmentAudit,
    BenefitCategory,
    BenefitMaximizer,
    BenevolenceScorer,
    EmpathyAssessment,
    EmpathyModule,
    EquityCalculator,
    EthicalPrinciple,
    EthicalScore,
    HarmCategory,
    HarmReducer,
    ValuePreservation,
    ValuePreserver,
)


class TestHarmReducer:
    """Tests for HarmReducer class."""

    def test_init(self):
        """Test harm reducer initialization."""
        reducer = HarmReducer()
        assert reducer._evaluation_counter == 0

    def test_evaluate_harm_neutral(self):
        """Test harm evaluation for neutral action."""
        reducer = HarmReducer()
        harm_score, breakdown = reducer.evaluate_harm(
            action="check_status",
            context={"type": "monitoring"},
        )

        assert 0 <= harm_score <= 1
        assert len(breakdown) == len(HarmCategory)

    def test_evaluate_harm_harmful_action(self):
        """Test harm evaluation for potentially harmful action."""
        reducer = HarmReducer()
        harm_score, breakdown = reducer.evaluate_harm(
            action="expose_private_data",
            context={"potential_harm": True},
        )

        assert harm_score > 0.2
        assert breakdown["privacy"] > 0

    def test_evaluate_harm_physical(self):
        """Test physical harm detection."""
        reducer = HarmReducer()
        harm_score, breakdown = reducer.evaluate_harm(
            action="cause_injury",
            context={},
        )

        assert breakdown["physical"] > 0

    def test_evaluate_harm_psychological(self):
        """Test psychological harm detection."""
        reducer = HarmReducer()
        harm_score, breakdown = reducer.evaluate_harm(
            action="cause_stress_and_anxiety",
            context={},
        )

        assert breakdown["psychological"] > 0


class TestBenefitMaximizer:
    """Tests for BenefitMaximizer class."""

    def test_init(self):
        """Test benefit maximizer initialization."""
        maximizer = BenefitMaximizer()
        assert maximizer._evaluation_counter == 0

    def test_evaluate_benefit_neutral(self):
        """Test benefit evaluation for neutral action."""
        maximizer = BenefitMaximizer()
        benefit_score, breakdown = maximizer.evaluate_benefit(
            action="check_status",
            context={},
        )

        assert 0 <= benefit_score <= 1
        assert len(breakdown) == len(BenefitCategory)

    def test_evaluate_benefit_positive_action(self):
        """Test benefit evaluation for positive action."""
        maximizer = BenefitMaximizer()
        benefit_score, breakdown = maximizer.evaluate_benefit(
            action="protect_and_secure_users",
            context={"humanitarian": True},
        )

        assert benefit_score > 0.3
        assert breakdown["safety"] > 0

    def test_evaluate_benefit_humanitarian(self):
        """Test humanitarian benefit detection."""
        maximizer = BenefitMaximizer()
        benefit_score, breakdown = maximizer.evaluate_benefit(
            action="humanitarian_aid_relief",
            context={},
        )

        assert breakdown["humanitarian"] > 0

    def test_evaluate_benefit_knowledge(self):
        """Test knowledge benefit detection."""
        maximizer = BenefitMaximizer()
        benefit_score, breakdown = maximizer.evaluate_benefit(
            action="research_and_educate",
            context={},
        )

        assert breakdown["knowledge"] > 0


class TestEquityCalculator:
    """Tests for EquityCalculator class."""

    def test_init(self):
        """Test equity calculator initialization."""
        calculator = EquityCalculator()
        assert calculator is not None

    def test_calculate_gini_equal(self):
        """Test Gini coefficient for equal distribution."""
        calculator = EquityCalculator()
        gini = calculator.calculate_gini([1, 1, 1, 1, 1])

        assert gini == 0.0

    def test_calculate_gini_unequal(self):
        """Test Gini coefficient for unequal distribution."""
        calculator = EquityCalculator()
        gini = calculator.calculate_gini([0, 0, 0, 0, 100])

        assert gini > 0.5

    def test_calculate_gini_empty(self):
        """Test Gini coefficient for empty list."""
        calculator = EquityCalculator()
        gini = calculator.calculate_gini([])

        assert gini == 0.0

    def test_calculate_gini_single(self):
        """Test Gini coefficient for single value."""
        calculator = EquityCalculator()
        gini = calculator.calculate_gini([100])

        assert gini == 0.0

    def test_evaluate_equity_neutral(self):
        """Test equity evaluation for neutral action."""
        calculator = EquityCalculator()
        equity = calculator.evaluate_equity(
            action="process_data",
            context={},
        )

        assert 0 <= equity <= 1

    def test_evaluate_equity_positive(self):
        """Test equity evaluation for equitable action."""
        calculator = EquityCalculator()
        equity = calculator.evaluate_equity(
            action="ensure_fair_and_equal_access",
            context={},
        )

        assert equity > 0.7

    def test_evaluate_equity_negative(self):
        """Test equity evaluation for inequitable action."""
        calculator = EquityCalculator()
        equity = calculator.evaluate_equity(
            action="discriminate_and_exclude",
            context={},
        )

        assert equity < 0.7

    def test_evaluate_equity_with_distribution(self):
        """Test equity evaluation with distribution context."""
        calculator = EquityCalculator()
        equity = calculator.evaluate_equity(
            action="distribute_resources",
            context={"distribution": [0, 0, 0, 100]},
        )

        assert equity < 0.8


class TestEmpathyModule:
    """Tests for EmpathyModule class."""

    def test_init(self):
        """Test empathy module initialization."""
        module = EmpathyModule()
        assert module._assessment_counter == 0

    def test_assess_empathy_basic(self):
        """Test basic empathy assessment."""
        module = EmpathyModule()
        assessment = module.assess_empathy(
            action="help_users",
            context={},
        )

        assert isinstance(assessment, EmpathyAssessment)
        assert assessment.assessment_id.startswith("empathy_")
        assert 0 <= assessment.overall_empathy_score <= 1

    def test_assess_empathy_with_users(self):
        """Test empathy assessment with users context."""
        module = EmpathyModule()
        assessment = module.assess_empathy(
            action="assist",
            context={"users": ["user1", "user2"]},
        )

        assert "direct_users" in assessment.affected_parties

    def test_assess_empathy_with_vulnerabilities(self):
        """Test empathy assessment with vulnerability factors."""
        module = EmpathyModule()
        assessment = module.assess_empathy(
            action="process",
            context={
                "children_involved": True,
                "elderly_involved": True,
            },
        )

        assert len(assessment.vulnerability_factors) >= 2
        assert len(assessment.mitigation_suggestions) >= 2

    def test_assess_empathy_humanitarian(self):
        """Test empathy assessment for humanitarian action."""
        module = EmpathyModule()
        assessment = module.assess_empathy(
            action="humanitarian_relief",
            context={},
        )

        assert assessment.overall_empathy_score >= 0.5


class TestValuePreserver:
    """Tests for ValuePreserver class."""

    def test_init(self):
        """Test value preserver initialization."""
        preserver = ValuePreserver()
        assert preserver._preservation_counter == 0

    def test_analyze_preservation_safe(self):
        """Test preservation analysis for safe action."""
        preserver = ValuePreserver()
        preservation = preserver.analyze_preservation(
            action="monitor_system",
            context={},
        )

        assert isinstance(preservation, ValuePreservation)
        assert preservation.preservation_id.startswith("preserve_")
        assert preservation.preservation_score >= 0.8
        assert preservation.default_to_positive is True

    def test_analyze_preservation_risky(self):
        """Test preservation analysis for risky action."""
        preserver = ValuePreserver()
        preservation = preserver.analyze_preservation(
            action="expose_and_track_users",
            context={},
        )

        assert preservation.preservation_score < 1.0
        assert len(preservation.values_at_risk) > 0
        assert len(preservation.safeguards_needed) > 0

    def test_analyze_preservation_multiple_risks(self):
        """Test preservation analysis with multiple value risks."""
        preserver = ValuePreserver()
        preservation = preserver.analyze_preservation(
            action="force_coerce_and_deceive",
            context={},
        )

        assert len(preservation.values_at_risk) >= 2


class TestBenevolenceScorer:
    """Tests for BenevolenceScorer class."""

    def test_init(self):
        """Test benevolence scorer initialization."""
        scorer = BenevolenceScorer()
        assert scorer.benevolence_threshold == 0.99
        assert scorer._score_counter == 0

    def test_init_custom_threshold(self):
        """Test benevolence scorer with custom threshold."""
        scorer = BenevolenceScorer(benevolence_threshold=0.95)
        assert scorer.benevolence_threshold == 0.95

    def test_score_action_basic(self):
        """Test basic action scoring."""
        scorer = BenevolenceScorer()
        score = scorer.score_action(
            action="check_status",
            context={},
        )

        assert isinstance(score, EthicalScore)
        assert score.score_id.startswith("ethical_")
        assert 0 <= score.benevolence_score <= 1
        assert 0 <= score.harm_score <= 1
        assert 0 <= score.benefit_score <= 1

    def test_score_action_positive(self):
        """Test scoring for positive action."""
        scorer = BenevolenceScorer(benevolence_threshold=0.5)
        score = scorer.score_action(
            action="protect_and_help_users_safely",
            context={"humanitarian": True, "sustainable": True},
        )

        assert score.benevolence_score > 0.5
        assert score.benefit_score > 0

    def test_score_action_negative(self):
        """Test scoring for negative action."""
        scorer = BenevolenceScorer()
        score = scorer.score_action(
            action="harm_and_exploit_users",
            context={"potential_harm": True},
        )

        assert score.harm_score > 0
        assert score.is_permissible is False

    def test_score_action_principles(self):
        """Test principle scoring."""
        scorer = BenevolenceScorer()
        score = scorer.score_action(
            action="research_with_integrity",
            context={},
        )

        assert len(score.principle_scores) == len(EthicalPrinciple)
        for principle_score in score.principle_scores.values():
            assert 0 <= principle_score <= 1

    def test_score_action_explanation(self):
        """Test explanation generation."""
        scorer = BenevolenceScorer()
        score = scorer.score_action(
            action="test_action",
            context={},
        )

        assert len(score.explanation) > 0
        assert "test_action" in score.explanation

    def test_full_audit(self):
        """Test full alignment audit."""
        scorer = BenevolenceScorer()
        audit = scorer.full_audit(
            action="safe_monitoring",
            context={},
        )

        assert isinstance(audit, AlignmentAudit)
        assert audit.audit_id.startswith("audit_")
        assert audit.ethical_score is not None
        assert audit.empathy_assessment is not None
        assert audit.value_preservation is not None

    def test_full_audit_passed(self):
        """Test full audit that passes."""
        scorer = BenevolenceScorer(benevolence_threshold=0.3)
        audit = scorer.full_audit(
            action="protect_users_safely",
            context={"humanitarian": True},
        )

        assert audit.passed is True
        assert len(audit.failure_reasons) == 0

    def test_full_audit_failed(self):
        """Test full audit that fails."""
        scorer = BenevolenceScorer(benevolence_threshold=0.99)
        audit = scorer.full_audit(
            action="harm_and_exploit",
            context={"potential_harm": True},
        )

        assert audit.passed is False
        assert len(audit.failure_reasons) > 0

    def test_is_action_permissible(self):
        """Test quick permissibility check."""
        scorer = BenevolenceScorer(benevolence_threshold=0.5)
        is_permissible, score, explanation = scorer.is_action_permissible(
            action="safe_action",
            context={},
        )

        assert isinstance(is_permissible, bool)
        assert 0 <= score <= 1
        assert len(explanation) > 0

    def test_get_statistics(self):
        """Test statistics retrieval."""
        scorer = BenevolenceScorer()
        scorer.score_action("action1", {})
        scorer.full_audit("action2", {})

        stats = scorer.get_statistics()

        assert stats["scores_generated"] >= 2
        assert stats["audits_performed"] >= 1
        assert "benevolence_threshold" in stats

    def test_get_audit_history(self):
        """Test audit history retrieval."""
        scorer = BenevolenceScorer()
        scorer.full_audit("action1", {})
        scorer.full_audit("action2", {})

        history = scorer.get_audit_history(limit=10)

        assert len(history) >= 2


class TestEthicalPrinciple:
    """Tests for EthicalPrinciple enum."""

    def test_all_principles_exist(self):
        """Test all ethical principles exist."""
        assert EthicalPrinciple.COMPASSION.value == "compassion"
        assert EthicalPrinciple.EVIDENCE.value == "evidence"
        assert EthicalPrinciple.JUSTICE.value == "justice"
        assert EthicalPrinciple.ALTRUISM.value == "altruism"
        assert EthicalPrinciple.CONTROL.value == "control"
        assert EthicalPrinciple.CHARACTER.value == "character"
        assert EthicalPrinciple.COMPETENCE.value == "competence"
        assert EthicalPrinciple.COMMITMENT.value == "commitment"


class TestHarmCategory:
    """Tests for HarmCategory enum."""

    def test_all_categories_exist(self):
        """Test all harm categories exist."""
        assert HarmCategory.PHYSICAL.value == "physical"
        assert HarmCategory.PSYCHOLOGICAL.value == "psychological"
        assert HarmCategory.FINANCIAL.value == "financial"
        assert HarmCategory.PRIVACY.value == "privacy"
        assert HarmCategory.AUTONOMY.value == "autonomy"
        assert HarmCategory.DIGNITY.value == "dignity"
        assert HarmCategory.ENVIRONMENTAL.value == "environmental"
        assert HarmCategory.SOCIETAL.value == "societal"


class TestBenefitCategory:
    """Tests for BenefitCategory enum."""

    def test_all_categories_exist(self):
        """Test all benefit categories exist."""
        assert BenefitCategory.SAFETY.value == "safety"
        assert BenefitCategory.WELLBEING.value == "wellbeing"
        assert BenefitCategory.KNOWLEDGE.value == "knowledge"
        assert BenefitCategory.EFFICIENCY.value == "efficiency"
        assert BenefitCategory.EQUITY.value == "equity"
        assert BenefitCategory.SUSTAINABILITY.value == "sustainability"
        assert BenefitCategory.EMPOWERMENT.value == "empowerment"
        assert BenefitCategory.HUMANITARIAN.value == "humanitarian"


class TestDataclasses:
    """Tests for dataclasses."""

    def test_ethical_score(self):
        """Test EthicalScore dataclass."""
        score = EthicalScore(
            score_id="test_001",
            action="test_action",
            benevolence_score=0.95,
            harm_score=0.1,
            benefit_score=0.8,
            equity_score=0.9,
            long_term_score=0.85,
            is_permissible=True,
            principle_scores={"compassion": 0.9},
            harm_breakdown={"physical": 0.1},
            benefit_breakdown={"safety": 0.8},
            explanation="Test explanation",
            recommendations=["Recommendation 1"],
        )

        assert score.score_id == "test_001"
        assert score.benevolence_score == 0.95
        assert score.is_permissible is True

    def test_empathy_assessment(self):
        """Test EmpathyAssessment dataclass."""
        assessment = EmpathyAssessment(
            assessment_id="empathy_001",
            affected_parties=["users"],
            impact_scores={"users": 0.8},
            vulnerability_factors=["children_at_risk"],
            mitigation_suggestions=["Add safeguards"],
            overall_empathy_score=0.75,
        )

        assert assessment.assessment_id == "empathy_001"
        assert assessment.overall_empathy_score == 0.75

    def test_value_preservation(self):
        """Test ValuePreservation dataclass."""
        preservation = ValuePreservation(
            preservation_id="preserve_001",
            values_at_risk=["privacy"],
            preservation_score=0.8,
            default_to_positive=True,
            safeguards_needed=["Encrypt data"],
        )

        assert preservation.preservation_id == "preserve_001"
        assert preservation.default_to_positive is True

    def test_alignment_audit(self):
        """Test AlignmentAudit dataclass."""
        ethical_score = EthicalScore(
            score_id="test_001",
            action="test",
            benevolence_score=0.95,
            harm_score=0.1,
            benefit_score=0.8,
            equity_score=0.9,
            long_term_score=0.85,
            is_permissible=True,
            principle_scores={},
            harm_breakdown={},
            benefit_breakdown={},
            explanation="Test",
            recommendations=[],
        )

        audit = AlignmentAudit(
            audit_id="audit_001",
            action="test_action",
            ethical_score=ethical_score,
            empathy_assessment=None,
            value_preservation=None,
            passed=True,
            failure_reasons=[],
        )

        assert audit.audit_id == "audit_001"
        assert audit.passed is True


class TestIntegration:
    """Integration tests for ethical bounding."""

    def test_full_ethical_evaluation_pipeline(self):
        """Test complete ethical evaluation pipeline."""
        scorer = BenevolenceScorer(benevolence_threshold=0.5)

        audit = scorer.full_audit(
            action="protect_and_secure_user_data",
            context={
                "users": ["user1", "user2"],
                "humanitarian": True,
                "sustainable": True,
            },
        )

        assert audit.ethical_score.benevolence_score > 0.3
        assert audit.empathy_assessment.overall_empathy_score > 0.5
        assert audit.value_preservation.preservation_score > 0.5

    def test_harmful_action_blocked(self):
        """Test that harmful actions are blocked."""
        scorer = BenevolenceScorer(benevolence_threshold=0.99)

        is_permissible, score, explanation = scorer.is_action_permissible(
            action="exploit_and_harm_vulnerable_users",
            context={
                "potential_harm": True,
                "children_involved": True,
            },
        )

        assert is_permissible is False
        assert "BLOCKED" in explanation

    def test_benevolent_action_approved(self):
        """Test that benevolent actions are approved."""
        scorer = BenevolenceScorer(benevolence_threshold=0.3)

        is_permissible, score, explanation = scorer.is_action_permissible(
            action="humanitarian_aid_and_relief",
            context={
                "humanitarian": True,
                "sustainable": True,
            },
        )

        assert is_permissible is True
        assert "APPROVED" in explanation

    def test_equity_impact_on_score(self):
        """Test that equity impacts benevolence score."""
        scorer = BenevolenceScorer()

        score_equal = scorer.score_action(
            action="distribute_resources",
            context={"distribution": [25, 25, 25, 25]},
        )

        score_unequal = scorer.score_action(
            action="distribute_resources",
            context={"distribution": [0, 0, 0, 100]},
        )

        assert score_equal.equity_score > score_unequal.equity_score

    def test_multiple_audits_tracked(self):
        """Test that multiple audits are tracked."""
        scorer = BenevolenceScorer()

        for i in range(5):
            scorer.full_audit(f"action_{i}", {})

        stats = scorer.get_statistics()
        history = scorer.get_audit_history()

        assert stats["audits_performed"] == 5
        assert len(history) == 5
