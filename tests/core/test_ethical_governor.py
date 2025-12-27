"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Tests for Ethical Governor module
"""

import numpy as np

from omni_anomaly_engine.core.ethical_config import DEFAULT_CONFIG
from omni_anomaly_engine.core.ethical_governor import (
    BiasMetrics,
    EthicalAutonomyGovernor,
    EthicalDecision,
    SigmaDirective,
)


class TestSigmaDirective:
    """Test Sigma Directive."""

    def test_initialization(self):
        """Test initialization."""
        directive = SigmaDirective(DEFAULT_CONFIG.ethical_scalars)

        assert directive.ethical_scalars is not None
        assert len(directive.directive_weights) == 4
        assert SigmaDirective.JUSTICE == "justice"
        assert SigmaDirective.ALTRUISM == "altruism"

    def test_apply_directive_approved(self):
        """Test directive approves ethical action."""
        directive = SigmaDirective(DEFAULT_CONFIG.ethical_scalars)
        context = {
            "fairness_score": 0.9,
            "societal_benefit": 0.9,
            "harm_prevention": 0.9,
            "suffering_mitigation": 0.9,
            "transparency": 0.9,
            "honesty": 0.9,
        }

        allow, reasoning = directive.apply_directive("test_action", context)

        assert allow is True
        assert "approved" in reasoning.lower()

    def test_apply_directive_rejected(self):
        """Test directive rejects unethical action."""
        directive = SigmaDirective(DEFAULT_CONFIG.ethical_scalars)
        context = {
            "fairness_score": 0.1,
            "bias_detected": True,
            "societal_benefit": 0.1,
            "potential_harm": 0.9,
        }

        allow, reasoning = directive.apply_directive("test_action", context)

        assert allow is False
        assert "Override" in reasoning or "violation" in reasoning.lower()

    def test_evaluate_justice(self):
        """Test justice evaluation."""
        directive = SigmaDirective(DEFAULT_CONFIG.ethical_scalars)

        score = directive._evaluate_justice({"fairness_score": 0.8})
        assert score == 0.8

        score = directive._evaluate_justice({"bias_detected": True})
        assert score == 0.0

    def test_evaluate_altruism(self):
        """Test altruism evaluation."""
        directive = SigmaDirective(DEFAULT_CONFIG.ethical_scalars)

        score = directive._evaluate_altruism({"societal_benefit": 0.8, "potential_harm": 0.2})
        assert abs(score - 0.6) < 0.01

    def test_evaluate_compassion(self):
        """Test compassion evaluation."""
        directive = SigmaDirective(DEFAULT_CONFIG.ethical_scalars)

        score = directive._evaluate_compassion(
            {"harm_prevention": 0.8, "suffering_mitigation": 0.6}
        )
        assert abs(score - 0.7) < 0.01

    def test_evaluate_truth(self):
        """Test truth evaluation."""
        directive = SigmaDirective(DEFAULT_CONFIG.ethical_scalars)

        score = directive._evaluate_truth({"transparency": 0.9, "honesty": 0.7})
        assert abs(score - 0.8) < 0.01

    def test_generate_override_reasoning(self):
        """Test reasoning generation."""
        directive = SigmaDirective(DEFAULT_CONFIG.ethical_scalars)

        reasoning = directive._generate_override_reasoning(0.5, 0.5, 0.5, 0.5)

        assert "justice" in reasoning.lower() or "Override" in reasoning


class TestEthicalAutonomyGovernor:
    """Test Ethical Autonomy Governor."""

    def test_initialization(self):
        """Test initialization."""
        governor = EthicalAutonomyGovernor()

        assert governor.enable_bias_audits is True
        assert governor.enable_sigma_directives is True
        assert governor.p_value_threshold == 0.05
        assert governor.ethical_threshold == 0.8
        assert governor.sigma_directive is not None

    def test_initialization_disabled(self):
        """Test initialization with features disabled."""
        governor = EthicalAutonomyGovernor(enable_bias_audits=False, enable_sigma_directives=False)

        assert governor.enable_bias_audits is False
        assert governor.sigma_directive is None

    def test_evaluate_decision_ethical(self):
        """Test evaluating ethical decision."""
        governor = EthicalAutonomyGovernor()
        context = {
            "fairness_score": 0.9,
            "societal_benefit": 0.9,
            "harm_prevention": 0.9,
            "transparency": 0.9,
            "honesty": 0.9,
        }

        decision = governor.evaluate_decision("ethical_action", context)

        assert isinstance(decision, EthicalDecision)
        assert decision.action == "ethical_action"

    def test_evaluate_decision_with_data(self):
        """Test evaluating decision with data for bias audit."""
        governor = EthicalAutonomyGovernor()
        data = np.random.randn(100)
        context = {}

        decision = governor.evaluate_decision("test_action", context, data)

        assert isinstance(decision, EthicalDecision)

    def test_compute_ethical_score(self):
        """Test ethical score computation."""
        governor = EthicalAutonomyGovernor()

        score = governor._compute_ethical_score(
            "action", {"harm": 0.1, "benefit": 0.9, "fairness": 0.8}
        )

        assert score > 0

    def test_compute_ethical_score_critical(self):
        """Test ethical score with critical context."""
        governor = EthicalAutonomyGovernor()

        base_score = governor._compute_ethical_score("action", {})
        critical_score = governor._compute_ethical_score("action", {"critical": True})

        assert critical_score > base_score

    def test_audit_bias(self):
        """Test bias auditing."""
        governor = EthicalAutonomyGovernor()
        data = np.random.randn(100)

        metrics = governor._audit_bias(data, {})

        assert isinstance(metrics, BiasMetrics)
        assert 0 <= metrics.statistical_parity <= 1

    def test_audit_bias_1d_data(self):
        """Test bias auditing with 1D data."""
        governor = EthicalAutonomyGovernor()
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        metrics = governor._audit_bias(data, {})

        assert isinstance(metrics, BiasMetrics)

    def test_statistical_validation(self):
        """Test statistical validation."""
        governor = EthicalAutonomyGovernor()

        p_value = governor._statistical_validation(1.0, {})

        assert 0 <= p_value <= 1

    def test_should_rollback_low_score(self):
        """Test rollback triggered by low ethical score."""
        governor = EthicalAutonomyGovernor(ethical_threshold=0.8)
        decision = EthicalDecision(
            decision_id="test",
            action="test",
            ethical_score=0.5,
            bias_audit_passed=True,
            p_value=0.5,
            sigma_directive_applied=False,
        )

        assert governor._should_rollback(decision) is True

    def test_should_rollback_bias_failed(self):
        """Test rollback triggered by bias audit failure."""
        governor = EthicalAutonomyGovernor()
        decision = EthicalDecision(
            decision_id="test",
            action="test",
            ethical_score=1.0,
            bias_audit_passed=False,
            p_value=0.5,
            sigma_directive_applied=False,
        )

        assert governor._should_rollback(decision) is True

    def test_should_rollback_sigma_directive(self):
        """Test rollback triggered by Sigma Directive."""
        governor = EthicalAutonomyGovernor()
        decision = EthicalDecision(
            decision_id="test",
            action="test",
            ethical_score=1.0,
            bias_audit_passed=True,
            p_value=0.5,
            sigma_directive_applied=True,
        )

        assert governor._should_rollback(decision) is True

    def test_get_governance_report_empty(self):
        """Test governance report with no decisions."""
        governor = EthicalAutonomyGovernor()

        report = governor.get_governance_report()

        assert report["total_decisions"] == 0
        assert report["total_rollbacks"] == 0
        assert "governance_status" in report

    def test_get_governance_report_with_decisions(self):
        """Test governance report with decisions."""
        governor = EthicalAutonomyGovernor()
        context = {
            "fairness_score": 0.9,
            "societal_benefit": 0.9,
            "harm_prevention": 0.9,
            "transparency": 0.9,
            "honesty": 0.9,
        }

        for _ in range(5):
            governor.evaluate_decision("test_action", context)

        report = governor.get_governance_report()

        assert report["total_decisions"] > 0
        assert "avg_ethical_score" in report
        assert "bias_audit_pass_rate" in report


class TestBiasMetrics:
    """Test BiasMetrics dataclass."""

    def test_creation(self):
        """Test creating BiasMetrics."""
        metrics = BiasMetrics(
            demographic_parity_diff=0.1,
            equalized_odds_diff=0.05,
            statistical_parity=0.9,
            bias_detected=False,
            mitigation_applied=False,
        )

        assert metrics.demographic_parity_diff == 0.1
        assert metrics.bias_detected is False


class TestEthicalDecision:
    """Test EthicalDecision dataclass."""

    def test_creation(self):
        """Test creating EthicalDecision."""
        decision = EthicalDecision(
            decision_id="test_001",
            action="approve_loan",
            ethical_score=0.95,
            bias_audit_passed=True,
            p_value=0.03,
            sigma_directive_applied=False,
        )

        assert decision.decision_id == "test_001"
        assert decision.action == "approve_loan"
        assert decision.ethical_score == 0.95
        assert decision.rollback_triggered is False
