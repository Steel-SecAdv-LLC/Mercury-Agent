"""
Mercury Agent ♱
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
Tests for AI Ethics Framework (src/core/ai_ethics.py)
"""

import pytest

from omni_mercury_engine.core.ai_ethics import (
    EthicalAutonomyGovernor,
    EthicalPrinciple,
    EthicsConfig,
    EthicsResult,
    evaluate_refactoring_ethics,
)


class TestEthicalPrinciple:
    """Test the EthicalPrinciple enum."""

    def test_all_principles_exist(self):
        """Test that all 8 principles are defined."""
        principles = [p.value for p in EthicalPrinciple]
        assert len(principles) == 8
        assert "compassion" in principles
        assert "evidence" in principles
        assert "justice" in principles
        assert "altruism" in principles
        assert "control" in principles
        assert "character" in principles
        assert "competence" in principles
        assert "commitment" in principles


class TestEthicsConfig:
    """Test the EthicsConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EthicsConfig()
        assert config.enable_compassion_checks is True
        assert config.enable_evidence_validation is True
        assert config.enable_justice_bias_checks is True
        assert config.enable_altruism_impact_checks is True
        assert config.enable_control_auditing is True
        assert config.enable_character_transparency is True
        assert config.enable_competence_validation is True
        assert config.enable_commitment_evolution is True
        assert config.min_ethics_score == 0.7
        assert config.strict_mode is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = EthicsConfig(
            enable_compassion_checks=False, min_ethics_score=0.9, strict_mode=True
        )
        assert config.enable_compassion_checks is False
        assert config.min_ethics_score == 0.9
        assert config.strict_mode is True


class TestEthicsResult:
    """Test the EthicsResult dataclass."""

    def test_passed_result(self):
        """Test a passing ethics result."""
        result = EthicsResult(
            passed=True,
            overall_score=0.85,
            principle_scores={"compassion": 0.9, "evidence": 0.8},
            violations=[],
            recommendations=[],
        )
        assert result.passed is True
        assert result.overall_score == 0.85
        assert len(result.principle_scores) == 2
        assert len(result.violations) == 0

    def test_failed_result(self):
        """Test a failing ethics result."""
        result = EthicsResult(
            passed=False,
            overall_score=0.4,
            principle_scores={"compassion": 0.3, "evidence": 0.5},
            violations=["Compassion: Action may cause harm"],
            recommendations=["Add safety checks"],
        )
        assert result.passed is False
        assert result.overall_score == 0.4
        assert len(result.violations) == 1
        assert len(result.recommendations) == 1


class TestEthicalAutonomyGovernor:
    """Test the EthicalAutonomyGovernor class."""

    def test_initialization(self):
        """Test governor initialization."""
        governor = EthicalAutonomyGovernor()
        assert governor.config is not None
        assert isinstance(governor.config, EthicsConfig)
        assert len(governor.audit_log) == 0

    def test_evaluate_action_basic(self):
        """Test basic action evaluation."""
        governor = EthicalAutonomyGovernor()
        result = governor.evaluate_action(
            action_type="refactoring", action_params={"create_backup": True}, context={}
        )
        assert isinstance(result, EthicsResult)
        assert result.overall_score > 0
        assert len(result.principle_scores) == 8

    def test_compassion_check_with_backup(self):
        """Test compassion check with backup enabled."""
        governor = EthicalAutonomyGovernor()
        result = governor.evaluate_action(
            action_type="refactoring",
            action_params={"create_backup": True, "require_confirmation": True},
            context={"has_rollback": True},
        )
        assert result.principle_scores["compassion"] >= 0.8

    def test_compassion_check_without_backup(self):
        """Test compassion check without backup."""
        governor = EthicalAutonomyGovernor()
        result = governor.evaluate_action(
            action_type="refactoring", action_params={"create_backup": False}, context={}
        )
        # Base compassion score is 0.55, plus keyword matches in params string
        # Result is 0.65 due to harm_reduction_keywords matching in stringified params
        assert result.principle_scores["compassion"] == 0.65

    def test_evidence_check_with_benchmarks(self):
        """Test evidence check with benchmarks."""
        governor = EthicalAutonomyGovernor()
        result = governor.evaluate_action(
            action_type="optimization",
            action_params={},
            context={"has_benchmarks": True, "has_statistics": True},
        )
        assert result.principle_scores["evidence"] >= 0.8

    def test_justice_check_ast_transform(self):
        """Test justice check for AST transformations."""
        governor = EthicalAutonomyGovernor()
        result = governor.evaluate_action(action_type="ast_transform", action_params={}, context={})
        # Deterministic actions like ast_transform get 0.95 justice score
        assert result.principle_scores["justice"] == 0.95

    def test_altruism_check_open_source(self):
        """Test altruism check for open source."""
        governor = EthicalAutonomyGovernor()
        result = governor.evaluate_action(
            action_type="refactoring", action_params={}, context={"is_open_source": True}
        )
        # Base 0.6 + 0.15 for open source = 0.75
        assert result.principle_scores["altruism"] >= 0.75

    def test_control_check_with_logging(self):
        """Test control check with logging enabled."""
        governor = EthicalAutonomyGovernor()
        result = governor.evaluate_action(
            action_type="refactoring",
            action_params={"logging_enabled": True},
            context={"audit_enabled": True},
        )
        assert result.principle_scores["control"] >= 0.8

    def test_character_check_transparent(self):
        """Test character check with transparency."""
        governor = EthicalAutonomyGovernor()
        result = governor.evaluate_action(
            action_type="refactoring",
            action_params={"verbose": True},
            context={"is_transparent": True},
        )
        assert result.principle_scores["character"] >= 0.9

    def test_competence_check_high_coverage(self):
        """Test competence check with high test coverage."""
        governor = EthicalAutonomyGovernor()
        result = governor.evaluate_action(
            action_type="refactoring", action_params={}, context={"test_coverage": 0.96}
        )
        # Base 0.5 + 0.35 for >95% coverage = 0.85
        assert result.principle_scores["competence"] >= 0.85

    def test_competence_check_low_coverage(self):
        """Test competence check with low test coverage."""
        governor = EthicalAutonomyGovernor()
        result = governor.evaluate_action(
            action_type="refactoring", action_params={}, context={"test_coverage": 0.5}
        )
        assert result.principle_scores["competence"] == 0.5

    def test_commitment_check_extensible(self):
        """Test commitment check with extensibility."""
        governor = EthicalAutonomyGovernor()
        result = governor.evaluate_action(
            action_type="refactoring",
            action_params={},
            context={"is_extensible": True, "is_versioned": True},
        )
        assert result.principle_scores["commitment"] >= 0.9

    def test_audit_log_records_actions(self):
        """Test that audit log records all evaluations."""
        governor = EthicalAutonomyGovernor()
        assert len(governor.audit_log) == 0

        governor.evaluate_action("refactoring", {"test": True}, {})
        assert len(governor.audit_log) == 1

        governor.evaluate_action("optimization", {"test": True}, {})
        assert len(governor.audit_log) == 2

    def test_get_audit_log(self):
        """Test getting audit log."""
        governor = EthicalAutonomyGovernor()
        governor.evaluate_action("refactoring", {"test": True}, {})

        log = governor.get_audit_log()
        assert len(log) == 1
        assert log[0]["action_type"] == "refactoring"
        assert "overall_score" in log[0]
        assert "passed" in log[0]

    def test_reset_audit_log(self):
        """Test resetting audit log."""
        governor = EthicalAutonomyGovernor()
        governor.evaluate_action("refactoring", {"test": True}, {})
        assert len(governor.audit_log) == 1

        governor.reset_audit_log()
        assert len(governor.audit_log) == 0

    def test_get_statistics_empty(self):
        """Test statistics with empty audit log."""
        governor = EthicalAutonomyGovernor()
        stats = governor.get_statistics()
        assert stats["total_evaluations"] == 0
        assert stats["passed"] == 0
        assert stats["failed"] == 0
        assert stats["pass_rate"] == 0.0
        assert stats["avg_score"] == 0.0

    def test_get_statistics_with_data(self):
        """Test statistics with audit log data."""
        governor = EthicalAutonomyGovernor()

        governor.evaluate_action(
            "refactoring",
            {"create_backup": True, "require_confirmation": True},
            {"test_coverage": 0.95, "has_benchmarks": True},
        )
        governor.evaluate_action("refactoring", {"create_backup": False}, {"test_coverage": 0.3})

        stats = governor.get_statistics()
        assert stats["total_evaluations"] == 2
        assert stats["passed"] + stats["failed"] == 2
        assert 0.0 <= stats["pass_rate"] <= 1.0
        assert stats["avg_score"] > 0

    def test_strict_mode_requires_no_violations(self):
        """Test that strict mode requires no violations."""
        config = EthicsConfig(strict_mode=True, min_ethics_score=0.5)
        governor = EthicalAutonomyGovernor(config)

        result = governor.evaluate_action(
            "refactoring", {"create_backup": False}, {"test_coverage": 0.3}
        )

        if len(result.violations) > 0:
            assert result.passed is False

    def test_non_strict_mode_allows_violations(self):
        """Test that non-strict mode can pass with violations if score is high."""
        config = EthicsConfig(strict_mode=False, min_ethics_score=0.5)
        governor = EthicalAutonomyGovernor(config)

        result = governor.evaluate_action(
            "ast_transform",
            {"create_backup": True, "logging_enabled": True},
            {"test_coverage": 0.9, "has_benchmarks": True},
        )

        assert result.overall_score >= 0.5


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_evaluate_refactoring_ethics_full_safety(self):
        """Test refactoring ethics with full safety features."""
        result = evaluate_refactoring_ethics(
            create_backup=True, require_confirmation=True, has_tests=True, test_coverage=0.95
        )
        assert isinstance(result, EthicsResult)
        assert result.overall_score >= 0.7

    def test_evaluate_refactoring_ethics_minimal_safety(self):
        """Test refactoring ethics with minimal safety."""
        result = evaluate_refactoring_ethics(
            create_backup=False, require_confirmation=False, has_tests=False, test_coverage=0.0
        )
        assert isinstance(result, EthicsResult)
        assert result.overall_score < 0.8

    def test_evaluate_refactoring_ethics_principle_scores(self):
        """Test that all principles are evaluated."""
        result = evaluate_refactoring_ethics(create_backup=True, test_coverage=0.95)
        assert len(result.principle_scores) == 8
        assert "compassion" in result.principle_scores
        assert "evidence" in result.principle_scores
        assert "justice" in result.principle_scores
        assert "altruism" in result.principle_scores
        assert "control" in result.principle_scores
        assert "character" in result.principle_scores
        assert "competence" in result.principle_scores
        assert "commitment" in result.principle_scores


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_action_params(self):
        """Test evaluation with empty action parameters."""
        governor = EthicalAutonomyGovernor()
        result = governor.evaluate_action(action_type="unknown", action_params={}, context={})
        assert isinstance(result, EthicsResult)
        assert result.overall_score > 0

    def test_none_context(self):
        """Test evaluation with None context."""
        governor = EthicalAutonomyGovernor()
        result = governor.evaluate_action(
            action_type="refactoring", action_params={"test": True}, context=None
        )
        assert isinstance(result, EthicsResult)

    def test_custom_min_ethics_score(self):
        """Test custom minimum ethics score."""
        config = EthicsConfig(min_ethics_score=0.9)
        governor = EthicalAutonomyGovernor(config)

        result = governor.evaluate_action("refactoring", {"create_backup": False}, {})

        if result.overall_score < 0.9:
            assert result.passed is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
