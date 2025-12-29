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
Ethical Bounding and Benevolence Scoring

Implements Phase 6 of the neuro-symbolic evolution:
- Hardcoded utility function for scoring actions on good/evil metrics
- Benevolence threshold enforcement (>=0.99 required)
- Empathy modules for human-centric choices
- Value preservation for positive outcomes
- Audit mechanisms for alignment verification

Research Sources:
- AI Safety (Amodei et al., 2016)
- Value Alignment (Russell, 2019)
- Ethical AI (Floridi & Cowls, 2019)
- Gini Coefficient for Equity (Gini, 1912)

Integration:
    This module provides ethical bounding that integrates with
    the autonomous agent to ensure all actions meet benevolence
    requirements before execution.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EthicalPrinciple(Enum):
    """Core ethical principles."""

    COMPASSION = "compassion"
    EVIDENCE = "evidence"
    JUSTICE = "justice"
    ALTRUISM = "altruism"
    CONTROL = "control"
    CHARACTER = "character"
    COMPETENCE = "competence"
    COMMITMENT = "commitment"


class HarmCategory(Enum):
    """Categories of potential harm."""

    PHYSICAL = "physical"
    PSYCHOLOGICAL = "psychological"
    FINANCIAL = "financial"
    PRIVACY = "privacy"
    AUTONOMY = "autonomy"
    DIGNITY = "dignity"
    ENVIRONMENTAL = "environmental"
    SOCIETAL = "societal"


class BenefitCategory(Enum):
    """Categories of potential benefit."""

    SAFETY = "safety"
    WELLBEING = "wellbeing"
    KNOWLEDGE = "knowledge"
    EFFICIENCY = "efficiency"
    EQUITY = "equity"
    SUSTAINABILITY = "sustainability"
    EMPOWERMENT = "empowerment"
    HUMANITARIAN = "humanitarian"


@dataclass
class EthicalScore:
    """Comprehensive ethical evaluation score."""

    score_id: str
    action: str
    benevolence_score: float
    harm_score: float
    benefit_score: float
    equity_score: float
    long_term_score: float
    is_permissible: bool
    principle_scores: dict[str, float]
    harm_breakdown: dict[str, float]
    benefit_breakdown: dict[str, float]
    explanation: str
    recommendations: list[str]
    timestamp: float = field(default_factory=time.time)


@dataclass
class EmpathyAssessment:
    """Assessment of human-centric impact."""

    assessment_id: str
    affected_parties: list[str]
    impact_scores: dict[str, float]
    vulnerability_factors: list[str]
    mitigation_suggestions: list[str]
    overall_empathy_score: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class ValuePreservation:
    """Value preservation analysis."""

    preservation_id: str
    values_at_risk: list[str]
    preservation_score: float
    default_to_positive: bool
    safeguards_needed: list[str]
    timestamp: float = field(default_factory=time.time)


@dataclass
class AlignmentAudit:
    """Audit record for alignment verification."""

    audit_id: str
    action: str
    ethical_score: EthicalScore
    empathy_assessment: EmpathyAssessment | None
    value_preservation: ValuePreservation | None
    passed: bool
    failure_reasons: list[str]
    timestamp: float = field(default_factory=time.time)


class HarmReducer:
    """
    Evaluates and minimizes potential harm from actions.

    Uses weighted scoring across harm categories to ensure
    actions minimize negative impacts.
    """

    HARM_WEIGHTS = {
        HarmCategory.PHYSICAL: 1.0,
        HarmCategory.PSYCHOLOGICAL: 0.9,
        HarmCategory.FINANCIAL: 0.7,
        HarmCategory.PRIVACY: 0.8,
        HarmCategory.AUTONOMY: 0.85,
        HarmCategory.DIGNITY: 0.9,
        HarmCategory.ENVIRONMENTAL: 0.75,
        HarmCategory.SOCIETAL: 0.8,
    }

    def __init__(self) -> None:
        """Initialize harm reducer."""
        self._evaluation_counter = 0

    def evaluate_harm(
        self,
        action: str,
        context: dict[str, Any],
    ) -> tuple[float, dict[str, float]]:
        """
        Evaluate potential harm from an action.

        Args:
            action: Action to evaluate
            context: Context for the action

        Returns:
            Tuple of (overall_harm_score, category_breakdown)
        """
        breakdown = {}

        for category in HarmCategory:
            harm_level = self._assess_category_harm(action, context, category)
            breakdown[category.value] = harm_level

        weighted_sum = sum(breakdown[cat.value] * self.HARM_WEIGHTS[cat] for cat in HarmCategory)
        max_weighted = sum(self.HARM_WEIGHTS.values())
        overall_harm = weighted_sum / max_weighted

        return overall_harm, breakdown

    def _assess_category_harm(
        self,
        action: str,
        context: dict[str, Any],
        category: HarmCategory,
    ) -> float:
        """Assess harm level for a specific category."""
        harm_keywords = {
            HarmCategory.PHYSICAL: ["injury", "damage", "hurt", "harm", "violence"],
            HarmCategory.PSYCHOLOGICAL: ["stress", "anxiety", "fear", "trauma", "distress"],
            HarmCategory.FINANCIAL: ["loss", "cost", "expense", "debt", "bankruptcy"],
            HarmCategory.PRIVACY: ["expose", "leak", "reveal", "track", "surveil"],
            HarmCategory.AUTONOMY: ["force", "coerce", "manipulate", "control", "restrict"],
            HarmCategory.DIGNITY: ["humiliate", "degrade", "demean", "disrespect"],
            HarmCategory.ENVIRONMENTAL: ["pollute", "destroy", "deplete", "waste"],
            HarmCategory.SOCIETAL: ["divide", "discriminate", "exclude", "marginalize"],
        }

        action_lower = action.lower()
        context_str = str(context).lower()
        combined = action_lower + " " + context_str

        keywords = harm_keywords.get(category, [])
        matches = sum(1 for kw in keywords if kw in combined)

        harm_level = min(1.0, matches * 0.25)

        if context.get("potential_harm"):
            harm_level = min(1.0, harm_level + 0.3)

        return harm_level


class BenefitMaximizer:
    """
    Evaluates and maximizes potential benefits from actions.

    Uses weighted scoring across benefit categories to ensure
    actions maximize positive impacts.
    """

    BENEFIT_WEIGHTS = {
        BenefitCategory.SAFETY: 1.0,
        BenefitCategory.WELLBEING: 0.95,
        BenefitCategory.KNOWLEDGE: 0.7,
        BenefitCategory.EFFICIENCY: 0.6,
        BenefitCategory.EQUITY: 0.85,
        BenefitCategory.SUSTAINABILITY: 0.8,
        BenefitCategory.EMPOWERMENT: 0.75,
        BenefitCategory.HUMANITARIAN: 1.0,
    }

    def __init__(self) -> None:
        """Initialize benefit maximizer."""
        self._evaluation_counter = 0

    def evaluate_benefit(
        self,
        action: str,
        context: dict[str, Any],
    ) -> tuple[float, dict[str, float]]:
        """
        Evaluate potential benefit from an action.

        Args:
            action: Action to evaluate
            context: Context for the action

        Returns:
            Tuple of (overall_benefit_score, category_breakdown)
        """
        breakdown = {}

        for category in BenefitCategory:
            benefit_level = self._assess_category_benefit(action, context, category)
            breakdown[category.value] = benefit_level

        weighted_sum = sum(
            breakdown[cat.value] * self.BENEFIT_WEIGHTS[cat] for cat in BenefitCategory
        )
        max_weighted = sum(self.BENEFIT_WEIGHTS.values())
        overall_benefit = weighted_sum / max_weighted

        return overall_benefit, breakdown

    def _assess_category_benefit(
        self,
        action: str,
        context: dict[str, Any],
        category: BenefitCategory,
    ) -> float:
        """Assess benefit level for a specific category."""
        benefit_keywords = {
            BenefitCategory.SAFETY: ["protect", "secure", "safe", "prevent", "guard"],
            BenefitCategory.WELLBEING: ["health", "wellness", "care", "support", "help"],
            BenefitCategory.KNOWLEDGE: ["learn", "discover", "research", "educate", "inform"],
            BenefitCategory.EFFICIENCY: ["optimize", "improve", "streamline", "automate"],
            BenefitCategory.EQUITY: ["fair", "equal", "inclusive", "accessible", "just"],
            BenefitCategory.SUSTAINABILITY: ["sustain", "renew", "conserve", "preserve"],
            BenefitCategory.EMPOWERMENT: ["enable", "empower", "assist", "facilitate"],
            BenefitCategory.HUMANITARIAN: ["humanitarian", "rescue", "aid", "relief", "crisis"],
        }

        action_lower = action.lower()
        context_str = str(context).lower()
        combined = action_lower + " " + context_str

        keywords = benefit_keywords.get(category, [])
        matches = sum(1 for kw in keywords if kw in combined)

        benefit_level = min(1.0, matches * 0.25)

        if context.get("humanitarian"):
            benefit_level = min(1.0, benefit_level + 0.4)

        return benefit_level


class EquityCalculator:
    """
    Calculates equity metrics using Gini-like coefficients.

    Ensures actions promote fairness and reduce inequality.
    """

    def __init__(self) -> None:
        """Initialize equity calculator."""
        pass

    def calculate_gini(self, values: list[float]) -> float:
        """
        Calculate Gini coefficient for a distribution.

        Args:
            values: List of values representing distribution

        Returns:
            Gini coefficient (0 = perfect equality, 1 = perfect inequality)
        """
        if not values or len(values) < 2:
            return 0.0

        values = sorted(values)
        n = len(values)
        total = sum(values)

        if total == 0:
            return 0.0

        cumulative = 0
        gini_sum = 0
        for i, v in enumerate(values):
            cumulative += v
            gini_sum += (2 * (i + 1) - n - 1) * v

        gini = gini_sum / (n * total)
        return max(0.0, min(1.0, gini))

    def evaluate_equity(
        self,
        action: str,
        context: dict[str, Any],
    ) -> float:
        """
        Evaluate equity impact of an action.

        Args:
            action: Action to evaluate
            context: Context for the action

        Returns:
            Equity score (higher = more equitable)
        """
        base_equity = 0.7

        equity_positive = ["fair", "equal", "inclusive", "accessible", "diverse"]
        equity_negative = ["discriminate", "exclude", "bias", "unfair", "privilege"]

        combined = (action + " " + str(context)).lower()

        for word in equity_positive:
            if word in combined:
                base_equity += 0.1

        for word in equity_negative:
            if word in combined:
                base_equity -= 0.15

        if "distribution" in context:
            dist = context["distribution"]
            if isinstance(dist, list) and len(dist) > 1:
                gini = self.calculate_gini(dist)
                base_equity -= gini * 0.3

        return max(0.0, min(1.0, base_equity))


class EmpathyModule:
    """
    Empathy module for human-centric decision making.

    Considers impact on affected parties and vulnerable populations.
    """

    def __init__(self) -> None:
        """Initialize empathy module."""
        self._assessment_counter = 0

    def assess_empathy(
        self,
        action: str,
        context: dict[str, Any],
    ) -> EmpathyAssessment:
        """
        Assess human-centric impact of an action.

        Args:
            action: Action to assess
            context: Context for the action

        Returns:
            EmpathyAssessment with detailed analysis
        """
        self._assessment_counter += 1
        assessment_id = f"empathy_{self._assessment_counter:06d}"

        affected_parties = self._identify_affected_parties(context)
        impact_scores = self._calculate_impact_scores(action, context, affected_parties)
        vulnerability_factors = self._identify_vulnerabilities(context)
        mitigation_suggestions = self._generate_mitigations(vulnerability_factors)

        overall_score = self._calculate_overall_empathy(impact_scores, vulnerability_factors)

        return EmpathyAssessment(
            assessment_id=assessment_id,
            affected_parties=affected_parties,
            impact_scores=impact_scores,
            vulnerability_factors=vulnerability_factors,
            mitigation_suggestions=mitigation_suggestions,
            overall_empathy_score=overall_score,
        )

    def _identify_affected_parties(self, context: dict[str, Any]) -> list[str]:
        """Identify parties affected by the action."""
        parties = ["general_public"]

        if context.get("users"):
            parties.append("direct_users")
        if context.get("stakeholders"):
            parties.extend(context["stakeholders"])
        if context.get("vulnerable_groups"):
            parties.extend(context["vulnerable_groups"])

        return list(set(parties))

    def _calculate_impact_scores(
        self,
        action: str,
        context: dict[str, Any],
        parties: list[str],
    ) -> dict[str, float]:
        """Calculate impact scores for each affected party."""
        scores = {}

        for party in parties:
            base_score = 0.7

            if "vulnerable" in party.lower():
                base_score -= 0.1
            if "humanitarian" in action.lower():
                base_score += 0.2

            scores[party] = max(0.0, min(1.0, base_score))

        return scores

    def _identify_vulnerabilities(self, context: dict[str, Any]) -> list[str]:
        """Identify vulnerability factors."""
        vulnerabilities = []

        if context.get("children_involved"):
            vulnerabilities.append("children_at_risk")
        if context.get("elderly_involved"):
            vulnerabilities.append("elderly_at_risk")
        if context.get("medical_context"):
            vulnerabilities.append("health_sensitive")
        if context.get("financial_hardship"):
            vulnerabilities.append("economic_vulnerability")

        return vulnerabilities

    def _generate_mitigations(self, vulnerabilities: list[str]) -> list[str]:
        """Generate mitigation suggestions for vulnerabilities."""
        mitigations = []

        mitigation_map = {
            "children_at_risk": "Implement additional safeguards for minors",
            "elderly_at_risk": "Ensure accessibility and clear communication",
            "health_sensitive": "Consult medical ethics guidelines",
            "economic_vulnerability": "Consider financial impact mitigation",
        }

        for vuln in vulnerabilities:
            if vuln in mitigation_map:
                mitigations.append(mitigation_map[vuln])

        return mitigations

    def _calculate_overall_empathy(
        self,
        impact_scores: dict[str, float],
        vulnerabilities: list[str],
    ) -> float:
        """Calculate overall empathy score."""
        if not impact_scores:
            return 0.7

        avg_impact = sum(impact_scores.values()) / len(impact_scores)

        vulnerability_penalty = len(vulnerabilities) * 0.05

        return max(0.0, min(1.0, avg_impact - vulnerability_penalty))


class ValuePreserver:
    """
    Value preservation module for maintaining positive outcomes.

    Ensures actions default to positive outcomes and preserve
    important values.
    """

    CORE_VALUES = [
        "human_dignity",
        "autonomy",
        "privacy",
        "safety",
        "fairness",
        "transparency",
        "accountability",
        "beneficence",
    ]

    def __init__(self) -> None:
        """Initialize value preserver."""
        self._preservation_counter = 0

    def analyze_preservation(
        self,
        action: str,
        context: dict[str, Any],
    ) -> ValuePreservation:
        """
        Analyze value preservation for an action.

        Args:
            action: Action to analyze
            context: Context for the action

        Returns:
            ValuePreservation analysis
        """
        self._preservation_counter += 1
        preservation_id = f"preserve_{self._preservation_counter:06d}"

        values_at_risk = self._identify_values_at_risk(action, context)
        preservation_score = self._calculate_preservation_score(values_at_risk)
        default_to_positive = preservation_score >= 0.7
        safeguards = self._recommend_safeguards(values_at_risk)

        return ValuePreservation(
            preservation_id=preservation_id,
            values_at_risk=values_at_risk,
            preservation_score=preservation_score,
            default_to_positive=default_to_positive,
            safeguards_needed=safeguards,
        )

    def _identify_values_at_risk(
        self,
        action: str,
        context: dict[str, Any],
    ) -> list[str]:
        """Identify values potentially at risk."""
        at_risk = []

        risk_indicators = {
            "human_dignity": ["degrade", "humiliate", "demean"],
            "autonomy": ["force", "coerce", "manipulate"],
            "privacy": ["expose", "track", "surveil", "collect"],
            "safety": ["danger", "risk", "harm", "threat"],
            "fairness": ["bias", "discriminate", "unfair"],
            "transparency": ["hide", "obscure", "deceive"],
            "accountability": ["anonymous", "untraceable"],
            "beneficence": ["harm", "damage", "hurt"],
        }

        combined = (action + " " + str(context)).lower()

        for value, indicators in risk_indicators.items():
            for indicator in indicators:
                if indicator in combined:
                    at_risk.append(value)
                    break

        return at_risk

    def _calculate_preservation_score(self, values_at_risk: list[str]) -> float:
        """Calculate preservation score based on values at risk."""
        if not values_at_risk:
            return 1.0

        risk_ratio = len(values_at_risk) / len(self.CORE_VALUES)
        return max(0.0, 1.0 - risk_ratio)

    def _recommend_safeguards(self, values_at_risk: list[str]) -> list[str]:
        """Recommend safeguards for at-risk values."""
        safeguards = []

        safeguard_map = {
            "human_dignity": "Ensure respectful treatment in all interactions",
            "autonomy": "Provide opt-out mechanisms and informed consent",
            "privacy": "Implement data minimization and encryption",
            "safety": "Add safety checks and human oversight",
            "fairness": "Conduct bias audits and fairness testing",
            "transparency": "Document decision-making processes",
            "accountability": "Maintain audit trails and responsibility chains",
            "beneficence": "Verify positive outcome likelihood",
        }

        for value in values_at_risk:
            if value in safeguard_map:
                safeguards.append(safeguard_map[value])

        return safeguards


class BenevolenceScorer:
    """
    Main benevolence scoring engine.

    Combines harm reduction, benefit maximization, equity,
    empathy, and value preservation into a unified score.
    """

    BENEVOLENCE_THRESHOLD = 0.99

    def __init__(self, benevolence_threshold: float = 0.99) -> None:
        """
        Initialize benevolence scorer.

        Args:
            benevolence_threshold: Minimum score for action approval
        """
        self.benevolence_threshold = benevolence_threshold

        self.harm_reducer = HarmReducer()
        self.benefit_maximizer = BenefitMaximizer()
        self.equity_calculator = EquityCalculator()
        self.empathy_module = EmpathyModule()
        self.value_preserver = ValuePreserver()

        self._score_counter = 0
        self._audit_counter = 0

        self.audit_history: list[AlignmentAudit] = []

        logger.info(f"BenevolenceScorer initialized with threshold {benevolence_threshold}")

    def score_action(
        self,
        action: str,
        context: dict[str, Any],
    ) -> EthicalScore:
        """
        Score an action for benevolence.

        Args:
            action: Action to score
            context: Context for the action

        Returns:
            EthicalScore with comprehensive evaluation
        """
        self._score_counter += 1
        score_id = f"ethical_{self._score_counter:06d}"

        harm_score, harm_breakdown = self.harm_reducer.evaluate_harm(action, context)
        benefit_score, benefit_breakdown = self.benefit_maximizer.evaluate_benefit(action, context)
        equity_score = self.equity_calculator.evaluate_equity(action, context)

        principle_scores = self._evaluate_principles(action, context)

        long_term_score = self._evaluate_long_term(action, context, benefit_score, harm_score)

        benevolence_score = self._calculate_benevolence(
            harm_score=harm_score,
            benefit_score=benefit_score,
            equity_score=equity_score,
            principle_scores=principle_scores,
            long_term_score=long_term_score,
        )

        is_permissible = benevolence_score >= self.benevolence_threshold

        explanation = self._generate_explanation(
            action, benevolence_score, harm_score, benefit_score, is_permissible
        )
        recommendations = self._generate_recommendations(
            harm_breakdown, benefit_breakdown, is_permissible
        )

        return EthicalScore(
            score_id=score_id,
            action=action,
            benevolence_score=benevolence_score,
            harm_score=harm_score,
            benefit_score=benefit_score,
            equity_score=equity_score,
            long_term_score=long_term_score,
            is_permissible=is_permissible,
            principle_scores=principle_scores,
            harm_breakdown=harm_breakdown,
            benefit_breakdown=benefit_breakdown,
            explanation=explanation,
            recommendations=recommendations,
        )

    def _evaluate_principles(
        self,
        action: str,
        context: dict[str, Any],
    ) -> dict[str, float]:
        """Evaluate action against ethical principles."""
        scores = {}

        for principle in EthicalPrinciple:
            scores[principle.value] = self._score_principle(action, context, principle)

        return scores

    def _score_principle(
        self,
        action: str,
        context: dict[str, Any],
        principle: EthicalPrinciple,
    ) -> float:
        """Score action against a specific principle."""
        base_score = 0.8

        principle_keywords = {
            EthicalPrinciple.COMPASSION: ["care", "help", "support", "empathy"],
            EthicalPrinciple.EVIDENCE: ["data", "research", "verify", "prove"],
            EthicalPrinciple.JUSTICE: ["fair", "just", "equal", "rights"],
            EthicalPrinciple.ALTRUISM: ["selfless", "benefit", "humanitarian", "aid"],
            EthicalPrinciple.CONTROL: ["oversight", "review", "approve", "monitor"],
            EthicalPrinciple.CHARACTER: ["integrity", "honest", "ethical", "moral"],
            EthicalPrinciple.COMPETENCE: ["capable", "skilled", "qualified", "expert"],
            EthicalPrinciple.COMMITMENT: ["dedicated", "persistent", "reliable", "consistent"],
        }

        combined = (action + " " + str(context)).lower()
        keywords = principle_keywords.get(principle, [])

        for kw in keywords:
            if kw in combined:
                base_score += 0.05

        return min(1.0, base_score)

    def _evaluate_long_term(
        self,
        action: str,
        context: dict[str, Any],
        benefit_score: float,
        harm_score: float,
    ) -> float:
        """Evaluate long-term societal impact."""
        base_score = 0.7

        base_score += benefit_score * 0.2
        base_score -= harm_score * 0.3

        if context.get("sustainable"):
            base_score += 0.1
        if context.get("short_term_only"):
            base_score -= 0.1

        return max(0.0, min(1.0, base_score))

    def _calculate_benevolence(
        self,
        harm_score: float,
        benefit_score: float,
        equity_score: float,
        principle_scores: dict[str, float],
        long_term_score: float,
    ) -> float:
        """
        Calculate overall benevolence score.

        Formula weights:
        - Harm reduction: 30% (inverted)
        - Benefit maximization: 25%
        - Equity: 20%
        - Principles average: 15%
        - Long-term impact: 10%
        """
        harm_component = (1 - harm_score) * 0.30
        benefit_component = benefit_score * 0.25
        equity_component = equity_score * 0.20

        principles_avg = sum(principle_scores.values()) / len(principle_scores)
        principles_component = principles_avg * 0.15

        long_term_component = long_term_score * 0.10

        benevolence = (
            harm_component
            + benefit_component
            + equity_component
            + principles_component
            + long_term_component
        )

        return max(0.0, min(1.0, benevolence))

    def _generate_explanation(
        self,
        action: str,
        benevolence_score: float,
        harm_score: float,
        benefit_score: float,
        is_permissible: bool,
    ) -> str:
        """Generate explanation for the ethical score."""
        status = "APPROVED" if is_permissible else "BLOCKED"

        return (
            f"Action '{action}' scored {benevolence_score:.2%} benevolence ({status}). "
            f"Harm potential: {harm_score:.0%}, Benefit potential: {benefit_score:.0%}. "
            f"Threshold: {self.benevolence_threshold:.0%}."
        )

    def _generate_recommendations(
        self,
        harm_breakdown: dict[str, float],
        benefit_breakdown: dict[str, float],
        is_permissible: bool,
    ) -> list[str]:
        """Generate recommendations for improving ethical score."""
        recommendations = []

        if not is_permissible:
            recommendations.append("Action does not meet benevolence threshold")

        high_harm_categories = [cat for cat, score in harm_breakdown.items() if score > 0.3]
        for cat in high_harm_categories:
            recommendations.append(f"Reduce {cat} harm potential")

        low_benefit_categories = [cat for cat, score in benefit_breakdown.items() if score < 0.3]
        if low_benefit_categories:
            recommendations.append("Consider ways to increase positive impact")

        return recommendations

    def full_audit(
        self,
        action: str,
        context: dict[str, Any],
    ) -> AlignmentAudit:
        """
        Perform full alignment audit on an action.

        Args:
            action: Action to audit
            context: Context for the action

        Returns:
            AlignmentAudit with comprehensive analysis
        """
        self._audit_counter += 1
        audit_id = f"audit_{self._audit_counter:06d}"

        ethical_score = self.score_action(action, context)
        empathy_assessment = self.empathy_module.assess_empathy(action, context)
        value_preservation = self.value_preserver.analyze_preservation(action, context)

        failure_reasons = []

        if not ethical_score.is_permissible:
            failure_reasons.append(
                f"Benevolence score {ethical_score.benevolence_score:.2%} below threshold"
            )

        if empathy_assessment.overall_empathy_score < 0.7:
            failure_reasons.append(
                f"Empathy score {empathy_assessment.overall_empathy_score:.2%} too low"
            )

        if not value_preservation.default_to_positive:
            failure_reasons.append("Action does not default to positive outcomes")

        passed = len(failure_reasons) == 0

        audit = AlignmentAudit(
            audit_id=audit_id,
            action=action,
            ethical_score=ethical_score,
            empathy_assessment=empathy_assessment,
            value_preservation=value_preservation,
            passed=passed,
            failure_reasons=failure_reasons,
        )

        self.audit_history.append(audit)

        return audit

    def is_action_permissible(
        self,
        action: str,
        context: dict[str, Any],
    ) -> tuple[bool, float, str]:
        """
        Quick check if action is permissible.

        Args:
            action: Action to check
            context: Context for the action

        Returns:
            Tuple of (is_permissible, benevolence_score, explanation)
        """
        score = self.score_action(action, context)
        return score.is_permissible, score.benevolence_score, score.explanation

    def get_statistics(self) -> dict[str, Any]:
        """Get scorer statistics."""
        passed_audits = sum(1 for a in self.audit_history if a.passed)

        return {
            "scores_generated": self._score_counter,
            "audits_performed": self._audit_counter,
            "audits_passed": passed_audits,
            "pass_rate": passed_audits / self._audit_counter if self._audit_counter > 0 else 0,
            "benevolence_threshold": self.benevolence_threshold,
        }

    def get_audit_history(self, limit: int = 100) -> list[AlignmentAudit]:
        """Get recent audit history."""
        return self.audit_history[-limit:]
