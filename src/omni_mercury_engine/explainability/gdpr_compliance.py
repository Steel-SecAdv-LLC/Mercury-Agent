"""
GDPR Article 22 Compliance for Mercury Agent.

Implements explanation generation and documentation for automated
decision-making systems to comply with GDPR Article 22 requirements.

References:
- GDPR Article 22: Automated individual decision-making
- Article 29 Working Party Guidelines on Automated Decision-Making
- ICO Guidance on Explaining Decisions Made with AI
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.explainability.counterfactuals import (
    CounterfactualSet,
    create_counterfactual_generator,
)
from omni_mercury_engine.explainability.shap import (
    ShapExplanation,
    create_shap_explainer,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np


logger = logging.getLogger(__name__)


class DecisionCategory(Enum):
    """Categories of automated decisions under GDPR."""

    HIGH_IMPACT = auto()
    LEGALLY_SIGNIFICANT = auto()
    PROFILING = auto()
    STANDARD = auto()


class ExplanationLevel(Enum):
    """Level of explanation detail."""

    MINIMAL = auto()
    STANDARD = auto()
    DETAILED = auto()
    FULL = auto()


@dataclass
class DataSubjectInfo:
    """Information about the data subject."""

    subject_id: str
    consent_given: bool = False
    consent_timestamp: float | None = None
    legitimate_interest: bool = False
    contract_necessity: bool = False
    legal_obligation: bool = False
    special_category_data: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionInfo:
    """Information about an automated decision."""

    decision_id: str
    decision_value: Any
    confidence: float
    timestamp: float
    model_id: str
    model_version: str
    input_features: dict[str, Any]
    category: DecisionCategory
    human_reviewed: bool = False
    reviewer_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExplanationReport:
    """
    GDPR-compliant explanation report.

    Contains all information required for Article 22 compliance,
    including meaningful information about the logic involved,
    significance, and envisaged consequences.
    """

    report_id: str
    decision_info: DecisionInfo
    subject_info: DataSubjectInfo

    logic_explanation: str
    feature_contributions: dict[str, float]
    top_factors: list[tuple[str, float, str]]
    counterfactual_actions: list[dict[str, Any]]

    significance: str
    consequences: str
    rights_info: str

    generated_at: float
    explanation_level: ExplanationLevel
    shap_explanation: ShapExplanation | None = None
    counterfactual_set: CounterfactualSet | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary format."""
        return {
            "report_id": self.report_id,
            "decision": {
                "id": self.decision_info.decision_id,
                "value": self.decision_info.decision_value,
                "confidence": self.decision_info.confidence,
                "timestamp": self.decision_info.timestamp,
                "model": self.decision_info.model_id,
                "category": self.decision_info.category.name,
            },
            "subject": {
                "id": self.subject_info.subject_id,
                "consent_given": self.subject_info.consent_given,
            },
            "explanation": {
                "logic": self.logic_explanation,
                "feature_contributions": self.feature_contributions,
                "top_factors": [
                    {"feature": f, "contribution": c, "description": d}
                    for f, c, d in self.top_factors
                ],
                "counterfactual_actions": self.counterfactual_actions,
            },
            "impact": {
                "significance": self.significance,
                "consequences": self.consequences,
            },
            "rights": self.rights_info,
            "generated_at": self.generated_at,
            "explanation_level": self.explanation_level.name,
        }

    def to_human_readable(self) -> str:
        """Generate human-readable explanation text."""
        lines = [
            f"EXPLANATION REPORT: {self.report_id}",
            "=" * 60,
            "",
            "DECISION SUMMARY",
            "-" * 40,
            f"Decision: {self.decision_info.decision_value}",
            f"Confidence: {self.decision_info.confidence:.1%}",
            f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.decision_info.timestamp))}",
            "",
            "HOW THIS DECISION WAS MADE",
            "-" * 40,
            self.logic_explanation,
            "",
            "KEY FACTORS IN THIS DECISION",
            "-" * 40,
        ]

        for feature, contribution, description in self.top_factors[:5]:
            direction = "increased" if contribution > 0 else "decreased"
            lines.append(f"- {feature}: {direction} the likelihood ({contribution:+.2f})")
            if description:
                lines.append(f"  {description}")

        if self.counterfactual_actions:
            lines.extend(
                [
                    "",
                    "WHAT COULD CHANGE THIS DECISION",
                    "-" * 40,
                ]
            )
            for i, action in enumerate(self.counterfactual_actions[:3], 1):
                lines.append(f"{i}. {action.get('description', 'Change not specified')}")

        lines.extend(
            [
                "",
                "SIGNIFICANCE AND CONSEQUENCES",
                "-" * 40,
                f"Significance: {self.significance}",
                f"Potential consequences: {self.consequences}",
                "",
                "YOUR RIGHTS",
                "-" * 40,
                self.rights_info,
            ]
        )

        return "\n".join(lines)


@dataclass
class ComplianceAuditRecord:
    """Audit record for compliance tracking."""

    record_id: str
    decision_id: str
    subject_id: str
    explanation_provided: bool
    explanation_timestamp: float
    human_review_requested: bool
    human_review_completed: bool
    objection_raised: bool
    objection_resolved: bool
    data_subject_notified: bool
    notification_timestamp: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


class GDPRExplainer:
    """
    GDPR Article 22 compliant explainer.

    Generates explanations that satisfy the requirements of GDPR
    Article 22 for automated decision-making:

    1. Meaningful information about the logic involved
    2. Significance and envisaged consequences
    3. Right to obtain human intervention
    4. Right to express views and contest decision

    Example:
        explainer = GDPRExplainer(
            model=my_model,
            background_data=X_train,
            feature_names=feature_names,
        )

        # Generate explanation for a decision
        report = explainer.explain_decision(
            instance=X_test[0],
            decision_value=model.predict(X_test[0:1])[0],
            subject_id="user_123",
        )

        # Get human-readable text
        print(report.to_human_readable())
    """

    RIGHTS_TEXT_TEMPLATE = """
Under GDPR Article 22, you have the right to:
1. Request human review of this decision
2. Express your point of view about this decision
3. Contest this decision
4. Request information about the logic involved
5. Object to decisions based solely on automated processing

To exercise these rights, contact: {contact_info}
Reference ID: {decision_id}
"""

    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray] | Any,
        background_data: np.ndarray,
        feature_names: list[str] | None = None,
        feature_descriptions: dict[str, str] | None = None,
        model_id: str = "unknown",
        model_version: str = "1.0",
        contact_info: str = "data-protection@organization.com",
        shap_method: str = "auto",
        counterfactual_method: str = "wachter",
    ) -> None:
        """
        Initialize GDPR explainer.

        Args:
            model: Model or prediction function
            background_data: Background data for SHAP
            feature_names: Feature names
            feature_descriptions: Human-readable feature descriptions
            model_id: Model identifier
            model_version: Model version
            contact_info: Contact information for rights requests
            shap_method: SHAP method to use
            counterfactual_method: Counterfactual method to use
        """
        self._model = model
        self._background_data = background_data
        self._feature_names = feature_names or [
            f"feature_{i}" for i in range(background_data.shape[1])
        ]
        self._feature_descriptions = feature_descriptions or {}
        self._model_id = model_id
        self._model_version = model_version
        self._contact_info = contact_info

        if callable(model):
            self._predict = model
        elif hasattr(model, "predict_proba"):
            self._predict = lambda x: model.predict_proba(x)[:, 1]
        elif hasattr(model, "predict"):
            self._predict = model.predict
        else:
            raise ValueError("Model must be callable or have predict method")

        self._shap_explainer = create_shap_explainer(
            model,
            background_data,
            feature_names,
            shap_method,
        )

        self._cf_generator = create_counterfactual_generator(
            model,
            counterfactual_method,
            feature_names=feature_names,
        )

        self._audit_records: list[ComplianceAuditRecord] = []
        self._report_counter = 0

    def explain_decision(
        self,
        instance: np.ndarray,
        decision_value: Any,
        subject_id: str,
        confidence: float | None = None,
        decision_category: DecisionCategory = DecisionCategory.STANDARD,
        explanation_level: ExplanationLevel = ExplanationLevel.STANDARD,
        consent_given: bool = False,
        include_counterfactuals: bool = True,
    ) -> ExplanationReport:
        """
        Generate GDPR-compliant explanation for a decision.

        Args:
            instance: Input features for the decision
            decision_value: The automated decision
            subject_id: Identifier for the data subject
            confidence: Decision confidence score
            decision_category: Category of the decision
            explanation_level: Level of detail
            consent_given: Whether explicit consent was given
            include_counterfactuals: Whether to include counterfactual actions

        Returns:
            ExplanationReport with all required information
        """
        self._report_counter += 1
        report_id = f"EXPL-{int(time.time())}-{self._report_counter:04d}"
        decision_id = f"DEC-{int(time.time())}-{self._report_counter:04d}"

        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        if confidence is None:
            pred = self._predict(instance)
            confidence = float(abs(pred[0] - 0.5) * 2) if hasattr(pred, "__len__") else 0.5

        subject_info = DataSubjectInfo(
            subject_id=subject_id,
            consent_given=consent_given,
            consent_timestamp=time.time() if consent_given else None,
        )

        input_features = {
            self._feature_names[i]: float(instance[0, i]) for i in range(len(self._feature_names))
        }

        decision_info = DecisionInfo(
            decision_id=decision_id,
            decision_value=decision_value,
            confidence=confidence,
            timestamp=time.time(),
            model_id=self._model_id,
            model_version=self._model_version,
            input_features=input_features,
            category=decision_category,
        )

        shap_result = self._shap_explainer.explain(instance[0])
        shap_explanation: ShapExplanation = (
            shap_result if isinstance(shap_result, ShapExplanation) else shap_result[0]
        )

        feature_contributions = shap_explanation.get_feature_importance()

        top_factors = self._generate_top_factors(
            shap_explanation,
            explanation_level,
        )

        counterfactual_set = None
        counterfactual_actions = []
        if include_counterfactuals:
            target_class = 0 if decision_value == 1 else 1
            counterfactual_set = self._cf_generator.generate(
                instance[0],
                target_class=target_class,
                n_counterfactuals=3,
            )
            counterfactual_actions = self._generate_cf_actions(counterfactual_set)

        logic_explanation = self._generate_logic_explanation(
            decision_value,
            top_factors,
            explanation_level,
        )

        significance = self._generate_significance(decision_category, decision_value)
        consequences = self._generate_consequences(decision_category, decision_value)

        rights_info = self.RIGHTS_TEXT_TEMPLATE.format(
            contact_info=self._contact_info,
            decision_id=decision_id,
        )

        report = ExplanationReport(
            report_id=report_id,
            decision_info=decision_info,
            subject_info=subject_info,
            logic_explanation=logic_explanation,
            feature_contributions=feature_contributions,
            top_factors=top_factors,
            counterfactual_actions=counterfactual_actions,
            significance=significance,
            consequences=consequences,
            rights_info=rights_info,
            generated_at=time.time(),
            explanation_level=explanation_level,
            shap_explanation=shap_explanation,
            counterfactual_set=counterfactual_set,
        )

        self._record_audit(report)

        return report

    def _generate_top_factors(
        self,
        shap_explanation: ShapExplanation,
        level: ExplanationLevel,
    ) -> list[tuple[str, float, str]]:
        """Generate top contributing factors with descriptions."""
        n_factors = {
            ExplanationLevel.MINIMAL: 3,
            ExplanationLevel.STANDARD: 5,
            ExplanationLevel.DETAILED: 10,
            ExplanationLevel.FULL: len(shap_explanation.shap_values),
        }[level]

        top_features = shap_explanation.get_top_features(n_factors)

        factors = []
        for feature, contribution in top_features:
            description = self._feature_descriptions.get(feature, "")
            factors.append((feature, contribution, description))

        return factors

    def _generate_cf_actions(
        self,
        cf_set: CounterfactualSet,
    ) -> list[dict[str, Any]]:
        """Generate actionable counterfactual descriptions."""
        actions = []

        for cf in cf_set.counterfactuals:
            if not cf.validity:
                continue

            changes = []
            for feature, (old, new) in cf.feature_changes.items():
                direction = "increase" if new > old else "decrease"
                amount = abs(new - old)
                changes.append(
                    {
                        "feature": feature,
                        "direction": direction,
                        "amount": amount,
                        "from": old,
                        "to": new,
                    }
                )

            if changes:
                change_descriptions = [
                    f"{c['direction']} {c['feature']} by {c['amount']:.2f}" for c in changes[:3]
                ]
                description = "To change this decision, you could: " + ", ".join(
                    change_descriptions
                )

                actions.append(
                    {
                        "description": description,
                        "changes": changes,
                        "distance": cf.distance,
                        "feasibility": 1.0 / (1.0 + cf.distance),
                    }
                )

        return actions

    def _generate_logic_explanation(
        self,
        decision_value: Any,
        top_factors: list[tuple[str, float, str]],
        level: ExplanationLevel,
    ) -> str:
        """Generate human-readable logic explanation."""
        positive_factors = [f for f in top_factors if f[1] > 0]
        negative_factors = [f for f in top_factors if f[1] < 0]

        parts = [
            f"This decision ({decision_value}) was made by analyzing multiple factors "
            f"from your data using a machine learning model."
        ]

        if level in [ExplanationLevel.STANDARD, ExplanationLevel.DETAILED, ExplanationLevel.FULL]:
            if positive_factors:
                factor_names = [f[0] for f in positive_factors[:3]]
                parts.append(
                    f"Factors that contributed positively to this decision include: "
                    f"{', '.join(factor_names)}."
                )

            if negative_factors:
                factor_names = [f[0] for f in negative_factors[:3]]
                parts.append(
                    f"Factors that contributed negatively include: " f"{', '.join(factor_names)}."
                )

        if level == ExplanationLevel.FULL:
            parts.append(
                f"The model ({self._model_id} v{self._model_version}) processed "
                f"{len(top_factors)} features to arrive at this decision."
            )

        return " ".join(parts)

    def _generate_significance(
        self,
        category: DecisionCategory,
        decision_value: Any,
    ) -> str:
        """Generate significance statement."""
        significance_map = {
            DecisionCategory.HIGH_IMPACT: (
                "This decision has significant impact on your situation "
                "and may affect important aspects of your life or circumstances."
            ),
            DecisionCategory.LEGALLY_SIGNIFICANT: (
                "This decision has legal significance and may affect your "
                "legal rights, status, or obligations."
            ),
            DecisionCategory.PROFILING: (
                "This decision involves profiling based on analysis of "
                "personal aspects including behavior, preferences, or characteristics."
            ),
            DecisionCategory.STANDARD: (
                "This decision may affect the services or information " "provided to you."
            ),
        }
        return significance_map.get(category, significance_map[DecisionCategory.STANDARD])

    def _generate_consequences(
        self,
        category: DecisionCategory,
        decision_value: Any,
    ) -> str:
        """Generate consequences statement."""
        consequences_map = {
            DecisionCategory.HIGH_IMPACT: (
                "Potential consequences include changes to services, eligibility, "
                "or treatment you receive. You may request human review of this decision."
            ),
            DecisionCategory.LEGALLY_SIGNIFICANT: (
                "This may affect your legal standing, contractual rights, or "
                "regulatory compliance. Legal consultation may be advisable."
            ),
            DecisionCategory.PROFILING: (
                "Your profile may be used for personalized services or targeted "
                "communications. You have the right to object to profiling."
            ),
            DecisionCategory.STANDARD: (
                "This may affect the content, recommendations, or services " "shown to you."
            ),
        }
        return consequences_map.get(category, consequences_map[DecisionCategory.STANDARD])

    def _record_audit(self, report: ExplanationReport) -> None:
        """Record audit trail for the explanation."""
        record = ComplianceAuditRecord(
            record_id=f"AUDIT-{report.report_id}",
            decision_id=report.decision_info.decision_id,
            subject_id=report.subject_info.subject_id,
            explanation_provided=True,
            explanation_timestamp=report.generated_at,
            human_review_requested=False,
            human_review_completed=report.decision_info.human_reviewed,
            objection_raised=False,
            objection_resolved=False,
            data_subject_notified=False,
            notification_timestamp=None,
        )
        self._audit_records.append(record)

    def request_human_review(
        self,
        decision_id: str,
        subject_id: str,
        reason: str = "",
    ) -> bool:
        """
        Request human review of a decision.

        Args:
            decision_id: ID of the decision
            subject_id: ID of the data subject
            reason: Reason for review request

        Returns:
            True if request was recorded
        """
        for record in self._audit_records:
            if record.decision_id == decision_id and record.subject_id == subject_id:
                record.human_review_requested = True
                record.metadata["review_reason"] = reason
                record.metadata["review_request_time"] = time.time()
                logger.info(f"Human review requested for decision {decision_id}")
                return True

        return False

    def record_objection(
        self,
        decision_id: str,
        subject_id: str,
        objection_text: str,
    ) -> bool:
        """
        Record data subject's objection to a decision.

        Args:
            decision_id: ID of the decision
            subject_id: ID of the data subject
            objection_text: Text of the objection

        Returns:
            True if objection was recorded
        """
        for record in self._audit_records:
            if record.decision_id == decision_id and record.subject_id == subject_id:
                record.objection_raised = True
                record.metadata["objection_text"] = objection_text
                record.metadata["objection_time"] = time.time()
                logger.info(f"Objection recorded for decision {decision_id}")
                return True

        return False

    def get_audit_records(
        self,
        subject_id: str | None = None,
    ) -> list[ComplianceAuditRecord]:
        """
        Get audit records, optionally filtered by subject.

        Args:
            subject_id: Optional subject ID to filter by

        Returns:
            List of audit records
        """
        if subject_id is None:
            return self._audit_records.copy()

        return [r for r in self._audit_records if r.subject_id == subject_id]

    def generate_compliance_report(self) -> dict[str, Any]:
        """
        Generate compliance summary report.

        Returns:
            Dictionary with compliance statistics
        """
        total_decisions = len(self._audit_records)
        explanations_provided = sum(1 for r in self._audit_records if r.explanation_provided)
        human_reviews_requested = sum(1 for r in self._audit_records if r.human_review_requested)
        human_reviews_completed = sum(1 for r in self._audit_records if r.human_review_completed)
        objections_raised = sum(1 for r in self._audit_records if r.objection_raised)
        objections_resolved = sum(1 for r in self._audit_records if r.objection_resolved)

        return {
            "report_generated": time.time(),
            "model_id": self._model_id,
            "model_version": self._model_version,
            "statistics": {
                "total_decisions": total_decisions,
                "explanations_provided": explanations_provided,
                "explanation_rate": explanations_provided / max(1, total_decisions),
                "human_reviews_requested": human_reviews_requested,
                "human_reviews_completed": human_reviews_completed,
                "human_review_completion_rate": human_reviews_completed
                / max(1, human_reviews_requested),
                "objections_raised": objections_raised,
                "objections_resolved": objections_resolved,
                "objection_resolution_rate": objections_resolved / max(1, objections_raised),
            },
            "compliance_status": {
                "article_22_compliance": explanations_provided == total_decisions,
                "human_review_available": True,
                "objection_mechanism_available": True,
            },
        }
