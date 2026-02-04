"""
Unified Explainability Interface for Mercury Agent.

High-level interface for generating explanations for anomaly detection
models using SHAP, counterfactuals, and GDPR-compliant reports.

Example:
    explainer = MercuryExplainer(
        model=my_model,
        background_data=X_train,
        feature_names=feature_names,
    )

    # Generate comprehensive explanation
    explanation = explainer.explain(
        instance=X_test[0],
        include_shap=True,
        include_counterfactuals=True,
        include_gdpr_report=True,
    )

    # Access different explanation types
    print(explanation.shap_values)
    print(explanation.counterfactuals)
    print(explanation.gdpr_report.to_human_readable())
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.explainability.counterfactuals import (
    CounterfactualSet,
    FeatureConstraint,
    create_counterfactual_generator,
)
from omni_mercury_engine.explainability.gdpr_compliance import (
    DecisionCategory,
    ExplanationLevel,
    ExplanationReport,
    GDPRExplainer,
)
from omni_mercury_engine.explainability.shap import (
    GlobalExplanation,
    ShapExplanation,
    create_shap_explainer,
)


if TYPE_CHECKING:
    from collections.abc import Callable


logger = logging.getLogger(__name__)


@dataclass
class AnomalyExplanation:
    """
    Comprehensive explanation for an anomaly detection result.

    Combines multiple explanation methods into a unified result.
    """

    instance: np.ndarray
    anomaly_score: float
    is_anomaly: bool
    threshold: float

    shap_explanation: ShapExplanation | None = None
    counterfactual_set: CounterfactualSet | None = None
    gdpr_report: ExplanationReport | None = None

    feature_names: list[str] | None = None
    explanation_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_top_anomaly_factors(self, n: int = 5) -> list[tuple[str, float]]:
        """Get top features contributing to anomaly classification."""
        if self.shap_explanation is None:
            return []

        return self.shap_explanation.get_top_features(n)

    def get_actionable_changes(self) -> list[dict[str, Any]]:
        """Get actionable changes to reduce anomaly score."""
        if self.counterfactual_set is None:
            return []

        actions = []
        for cf in self.counterfactual_set.counterfactuals:
            if cf.validity:
                actions.append({
                    "changes": cf.feature_changes,
                    "predicted_score": cf.counterfactual_prediction,
                    "distance": cf.distance,
                })

        return sorted(actions, key=lambda x: x["distance"])

    def to_dict(self) -> dict[str, Any]:
        """Convert explanation to dictionary."""
        result = {
            "instance": self.instance.tolist(),
            "anomaly_score": self.anomaly_score,
            "is_anomaly": self.is_anomaly,
            "threshold": self.threshold,
            "explanation_time": self.explanation_time,
        }

        if self.shap_explanation is not None:
            result["shap"] = self.shap_explanation.to_dict()

        if self.counterfactual_set is not None:
            result["counterfactuals"] = [
                cf.to_dict() for cf in self.counterfactual_set.counterfactuals
            ]

        if self.gdpr_report is not None:
            result["gdpr_report"] = self.gdpr_report.to_dict()

        return result

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Anomaly Explanation",
            "=" * 50,
            f"Anomaly Score: {self.anomaly_score:.4f}",
            f"Is Anomaly: {self.is_anomaly}",
            f"Threshold: {self.threshold:.4f}",
            "",
        ]

        if self.shap_explanation is not None:
            lines.append("Top Contributing Factors:")
            lines.append("-" * 30)
            for feature, contribution in self.get_top_anomaly_factors(5):
                direction = "+" if contribution > 0 else ""
                lines.append(f"  {feature}: {direction}{contribution:.4f}")
            lines.append("")

        if self.counterfactual_set is not None:
            valid_cfs = [cf for cf in self.counterfactual_set.counterfactuals if cf.validity]
            if valid_cfs:
                lines.append("Suggested Changes to Reduce Anomaly Score:")
                lines.append("-" * 30)
                for cf in valid_cfs[:2]:
                    for feature, (old, new) in list(cf.feature_changes.items())[:3]:
                        lines.append(f"  {feature}: {old:.4f} -> {new:.4f}")
                lines.append("")

        return "\n".join(lines)


@dataclass
class GlobalAnomalyExplanation:
    """Global explanation for anomaly detection across a dataset."""

    global_shap: GlobalExplanation | None
    feature_importance: dict[str, float]
    anomaly_rate: float
    threshold: float
    n_samples: int

    feature_names: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_most_important_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Get most important features globally."""
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return sorted_features[:n]


class MercuryExplainer:
    """
    Unified explainability interface for Mercury Agent.

    Provides comprehensive explanations for anomaly detection models
    using multiple explanation methods.

    Features:
    - SHAP values for feature attribution
    - Counterfactual explanations for actionable insights
    - GDPR Article 22 compliant reports
    - Global and local explanations
    - Customizable explanation depth

    Example:
        explainer = MercuryExplainer(
            model=anomaly_detector,
            background_data=X_train,
            feature_names=["feature_1", "feature_2", ...],
            model_id="mercury_anomaly_v1",
        )

        # Explain single instance
        explanation = explainer.explain(
            instance=X_test[0],
            anomaly_score=0.85,
            is_anomaly=True,
        )

        # Get actionable recommendations
        for action in explanation.get_actionable_changes():
            print(f"Change {action['changes']} to reduce score")
    """

    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray] | Any,
        background_data: np.ndarray,
        feature_names: list[str] | None = None,
        feature_descriptions: dict[str, str] | None = None,
        feature_constraints: list[FeatureConstraint] | None = None,
        model_id: str = "mercury_model",
        model_version: str = "1.0",
        threshold: float = 0.5,
        shap_method: str = "auto",
        counterfactual_method: str = "wachter",
        contact_info: str = "support@organization.com",
    ) -> None:
        """
        Initialize Mercury Explainer.

        Args:
            model: Anomaly detection model or scoring function
            background_data: Background data for explanation methods
            feature_names: Names of input features
            feature_descriptions: Human-readable feature descriptions
            feature_constraints: Constraints for counterfactual generation
            model_id: Model identifier
            model_version: Model version string
            threshold: Anomaly threshold
            shap_method: SHAP method ("auto", "kernel", "sampling", etc.)
            counterfactual_method: Counterfactual method ("wachter", "dice", etc.)
            contact_info: Contact for GDPR requests
        """
        self._model = model
        self._background_data = background_data
        self._feature_names = feature_names or [
            f"feature_{i}" for i in range(background_data.shape[1])
        ]
        self._feature_descriptions = feature_descriptions or {}
        self._feature_constraints = feature_constraints
        self._model_id = model_id
        self._model_version = model_version
        self._threshold = threshold

        if callable(model):
            self._predict = model
        elif hasattr(model, "decision_function"):
            self._predict = model.decision_function
        elif hasattr(model, "predict_proba"):
            self._predict = lambda x: model.predict_proba(x)[:, 1]
        elif hasattr(model, "predict"):
            self._predict = model.predict
        else:
            raise ValueError("Model must be callable or have predict method")

        self._shap_explainer = create_shap_explainer(
            self._predict,
            background_data,
            feature_names,
            shap_method,
        )

        self._cf_generator = create_counterfactual_generator(
            self._predict,
            counterfactual_method,
            feature_names=feature_names,
            feature_constraints=feature_constraints,
        )

        self._gdpr_explainer = GDPRExplainer(
            model=self._predict,
            background_data=background_data,
            feature_names=feature_names,
            feature_descriptions=feature_descriptions,
            model_id=model_id,
            model_version=model_version,
            contact_info=contact_info,
            shap_method=shap_method,
            counterfactual_method=counterfactual_method,
        )

    def explain(
        self,
        instance: np.ndarray,
        anomaly_score: float | None = None,
        is_anomaly: bool | None = None,
        include_shap: bool = True,
        include_counterfactuals: bool = True,
        include_gdpr_report: bool = False,
        subject_id: str | None = None,
        n_counterfactuals: int = 3,
    ) -> AnomalyExplanation:
        """
        Generate comprehensive explanation for an instance.

        Args:
            instance: Instance to explain
            anomaly_score: Pre-computed anomaly score (computed if None)
            is_anomaly: Pre-computed anomaly label (computed if None)
            include_shap: Whether to include SHAP values
            include_counterfactuals: Whether to include counterfactuals
            include_gdpr_report: Whether to generate GDPR report
            subject_id: Data subject ID (required for GDPR report)
            n_counterfactuals: Number of counterfactuals to generate

        Returns:
            AnomalyExplanation with requested explanation types
        """
        start_time = time.time()

        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        if anomaly_score is None:
            anomaly_score = float(self._predict(instance)[0])

        if is_anomaly is None:
            is_anomaly = anomaly_score > self._threshold

        shap_explanation = None
        if include_shap:
            shap_explanation = self._shap_explainer.explain(instance[0])

        counterfactual_set = None
        if include_counterfactuals:
            target_class = 0 if is_anomaly else 1
            counterfactual_set = self._cf_generator.generate(
                instance[0],
                target_class=target_class,
                n_counterfactuals=n_counterfactuals,
            )

        gdpr_report = None
        if include_gdpr_report:
            if subject_id is None:
                subject_id = f"subject_{int(time.time())}"

            decision_category = (
                DecisionCategory.HIGH_IMPACT if is_anomaly
                else DecisionCategory.STANDARD
            )

            gdpr_report = self._gdpr_explainer.explain_decision(
                instance=instance[0],
                decision_value="Anomaly" if is_anomaly else "Normal",
                subject_id=subject_id,
                confidence=anomaly_score,
                decision_category=decision_category,
                include_counterfactuals=include_counterfactuals,
            )

        explanation_time = time.time() - start_time

        return AnomalyExplanation(
            instance=instance[0],
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            threshold=self._threshold,
            shap_explanation=shap_explanation,
            counterfactual_set=counterfactual_set,
            gdpr_report=gdpr_report,
            feature_names=self._feature_names,
            explanation_time=explanation_time,
        )

    def explain_batch(
        self,
        instances: np.ndarray,
        anomaly_scores: np.ndarray | None = None,
        include_shap: bool = True,
        include_counterfactuals: bool = False,
    ) -> list[AnomalyExplanation]:
        """
        Generate explanations for multiple instances.

        Args:
            instances: Batch of instances
            anomaly_scores: Pre-computed anomaly scores
            include_shap: Whether to include SHAP values
            include_counterfactuals: Whether to include counterfactuals

        Returns:
            List of AnomalyExplanation objects
        """
        if anomaly_scores is None:
            anomaly_scores = self._predict(instances)

        explanations = []
        for i, instance in enumerate(instances):
            explanation = self.explain(
                instance=instance,
                anomaly_score=float(anomaly_scores[i]),
                include_shap=include_shap,
                include_counterfactuals=include_counterfactuals,
                include_gdpr_report=False,
            )
            explanations.append(explanation)

        return explanations

    def explain_global(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> GlobalAnomalyExplanation:
        """
        Generate global explanation for the model.

        Args:
            X: Dataset to analyze
            y: Optional labels (anomaly=1, normal=0)

        Returns:
            GlobalAnomalyExplanation with feature importance
        """
        scores = self._predict(X)
        anomalies = scores > self._threshold
        anomaly_rate = np.mean(anomalies)

        global_shap = self._shap_explainer.explain_global(X)

        feature_importance = global_shap.get_feature_importance()

        return GlobalAnomalyExplanation(
            global_shap=global_shap,
            feature_importance=feature_importance,
            anomaly_rate=float(anomaly_rate),
            threshold=self._threshold,
            n_samples=len(X),
            feature_names=self._feature_names,
        )

    def get_feature_importance(
        self,
        X: np.ndarray,
        n_samples: int | None = None,
    ) -> dict[str, float]:
        """
        Compute global feature importance.

        Args:
            X: Dataset for importance computation
            n_samples: Number of samples to use (all if None)

        Returns:
            Dictionary of feature importances
        """
        if n_samples is not None and n_samples < len(X):
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]

        global_explanation = self.explain_global(X)
        return global_explanation.feature_importance

    def generate_report(
        self,
        instance: np.ndarray,
        subject_id: str,
        anomaly_score: float | None = None,
        explanation_level: ExplanationLevel = ExplanationLevel.STANDARD,
    ) -> ExplanationReport:
        """
        Generate GDPR-compliant explanation report.

        Args:
            instance: Instance to explain
            subject_id: Data subject identifier
            anomaly_score: Pre-computed anomaly score
            explanation_level: Level of detail

        Returns:
            ExplanationReport for GDPR compliance
        """
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        if anomaly_score is None:
            anomaly_score = float(self._predict(instance)[0])

        is_anomaly = anomaly_score > self._threshold

        return self._gdpr_explainer.explain_decision(
            instance=instance[0],
            decision_value="Anomaly" if is_anomaly else "Normal",
            subject_id=subject_id,
            confidence=anomaly_score,
            decision_category=DecisionCategory.HIGH_IMPACT if is_anomaly else DecisionCategory.STANDARD,
            explanation_level=explanation_level,
            include_counterfactuals=True,
        )

    def request_human_review(
        self,
        decision_id: str,
        subject_id: str,
        reason: str = "",
    ) -> bool:
        """Request human review of a decision."""
        return self._gdpr_explainer.request_human_review(
            decision_id,
            subject_id,
            reason,
        )

    def get_compliance_report(self) -> dict[str, Any]:
        """Get GDPR compliance report."""
        return self._gdpr_explainer.generate_compliance_report()

    @property
    def threshold(self) -> float:
        """Get anomaly threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set anomaly threshold."""
        self._threshold = value

    @property
    def feature_names(self) -> list[str]:
        """Get feature names."""
        return self._feature_names.copy()
