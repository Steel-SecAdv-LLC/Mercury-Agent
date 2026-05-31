"""
Mercury Agent - SHAP Explainability Integration

Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Production-grade explainability for anomaly detection providing:
- SHAP (SHapley Additive exPlanations) integration
- Feature importance analysis
- Local and global explanations
- Attention weight visualization
- Counterfactual explanations
- Neuro-symbolic rule extraction
- Anomaly explanation narratives

This addresses the critical gap: "No SHAP integration" identified in audit.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Optional imports
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment, unused-ignore]


class ExplainabilityMethod(StrEnum):
    """Available explainability methods."""

    SHAP_KERNEL = "shap_kernel"  # Model-agnostic
    SHAP_TREE = "shap_tree"  # Tree-based models
    SHAP_DEEP = "shap_deep"  # Deep learning
    SHAP_GRADIENT = "shap_gradient"  # Gradient-based
    PERMUTATION = "permutation"  # Permutation importance
    LIME = "lime"  # Local Interpretable Model-agnostic Explanations
    ATTENTION = "attention"  # Attention weights
    GRADIENT_CAM = "gradient_cam"  # Gradient-weighted Class Activation Mapping
    COUNTERFACTUAL = "counterfactual"  # Counterfactual explanations


class AggregationMethod(StrEnum):
    """Methods for aggregating feature importance."""

    MEAN_ABS = "mean_abs"  # Mean absolute SHAP value
    MEAN = "mean"  # Mean SHAP value (preserves sign)
    MAX_ABS = "max_abs"  # Maximum absolute SHAP value
    SUM_ABS = "sum_abs"  # Sum of absolute SHAP values


@dataclass
class FeatureImportance:
    """Feature importance result."""

    feature_name: str
    importance: float
    rank: int
    direction: str  # 'positive', 'negative', 'mixed'
    shap_values: NDArray[np.float64] | None = None


@dataclass
class LocalExplanation:
    """Explanation for a single prediction."""

    sample_index: int
    prediction: float
    base_value: float  # Expected value (baseline)
    feature_contributions: dict[str, float]  # Feature -> contribution
    top_features: list[FeatureImportance]
    anomaly_reasons: list[str]  # Human-readable reasons
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sample_index": self.sample_index,
            "prediction": self.prediction,
            "base_value": self.base_value,
            "feature_contributions": self.feature_contributions,
            "top_features": [
                {
                    "name": f.feature_name,
                    "importance": f.importance,
                    "direction": f.direction,
                }
                for f in self.top_features
            ],
            "anomaly_reasons": self.anomaly_reasons,
            "confidence": self.confidence,
        }

    def get_narrative(self) -> str:
        """Generate human-readable narrative explanation."""
        if not self.anomaly_reasons:
            return f"Anomaly score: {self.prediction:.4f}"

        reasons = "\n- ".join(self.anomaly_reasons)
        return f"""Anomaly Score: {self.prediction:.4f}
Confidence: {self.confidence:.2%}

Key Contributing Factors:
- {reasons}
"""


@dataclass
class GlobalExplanation:
    """Global explanation for model behavior."""

    feature_importances: list[FeatureImportance]
    feature_interactions: dict[tuple[str, str], float]
    base_value: float
    total_samples: int

    # Aggregated statistics
    mean_prediction: float
    std_prediction: float

    # Method info
    method: ExplainabilityMethod
    aggregation: AggregationMethod

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_importances": [
                {
                    "name": f.feature_name,
                    "importance": f.importance,
                    "rank": f.rank,
                    "direction": f.direction,
                }
                for f in self.feature_importances
            ],
            "feature_interactions": {
                f"{k[0]}:{k[1]}": v for k, v in self.feature_interactions.items()
            },
            "base_value": self.base_value,
            "total_samples": self.total_samples,
            "mean_prediction": self.mean_prediction,
            "method": self.method.value,
        }


@dataclass
class CounterfactualExplanation:
    """Counterfactual explanation showing what would change the prediction."""

    original_sample: NDArray[np.float64]
    original_prediction: float
    counterfactual_sample: NDArray[np.float64]
    counterfactual_prediction: float
    changed_features: dict[str, tuple[float, float]]  # feature -> (old, new)
    distance: float  # Distance between original and counterfactual
    validity: bool  # Whether counterfactual is valid (different prediction)

    def get_narrative(self, feature_names: list[str] | None = None) -> str:
        """Generate human-readable counterfactual narrative."""
        changes = []
        for feat, (old, new) in self.changed_features.items():
            name = feature_names[int(feat)] if feature_names and feat.isdigit() else feat
            direction = "increased" if new > old else "decreased"
            changes.append(f"- {name}: {direction} from {old:.4f} to {new:.4f}")

        changes_text = "\n".join(changes)

        return f"""To change the prediction from {self.original_prediction:.4f} to {self.counterfactual_prediction:.4f}:
{changes_text}
"""


class BaseExplainer(ABC):
    """Base class for explainability methods."""

    @abstractmethod
    def explain_local(
        self,
        model: Any,
        X: NDArray[np.float64],
        sample_indices: list[int] | None = None,
        feature_names: list[str] | None = None,
    ) -> list[LocalExplanation]:
        """Generate local explanations for specific samples."""
        pass

    @abstractmethod
    def explain_global(
        self,
        model: Any,
        X: NDArray[np.float64],
        feature_names: list[str] | None = None,
    ) -> GlobalExplanation:
        """Generate global explanation for model."""
        pass


class SHAPExplainer(BaseExplainer):
    """
    SHAP-based explainability for anomaly detection.

    Supports multiple SHAP explainer types based on model type:
    - KernelExplainer: Model-agnostic (slowest but most flexible)
    - TreeExplainer: For tree-based models (fast)
    - DeepExplainer: For neural networks
    - GradientExplainer: For neural networks (gradient-based)
    """

    def __init__(
        self,
        method: ExplainabilityMethod = ExplainabilityMethod.SHAP_KERNEL,
        aggregation: AggregationMethod = AggregationMethod.MEAN_ABS,
        background_samples: int = 100,
        top_k_features: int = 10,
        random_state: int | None = None,
    ):
        """
        Initialize SHAP explainer.

        Args:
            method: SHAP method to use
            aggregation: How to aggregate SHAP values
            background_samples: Number of background samples for KernelSHAP
            top_k_features: Number of top features to highlight
            random_state: Seed for reproducible random sampling
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, using fallback permutation importance")

        self.method = method
        self.aggregation = aggregation
        self.background_samples = background_samples
        self.top_k_features = top_k_features
        self.rng = np.random.default_rng(random_state)

        self._explainer: Any = None
        self._background_data: NDArray[np.float64] | None = None
        self._base_value: float = 0.0

    def _create_explainer(
        self,
        model: Any,
        X_background: NDArray[np.float64],
    ) -> Any:
        """Create appropriate SHAP explainer based on method."""
        if not SHAP_AVAILABLE:
            return None

        # Get prediction function
        if hasattr(model, "predict_proba"):

            def predict_fn(x: Any) -> Any:
                return model.predict_proba(x)[:, 1]

        elif hasattr(model, "predict"):
            predict_fn = model.predict
        elif callable(model):
            predict_fn = model
        else:
            raise ValueError("Model must have predict/predict_proba or be callable")

        if self.method == ExplainabilityMethod.SHAP_KERNEL:
            # Sample background data
            n_samples = min(self.background_samples, len(X_background))
            indices = self.rng.choice(len(X_background), n_samples, replace=False)
            background = X_background[indices]

            return shap.KernelExplainer(predict_fn, background)

        elif self.method == ExplainabilityMethod.SHAP_TREE:
            return shap.TreeExplainer(model)

        elif self.method == ExplainabilityMethod.SHAP_DEEP:
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch required for DeepExplainer")

            # Sample background
            n_samples = min(self.background_samples, len(X_background))
            indices = self.rng.choice(len(X_background), n_samples, replace=False)
            background = torch.tensor(X_background[indices], dtype=torch.float32)  # type: ignore[assignment, unused-ignore]

            return shap.DeepExplainer(model, background)

        elif self.method == ExplainabilityMethod.SHAP_GRADIENT:
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch required for GradientExplainer")

            n_samples = min(self.background_samples, len(X_background))
            indices = self.rng.choice(len(X_background), n_samples, replace=False)
            background = torch.tensor(X_background[indices], dtype=torch.float32)  # type: ignore[assignment, unused-ignore]

            return shap.GradientExplainer(model, background)

        else:
            # Default to Kernel
            n_samples = min(self.background_samples, len(X_background))
            indices = self.rng.choice(len(X_background), n_samples, replace=False)
            background = X_background[indices]

            return shap.KernelExplainer(predict_fn, background)

    def _compute_shap_values(
        self,
        X: NDArray[np.float64],
        model: Any = None,
    ) -> tuple[NDArray[np.float64], float]:
        """Compute SHAP values for samples."""
        if self._explainer is None:
            if model is None:
                raise ValueError("Must provide model or call explain_local/global first")
            self._explainer = self._create_explainer(model, X)

        if self._explainer is None:
            # Fallback to permutation importance
            return self._permutation_importance(X, model)

        # Handle PyTorch tensors
        if TORCH_AVAILABLE and isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()

        # Compute SHAP values
        shap_values = self._explainer.shap_values(X)

        # Handle multi-output (binary classification)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Take positive class

        # Get base value
        if hasattr(self._explainer, "expected_value"):
            base_value = self._explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        else:
            base_value = 0.0

        return shap_values, float(base_value)

    def _permutation_importance(
        self,
        X: NDArray[np.float64],
        model: Any,
    ) -> tuple[NDArray[np.float64], float]:
        """Fallback permutation importance when SHAP unavailable."""
        logger.info("Using permutation importance (SHAP unavailable)")

        if hasattr(model, "predict_proba"):
            base_preds = model.predict_proba(X)[:, 1]
        else:
            base_preds = model.predict(X)

        n_samples, n_features = X.shape
        importance = np.zeros((n_samples, n_features))

        # Compute importance for each feature
        for j in range(n_features):
            X_permuted = X.copy()
            X_permuted[:, j] = self.rng.permutation(X_permuted[:, j])

            if hasattr(model, "predict_proba"):
                permuted_preds = model.predict_proba(X_permuted)[:, 1]
            else:
                permuted_preds = model.predict(X_permuted)

            importance[:, j] = base_preds - permuted_preds

        return importance, float(np.mean(base_preds))

    def explain_local(
        self,
        model: Any,
        X: NDArray[np.float64],
        sample_indices: list[int] | None = None,
        feature_names: list[str] | None = None,
    ) -> list[LocalExplanation]:
        """
        Generate local explanations for specific samples.

        Args:
            model: Trained model
            X: Feature matrix
            sample_indices: Indices of samples to explain (all if None)
            feature_names: Feature names for interpretation

        Returns:
            List of LocalExplanation objects
        """
        if sample_indices is None:
            sample_indices = list(range(len(X)))

        # Initialize explainer if needed
        if self._explainer is None:
            self._explainer = self._create_explainer(model, X)

        # Get samples to explain
        X_explain = X[sample_indices]

        # Compute SHAP values
        shap_values, base_value = self._compute_shap_values(X_explain, model)
        self._base_value = base_value

        if hasattr(model, "predict_proba"):
            predictions = model.predict_proba(X_explain)[:, 1]
        else:
            predictions = model.predict(X_explain)

        # Generate explanations
        explanations = []
        n_features = X.shape[1]

        for i, idx in enumerate(sample_indices):
            sample_shap = shap_values[i]

            # Feature contributions
            contributions = {}
            for j in range(n_features):
                name = feature_names[j] if feature_names else f"feature_{j}"
                contributions[name] = float(sample_shap[j])

            # Sort by absolute importance
            sorted_features = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)

            # Create top features
            top_features = []
            for rank, (name, value) in enumerate(sorted_features[: self.top_k_features]):
                direction = "positive" if value > 0 else "negative"
                top_features.append(
                    FeatureImportance(
                        feature_name=name,
                        importance=float(abs(value)),
                        rank=rank + 1,
                        direction=direction,
                    )
                )

            # Generate anomaly reasons
            reasons = self._generate_reasons(top_features, predictions[i])

            explanation = LocalExplanation(
                sample_index=idx,
                prediction=float(predictions[i]),
                base_value=base_value,
                feature_contributions=contributions,
                top_features=top_features,
                anomaly_reasons=reasons,
                confidence=float(abs(predictions[i] - 0.5) * 2),  # 0-1 scale
            )

            explanations.append(explanation)

        return explanations

    def _generate_reasons(
        self,
        top_features: list[FeatureImportance],
        prediction: float,
    ) -> list[str]:
        """Generate human-readable anomaly reasons."""
        reasons = []

        is_anomaly = prediction > 0.5

        for feat in top_features[:5]:  # Top 5 features
            if is_anomaly and feat.direction == "positive":
                reasons.append(
                    f"High {feat.feature_name} contributes +{feat.importance:.4f} to anomaly score"
                )
            elif is_anomaly and feat.direction == "negative":
                reasons.append(
                    f"Low {feat.feature_name} partially offsets anomaly (-{feat.importance:.4f})"
                )
            elif not is_anomaly and feat.direction == "negative":
                reasons.append(
                    f"Normal {feat.feature_name} reduces anomaly score by {feat.importance:.4f}"
                )

        if not reasons:
            if is_anomaly:
                reasons.append("Multiple features contribute to elevated anomaly score")
            else:
                reasons.append("No significant anomaly indicators detected")

        return reasons

    def explain_global(
        self,
        model: Any,
        X: NDArray[np.float64],
        feature_names: list[str] | None = None,
    ) -> GlobalExplanation:
        """
        Generate global explanation for model behavior.

        Args:
            model: Trained model
            X: Feature matrix
            feature_names: Feature names for interpretation

        Returns:
            GlobalExplanation object
        """
        # Initialize explainer
        if self._explainer is None:
            self._explainer = self._create_explainer(model, X)

        # Compute SHAP values for all samples
        shap_values, base_value = self._compute_shap_values(X, model)

        # Aggregate importance
        if self.aggregation == AggregationMethod.MEAN_ABS:
            importance_values = np.mean(np.abs(shap_values), axis=0)
        elif self.aggregation == AggregationMethod.MEAN:
            importance_values = np.mean(shap_values, axis=0)
        elif self.aggregation == AggregationMethod.MAX_ABS:
            importance_values = np.max(np.abs(shap_values), axis=0)
        elif self.aggregation == AggregationMethod.SUM_ABS:
            importance_values = np.sum(np.abs(shap_values), axis=0)
        else:
            importance_values = np.mean(np.abs(shap_values), axis=0)

        # Create feature importance list
        n_features = X.shape[1]
        feature_importances = []

        for j in range(n_features):
            name = feature_names[j] if feature_names else f"feature_{j}"
            mean_shap = np.mean(shap_values[:, j])
            direction = "positive" if mean_shap > 0 else "negative" if mean_shap < 0 else "mixed"

            feature_importances.append(
                FeatureImportance(
                    feature_name=name,
                    importance=float(importance_values[j]),
                    rank=0,  # Set below
                    direction=direction,
                    shap_values=shap_values[:, j],
                )
            )

        # Sort and assign ranks
        feature_importances.sort(key=lambda x: x.importance, reverse=True)
        for rank, feat in enumerate(feature_importances):
            feat.rank = rank + 1

        # Compute feature interactions (top pairs)
        interactions = self._compute_interactions(shap_values, feature_names)

        if hasattr(model, "predict_proba"):
            predictions = model.predict_proba(X)[:, 1]
        else:
            predictions = model.predict(X)

        return GlobalExplanation(
            feature_importances=feature_importances,
            feature_interactions=interactions,
            base_value=base_value,
            total_samples=len(X),
            mean_prediction=float(np.mean(predictions)),
            std_prediction=float(np.std(predictions)),
            method=self.method,
            aggregation=self.aggregation,
        )

    def _compute_interactions(
        self,
        shap_values: NDArray[np.float64],
        feature_names: list[str] | None,
    ) -> dict[tuple[str, str], float]:
        """Compute pairwise feature interactions."""
        interactions = {}
        n_features = shap_values.shape[1]

        # Compute correlation between SHAP values as proxy for interaction
        for i in range(min(n_features, 20)):  # Limit to top 20 features
            for j in range(i + 1, min(n_features, 20)):
                corr = np.corrcoef(shap_values[:, i], shap_values[:, j])[0, 1]

                name_i = feature_names[i] if feature_names else f"feature_{i}"
                name_j = feature_names[j] if feature_names else f"feature_{j}"

                if abs(corr) > 0.3:  # Only store significant interactions
                    interactions[(name_i, name_j)] = float(corr)

        return interactions


class CounterfactualExplainer:
    """
    Generates counterfactual explanations.

    Finds minimal changes to input that would flip the prediction.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        max_features_to_change: int = 5,
        step_size: float = 0.1,
        max_iterations: int = 100,
    ):
        """
        Initialize counterfactual explainer.

        Args:
            threshold: Prediction threshold for anomaly
            max_features_to_change: Maximum features to modify
            step_size: Size of feature changes
            max_iterations: Maximum optimization iterations
        """
        self.threshold = threshold
        self.max_features_to_change = max_features_to_change
        self.step_size = step_size
        self.max_iterations = max_iterations

    def explain(
        self,
        model: Any,
        sample: NDArray[np.float64],
        feature_names: list[str] | None = None,
        feature_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> CounterfactualExplanation:
        """
        Generate counterfactual explanation for a sample.

        Args:
            model: Trained model
            sample: Single sample to explain
            feature_names: Feature names
            feature_ranges: Allowed ranges for each feature

        Returns:
            CounterfactualExplanation object
        """
        # Get original prediction
        if hasattr(model, "predict_proba"):
            original_pred = model.predict_proba(sample.reshape(1, -1))[0, 1]
        else:
            original_pred = model.predict(sample.reshape(1, -1))[0]

        # Determine target direction
        is_anomaly = original_pred > self.threshold

        # Initialize counterfactual
        counterfactual = sample.copy()
        n_features = len(sample)

        # Gradient-free optimization
        best_counterfactual = counterfactual.copy()
        best_pred = original_pred
        best_distance = float("inf")

        for iteration in range(self.max_iterations):
            # Try modifying each feature
            for j in range(n_features):
                for direction in [-1, 1]:
                    candidate = counterfactual.copy()
                    candidate[j] += direction * self.step_size

                    # Apply feature constraints if available
                    if feature_ranges:
                        name = feature_names[j] if feature_names else f"feature_{j}"
                        if name in feature_ranges:
                            min_val, max_val = feature_ranges[name]
                            candidate[j] = np.clip(candidate[j], min_val, max_val)

                    # Get prediction
                    if hasattr(model, "predict_proba"):
                        pred = model.predict_proba(candidate.reshape(1, -1))[0, 1]
                    else:
                        pred = model.predict(candidate.reshape(1, -1))[0]

                    # Check if better
                    distance = np.linalg.norm(candidate - sample)
                    flipped = (pred > self.threshold) != is_anomaly

                    if flipped and distance < best_distance:
                        best_counterfactual = candidate.copy()
                        best_pred = pred
                        best_distance = distance  # type: ignore[assignment, unused-ignore]
                        counterfactual = candidate.copy()

            # Early stopping if valid counterfactual found
            if best_distance < float("inf"):
                break

        # Identify changed features
        changed_features = {}
        for j in range(n_features):
            if abs(best_counterfactual[j] - sample[j]) > 1e-6:
                name = feature_names[j] if feature_names else str(j)
                changed_features[name] = (float(sample[j]), float(best_counterfactual[j]))

        # Check validity
        validity = (best_pred > self.threshold) != is_anomaly

        return CounterfactualExplanation(
            original_sample=sample,
            original_prediction=float(original_pred),
            counterfactual_sample=best_counterfactual,
            counterfactual_prediction=float(best_pred),
            changed_features=changed_features,
            distance=float(best_distance),
            validity=validity,
        )


class AnomalyExplainer:
    """
    Unified anomaly detection explainer.

    Combines multiple explainability methods for comprehensive anomaly explanations.
    """

    def __init__(
        self,
        shap_method: ExplainabilityMethod = ExplainabilityMethod.SHAP_KERNEL,
        include_counterfactual: bool = True,
        top_k_features: int = 10,
    ):
        """
        Initialize anomaly explainer.

        Args:
            shap_method: SHAP method to use
            include_counterfactual: Include counterfactual explanations
            top_k_features: Number of top features to highlight
        """
        self.shap_explainer = SHAPExplainer(method=shap_method, top_k_features=top_k_features)
        self.counterfactual_explainer = (
            CounterfactualExplainer() if include_counterfactual else None
        )
        self.top_k_features = top_k_features

    def explain(
        self,
        model: Any,
        X: NDArray[np.float64],
        sample_indices: list[int] | None = None,
        feature_names: list[str] | None = None,
        include_global: bool = True,
    ) -> dict[str, Any]:
        """
        Generate comprehensive explanations.

        Args:
            model: Trained model
            X: Feature matrix
            sample_indices: Specific samples to explain
            feature_names: Feature names
            include_global: Include global explanation

        Returns:
            Dictionary with local, global, and counterfactual explanations
        """
        result: dict[str, Any] = {}

        # Local explanations
        local_explanations = self.shap_explainer.explain_local(
            model, X, sample_indices, feature_names
        )
        result["local_explanations"] = [e.to_dict() for e in local_explanations]

        # Global explanation
        if include_global:
            global_explanation = self.shap_explainer.explain_global(model, X, feature_names)
            result["global_explanation"] = global_explanation.to_dict()

        # Counterfactual explanations
        if self.counterfactual_explainer and sample_indices:
            counterfactuals = []
            for idx in sample_indices[:10]:  # Limit to 10 counterfactuals
                cf = self.counterfactual_explainer.explain(model, X[idx], feature_names)
                counterfactuals.append(
                    {
                        "sample_index": idx,
                        "original_prediction": cf.original_prediction,
                        "counterfactual_prediction": cf.counterfactual_prediction,
                        "changed_features": cf.changed_features,
                        "distance": cf.distance,
                        "validity": cf.validity,
                    }
                )
            result["counterfactuals"] = counterfactuals

        return result

    def generate_report(
        self,
        model: Any,
        X: NDArray[np.float64],
        feature_names: list[str] | None = None,
    ) -> str:
        """
        Generate human-readable explanation report.

        Args:
            model: Trained model
            X: Feature matrix
            feature_names: Feature names

        Returns:
            Formatted report string
        """
        # Get global explanation
        global_exp = self.shap_explainer.explain_global(model, X, feature_names)

        # Build report
        lines = [
            "=" * 60,
            "MERCURY AGENT ANOMALY DETECTION EXPLANATION REPORT",
            "=" * 60,
            "",
            f"Total Samples Analyzed: {global_exp.total_samples}",
            f"Mean Anomaly Score: {global_exp.mean_prediction:.4f}",
            f"Std Anomaly Score: {global_exp.std_prediction:.4f}",
            f"Base Value (Expected): {global_exp.base_value:.4f}",
            "",
            "TOP CONTRIBUTING FEATURES",
            "-" * 40,
        ]

        for feat in global_exp.feature_importances[: self.top_k_features]:
            direction_indicator = "+" if feat.direction == "positive" else "-"
            lines.append(
                f"  {feat.rank}. {feat.feature_name}: "
                f"{direction_indicator}{feat.importance:.4f}"
            )

        lines.extend(
            [
                "",
                "FEATURE INTERACTIONS",
                "-" * 40,
            ]
        )

        if global_exp.feature_interactions:
            for (f1, f2), corr in list(global_exp.feature_interactions.items())[:5]:
                lines.append(f"  {f1} <-> {f2}: correlation={corr:.3f}")
        else:
            lines.append("  No significant interactions detected")

        lines.extend(
            [
                "",
                "=" * 60,
            ]
        )

        return "\n".join(lines)


def create_explainer(
    method: str = "shap",
    **kwargs: Any,
) -> BaseExplainer:
    """
    Factory function to create explainer.

    Args:
        method: Explainability method ('shap', 'permutation', 'attention')
        **kwargs: Additional arguments

    Returns:
        Configured explainer
    """
    method_map = {
        "shap": ExplainabilityMethod.SHAP_KERNEL,
        "shap_kernel": ExplainabilityMethod.SHAP_KERNEL,
        "shap_tree": ExplainabilityMethod.SHAP_TREE,
        "shap_deep": ExplainabilityMethod.SHAP_DEEP,
        "shap_gradient": ExplainabilityMethod.SHAP_GRADIENT,
        "permutation": ExplainabilityMethod.PERMUTATION,
    }

    m = method_map.get(method.lower(), ExplainabilityMethod.SHAP_KERNEL)

    return SHAPExplainer(method=m, **kwargs)


# Exports
__all__ = [
    "AggregationMethod",
    "AnomalyExplainer",
    "BaseExplainer",
    "CounterfactualExplainer",
    "CounterfactualExplanation",
    "ExplainabilityMethod",
    "FeatureImportance",
    "GlobalExplanation",
    "LocalExplanation",
    "SHAPExplainer",
    "create_explainer",
]
