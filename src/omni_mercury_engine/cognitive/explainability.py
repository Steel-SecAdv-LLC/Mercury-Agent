"""
Mercury Agent - Explainability Module with SHAP/LIME Integration

Production-grade explainability for anomaly detection models.
Provides feature importance, counterfactual explanations, and faithfulness metrics.

Features:
- SHAP (SHapley Additive exPlanations) integration
- LIME (Local Interpretable Model-agnostic Explanations) integration
- Custom faithfulness and consistency metrics
- Counterfactual explanation generation
- Attention-based explanation extraction
- Human-readable explanation generation

Research References:
- SHAP: Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"
- LIME: Ribeiro et al. (2016) "Why Should I Trust You?"
- Integrated Gradients: Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks"
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable

    import lime as lime_module  # noqa: F401
    import lime.lime_tabular as lime_tabular_module  # noqa: F401
    import shap as shap_module  # noqa: F401

import numpy as np


logger = logging.getLogger(__name__)

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

# Optional SHAP import
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

# Optional LIME import
try:
    import lime
    import lime.lime_tabular

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    lime = None


class ExplanationType(Enum):
    """Types of explanations."""

    SHAP = "shap"
    LIME = "lime"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    ATTENTION = "attention"
    COUNTERFACTUAL = "counterfactual"
    RULE_BASED = "rule_based"


class FaithfulnessMetric(Enum):
    """Metrics for measuring explanation faithfulness."""

    COMPREHENSIVENESS = "comprehensiveness"
    SUFFICIENCY = "sufficiency"
    MONOTONICITY = "monotonicity"
    FAITHFULNESS_CORRELATION = "faithfulness_correlation"


@dataclass
class FeatureImportance:
    """Feature importance score."""

    feature_name: str
    feature_index: int
    importance: float
    direction: str  # "positive", "negative", "neutral"
    confidence: float = 1.0


@dataclass
class Explanation:
    """Explanation for a prediction."""

    explanation_id: str
    explanation_type: ExplanationType
    prediction: float
    feature_importances: list[FeatureImportance]
    base_value: float
    local_accuracy: float
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    human_readable: str = ""
    counterfactuals: list[dict[str, Any]] = field(default_factory=list)
    faithfulness_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "explanation_id": self.explanation_id,
            "explanation_type": self.explanation_type.value,
            "prediction": self.prediction,
            "base_value": self.base_value,
            "local_accuracy": self.local_accuracy,
            "feature_importances": [
                {
                    "feature": fi.feature_name,
                    "importance": fi.importance,
                    "direction": fi.direction,
                    "confidence": fi.confidence,
                }
                for fi in self.feature_importances
            ],
            "human_readable": self.human_readable,
            "counterfactuals": self.counterfactuals,
            "faithfulness_scores": self.faithfulness_scores,
        }


class BaseExplainer(ABC):
    """Abstract base class for explainers."""

    @abstractmethod
    def explain(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        instance: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> Explanation:
        """Generate explanation for an instance."""
        pass


class SHAPExplainer(BaseExplainer):
    """
    SHAP-based explainer for anomaly detection models.

    Uses Kernel SHAP for model-agnostic explanations with
    game-theoretic feature attribution.
    """

    def __init__(
        self,
        background_data: np.ndarray | None = None,
        n_samples: int = 100,
        link: str = "identity",
    ):
        """
        Initialize SHAP explainer.

        Args:
            background_data: Background dataset for SHAP values
            n_samples: Number of samples for Kernel SHAP
            link: Link function ("identity" or "logit")
        """
        self.background_data = background_data
        self.n_samples = n_samples
        self.link = link
        self._explainer: Any = None

    def _create_explainer(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        background: np.ndarray,
    ) -> Any:
        """Create SHAP explainer instance."""
        if not SHAP_AVAILABLE:
            return None

        return shap.KernelExplainer(model, background)

    def explain(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        instance: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> Explanation:
        """Generate SHAP explanation."""
        import uuid

        instance = np.atleast_2d(instance)
        n_features = instance.shape[1]

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        if SHAP_AVAILABLE and self.background_data is not None:
            if self._explainer is None:
                self._explainer = self._create_explainer(model, self.background_data)

            shap_values = self._explainer.shap_values(instance, nsamples=self.n_samples)
            base_value = self._explainer.expected_value

            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            shap_values = shap_values.flatten()

            if isinstance(base_value, (list, np.ndarray)):
                base_value = float(base_value[0])

        else:
            shap_values = self._approximate_shap(model, instance)
            base_value = float(model(instance.mean(axis=0, keepdims=True))[0])

        prediction = float(model(instance)[0])

        feature_importances = []
        for i, (name, value) in enumerate(zip(feature_names, shap_values)):
            direction = "positive" if value > 0 else ("negative" if value < 0 else "neutral")
            feature_importances.append(
                FeatureImportance(
                    feature_name=name,
                    feature_index=i,
                    importance=abs(float(value)),
                    direction=direction,
                )
            )

        feature_importances.sort(key=lambda x: x.importance, reverse=True)

        reconstructed = base_value + sum(shap_values)
        local_accuracy = 1.0 - abs(prediction - reconstructed) / (abs(prediction) + 1e-8)

        human_readable = self._generate_human_readable(
            feature_importances[:5], prediction, base_value
        )

        return Explanation(
            explanation_id=str(uuid.uuid4()),
            explanation_type=ExplanationType.SHAP,
            prediction=prediction,
            feature_importances=feature_importances,
            base_value=base_value,
            local_accuracy=float(local_accuracy),
            human_readable=human_readable,
            metadata={"n_samples": self.n_samples, "method": "kernel_shap"},
        )

    def _approximate_shap(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        instance: np.ndarray,
    ) -> np.ndarray:
        """Approximate SHAP values without SHAP library."""
        instance = instance.flatten()
        n_features = len(instance)
        shap_values = np.zeros(n_features)

        baseline = np.zeros_like(instance)
        n_samples = min(self.n_samples, 2**n_features)

        for _ in range(n_samples):
            coalition = np.random.randint(0, 2, n_features)

            with_feature = baseline.copy()
            without_feature = baseline.copy()

            for j in range(n_features):
                if coalition[j] == 1:
                    with_feature[j] = instance[j]
                    without_feature[j] = instance[j]

            for j in range(n_features):
                temp_with = with_feature.copy()
                temp_without = without_feature.copy()

                if coalition[j] == 0:
                    temp_with[j] = instance[j]
                else:
                    temp_without[j] = baseline[j]

                pred_with = model(temp_with.reshape(1, -1))[0]
                pred_without = model(temp_without.reshape(1, -1))[0]

                shap_values[j] += (pred_with - pred_without) / n_samples

        return shap_values

    def _generate_human_readable(
        self,
        top_features: list[FeatureImportance],
        prediction: float,
        base_value: float,
    ) -> str:
        """Generate human-readable explanation."""
        lines = [f"Prediction: {prediction:.4f} (baseline: {base_value:.4f})"]
        lines.append("Top contributing features:")

        for fi in top_features:
            arrow = "↑" if fi.direction == "positive" else "↓"
            lines.append(f"  {arrow} {fi.feature_name}: {fi.importance:.4f} ({fi.direction})")

        return "\n".join(lines)


class LIMEExplainer(BaseExplainer):
    """
    LIME-based explainer for anomaly detection models.

    Provides local interpretable explanations using
    surrogate linear models.
    """

    def __init__(
        self,
        training_data: np.ndarray | None = None,
        mode: str = "regression",
        n_samples: int = 5000,
        kernel_width: float | None = None,
    ):
        """
        Initialize LIME explainer.

        Args:
            training_data: Training data for discretization
            mode: "regression" or "classification"
            n_samples: Number of samples for local surrogate
            kernel_width: Kernel width for sample weighting
        """
        self.training_data = training_data
        self.mode = mode
        self.n_samples = n_samples
        self.kernel_width = kernel_width
        self._explainer: Any = None

    def _create_explainer(
        self,
        training_data: np.ndarray,
        feature_names: list[str],
    ) -> Any:
        """Create LIME explainer instance."""
        if not LIME_AVAILABLE:
            return None

        return lime.lime_tabular.LimeTabularExplainer(
            training_data,
            feature_names=feature_names,
            mode=self.mode,
            kernel_width=self.kernel_width,
        )

    def explain(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        instance: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> Explanation:
        """Generate LIME explanation."""
        import uuid

        instance = np.atleast_1d(instance.flatten())
        n_features = len(instance)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        if LIME_AVAILABLE and self.training_data is not None:
            if self._explainer is None:
                self._explainer = self._create_explainer(self.training_data, feature_names)

            exp = self._explainer.explain_instance(
                instance,
                model,
                num_samples=self.n_samples,
            )

            lime_weights = dict(exp.as_list())
            local_accuracy = exp.score

        else:
            lime_weights, local_accuracy = self._approximate_lime(model, instance, feature_names)

        prediction = float(model(instance.reshape(1, -1))[0])

        feature_importances = []
        for i, name in enumerate(feature_names):
            weight = lime_weights.get(name, 0.0)
            if isinstance(weight, (tuple, list)):
                weight = weight[1] if len(weight) > 1 else weight[0]
            direction = "positive" if weight > 0 else ("negative" if weight < 0 else "neutral")
            feature_importances.append(
                FeatureImportance(
                    feature_name=name,
                    feature_index=i,
                    importance=abs(float(weight)),
                    direction=direction,
                )
            )

        feature_importances.sort(key=lambda x: x.importance, reverse=True)

        human_readable = self._generate_human_readable(feature_importances[:5], prediction)

        base_value = prediction - sum(
            fi.importance * (1 if fi.direction == "positive" else -1) for fi in feature_importances
        )

        return Explanation(
            explanation_id=str(uuid.uuid4()),
            explanation_type=ExplanationType.LIME,
            prediction=prediction,
            feature_importances=feature_importances,
            base_value=float(base_value),
            local_accuracy=float(local_accuracy),
            human_readable=human_readable,
            metadata={"n_samples": self.n_samples, "method": "lime"},
        )

    def _approximate_lime(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        instance: np.ndarray,
        feature_names: list[str],
    ) -> tuple[dict[str, float], float]:
        """Approximate LIME without LIME library."""
        n_features = len(instance)

        samples = np.random.randn(self.n_samples, n_features) * 0.1 + instance
        predictions = model(samples).flatten()

        distances = np.sqrt(np.sum((samples - instance) ** 2, axis=1))
        kernel_width = self.kernel_width or np.sqrt(n_features) * 0.75
        weights = np.exp(-(distances**2) / (kernel_width**2))

        samples_weighted = samples * np.sqrt(weights)[:, np.newaxis]
        predictions_weighted = predictions * np.sqrt(weights)

        try:
            coeffs, residuals, _, _ = np.linalg.lstsq(
                samples_weighted, predictions_weighted, rcond=None
            )
        except np.linalg.LinAlgError:
            coeffs = np.zeros(n_features)
            residuals = np.array([1e10])

        lime_weights = {name: float(coeffs[i]) for i, name in enumerate(feature_names)}

        ss_res = (
            residuals[0]
            if len(residuals) > 0
            else np.sum((predictions_weighted - samples_weighted @ coeffs) ** 2)
        )
        ss_tot = np.sum((predictions_weighted - np.mean(predictions_weighted)) ** 2) + 1e-8
        r2 = 1 - ss_res / ss_tot

        return lime_weights, float(max(0, r2))

    def _generate_human_readable(
        self,
        top_features: list[FeatureImportance],
        prediction: float,
    ) -> str:
        """Generate human-readable explanation."""
        lines = [f"Prediction: {prediction:.4f}"]
        lines.append("Key factors (local linear approximation):")

        for fi in top_features:
            sign = "+" if fi.direction == "positive" else "-"
            lines.append(f"  {sign} {fi.feature_name}: coefficient {fi.importance:.4f}")

        return "\n".join(lines)


class IntegratedGradientsExplainer(BaseExplainer):
    """
    Integrated Gradients explainer for differentiable models.

    Provides theoretically grounded attributions using
    path integrals from baseline to input.
    """

    def __init__(
        self,
        baseline: np.ndarray | None = None,
        n_steps: int = 50,
    ):
        """
        Initialize Integrated Gradients explainer.

        Args:
            baseline: Baseline input (defaults to zeros)
            n_steps: Number of interpolation steps
        """
        self.baseline = baseline
        self.n_steps = n_steps

    def explain(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        instance: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> Explanation:
        """Generate Integrated Gradients explanation."""
        import uuid

        instance = np.atleast_2d(instance)
        n_features = instance.shape[1]

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        baseline = self.baseline if self.baseline is not None else np.zeros_like(instance)
        baseline = np.atleast_2d(baseline)

        if TORCH_AVAILABLE:
            attributions = self._compute_ig_torch(model, instance, baseline)
        else:
            attributions = self._compute_ig_finite_diff(model, instance, baseline)

        attributions = attributions.flatten()
        prediction = float(model(instance)[0])
        baseline_pred = float(model(baseline)[0])

        feature_importances = []
        for i, (name, attr) in enumerate(zip(feature_names, attributions)):
            direction = "positive" if attr > 0 else ("negative" if attr < 0 else "neutral")
            feature_importances.append(
                FeatureImportance(
                    feature_name=name,
                    feature_index=i,
                    importance=abs(float(attr)),
                    direction=direction,
                )
            )

        feature_importances.sort(key=lambda x: x.importance, reverse=True)

        completeness_error = abs(sum(attributions) - (prediction - baseline_pred))
        local_accuracy = 1.0 - completeness_error / (abs(prediction - baseline_pred) + 1e-8)

        human_readable = self._generate_human_readable(
            feature_importances[:5], prediction, baseline_pred
        )

        return Explanation(
            explanation_id=str(uuid.uuid4()),
            explanation_type=ExplanationType.INTEGRATED_GRADIENTS,
            prediction=prediction,
            feature_importances=feature_importances,
            base_value=baseline_pred,
            local_accuracy=float(local_accuracy),
            human_readable=human_readable,
            metadata={"n_steps": self.n_steps, "method": "integrated_gradients"},
        )

    def _compute_ig_torch(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        instance: np.ndarray,
        baseline: np.ndarray,
    ) -> np.ndarray:
        """Compute IG using PyTorch gradients."""
        instance_t = torch.tensor(instance, dtype=torch.float32, requires_grad=True)
        baseline_t = torch.tensor(baseline, dtype=torch.float32)

        alphas = torch.linspace(0, 1, self.n_steps)
        gradients = []

        for alpha in alphas:
            interpolated = baseline_t + alpha * (instance_t - baseline_t)
            interpolated.requires_grad_(True)

            output = torch.tensor(model(interpolated.detach().numpy()), dtype=torch.float32)

            if output.requires_grad:
                grad = torch.autograd.grad(output.sum(), interpolated)[0]
                gradients.append(grad.detach().numpy())
            else:
                grad = self._numerical_gradient(model, interpolated.detach().numpy())
                gradients.append(grad)

        avg_gradients = np.mean(gradients, axis=0)
        attributions = (instance - baseline) * avg_gradients

        return attributions

    def _compute_ig_finite_diff(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        instance: np.ndarray,
        baseline: np.ndarray,
    ) -> np.ndarray:
        """Compute IG using finite differences."""
        alphas = np.linspace(0, 1, self.n_steps)
        gradients = []

        for alpha in alphas:
            interpolated = baseline + alpha * (instance - baseline)
            grad = self._numerical_gradient(model, interpolated)
            gradients.append(grad)

        avg_gradients = np.mean(gradients, axis=0)
        attributions = (instance - baseline) * avg_gradients

        return attributions

    def _numerical_gradient(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        epsilon: float = 1e-5,
    ) -> np.ndarray:
        """Compute numerical gradient."""
        x = x.flatten()
        grad = np.zeros_like(x)

        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += epsilon
            x_minus[i] -= epsilon

            grad[i] = (model(x_plus.reshape(1, -1))[0] - model(x_minus.reshape(1, -1))[0]) / (
                2 * epsilon
            )

        return grad.reshape(1, -1)

    def _generate_human_readable(
        self,
        top_features: list[FeatureImportance],
        prediction: float,
        baseline_pred: float,
    ) -> str:
        """Generate human-readable explanation."""
        lines = [
            f"Prediction: {prediction:.4f}",
            f"Change from baseline: {prediction - baseline_pred:.4f}",
            "Attributions (integrated gradients):",
        ]

        for fi in top_features:
            arrow = "↑" if fi.direction == "positive" else "↓"
            lines.append(f"  {arrow} {fi.feature_name}: {fi.importance:.4f}")

        return "\n".join(lines)


class CounterfactualExplainer:
    """
    Generate counterfactual explanations.

    Finds minimal changes to input that flip the prediction,
    answering "what would need to change?"
    """

    def __init__(
        self,
        threshold: float = 0.5,
        max_iterations: int = 100,
        step_size: float = 0.1,
        feature_constraints: dict[int, tuple[float, float]] | None = None,
    ):
        """
        Initialize counterfactual explainer.

        Args:
            threshold: Decision threshold for classification
            max_iterations: Maximum optimization iterations
            step_size: Step size for perturbation
            feature_constraints: Min/max constraints per feature
        """
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.feature_constraints = feature_constraints or {}

    def generate_counterfactual(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        instance: np.ndarray,
        target_class: int = 0,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a counterfactual explanation.

        Args:
            model: Prediction function
            instance: Original instance
            target_class: Target class for counterfactual
            feature_names: Feature names

        Returns:
            Dictionary with counterfactual information
        """
        instance = np.atleast_1d(instance.flatten())
        n_features = len(instance)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        original_pred = float(model(instance.reshape(1, -1))[0])
        original_class = 1 if original_pred > self.threshold else 0
        _target_pred = 1.0 if target_class == 1 else 0.0

        counterfactual = instance.copy()

        for iteration in range(self.max_iterations):
            current_pred = float(model(counterfactual.reshape(1, -1))[0])

            if (target_class == 1 and current_pred > self.threshold) or (
                target_class == 0 and current_pred <= self.threshold
            ):
                break

            gradient = self._estimate_gradient(model, counterfactual)

            if target_class == 1:
                counterfactual += self.step_size * gradient
            else:
                counterfactual -= self.step_size * gradient

            for i, (lo, hi) in self.feature_constraints.items():
                counterfactual[i] = np.clip(counterfactual[i], lo, hi)

        final_pred = float(model(counterfactual.reshape(1, -1))[0])

        changes = []
        for i, (orig, cf) in enumerate(zip(instance, counterfactual)):
            if abs(orig - cf) > 1e-6:
                changes.append(
                    {
                        "feature": feature_names[i],
                        "original_value": float(orig),
                        "counterfactual_value": float(cf),
                        "change": float(cf - orig),
                    }
                )

        return {
            "original": instance.tolist(),
            "counterfactual": counterfactual.tolist(),
            "original_prediction": original_pred,
            "counterfactual_prediction": final_pred,
            "original_class": original_class,
            "counterfactual_class": 1 if final_pred > self.threshold else 0,
            "target_class": target_class,
            "success": (final_pred > self.threshold) == (target_class == 1),
            "changes": changes,
            "total_change": float(np.sum(np.abs(counterfactual - instance))),
            "iterations": iteration + 1,
        }

    def _estimate_gradient(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        epsilon: float = 1e-4,
    ) -> np.ndarray:
        """Estimate gradient numerically."""
        grad = np.zeros_like(x)

        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += epsilon
            x_minus[i] -= epsilon

            grad[i] = (model(x_plus.reshape(1, -1))[0] - model(x_minus.reshape(1, -1))[0]) / (
                2 * epsilon
            )

        return grad


class FaithfulnessEvaluator:
    """
    Evaluate explanation faithfulness using various metrics.

    Metrics:
    - Comprehensiveness: Does removing important features change prediction?
    - Sufficiency: Do important features alone predict well?
    - Monotonicity: Are feature rankings consistent?
    """

    def __init__(
        self,
        n_steps: int = 10,
    ):
        """
        Initialize faithfulness evaluator.

        Args:
            n_steps: Number of removal/addition steps
        """
        self.n_steps = n_steps

    def evaluate(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        instance: np.ndarray,
        explanation: Explanation,
    ) -> dict[str, float]:
        """
        Evaluate explanation faithfulness.

        Args:
            model: Prediction function
            instance: Original instance
            explanation: Explanation to evaluate

        Returns:
            Dictionary of metric scores
        """
        instance = np.atleast_1d(instance.flatten())
        original_pred = float(model(instance.reshape(1, -1))[0])

        ranked_features = sorted(
            explanation.feature_importances,
            key=lambda x: x.importance,
            reverse=True,
        )

        comprehensiveness = self._compute_comprehensiveness(
            model, instance, ranked_features, original_pred
        )

        sufficiency = self._compute_sufficiency(model, instance, ranked_features, original_pred)

        monotonicity = self._compute_monotonicity(model, instance, ranked_features, original_pred)

        return {
            "comprehensiveness": comprehensiveness,
            "sufficiency": sufficiency,
            "monotonicity": monotonicity,
        }

    def _compute_comprehensiveness(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        instance: np.ndarray,
        ranked_features: list[FeatureImportance],
        original_pred: float,
    ) -> float:
        """Compute comprehensiveness (removing important features should change prediction)."""
        changes = []
        modified = instance.copy()
        baseline_value = 0.0

        for fi in ranked_features:
            modified[fi.feature_index] = baseline_value
            new_pred = float(model(modified.reshape(1, -1))[0])
            changes.append(abs(original_pred - new_pred))

        return float(np.mean(changes)) if changes else 0.0

    def _compute_sufficiency(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        instance: np.ndarray,
        ranked_features: list[FeatureImportance],
        original_pred: float,
    ) -> float:
        """Compute sufficiency (important features alone should predict well)."""
        baseline = np.zeros_like(instance)
        changes = []

        for i, fi in enumerate(ranked_features):
            baseline[fi.feature_index] = instance[fi.feature_index]
            new_pred = float(model(baseline.reshape(1, -1))[0])
            error = abs(original_pred - new_pred)
            changes.append(1.0 - error / (abs(original_pred) + 1e-8))

        return float(np.mean(changes)) if changes else 0.0

    def _compute_monotonicity(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        instance: np.ndarray,
        ranked_features: list[FeatureImportance],
        original_pred: float,
    ) -> float:
        """Compute monotonicity (more important features should have larger effect)."""
        if len(ranked_features) < 2:
            return 1.0

        effects = []
        for fi in ranked_features:
            modified = instance.copy()
            modified[fi.feature_index] = 0.0
            effect = abs(original_pred - float(model(modified.reshape(1, -1))[0]))
            effects.append(effect)

        monotonic_pairs = 0
        total_pairs = 0

        for i in range(len(effects) - 1):
            for j in range(i + 1, len(effects)):
                total_pairs += 1
                if effects[i] >= effects[j]:
                    monotonic_pairs += 1

        return monotonic_pairs / total_pairs if total_pairs > 0 else 1.0


class ExplainabilityEngine:
    """
    Unified explainability engine combining multiple explanation methods.

    Provides a single interface for generating and evaluating explanations
    using SHAP, LIME, Integrated Gradients, and counterfactuals.
    """

    def __init__(
        self,
        training_data: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        default_method: ExplanationType = ExplanationType.SHAP,
    ):
        """
        Initialize explainability engine.

        Args:
            training_data: Background/training data for explainers
            feature_names: Feature names
            default_method: Default explanation method
        """
        self.training_data = training_data
        self.feature_names = feature_names
        self.default_method = default_method

        self.shap_explainer = SHAPExplainer(background_data=training_data)
        self.lime_explainer = LIMEExplainer(training_data=training_data)
        self.ig_explainer = IntegratedGradientsExplainer()
        self.cf_explainer = CounterfactualExplainer()
        self.faithfulness_evaluator = FaithfulnessEvaluator()

        logger.info(
            f"ExplainabilityEngine initialized " f"(SHAP={SHAP_AVAILABLE}, LIME={LIME_AVAILABLE})"
        )

    def explain(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        instance: np.ndarray,
        method: ExplanationType | None = None,
        include_counterfactual: bool = False,
        include_faithfulness: bool = True,
    ) -> Explanation:
        """
        Generate explanation using specified method.

        Args:
            model: Prediction function
            instance: Instance to explain
            method: Explanation method (defaults to engine default)
            include_counterfactual: Generate counterfactual explanation
            include_faithfulness: Compute faithfulness metrics

        Returns:
            Explanation object
        """
        method = method or self.default_method

        if method == ExplanationType.SHAP:
            explanation = self.shap_explainer.explain(model, instance, self.feature_names)
        elif method == ExplanationType.LIME:
            explanation = self.lime_explainer.explain(model, instance, self.feature_names)
        elif method == ExplanationType.INTEGRATED_GRADIENTS:
            explanation = self.ig_explainer.explain(model, instance, self.feature_names)
        else:
            explanation = self.shap_explainer.explain(model, instance, self.feature_names)

        if include_counterfactual:
            cf_result = self.cf_explainer.generate_counterfactual(
                model, instance, feature_names=self.feature_names
            )
            explanation.counterfactuals = [cf_result]

        if include_faithfulness:
            faithfulness = self.faithfulness_evaluator.evaluate(model, instance, explanation)
            explanation.faithfulness_scores = faithfulness

        return explanation

    def compare_methods(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        instance: np.ndarray,
    ) -> dict[str, Explanation]:
        """
        Compare explanations from all methods.

        Args:
            model: Prediction function
            instance: Instance to explain

        Returns:
            Dictionary of method -> explanation
        """
        results = {}

        for method in [
            ExplanationType.SHAP,
            ExplanationType.LIME,
            ExplanationType.INTEGRATED_GRADIENTS,
        ]:
            try:
                results[method.value] = self.explain(
                    model, instance, method=method, include_faithfulness=True
                )
            except Exception as e:
                logger.warning(f"Failed to generate {method.value} explanation: {e}")

        return results
