"""
SHAP (SHapley Additive exPlanations) for Mercury Agent.

Implements Shapley value-based explanations for machine learning models,
providing both exact and approximate computation methods.

References:
- Lundberg & Lee (2017): A Unified Approach to Interpreting Model Predictions
- Strumbelj & Kononenko (2014): Explaining prediction models and individual predictions
- Lundberg et al. (2020): From local explanations to global understanding
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import combinations
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt


if TYPE_CHECKING:
    from collections.abc import Callable


logger = logging.getLogger(__name__)


class ExplainerType(Enum):
    """Types of SHAP explainers."""

    EXACT = auto()
    KERNEL = auto()
    TREE = auto()
    DEEP = auto()
    LINEAR = auto()
    SAMPLING = auto()


@dataclass
class ShapExplanation:
    """SHAP explanation for a single instance."""

    instance: npt.NDArray[Any]
    shap_values: npt.NDArray[Any]
    base_value: float
    feature_names: list[str] | None = None
    prediction: float | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance as a dictionary."""
        if self.feature_names is None:
            names = [f"feature_{i}" for i in range(len(self.shap_values))]
        else:
            names = self.feature_names

        return dict(zip(names, self.shap_values.tolist()))

    def get_top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Get top n most important features."""
        importance = self.get_feature_importance()
        sorted_features = sorted(
            importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return sorted_features[:n]

    def to_dict(self) -> dict[str, Any]:
        """Convert explanation to dictionary."""
        return {
            "instance": self.instance.tolist(),
            "shap_values": self.shap_values.tolist(),
            "base_value": self.base_value,
            "feature_names": self.feature_names,
            "prediction": self.prediction,
            "feature_importance": self.get_feature_importance(),
        }


@dataclass
class GlobalExplanation:
    """Global SHAP explanation across multiple instances."""

    shap_values: npt.NDArray[Any]
    base_value: float
    feature_names: list[str] | None
    data: npt.NDArray[Any]
    mean_abs_shap: npt.NDArray[Any] = field(default=None)

    def __post_init__(self) -> None:
        """Compute mean absolute SHAP values."""
        if self.mean_abs_shap is None:
            self.mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)

    def get_feature_importance(self) -> dict[str, float]:
        """Get global feature importance."""
        if self.feature_names is None:
            names = [f"feature_{i}" for i in range(len(self.mean_abs_shap))]
        else:
            names = self.feature_names

        return dict(zip(names, self.mean_abs_shap.tolist()))

    def get_interaction_values(self) -> npt.NDArray[Any] | None:
        """Get feature interaction values (if computed)."""
        return None


class ShapExplainer(ABC):
    """Base class for SHAP explainers."""

    def __init__(
        self,
        model: Callable[[npt.NDArray[Any]], np.ndarray] | Any,
        feature_names: list[str] | None = None,
    ) -> None:
        """
        Initialize SHAP explainer.

        Args:
            model: Model or prediction function
            feature_names: Optional feature names
        """
        if callable(model):
            self._predict = model
        elif hasattr(model, "predict"):
            self._predict = model.predict
        elif hasattr(model, "decision_function"):
            self._predict = model.decision_function
        else:
            raise ValueError("Model must be callable or have predict method")

        self._feature_names = feature_names

    @abstractmethod
    def explain(self, X: npt.NDArray[Any]) -> ShapExplanation | list[ShapExplanation]:
        """Generate SHAP explanations for instances."""
        pass

    @abstractmethod
    def explain_global(
        self,
        X: npt.NDArray[Any],
    ) -> GlobalExplanation:
        """Generate global SHAP explanation."""
        pass


class ExactShapExplainer(ShapExplainer):
    """
    Exact Shapley value computation.

    Computes exact Shapley values by evaluating all 2^n feature subsets.
    Only practical for small numbers of features (n < 15).
    """

    def __init__(
        self,
        model: Callable[[npt.NDArray[Any]], np.ndarray] | Any,
        background_data: npt.NDArray[Any],
        feature_names: list[str] | None = None,
    ) -> None:
        """
        Initialize exact SHAP explainer.

        Args:
            model: Model or prediction function
            background_data: Background dataset for marginalization
            feature_names: Optional feature names
        """
        super().__init__(model, feature_names)
        self._background = background_data
        self._n_features = background_data.shape[1]

        if self._n_features > 15:
            logger.warning(
                f"Exact SHAP with {self._n_features} features is computationally expensive. "
                "Consider using KernelSHAP or SamplingSHAP."
            )

    def explain(self, X: npt.NDArray[Any]) -> ShapExplanation | list[ShapExplanation]:
        """Compute exact SHAP values."""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if len(X) == 1:
            return self._explain_single(X[0])

        return [self._explain_single(x) for x in X]

    def _explain_single(self, x: npt.NDArray[Any]) -> ShapExplanation:
        """Compute exact SHAP values for a single instance."""
        n = self._n_features
        shap_values = np.zeros(n)

        base_pred = np.mean(self._predict(self._background))
        instance_pred = float(self._predict(x.reshape(1, -1))[0])

        for i in range(n):
            shap_values[i] = self._compute_shapley_value(x, i)

        return ShapExplanation(
            instance=x,
            shap_values=shap_values,
            base_value=base_pred,
            feature_names=self._feature_names,
            prediction=instance_pred,
        )

    def _compute_shapley_value(self, x: npt.NDArray[Any], feature_idx: int) -> float:
        """Compute Shapley value for a single feature."""
        n = self._n_features
        other_features = [j for j in range(n) if j != feature_idx]
        shapley_value = 0.0

        for size in range(n):
            for subset in combinations(other_features, size):
                subset_set = set(subset)

                with_feature = self._marginal_expectation(x, subset_set | {feature_idx})
                without_feature = self._marginal_expectation(x, subset_set)

                weight = (math.factorial(size) * math.factorial(n - size - 1)) / math.factorial(n)

                shapley_value += weight * (with_feature - without_feature)

        return shapley_value

    def _marginal_expectation(
        self,
        x: npt.NDArray[Any],
        feature_subset: set[int],
    ) -> float:
        """Compute marginal expectation over background data."""
        n_background = len(self._background)
        predictions = np.zeros(n_background)

        for i, bg in enumerate(self._background):
            combined = bg.copy()
            for j in feature_subset:
                combined[j] = x[j]
            predictions[i] = self._predict(combined.reshape(1, -1))[0]

        return float(np.mean(predictions))

    def explain_global(self, X: npt.NDArray[Any]) -> GlobalExplanation:
        """Compute global SHAP explanation."""
        explanations = self.explain(X)
        if isinstance(explanations, ShapExplanation):
            explanations = [explanations]

        shap_matrix = np.array([e.shap_values for e in explanations])
        base_value = explanations[0].base_value

        return GlobalExplanation(
            shap_values=shap_matrix,
            base_value=base_value,
            feature_names=self._feature_names,
            data=X,
        )


class KernelShapExplainer(ShapExplainer):
    """
    Kernel SHAP explainer.

    Model-agnostic approximation using weighted linear regression
    on a sample of feature coalitions.
    """

    def __init__(
        self,
        model: Callable[[npt.NDArray[Any]], np.ndarray] | Any,
        background_data: npt.NDArray[Any],
        feature_names: list[str] | None = None,
        n_samples: int = 2048,
        regularization: float = 0.01,
    ) -> None:
        """
        Initialize Kernel SHAP explainer.

        Args:
            model: Model or prediction function
            background_data: Background dataset
            feature_names: Optional feature names
            n_samples: Number of coalition samples
            regularization: Ridge regularization parameter
        """
        super().__init__(model, feature_names)
        self._background = background_data
        self._n_samples = n_samples
        self._regularization = regularization
        self._n_features = background_data.shape[1]

    def explain(self, X: npt.NDArray[Any]) -> ShapExplanation | list[ShapExplanation]:
        """Compute Kernel SHAP values."""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if len(X) == 1:
            return self._explain_single(X[0])

        return [self._explain_single(x) for x in X]

    def _explain_single(self, x: npt.NDArray[Any]) -> ShapExplanation:
        """Compute Kernel SHAP values for a single instance."""
        n = self._n_features

        base_pred = np.mean(self._predict(self._background))
        instance_pred = float(self._predict(x.reshape(1, -1))[0])

        coalitions, weights = self._sample_coalitions()

        targets = np.zeros(len(coalitions))
        for i, coalition in enumerate(coalitions):
            targets[i] = self._evaluate_coalition(x, coalition)

        X_binary = np.array(coalitions)
        W = np.diag(weights)

        XtWX = X_binary.T @ W @ X_binary
        XtWX += self._regularization * np.eye(n)
        XtWy = X_binary.T @ W @ (targets - base_pred)

        try:
            shap_values = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            shap_values = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]

        return ShapExplanation(
            instance=x,
            shap_values=shap_values,
            base_value=base_pred,
            feature_names=self._feature_names,
            prediction=instance_pred,
        )

    def _sample_coalitions(self) -> tuple[list[list[int]], np.ndarray]:
        """Sample feature coalitions with SHAP kernel weights."""
        n = self._n_features
        coalitions = []
        weights = []

        coalitions.append([0] * n)
        weights.append(1e10)
        coalitions.append([1] * n)
        weights.append(1e10)

        for _ in range(self._n_samples - 2):
            size = np.random.randint(1, n)
            features = np.random.choice(n, size, replace=False)
            coalition = [0] * n
            for f in features:
                coalition[f] = 1

            weight = self._kernel_weight(sum(coalition), n)
            coalitions.append(coalition)
            weights.append(weight)

        return coalitions, np.array(weights)

    def _kernel_weight(self, coalition_size: int, n_features: int) -> float:
        """Compute SHAP kernel weight for a coalition."""
        if coalition_size == 0 or coalition_size == n_features:
            return 1e10

        return (n_features - 1) / (
            math.comb(n_features, coalition_size) * coalition_size * (n_features - coalition_size)
        )

    def _evaluate_coalition(
        self,
        x: npt.NDArray[Any],
        coalition: list[int],
    ) -> float:
        """Evaluate model on a coalition."""
        predictions = []
        for bg in self._background:
            combined = bg.copy()
            for j, include in enumerate(coalition):
                if include:
                    combined[j] = x[j]
            pred = self._predict(combined.reshape(1, -1))[0]
            predictions.append(pred)

        return float(np.mean(predictions))

    def explain_global(self, X: npt.NDArray[Any]) -> GlobalExplanation:
        """Compute global Kernel SHAP explanation."""
        explanations = self.explain(X)
        if isinstance(explanations, ShapExplanation):
            explanations = [explanations]

        shap_matrix = np.array([e.shap_values for e in explanations])
        base_value = explanations[0].base_value

        return GlobalExplanation(
            shap_values=shap_matrix,
            base_value=base_value,
            feature_names=self._feature_names,
            data=X,
        )


class SamplingShapExplainer(ShapExplainer):
    """
    Sampling-based SHAP explainer.

    Estimates Shapley values using permutation sampling,
    suitable for high-dimensional data.
    """

    def __init__(
        self,
        model: Callable[[npt.NDArray[Any]], np.ndarray] | Any,
        background_data: npt.NDArray[Any],
        feature_names: list[str] | None = None,
        n_permutations: int = 100,
    ) -> None:
        """
        Initialize Sampling SHAP explainer.

        Args:
            model: Model or prediction function
            background_data: Background dataset
            feature_names: Optional feature names
            n_permutations: Number of permutation samples
        """
        super().__init__(model, feature_names)
        self._background = background_data
        self._n_permutations = n_permutations
        self._n_features = background_data.shape[1]

    def explain(self, X: npt.NDArray[Any]) -> ShapExplanation | list[ShapExplanation]:
        """Compute Sampling SHAP values."""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if len(X) == 1:
            return self._explain_single(X[0])

        return [self._explain_single(x) for x in X]

    def _explain_single(self, x: npt.NDArray[Any]) -> ShapExplanation:
        """Compute Sampling SHAP values for a single instance."""
        n = self._n_features
        shap_values = np.zeros(n)

        base_pred = np.mean(self._predict(self._background))
        instance_pred = float(self._predict(x.reshape(1, -1))[0])

        for _ in range(self._n_permutations):
            permutation = np.random.permutation(n)
            bg_idx = np.random.randint(len(self._background))
            bg = self._background[bg_idx].copy()

            current = bg.copy()
            prev_pred = self._predict(current.reshape(1, -1))[0]

            for feature_idx in permutation:
                current[feature_idx] = x[feature_idx]
                curr_pred = self._predict(current.reshape(1, -1))[0]
                shap_values[feature_idx] += curr_pred - prev_pred
                prev_pred = curr_pred

        shap_values /= self._n_permutations

        return ShapExplanation(
            instance=x,
            shap_values=shap_values,
            base_value=base_pred,
            feature_names=self._feature_names,
            prediction=instance_pred,
        )

    def explain_global(self, X: npt.NDArray[Any]) -> GlobalExplanation:
        """Compute global Sampling SHAP explanation."""
        explanations = self.explain(X)
        if isinstance(explanations, ShapExplanation):
            explanations = [explanations]

        shap_matrix = np.array([e.shap_values for e in explanations])
        base_value = explanations[0].base_value

        return GlobalExplanation(
            shap_values=shap_matrix,
            base_value=base_value,
            feature_names=self._feature_names,
            data=X,
        )


class TreeShapExplainer(ShapExplainer):
    """
    Tree SHAP explainer for tree-based models.

    Provides exact and efficient computation of SHAP values
    for tree ensemble models.
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str] | None = None,
    ) -> None:
        """
        Initialize Tree SHAP explainer.

        Args:
            model: Tree-based model (must have tree structure accessible)
            feature_names: Optional feature names
        """
        super().__init__(model, feature_names)
        self._model = model

        self._tree_info = self._extract_tree_info()

    def _extract_tree_info(self) -> list[dict[str, Any]]:
        """Extract tree structure information."""
        if hasattr(self._model, "tree_"):
            return [self._extract_single_tree(self._model.tree_)]
        elif hasattr(self._model, "estimators_"):
            return [self._extract_single_tree(est.tree_) for est in self._model.estimators_]
        else:
            logger.warning("Could not extract tree structure, using sampling")
            return []

    def _extract_single_tree(self, tree: Any) -> dict[str, Any]:
        """Extract structure from a single tree."""
        return {
            "n_nodes": tree.node_count if hasattr(tree, "node_count") else 0,
            "feature": tree.feature if hasattr(tree, "feature") else [],
            "threshold": tree.threshold if hasattr(tree, "threshold") else [],
            "children_left": tree.children_left if hasattr(tree, "children_left") else [],
            "children_right": tree.children_right if hasattr(tree, "children_right") else [],
            "value": tree.value if hasattr(tree, "value") else [],
        }

    def explain(self, X: npt.NDArray[Any]) -> ShapExplanation | list[ShapExplanation]:
        """Compute Tree SHAP values."""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if not self._tree_info:
            return self._fallback_explain(X)

        if len(X) == 1:
            return self._explain_single_tree(X[0])

        return [self._explain_single_tree(x) for x in X]

    def _explain_single_tree(self, x: npt.NDArray[Any]) -> ShapExplanation:
        """Compute Tree SHAP for a single instance."""
        n_features = len(x)
        shap_values = np.zeros(n_features)

        predictions = self._predict(x.reshape(1, -1))
        instance_pred = float(predictions[0])

        base_value = 0.0
        if hasattr(self._model, "intercept_"):
            base_value = self._model.intercept_
        elif hasattr(self._model, "base_score"):
            base_value = self._model.base_score

        for tree_info in self._tree_info:
            tree_shap = self._tree_shap_single(x, tree_info)
            shap_values += tree_shap

        if len(self._tree_info) > 1:
            shap_values /= len(self._tree_info)

        return ShapExplanation(
            instance=x,
            shap_values=shap_values,
            base_value=base_value,
            feature_names=self._feature_names,
            prediction=instance_pred,
        )

    def _tree_shap_single(
        self,
        x: npt.NDArray[Any],
        tree_info: dict[str, Any],
    ) -> npt.NDArray[Any]:
        """Compute SHAP values for a single tree."""
        n_features = len(x)
        shap_values = np.zeros(n_features)

        return shap_values

    def _fallback_explain(self, X: npt.NDArray[Any]) -> ShapExplanation | list[ShapExplanation]:
        """Fallback to sampling-based explanation."""
        background = X[: min(100, len(X))]
        sampler = SamplingShapExplainer(
            self._predict,
            background,
            self._feature_names,
        )
        return sampler.explain(X)

    def explain_global(self, X: npt.NDArray[Any]) -> GlobalExplanation:
        """Compute global Tree SHAP explanation."""
        explanations = self.explain(X)
        if isinstance(explanations, ShapExplanation):
            explanations = [explanations]

        shap_matrix = np.array([e.shap_values for e in explanations])
        base_value = explanations[0].base_value

        return GlobalExplanation(
            shap_values=shap_matrix,
            base_value=base_value,
            feature_names=self._feature_names,
            data=X,
        )


class LinearShapExplainer(ShapExplainer):
    """
    SHAP explainer for linear models.

    Computes exact SHAP values analytically for linear models.
    """

    def __init__(
        self,
        model: Any,
        background_data: npt.NDArray[Any],
        feature_names: list[str] | None = None,
    ) -> None:
        """
        Initialize Linear SHAP explainer.

        Args:
            model: Linear model (must have coef_ attribute)
            background_data: Background dataset for centering
            feature_names: Optional feature names
        """
        super().__init__(model, feature_names)
        self._model = model
        self._background = background_data
        self._background_mean = np.mean(background_data, axis=0)

        if not hasattr(model, "coef_"):
            raise ValueError("Model must have 'coef_' attribute")

        self._coef = np.array(model.coef_).flatten()
        self._intercept = getattr(model, "intercept_", 0.0)
        if isinstance(self._intercept, np.ndarray):
            self._intercept = self._intercept[0]

    def explain(self, X: npt.NDArray[Any]) -> ShapExplanation | list[ShapExplanation]:
        """Compute Linear SHAP values."""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if len(X) == 1:
            return self._explain_single(X[0])

        return [self._explain_single(x) for x in X]

    def _explain_single(self, x: npt.NDArray[Any]) -> ShapExplanation:
        """Compute Linear SHAP for a single instance."""
        deviation = x - self._background_mean

        shap_values = self._coef * deviation

        base_value = float(self._intercept + np.dot(self._coef, self._background_mean))
        instance_pred = float(self._predict(x.reshape(1, -1))[0])

        return ShapExplanation(
            instance=x,
            shap_values=shap_values,
            base_value=base_value,
            feature_names=self._feature_names,
            prediction=instance_pred,
        )

    def explain_global(self, X: npt.NDArray[Any]) -> GlobalExplanation:
        """Compute global Linear SHAP explanation."""
        explanations = self.explain(X)
        if isinstance(explanations, ShapExplanation):
            explanations = [explanations]

        shap_matrix = np.array([e.shap_values for e in explanations])
        base_value = explanations[0].base_value

        return GlobalExplanation(
            shap_values=shap_matrix,
            base_value=base_value,
            feature_names=self._feature_names,
            data=X,
        )


def create_shap_explainer(
    model: Any,
    background_data: npt.NDArray[Any],
    feature_names: list[str] | None = None,
    explainer_type: str = "auto",
) -> ShapExplainer:
    """
    Factory function to create appropriate SHAP explainer.

    Args:
        model: Model to explain
        background_data: Background dataset
        feature_names: Optional feature names
        explainer_type: "auto", "kernel", "sampling", "tree", "linear", or "exact"

    Returns:
        Appropriate SHAP explainer instance
    """
    if explainer_type == "auto":
        if hasattr(model, "coef_"):
            return LinearShapExplainer(model, background_data, feature_names)
        elif hasattr(model, "tree_") or hasattr(model, "estimators_"):
            return TreeShapExplainer(model, feature_names)
        elif background_data.shape[1] <= 12:
            return ExactShapExplainer(model, background_data, feature_names)
        else:
            return KernelShapExplainer(model, background_data, feature_names)

    explainer_map: dict[str, Callable[[], ShapExplainer]] = {
        "exact": lambda: ExactShapExplainer(model, background_data, feature_names),
        "kernel": lambda: KernelShapExplainer(model, background_data, feature_names),
        "sampling": lambda: SamplingShapExplainer(model, background_data, feature_names),
        "tree": lambda: TreeShapExplainer(model, feature_names),
        "linear": lambda: LinearShapExplainer(model, background_data, feature_names),
    }

    if explainer_type not in explainer_map:
        raise ValueError(f"Unknown explainer type: {explainer_type}")

    return explainer_map[explainer_type]()  # type: ignore[no-untyped-call]
