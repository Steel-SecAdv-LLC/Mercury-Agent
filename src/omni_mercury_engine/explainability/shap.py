# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""SHAP (SHapley Additive exPlanations) for Mercury Agent.

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

    instance: np.ndarray[Any, Any]
    shap_values: np.ndarray[Any, Any]
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

    shap_values: np.ndarray[Any, Any]
    base_value: float
    feature_names: list[str] | None
    data: np.ndarray[Any, Any]
    mean_abs_shap: np.ndarray[Any, Any] = field(default=None)  # type: ignore[arg-type, unused-ignore]

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

    def get_interaction_values(self) -> np.ndarray[Any, Any] | None:
        """Global feature-interaction matrix from SHAP-contribution covariance.

        Returns an ``(n_features, n_features)`` symmetric matrix whose ``(i, j)``
        entry is the covariance of feature ``i``'s and feature ``j``'s SHAP
        contributions across the explained instances. Two features whose
        attributions move together (or in opposition) across the dataset carry a
        large-magnitude off-diagonal entry, which is a legitimate *global*
        interaction signal; the diagonal is each feature's attribution variance.

        This is a covariance-based interaction proxy, not the exact Shapley
        interaction index (which requires re-querying the model over feature
        pairs and is not available from a materialised global explanation). It
        is defined for any explainer, so it is computed here rather than left
        unpopulated. Returns ``None`` only when there are too few instances
        (< 2) for covariance to be defined.
        """
        shap_matrix = np.asarray(self.shap_values, dtype=float)
        if shap_matrix.ndim != 2 or shap_matrix.shape[0] < 2:
            return None
        # rowvar=False -> variables are columns (features), observations are rows.
        interaction = np.cov(shap_matrix, rowvar=False)
        return np.atleast_2d(interaction)


class ShapExplainer(ABC):
    """Base class for SHAP explainers."""

    def __init__(
        self,
        model: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]] | Any,
        feature_names: list[str] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize SHAP explainer.

        Args:
            model: Model or prediction function
            feature_names: Optional feature names
            seed: Optional seed for the per-instance ``Generator`` shared
                with all subclasses (``ExactShapExplainer``,
                ``KernelShapExplainer``, ``SamplingShapExplainer``).
                Drives coalition sampling and background permutation.
                ``None`` (default) uses OS entropy.
        """
        self._rng: np.random.Generator = np.random.default_rng(seed)
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
    def explain(self, X: np.ndarray[Any, Any]) -> ShapExplanation | list[ShapExplanation]:
        """Generate SHAP explanations for instances."""
        pass

    @abstractmethod
    def explain_global(
        self,
        X: np.ndarray[Any, Any],
    ) -> GlobalExplanation:
        """Generate global SHAP explanation."""
        pass


class ExactShapExplainer(ShapExplainer):
    """Exact Shapley value computation.

    Computes exact Shapley values by evaluating all 2^n feature subsets. Only practical for small
    numbers of features (n < 15).
    """

    def __init__(
        self,
        model: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]] | Any,
        background_data: np.ndarray[Any, Any],
        feature_names: list[str] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize exact SHAP explainer.

        Args:
            model: Model or prediction function
            background_data: Background dataset for marginalization
            feature_names: Optional feature names
            seed: Optional seed for the per-instance ``Generator``
                driving exact-shap subset enumeration tie-breaking.
                ``None`` (default) uses OS entropy.
        """
        super().__init__(model, feature_names, seed=seed)
        self._background = background_data
        self._n_features = background_data.shape[1]

        if self._n_features > 15:
            logger.warning(
                f"Exact SHAP with {self._n_features} features is computationally expensive. "
                "Consider using KernelSHAP or SamplingSHAP."
            )

    def explain(self, X: np.ndarray[Any, Any]) -> ShapExplanation | list[ShapExplanation]:
        """Compute exact SHAP values."""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if len(X) == 1:
            return self._explain_single(X[0])

        return [self._explain_single(x) for x in X]

    def _explain_single(self, x: np.ndarray[Any, Any]) -> ShapExplanation:
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

    def _compute_shapley_value(self, x: np.ndarray[Any, Any], feature_idx: int) -> float:
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
        x: np.ndarray[Any, Any],
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

    def explain_global(self, X: np.ndarray[Any, Any]) -> GlobalExplanation:
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
    """Kernel SHAP explainer.

    Model-agnostic approximation using weighted linear regression on a sample of feature coalitions.
    """

    def __init__(
        self,
        model: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]] | Any,
        background_data: np.ndarray[Any, Any],
        feature_names: list[str] | None = None,
        n_samples: int = 2048,
        regularization: float = 0.01,
        seed: int | None = None,
    ) -> None:
        """Initialize Kernel SHAP explainer.

        Args:
            model: Model or prediction function
            background_data: Background dataset
            feature_names: Optional feature names
            n_samples: Number of coalition samples
            regularization: Ridge regularization parameter
            seed: Optional seed for the per-instance ``Generator``
                driving coalition sampling.  ``None`` (default) uses
                OS entropy.
        """
        super().__init__(model, feature_names, seed=seed)
        self._background = background_data
        self._n_samples = n_samples
        self._regularization = regularization
        self._n_features = background_data.shape[1]

    def explain(self, X: np.ndarray[Any, Any]) -> ShapExplanation | list[ShapExplanation]:
        """Compute Kernel SHAP values."""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if len(X) == 1:
            return self._explain_single(X[0])

        return [self._explain_single(x) for x in X]

    def _explain_single(self, x: np.ndarray[Any, Any]) -> ShapExplanation:
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

    def _sample_coalitions(self) -> tuple[list[list[int]], np.ndarray[Any, Any]]:
        """Sample feature coalitions with SHAP kernel weights."""
        n = self._n_features
        coalitions = []
        weights = []

        coalitions.append([0] * n)
        weights.append(1e10)
        coalitions.append([1] * n)
        weights.append(1e10)

        for _ in range(self._n_samples - 2):
            size = int(self._rng.integers(1, n))
            features = self._rng.choice(n, size, replace=False)
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
        x: np.ndarray[Any, Any],
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

    def explain_global(self, X: np.ndarray[Any, Any]) -> GlobalExplanation:
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
    """Sampling-based SHAP explainer.

    Estimates Shapley values using permutation sampling, suitable for high-dimensional data.
    """

    def __init__(
        self,
        model: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]] | Any,
        background_data: np.ndarray[Any, Any],
        feature_names: list[str] | None = None,
        n_permutations: int = 100,
        seed: int | None = None,
    ) -> None:
        """Initialize Sampling SHAP explainer.

        Args:
            model: Model or prediction function
            background_data: Background dataset
            feature_names: Optional feature names
            n_permutations: Number of permutation samples
            seed: Optional seed for the per-instance ``Generator``
                driving feature-permutation order.  ``None`` (default)
                uses OS entropy.
        """
        super().__init__(model, feature_names, seed=seed)
        self._background = background_data
        self._n_permutations = n_permutations
        self._n_features = background_data.shape[1]

    def explain(self, X: np.ndarray[Any, Any]) -> ShapExplanation | list[ShapExplanation]:
        """Compute Sampling SHAP values."""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if len(X) == 1:
            return self._explain_single(X[0])

        return [self._explain_single(x) for x in X]

    def _explain_single(self, x: np.ndarray[Any, Any]) -> ShapExplanation:
        """Compute Sampling SHAP values for a single instance."""
        n = self._n_features
        shap_values = np.zeros(n)

        base_pred = np.mean(self._predict(self._background))
        instance_pred = float(self._predict(x.reshape(1, -1))[0])

        for _ in range(self._n_permutations):
            permutation = self._rng.permutation(n)
            bg_idx = int(self._rng.integers(len(self._background)))
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

    def explain_global(self, X: np.ndarray[Any, Any]) -> GlobalExplanation:
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
    """Tree SHAP explainer for tree-based models.

    Provides exact and efficient computation of SHAP values for tree ensemble models.
    """

    #: Max number of *used* features in a single tree for which the Shapley sum
    #: is enumerated exactly (2**k coalitions). Above this the per-tree value is
    #: estimated by permutation sampling instead, keeping the cost bounded.
    _MAX_EXACT_TREE_FEATURES = 12

    def __init__(
        self,
        model: Any,
        feature_names: list[str] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize Tree SHAP explainer.

        Args:
            model: Tree-based model (must have tree structure accessible)
            feature_names: Optional feature names
            seed: Optional seed for the ``Generator`` used by the
                permutation-sampling branch. Exact Tree SHAP (a tree with at
                most ``_MAX_EXACT_TREE_FEATURES`` used features) is deterministic
                given a fixed input; wider trees fall back to a seeded
                ``n_permutations`` estimate whose values are additive and
                unbiased but seed-dependent, so pass a fixed ``seed`` for
                reproducibility. ``None`` (default) uses OS entropy.
        """
        super().__init__(model, feature_names, seed=seed)
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
        """Extract structure from a single tree.

        ``weighted_n_node_samples`` is the training coverage of each node and
        is what path-dependent Tree SHAP uses as the implicit background: when a
        feature is "absent" from a coalition the tree marginalises over its
        split by taking the coverage-weighted average of the two children. Falls
        back to ``n_node_samples`` and then to uniform ones if a tree object
        exposes neither (so any tree exposing the standard ``tree_`` array
        interface still yields defined values).
        """
        n_nodes = int(tree.node_count) if hasattr(tree, "node_count") else 0
        coverage = None
        if hasattr(tree, "weighted_n_node_samples"):
            coverage = np.asarray(tree.weighted_n_node_samples, dtype=float)
        elif hasattr(tree, "n_node_samples"):
            coverage = np.asarray(tree.n_node_samples, dtype=float)
        elif n_nodes:
            coverage = np.ones(n_nodes, dtype=float)
        else:
            coverage = np.asarray([], dtype=float)
        feature_arr = np.asarray(tree.feature) if hasattr(tree, "feature") else np.asarray([])
        return {
            "n_nodes": n_nodes,
            "feature": feature_arr,
            # Sorted unique split-feature indices, computed once per tree at
            # extraction: the bounds check, the full-coalition `present` mask,
            # and the per-tree Shapley enumeration all consume this, and
            # recomputing it from the per-node array on every explanation is
            # avoidable O(n_nodes) work for large trees/ensembles.
            "used_features": (
                np.unique(feature_arr[feature_arr >= 0]).astype(int)
                if feature_arr.size
                else np.asarray([], dtype=int)
            ),
            "threshold": (
                np.asarray(tree.threshold, dtype=float)
                if hasattr(tree, "threshold")
                else np.asarray([])
            ),
            "children_left": (
                np.asarray(tree.children_left) if hasattr(tree, "children_left") else np.asarray([])
            ),
            "children_right": (
                np.asarray(tree.children_right)
                if hasattr(tree, "children_right")
                else np.asarray([])
            ),
            "value": np.asarray(tree.value) if hasattr(tree, "value") else np.asarray([]),
            "weighted_n_node_samples": coverage,
        }

    def explain(self, X: np.ndarray[Any, Any]) -> ShapExplanation | list[ShapExplanation]:
        """Compute Tree SHAP values."""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if not self._tree_info:
            return self._fallback_explain(X)

        if len(X) == 1:
            return self._explain_single_tree(X[0])

        return [self._explain_single_tree(x) for x in X]

    def _explain_single_tree(self, x: np.ndarray[Any, Any]) -> ShapExplanation:
        """Compute Tree SHAP for a single instance.

        The ensemble decomposition is computed entirely in the tree's own
        output space (the coverage-marginalised conditional expectation), so the
        SHAP additivity property holds *exactly*: ``base_value + sum(shap) ==
        prediction`` for every instance, where ``prediction`` is the trees'
        averaged path-dependent output and ``base_value`` is the averaged
        empty-coalition expectation. Averaging across the ensemble preserves
        additivity because each tree satisfies it individually.
        """
        n_features = len(x)
        # Fail fast with an actionable message rather than an opaque IndexError
        # deep in the recursion / boolean-mask assignment when a tree splits on a
        # feature index the instance vector cannot address (e.g. the caller passed
        # a reduced feature set the model was not trained on).
        for tree_info in self._tree_info:
            used_features = tree_info["used_features"]
            max_feat = int(used_features[-1]) if used_features.size else -1
            if max_feat >= n_features:
                raise ValueError(
                    f"tree splits on feature index {max_feat} but the instance has "
                    f"only {n_features} features; the explainer was built for a model "
                    "trained on more features than were supplied"
                )

        shap_values = np.zeros(n_features)
        base_sum = 0.0
        pred_sum = 0.0

        empty = np.zeros(n_features, dtype=bool)
        for tree_info in self._tree_info:
            shap_values += self._tree_shap_single(x, tree_info)
            base_sum += self._cond_expectation(tree_info, x, empty)
            used = np.zeros(n_features, dtype=bool)
            used[tree_info["used_features"]] = True
            pred_sum += self._cond_expectation(tree_info, x, used)

        n_trees = max(1, len(self._tree_info))
        shap_values /= n_trees
        base_value = base_sum / n_trees
        prediction = pred_sum / n_trees

        return ShapExplanation(
            instance=x,
            shap_values=shap_values,
            base_value=float(base_value),
            feature_names=self._feature_names,
            prediction=float(prediction),
        )

    def _leaf_value(self, tree_info: dict[str, Any], node: int) -> float:
        """Scalar output of a leaf node.

        The regression-vs-classification distinction is made from the *last*
        axis of the tree's ``value`` array (the ``n_classes`` dimension in the
        standard ``tree_`` layout), not from the flattened size -- otherwise a
        multi-output-regression leaf of shape ``(n_outputs, 1)`` and a binary
        classification leaf of shape ``(1, 2)`` are indistinguishable after
        ``ravel()``. For regression (last axis 1) the leaf's first output value
        is returned; for classification the positive-class (last-class)
        probability of the last output's class counts is returned
        (``predict_proba[:, -1]`` for the common binary case). Multi-output
        regression is reduced to its first output (a documented scalar-SHAP
        limitation, kept self-consistent so additivity still holds).
        """
        node_value = np.asarray(tree_info["value"][node])
        n_classes = int(np.asarray(tree_info["value"]).shape[-1])
        if n_classes == 1:
            # Regression: the (first) output value directly.
            return float(node_value.reshape(-1)[0])
        # Classification: positive-class probability from the last output's
        # class counts. reshape(-1)[-n_classes:] selects that last output row.
        counts = node_value.reshape(-1)[-n_classes:]
        total = float(counts.sum())
        if total <= 0.0:
            return 0.0
        return float(counts[-1] / total)

    def _cond_expectation(
        self,
        tree_info: dict[str, Any],
        x: np.ndarray[Any, Any],
        present: np.ndarray[Any, Any],
    ) -> float:
        """Path-dependent conditional expectation ``E[tree(x) | x_S]``.

        Features in ``present`` follow the instance down its split; absent
        features are marginalised by taking the coverage-weighted mean of both
        children (Lundberg et al. 2020, path-dependent feature perturbation).

        Implemented with an explicit work-stack rather than recursion: a legit
        deep tree (sklearn defaults to ``max_depth=None``) can exceed Python's
        recursion limit, so a recursive value function would raise
        ``RecursionError`` on any explanation of such a tree. The stack
        accumulates each reached leaf's value weighted by the path probability,
        which is exactly the recursive coverage-weighted mean by linearity.
        """
        left = tree_info["children_left"]
        right = tree_info["children_right"]
        feat = tree_info["feature"]
        thr = tree_info["threshold"]
        cov = tree_info["weighted_n_node_samples"]

        total_value = 0.0
        stack: list[tuple[int, float]] = [(0, 1.0)]
        while stack:
            node, weight = stack.pop()
            f = int(feat[node])
            if f < 0 or int(left[node]) < 0:  # leaf (standard tree_ sentinel: feature == -2)
                total_value += weight * self._leaf_value(tree_info, node)
                continue
            lc = int(left[node])
            rc = int(right[node])
            if present[f]:
                child = lc if x[f] <= thr[node] else rc
                stack.append((child, weight))
                continue
            wl = float(cov[lc])
            wr = float(cov[rc])
            branch_total = wl + wr
            if branch_total <= 0.0:
                stack.append((lc, weight * 0.5))
                stack.append((rc, weight * 0.5))
            else:
                stack.append((lc, weight * wl / branch_total))
                stack.append((rc, weight * wr / branch_total))

        return total_value

    def _tree_shap_single(
        self,
        x: np.ndarray[Any, Any],
        tree_info: dict[str, Any],
    ) -> np.ndarray[Any, Any]:
        """Exact (small trees) or sampled path-dependent SHAP for one tree.

        Only features that actually appear in the tree's split nodes can carry
        non-zero attribution, so the coalition space is over those used features
        alone. When there are few enough of them the Shapley sum is enumerated
        exactly; otherwise it is estimated by permutation sampling over the same
        conditional-expectation value function (no external background needed --
        the tree's node coverage is the background).
        """
        n_features = len(x)
        shap_values = np.zeros(n_features)
        used = tree_info["used_features"].tolist()  # cached sorted-unique at extraction
        if not used:
            return shap_values

        if len(used) <= self._MAX_EXACT_TREE_FEATURES:
            n = len(used)
            for feature_idx in used:
                others = [u for u in used if u != feature_idx]
                phi = 0.0
                for size in range(len(others) + 1):
                    weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
                    for combo in combinations(others, size):
                        present = np.zeros(n_features, dtype=bool)
                        present[list(combo)] = True
                        without = self._cond_expectation(tree_info, x, present)
                        present[feature_idx] = True
                        with_feature = self._cond_expectation(tree_info, x, present)
                        phi += weight * (with_feature - without)
                shap_values[feature_idx] = phi
        else:
            n_permutations = 128
            for _ in range(n_permutations):
                order = self._rng.permutation(used)
                present = np.zeros(n_features, dtype=bool)
                prev = self._cond_expectation(tree_info, x, present)
                for feature_idx in order:
                    present[feature_idx] = True
                    current = self._cond_expectation(tree_info, x, present)
                    shap_values[feature_idx] += current - prev
                    prev = current
            shap_values /= n_permutations

        return shap_values

    def _fallback_explain(self, X: np.ndarray[Any, Any]) -> ShapExplanation | list[ShapExplanation]:
        """Fallback to sampling-based explanation."""
        background = X[: min(100, len(X))]
        sampler = SamplingShapExplainer(
            self._predict,
            background,
            self._feature_names,
        )
        return sampler.explain(X)

    def explain_global(self, X: np.ndarray[Any, Any]) -> GlobalExplanation:
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
    """SHAP explainer for linear models.

    Computes exact SHAP values analytically for linear models.
    """

    def __init__(
        self,
        model: Any,
        background_data: np.ndarray[Any, Any],
        feature_names: list[str] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize Linear SHAP explainer.

        Args:
            model: Linear model (must have coef_ attribute)
            background_data: Background dataset for centering
            feature_names: Optional feature names
            seed: Optional seed forwarded to the base ``BaseSHAPExplainer``
                ``Generator`` (Linear SHAP is closed-form deterministic;
                the seed is only consumed if the caller falls through to
                a sampling-based fallback).  ``None`` (default) uses OS
                entropy.
        """
        super().__init__(model, feature_names, seed=seed)
        self._model = model
        self._background = background_data
        self._background_mean = np.mean(background_data, axis=0)

        if not hasattr(model, "coef_"):
            raise ValueError("Model must have 'coef_' attribute")

        self._coef = np.array(model.coef_).flatten()
        self._intercept = getattr(model, "intercept_", 0.0)
        if isinstance(self._intercept, np.ndarray):
            self._intercept = self._intercept[0]

    def explain(self, X: np.ndarray[Any, Any]) -> ShapExplanation | list[ShapExplanation]:
        """Compute Linear SHAP values."""
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if len(X) == 1:
            return self._explain_single(X[0])

        return [self._explain_single(x) for x in X]

    def _explain_single(self, x: np.ndarray[Any, Any]) -> ShapExplanation:
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

    def explain_global(self, X: np.ndarray[Any, Any]) -> GlobalExplanation:
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
    background_data: np.ndarray[Any, Any],
    feature_names: list[str] | None = None,
    explainer_type: str = "auto",
) -> ShapExplainer:
    """Factory function to create appropriate SHAP explainer.

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

    return explainer_map[explainer_type]()
