"""
Counterfactual Explanations for Mercury Agent.

Implements counterfactual explanation methods that answer "what would need
to change for the model to give a different prediction?"

References:
- Wachter et al. (2017): Counterfactual Explanations without Opening the Black Box
- Mothilal et al. (2020): DiCE: Diverse Counterfactual Explanations
- Karimi et al. (2020): Algorithmic Recourse: from Counterfactual Explanations to Interventions
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import minimize

if TYPE_CHECKING:
    from collections.abc import Callable


logger = logging.getLogger(__name__)


class DistanceMetric(Enum):
    """Distance metrics for counterfactual generation."""

    L1 = auto()
    L2 = auto()
    MAD = auto()
    GOWER = auto()


class CounterfactualMethod(Enum):
    """Counterfactual generation methods."""

    WACHTER = auto()
    DICE = auto()
    GROWING_SPHERES = auto()
    GENETIC = auto()
    PROTOTYPE = auto()


@dataclass
class FeatureConstraint:
    """Constraint on a feature for counterfactual generation."""

    name: str
    feature_idx: int
    is_mutable: bool = True
    is_categorical: bool = False
    categories: list[Any] | None = None
    min_value: float | None = None
    max_value: float | None = None
    step_size: float | None = None


@dataclass
class Counterfactual:
    """A single counterfactual explanation."""

    original: np.ndarray[Any, Any]
    counterfactual: np.ndarray[Any, Any]
    original_prediction: float
    counterfactual_prediction: float
    feature_changes: dict[str, tuple[Any, Any]]
    distance: float
    validity: bool
    sparsity: int
    feature_names: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_changes_description(self) -> list[str]:
        """Get human-readable description of changes."""
        descriptions = []
        for name, (old, new) in self.feature_changes.items():
            descriptions.append(f"{name}: {old:.4f} -> {new:.4f}")
        return descriptions

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original": self.original.tolist(),
            "counterfactual": self.counterfactual.tolist(),
            "original_prediction": self.original_prediction,
            "counterfactual_prediction": self.counterfactual_prediction,
            "feature_changes": {
                k: [float(v[0]), float(v[1])] for k, v in self.feature_changes.items()
            },
            "distance": self.distance,
            "validity": self.validity,
            "sparsity": self.sparsity,
        }


@dataclass
class CounterfactualSet:
    """Set of diverse counterfactual explanations."""

    original: np.ndarray[Any, Any]
    counterfactuals: list[Counterfactual]
    original_prediction: float
    target_class: int | None
    diversity_score: float
    coverage_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_best(self) -> Counterfactual | None:
        """Get the best counterfactual (smallest distance)."""
        if not self.counterfactuals:
            return None
        return min(self.counterfactuals, key=lambda x: x.distance)

    def get_most_sparse(self) -> Counterfactual | None:
        """Get the counterfactual with fewest changes."""
        if not self.counterfactuals:
            return None
        return min(self.counterfactuals, key=lambda x: x.sparsity)


class CounterfactualGenerator(ABC):
    """Base class for counterfactual generators."""

    def __init__(
        self,
        model: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]] | Any,
        feature_names: list[str] | None = None,
        feature_constraints: list[FeatureConstraint] | None = None,
    ) -> None:
        """
        Initialize counterfactual generator.

        Args:
            model: Model or prediction function
            feature_names: Optional feature names
            feature_constraints: Optional feature constraints
        """
        if callable(model):
            self._predict = model
        elif hasattr(model, "predict_proba"):
            self._predict = lambda x: model.predict_proba(x)[:, 1]
        elif hasattr(model, "predict"):
            self._predict = model.predict
        elif hasattr(model, "decision_function"):
            self._predict = model.decision_function
        else:
            raise ValueError("Model must be callable or have predict method")

        self._feature_names = feature_names
        self._constraints = feature_constraints or []

    @abstractmethod
    def generate(
        self,
        instance: np.ndarray[Any, Any],
        target_class: int | None = None,
        n_counterfactuals: int = 1,
    ) -> CounterfactualSet:
        """Generate counterfactual explanations."""
        pass

    def _compute_distance(
        self,
        x1: np.ndarray[Any, Any],
        x2: np.ndarray[Any, Any],
        metric: DistanceMetric = DistanceMetric.L2,
        feature_weights: np.ndarray[Any, Any] | None = None,
    ) -> float:
        """Compute distance between two instances."""
        diff = x1 - x2

        if feature_weights is not None:
            diff = diff * feature_weights

        if metric == DistanceMetric.L1:
            return float(np.sum(np.abs(diff)))
        elif metric == DistanceMetric.L2:
            return float(np.sqrt(np.sum(diff**2)))
        elif metric == DistanceMetric.MAD:
            return float(np.sum(np.abs(diff)))
        else:
            return float(np.sqrt(np.sum(diff**2)))

    def _get_feature_changes(
        self,
        original: np.ndarray[Any, Any],
        counterfactual: np.ndarray[Any, Any],
    ) -> dict[str, tuple[Any, Any]]:
        """Get dictionary of feature changes."""
        changes = {}
        for i in range(len(original)):
            if not np.isclose(original[i], counterfactual[i]):
                name = self._feature_names[i] if self._feature_names else f"feature_{i}"
                changes[name] = (original[i], counterfactual[i])
        return changes


class WachterCounterfactual(CounterfactualGenerator):
    """
    Wachter et al.

    counterfactual generation.     Uses gradient-based optimization to find counterfactuals that
    minimize distance while achieving the desired prediction.
    """

    def __init__(
        self,
        model: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]] | Any,
        feature_names: list[str] | None = None,
        feature_constraints: list[FeatureConstraint] | None = None,
        lambda_param: float = 0.1,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
    ) -> None:
        """
        Initialize Wachter counterfactual generator.

        Args:
            model: Model or prediction function
            feature_names: Feature names
            feature_constraints: Feature constraints
            lambda_param: Trade-off between proximity and validity
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance
        """
        super().__init__(model, feature_names, feature_constraints)
        self._lambda = lambda_param
        self._max_iter = max_iterations
        self._tolerance = tolerance

    def generate(
        self,
        instance: np.ndarray[Any, Any],
        target_class: int | None = None,
        n_counterfactuals: int = 1,
    ) -> CounterfactualSet:
        """Generate Wachter counterfactuals."""
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        original = instance[0].copy()
        original_pred = float(self._predict(original.reshape(1, -1))[0])

        if target_class is None:
            target_class = 1 if original_pred < 0.5 else 0

        target_pred = float(target_class)

        counterfactuals = []
        for i in range(n_counterfactuals):
            init_point = original + np.random.randn(len(original)) * 0.1 * (i + 1)

            cf = self._optimize(original, init_point, target_pred)

            if cf is not None:
                cf_pred = float(self._predict(cf.reshape(1, -1))[0])
                changes = self._get_feature_changes(original, cf)
                distance = self._compute_distance(original, cf)

                is_valid = (target_class == 1 and cf_pred >= 0.5) or (
                    target_class == 0 and cf_pred < 0.5
                )

                counterfactuals.append(
                    Counterfactual(
                        original=original,
                        counterfactual=cf,
                        original_prediction=original_pred,
                        counterfactual_prediction=cf_pred,
                        feature_changes=changes,
                        distance=distance,
                        validity=is_valid,
                        sparsity=len(changes),
                        feature_names=self._feature_names,
                    )
                )

        diversity = self._compute_diversity(counterfactuals) if counterfactuals else 0.0
        coverage = sum(1 for cf in counterfactuals if cf.validity) / max(1, n_counterfactuals)

        return CounterfactualSet(
            original=original,
            counterfactuals=counterfactuals,
            original_prediction=original_pred,
            target_class=target_class,
            diversity_score=diversity,
            coverage_score=coverage,
        )

    def _optimize(
        self,
        original: np.ndarray[Any, Any],
        init_point: np.ndarray[Any, Any],
        target_pred: float,
    ) -> np.ndarray[Any, Any] | None:
        """Optimize to find counterfactual."""

        def objective(x: np.ndarray[Any, Any]) -> float:
            pred = self._predict(x.reshape(1, -1))[0]
            pred_loss = (pred - target_pred) ** 2
            dist_loss = self._lambda * np.sum((x - original) ** 2)
            return float(pred_loss + dist_loss)

        bounds = self._get_bounds(original)

        try:
            result = minimize(
                objective,
                init_point,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": self._max_iter, "ftol": self._tolerance},
            )

            if result.success:
                return result.x
        except Exception as e:
            logger.warning(f"Optimization failed: {e}")

        return None

    def _get_bounds(self, original: np.ndarray[Any, Any]) -> list[tuple[float, float]]:
        """Get optimization bounds from constraints."""
        bounds = []
        for i in range(len(original)):
            constraint = next(
                (c for c in self._constraints if c.feature_idx == i),
                None,
            )

            if constraint is not None:
                if not constraint.is_mutable:
                    bounds.append((original[i], original[i]))
                else:
                    low = constraint.min_value if constraint.min_value is not None else -np.inf
                    high = constraint.max_value if constraint.max_value is not None else np.inf
                    bounds.append((low, high))
            else:
                bounds.append((-np.inf, np.inf))

        return bounds

    def _compute_diversity(self, counterfactuals: list[Counterfactual]) -> float:
        """Compute diversity score for counterfactual set."""
        if len(counterfactuals) < 2:
            return 0.0

        distances = []
        for i in range(len(counterfactuals)):
            for j in range(i + 1, len(counterfactuals)):
                dist = self._compute_distance(
                    counterfactuals[i].counterfactual,
                    counterfactuals[j].counterfactual,
                )
                distances.append(dist)

        return float(np.mean(distances)) if distances else 0.0


class DiCECounterfactual(CounterfactualGenerator):
    """
    DiCE: Diverse Counterfactual Explanations.

    Generates a diverse set of counterfactuals using a diversity-promoting loss function.
    """

    def __init__(
        self,
        model: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]] | Any,
        feature_names: list[str] | None = None,
        feature_constraints: list[FeatureConstraint] | None = None,
        proximity_weight: float = 0.5,
        diversity_weight: float = 1.0,
        max_iterations: int = 500,
    ) -> None:
        """
        Initialize DiCE counterfactual generator.

        Args:
            model: Model or prediction function
            feature_names: Feature names
            feature_constraints: Feature constraints
            proximity_weight: Weight for proximity loss
            diversity_weight: Weight for diversity loss
            max_iterations: Maximum optimization iterations
        """
        super().__init__(model, feature_names, feature_constraints)
        self._proximity_weight = proximity_weight
        self._diversity_weight = diversity_weight
        self._max_iter = max_iterations

    def generate(
        self,
        instance: np.ndarray[Any, Any],
        target_class: int | None = None,
        n_counterfactuals: int = 4,
    ) -> CounterfactualSet:
        """Generate DiCE counterfactuals."""
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        original = instance[0].copy()
        original_pred = float(self._predict(original.reshape(1, -1))[0])

        if target_class is None:
            target_class = 1 if original_pred < 0.5 else 0

        cf_points = self._optimize_diverse(
            original,
            target_class,
            n_counterfactuals,
        )

        counterfactuals = []
        for cf in cf_points:
            cf_pred = float(self._predict(cf.reshape(1, -1))[0])
            changes = self._get_feature_changes(original, cf)
            distance = self._compute_distance(original, cf)

            is_valid = (target_class == 1 and cf_pred >= 0.5) or (
                target_class == 0 and cf_pred < 0.5
            )

            counterfactuals.append(
                Counterfactual(
                    original=original,
                    counterfactual=cf,
                    original_prediction=original_pred,
                    counterfactual_prediction=cf_pred,
                    feature_changes=changes,
                    distance=distance,
                    validity=is_valid,
                    sparsity=len(changes),
                    feature_names=self._feature_names,
                )
            )

        diversity = self._compute_pairwise_diversity(cf_points)
        coverage = sum(1 for cf in counterfactuals if cf.validity) / max(1, n_counterfactuals)

        return CounterfactualSet(
            original=original,
            counterfactuals=counterfactuals,
            original_prediction=original_pred,
            target_class=target_class,
            diversity_score=diversity,
            coverage_score=coverage,
        )

    def _optimize_diverse(
        self,
        original: np.ndarray[Any, Any],
        target_class: int,
        n_cfs: int,
    ) -> list[np.ndarray[Any, Any]]:
        """Optimize for diverse counterfactuals simultaneously."""
        n_features = len(original)
        target_pred = float(target_class)

        init_points = [original + np.random.randn(n_features) * 0.1 * (i + 1) for i in range(n_cfs)]
        init_flat = np.concatenate(init_points)

        def objective(x_flat: np.ndarray[Any, Any]) -> float:
            cfs = [x_flat[i * n_features : (i + 1) * n_features] for i in range(n_cfs)]

            validity_loss = 0.0
            proximity_loss = 0.0

            for cf in cfs:
                pred = self._predict(cf.reshape(1, -1))[0]
                validity_loss += (pred - target_pred) ** 2
                proximity_loss += np.sum((cf - original) ** 2)

            diversity_loss = 0.0
            for i in range(n_cfs):
                for j in range(i + 1, n_cfs):
                    dist = np.sum((cfs[i] - cfs[j]) ** 2)
                    diversity_loss -= dist

            total = (
                validity_loss
                + self._proximity_weight * proximity_loss
                + self._diversity_weight * diversity_loss / max(1, n_cfs * (n_cfs - 1) / 2)
            )

            return total

        try:
            result = minimize(
                objective,
                init_flat,
                method="L-BFGS-B",
                options={"maxiter": self._max_iter},
            )

            cfs = [result.x[i * n_features : (i + 1) * n_features] for i in range(n_cfs)]
            return cfs

        except Exception as e:
            logger.warning(f"DiCE optimization failed: {e}")
            return init_points

    def _compute_pairwise_diversity(self, points: list[np.ndarray[Any, Any]]) -> float:
        """Compute pairwise diversity."""
        if len(points) < 2:
            return 0.0

        distances = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = np.sqrt(np.sum((points[i] - points[j]) ** 2))
                distances.append(dist)

        return float(np.mean(distances))


class GrowingSpheresCounterfactual(CounterfactualGenerator):
    """
    Growing Spheres counterfactual generation.

    Finds counterfactuals by growing hyperspheres around the instance until the decision boundary is
    crossed.
    """

    def __init__(
        self,
        model: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]] | Any,
        feature_names: list[str] | None = None,
        feature_constraints: list[FeatureConstraint] | None = None,
        n_samples: int = 1000,
        step_size: float = 0.1,
        max_iterations: int = 100,
    ) -> None:
        """
        Initialize Growing Spheres generator.

        Args:
            model: Model or prediction function
            feature_names: Feature names
            feature_constraints: Feature constraints
            n_samples: Samples per sphere
            step_size: Sphere growth step
            max_iterations: Maximum growth iterations
        """
        super().__init__(model, feature_names, feature_constraints)
        self._n_samples = n_samples
        self._step_size = step_size
        self._max_iter = max_iterations

    def generate(
        self,
        instance: np.ndarray[Any, Any],
        target_class: int | None = None,
        n_counterfactuals: int = 1,
    ) -> CounterfactualSet:
        """Generate Growing Spheres counterfactuals."""
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        original = instance[0].copy()
        original_pred = float(self._predict(original.reshape(1, -1))[0])

        if target_class is None:
            target_class = 1 if original_pred < 0.5 else 0

        counterfactuals = []

        for _ in range(n_counterfactuals):
            cf = self._grow_sphere(original, target_class)

            if cf is not None:
                cf_pred = float(self._predict(cf.reshape(1, -1))[0])
                changes = self._get_feature_changes(original, cf)
                distance = self._compute_distance(original, cf)

                is_valid = (target_class == 1 and cf_pred >= 0.5) or (
                    target_class == 0 and cf_pred < 0.5
                )

                counterfactuals.append(
                    Counterfactual(
                        original=original,
                        counterfactual=cf,
                        original_prediction=original_pred,
                        counterfactual_prediction=cf_pred,
                        feature_changes=changes,
                        distance=distance,
                        validity=is_valid,
                        sparsity=len(changes),
                        feature_names=self._feature_names,
                    )
                )

        diversity = 0.0
        if len(counterfactuals) > 1:
            distances = []
            for i in range(len(counterfactuals)):
                for j in range(i + 1, len(counterfactuals)):
                    distances.append(
                        self._compute_distance(
                            counterfactuals[i].counterfactual,
                            counterfactuals[j].counterfactual,
                        )
                    )
            diversity = float(np.mean(distances))

        coverage = sum(1 for cf in counterfactuals if cf.validity) / max(1, n_counterfactuals)

        return CounterfactualSet(
            original=original,
            counterfactuals=counterfactuals,
            original_prediction=original_pred,
            target_class=target_class,
            diversity_score=diversity,
            coverage_score=coverage,
        )

    def _grow_sphere(
        self,
        original: np.ndarray[Any, Any],
        target_class: int,
    ) -> np.ndarray[Any, Any] | None:
        """Grow sphere until crossing decision boundary."""
        len(original)
        radius = self._step_size

        for _ in range(self._max_iter):
            samples = self._sample_sphere(original, radius, self._n_samples)

            for sample in samples:
                pred = self._predict(sample.reshape(1, -1))[0]

                if (target_class == 1 and pred >= 0.5) or (target_class == 0 and pred < 0.5):
                    refined = self._refine_counterfactual(original, sample, target_class)
                    return refined

            radius += self._step_size

        return None

    def _sample_sphere(
        self,
        center: np.ndarray[Any, Any],
        radius: float,
        n_samples: int,
    ) -> np.ndarray[Any, Any]:
        """Sample points uniformly on a hypersphere."""
        n_features = len(center)

        directions = np.random.randn(n_samples, n_features)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        radii = radius * np.random.uniform(0.8, 1.0, n_samples)

        samples = center + directions * radii[:, np.newaxis]
        return samples

    def _refine_counterfactual(
        self,
        original: np.ndarray[Any, Any],
        candidate: np.ndarray[Any, Any],
        target_class: int,
    ) -> np.ndarray[Any, Any]:
        """Refine counterfactual using binary search."""
        low = 0.0
        high = 1.0

        for _ in range(20):
            mid = (low + high) / 2
            point = original + mid * (candidate - original)
            pred = self._predict(point.reshape(1, -1))[0]

            if (target_class == 1 and pred >= 0.5) or (target_class == 0 and pred < 0.5):
                high = mid
            else:
                low = mid

        return original + high * (candidate - original)


class PrototypeCounterfactual(CounterfactualGenerator):
    """
    Prototype-based counterfactual generation.

    Finds counterfactuals by moving towards prototypes of the target class.
    """

    def __init__(
        self,
        model: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]] | Any,
        training_data: np.ndarray[Any, Any],
        training_labels: np.ndarray[Any, Any],
        feature_names: list[str] | None = None,
        feature_constraints: list[FeatureConstraint] | None = None,
        n_prototypes: int = 5,
    ) -> None:
        """
        Initialize Prototype counterfactual generator.

        Args:
            model: Model or prediction function
            training_data: Training data for prototype extraction
            training_labels: Training labels
            feature_names: Feature names
            feature_constraints: Feature constraints
            n_prototypes: Number of prototypes per class
        """
        super().__init__(model, feature_names, feature_constraints)
        self._training_data = training_data
        self._training_labels = training_labels
        self._n_prototypes = n_prototypes

        self._prototypes = self._compute_prototypes()

    def _compute_prototypes(self) -> dict[int, np.ndarray[Any, Any]]:
        """Compute class prototypes using k-means-like approach."""
        prototypes = {}
        unique_labels = np.unique(self._training_labels)

        for label in unique_labels:
            class_data = self._training_data[self._training_labels == label]

            if len(class_data) <= self._n_prototypes:
                prototypes[int(label)] = class_data
            else:
                indices = np.random.choice(
                    len(class_data),
                    self._n_prototypes,
                    replace=False,
                )
                prototypes[int(label)] = class_data[indices]

        return prototypes

    def generate(
        self,
        instance: np.ndarray[Any, Any],
        target_class: int | None = None,
        n_counterfactuals: int = 1,
    ) -> CounterfactualSet:
        """Generate prototype-based counterfactuals."""
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        original = instance[0].copy()
        original_pred = float(self._predict(original.reshape(1, -1))[0])

        if target_class is None:
            target_class = 1 if original_pred < 0.5 else 0

        if target_class not in self._prototypes:
            return CounterfactualSet(
                original=original,
                counterfactuals=[],
                original_prediction=original_pred,
                target_class=target_class,
                diversity_score=0.0,
                coverage_score=0.0,
            )

        target_prototypes = self._prototypes[target_class]
        distances = [self._compute_distance(original, p) for p in target_prototypes]
        sorted_indices = np.argsort(distances)

        counterfactuals = []
        for idx in sorted_indices[:n_counterfactuals]:
            prototype = target_prototypes[idx]

            cf = self._interpolate_to_boundary(original, prototype, target_class)

            if cf is not None:
                cf_pred = float(self._predict(cf.reshape(1, -1))[0])
                changes = self._get_feature_changes(original, cf)
                distance = self._compute_distance(original, cf)

                is_valid = (target_class == 1 and cf_pred >= 0.5) or (
                    target_class == 0 and cf_pred < 0.5
                )

                counterfactuals.append(
                    Counterfactual(
                        original=original,
                        counterfactual=cf,
                        original_prediction=original_pred,
                        counterfactual_prediction=cf_pred,
                        feature_changes=changes,
                        distance=distance,
                        validity=is_valid,
                        sparsity=len(changes),
                        feature_names=self._feature_names,
                    )
                )

        diversity = 0.0
        if len(counterfactuals) > 1:
            dist_pairs = []
            for i in range(len(counterfactuals)):
                for j in range(i + 1, len(counterfactuals)):
                    dist_pairs.append(
                        self._compute_distance(
                            counterfactuals[i].counterfactual,
                            counterfactuals[j].counterfactual,
                        )
                    )
            diversity = float(np.mean(dist_pairs))

        coverage = sum(1 for cf in counterfactuals if cf.validity) / max(1, n_counterfactuals)

        return CounterfactualSet(
            original=original,
            counterfactuals=counterfactuals,
            original_prediction=original_pred,
            target_class=target_class,
            diversity_score=diversity,
            coverage_score=coverage,
        )

    def _interpolate_to_boundary(
        self,
        original: np.ndarray[Any, Any],
        prototype: np.ndarray[Any, Any],
        target_class: int,
    ) -> np.ndarray[Any, Any] | None:
        """Interpolate between original and prototype to find boundary."""
        low = 0.0
        high = 1.0

        for _ in range(30):
            mid = (low + high) / 2
            point = original + mid * (prototype - original)
            pred = self._predict(point.reshape(1, -1))[0]

            if (target_class == 1 and pred >= 0.5) or (target_class == 0 and pred < 0.5):
                high = mid
            else:
                low = mid

        cf = original + high * (prototype - original)

        final_pred = self._predict(cf.reshape(1, -1))[0]
        if (target_class == 1 and final_pred >= 0.5) or (target_class == 0 and final_pred < 0.5):
            return cf

        return None


def create_counterfactual_generator(
    model: Any,
    method: str = "wachter",
    training_data: np.ndarray[Any, Any] | None = None,
    training_labels: np.ndarray[Any, Any] | None = None,
    feature_names: list[str] | None = None,
    feature_constraints: list[FeatureConstraint] | None = None,
    **kwargs: Any,
) -> CounterfactualGenerator:
    """
    Factory function to create counterfactual generator.

    Args:
        model: Model to explain
        method: "wachter", "dice", "growing_spheres", or "prototype"
        training_data: Training data (required for prototype method)
        training_labels: Training labels (required for prototype method)
        feature_names: Optional feature names
        feature_constraints: Optional feature constraints
        **kwargs: Additional method-specific parameters

    Returns:
        Counterfactual generator instance
    """
    method = method.lower()

    if method == "wachter":
        return WachterCounterfactual(
            model,
            feature_names,
            feature_constraints,
            **kwargs,
        )
    elif method == "dice":
        return DiCECounterfactual(
            model,
            feature_names,
            feature_constraints,
            **kwargs,
        )
    elif method == "growing_spheres":
        return GrowingSpheresCounterfactual(
            model,
            feature_names,
            feature_constraints,
            **kwargs,
        )
    elif method == "prototype":
        if training_data is None or training_labels is None:
            raise ValueError("Prototype method requires training data and labels")
        return PrototypeCounterfactual(
            model,
            training_data,
            training_labels,
            feature_names,
            feature_constraints,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown counterfactual method: {method}")
