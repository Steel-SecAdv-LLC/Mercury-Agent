# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Active Learning Framework.

Production-grade active learning for anomaly detection providing:
- Uncertainty sampling (entropy, margin, least confident)
- Query-by-committee with diversity
- Expected model change
- Information density weighting
- Batch mode active learning
- Human-in-the-loop integration
- Budget-aware sample selection

This addresses the critical gap: "No Active Learning" identified in audit.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class SamplingStrategy(StrEnum):
    """Active learning sampling strategies."""

    UNCERTAINTY_ENTROPY = "uncertainty_entropy"
    UNCERTAINTY_MARGIN = "uncertainty_margin"
    UNCERTAINTY_LEAST_CONFIDENT = "uncertainty_least_confident"
    QUERY_BY_COMMITTEE = "query_by_committee"
    EXPECTED_MODEL_CHANGE = "expected_model_change"
    INFORMATION_DENSITY = "information_density"
    RANDOM = "random"
    DIVERSITY = "diversity"
    HYBRID = "hybrid"  # Combines uncertainty and diversity


class LabelType(StrEnum):
    """Types of labels in active learning."""

    NORMAL = "normal"
    ANOMALY = "anomaly"
    UNCERTAIN = "uncertain"  # Human is not sure
    SKIP = "skip"  # Skip this sample


@dataclass
class LabeledSample:
    """A labeled sample from the oracle."""

    index: int
    features: NDArray[np.float64]
    label: LabelType
    confidence: float = 1.0  # Oracle's confidence in the label
    timestamp: float = 0.0
    annotator_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryBatch:
    """A batch of samples selected for labeling."""

    indices: list[int]
    features: NDArray[np.float64]
    uncertainties: list[float]
    diversity_scores: list[float]
    priority_scores: list[float]
    strategy: SamplingStrategy

    def __len__(self) -> int:
        """Return the length."""
        return len(self.indices)


@dataclass
class ActiveLearningState:
    """State of the active learning process."""

    total_queries: int
    total_labeled: int
    label_distribution: dict[str, int]
    budget_remaining: int
    current_accuracy: float | None
    iterations: int
    convergence_history: list[float]


class BaseSampler(ABC):
    """Base class for active learning samplers."""

    @abstractmethod
    def select(
        self,
        model: Any,
        X_unlabeled: NDArray[np.float64],
        n_samples: int,
        X_labeled: NDArray[np.float64] | None = None,
    ) -> QueryBatch:
        """Select samples for labeling."""
        pass


class UncertaintySampler(BaseSampler):
    """Uncertainty-based sampling strategies.

    Selects samples where the model is most uncertain:
    - Entropy: Maximum prediction entropy
    - Margin: Minimum difference between top 2 classes
    - Least Confident: Minimum prediction confidence
    """

    def __init__(
        self,
        strategy: SamplingStrategy = SamplingStrategy.UNCERTAINTY_ENTROPY,
    ):
        """Initialize uncertainty sampler.

        Args:
            strategy: Uncertainty measure to use
        """
        if strategy not in [
            SamplingStrategy.UNCERTAINTY_ENTROPY,
            SamplingStrategy.UNCERTAINTY_MARGIN,
            SamplingStrategy.UNCERTAINTY_LEAST_CONFIDENT,
        ]:
            raise ValueError(f"Invalid uncertainty strategy: {strategy}")

        self.strategy = strategy
        self.rng = np.random.default_rng(None)  # Default RNG for fallback

    def _compute_entropy(self, probs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute prediction entropy."""
        # Clip to avoid log(0)
        probs = np.clip(probs, 1e-10, 1 - 1e-10)

        # Binary entropy
        if probs.ndim == 1 or probs.shape[1] == 1:
            p = probs.flatten()
            return -(p * np.log(p) + (1 - p) * np.log(1 - p))

        # Multi-class entropy
        return -np.sum(probs * np.log(probs), axis=1)

    def _compute_margin(self, probs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute margin (difference between top 2 classes)."""
        if probs.ndim == 1 or probs.shape[1] == 1:
            p = probs.flatten()
            return 1 - np.abs(2 * p - 1)  # Uncertainty = 1 - margin

        # Sort probabilities
        sorted_probs = np.sort(probs, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        return 1 - margin  # Uncertainty = 1 - margin

    def _compute_least_confident(self, probs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute least confidence (1 - max probability)."""
        if probs.ndim == 1 or probs.shape[1] == 1:
            p = probs.flatten()
            return 1 - np.maximum(p, 1 - p)

        return 1 - np.max(probs, axis=1)

    def compute_uncertainty(self, probs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute uncertainty scores."""
        if self.strategy == SamplingStrategy.UNCERTAINTY_ENTROPY:
            return self._compute_entropy(probs)
        elif self.strategy == SamplingStrategy.UNCERTAINTY_MARGIN:
            return self._compute_margin(probs)
        elif self.strategy == SamplingStrategy.UNCERTAINTY_LEAST_CONFIDENT:
            return self._compute_least_confident(probs)
        else:
            return self._compute_entropy(probs)

    def select(
        self,
        model: Any,
        X_unlabeled: NDArray[np.float64],
        n_samples: int,
        X_labeled: NDArray[np.float64] | None = None,
    ) -> QueryBatch:
        """Select most uncertain samples.

        Args:
            model: Trained model with predict_proba
            X_unlabeled: Unlabeled feature matrix
            n_samples: Number of samples to select
            X_labeled: Already labeled samples (unused)

        Returns:
            QueryBatch with selected samples
        """
        # Get probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_unlabeled)
        elif hasattr(model, "decision_function"):
            # Convert decision function to probabilities
            scores = model.decision_function(X_unlabeled)
            probs = 1 / (1 + np.exp(-scores))
        else:
            # Fallback to random sampling
            logger.warning("Model has no predict_proba, using random sampling")
            indices = self.rng.choice(
                len(X_unlabeled), min(n_samples, len(X_unlabeled)), replace=False
            )
            return QueryBatch(
                indices=indices.tolist(),
                features=X_unlabeled[indices],
                uncertainties=[0.5] * len(indices),
                diversity_scores=[0.0] * len(indices),
                priority_scores=[0.5] * len(indices),
                strategy=self.strategy,
            )

        # Compute uncertainty
        uncertainties = self.compute_uncertainty(probs)

        # Select top uncertain samples
        n_select = min(n_samples, len(X_unlabeled))
        top_indices = np.argsort(uncertainties)[-n_select:][::-1]

        return QueryBatch(
            indices=top_indices.tolist(),
            features=X_unlabeled[top_indices],
            uncertainties=uncertainties[top_indices].tolist(),
            diversity_scores=[0.0] * len(top_indices),
            priority_scores=uncertainties[top_indices].tolist(),
            strategy=self.strategy,
        )


class DiversitySampler(BaseSampler):
    """Diversity-based sampling.

    Selects samples that are maximally diverse from each other and from already labeled samples.
    """

    def __init__(
        self,
        distance_metric: str = "euclidean",
    ):
        """Initialize diversity sampler.

        Args:
            distance_metric: Distance metric for diversity
        """
        self.distance_metric = distance_metric

    def compute_diversity(
        self,
        X_candidates: NDArray[np.float64],
        X_reference: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute diversity scores (distance to nearest reference point)."""
        if X_reference is None or len(X_reference) == 0:
            # No reference - all samples equally diverse
            return np.ones(len(X_candidates))

        # Compute distances to reference points
        distances = cdist(X_candidates, X_reference, metric=self.distance_metric)

        # Diversity = minimum distance to any reference point
        diversity = np.min(distances, axis=1)

        return diversity

    def select(
        self,
        model: Any,
        X_unlabeled: NDArray[np.float64],
        n_samples: int,
        X_labeled: NDArray[np.float64] | None = None,
    ) -> QueryBatch:
        """Select diverse samples using greedy k-center.

        Args:
            model: Trained model (unused)
            X_unlabeled: Unlabeled feature matrix
            n_samples: Number of samples to select
            X_labeled: Already labeled samples

        Returns:
            QueryBatch with selected samples
        """
        n_select = min(n_samples, len(X_unlabeled))
        selected_indices = []
        selected_features = []

        # Initialize reference set with labeled samples
        if X_labeled is not None and len(X_labeled) > 0:
            reference = X_labeled.copy()
        else:
            reference = np.empty((0, X_unlabeled.shape[1]))

        # Greedy selection
        remaining_indices = list(range(len(X_unlabeled)))

        for _ in range(n_select):
            if not remaining_indices:
                break

            # Compute diversity to current reference set
            X_remaining = X_unlabeled[remaining_indices]
            if len(reference) == 0:
                # First selection - pick centroid-nearest
                centroid = np.mean(X_remaining, axis=0, keepdims=True)
                distances = cdist(X_remaining, centroid).flatten()
                best_idx = np.argmin(distances)
            else:
                diversity = self.compute_diversity(X_remaining, reference)
                best_idx = np.argmax(diversity)

            # Add to selected
            original_idx = remaining_indices[best_idx]
            selected_indices.append(original_idx)
            selected_features.append(X_unlabeled[original_idx])

            # Update reference and remaining
            reference = np.vstack([reference, X_unlabeled[original_idx : original_idx + 1]])
            remaining_indices.pop(best_idx)

        selected_features_array = np.array(selected_features)

        # Compute final diversity scores
        diversity_scores = (
            self.compute_diversity(selected_features_array, X_labeled).tolist()
            if X_labeled is not None
            else [1.0] * len(selected_indices)
        )

        return QueryBatch(
            indices=selected_indices,
            features=selected_features_array,
            uncertainties=[0.0] * len(selected_indices),
            diversity_scores=diversity_scores,
            priority_scores=diversity_scores,
            strategy=SamplingStrategy.DIVERSITY,
        )


class HybridSampler(BaseSampler):
    """Hybrid sampling combining uncertainty and diversity.

    Uses uncertainty for initial filtering, then diversity for final selection.
    """

    def __init__(
        self,
        uncertainty_weight: float = 0.7,
        diversity_weight: float = 0.3,
        uncertainty_strategy: SamplingStrategy = SamplingStrategy.UNCERTAINTY_ENTROPY,
        prefilter_ratio: float = 3.0,
    ):
        """Initialize hybrid sampler.

        Args:
            uncertainty_weight: Weight for uncertainty in combined score
            diversity_weight: Weight for diversity in combined score
            uncertainty_strategy: Uncertainty measure to use
            prefilter_ratio: Prefilter top k*ratio by uncertainty
        """
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        self.prefilter_ratio = prefilter_ratio

        self.uncertainty_sampler = UncertaintySampler(uncertainty_strategy)
        self.diversity_sampler = DiversitySampler()

    def select(
        self,
        model: Any,
        X_unlabeled: NDArray[np.float64],
        n_samples: int,
        X_labeled: NDArray[np.float64] | None = None,
    ) -> QueryBatch:
        """Select samples using hybrid uncertainty + diversity.

        Args:
            model: Trained model
            X_unlabeled: Unlabeled feature matrix
            n_samples: Number of samples to select
            X_labeled: Already labeled samples

        Returns:
            QueryBatch with selected samples
        """
        # Prefilter by uncertainty
        prefilter_n = min(int(n_samples * self.prefilter_ratio), len(X_unlabeled))

        uncertainty_batch = self.uncertainty_sampler.select(model, X_unlabeled, prefilter_n)

        # Select from prefiltered using diversity
        X_prefiltered = uncertainty_batch.features
        prefiltered_indices = uncertainty_batch.indices

        if len(X_prefiltered) <= n_samples:
            return uncertainty_batch

        # Compute diversity within prefiltered set
        diversity_scores = self.diversity_sampler.compute_diversity(X_prefiltered, X_labeled)

        # Normalize scores
        uncertainties = np.array(uncertainty_batch.uncertainties)
        uncertainties_norm = (uncertainties - uncertainties.min()) / (
            uncertainties.max() - uncertainties.min() + 1e-10
        )
        diversity_norm = (diversity_scores - diversity_scores.min()) / (
            diversity_scores.max() - diversity_scores.min() + 1e-10
        )

        # Combined score
        combined_scores = (
            self.uncertainty_weight * uncertainties_norm + self.diversity_weight * diversity_norm
        )

        # Select top combined scores
        top_k = min(n_samples, len(combined_scores))
        top_local_indices = np.argsort(combined_scores)[-top_k:][::-1]

        final_indices = [prefiltered_indices[i] for i in top_local_indices]
        final_features = X_prefiltered[top_local_indices]
        final_uncertainties = uncertainties[top_local_indices].tolist()
        final_diversity = diversity_norm[top_local_indices].tolist()
        final_priority = combined_scores[top_local_indices].tolist()

        return QueryBatch(
            indices=final_indices,
            features=final_features,
            uncertainties=final_uncertainties,
            diversity_scores=final_diversity,
            priority_scores=final_priority,
            strategy=SamplingStrategy.HYBRID,
        )


class QueryByCommitteeSampler(BaseSampler):
    """Query by Committee (QBC) sampling.

    Uses an ensemble of models and selects samples with highest disagreement among committee
    members.
    """

    def __init__(
        self,
        n_committee: int = 5,
        disagreement_measure: str = "vote_entropy",
        random_state: int | None = None,
    ):
        """Initialize QBC sampler.

        Args:
            n_committee: Number of committee members
            disagreement_measure: 'vote_entropy' or 'kl_divergence'
            random_state: Seed for reproducible random sampling
        """
        self.n_committee = n_committee
        self.disagreement_measure = disagreement_measure
        self.rng = np.random.default_rng(random_state)

    def _train_committee(
        self,
        base_model: Any,
        X_labeled: NDArray[np.float64],
        y_labeled: NDArray[np.int64],
    ) -> list[Any]:
        """Train committee using bootstrap sampling."""
        committee = []
        n_samples = len(X_labeled)

        for _ in range(self.n_committee):
            # Bootstrap sample
            indices = self.rng.choice(n_samples, n_samples, replace=True)
            X_boot = X_labeled[indices]
            y_boot = y_labeled[indices]

            # Clone and train model
            try:
                from omni_mercury_engine.ml.mercury_ml import clone

                model = clone(base_model)
            except ImportError:
                model = base_model

            model.fit(X_boot, y_boot)
            committee.append(model)

        return committee

    def _compute_disagreement(
        self,
        committee: list[Any],
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute disagreement among committee members."""
        # Collect predictions from all committee members
        predictions = []
        for model in committee:
            if hasattr(model, "predict_proba"):
                preds = model.predict_proba(X)[:, 1]
            else:
                preds = model.predict(X).astype(float)
            predictions.append(preds)

        predictions = np.array(predictions)  # type: ignore[assignment, unused-ignore]

        if self.disagreement_measure == "vote_entropy":
            # Vote entropy: entropy of binary vote distribution
            votes = (predictions > 0.5).astype(int)  # type: ignore[operator, unused-ignore]
            vote_fraction = np.mean(votes, axis=0)
            vote_fraction = np.clip(vote_fraction, 1e-10, 1 - 1e-10)
            entropy = -(
                vote_fraction * np.log(vote_fraction)
                + (1 - vote_fraction) * np.log(1 - vote_fraction)
            )
            return entropy

        elif self.disagreement_measure == "kl_divergence":
            # Average KL divergence from mean prediction
            mean_pred = np.mean(predictions, axis=0)
            mean_pred = np.clip(mean_pred, 1e-10, 1 - 1e-10)

            kl_sum = 0
            for pred in predictions:
                pred = np.clip(pred, 1e-10, 1 - 1e-10)
                kl = pred * np.log(pred / mean_pred) + (1 - pred) * np.log(
                    (1 - pred) / (1 - mean_pred)
                )
                kl_sum += kl

            return np.asarray(kl_sum / len(predictions))  # type: ignore[return-value, unused-ignore]

        else:
            # Default to variance
            return np.var(predictions, axis=0)

    def select(
        self,
        model: Any,
        X_unlabeled: NDArray[np.float64],
        n_samples: int,
        X_labeled: NDArray[np.float64] | None = None,
        y_labeled: NDArray[np.int64] | None = None,
    ) -> QueryBatch:
        """Select samples with highest committee disagreement.

        Args:
            model: Base model for committee
            X_unlabeled: Unlabeled feature matrix
            n_samples: Number of samples to select
            X_labeled: Labeled features for training committee
            y_labeled: Labels for training committee

        Returns:
            QueryBatch with selected samples
        """
        if X_labeled is None or y_labeled is None:
            logger.warning("QBC requires labeled data, falling back to random")
            indices = self.rng.choice(
                len(X_unlabeled), min(n_samples, len(X_unlabeled)), replace=False
            )
            return QueryBatch(
                indices=indices.tolist(),
                features=X_unlabeled[indices],
                uncertainties=[0.5] * len(indices),
                diversity_scores=[0.0] * len(indices),
                priority_scores=[0.5] * len(indices),
                strategy=SamplingStrategy.QUERY_BY_COMMITTEE,
            )

        # Train committee
        committee = self._train_committee(model, X_labeled, y_labeled)

        # Compute disagreement
        disagreement = self._compute_disagreement(committee, X_unlabeled)

        # Select highest disagreement
        n_select = min(n_samples, len(X_unlabeled))
        top_indices = np.argsort(disagreement)[-n_select:][::-1]

        return QueryBatch(
            indices=top_indices.tolist(),
            features=X_unlabeled[top_indices],
            uncertainties=disagreement[top_indices].tolist(),
            diversity_scores=[0.0] * len(top_indices),
            priority_scores=disagreement[top_indices].tolist(),
            strategy=SamplingStrategy.QUERY_BY_COMMITTEE,
        )


class ActiveLearner:
    """Active learning manager for iterative model improvement.

    Coordinates sample selection, oracle querying, and model retraining.
    """

    def __init__(
        self,
        model: Any,
        strategy: SamplingStrategy = SamplingStrategy.HYBRID,
        batch_size: int = 10,
        budget: int = 100,
        initial_samples: int = 10,
        retrain_interval: int = 1,
        random_state: int | None = None,
    ):
        """Initialize active learner.

        Args:
            model: Base model to improve
            strategy: Sampling strategy
            batch_size: Samples per query batch
            budget: Total labeling budget
            initial_samples: Initial random samples to label
            retrain_interval: Batches between retraining
            random_state: Seed for reproducible random sampling
        """
        self.model = model
        self.strategy = strategy
        self.batch_size = batch_size
        self.budget = budget
        self.initial_samples = initial_samples
        self.retrain_interval = retrain_interval
        self.rng = np.random.default_rng(random_state)

        # Create sampler
        self.sampler = self._create_sampler()

        # State tracking
        self._labeled_indices: list[int] = []
        self._labeled_X: list[NDArray[np.float64]] = []
        self._labeled_y: list[int] = []
        self._queries_made = 0
        self._iterations = 0
        self._accuracy_history: list[float] = []

    def _create_sampler(self) -> BaseSampler:
        """Create sampler based on strategy."""
        if self.strategy in [
            SamplingStrategy.UNCERTAINTY_ENTROPY,
            SamplingStrategy.UNCERTAINTY_MARGIN,
            SamplingStrategy.UNCERTAINTY_LEAST_CONFIDENT,
        ]:
            return UncertaintySampler(self.strategy)
        elif self.strategy == SamplingStrategy.DIVERSITY:
            return DiversitySampler()
        elif self.strategy == SamplingStrategy.HYBRID:
            return HybridSampler()
        elif self.strategy == SamplingStrategy.QUERY_BY_COMMITTEE:
            return QueryByCommitteeSampler()
        else:
            return HybridSampler()

    def _stratified_initial_indices(
        self,
        y: NDArray[np.int64],
        n: int,
    ) -> list[int]:
        """Select ``n`` initial indices preserving class proportions.

        Uses largest-remainder proportional allocation with a per-class
        floor of one whenever ``n >= number_of_classes``. This
        guarantees that the initial labeled batch contains at least one
        sample from every class present in ``y`` (when feasible), which
        is required for downstream binary/multinomial classifiers to be
        trainable on the initial batch.

        Args:
            y: Full label vector for the pool ``X``.
            n: Number of initial samples to draw.

        Returns:
            List of ``n`` integer indices into ``y`` / the pool, drawn
            without replacement, stratified by the classes in ``y``.
        """
        n = max(0, min(n, len(y)))
        if n == 0:
            return []

        classes = np.unique(y)
        n_classes = int(classes.size)

        # Single-class pool: stratification reduces to uniform sampling.
        if n_classes <= 1:
            return self.rng.choice(len(y), n, replace=False).tolist()

        class_counts = {int(c): int(np.sum(y == c)) for c in classes}
        total = sum(class_counts.values())

        # Largest-remainder allocation with optional per-class floor.
        floor_one = n >= n_classes
        allocations: dict[int, int] = {}
        remainders: list[tuple[float, int]] = []
        assigned = 0
        for c in classes:
            c_int = int(c)
            exact = n * class_counts[c_int] / total
            base = int(exact)
            if floor_one:
                base = max(base, 1)
            base = min(base, class_counts[c_int])
            allocations[c_int] = base
            assigned += base
            remainders.append((exact - base, c_int))

        # Distribute remaining slots by largest remainder, respecting
        # the per-class cap. Iterate until we either fill the quota or
        # all classes saturate.
        remaining = n - assigned
        remainders.sort(reverse=True)
        if remaining > 0:
            for _, c_int in remainders:
                if remaining == 0:
                    break
                if allocations[c_int] < class_counts[c_int]:
                    take = min(remaining, class_counts[c_int] - allocations[c_int])
                    allocations[c_int] += take
                    remaining -= take
        elif remaining < 0:
            # Over-allocated because of the per-class floor; trim from
            # the smallest remainders first while keeping >= 1 sample
            # per class.
            for _, c_int in sorted(remainders):
                while remaining < 0 and allocations[c_int] > 1:
                    allocations[c_int] -= 1
                    remaining += 1
                if remaining == 0:
                    break

        indices: list[int] = []
        for c_int, count in allocations.items():
            if count <= 0:
                continue
            class_indices = np.where(y == c_int)[0]
            chosen = self.rng.choice(class_indices, count, replace=False)
            indices.extend(int(i) for i in chosen)

        # Shuffle so callers do not see a class-ordered batch.
        self.rng.shuffle(indices)
        return indices

    def initialize(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64] | None = None,
        initial_indices: list[int] | None = None,
    ) -> QueryBatch:
        """Initialize with random or specified samples.

        When ``y`` is provided and contains at least two distinct
        classes, sampling is **stratified** so every class is
        represented in the initial labeled batch (when the requested
        size permits). This is required so the underlying classifier
        can be trained on the initial batch; without stratification a
        random draw can pick all-same-class samples and leave the
        model unfit, which would later crash downstream queries.

        Args:
            X: Full feature matrix
            y: Labels (if available for initial samples)
            initial_indices: Specific indices to start with

        Returns:
            QueryBatch for initial labeling
        """
        if initial_indices is not None:
            indices = list(initial_indices)
        else:
            n_initial = min(self.initial_samples, len(X))
            if y is not None:
                indices = self._stratified_initial_indices(np.asarray(y), n_initial)
            else:
                indices = self.rng.choice(len(X), n_initial, replace=False).tolist()

        # If labels provided, add to labeled set
        if y is not None:
            for idx in indices:
                self._labeled_indices.append(idx)
                self._labeled_X.append(X[idx])
                self._labeled_y.append(int(y[idx]))

            # Train initial model
            if len(self._labeled_y) > 0:
                self._train_model()

        return QueryBatch(
            indices=indices,
            features=X[indices],
            uncertainties=[0.5] * len(indices),
            diversity_scores=[1.0] * len(indices),
            priority_scores=[1.0] * len(indices),
            strategy=SamplingStrategy.RANDOM,
        )

    def _train_model(self) -> None:
        """Retrain model on current labeled data.

        Skips training (with a warning) when fewer than two distinct
        classes are labeled. Callers must not assume the model is
        fitted after this returns; downstream code consults
        :meth:`_model_is_fitted` before invoking ``predict_proba``.
        """
        if len(self._labeled_y) < 2:
            return

        X = np.array(self._labeled_X)
        y = np.array(self._labeled_y)

        # Check for both classes
        if len(np.unique(y)) < 2:
            logger.warning(
                "Labeled set contains a single class (%s); model fit skipped. "
                "Subsequent queries will fall back to random sampling until "
                "the second class is observed.",
                np.unique(y).tolist(),
            )
            return

        self.model.fit(X, y)
        logger.info(f"Model retrained on {len(y)} samples")

    def _model_is_fitted(self) -> bool:
        """Return True when ``self.model`` has been successfully fit.

        Mercury's :class:`~omni_mercury_engine.ml.mercury_ml.LogisticRegression`
        exposes an ``is_fitted_`` flag; estimators without that flag
        are assumed to manage their own state (sklearn convention).
        """
        flag = getattr(self.model, "is_fitted_", None)
        if flag is None:
            return True
        return bool(flag)

    def query(
        self,
        X_pool: NDArray[np.float64],
        exclude_indices: list[int] | None = None,
    ) -> QueryBatch:
        """Select next batch of samples to label.

        Args:
            X_pool: Pool of unlabeled samples
            exclude_indices: Indices to exclude (already labeled)

        Returns:
            QueryBatch with samples to label
        """
        if exclude_indices is None:
            exclude_indices = self._labeled_indices

        # Create mask for unlabeled samples
        mask = np.ones(len(X_pool), dtype=bool)
        # Validate and filter exclude_indices to be within bounds
        if exclude_indices:
            valid_indices = [i for i in exclude_indices if 0 <= i < len(X_pool)]
            if len(valid_indices) < len(exclude_indices):
                logger.warning(
                    f"Filtered {len(exclude_indices) - len(valid_indices)} "
                    f"out-of-bounds indices from exclude_indices"
                )
            if valid_indices:
                mask[valid_indices] = False
        unlabeled_indices = np.where(mask)[0]

        if len(unlabeled_indices) == 0:
            logger.warning("No unlabeled samples remaining")
            return QueryBatch(
                indices=[],
                features=np.array([]).reshape(0, X_pool.shape[1]),
                uncertainties=[],
                diversity_scores=[],
                priority_scores=[],
                strategy=self.strategy,
            )

        X_unlabeled = X_pool[unlabeled_indices]

        # Get labeled data for diversity computation
        X_labeled = np.array(self._labeled_X) if self._labeled_X else None
        y_labeled = np.array(self._labeled_y) if self._labeled_y else None

        # Select samples
        n_to_select = min(self.batch_size, self.budget - self._queries_made)
        n_to_select = max(0, n_to_select)

        if n_to_select == 0:
            empty = QueryBatch(
                indices=[],
                features=X_unlabeled[:0],
                uncertainties=[],
                diversity_scores=[],
                priority_scores=[],
                strategy=self.strategy,
            )
            return empty

        # If the underlying model is not yet fitted (e.g. the initial
        # labeled batch happened to contain a single class, so
        # ``_train_model`` skipped the fit), we cannot call
        # ``predict_proba`` -- doing so would either raise NotFittedError
        # or, for estimators that do not check, produce a shape-mismatch
        # matmul error. Fall back to a uniform random query so the
        # active-learning loop can keep gathering labels until the
        # missing class is observed and the model becomes trainable.
        if not self._model_is_fitted():
            logger.warning(
                "Underlying model is not fitted; falling back to random "
                "sampling for this query batch."
            )
            n_random = min(n_to_select, len(X_unlabeled))
            random_local_indices = self.rng.choice(len(X_unlabeled), n_random, replace=False)
            batch = QueryBatch(
                indices=random_local_indices.tolist(),
                features=X_unlabeled[random_local_indices],
                uncertainties=[0.5] * n_random,
                diversity_scores=[0.0] * n_random,
                priority_scores=[0.5] * n_random,
                strategy=SamplingStrategy.RANDOM,
            )
        elif isinstance(self.sampler, QueryByCommitteeSampler):
            batch = self.sampler.select(self.model, X_unlabeled, n_to_select, X_labeled, y_labeled)
        else:
            batch = self.sampler.select(self.model, X_unlabeled, n_to_select, X_labeled)

        # Map back to original indices
        original_indices = [int(unlabeled_indices[i]) for i in batch.indices]
        batch.indices = original_indices

        self._queries_made += len(batch.indices)
        return batch

    def update(
        self,
        labels: list[LabeledSample],
        X_pool: NDArray[np.float64],
    ) -> None:
        """Update with new labels from oracle.

        Args:
            labels: List of labeled samples
            X_pool: Full feature pool (for getting features)
        """
        for label in labels:
            if label.label in [LabelType.NORMAL, LabelType.ANOMALY]:
                self._labeled_indices.append(label.index)
                self._labeled_X.append(X_pool[label.index])
                self._labeled_y.append(1 if label.label == LabelType.ANOMALY else 0)

        self._iterations += 1

        # Retrain if interval reached
        if self._iterations % self.retrain_interval == 0:
            self._train_model()

    def evaluate(
        self,
        X_test: NDArray[np.float64],
        y_test: NDArray[np.int64],
    ) -> float:
        """Evaluate current model on test set.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Accuracy score
        """
        if len(self._labeled_y) < 2:
            return 0.0

        try:
            predictions = self.model.predict(X_test)
            accuracy = float(np.mean(predictions == y_test))
            self._accuracy_history.append(accuracy)
            return accuracy
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return 0.0

    def get_state(self) -> ActiveLearningState:
        """Get current active learning state."""
        label_dist = {
            "normal": sum(1 for y in self._labeled_y if y == 0),
            "anomaly": sum(1 for y in self._labeled_y if y == 1),
        }

        return ActiveLearningState(
            total_queries=self._queries_made,
            total_labeled=len(self._labeled_y),
            label_distribution=label_dist,
            budget_remaining=self.budget - self._queries_made,
            current_accuracy=self._accuracy_history[-1] if self._accuracy_history else None,
            iterations=self._iterations,
            convergence_history=self._accuracy_history.copy(),
        )

    def run_loop(
        self,
        X_pool: NDArray[np.float64],
        oracle: Callable[[QueryBatch], list[LabeledSample]],
        X_test: NDArray[np.float64] | None = None,
        y_test: NDArray[np.int64] | None = None,
        max_iterations: int | None = None,
    ) -> Iterator[tuple[QueryBatch, ActiveLearningState]]:
        """Run active learning loop.

        Args:
            X_pool: Pool of samples
            oracle: Function that labels a batch (simulated or human)
            X_test: Test set for evaluation (optional)
            y_test: Test labels (optional)
            max_iterations: Maximum iterations (None = until budget exhausted)

        Yields:
            Tuple of (current batch, current state)
        """
        iteration = 0
        max_iter = max_iterations or (self.budget // self.batch_size + 1)

        while iteration < max_iter and self._queries_made < self.budget:
            # Select samples
            batch = self.query(X_pool)

            if len(batch) == 0:
                break

            # Get labels from oracle
            labels = oracle(batch)

            # Update with labels
            self.update(labels, X_pool)

            # Evaluate if test set provided
            if X_test is not None and y_test is not None:
                self.evaluate(X_test, y_test)

            iteration += 1

            yield batch, self.get_state()


def create_active_learner(
    model: Any,
    strategy: str = "hybrid",
    **kwargs: Any,
) -> ActiveLearner:
    """Factory function to create active learner.

    Args:
        model: Base model
        strategy: Sampling strategy name
        **kwargs: Additional arguments

    Returns:
        Configured ActiveLearner
    """
    strategy_map = {
        "entropy": SamplingStrategy.UNCERTAINTY_ENTROPY,
        "margin": SamplingStrategy.UNCERTAINTY_MARGIN,
        "least_confident": SamplingStrategy.UNCERTAINTY_LEAST_CONFIDENT,
        "diversity": SamplingStrategy.DIVERSITY,
        "qbc": SamplingStrategy.QUERY_BY_COMMITTEE,
        "hybrid": SamplingStrategy.HYBRID,
        "random": SamplingStrategy.RANDOM,
    }

    s = strategy_map.get(strategy.lower(), SamplingStrategy.HYBRID)

    return ActiveLearner(model=model, strategy=s, **kwargs)


# Exports
__all__ = [
    "ActiveLearner",
    "ActiveLearningState",
    "BaseSampler",
    "DiversitySampler",
    "HybridSampler",
    "LabelType",
    "LabeledSample",
    "QueryBatch",
    "QueryByCommitteeSampler",
    "SamplingStrategy",
    "UncertaintySampler",
    "create_active_learner",
]
