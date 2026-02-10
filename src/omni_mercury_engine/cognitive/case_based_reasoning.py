"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

from __future__ import annotations

"""
Case-Based Reasoning Engine

Implements case-based reasoning for learning from historical anomalies:
- Case retrieval: Find similar past cases
- Case adaptation: Modify solutions for new situations
- Case retention: Learn from new cases
- Experience-based decision making

Research Sources:
- Neuro-Symbolic AI Lab: Case-based reasoning
- CBR Cycle: Retrieve, Reuse, Revise, Retain
- Analogical reasoning for AI systems
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class CaseOutcome(Enum):
    """Outcomes of case resolutions."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    UNKNOWN = "unknown"


class SimilarityMetric(Enum):
    """Similarity metrics for case comparison."""

    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    MANHATTAN = "manhattan"
    WEIGHTED = "weighted"


@dataclass
class Case:
    """A case in the case base."""

    case_id: str
    problem_description: str
    problem_features: dict[str, Any]
    feature_vector: np.ndarray[Any, Any] | None
    solution: dict[str, Any]
    outcome: CaseOutcome
    outcome_score: float  # 0-1 success measure
    domain: str
    timestamp: float = field(default_factory=time.time)
    retrieval_count: int = 0
    adaptation_history: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.case_id,
            "problem": self.problem_description,
            "features": self.problem_features,
            "solution": self.solution,
            "outcome": self.outcome.value,
            "score": self.outcome_score,
            "domain": self.domain,
            "retrievals": self.retrieval_count,
        }


@dataclass
class RetrievalResult:
    """Result of case retrieval."""

    query_case: Case | dict[str, Any]
    retrieved_cases: list[tuple[Case, float]]  # (case, similarity)
    best_match: Case | None
    best_similarity: float
    retrieval_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_retrieved": len(self.retrieved_cases),
            "best_match": self.best_match.case_id if self.best_match else None,
            "best_similarity": self.best_similarity,
            "time_ms": self.retrieval_time_ms,
        }


@dataclass
class AdaptationResult:
    """Result of case adaptation."""

    source_case: Case
    adapted_solution: dict[str, Any]
    adaptations_made: list[str]
    confidence: float
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_case": self.source_case.case_id,
            "adapted_solution": self.adapted_solution,
            "adaptations": self.adaptations_made,
            "confidence": self.confidence,
        }


class CaseBasedReasoner:
    """
    Case-Based Reasoning Engine.

    Implements the CBR cycle:
    1. RETRIEVE: Find similar past cases
    2. REUSE: Apply solution from retrieved case
    3. REVISE: Adapt solution if needed
    4. RETAIN: Store successful cases for future use

    This enables learning from experience and analogical reasoning.
    """

    PHI = (1 + np.sqrt(5)) / 2  # Golden ratio

    def __init__(
        self,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        retrieval_threshold: float = 0.5,
        max_cases: int = 10000,
        enable_forgetting: bool = True,
        forgetting_threshold: float = 0.3,
    ):
        """
        Initialize Case-Based Reasoner.

        Args:
            similarity_metric: Metric for case comparison
            retrieval_threshold: Minimum similarity for retrieval
            max_cases: Maximum cases to store
            enable_forgetting: Allow removing low-utility cases
            forgetting_threshold: Threshold for forgetting
        """
        self.similarity_metric = similarity_metric
        self.retrieval_threshold = retrieval_threshold
        self.max_cases = max_cases
        self.enable_forgetting = enable_forgetting
        self.forgetting_threshold = forgetting_threshold

        # Case base
        self._cases: dict[str, Case] = {}
        self._domain_index: dict[str, list[str]] = {}
        self._feature_index: dict[str, list[str]] = {}

        # Feature weights for similarity (can be learned)
        self._feature_weights: dict[str, float] = {}

        # Statistics
        self._stats = {
            "cases_stored": 0,
            "retrievals": 0,
            "adaptations": 0,
            "successful_reuses": 0,
        }

        logger.info(f"CaseBasedReasoner initialized (metric={similarity_metric.value})")

    def add_case(self, case: Case) -> None:
        """
        Add a case to the case base (RETAIN).

        Args:
            case: Case to add
        """
        if len(self._cases) >= self.max_cases:
            self._forget_low_utility_cases()

        self._cases[case.case_id] = case

        # Update indices
        if case.domain not in self._domain_index:
            self._domain_index[case.domain] = []
        self._domain_index[case.domain].append(case.case_id)

        for feature in case.problem_features:
            if feature not in self._feature_index:
                self._feature_index[feature] = []
            self._feature_index[feature].append(case.case_id)

        self._stats["cases_stored"] += 1
        logger.debug(f"Added case: {case.case_id}")

    def retrieve(
        self,
        query: Case | dict[str, Any],
        k: int = 5,
        domain_filter: str | None = None,
    ) -> RetrievalResult:
        """
        Retrieve similar cases from the case base (RETRIEVE).

        Args:
            query: Query case or problem features
            k: Number of cases to retrieve
            domain_filter: Optional domain to filter by

        Returns:
            Retrieved cases with similarities
        """
        start_time = time.time()
        self._stats["retrievals"] += 1

        # Convert query to features
        if isinstance(query, Case):
            query_features = query.problem_features
            query_vector = query.feature_vector
        else:
            query_features = query
            query_vector = self._dict_to_vector(query)

        # Get candidate cases
        if domain_filter and domain_filter in self._domain_index:
            candidate_ids = self._domain_index[domain_filter]
        else:
            candidate_ids = list(self._cases.keys())

        # Compute similarities
        similarities: list[tuple[Case, float]] = []
        for case_id in candidate_ids:
            case = self._cases[case_id]
            similarity = self._compute_similarity(query_features, query_vector, case)
            if similarity >= self.retrieval_threshold:
                similarities.append((case, similarity))
                case.retrieval_count += 1

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]

        elapsed = (time.time() - start_time) * 1000

        return RetrievalResult(
            query_case=query,
            retrieved_cases=top_k,
            best_match=top_k[0][0] if top_k else None,
            best_similarity=top_k[0][1] if top_k else 0.0,
            retrieval_time_ms=elapsed,
        )

    def adapt(
        self,
        source_case: Case,
        target_problem: dict[str, Any],
        adaptation_rules: list[Callable[..., Any]] | None = None,
    ) -> AdaptationResult:
        """
        Adapt a retrieved case's solution to a new problem (REVISE).

        Args:
            source_case: Case to adapt from
            target_problem: New problem features
            adaptation_rules: Custom adaptation functions

        Returns:
            Adapted solution
        """
        self._stats["adaptations"] += 1

        adapted_solution = dict(source_case.solution)
        adaptations_made = []

        # Identify differences
        differences = self._identify_differences(source_case.problem_features, target_problem)

        # Apply adaptation rules
        if adaptation_rules:
            for rule in adaptation_rules:
                try:
                    result = rule(source_case, target_problem, adapted_solution)
                    if result:
                        adapted_solution.update(result)
                        adaptations_made.append(f"Applied rule: {rule.__name__}")
                except Exception as e:
                    logger.warning(f"Adaptation rule failed: {e}")

        # Default adaptations based on differences
        for feature, (old_val, new_val) in differences.items():
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                # Proportional adjustment
                if old_val != 0:
                    ratio = new_val / old_val
                    for sol_key, sol_val in adapted_solution.items():
                        if isinstance(sol_val, (int, float)):
                            # Scale related solution parameters
                            if feature in sol_key.lower():
                                adapted_solution[sol_key] = sol_val * ratio
                                adaptations_made.append(
                                    f"Scaled {sol_key} by {ratio:.2f} due to {feature} change"
                                )

        # Calculate confidence based on similarity and outcome history
        similarity = self._compute_similarity(target_problem, None, source_case)
        confidence = similarity * source_case.outcome_score

        # Record adaptation
        source_case.adaptation_history.append(f"Adapted for problem at {time.time()}")

        return AdaptationResult(
            source_case=source_case,
            adapted_solution=adapted_solution,
            adaptations_made=adaptations_made,
            confidence=confidence,
            explanation=f"Adapted from case {source_case.case_id} with {len(adaptations_made)} modifications",
        )

    def solve(
        self,
        problem: dict[str, Any],
        domain: str | None = None,
        k: int = 3,
    ) -> dict[str, Any]:
        """
        Complete CBR cycle: Retrieve, Reuse, Revise.

        Args:
            problem: Problem features
            domain: Problem domain
            k: Number of cases to consider

        Returns:
            Proposed solution with metadata
        """
        # RETRIEVE
        retrieval = self.retrieve(problem, k=k, domain_filter=domain)

        if not retrieval.best_match:
            return {
                "solution": None,
                "confidence": 0.0,
                "status": "no_matching_cases",
                "explanation": "No similar cases found in case base",
            }

        # REUSE best match
        best_case = retrieval.best_match
        best_similarity = retrieval.best_similarity

        if best_similarity > 0.9:
            # Very similar - reuse directly
            return {
                "solution": best_case.solution,
                "confidence": best_similarity * best_case.outcome_score,
                "status": "direct_reuse",
                "source_case": best_case.case_id,
                "similarity": best_similarity,
                "explanation": f"Directly reusing solution from highly similar case ({best_similarity:.2%})",
            }

        # REVISE - need adaptation
        adaptation = self.adapt(best_case, problem)

        return {
            "solution": adaptation.adapted_solution,
            "confidence": adaptation.confidence,
            "status": "adapted",
            "source_case": best_case.case_id,
            "similarity": best_similarity,
            "adaptations": adaptation.adaptations_made,
            "explanation": adaptation.explanation,
        }

    def learn_from_outcome(
        self,
        case_id: str,
        outcome: CaseOutcome,
        outcome_score: float,
        feedback: dict[str, Any] | None = None,
    ) -> None:
        """
        Learn from the outcome of a case solution (RETAIN enhancement).

        Args:
            case_id: Case that was used
            outcome: Outcome of the solution
            outcome_score: Success measure (0-1)
            feedback: Additional feedback
        """
        if case_id in self._cases:
            case = self._cases[case_id]
            case.outcome = outcome
            case.outcome_score = outcome_score

            if feedback:
                case.metadata["feedback"] = feedback

            if outcome == CaseOutcome.SUCCESS:
                self._stats["successful_reuses"] += 1

            logger.debug(f"Updated case {case_id} with outcome {outcome.value}")

    def _compute_similarity(
        self,
        query_features: dict[str, Any],
        query_vector: np.ndarray[Any, Any] | None,
        case: Case,
    ) -> float:
        """Compute similarity between query and case."""
        if self.similarity_metric == SimilarityMetric.EUCLIDEAN:
            return self._euclidean_similarity(query_features, case.problem_features)
        elif self.similarity_metric == SimilarityMetric.COSINE:
            return self._cosine_similarity(
                query_vector, case.feature_vector, query_features, case.problem_features
            )
        elif self.similarity_metric == SimilarityMetric.MANHATTAN:
            return self._manhattan_similarity(query_features, case.problem_features)
        else:  # WEIGHTED
            return self._weighted_similarity(query_features, case.problem_features)

    def _euclidean_similarity(
        self,
        features1: dict[str, Any],
        features2: dict[str, Any],
    ) -> float:
        """Compute Euclidean similarity."""
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0

        distances = []
        for key in common_keys:
            v1, v2 = features1[key], features2[key]
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                distances.append((v1 - v2) ** 2)
            elif v1 == v2:
                distances.append(0)
            else:
                distances.append(1)

        if not distances:
            return 0.0

        euclidean_dist = np.sqrt(np.sum(distances))
        # Convert distance to similarity
        return float(1.0 / (1.0 + euclidean_dist))

    def _cosine_similarity(
        self,
        vec1: np.ndarray[Any, Any] | None,
        vec2: np.ndarray[Any, Any] | None,
        features1: dict[str, Any],
        features2: dict[str, Any],
    ) -> float:
        """Compute cosine similarity."""
        if vec1 is not None and vec2 is not None:
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 > 0 and norm2 > 0:
                return float(np.dot(vec1, vec2) / (norm1 * norm2))

        # Fallback to feature-based
        return self._feature_cosine(features1, features2)

    def _feature_cosine(
        self,
        features1: dict[str, Any],
        features2: dict[str, Any],
    ) -> float:
        """Cosine similarity for feature dicts."""
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0

        v1, v2 = [], []
        for key in common_keys:
            val1, val2 = features1[key], features2[key]
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                v1.append(val1)
                v2.append(val2)

        if not v1:
            return 0.0

        v1, v2 = np.array(v1), np.array(v2)  # type: ignore[assignment, unused-ignore]
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 > 0 and norm2 > 0:
            return float(np.dot(v1, v2) / (norm1 * norm2))
        return 0.0

    def _manhattan_similarity(
        self,
        features1: dict[str, Any],
        features2: dict[str, Any],
    ) -> float:
        """Compute Manhattan similarity."""
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0

        distances = []
        for key in common_keys:
            v1, v2 = features1[key], features2[key]
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                distances.append(abs(v1 - v2))

        if not distances:
            return 0.0

        manhattan_dist = np.sum(distances)
        return float(1.0 / (1.0 + manhattan_dist))

    def _weighted_similarity(
        self,
        features1: dict[str, Any],
        features2: dict[str, Any],
    ) -> float:
        """Compute weighted similarity using learned feature weights."""
        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0

        weighted_sim = 0.0
        total_weight = 0.0

        for key in common_keys:
            weight = self._feature_weights.get(key, 1.0)
            v1, v2 = features1[key], features2[key]

            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                sim = 1.0 / (1.0 + abs(v1 - v2))
            elif v1 == v2:
                sim = 1.0
            else:
                sim = 0.0

            weighted_sim += weight * sim
            total_weight += weight

        return weighted_sim / total_weight if total_weight > 0 else 0.0

    def _identify_differences(
        self,
        source_features: dict[str, Any],
        target_features: dict[str, Any],
    ) -> dict[str, tuple[Any, Any]]:
        """Identify differences between source and target problems."""
        differences = {}

        for key in set(source_features.keys()) | set(target_features.keys()):
            source_val = source_features.get(key)
            target_val = target_features.get(key)

            if source_val != target_val:
                differences[key] = (source_val, target_val)

        return differences

    def _dict_to_vector(self, features: dict[str, Any]) -> np.ndarray[Any, Any] | None:
        """Convert feature dict to vector."""
        numeric_values = []
        for key in sorted(features.keys()):
            val = features[key]
            if isinstance(val, (int, float)):
                numeric_values.append(val)

        return np.array(numeric_values) if numeric_values else None

    def _forget_low_utility_cases(self) -> None:
        """Remove low-utility cases to make room."""
        if not self.enable_forgetting:
            return

        # Calculate utility scores
        utilities = []
        for case_id, case in self._cases.items():
            utility = self._calculate_utility(case)
            utilities.append((case_id, utility))

        # Sort and remove lowest utility
        utilities.sort(key=lambda x: x[1])
        num_to_remove = max(1, len(utilities) // 10)

        for case_id, utility in utilities[:num_to_remove]:
            if utility < self.forgetting_threshold:
                del self._cases[case_id]
                logger.debug(f"Forgot low-utility case: {case_id}")

    def _calculate_utility(self, case: Case) -> float:
        """Calculate utility score for a case."""
        # Factors: outcome, retrieval frequency, recency
        outcome_score = case.outcome_score
        retrieval_bonus = min(1.0, case.retrieval_count / 10)
        recency = 1.0 / (1.0 + (time.time() - case.timestamp) / (86400 * 30))  # 30-day decay

        return (outcome_score + retrieval_bonus + recency) / 3

    def get_statistics(self) -> dict[str, Any]:
        """Get reasoner statistics."""
        return {
            **self._stats,
            "total_cases": len(self._cases),
            "domains": list(self._domain_index.keys()),
            "success_rate": (
                self._stats["successful_reuses"] / self._stats["retrievals"]
                if self._stats["retrievals"] > 0
                else 0
            ),
        }
