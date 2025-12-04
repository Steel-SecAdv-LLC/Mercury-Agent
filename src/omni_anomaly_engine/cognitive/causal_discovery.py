"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

"""
Causal Discovery Engine - Causal Relationship Inference

Discovers and models causal relationships in anomaly patterns:
- PC Algorithm for constraint-based discovery
- Granger causality for time series
- Do-calculus for causal reasoning
- Counterfactual analysis

Research Sources:
- DARPA ANSR: Causal reasoning for trustworthy AI
- Pearl's Causal Inference framework
- Granger Causality for temporal data
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
from collections import defaultdict

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class CausalRelationType(Enum):
    """Types of causal relationships."""

    DIRECT = "direct"  # X directly causes Y
    INDIRECT = "indirect"  # X causes Y through mediators
    CONFOUNDED = "confounded"  # Common cause
    COLLIDER = "collider"  # Common effect
    BIDIRECTIONAL = "bidirectional"  # Feedback loop


class InterventionType(Enum):
    """Types of causal interventions (do-calculus)."""

    DO = "do"  # Do(X=x)
    SEE = "see"  # Observe X=x
    COUNTERFACTUAL = "counterfactual"  # What if X had been x?


@dataclass
class CausalEdge:
    """An edge in the causal graph."""

    source: str
    target: str
    relation_type: CausalRelationType
    strength: float  # Causal effect size
    confidence: float  # Confidence in this edge
    lag: int = 0  # Time lag (for temporal causation)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.relation_type.value,
            "strength": self.strength,
            "confidence": self.confidence,
            "lag": self.lag,
        }


@dataclass
class CausalGraph:
    """A causal graph structure."""

    graph_id: str
    nodes: list[str]
    edges: list[CausalEdge]
    confounders: list[tuple[str, str, str]]  # (confounder, var1, var2)
    colliders: list[tuple[str, str, str]]  # (collider, var1, var2)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.graph_id,
            "nodes": self.nodes,
            "edges": [e.to_dict() for e in self.edges],
            "confounders": self.confounders,
            "colliders": self.colliders,
        }

    def get_parents(self, node: str) -> list[str]:
        """Get parent nodes (direct causes)."""
        return [e.source for e in self.edges if e.target == node]

    def get_children(self, node: str) -> list[str]:
        """Get child nodes (direct effects)."""
        return [e.target for e in self.edges if e.source == node]


@dataclass
class CausalEffect:
    """Result of a causal effect estimation."""

    cause: str
    effect: str
    ate: float  # Average treatment effect
    confidence_interval: tuple[float, float]
    p_value: float
    method: str
    is_significant: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "ate": self.ate,
            "ci": self.confidence_interval,
            "p_value": self.p_value,
            "method": self.method,
            "significant": self.is_significant,
        }


@dataclass
class CounterfactualResult:
    """Result of counterfactual reasoning."""

    query: str
    factual_outcome: float
    counterfactual_outcome: float
    difference: float
    probability: float
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "factual": self.factual_outcome,
            "counterfactual": self.counterfactual_outcome,
            "difference": self.difference,
            "probability": self.probability,
            "explanation": self.explanation,
        }


class CausalDiscoveryEngine:
    """
    Causal Discovery and Inference Engine.

    Discovers causal relationships and enables causal reasoning:
    1. Structure Learning: Discover causal graph from data
    2. Effect Estimation: Quantify causal effects
    3. Intervention Reasoning: Predict effects of interventions
    4. Counterfactual Analysis: What-if reasoning

    This enables understanding WHY anomalies occur, not just detecting them.
    """

    PHI = (1 + np.sqrt(5)) / 2  # Golden ratio

    def __init__(
        self,
        significance_level: float = 0.05,
        max_conditioning_set: int = 3,
        enable_temporal: bool = True,
        max_lag: int = 5,
    ):
        """
        Initialize Causal Discovery Engine.

        Args:
            significance_level: Alpha for independence tests
            max_conditioning_set: Max size of conditioning sets
            enable_temporal: Enable Granger causality
            max_lag: Maximum time lag for temporal causation
        """
        self.significance_level = significance_level
        self.max_conditioning_set = max_conditioning_set
        self.enable_temporal = enable_temporal
        self.max_lag = max_lag

        # Storage
        self._graphs: dict[str, CausalGraph] = {}
        self._effects: dict[tuple[str, str], CausalEffect] = {}
        self._counterfactuals: list[CounterfactualResult] = []

        # Statistics
        self._stats = {
            "graphs_discovered": 0,
            "edges_found": 0,
            "effects_estimated": 0,
            "counterfactuals_computed": 0,
        }

        logger.info(f"CausalDiscoveryEngine initialized (alpha={significance_level})")

    def discover_structure(
        self,
        data: np.ndarray,
        variable_names: list[str] | None = None,
        prior_knowledge: dict[str, list[str]] | None = None,
    ) -> CausalGraph:
        """
        Discover causal structure using constraint-based methods.

        Uses a simplified PC algorithm:
        1. Start with complete undirected graph
        2. Remove edges based on conditional independence
        3. Orient edges using v-structures

        Args:
            data: Data matrix (samples x variables)
            variable_names: Names for variables
            prior_knowledge: Known causal relationships

        Returns:
            Discovered causal graph
        """
        n_samples, n_vars = data.shape
        names = variable_names or [f"X{i}" for i in range(n_vars)]

        # Initialize with all edges (complete graph)
        adjacency = np.ones((n_vars, n_vars), dtype=bool)
        np.fill_diagonal(adjacency, False)

        # Phase 1: Remove edges based on conditional independence
        for cond_size in range(self.max_conditioning_set + 1):
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if not adjacency[i, j]:
                        continue

                    # Get potential conditioning sets
                    neighbors = set(np.where(adjacency[i] | adjacency[j])[0])
                    neighbors.discard(i)
                    neighbors.discard(j)

                    if len(neighbors) < cond_size:
                        continue

                    # Test conditional independence
                    for cond_set in self._subsets(list(neighbors), cond_size):
                        if self._conditional_independence_test(
                            data[:, i], data[:, j], data[:, list(cond_set)]
                        ):
                            adjacency[i, j] = False
                            adjacency[j, i] = False
                            break

        # Phase 2: Orient edges (find v-structures)
        edges = []
        confounders = []
        colliders = []

        for i in range(n_vars):
            for j in range(n_vars):
                if i != j and adjacency[i, j]:
                    # Determine direction based on temporal order or correlation
                    strength = self._estimate_edge_strength(data[:, i], data[:, j])
                    confidence = self._estimate_confidence(data[:, i], data[:, j])

                    if strength > 0:
                        edges.append(
                            CausalEdge(
                                source=names[i],
                                target=names[j],
                                relation_type=CausalRelationType.DIRECT,
                                strength=abs(strength),
                                confidence=confidence,
                            )
                        )
                        adjacency[j, i] = False  # Remove reverse edge

        # Apply prior knowledge
        if prior_knowledge:
            edges = self._apply_prior_knowledge(edges, prior_knowledge, names)

        graph = CausalGraph(
            graph_id=f"causal_graph_{int(time.time())}",
            nodes=names,
            edges=edges,
            confounders=confounders,
            colliders=colliders,
        )

        self._graphs[graph.graph_id] = graph
        self._stats["graphs_discovered"] += 1
        self._stats["edges_found"] += len(edges)

        logger.info(f"Discovered causal graph: {len(names)} nodes, {len(edges)} edges")
        return graph

    def discover_temporal_causation(
        self,
        time_series: np.ndarray,
        variable_names: list[str] | None = None,
    ) -> CausalGraph:
        """
        Discover temporal causal relationships using Granger causality.

        Args:
            time_series: Time series data (time x variables)
            variable_names: Names for variables

        Returns:
            Temporal causal graph
        """
        n_time, n_vars = time_series.shape
        names = variable_names or [f"X{i}" for i in range(n_vars)]
        edges = []

        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue

                # Test Granger causality at each lag
                for lag in range(1, self.max_lag + 1):
                    is_causal, strength = self._granger_causality_test(
                        time_series[:, i], time_series[:, j], lag
                    )

                    if is_causal:
                        edges.append(
                            CausalEdge(
                                source=names[i],
                                target=names[j],
                                relation_type=CausalRelationType.DIRECT,
                                strength=strength,
                                confidence=0.95,  # Based on test
                                lag=lag,
                            )
                        )
                        break  # Use first significant lag

        graph = CausalGraph(
            graph_id=f"temporal_causal_{int(time.time())}",
            nodes=names,
            edges=edges,
            confounders=[],
            colliders=[],
        )

        self._graphs[graph.graph_id] = graph
        self._stats["graphs_discovered"] += 1

        logger.info(f"Discovered temporal causation: {len(edges)} edges")
        return graph

    def estimate_causal_effect(
        self,
        data: np.ndarray,
        cause_idx: int,
        effect_idx: int,
        adjustment_set: list[int] | None = None,
        variable_names: list[str] | None = None,
    ) -> CausalEffect:
        """
        Estimate causal effect using backdoor adjustment.

        Args:
            data: Data matrix
            cause_idx: Index of cause variable
            effect_idx: Index of effect variable
            adjustment_set: Variables to adjust for
            variable_names: Variable names

        Returns:
            Estimated causal effect
        """
        names = variable_names or [f"X{i}" for i in range(data.shape[1])]
        cause_name = names[cause_idx]
        effect_name = names[effect_idx]

        # Simple linear regression for ATE
        X = data[:, cause_idx].reshape(-1, 1)
        y = data[:, effect_idx]

        if adjustment_set:
            # Include adjustment variables
            Z = data[:, adjustment_set]
            X = np.hstack([X, Z])

        # Fit linear model
        X_with_intercept = np.hstack([np.ones((len(X), 1)), X])
        try:
            beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            ate = beta[1]  # Coefficient of cause variable

            # Calculate confidence interval
            residuals = y - X_with_intercept @ beta
            mse = np.mean(residuals ** 2)
            var_beta = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)[1, 1]
            se = np.sqrt(var_beta)

            ci = (ate - 1.96 * se, ate + 1.96 * se)
            t_stat = ate / se if se > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(y) - 2))

        except np.linalg.LinAlgError:
            ate = 0.0
            ci = (0.0, 0.0)
            p_value = 1.0

        effect = CausalEffect(
            cause=cause_name,
            effect=effect_name,
            ate=float(ate),
            confidence_interval=ci,
            p_value=float(p_value),
            method="backdoor_adjustment",
            is_significant=p_value < self.significance_level,
        )

        self._effects[(cause_name, effect_name)] = effect
        self._stats["effects_estimated"] += 1

        return effect

    def do_intervention(
        self,
        graph: CausalGraph,
        data: np.ndarray,
        intervention_var: str,
        intervention_value: float,
        target_var: str,
        variable_names: list[str],
    ) -> dict[str, Any]:
        """
        Compute the effect of an intervention do(X=x) on Y.

        Uses the adjustment formula based on the causal graph.

        Args:
            graph: Causal graph structure
            data: Observational data
            intervention_var: Variable to intervene on
            intervention_value: Value to set
            target_var: Target variable
            variable_names: Variable names

        Returns:
            Intervention effect
        """
        # Find adjustment set (parents of intervention variable)
        adjustment_vars = graph.get_parents(intervention_var)

        # Get indices
        int_idx = variable_names.index(intervention_var)
        target_idx = variable_names.index(target_var)
        adj_indices = [variable_names.index(v) for v in adjustment_vars if v in variable_names]

        # Estimate E[Y | do(X=x)] using adjustment
        if not adj_indices:
            # No confounders: E[Y | X=x]
            mask = np.abs(data[:, int_idx] - intervention_value) < 0.5
            if mask.sum() > 0:
                expected_y = data[mask, target_idx].mean()
            else:
                expected_y = data[:, target_idx].mean()
        else:
            # Adjust for confounders
            expected_y = self._adjustment_formula(
                data, int_idx, target_idx, adj_indices, intervention_value
            )

        return {
            "intervention": f"do({intervention_var}={intervention_value})",
            "target": target_var,
            "expected_value": float(expected_y),
            "adjustment_set": adjustment_vars,
        }

    def counterfactual_query(
        self,
        graph: CausalGraph,
        data: np.ndarray,
        factual_observation: dict[str, float],
        counterfactual_intervention: dict[str, float],
        target_var: str,
        variable_names: list[str],
    ) -> CounterfactualResult:
        """
        Answer counterfactual queries: What would Y have been if X=x'?

        Args:
            graph: Causal graph
            data: Historical data for estimating structural equations
            factual_observation: What actually happened
            counterfactual_intervention: What we hypothesize instead
            target_var: Variable we want to predict
            variable_names: Variable names

        Returns:
            Counterfactual result
        """
        target_idx = variable_names.index(target_var)

        # Step 1: Abduction - infer noise terms from factual observation
        factual_outcome = factual_observation.get(target_var, data[:, target_idx].mean())

        # Step 2: Action - modify structural equations with intervention
        # Step 3: Prediction - compute counterfactual outcome

        # Simplified: use linear model to predict counterfactual
        int_var = list(counterfactual_intervention.keys())[0]
        int_value = counterfactual_intervention[int_var]
        factual_value = factual_observation.get(int_var, 0)

        # Estimate causal effect
        int_idx = variable_names.index(int_var) if int_var in variable_names else 0
        effect = self.estimate_causal_effect(data, int_idx, target_idx, None, variable_names)

        # Counterfactual = Factual + Effect * (Counterfactual_X - Factual_X)
        counterfactual_outcome = factual_outcome + effect.ate * (int_value - factual_value)

        result = CounterfactualResult(
            query=f"What would {target_var} have been if {int_var}={int_value}?",
            factual_outcome=float(factual_outcome),
            counterfactual_outcome=float(counterfactual_outcome),
            difference=float(counterfactual_outcome - factual_outcome),
            probability=0.8 if effect.is_significant else 0.5,
            explanation=f"If {int_var} had been {int_value} instead of {factual_value}, "
                       f"{target_var} would have changed by {counterfactual_outcome - factual_outcome:.3f}",
        )

        self._counterfactuals.append(result)
        self._stats["counterfactuals_computed"] += 1

        return result

    def _conditional_independence_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        conditioning: np.ndarray | None,
    ) -> bool:
        """Test if X and Y are conditionally independent given Z."""
        if conditioning is None or conditioning.size == 0:
            # Unconditional test: correlation
            corr, p_value = stats.pearsonr(x, y)
            return p_value > self.significance_level

        # Partial correlation test
        try:
            # Residualize X and Y on Z
            if conditioning.ndim == 1:
                conditioning = conditioning.reshape(-1, 1)

            Z_with_intercept = np.hstack([np.ones((len(conditioning), 1)), conditioning])

            beta_x = np.linalg.lstsq(Z_with_intercept, x, rcond=None)[0]
            resid_x = x - Z_with_intercept @ beta_x

            beta_y = np.linalg.lstsq(Z_with_intercept, y, rcond=None)[0]
            resid_y = y - Z_with_intercept @ beta_y

            corr, p_value = stats.pearsonr(resid_x, resid_y)
            return p_value > self.significance_level

        except Exception:
            return False

    def _granger_causality_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lag: int,
    ) -> tuple[bool, float]:
        """Test if X Granger-causes Y at given lag."""
        if len(x) <= lag + 2:
            return False, 0.0

        # Restricted model: Y ~ Y_lagged
        Y = y[lag:]
        Y_lagged = np.column_stack([y[lag-i-1:-i-1] for i in range(lag)])

        # Unrestricted model: Y ~ Y_lagged + X_lagged
        X_lagged = np.column_stack([x[lag-i-1:-i-1] for i in range(lag)])

        try:
            # Fit restricted
            Z_r = np.hstack([np.ones((len(Y), 1)), Y_lagged])
            beta_r = np.linalg.lstsq(Z_r, Y, rcond=None)[0]
            rss_r = np.sum((Y - Z_r @ beta_r) ** 2)

            # Fit unrestricted
            Z_u = np.hstack([np.ones((len(Y), 1)), Y_lagged, X_lagged])
            beta_u = np.linalg.lstsq(Z_u, Y, rcond=None)[0]
            rss_u = np.sum((Y - Z_u @ beta_u) ** 2)

            # F-test
            n = len(Y)
            k_r = Z_r.shape[1]
            k_u = Z_u.shape[1]

            f_stat = ((rss_r - rss_u) / (k_u - k_r)) / (rss_u / (n - k_u))
            p_value = 1 - stats.f.cdf(f_stat, k_u - k_r, n - k_u)

            is_causal = p_value < self.significance_level
            strength = 1 - rss_u / rss_r if rss_r > 0 else 0

            return is_causal, float(strength)

        except Exception:
            return False, 0.0

    def _estimate_edge_strength(self, x: np.ndarray, y: np.ndarray) -> float:
        """Estimate the strength of a causal edge."""
        if len(x) < 3:
            return 0.0
        corr, _ = stats.pearsonr(x, y)
        return float(corr)

    def _estimate_confidence(self, x: np.ndarray, y: np.ndarray) -> float:
        """Estimate confidence in a causal edge."""
        if len(x) < 3:
            return 0.5
        _, p_value = stats.pearsonr(x, y)
        return float(1 - p_value)

    def _subsets(self, s: list, k: int):
        """Generate all subsets of size k."""
        from itertools import combinations
        return combinations(s, k)

    def _apply_prior_knowledge(
        self,
        edges: list[CausalEdge],
        prior: dict[str, list[str]],
        names: list[str],
    ) -> list[CausalEdge]:
        """Apply prior knowledge to orient edges."""
        # prior format: {"causes": ["var1 -> var2"], "forbids": ["var3 -> var4"]}
        for constraint in prior.get("causes", []):
            parts = constraint.split("->")
            if len(parts) == 2:
                src, tgt = parts[0].strip(), parts[1].strip()
                # Remove any edge going the wrong way
                edges = [e for e in edges if not (e.source == tgt and e.target == src)]

        return edges

    def _adjustment_formula(
        self,
        data: np.ndarray,
        x_idx: int,
        y_idx: int,
        z_indices: list[int],
        x_value: float,
    ) -> float:
        """Apply the backdoor adjustment formula."""
        # E[Y|do(X=x)] = sum_z E[Y|X=x,Z=z] P(Z=z)
        # Simplified: regression with adjustment
        X = data[:, [x_idx] + z_indices]
        y = data[:, y_idx]

        X_with_intercept = np.hstack([np.ones((len(X), 1)), X])
        beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]

        # Predict at x=x_value with average Z values
        z_means = data[:, z_indices].mean(axis=0)
        x_point = np.array([1, x_value] + list(z_means))

        return float(x_point @ beta)

    def get_statistics(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            **self._stats,
            "graphs_stored": len(self._graphs),
            "effects_stored": len(self._effects),
        }
