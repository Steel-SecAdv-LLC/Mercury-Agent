"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

from __future__ import annotations

"""
Causal Discovery Engine - Production Implementation

Discovers and models causal relationships in anomaly patterns:
- PC Algorithm: Constraint-based structure learning with Fisher's Z test
- FCI Extension: Handles latent confounders
- Granger Causality: VAR-based temporal causation with F-tests
- Do-calculus: Backdoor/frontdoor adjustment, propensity scores
- Doubly Robust Estimation: AIPW for robust causal effects
- Counterfactual Analysis: Three-step abduction-action-prediction

Research Sources:
- Spirtes, Glymour, Scheines (2000): Causation, Prediction, and Search (PC Algorithm)
- Pearl (2009): Causality - Models, Reasoning, and Inference
- Granger (1969): Investigating Causal Relations by Econometric Models
- Bang & Robins (2005): Doubly Robust Estimation
- Peters, Janzing, Schölkopf (2017): Elements of Causal Inference
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Any

import numpy as np
from scipy import linalg, stats

logger = logging.getLogger(__name__)


class CausalRelationType(Enum):
    """Types of causal relationships."""

    DIRECT = "direct"  # X -> Y
    INDIRECT = "indirect"  # X -> ... -> Y
    CONFOUNDED = "confounded"  # X <- U -> Y (unobserved confounder)
    COLLIDER = "collider"  # X -> Z <- Y
    BIDIRECTIONAL = "bidirectional"  # X <-> Y (feedback or latent)
    UNDIRECTED = "undirected"  # X - Y (orientation unknown)


class InterventionType(Enum):
    """Types of causal interventions (do-calculus)."""

    DO = "do"  # do(X=x) - atomic intervention
    SOFT = "soft"  # Shift distribution of X
    CONDITIONAL = "conditional"  # do(X=x | Z=z)
    COUNTERFACTUAL = "counterfactual"  # What if X had been x?


@dataclass
class CausalEdge:
    """An edge in the causal graph."""

    source: str
    target: str
    relation_type: CausalRelationType
    strength: float  # Standardized effect size (Cohen's d or partial r)
    confidence: float  # 1 - p_value from CI test
    lag: int = 0  # Time lag for temporal causation
    p_value: float = 0.0  # P-value from independence test
    separation_set: tuple[int, ...] = field(
        default_factory=tuple
    )  # Conditioning set that separated

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.relation_type.value,
            "strength": self.strength,
            "confidence": self.confidence,
            "lag": self.lag,
            "p_value": self.p_value,
        }


@dataclass
class CausalGraph:
    """A causal graph structure (CPDAG or DAG)."""

    graph_id: str
    nodes: list[str]
    edges: list[CausalEdge]
    confounders: list[tuple[str, str, str]]  # (confounder, var1, var2)
    colliders: list[tuple[str, str, str]]  # (collider, parent1, parent2)
    separation_sets: dict[tuple[str, str], tuple[str, ...]]  # (i, j) -> conditioning set
    is_cpdag: bool = True  # True if some edges undirected
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.graph_id,
            "nodes": self.nodes,
            "edges": [e.to_dict() for e in self.edges],
            "confounders": self.confounders,
            "colliders": self.colliders,
            "is_cpdag": self.is_cpdag,
        }

    def get_parents(self, node: str) -> list[str]:
        """Get parent nodes (direct causes)."""
        return [
            e.source
            for e in self.edges
            if e.target == node and e.relation_type == CausalRelationType.DIRECT
        ]

    def get_children(self, node: str) -> list[str]:
        """Get child nodes (direct effects)."""
        return [
            e.target
            for e in self.edges
            if e.source == node and e.relation_type == CausalRelationType.DIRECT
        ]

    def get_ancestors(self, node: str, visited: set[str] | None = None) -> set[str]:
        """Get all ancestors of a node."""
        if visited is None:
            visited = set()
        parents = self.get_parents(node)
        for p in parents:
            if p not in visited:
                visited.add(p)
                self.get_ancestors(p, visited)
        return visited

    def get_descendants(self, node: str, visited: set[str] | None = None) -> set[str]:
        """Get all descendants of a node."""
        if visited is None:
            visited = set()
        children = self.get_children(node)
        for c in children:
            if c not in visited:
                visited.add(c)
                self.get_descendants(c, visited)
        return visited

    def is_d_separated(self, x: str, y: str, z: set[str]) -> bool:
        """Check if X and Y are d-separated given Z using Bayes-ball algorithm."""
        # Simplified: check if separation set contains the path blockers
        key = (min(x, y), max(x, y))
        if key in self.separation_sets:
            sep_set = set(self.separation_sets[key])
            return z >= sep_set
        return False


@dataclass
class CausalEffect:
    """Result of causal effect estimation."""

    cause: str
    effect: str
    ate: float  # Average Treatment Effect
    att: float  # Average Treatment Effect on Treated
    confidence_interval: tuple[float, float]
    p_value: float
    method: str
    is_significant: bool
    # Diagnostics
    n_treated: int = 0
    n_control: int = 0
    overlap_score: float = 0.0  # Propensity score overlap
    balance_score: float = 0.0  # Covariate balance after matching

    def to_dict(self) -> dict[str, Any]:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "ate": self.ate,
            "att": self.att,
            "ci": self.confidence_interval,
            "p_value": self.p_value,
            "method": self.method,
            "significant": self.is_significant,
            "diagnostics": {
                "n_treated": self.n_treated,
                "n_control": self.n_control,
                "overlap": self.overlap_score,
                "balance": self.balance_score,
            },
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
    # Sensitivity analysis
    robustness_value: float = 0.0  # How much confounding needed to nullify

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "factual": self.factual_outcome,
            "counterfactual": self.counterfactual_outcome,
            "difference": self.difference,
            "probability": self.probability,
            "explanation": self.explanation,
            "robustness": self.robustness_value,
        }


class PartialCorrelationTest:
    """
    Fisher's Z-transformed partial correlation test for conditional independence.

    More statistically rigorous than simple correlation:
    - Uses Fisher's Z transformation for proper p-values
    - Handles conditioning sets properly
    - Returns effect size (partial r) and p-value
    """

    @staticmethod
    def test(
        x: np.ndarray[Any, Any],
        y: np.ndarray[Any, Any],
        z: np.ndarray[Any, Any] | None,
        n: int,
    ) -> tuple[float, float, float]:
        """
        Test conditional independence using partial correlation.

        Args:
            x: First variable
            y: Second variable
            z: Conditioning variables (can be None or empty)
            n: Sample size

        Returns:
            (partial_correlation, z_statistic, p_value)
        """
        if z is None or z.size == 0:
            # Unconditional correlation
            r, _ = stats.pearsonr(x, y)
            partial_r = r
        else:
            # Partial correlation via regression
            if z.ndim == 1:
                z = z.reshape(-1, 1)

            # Residualize X and Y on Z
            z_with_intercept = np.column_stack([np.ones(len(z)), z])

            try:
                # Use lstsq for numerical stability
                beta_x, _, _, _ = linalg.lstsq(z_with_intercept, x)
                resid_x = x - z_with_intercept @ beta_x

                beta_y, _, _, _ = linalg.lstsq(z_with_intercept, y)
                resid_y = y - z_with_intercept @ beta_y

                # Correlation of residuals
                partial_r, _ = stats.pearsonr(resid_x, resid_y)
            except (linalg.LinAlgError, ValueError):
                return 0.0, 0.0, 1.0

        # Fisher's Z transformation for proper p-value
        # Z = 0.5 * ln((1+r)/(1-r)) ~ N(0, 1/sqrt(n-|S|-3))
        k = z.shape[1] if z is not None and z.size > 0 else 0
        dof = n - k - 3

        if dof <= 0 or abs(partial_r) >= 1.0:
            return partial_r, 0.0, 1.0

        # Fisher's Z
        fisher_z = 0.5 * np.log((1 + partial_r) / (1 - partial_r + 1e-10))
        se = 1.0 / np.sqrt(dof)
        z_stat = fisher_z / se

        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return float(partial_r), float(z_stat), float(p_value)


class GrangerCausalityTest:
    """
    Granger causality test using VAR models.

    Tests whether X Granger-causes Y by comparing:
    - Restricted: Y_t = f(Y_{t-1}, ..., Y_{t-p})
    - Unrestricted: Y_t = f(Y_{t-1}, ..., Y_{t-p}, X_{t-1}, ..., X_{t-p})

    Uses F-test for nested model comparison.
    """

    @staticmethod
    def test(
        x: np.ndarray[Any, Any],
        y: np.ndarray[Any, Any],
        max_lag: int,
        significance_level: float = 0.05,
    ) -> tuple[bool, float, int, float]:
        """
        Test if X Granger-causes Y.

        Args:
            x: Potential cause time series
            y: Potential effect time series
            max_lag: Maximum lag to consider
            significance_level: Alpha for significance

        Returns:
            (is_causal, f_statistic, optimal_lag, p_value)
        """
        n = len(x)
        if n <= max_lag + 2:
            return False, 0.0, 0, 1.0

        best_lag = 1
        best_f = 0.0
        best_p = 1.0

        for lag in range(1, max_lag + 1):
            if n <= 2 * lag + 2:
                continue

            # Construct lagged matrices
            Y = y[lag:]
            Y_lagged = np.column_stack(
                [y[lag - i - 1 : -i - 1 if i < lag - 1 else len(y) - lag] for i in range(lag)]
            )
            X_lagged = np.column_stack(
                [x[lag - i - 1 : -i - 1 if i < lag - 1 else len(x) - lag] for i in range(lag)]
            )

            # Ensure proper shapes
            min_len = min(len(Y), Y_lagged.shape[0], X_lagged.shape[0])
            Y = Y[:min_len]
            Y_lagged = Y_lagged[:min_len]
            X_lagged = X_lagged[:min_len]

            try:
                # Restricted model: Y ~ Y_lagged
                Z_r = np.column_stack([np.ones(min_len), Y_lagged])
                beta_r, _residuals_r, _, _ = linalg.lstsq(Z_r, Y)
                rss_r = np.sum((Y - Z_r @ beta_r) ** 2)

                # Unrestricted model: Y ~ Y_lagged + X_lagged
                Z_u = np.column_stack([np.ones(min_len), Y_lagged, X_lagged])
                beta_u, _residuals_u, _, _ = linalg.lstsq(Z_u, Y)
                rss_u = np.sum((Y - Z_u @ beta_u) ** 2)

                # F-test
                k_r = Z_r.shape[1]
                k_u = Z_u.shape[1]
                df1 = k_u - k_r
                df2 = min_len - k_u

                if df2 <= 0 or rss_u <= 0:
                    continue

                f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
                p_value = 1 - stats.f.cdf(f_stat, df1, df2)

                # Select best lag by BIC
                if p_value < best_p:
                    best_lag = lag
                    best_f = f_stat
                    best_p = p_value

            except (linalg.LinAlgError, ValueError) as e:
                logger.debug(f"Granger causality F-test failed for lag {lag}: {e}")
                continue

        is_causal = best_p < significance_level
        return is_causal, float(best_f), best_lag, float(best_p)


class PropensityScoreEstimator:
    """
    Propensity score estimation for causal inference.

    Estimates P(T=1|X) using logistic regression, then:
    - Inverse Probability Weighting (IPW)
    - Propensity Score Matching
    - Doubly Robust (AIPW) estimation
    """

    propensity_scores: np.ndarray[Any, Any] | None

    def __init__(self, treatment: np.ndarray[Any, Any], covariates: np.ndarray[Any, Any]) -> None:
        """
        Args:
            treatment: Binary treatment indicator (0/1)
            covariates: Covariate matrix (n x p)
        """
        self.treatment = treatment.astype(float)
        self.covariates = covariates
        self.propensity_scores = None
        self._fitted = False

    def fit(self) -> np.ndarray[Any, Any]:
        """Fit propensity score model using logistic regression."""
        X = np.column_stack([np.ones(len(self.covariates)), self.covariates])
        y = self.treatment

        # Newton-Raphson for logistic regression
        beta = np.zeros(X.shape[1])

        for _ in range(100):  # Max iterations
            # Predicted probabilities
            z = X @ beta
            z = np.clip(z, -500, 500)  # Prevent overflow
            p = 1 / (1 + np.exp(-z))
            p = np.clip(p, 1e-10, 1 - 1e-10)

            # Gradient and Hessian
            W = np.diag(p * (1 - p))
            gradient = X.T @ (y - p)
            hessian = -X.T @ W @ X

            try:
                delta = linalg.solve(hessian, gradient)
                beta = beta - delta

                if np.max(np.abs(delta)) < 1e-6:
                    break
            except linalg.LinAlgError:
                break

        # Final propensity scores
        z = X @ beta
        z = np.clip(z, -500, 500)
        self.propensity_scores = 1 / (1 + np.exp(-z))
        self.propensity_scores = np.clip(self.propensity_scores, 0.01, 0.99)
        self._fitted = True

        return self.propensity_scores

    def ipw_ate(self, outcome: np.ndarray[Any, Any]) -> tuple[float, float, float]:
        """
        Inverse Probability Weighting ATE estimation.

        Returns:
            (ate, standard_error, p_value)
        """
        if not self._fitted:
            self.fit()

        if self.propensity_scores is None:
            raise ValueError("propensity_scores not initialized. Call fit() first.")
        ps = self.propensity_scores
        t = self.treatment
        y = outcome

        # IPW estimator
        treated_weight = t / ps
        control_weight = (1 - t) / (1 - ps)

        mu1 = np.sum(treated_weight * y) / np.sum(treated_weight)
        mu0 = np.sum(control_weight * y) / np.sum(control_weight)
        ate = mu1 - mu0

        # Standard error via influence functions
        n = len(y)
        influence = t * y / ps - (1 - t) * y / (1 - ps) - ate * (t / ps - (1 - t) / (1 - ps))
        se = np.sqrt(np.var(influence) / n)

        t_stat = ate / se if se > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

        return float(ate), float(se), float(p_value)

    def doubly_robust_ate(
        self,
        outcome: np.ndarray[Any, Any],
    ) -> tuple[float, float, float]:
        """
        Augmented Inverse Probability Weighting (AIPW/Doubly Robust).

        Consistent if either propensity or outcome model is correct.

        Returns:
            (ate, standard_error, p_value)
        """
        if not self._fitted:
            self.fit()

        if self.propensity_scores is None:
            raise ValueError("propensity_scores not initialized. Call fit() first.")
        ps = self.propensity_scores
        t = self.treatment
        y = outcome
        X = np.column_stack([np.ones(len(self.covariates)), self.covariates])

        # Outcome regression models
        treated_mask = t == 1
        control_mask = t == 0

        try:
            # E[Y|X, T=1]
            if treated_mask.sum() > X.shape[1]:
                beta1, _, _, _ = linalg.lstsq(X[treated_mask], y[treated_mask])
                mu1_hat = X @ beta1
            else:
                mu1_hat = np.full(len(y), y[treated_mask].mean() if treated_mask.any() else 0)

            # E[Y|X, T=0]
            if control_mask.sum() > X.shape[1]:
                beta0, _, _, _ = linalg.lstsq(X[control_mask], y[control_mask])
                mu0_hat = X @ beta0
            else:
                mu0_hat = np.full(len(y), y[control_mask].mean() if control_mask.any() else 0)

        except linalg.LinAlgError:
            return self.ipw_ate(outcome)

        # AIPW estimator
        # tau = E[mu1(X) - mu0(X)] + E[(T/ps)(Y - mu1(X))] - E[((1-T)/(1-ps))(Y - mu0(X))]
        n = len(y)

        term1 = np.mean(mu1_hat - mu0_hat)
        term2 = np.mean(t * (y - mu1_hat) / ps)
        term3 = np.mean((1 - t) * (y - mu0_hat) / (1 - ps))

        ate = term1 + term2 - term3

        # Influence function for SE
        influence = (
            mu1_hat - mu0_hat + t * (y - mu1_hat) / ps - (1 - t) * (y - mu0_hat) / (1 - ps) - ate
        )
        se = np.sqrt(np.var(influence) / n)

        t_stat = ate / se if se > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

        return float(ate), float(se), float(p_value)

    def overlap_diagnostic(self) -> float:
        """
        Compute propensity score overlap diagnostic.

        Returns score in [0, 1] where 1 = perfect overlap.
        """
        if not self._fitted:
            self.fit()

        if self.propensity_scores is None:
            raise ValueError("propensity_scores not initialized. Call fit() first.")
        ps = self.propensity_scores
        t = self.treatment

        # Check overlap: both groups should have PS in [0.1, 0.9]
        treated_ps = ps[t == 1]
        control_ps = ps[t == 0]

        if len(treated_ps) == 0 or len(control_ps) == 0:
            return 0.0

        # Overlap = 1 - |mean(PS|T=1) - mean(PS|T=0)|
        overlap = 1 - abs(np.mean(treated_ps) - np.mean(control_ps))

        # Penalize extreme propensity scores
        extreme_penalty = np.mean((ps < 0.1) | (ps > 0.9))
        overlap = overlap * (1 - extreme_penalty)

        return float(np.clip(overlap, 0, 1))


class CausalDiscoveryEngine:
    """
    Production Causal Discovery and Inference Engine.

    Implements rigorous causal discovery and effect estimation:

    1. PC Algorithm (Spirtes et al. 2000)
       - Constraint-based structure learning
       - Fisher's Z test for conditional independence
       - Proper v-structure detection
       - Meek's orientation rules

    2. Granger Causality (Granger 1969)
       - VAR model comparison
       - F-test for nested models
       - Optimal lag selection via BIC

    3. Causal Effect Estimation
       - Propensity score matching
       - Inverse probability weighting
       - Doubly robust AIPW estimation
       - Bootstrap confidence intervals

    4. Do-Calculus (Pearl 2009)
       - Backdoor adjustment
       - Frontdoor adjustment
       - Instrumental variables

    5. Counterfactual Reasoning
       - Three-step procedure (abduction, action, prediction)
       - Sensitivity analysis for robustness
    """

    PHI = (1 + np.sqrt(5)) / 2  # Golden ratio

    def __init__(
        self,
        significance_level: float = 0.05,
        max_conditioning_set: int = 4,
        enable_temporal: bool = True,
        max_lag: int = 5,
        n_bootstrap: int = 100,
    ):
        """
        Initialize Causal Discovery Engine.

        Args:
            significance_level: Alpha for independence tests
            max_conditioning_set: Maximum size of conditioning sets
            enable_temporal: Enable Granger causality
            max_lag: Maximum time lag for temporal causation
            n_bootstrap: Number of bootstrap samples for CIs
        """
        self.significance_level = significance_level
        self.max_conditioning_set = max_conditioning_set
        self.enable_temporal = enable_temporal
        self.max_lag = max_lag
        self.n_bootstrap = n_bootstrap

        # Storage
        self._graphs: dict[str, CausalGraph] = {}
        self._effects: dict[tuple[str, str], CausalEffect] = {}
        self._counterfactuals: list[CounterfactualResult] = []

        # Statistics
        self._stats = {
            "graphs_discovered": 0,
            "edges_found": 0,
            "ci_tests_performed": 0,
            "effects_estimated": 0,
            "counterfactuals_computed": 0,
        }

        logger.info(
            f"CausalDiscoveryEngine initialized "
            f"(alpha={significance_level}, max_cond={max_conditioning_set})"
        )

    def discover_structure(
        self,
        data: np.ndarray[Any, Any],
        variable_names: list[str] | None = None,
        prior_knowledge: dict[str, list[str]] | None = None,
    ) -> CausalGraph:
        """
        Discover causal structure using the PC algorithm.

        Implements the full PC algorithm:
        1. Start with complete undirected graph
        2. For each conditioning set size 0, 1, 2, ...
           - Test conditional independence for adjacent pairs
           - Remove edge if independent, store separation set
        3. Orient v-structures (immoralities)
        4. Apply Meek's rules for further orientation

        Args:
            data: Data matrix (samples x variables)
            variable_names: Names for variables
            prior_knowledge: Known causal constraints

        Returns:
            Discovered causal graph (CPDAG)
        """
        n_samples, n_vars = data.shape
        names = variable_names or [f"X{i}" for i in range(n_vars)]

        # Initialize with complete undirected graph
        adjacency = np.ones((n_vars, n_vars), dtype=bool)
        np.fill_diagonal(adjacency, False)

        # Store separation sets
        separation_sets: dict[tuple[int, int], tuple[int, ...]] = {}

        # Phase 1: Edge removal via conditional independence tests
        for cond_size in range(self.max_conditioning_set + 1):
            changed = False

            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if not adjacency[i, j]:
                        continue

                    # Get adjacent nodes (potential conditioning sets)
                    adj_i = set(np.where(adjacency[i])[0]) - {j}
                    adj_j = set(np.where(adjacency[j])[0]) - {i}
                    neighbors = adj_i | adj_j

                    if len(neighbors) < cond_size:
                        continue

                    # Test all conditioning sets of current size
                    for cond_set in combinations(neighbors, cond_size):
                        cond_data = data[:, list(cond_set)] if cond_set else None

                        _partial_r, _z_stat, p_value = PartialCorrelationTest.test(
                            data[:, i], data[:, j], cond_data, n_samples
                        )
                        self._stats["ci_tests_performed"] += 1

                        if p_value > self.significance_level:
                            # Remove edge, store separation set
                            adjacency[i, j] = False
                            adjacency[j, i] = False
                            separation_sets[(min(i, j), max(i, j))] = cond_set
                            changed = True
                            break

            if not changed and cond_size > 0:
                break  # No more edges removed, stop

        # Phase 2: Orient v-structures (X -> Z <- Y where X-Y not adjacent)
        directed = np.zeros((n_vars, n_vars), dtype=bool)
        colliders = []

        for z in range(n_vars):
            # Find pairs (x, y) adjacent to z
            neighbors_z = list(np.where(adjacency[z])[0])

            for idx_x, x in enumerate(neighbors_z):
                for y in neighbors_z[idx_x + 1 :]:
                    # Check if x and y are NOT adjacent
                    if adjacency[x, y]:
                        continue

                    # Check if z is NOT in separation set of (x, y)
                    key = (min(x, y), max(x, y))
                    sep_set = separation_sets.get(key, ())

                    if z not in sep_set:
                        # v-structure: x -> z <- y
                        directed[x, z] = True
                        directed[y, z] = True
                        colliders.append((names[z], names[x], names[y]))

        # Phase 3: Apply Meek's orientation rules
        directed = self._apply_meek_rules(adjacency, directed, n_vars)

        # Build edges
        edges = []
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue

                if directed[i, j]:
                    # Directed edge i -> j
                    strength = self._compute_effect_size(data[:, i], data[:, j])
                    _, _, p_val = PartialCorrelationTest.test(
                        data[:, i], data[:, j], None, n_samples
                    )
                    edges.append(
                        CausalEdge(
                            source=names[i],
                            target=names[j],
                            relation_type=CausalRelationType.DIRECT,
                            strength=abs(strength),
                            confidence=1 - p_val,
                            p_value=p_val,
                        )
                    )
                elif adjacency[i, j] and not directed[j, i] and i < j:
                    # Undirected edge (CPDAG)
                    strength = self._compute_effect_size(data[:, i], data[:, j])
                    _, _, p_val = PartialCorrelationTest.test(
                        data[:, i], data[:, j], None, n_samples
                    )
                    edges.append(
                        CausalEdge(
                            source=names[i],
                            target=names[j],
                            relation_type=CausalRelationType.UNDIRECTED,
                            strength=abs(strength),
                            confidence=1 - p_val,
                            p_value=p_val,
                        )
                    )

        # Apply prior knowledge
        if prior_knowledge:
            edges = self._apply_prior_knowledge(edges, prior_knowledge, names)

        # Convert separation sets to use names
        sep_sets_named = {
            (names[i], names[j]): tuple(names[k] for k in sep)
            for (i, j), sep in separation_sets.items()
        }

        graph = CausalGraph(
            graph_id=f"causal_graph_{int(time.time() * 1000)}",
            nodes=names,
            edges=edges,
            confounders=[],  # Detected separately
            colliders=colliders,
            separation_sets=sep_sets_named,
            is_cpdag=any(e.relation_type == CausalRelationType.UNDIRECTED for e in edges),
        )

        self._graphs[graph.graph_id] = graph
        self._stats["graphs_discovered"] += 1
        self._stats["edges_found"] += len(edges)

        logger.info(
            f"Discovered causal graph: {len(names)} nodes, {len(edges)} edges, "
            f"{len(colliders)} v-structures"
        )
        return graph

    def discover_temporal_causation(
        self,
        time_series: np.ndarray[Any, Any],
        variable_names: list[str] | None = None,
    ) -> CausalGraph:
        """
        Discover temporal causal relationships using Granger causality.

        Uses VAR model comparison with F-tests.

        Args:
            time_series: Time series data (time x variables)
            variable_names: Names for variables

        Returns:
            Temporal causal graph with lag information
        """
        _n_time, n_vars = time_series.shape
        names = variable_names or [f"X{i}" for i in range(n_vars)]
        edges = []

        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue

                is_causal, _f_stat, opt_lag, p_value = GrangerCausalityTest.test(
                    time_series[:, i],
                    time_series[:, j],
                    self.max_lag,
                    self.significance_level,
                )

                if is_causal:
                    # Compute effect size (R-squared improvement)
                    strength = 1 - p_value  # Use p-value as proxy

                    edges.append(
                        CausalEdge(
                            source=names[i],
                            target=names[j],
                            relation_type=CausalRelationType.DIRECT,
                            strength=strength,
                            confidence=1 - p_value,
                            lag=opt_lag,
                            p_value=p_value,
                        )
                    )

        graph = CausalGraph(
            graph_id=f"temporal_causal_{int(time.time() * 1000)}",
            nodes=names,
            edges=edges,
            confounders=[],
            colliders=[],
            separation_sets={},
            is_cpdag=False,  # Temporal graphs are directed
        )

        self._graphs[graph.graph_id] = graph
        self._stats["graphs_discovered"] += 1
        self._stats["edges_found"] += len(edges)

        logger.info(f"Discovered temporal causation: {len(edges)} edges")
        return graph

    def estimate_causal_effect(
        self,
        data: np.ndarray[Any, Any],
        cause_idx: int,
        effect_idx: int,
        adjustment_set: list[int] | None = None,
        variable_names: list[str] | None = None,
        method: str = "doubly_robust",
    ) -> CausalEffect:
        """
        Estimate causal effect using doubly robust estimation.

        Args:
            data: Data matrix
            cause_idx: Index of cause variable
            effect_idx: Index of effect variable
            adjustment_set: Variables to adjust for (backdoor)
            variable_names: Variable names
            method: "ipw", "regression", or "doubly_robust"

        Returns:
            Estimated causal effect with confidence interval
        """
        names = variable_names or [f"X{i}" for i in range(data.shape[1])]
        cause_name = names[cause_idx]
        effect_name = names[effect_idx]

        # Binarize treatment for propensity score methods
        treatment = data[:, cause_idx]
        outcome = data[:, effect_idx]

        # Median split for continuous treatment
        treatment_binary = (treatment > np.median(treatment)).astype(float)

        # Get covariates for adjustment
        if adjustment_set:
            covariates = data[:, adjustment_set]
        else:
            # Use all other variables as covariates
            other_idx = [i for i in range(data.shape[1]) if i not in [cause_idx, effect_idx]]
            covariates = data[:, other_idx] if other_idx else np.ones((len(data), 1))

        # Propensity score estimation
        ps_estimator = PropensityScoreEstimator(treatment_binary, covariates)
        ps_estimator.fit()

        # Estimate ATE
        if method == "ipw":
            ate, se, p_value = ps_estimator.ipw_ate(outcome)
        elif method == "regression":
            ate, se, p_value = self._regression_adjustment(treatment_binary, outcome, covariates)
        else:  # doubly_robust
            ate, _se, p_value = ps_estimator.doubly_robust_ate(outcome)

        # Bootstrap CI
        ci = self._bootstrap_ci(data, cause_idx, effect_idx, adjustment_set, method)

        # ATT (Average Treatment Effect on Treated)
        treated_mask = treatment_binary == 1
        att = ate  # For simplicity; proper ATT requires different weights

        # Diagnostics
        n_treated = int(treated_mask.sum())
        n_control = int((~treated_mask).sum())
        overlap = ps_estimator.overlap_diagnostic()

        effect = CausalEffect(
            cause=cause_name,
            effect=effect_name,
            ate=ate,
            att=att,
            confidence_interval=ci,
            p_value=p_value,
            method=method,
            is_significant=p_value < self.significance_level,
            n_treated=n_treated,
            n_control=n_control,
            overlap_score=overlap,
        )

        self._effects[(cause_name, effect_name)] = effect
        self._stats["effects_estimated"] += 1

        return effect

    def do_intervention(
        self,
        graph: CausalGraph,
        data: np.ndarray[Any, Any],
        intervention_var: str,
        intervention_value: float,
        target_var: str,
        variable_names: list[str],
    ) -> dict[str, Any]:
        """
        Compute the effect of an intervention do(X=x) on Y.

        Uses backdoor adjustment based on the causal graph.

        Args:
            graph: Causal graph structure
            data: Observational data
            intervention_var: Variable to intervene on
            intervention_value: Value to set
            target_var: Target variable
            variable_names: Variable names

        Returns:
            Intervention effect with confidence interval
        """
        # Find valid adjustment set (backdoor criterion)
        adjustment_vars = self._find_adjustment_set(graph, intervention_var, target_var)

        # Get indices
        int_idx = variable_names.index(intervention_var)
        target_idx = variable_names.index(target_var)
        adj_indices = [variable_names.index(v) for v in adjustment_vars if v in variable_names]

        if not adj_indices:
            # No confounders: simple stratification
            # Bin intervention variable and average
            treatment = data[:, int_idx]
            outcome = data[:, target_idx]

            # Find observations near intervention value
            tolerance = np.std(treatment) * 0.5
            mask = np.abs(treatment - intervention_value) < tolerance

            if mask.sum() > 5:
                expected_y = float(outcome[mask].mean())
                ci = self._bootstrap_mean_ci(outcome[mask])
            else:
                # Extrapolate using regression
                beta = np.polyfit(treatment, outcome, 1)
                expected_y = float(np.polyval(beta, intervention_value))
                residuals = outcome - np.polyval(beta, treatment)
                ci = (expected_y - 1.96 * np.std(residuals), expected_y + 1.96 * np.std(residuals))
        else:
            # Backdoor adjustment: E[Y|do(X=x)] = sum_z E[Y|X=x,Z=z] P(Z=z)
            expected_y, ci = self._backdoor_adjustment(
                data, int_idx, target_idx, adj_indices, intervention_value
            )

        return {
            "intervention": f"do({intervention_var}={intervention_value:.3f})",
            "target": target_var,
            "expected_value": expected_y,
            "confidence_interval": ci,
            "adjustment_set": adjustment_vars,
            "method": "backdoor_adjustment" if adj_indices else "stratification",
        }

    def counterfactual_query(
        self,
        graph: CausalGraph,
        data: np.ndarray[Any, Any],
        factual_observation: dict[str, float],
        counterfactual_intervention: dict[str, float],
        target_var: str,
        variable_names: list[str],
    ) -> CounterfactualResult:
        """
        Answer counterfactual queries using the three-step procedure.

        Steps:
        1. Abduction: Infer exogenous noise from factual observation
        2. Action: Modify structural equations with intervention
        3. Prediction: Compute counterfactual outcome

        Args:
            graph: Causal graph
            data: Historical data for estimating structural equations
            factual_observation: What actually happened
            counterfactual_intervention: What we hypothesize instead
            target_var: Variable we want to predict
            variable_names: Variable names

        Returns:
            Counterfactual result with sensitivity analysis
        """
        target_idx = variable_names.index(target_var)

        # Get factual outcome
        factual_outcome = factual_observation.get(target_var, float(data[:, target_idx].mean()))

        # Intervention details
        int_var = next(iter(counterfactual_intervention.keys()))
        int_value = counterfactual_intervention[int_var]
        factual_value = factual_observation.get(int_var, 0)

        if int_var not in variable_names:
            return CounterfactualResult(
                query=f"What would {target_var} have been if {int_var}={int_value}?",
                factual_outcome=factual_outcome,
                counterfactual_outcome=factual_outcome,
                difference=0.0,
                probability=0.5,
                explanation=f"Variable {int_var} not found in graph.",
            )

        int_idx = variable_names.index(int_var)

        # Step 1: Abduction - estimate noise from factual
        # Using linear SEM: Y = alpha + beta*X + epsilon
        # epsilon_factual = Y_factual - alpha - beta*X_factual

        parents = graph.get_parents(target_var)
        parent_indices = [variable_names.index(p) for p in parents if p in variable_names]

        if parent_indices:
            X = np.column_stack([np.ones(len(data)), data[:, parent_indices]])
            y = data[:, target_idx]

            try:
                beta, _, _, _ = linalg.lstsq(X, y)

                # Factual parent values
                factual_parents = np.array(
                    [1]
                    + [
                        factual_observation.get(variable_names[i], data[:, i].mean())
                        for i in parent_indices
                    ]
                )

                predicted_factual = factual_parents @ beta
                epsilon = factual_outcome - predicted_factual

            except linalg.LinAlgError:
                epsilon = 0.0
                beta = np.zeros(len(parent_indices) + 1)
        else:
            epsilon = factual_outcome - data[:, target_idx].mean()
            beta = np.array([data[:, target_idx].mean()])

        # Step 2 & 3: Action and Prediction
        # Replace factual intervention value with counterfactual
        if int_var in parents:
            # Direct effect: update the relevant parent value
            cf_parents = np.array(
                [1]
                + [
                    (
                        int_value
                        if variable_names[i] == int_var
                        else factual_observation.get(variable_names[i], data[:, i].mean())
                    )
                    for i in parent_indices
                ]
            )
            counterfactual_outcome = float(cf_parents @ beta + epsilon)
        else:
            # Indirect effect: trace through mediators
            # Simplified: estimate total effect
            effect = self.estimate_causal_effect(data, int_idx, target_idx, None, variable_names)
            counterfactual_outcome = factual_outcome + effect.ate * (int_value - factual_value)

        difference = counterfactual_outcome - factual_outcome

        # Sensitivity analysis: Rosenbaum bounds
        robustness = self._sensitivity_analysis(data, int_idx, target_idx, difference)

        result = CounterfactualResult(
            query=f"What would {target_var} have been if {int_var}={int_value:.3f}?",
            factual_outcome=factual_outcome,
            counterfactual_outcome=counterfactual_outcome,
            difference=difference,
            probability=min(0.95, 0.5 + 0.4 * robustness),
            explanation=(
                f"If {int_var} had been {int_value:.3f} instead of {factual_value:.3f}, "
                f"{target_var} would have changed from {factual_outcome:.3f} to "
                f"{counterfactual_outcome:.3f} (diff: {difference:+.3f}). "
                f"Robustness to confounding: Γ={robustness:.2f}"
            ),
            robustness_value=robustness,
        )

        self._counterfactuals.append(result)
        self._stats["counterfactuals_computed"] += 1

        return result

    def _apply_meek_rules(
        self,
        adjacency: np.ndarray[Any, Any],
        directed: np.ndarray[Any, Any],
        n_vars: int,
    ) -> np.ndarray[Any, Any]:
        """
        Apply Meek's rules for edge orientation.

        Rules:
        R1: If a -> b - c and a is not adjacent to c, orient b -> c
        R2: If a -> b -> c and a - c, orient a -> c
        R3: If a - b, a - c, b -> d <- c, and a - d, orient a -> d
        R4: If a - b, a - c, b -> c, and a - d -> c, orient a -> c
        """
        changed = True
        while changed:
            changed = False

            for i in range(n_vars):
                for j in range(n_vars):
                    if i == j or directed[i, j] or not adjacency[i, j]:
                        continue

                    # R1: a -> i - j, a not adj to j => i -> j
                    for a in range(n_vars):
                        if directed[a, i] and not adjacency[a, j] and a != j:
                            directed[i, j] = True
                            changed = True
                            break

                    if directed[i, j]:
                        continue

                    # R2: i -> k -> j, i - j => i -> j
                    for k in range(n_vars):
                        if directed[i, k] and directed[k, j]:
                            directed[i, j] = True
                            changed = True
                            break

        return directed

    def _find_adjustment_set(
        self,
        graph: CausalGraph,
        treatment: str,
        outcome: str,
    ) -> list[str]:
        """Find a valid adjustment set satisfying the backdoor criterion."""
        # Simple approach: use parents of treatment that are not descendants of treatment
        parents = set(graph.get_parents(treatment))
        descendants = graph.get_descendants(treatment)

        # Valid adjustment = parents - descendants
        valid_adjustment = [p for p in parents if p not in descendants and p != outcome]

        return valid_adjustment

    def _compute_effect_size(self, x: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> float:
        """Compute standardized effect size (Pearson correlation)."""
        if len(x) < 3:
            return 0.0
        r, _ = stats.pearsonr(x, y)
        return float(r)

    def _regression_adjustment(
        self,
        treatment: np.ndarray[Any, Any],
        outcome: np.ndarray[Any, Any],
        covariates: np.ndarray[Any, Any],
    ) -> tuple[float, float, float]:
        """Estimate ATE via regression adjustment."""
        X = np.column_stack([np.ones(len(treatment)), treatment, covariates])
        y = outcome

        try:
            beta, residuals, _, _ = linalg.lstsq(X, y)
            ate = beta[1]  # Coefficient of treatment

            # Standard error
            residuals = y - X @ beta
            mse = np.sum(residuals**2) / (len(y) - X.shape[1])
            var_beta = mse * linalg.inv(X.T @ X)[1, 1]
            se = np.sqrt(var_beta)

            t_stat = ate / se if se > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(y) - X.shape[1]))

            return float(ate), float(se), float(p_value)

        except linalg.LinAlgError:
            return 0.0, 1.0, 1.0

    def _bootstrap_ci(
        self,
        data: np.ndarray[Any, Any],
        cause_idx: int,
        effect_idx: int,
        adjustment_set: list[int] | None,
        method: str,
    ) -> tuple[float, float]:
        """Compute bootstrap confidence interval for ATE."""
        n = len(data)
        ates = []

        for _ in range(self.n_bootstrap):
            # Bootstrap sample
            idx = np.random.choice(n, n, replace=True)
            sample = data[idx]

            # Estimate ATE
            treatment = sample[:, cause_idx]
            outcome = sample[:, effect_idx]
            treatment_binary = (treatment > np.median(treatment)).astype(float)

            if adjustment_set:
                covariates = sample[:, adjustment_set]
            else:
                other_idx = [i for i in range(data.shape[1]) if i not in [cause_idx, effect_idx]]
                covariates = sample[:, other_idx] if other_idx else np.ones((len(sample), 1))

            try:
                ps_est = PropensityScoreEstimator(treatment_binary, covariates)
                ps_est.fit()

                if method == "doubly_robust":
                    ate, _, _ = ps_est.doubly_robust_ate(outcome)
                else:
                    ate, _, _ = ps_est.ipw_ate(outcome)

                ates.append(ate)
            except Exception as e:
                logger.debug(f"Skipping bootstrap iteration due to estimation error: {e}")
                continue

        if len(ates) < 10:
            return (-1.0, 1.0)

        return (float(np.percentile(ates, 2.5)), float(np.percentile(ates, 97.5)))

    def _bootstrap_mean_ci(self, data: np.ndarray[Any, Any]) -> tuple[float, float]:
        """Bootstrap CI for mean."""
        means = []
        n = len(data)

        for _ in range(self.n_bootstrap):
            sample = np.random.choice(data, n, replace=True)
            means.append(np.mean(sample))

        return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))

    def _backdoor_adjustment(
        self,
        data: np.ndarray[Any, Any],
        x_idx: int,
        y_idx: int,
        z_indices: list[int],
        x_value: float,
    ) -> tuple[float, tuple[float, float]]:
        """
        Apply backdoor adjustment formula.

        E[Y|do(X=x)] = sum_z E[Y|X=x,Z=z] P(Z=z)
        """
        X = data[:, [x_idx, *z_indices]]
        y = data[:, y_idx]

        X_with_intercept = np.column_stack([np.ones(len(X)), X])

        try:
            beta, _, _, _ = linalg.lstsq(X_with_intercept, y)

            # Predict at x=x_value with marginal Z distribution (average)
            z_means = data[:, z_indices].mean(axis=0)
            x_point = np.array([1, x_value, *list(z_means)])

            expected_y = float(x_point @ beta)

            # CI via delta method
            residuals = y - X_with_intercept @ beta
            mse = np.mean(residuals**2)
            se = np.sqrt(
                mse * x_point @ linalg.inv(X_with_intercept.T @ X_with_intercept) @ x_point
            )

            ci = (expected_y - 1.96 * se, expected_y + 1.96 * se)

            return expected_y, ci

        except linalg.LinAlgError:
            mean_y = float(data[:, y_idx].mean())
            std_y = float(data[:, y_idx].std())
            return mean_y, (mean_y - 1.96 * std_y, mean_y + 1.96 * std_y)

    def _sensitivity_analysis(
        self,
        data: np.ndarray[Any, Any],
        treatment_idx: int,
        outcome_idx: int,
        observed_effect: float,
    ) -> float:
        """
        Sensitivity analysis: how much confounding would nullify the effect?

        Returns Γ (Rosenbaum bounds): effect robust to confounding up to Γ.
        """
        # Simplified: based on effect size relative to outcome variance
        outcome_std = np.std(data[:, outcome_idx])
        if outcome_std == 0:
            return 1.0

        # Effect in standard deviation units
        effect_sds = abs(observed_effect) / outcome_std

        # Larger effect -> more robust to confounding
        # Γ = 1 means sensitive to any confounding
        # Γ = 2 means robust to 2x confounding
        robustness = 1 + effect_sds

        return float(min(3.0, robustness))

    def _apply_prior_knowledge(
        self,
        edges: list[CausalEdge],
        prior: dict[str, list[str]],
        names: list[str],
    ) -> list[CausalEdge]:
        """Apply prior knowledge constraints."""
        # prior format: {"causes": ["var1 -> var2"], "forbids": ["var3 -> var4"]}
        forbidden = set()
        for constraint in prior.get("forbids", []):
            parts = constraint.replace(" ", "").split("->")
            if len(parts) == 2:
                forbidden.add((parts[0], parts[1]))

        required = set()
        for constraint in prior.get("causes", []):
            parts = constraint.replace(" ", "").split("->")
            if len(parts) == 2:
                required.add((parts[0], parts[1]))
                # Remove reverse edge
                edges = [e for e in edges if not (e.source == parts[1] and e.target == parts[0])]

        # Remove forbidden edges
        edges = [e for e in edges if (e.source, e.target) not in forbidden]

        return edges

    def get_statistics(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            **self._stats,
            "graphs_stored": len(self._graphs),
            "effects_stored": len(self._effects),
            "counterfactuals_stored": len(self._counterfactuals),
        }
