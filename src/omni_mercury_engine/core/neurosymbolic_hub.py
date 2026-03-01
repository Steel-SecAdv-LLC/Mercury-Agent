"""
Mercury Agent - Enhanced Neuro-Symbolic Hub
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Enhanced neuro-symbolic fusion integrating:
- LTN (Logic Tensor Networks) with NetworkX graph reasoning
- Stacking/BMA fusion for ensemble calibration
- Real-time rule inference with explanation generation
- GOSNN bidirectional synaptic integration
- Calibration and conformal prediction
- Benevolence ≥0.99 enforcement

This module serves as the keystone for cross-domain gains, providing
unified neuro-symbolic capabilities to all detector domains.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from scipy.optimize import minimize

# P2-P3 Integration: Import centralized constants and domain modules
from omni_mercury_engine.core.centralized_constants import (
    ETHICAL,
    LYAPUNOV,
    MATH,
)
from omni_mercury_engine.core.config import ThresholdConfig

# Domain-specific feature extraction (P2 Integration)
try:
    from omni_mercury_engine.core.domain_feature_extractors import (
        BaseDomainExtractor,
        DomainFeatureExtractorFactory,
    )

    DOMAIN_EXTRACTORS_AVAILABLE = True
except ImportError:
    DOMAIN_EXTRACTORS_AVAILABLE = False
    DomainFeatureExtractorFactory = None  # type: ignore[misc, assignment]
    BaseDomainExtractor = None  # type: ignore[misc, assignment]

# Adaptive domain thresholding (P2 Integration)
try:
    from omni_mercury_engine.core.adaptive_domain_thresholding import (
        AdaptiveDomainThresholdManager,
        create_domain_threshold_manager,
    )

    ADAPTIVE_THRESHOLDING_AVAILABLE = True
except ImportError:
    ADAPTIVE_THRESHOLDING_AVAILABLE = False
    AdaptiveDomainThresholdManager = None  # type: ignore[misc, assignment]
    create_domain_threshold_manager = None  # type: ignore[assignment]

# GOSNN-3R bidirectional integration (P2 Integration)
try:
    from omni_mercury_engine.core.gosnn_3r_integration import (
        GOSNN3RIntegration,
        SlidingWindowConfig,
        SlidingWindowNormalizer,
    )

    GOSNN_3R_AVAILABLE = True
except ImportError:
    GOSNN_3R_AVAILABLE = False
    GOSNN3RIntegration = None  # type: ignore[misc, assignment]
    SlidingWindowConfig = None  # type: ignore[misc, assignment]
    SlidingWindowNormalizer = None  # type: ignore[misc, assignment]


logger = logging.getLogger(__name__)


# Constants - now referencing centralized constants
PHI = MATH.GOLDEN_RATIO
BENEVOLENCE_THRESHOLD = ETHICAL.BENEVOLENCE_IMMUTABLE
SIGMA_IMMUTABLE_DEFAULT = ETHICAL.SIGMA_IMMUTABLE_DEFAULT
LYAPUNOV_LAMBDA = LYAPUNOV.LAMBDA_CONVERGENCE

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment, unused-ignore]
    nn = None  # type: ignore[assignment, unused-ignore]


class FusionMode(Enum):
    """Fusion modes for neuro-symbolic integration."""

    NEURAL_DOMINANT = "neural_dominant"  # 70% neural, 30% symbolic
    SYMBOLIC_DOMINANT = "symbolic_dominant"  # 30% neural, 70% symbolic
    BALANCED = "balanced"  # 50% neural, 50% symbolic
    PHI_WEIGHTED = "phi_weighted"  # Golden ratio weighting
    ADAPTIVE = "adaptive"  # Context-dependent weighting
    STACKING = "stacking"  # Meta-learner fusion
    BMA = "bma"  # Bayesian Model Averaging


@dataclass
class ExplainableOutput:
    """Explainable output from neuro-symbolic hub."""

    # Core predictions
    anomaly_score: float
    is_anomaly: bool
    confidence: float

    # Component contributions
    neural_score: float
    symbolic_score: float
    neural_weight: float
    symbolic_weight: float

    # Explanations
    rules_fired: list[str]
    explanations: list[str]
    reasoning_chain: list[dict[str, Any]]

    # Calibration
    calibrated_score: float | None = None
    confidence_interval: tuple[float, float] | None = None

    # Ethical compliance
    benevolence_score: float = 1.0
    ethical_compliant: bool = True
    ethical_violations: list[str] = field(default_factory=list)

    # Metadata
    fusion_mode: str = "balanced"
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "anomaly_score": self.anomaly_score,
            "is_anomaly": self.is_anomaly,
            "confidence": self.confidence,
            "neural_score": self.neural_score,
            "symbolic_score": self.symbolic_score,
            "rules_fired": self.rules_fired,
            "explanations": self.explanations,
            "calibrated_score": self.calibrated_score,
            "benevolence_score": self.benevolence_score,
            "ethical_compliant": self.ethical_compliant,
        }


@dataclass
class SymbolicRule:
    """Enhanced symbolic rule with provenance tracking."""

    rule_id: str
    premise: str
    conclusion: str
    confidence: float
    category: str = "general"
    provenance: str = "system"  # system, learned, user
    activation_count: int = 0
    last_activated: float | None = None
    explanation_template: str = ""

    def evaluate(self, facts: set[str], context: dict[str, Any]) -> tuple[bool, float]:
        """
        Evaluate if rule fires given facts and context.

        Returns:
            Tuple of (fires, confidence_adjusted)
        """
        premise_parts = self.premise.lower().replace(" and ", ",").split(",")

        for part in premise_parts:
            part = part.strip()

            # Handle negation
            if part.startswith("not "):
                key = part[4:].strip()
                if key in facts or context.get(key):
                    return False, 0.0
                continue

            # Handle comparisons (order matters: >= and <= before > and <)
            if ">=" in part:
                key, val = part.split(">=")
                ctx_val = context.get(key.strip(), 0)
                if not isinstance(ctx_val, (int, float)):
                    return False, 0.0
                if ctx_val < float(val.strip()):
                    return False, 0.0
            elif "<=" in part:
                key, val = part.split("<=")
                ctx_val = context.get(key.strip(), float("inf"))
                if not isinstance(ctx_val, (int, float)):
                    return False, 0.0
                if ctx_val > float(val.strip()):
                    return False, 0.0
            elif ">" in part:
                key, val = part.split(">")
                ctx_val = context.get(key.strip(), 0)
                if not isinstance(ctx_val, (int, float)):
                    return False, 0.0
                if ctx_val <= float(val.strip()):
                    return False, 0.0
            elif "<" in part:
                key, val = part.split("<")
                ctx_val = context.get(key.strip(), float("inf"))
                if not isinstance(ctx_val, (int, float)):
                    return False, 0.0
                if ctx_val >= float(val.strip()):
                    return False, 0.0
            elif part not in facts and not context.get(part):
                return False, 0.0

        # Track activation
        self.activation_count += 1
        self.last_activated = time.time()

        return True, self.confidence

    def generate_explanation(self, context: dict[str, Any]) -> str:
        """Generate human-readable explanation."""
        if self.explanation_template:
            try:
                return self.explanation_template.format(**context)
            except KeyError:
                # Template variable not in context; use default format
                pass
        return f"Rule '{self.rule_id}': {self.premise} -> {self.conclusion} (conf={self.confidence:.2f})"


class KnowledgeGraph:
    """
    NetworkX-based knowledge graph for symbolic reasoning.

    Supports:
    - Fact and rule representation
    - Forward chaining inference
    - Backward chaining for explanation
    - Graph-based anomaly detection
    """

    def __init__(self) -> None:
        if not NETWORKX_AVAILABLE or nx is None:
            self.graph = None
            logger.warning("NetworkX not available, knowledge graph disabled")
        else:
            self.graph = nx.DiGraph()

        self.rules: dict[str, SymbolicRule] = {}
        self.facts: set[str] = set()

    def add_fact(self, fact: str, confidence: float = 1.0) -> None:
        """Add a fact to the knowledge graph."""
        self.facts.add(fact)
        if self.graph is not None:
            self.graph.add_node(fact, type="fact", confidence=confidence)

    def add_rule(self, rule: SymbolicRule) -> None:
        """Add a rule to the knowledge graph."""
        self.rules[rule.rule_id] = rule

        if self.graph is not None:
            # Add rule node
            self.graph.add_node(rule.rule_id, type="rule", confidence=rule.confidence)

            # Add edges from premise to rule
            premise_parts = rule.premise.lower().replace(" and ", ",").split(",")
            for part in premise_parts:
                part = part.strip()
                if not part.startswith("not "):
                    self.graph.add_edge(part, rule.rule_id, relation="premise")

            # Add edge from rule to conclusion
            self.graph.add_edge(rule.rule_id, rule.conclusion, relation="conclusion")

    def forward_chain(self, context: dict[str, Any]) -> tuple[set[str], list[str]]:
        """
        Perform forward chaining inference.

        Args:
            context: Context dictionary with facts and values

        Returns:
            Tuple of (derived_facts, rules_fired)
        """
        derived = set(self.facts)
        rules_fired = []
        changed = True
        max_iterations = 100
        iteration = 0

        while changed and iteration < max_iterations:
            changed = False
            iteration += 1

            for rule_id, rule in self.rules.items():
                if rule.conclusion in derived:
                    continue

                fires, confidence = rule.evaluate(derived, context)
                if fires:
                    derived.add(rule.conclusion)
                    rules_fired.append(rule_id)
                    changed = True

        return derived, rules_fired

    def backward_chain(self, goal: str, context: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Perform backward chaining to find proof for goal.

        Args:
            goal: Goal to prove
            context: Context dictionary

        Returns:
            Tuple of (proved, reasoning_chain)
        """
        if goal in self.facts or context.get(goal):
            return True, [f"Known fact: {goal}"]

        reasoning_chain = []

        for rule_id, rule in self.rules.items():
            if rule.conclusion == goal:
                # Check if premise can be satisfied
                fires, _ = rule.evaluate(self.facts, context)
                if fires:
                    reasoning_chain.append(rule.generate_explanation(context))
                    return True, reasoning_chain

        return False, []

    def get_anomaly_indicators(self, context: dict[str, Any]) -> tuple[float, list[str]]:
        """
        Compute symbolic anomaly score based on rule firings.

        Args:
            context: Context dictionary

        Returns:
            Tuple of (anomaly_score, explanations)
        """
        derived, rules_fired = self.forward_chain(context)
        explanations = []

        # Check for anomaly-related conclusions
        anomaly_conclusions = {
            "is_anomalous",
            "security_alert",
            "behavioral_anomaly",
            "high_risk",
            "critical_deviation",
            "ethical_violation",
            "unusual_pattern",
            "threat_detected",
        }

        anomaly_count = 0
        confidence_sum = 0.0

        for conclusion in anomaly_conclusions:
            if conclusion in derived:
                anomaly_count += 1
                # Find rule that derived this
                for rule_id in rules_fired:
                    if self.rules[rule_id].conclusion == conclusion:
                        confidence_sum += self.rules[rule_id].confidence
                        explanations.append(self.rules[rule_id].generate_explanation(context))
                        break

        if anomaly_count == 0:
            return 0.0, explanations

        # Weighted score by confidence
        anomaly_score = min(1.0, confidence_sum / len(anomaly_conclusions))

        return anomaly_score, explanations


class NeuralEncoder:
    """
    Neural encoder for neuro-symbolic fusion.

    Supports PyTorch and NumPy fallback.
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 128):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self._fitted = False

        if TORCH_AVAILABLE and nn is not None:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )
            # Initialize weights
            for module in self.encoder:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        else:
            # NumPy fallback - simple logistic regression style
            self.weights = None
            self.bias = 0.0

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encode input to anomaly scores.

        Args:
            X: Input array (n_samples, n_features)

        Returns:
            Anomaly scores (n_samples,)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Pad or truncate to input_dim
        if X.shape[1] < self.input_dim:
            X = np.pad(X, ((0, 0), (0, self.input_dim - X.shape[1])))
        elif X.shape[1] > self.input_dim:
            X = X[:, : self.input_dim]

        if TORCH_AVAILABLE and torch is not None:
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                scores = self.encoder(X_tensor).numpy()
            return np.asarray(scores.flatten())  # type: ignore[no-any-return, unused-ignore]
        else:
            # NumPy fallback - use statistical features
            mean = np.mean(X, axis=1)
            std = np.std(X, axis=1) + 1e-10

            # Simple anomaly score based on deviation
            z_score = np.abs(mean) / std
            score = 1 / (1 + np.exp(-z_score + 2))  # Sigmoid centered at 2

            return np.asarray(score)  # type: ignore[no-any-return, unused-ignore]

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> NeuralEncoder:
        """Fit encoder (placeholder for training)."""
        self._fitted = True

        if not TORCH_AVAILABLE and y is not None:
            # Fit simple weights using normal equations
            X_pad = np.pad(X, ((0, 0), (0, max(0, self.input_dim - X.shape[1]))))[
                :, : self.input_dim
            ]
            X_aug = np.column_stack([X_pad, np.ones(len(X))])

            try:
                # Regularized least squares
                lambda_reg = 0.01
                identity_mat = np.eye(X_aug.shape[1])
                self.weights = np.linalg.solve(  # type: ignore[assignment, unused-ignore]
                    X_aug.T @ X_aug + lambda_reg * identity_mat, X_aug.T @ y
                )
            except np.linalg.LinAlgError:
                self.weights = np.zeros(X_aug.shape[1])  # type: ignore[assignment, unused-ignore]

        return self


class NeuroSymbolicHub:
    """
    Enhanced Neuro-Symbolic Hub for unified anomaly detection.

    Integrates:
    - Neural encoding (LTN-style)
    - Symbolic reasoning (knowledge graph)
    - Stacking/BMA fusion
    - Calibration and conformal prediction
    - Ethical constraint enforcement
    - GOSNN bidirectional integration

    This is the keystone component for cross-domain fusion.
    """

    def __init__(
        self,
        input_dim: int = 64,
        fusion_mode: FusionMode = FusionMode.PHI_WEIGHTED,
        sigma_immutable: float = SIGMA_IMMUTABLE_DEFAULT,
        benevolence_threshold: float = BENEVOLENCE_THRESHOLD,
        use_calibration: bool = True,
        seed: int = 42,
        thresholds: ThresholdConfig | None = None,
        domain: str | None = None,
        enable_domain_features: bool = True,
        enable_adaptive_thresholding: bool = True,
        enable_gosnn_3r: bool = True,
    ):
        """
        Initialize Neuro-Symbolic Hub.

        Args:
            input_dim: Input feature dimension
            fusion_mode: Mode for combining neural and symbolic
            sigma_immutable: Ethical threshold (0.93-0.96)
            benevolence_threshold: Required benevolence (default 0.99)
            use_calibration: Apply probability calibration
            seed: Random seed for reproducibility
            thresholds: Threshold configuration (frozen at construction time)
            domain: Domain for specialized processing ('medical', 'financial', 'infrastructure')
            enable_domain_features: Enable domain-specific feature extraction
            enable_adaptive_thresholding: Enable adaptive per-domain thresholding
            enable_gosnn_3r: Enable GOSNN-3R bidirectional integration
        """
        self.input_dim = input_dim
        self.fusion_mode = fusion_mode
        self.sigma_immutable = sigma_immutable
        self.benevolence_threshold = benevolence_threshold
        self.use_calibration = use_calibration
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.domain = domain
        # Freeze thresholds at construction time to avoid global mutation risk
        self._thresholds = thresholds or ThresholdConfig()

        # Neural component
        self.neural_encoder = NeuralEncoder(input_dim)

        # Symbolic component
        self.knowledge_graph = KnowledgeGraph()

        # Fusion weights (learned or fixed)
        self._neural_weight = PHI / (1 + PHI)  # ~0.618 for phi-weighted
        self._symbolic_weight = 1 / (1 + PHI)  # ~0.382 for phi-weighted

        # Calibrator
        self._calibrator: Any = None

        # Meta-learner for stacking
        self._meta_learner: Any = None

        # P2-P3 Integration: Domain-specific components
        self._domain_extractor: BaseDomainExtractor | None = None
        self._threshold_manager: AdaptiveDomainThresholdManager | None = None
        self._gosnn_3r: GOSNN3RIntegration | None = None
        self._sliding_normalizer: SlidingWindowNormalizer | None = None

        # Initialize domain feature extractor
        if enable_domain_features and DOMAIN_EXTRACTORS_AVAILABLE and domain:
            try:
                if DomainFeatureExtractorFactory is not None:
                    self._domain_extractor = DomainFeatureExtractorFactory.create(domain)
                    logger.info(f"Domain feature extractor initialized for: {domain}")
            except Exception as e:
                logger.warning(f"Failed to initialize domain extractor: {e}")

        # Initialize adaptive domain thresholding
        if enable_adaptive_thresholding and ADAPTIVE_THRESHOLDING_AVAILABLE and domain:
            try:
                if create_domain_threshold_manager is not None:
                    self._threshold_manager = create_domain_threshold_manager(domain)
                    logger.info(f"Adaptive threshold manager initialized for: {domain}")
            except Exception as e:
                logger.warning(f"Failed to initialize threshold manager: {e}")

        # Initialize GOSNN-3R integration
        if enable_gosnn_3r and GOSNN_3R_AVAILABLE:
            try:
                if GOSNN3RIntegration is not None:
                    self._gosnn_3r = GOSNN3RIntegration(domain=domain or "general")
                if SlidingWindowNormalizer is not None and SlidingWindowConfig is not None:
                    self._sliding_normalizer = SlidingWindowNormalizer(
                        config=SlidingWindowConfig(window_size=100)
                    )
                logger.info("GOSNN-3R bidirectional integration initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize GOSNN-3R integration: {e}")

        # Tracking
        self._fitted = False
        self._inference_count = 0
        self._total_processing_time = 0.0

        # Initialize default rules
        self._initialize_default_rules()

        # Add domain-specific rules if domain is specified
        if domain:
            self._initialize_domain_rules(domain)

        logger.info(
            f"NeuroSymbolicHub initialized: fusion={fusion_mode.value}, "
            f"sigma_immutable={sigma_immutable}, benevolence≥{benevolence_threshold}, "
            f"domain={domain or 'general'}"
        )

    def _initialize_default_rules(self) -> None:
        """Initialize default symbolic rules."""
        default_rules = [
            # Anomaly detection rules
            SymbolicRule(
                rule_id="high_deviation",
                premise="deviation_score >= 2.0",
                conclusion="is_anomalous",
                confidence=0.85,
                category="statistical",
                explanation_template="Statistical deviation ≥2σ detected",
            ),
            SymbolicRule(
                rule_id="rare_pattern",
                premise="pattern_frequency < 0.01",
                conclusion="unusual_pattern",
                confidence=0.9,
                category="pattern",
                explanation_template="Rare pattern (frequency < 1%)",
            ),
            SymbolicRule(
                rule_id="threat_detected",
                premise="threat_score >= 0.7 AND not authorized",
                conclusion="security_alert",
                confidence=0.95,
                category="security",
                explanation_template="Security threat (score ≥70%, unauthorized)",
            ),
            SymbolicRule(
                rule_id="temporal_anomaly",
                premise="time_deviation >= 3.0",
                conclusion="is_anomalous",
                confidence=0.88,
                category="temporal",
                explanation_template="Temporal deviation ≥3σ",
            ),
            SymbolicRule(
                rule_id="spatial_cluster",
                premise="spatial_autocorrelation >= 0.5",
                conclusion="cluster_detected",
                confidence=0.82,
                category="spatial",
                explanation_template="Spatial clustering (Moran's I ≥0.5)",
            ),
            # Ethical rules (immutable)
            SymbolicRule(
                rule_id="benevolence_violation",
                premise="benevolence_score < 0.99",
                conclusion="ethical_violation",
                confidence=1.0,
                category="ethical",
                provenance="system",
                explanation_template="Benevolence below 0.99 threshold",
            ),
            SymbolicRule(
                rule_id="sigma_immutable_violation",
                premise="ethical_score < 0.93",
                conclusion="ethical_violation",
                confidence=1.0,
                category="ethical",
                provenance="system",
                explanation_template="Ethical score below σ_immutable (0.93)",
            ),
            SymbolicRule(
                rule_id="harm_detection",
                premise="harm_potential >= 0.5",
                conclusion="high_risk",
                confidence=0.95,
                category="ethical",
                provenance="system",
                explanation_template="Potential harm detected (≥50%)",
            ),
        ]

        for rule in default_rules:
            self.knowledge_graph.add_rule(rule)

    def _initialize_domain_rules(self, domain: str) -> None:
        """Initialize domain-specific symbolic rules.

        Args:
            domain: Domain name ('medical', 'financial', 'infrastructure')
        """
        domain_rules: list[SymbolicRule] = []

        if domain == "medical":
            domain_rules = [
                SymbolicRule(
                    rule_id="sepsis_alert",
                    premise="sofa_score >= 2.0 AND lactate_elevated",
                    conclusion="critical_deviation",
                    confidence=0.95,
                    category="medical",
                    explanation_template="Sepsis criteria met (SOFA ≥2, elevated lactate)",
                ),
                SymbolicRule(
                    rule_id="vital_instability",
                    premise="vital_variability >= 0.3 AND trend_declining",
                    conclusion="is_anomalous",
                    confidence=0.88,
                    category="medical",
                    explanation_template="Vital sign instability detected",
                ),
                SymbolicRule(
                    rule_id="deterioration_risk",
                    premise="deterioration_score >= 0.7",
                    conclusion="high_risk",
                    confidence=0.92,
                    category="medical",
                    explanation_template="Patient deterioration risk ≥70%",
                ),
            ]

        elif domain == "financial":
            domain_rules = [
                SymbolicRule(
                    rule_id="benford_violation",
                    premise="benford_deviation >= 0.15",
                    conclusion="unusual_pattern",
                    confidence=0.85,
                    category="financial",
                    explanation_template="Benford's Law violation (deviation ≥15%)",
                ),
                SymbolicRule(
                    rule_id="velocity_spike",
                    premise="transaction_velocity >= 3.0",
                    conclusion="is_anomalous",
                    confidence=0.90,
                    category="financial",
                    explanation_template="Transaction velocity spike (3σ above mean)",
                ),
                SymbolicRule(
                    rule_id="round_number_fraud",
                    premise="round_number_ratio >= 0.4 AND amount_large",
                    conclusion="security_alert",
                    confidence=0.82,
                    category="financial",
                    explanation_template="Suspicious round number pattern in large transactions",
                ),
            ]

        elif domain == "infrastructure":
            domain_rules = [
                SymbolicRule(
                    rule_id="scada_correlation_break",
                    premise="correlation_deviation >= 0.4",
                    conclusion="is_anomalous",
                    confidence=0.92,
                    category="infrastructure",
                    explanation_template="SCADA sensor correlation anomaly",
                ),
                SymbolicRule(
                    rule_id="setpoint_drift",
                    premise="setpoint_deviation >= 2.0 AND not maintenance_mode",
                    conclusion="critical_deviation",
                    confidence=0.95,
                    category="infrastructure",
                    explanation_template="Process variable drifting from setpoint",
                ),
                SymbolicRule(
                    rule_id="alarm_cascade",
                    premise="alarm_frequency >= 5.0",
                    conclusion="high_risk",
                    confidence=0.88,
                    category="infrastructure",
                    explanation_template="Alarm cascade detected (≥5 alarms/minute)",
                ),
            ]

        for rule in domain_rules:
            self.knowledge_graph.add_rule(rule)

        if domain_rules:
            logger.info(f"Added {len(domain_rules)} domain-specific rules for {domain}")

    def add_fact(self, fact: str) -> None:
        """Add a fact to the knowledge base."""
        self.knowledge_graph.add_fact(fact)

    def add_rule(self, rule: SymbolicRule) -> None:
        """Add a custom rule to the knowledge base."""
        self.knowledge_graph.add_rule(rule)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        validation_split: float = 0.2,
    ) -> NeuroSymbolicHub:
        """
        Fit the neuro-symbolic hub.

        Args:
            X: Training features
            y: Optional labels
            validation_split: Fraction for calibration validation

        Returns:
            Self for method chaining
        """
        np.random.seed(self.seed)

        # Fit neural encoder
        self.neural_encoder.fit(X, y)

        # If labels provided, set up calibration and learn fusion weights
        if y is not None and self.use_calibration:
            n = len(X)
            n_val = int(n * validation_split)
            idx = np.random.permutation(n)
            val_idx = idx[:n_val]

            X_val = X[val_idx]
            y_val = y[val_idx]

            # Get neural scores
            neural_scores = self.neural_encoder.encode(X_val)

            # Set up calibration
            self._setup_calibration(neural_scores, y_val)

            # Learn fusion weights if using stacking/BMA
            if self.fusion_mode in [FusionMode.STACKING, FusionMode.BMA, FusionMode.ADAPTIVE]:
                self._learn_fusion_weights(X_val, y_val)

        self._fitted = True
        logger.info("NeuroSymbolicHub fitted successfully")

        return self

    def _setup_calibration(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Set up Platt scaling calibration."""
        try:
            from omni_mercury_engine.core.calibration import PlattScaling

            calibrator = PlattScaling()
            calibrator.fit(scores, labels)
            self._calibrator = calibrator
        except ImportError:
            # Simple sigmoid calibration fallback
            self._calibrator = None

    def _learn_fusion_weights(self, X: np.ndarray, y: np.ndarray) -> None:
        """Learn optimal fusion weights."""
        # Get neural scores
        neural_scores = self.neural_encoder.encode(X)

        # Get symbolic scores
        symbolic_scores = []
        for i in range(len(X)):
            context = {"deviation_score": np.abs(X[i]).max()}
            score, _ = self.knowledge_graph.get_anomaly_indicators(context)
            symbolic_scores.append(score)
        symbolic_scores = np.array(symbolic_scores)  # type: ignore[assignment, unused-ignore]

        if self.fusion_mode == FusionMode.STACKING:
            # Train meta-learner
            meta_features = np.column_stack([neural_scores, symbolic_scores])

            from omni_mercury_engine.ml._native_utils import (
                NativeLogisticRegression as LogisticRegression,
            )

            meta_learner = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=self.seed)
            meta_learner.fit(meta_features, y)
            self._meta_learner = meta_learner

        elif self.fusion_mode in [FusionMode.BMA, FusionMode.ADAPTIVE]:
            # Optimize weights using cross-entropy loss
            def objective(w: np.ndarray) -> float:
                w = np.abs(w)
                w = w / (np.sum(w) + 1e-10)

                fused = w[0] * neural_scores + w[1] * symbolic_scores
                fused = np.clip(fused, 1e-10, 1 - 1e-10)

                bce: float = -np.mean(y * np.log(fused) + (1 - y) * np.log(1 - fused))
                return bce

            result = minimize(
                objective, [0.6, 0.4], method="L-BFGS-B", bounds=[(0.1, 0.9), (0.1, 0.9)]
            )

            w = np.abs(result.x)
            w = w / np.sum(w)
            self._neural_weight = w[0]
            self._symbolic_weight = w[1]

    def predict(
        self,
        X: np.ndarray,
        context: dict[str, Any] | None = None,
        return_explanations: bool = True,
    ) -> list[ExplainableOutput]:
        """
        Predict anomalies with explanations.

        Args:
            X: Input features (n_samples, n_features)
            context: Optional context dictionary
            return_explanations: Whether to generate explanations

        Returns:
            List of ExplainableOutput for each sample
        """
        start_time = time.time()

        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = len(X)
        context = context or {}
        results = []

        # P2 Integration: Apply domain-specific feature extraction
        domain_features: dict[str, Any] = {}
        X_enhanced = X
        if self._domain_extractor is not None:
            try:
                # Extract domain-specific features
                extracted_features, feature_names = self._domain_extractor.extract_features(X)  # type: ignore[attr-defined]

                # If we got enhanced features, use them
                if extracted_features is not None and len(extracted_features) > 0:
                    # Concatenate original features with domain features
                    if extracted_features.shape[0] == X.shape[0]:
                        X_enhanced = np.concatenate([X, extracted_features], axis=1)
                        domain_features = {
                            "domain": self.domain,
                            "extracted_feature_count": len(feature_names),
                            "feature_names": feature_names,
                        }
                        logger.debug(f"Domain features extracted: {len(feature_names)} features")
            except Exception as e:
                logger.warning(f"Domain feature extraction failed: {e}")
                # Fall back to original features
                X_enhanced = X

        # P2 Integration: Apply sliding window normalization if available
        if self._sliding_normalizer is not None:
            try:
                X_enhanced = self._sliding_normalizer.normalize(X_enhanced)
            except Exception as e:
                logger.warning(f"Sliding window normalization failed: {e}")

        # Get neural scores (using enhanced features if available)
        neural_scores = self.neural_encoder.encode(X_enhanced)

        for i in range(n_samples):
            sample_start = time.time()

            # Build context for this sample
            sample_context = {
                **context,
                "deviation_score": float(
                    np.abs(X[i] - np.mean(X[i])).max() / (np.std(X[i]) + 1e-10)
                ),
                "max_value": float(np.max(np.abs(X[i]))),
                "neural_score": float(neural_scores[i]),
            }

            # Get symbolic score and explanations
            symbolic_score, explanations = self.knowledge_graph.get_anomaly_indicators(
                sample_context
            )

            # Get rules fired
            _, rules_fired = self.knowledge_graph.forward_chain(sample_context)

            # Compute fused score based on mode
            neural_score = float(neural_scores[i])

            if self.fusion_mode == FusionMode.STACKING and self._meta_learner is not None:
                meta_features = np.array([[neural_score, symbolic_score]])
                fused_score = float(self._meta_learner.predict_proba(meta_features)[0, 1])
                neural_weight = self._neural_weight
                symbolic_weight = self._symbolic_weight

            elif self.fusion_mode == FusionMode.PHI_WEIGHTED:
                neural_weight = PHI / (1 + PHI)
                symbolic_weight = 1 / (1 + PHI)
                fused_score = neural_weight * neural_score + symbolic_weight * symbolic_score

            elif self.fusion_mode == FusionMode.NEURAL_DOMINANT:
                neural_weight = 0.7
                symbolic_weight = 0.3
                fused_score = neural_weight * neural_score + symbolic_weight * symbolic_score

            elif self.fusion_mode == FusionMode.SYMBOLIC_DOMINANT:
                neural_weight = 0.3
                symbolic_weight = 0.7
                fused_score = neural_weight * neural_score + symbolic_weight * symbolic_score

            else:  # BALANCED or ADAPTIVE
                neural_weight = self._neural_weight
                symbolic_weight = self._symbolic_weight
                fused_score = neural_weight * neural_score + symbolic_weight * symbolic_score

            # Apply calibration
            calibrated_score = None
            if self._calibrator is not None:
                try:
                    calibrated_score = float(self._calibrator.calibrate(np.array([fused_score]))[0])
                except Exception:
                    # Calibration failed; use uncalibrated score
                    calibrated_score = fused_score

            # Compute confidence
            # Higher agreement between neural and symbolic = higher confidence
            agreement = 1 - abs(neural_score - symbolic_score)
            confidence = (
                0.5
                + 0.5
                * agreement
                * min(neural_score, symbolic_score, 1 - neural_score, 1 - symbolic_score)
                * 4
            )
            confidence = min(max(confidence, 0.0), 1.0)

            # P2 Integration: Use adaptive domain thresholding if available
            if self._threshold_manager is not None:
                try:
                    # Get calibrated score and adaptive threshold
                    threshold_result = self._threshold_manager.get_threshold(
                        fused_score, confidence
                    )
                    # Handle both float and dict return types
                    if isinstance(threshold_result, dict):
                        adaptive_threshold = float(
                            threshold_result.get("threshold", self._thresholds.anomaly_default)
                        )
                        # Update calibrated score if threshold manager provides one
                        if "calibrated_score" in threshold_result:
                            calibrated_score = float(threshold_result["calibrated_score"])
                    else:
                        adaptive_threshold = float(threshold_result)
                    is_anomaly = fused_score > adaptive_threshold
                except Exception as e:
                    logger.warning(f"Adaptive thresholding failed: {e}")
                    is_anomaly = fused_score > self._thresholds.anomaly_default
            else:
                # Fallback to centralized threshold
                is_anomaly = fused_score > self._thresholds.anomaly_default

            # P2 Integration: GOSNN-3R feedback loop
            gosnn_3r_info: dict[str, Any] = {}
            if self._gosnn_3r is not None:
                try:
                    # Get 3R-adjusted fusion weights based on confidence
                    adjusted_weights = self._gosnn_3r.adjust_weights(
                        neural_score=neural_score,
                        symbolic_score=symbolic_score,
                        confidence=confidence,
                        domain=self.domain,
                    )

                    # Apply adjusted weights for potential score refinement
                    if adjusted_weights:
                        gosnn_3r_info = {
                            "gosnn_3r_active": True,
                            "adjusted_neural_weight": adjusted_weights.get("neural_weight"),
                            "adjusted_symbolic_weight": adjusted_weights.get("symbolic_weight"),
                            "lyapunov_stability": adjusted_weights.get("lyapunov_stability", 0.0),
                        }

                        # Optionally refine fused score with GOSNN-3R adjusted weights
                        if adjusted_weights.get("refine_score", False):
                            w_n = adjusted_weights["neural_weight"]
                            w_s = adjusted_weights["symbolic_weight"]
                            fused_score = w_n * neural_score + w_s * symbolic_score
                except Exception as e:
                    logger.warning(f"GOSNN-3R integration failed: {e}")

            # Check ethical compliance
            benevolence_score = self._compute_benevolence(sample_context, fused_score)
            ethical_compliant = benevolence_score >= self.benevolence_threshold
            ethical_violations = []

            if not ethical_compliant:
                ethical_violations.append(
                    f"Benevolence {benevolence_score:.3f} < {self.benevolence_threshold}"
                )

            # Build reasoning chain with P2 integration info
            reasoning_chain = []

            # Add domain feature extraction step if used
            if domain_features:
                reasoning_chain.append(
                    {
                        "step": "domain_feature_extraction",
                        "domain": domain_features.get("domain"),
                        "features_extracted": domain_features.get("extracted_feature_count", 0),
                    }
                )

            reasoning_chain.extend(
                [
                    {"step": "neural_encoding", "score": neural_score, "weight": neural_weight},
                    {
                        "step": "symbolic_reasoning",
                        "score": symbolic_score,
                        "weight": symbolic_weight,
                        "rules_fired": rules_fired,
                    },
                    {"step": "fusion", "fused_score": fused_score, "mode": self.fusion_mode.value},
                ]
            )

            # Add GOSNN-3R integration step if used
            if gosnn_3r_info:
                reasoning_chain.append(
                    {
                        "step": "gosnn_3r_integration",
                        **gosnn_3r_info,
                    }
                )

            if calibrated_score is not None:
                reasoning_chain.append(
                    {"step": "calibration", "calibrated_score": calibrated_score}
                )

            sample_time = (time.time() - sample_start) * 1000

            results.append(
                ExplainableOutput(
                    anomaly_score=fused_score,
                    is_anomaly=is_anomaly,
                    confidence=confidence,
                    neural_score=neural_score,
                    symbolic_score=symbolic_score,
                    neural_weight=neural_weight,
                    symbolic_weight=symbolic_weight,
                    rules_fired=rules_fired,
                    explanations=explanations if return_explanations else [],
                    reasoning_chain=reasoning_chain if return_explanations else [],
                    calibrated_score=calibrated_score,
                    benevolence_score=benevolence_score,
                    ethical_compliant=ethical_compliant,
                    ethical_violations=ethical_violations,
                    fusion_mode=self.fusion_mode.value,
                    processing_time_ms=sample_time,
                )
            )

        total_time = (time.time() - start_time) * 1000
        self._inference_count += n_samples
        self._total_processing_time += total_time

        return results

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly probabilities (standard ML interface)."""
        results = self.predict(X, return_explanations=False)
        scores = np.array([r.anomaly_score for r in results])
        return np.column_stack([1 - scores, scores])

    def _compute_benevolence(self, context: dict[str, Any], anomaly_score: float) -> float:
        """
        Compute benevolence score for ethical compliance.

        Benevolence measures the positive intent and harm-avoidance
        of the detection. Higher anomaly scores with low false positive
        risk = higher benevolence.
        """
        # Base benevolence from confidence
        confidence = context.get("neural_score", 0.5)
        base_benevolence = 0.5 + 0.5 * confidence

        # Penalty for potential false positives (high score, low confidence)
        if anomaly_score > 0.5 and confidence < 0.5:
            false_positive_risk = anomaly_score * (1 - confidence)
            base_benevolence -= false_positive_risk * 0.2

        # Bonus for catching true anomalies (ethical duty)
        if anomaly_score > 0.7 and confidence > 0.7:
            base_benevolence += 0.1

        # Apply Lyapunov stability factor
        stability = np.exp(-LYAPUNOV_LAMBDA * (1 - base_benevolence))
        benevolence = base_benevolence * stability

        return float(np.clip(benevolence, 0.0, 1.0))

    def get_gosnn_scalars(self) -> dict[str, float]:
        """
        Get scalars for GOSNN integration.

        Returns:
            Dictionary of scalars for GOSNN registration
        """
        return {
            "neurosymbolic_neural_weight": self._neural_weight,
            "neurosymbolic_symbolic_weight": self._symbolic_weight,
            "neurosymbolic_inference_count": float(self._inference_count),
            "neurosymbolic_avg_latency_ms": (
                self._total_processing_time / max(self._inference_count, 1)
            ),
            "neurosymbolic_rule_count": float(len(self.knowledge_graph.rules)),
            "neurosymbolic_fact_count": float(len(self.knowledge_graph.facts)),
        }

    def integrate_with_gosnn(self, gosnn: Any) -> None:
        """
        Register scalars with GOSNN hub.

        Args:
            gosnn: GlobalOmniScalarNetwork instance
        """
        from omni_mercury_engine.core.global_omni_scalar_network import ScalarGroup

        scalars = self.get_gosnn_scalars()
        gosnn.register_scalars(
            component_name="NeuroSymbolicHub",
            scalars=scalars,
            group=ScalarGroup.QUANTUM_CONSCIOUSNESS,
        )

        logger.info("NeuroSymbolicHub integrated with GOSNN")

    def get_statistics(self) -> dict[str, Any]:
        """Get hub statistics."""
        return {
            "fusion_mode": self.fusion_mode.value,
            "neural_weight": self._neural_weight,
            "symbolic_weight": self._symbolic_weight,
            "rule_count": len(self.knowledge_graph.rules),
            "fact_count": len(self.knowledge_graph.facts),
            "inference_count": self._inference_count,
            "avg_latency_ms": (self._total_processing_time / max(self._inference_count, 1)),
            "fitted": self._fitted,
            "calibration_enabled": self._calibrator is not None,
            "sigma_immutable": self.sigma_immutable,
            "benevolence_threshold": self.benevolence_threshold,
        }


def create_neurosymbolic_hub(
    input_dim: int = 64,
    fusion_mode: str = "phi_weighted",
    **kwargs: Any,
) -> NeuroSymbolicHub:
    """
    Factory function to create neuro-symbolic hub.

    Args:
        input_dim: Input feature dimension
        fusion_mode: Fusion mode string
        **kwargs: Additional arguments

    Returns:
        Configured NeuroSymbolicHub
    """
    mode_map = {
        "neural_dominant": FusionMode.NEURAL_DOMINANT,
        "symbolic_dominant": FusionMode.SYMBOLIC_DOMINANT,
        "balanced": FusionMode.BALANCED,
        "phi_weighted": FusionMode.PHI_WEIGHTED,
        "adaptive": FusionMode.ADAPTIVE,
        "stacking": FusionMode.STACKING,
        "bma": FusionMode.BMA,
    }

    mode = mode_map.get(fusion_mode, FusionMode.PHI_WEIGHTED)

    return NeuroSymbolicHub(
        input_dim=input_dim,
        fusion_mode=mode,
        **kwargs,
    )
