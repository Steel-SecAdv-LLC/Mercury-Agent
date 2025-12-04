"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

"""
Unified Neurosymbolic Engine - Fusion of neural networks and symbolic reasoning

This module provides the core neurosymbolic AI capabilities for OMNI ♱ AVA,
combining Logic Tensor Networks (LTN) with symbolic reasoning for:
- Explainable anomaly detection
- Ethical constraint enforcement
- Hybrid neural-symbolic inference

Architecture:
    1. LogicTensorNetwork: Neural network with fuzzy logic operations
    2. SymbolicReasoningLayer: PyReason-inspired rule-based reasoning
    3. NeurosymbolicEngine: Unified interface for hybrid inference

Research Sources:
    - LTN: Logic Tensor Networks (Serafini & Garcez, 2016)
    - PyReason: Temporal First-Order Logic Explainable AI (AAAI 2023)
    - Neuro-symbolic AI: https://arxiv.org/pdf/2302.13482.pdf

Example:
    Basic usage::

        from omni_anomaly_engine.models.neurosymbolic import NeurosymbolicEngine
        import numpy as np

        engine = NeurosymbolicEngine(input_dim=64)
        engine.add_fact("missing_person")
        engine.add_fact("child")

        # Neural inference
        features = np.random.randn(10, 64)
        result = engine.predict(features)

        # Symbolic inference
        symbolic = engine.symbolic_inference("priority_high")
        print(f"Derived: {symbolic['result']}, Explanation: {symbolic['explanation']}")
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

_FOUNDATION_HASH = "D19L12E19A92"

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available, neurosymbolic engine will use limited functionality")


class ReasoningMode(Enum):
    """Reasoning modes for hybrid inference."""

    NEURAL_ONLY = "neural_only"
    SYMBOLIC_ONLY = "symbolic_only"
    HYBRID = "hybrid"
    NEURO_SYMBOLIC_ATTENTION = "neuro_symbolic_attention"


@dataclass
class SymbolicRule:
    """Represents a symbolic logical rule with explainability support.

    Attributes:
        premise: Logical condition(s) that must be satisfied
        conclusion: Result derived when premise is satisfied
        confidence: Rule confidence score (0.0 to 1.0)
        name: Optional human-readable rule name
        category: Rule category for organization
        explanation_template: Template for generating explanations
    """

    premise: str
    conclusion: str
    confidence: float
    name: str = ""
    category: str = "general"
    explanation_template: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"rule_{hash((self.premise, self.conclusion)) % 10000:04d}"
        if not self.explanation_template:
            self.explanation_template = "{conclusion} derived from: {premise}"

    def generate_explanation(self) -> str:
        """Generate human-readable explanation for this rule."""
        return self.explanation_template.format(
            premise=self.premise,
            conclusion=self.conclusion,
            confidence=self.confidence,
        )


@dataclass
class ReasoningResult:
    """Result from symbolic or hybrid reasoning.

    Attributes:
        result: Whether the query was derived
        confidence: Confidence in the result
        explanation: Human-readable explanation
        method: Reasoning method used
        rules_fired: List of rules that contributed
        neural_contribution: Neural network contribution (if hybrid)
        symbolic_contribution: Symbolic contribution (if hybrid)
    """

    result: bool
    confidence: float
    explanation: str
    method: str
    rules_fired: list[str] = field(default_factory=list)
    neural_contribution: float = 0.0
    symbolic_contribution: float = 0.0


class LogicTensorNetwork:
    """
    Logic Tensor Network for combining neural and symbolic reasoning.
    Implements fuzzy logic operations over neural network outputs.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if TORCH_AVAILABLE:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
            )
            self.logic_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        """Forward pass through LTN"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for neural forward pass")
        features = self.encoder(x)
        logits = self.logic_head(features)
        return torch.sigmoid(logits)


class SymbolicReasoningLayer:
    """
    Symbolic reasoning layer for explainable AI.

    Provides PyReason-inspired temporal first-order logic reasoning
    for generating human-readable explanations of decisions.

    Based on: PyReason - Temporal First-Order Logic Explainable AI (AAAI 2023)
    https://pyreason.readthedocs.io/

    Enhanced Features:
    - Configurable neural/symbolic weighting (default 60/40)
    - Optional confidence-weighted symbolic scoring
    - Case-insensitive context key normalization
    - Humanitarian context awareness for survivor-first principles
    """

    def __init__(
        self,
        explainability_threshold: float = 0.7,
        temporal_logic: bool = True,
        graph_based: bool = True,
        neural_weight: float = 0.6,
        symbolic_weight: float = 0.4,
        use_confidence_weights: bool = False,
    ):
        """Initialize symbolic reasoning layer.

        Args:
            explainability_threshold: Min confidence for explanations
            temporal_logic: Enable temporal reasoning
            graph_based: Enable graph-based reasoning
            neural_weight: Weight for neural component (default 0.6)
            symbolic_weight: Weight for symbolic component (default 0.4)
            use_confidence_weights: Use rule confidences for symbolic scoring
        """
        self.explainability_threshold = explainability_threshold
        self.temporal_logic = temporal_logic
        self.graph_based = graph_based
        self.neural_weight = neural_weight
        self.symbolic_weight = symbolic_weight
        self.use_confidence_weights = use_confidence_weights
        self.rules: list[SymbolicRule] = []

    def add_rule(self, rule: SymbolicRule) -> None:
        """Add a symbolic reasoning rule."""
        self.rules.append(rule)

    def reason(
        self,
        neural_score: float,
        context: dict[str, Any],
    ) -> ReasoningResult:
        """
        Perform hybrid neuro-symbolic reasoning.

        Combines neural network output with symbolic rule evaluation
        for explainable anomaly detection.

        Args:
            neural_score: Output from neural detector (0.0 to 1.0)
            context: Context dict with code metrics and features

        Returns:
            ReasoningResult with explanation
        """
        rules_fired: list[SymbolicRule] = []
        explanations: list[str] = []

        for rule in self.rules:
            satisfied = self._evaluate_rule(rule, context)
            if satisfied and rule.confidence >= self.explainability_threshold:
                rules_fired.append(rule)
                explanations.append(rule.generate_explanation())

        # Calculate symbolic confidence - optionally weighted by rule confidences
        if self.use_confidence_weights and self.rules:
            # Confidence-weighted scoring: sum of fired rule confidences / total possible
            eligible_rules = [
                r for r in self.rules if r.confidence >= self.explainability_threshold
            ]
            total_weight = sum(r.confidence for r in eligible_rules) if eligible_rules else 1.0
            fired_weight = sum(r.confidence for r in rules_fired) if rules_fired else 0.0
            symbolic_confidence = fired_weight / total_weight if total_weight > 0 else 0.0
        else:
            # Original behavior: simple ratio of rules fired
            symbolic_confidence = len(rules_fired) / len(self.rules) if self.rules else 0.0

        # Normalize weights to ensure they sum to 1.0
        total_weight = self.neural_weight + self.symbolic_weight
        nw = self.neural_weight / total_weight if total_weight > 0 else 0.6
        sw = self.symbolic_weight / total_weight if total_weight > 0 else 0.4

        # Weighted combination using configurable weights
        combined_confidence = nw * neural_score + sw * symbolic_confidence

        # Apply humanitarian context boost for survivor-first principles
        if context.get("humanitarian_context") or context.get("missing_person"):
            # Increase sensitivity for humanitarian applications
            combined_confidence = min(1.0, combined_confidence * 1.1)

        is_anomaly = combined_confidence > 0.5

        explanation = (
            "; ".join(explanations) if explanations else "Based on neural network analysis only."
        )

        return ReasoningResult(
            result=is_anomaly,
            confidence=combined_confidence,
            explanation=explanation,
            method="hybrid",
            rules_fired=[r.name for r in rules_fired],
            neural_contribution=neural_score,
            symbolic_contribution=symbolic_confidence,
        )

    def _evaluate_rule(self, rule: SymbolicRule, context: dict[str, Any]) -> bool:
        """Evaluate if a rule's premise is satisfied by context.

        Enhanced with case-insensitive key matching for robust premise evaluation.
        Supports: boolean checks, negation (not), comparisons (>, <, >=, <=, ==).
        """
        # Normalize context keys to lowercase for case-insensitive matching
        normalized_context: dict[str, Any] = {k.lower(): v for k, v in context.items()}

        # Parse premise into parts (supports 'and' conjunction)
        premise_parts = rule.premise.lower().replace(" and ", ",").split(",")

        for raw_part in premise_parts:
            part = raw_part.strip()
            if not part:
                continue

            if part.startswith("not "):
                key = part[4:].strip()
                if normalized_context.get(key):
                    return False
            # Check for comparisons with enhanced operator support
            elif ">=" in part:
                key, val = part.split(">=", 1)
                try:
                    if normalized_context.get(key.strip(), 0) < float(val.strip()):
                        return False
                except (ValueError, TypeError):
                    return False
            elif "<=" in part:
                key, val = part.split("<=", 1)
                try:
                    if normalized_context.get(key.strip(), 0) > float(val.strip()):
                        return False
                except (ValueError, TypeError):
                    return False
            elif "==" in part:
                key, val = part.split("==", 1)
                ctx_val = normalized_context.get(key.strip())
                try:
                    if ctx_val != float(val.strip()):
                        return False
                except (ValueError, TypeError):
                    # Fall back to string comparison
                    if str(ctx_val).lower() != val.strip().lower():
                        return False
            elif ">" in part:
                key, val = part.split(">", 1)
                try:
                    if normalized_context.get(key.strip(), 0) <= float(val.strip()):
                        return False
                except (ValueError, TypeError):
                    return False
            elif "<" in part:
                key, val = part.split("<", 1)
                try:
                    if normalized_context.get(key.strip(), 0) >= float(val.strip()):
                        return False
                except (ValueError, TypeError):
                    return False
            # Boolean check - key must be truthy
            elif not normalized_context.get(part):
                return False
        return True

    def explain_decision(
        self,
        reasoning_result: ReasoningResult,
    ) -> str:
        """Generate human-readable explanation of reasoning decision."""
        explanation = f"Decision: {'ANOMALOUS' if reasoning_result.result else 'NORMAL'} "
        explanation += f"(confidence: {reasoning_result.confidence:.2%})\n\n"

        if reasoning_result.rules_fired:
            explanation += "Reasoning:\n"
            for rule_name in reasoning_result.rules_fired:
                explanation += f"  - Rule '{rule_name}' fired\n"
        else:
            explanation += "Based on neural network analysis only.\n"

        explanation += f"\nNeural contribution: {reasoning_result.neural_contribution:.2%}\n"
        explanation += f"Symbolic contribution: {reasoning_result.symbolic_contribution:.2%}\n"

        return explanation


class NeurosymbolicEngine:
    """
    Unified Neurosymbolic reasoning engine combining LTN with symbolic logic.

    Provides:
    - Neural inference via Logic Tensor Networks
    - Symbolic inference via knowledge base and rules
    - Hybrid reasoning combining both approaches
    - Explainable AI with human-readable explanations
    - Ethical constraint enforcement
    - Optional integration with PercipienceEngine for ethical alignment

    This is the primary neurosymbolic component for OMNI ♱ AVA.

    Enhanced Features:
    - Configurable neural/symbolic weighting via SymbolicReasoningLayer
    - Optional ethical alignment verification via PercipienceEngine
    - Humanitarian context awareness for survivor-first principles
    - Robust feature indexing with class constants
    """

    def __init__(
        self,
        input_dim: int = 64,
        reasoning_mode: ReasoningMode = ReasoningMode.HYBRID,
        explainability_threshold: float = 0.7,
        ethical_alignment_engine: Any | None = None,
        neural_weight: float = 0.6,
        symbolic_weight: float = 0.4,
        use_confidence_weights: bool = False,
    ):
        """Initialize Neurosymbolic Engine.

        Args:
            input_dim: Input feature dimension for LTN
            reasoning_mode: Default reasoning mode (HYBRID, NEURAL_ONLY, SYMBOLIC_ONLY)
            explainability_threshold: Minimum confidence for explanations
            ethical_alignment_engine: Optional PercipienceEngine for ethical verification
            neural_weight: Weight for neural component in hybrid reasoning (default 0.6)
            symbolic_weight: Weight for symbolic component in hybrid reasoning (default 0.4)
            use_confidence_weights: Use rule confidences for symbolic scoring
        """
        self.input_dim = input_dim
        self.reasoning_mode = reasoning_mode
        self.golden_ratio = 0.618
        self.quantum_factor = 1.2
        self.ethical_alignment_engine = ethical_alignment_engine

        if TORCH_AVAILABLE:
            self.ltn = LogicTensorNetwork(input_dim)
        else:
            self.ltn = None

        self.knowledge_base: list[SymbolicRule] = []
        self.facts: set[str] = set()

        # Symbolic reasoning layer with configurable weights
        self.symbolic_layer = SymbolicReasoningLayer(
            explainability_threshold=explainability_threshold,
            neural_weight=neural_weight,
            symbolic_weight=symbolic_weight,
            use_confidence_weights=use_confidence_weights,
        )

        self.omni_scalars = {
            "omni_logic": 1.40,
            "omni_reason": 1.38,
            "omni_wisdom": 1.42,
            "omni_understanding": 1.36,
            "omni_interpretation": 1.35,
        }

        self._initialize_ethical_rules()
        self._initialize_anomaly_rules()

        logging.info(f"NeurosymbolicEngine initialized (mode={reasoning_mode.value})")

    def _initialize_ethical_rules(self):
        """Initialize fundamental ethical rules"""
        ethical_rules = [
            SymbolicRule(
                premise="missing_person AND child", conclusion="priority_high", confidence=1.0
            ),
            SymbolicRule(
                premise="requires_consent AND NOT consent_given",
                conclusion="action_blocked",
                confidence=1.0,
            ),
            SymbolicRule(
                premise="privacy_risk AND NOT explicit_authorization",
                conclusion="apply_privacy_filter",
                confidence=0.95,
            ),
        ]

        self.knowledge_base.extend(ethical_rules)

    def _initialize_anomaly_rules(self):
        """Initialize anomaly detection rules for symbolic layer."""
        anomaly_rules = [
            SymbolicRule(
                name="high_complexity_rule",
                premise="cyclomatic_complexity > 10 AND code_length > 100",
                conclusion="is_anomalous",
                confidence=0.8,
                category="code_analysis",
                explanation_template="High complexity detected: cyclomatic > 10, length > 100",
            ),
            SymbolicRule(
                name="unusual_pattern_rule",
                premise="pattern_frequency < 0.01",
                conclusion="is_anomalous",
                confidence=0.9,
                category="pattern_analysis",
                explanation_template="Unusual pattern detected: frequency < 1%",
            ),
            SymbolicRule(
                name="refactoring_candidate_rule",
                premise="code_duplication > 0.5 AND maintainability_index < 20",
                conclusion="needs_refactoring",
                confidence=0.85,
                category="code_quality",
                explanation_template="Refactoring needed: high duplication, low maintainability",
            ),
            SymbolicRule(
                name="security_anomaly_rule",
                premise="threat_score > 0.7 AND NOT authorized",
                conclusion="security_alert",
                confidence=0.95,
                category="security",
                explanation_template="Security anomaly: threat score > 70%, unauthorized",
            ),
            SymbolicRule(
                name="behavioral_anomaly_rule",
                premise="deviation_score > 2.0 AND frequency > 10",
                conclusion="behavioral_anomaly",
                confidence=0.85,
                category="behavior",
                explanation_template="Behavioral anomaly: deviation > 2σ, high frequency",
            ),
        ]

        for rule in anomaly_rules:
            self.symbolic_layer.add_rule(rule)

    def add_fact(self, fact: str):
        """Add a fact to the knowledge base"""
        self.facts.add(fact)
        logging.info(f"Added fact: {fact}")

    def neural_inference(self, features: np.ndarray) -> float:
        """
        Perform neural inference on features.

        Args:
            features: Input features (numpy array)

        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not TORCH_AVAILABLE or self.ltn is None:
            return 0.5

        try:
            if len(features.shape) == 1:
                features = features.reshape(1, -1)

            features_tensor = torch.FloatTensor(features)

            if features_tensor.shape[1] < self.input_dim:
                padding = torch.zeros(
                    features_tensor.shape[0], self.input_dim - features_tensor.shape[1]
                )
                features_tensor = torch.cat([features_tensor, padding], dim=1)
            elif features_tensor.shape[1] > self.input_dim:
                features_tensor = features_tensor[:, : self.input_dim]

            with torch.no_grad():
                output = self.ltn.forward(features_tensor)

            return float(output.item())

        except Exception as e:
            logging.error(f"Neural inference error: {e}")
            return 0.5

    def symbolic_inference(self, query: str) -> dict[str, Any]:
        """
        Perform symbolic inference using knowledge base.

        Args:
            query: Query in logical form

        Returns:
            Inference result with explanation
        """
        try:
            if query in self.facts:
                return {
                    "result": True,
                    "confidence": 1.0,
                    "explanation": f"{query} is a known fact",
                    "method": "direct_fact",
                }

            applicable_rules = []

            for rule in self.knowledge_base:
                if rule.conclusion == query:
                    premise_satisfied = self._evaluate_premise(rule.premise)

                    if premise_satisfied:
                        applicable_rules.append(rule)

            if applicable_rules:
                best_rule = max(applicable_rules, key=lambda r: r.confidence)

                return {
                    "result": True,
                    "confidence": float(best_rule.confidence),
                    "explanation": f"{query} derived from: {best_rule.premise}",
                    "method": "rule_based",
                    "rule": best_rule,
                }

            return {
                "result": False,
                "confidence": 0.0,
                "explanation": f"Cannot derive {query} from knowledge base",
                "method": "unknown",
            }

        except Exception as e:
            logging.error(f"Symbolic inference error: {e}")
            return {"result": False, "error": str(e)}

    def _evaluate_premise(self, premise: str) -> bool:
        """
        Evaluate if a premise is satisfied by current facts.

        Args:
            premise: Logical premise string

        Returns:
            True if premise is satisfied
        """
        try:
            premise_lower = premise.lower()

            if " and " in premise_lower:
                parts = premise_lower.split(" and ")
                return all(part.strip() in self.facts for part in parts)

            if " or " in premise_lower:
                parts = premise_lower.split(" or ")
                return any(part.strip() in self.facts for part in parts)

            if premise_lower.startswith("not "):
                fact = premise_lower[4:].strip()
                return fact not in self.facts

            return premise.strip() in self.facts

        except Exception as e:
            logging.error(f"Premise evaluation error: {e}")
            return False

    # Feature vector layout constants for robust indexing
    _NEURAL_CONF_OFFSET = -2  # Neural confidence is second-to-last
    _OMNI_LOGIC_OFFSET = -1  # Omni logic scalar is last

    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extract neurosymbolic features for anomaly detection.

        Feature vector layout: [data_features(10), neural_conf(1), omni_logic(1)]
        Total: 12 features per sample

        Use class constants _NEURAL_CONF_OFFSET and _OMNI_LOGIC_OFFSET for
        robust indexing that doesn't break if feature layout changes.
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        batch_size = data.shape[0]
        features = []

        for i in range(batch_size):
            neural_conf = self.neural_inference(data[i])

            feature_vec = np.concatenate(
                [
                    (
                        data[i][:10]
                        if data.shape[1] >= 10
                        else np.pad(data[i], (0, 10 - data.shape[1]))
                    ),
                    [neural_conf],
                    [self.omni_scalars["omni_logic"]],
                ]
            )
            features.append(feature_vec)

        return np.array(features).astype(np.float32)

    def predict(
        self,
        data: np.ndarray,
        context: dict[str, Any] | None = None,
        mode: ReasoningMode | None = None,
    ) -> dict[str, Any]:
        """Predict anomalies using neurosymbolic reasoning.

        Args:
            data: Input data array
            context: Optional context for symbolic reasoning
            mode: Override reasoning mode for this prediction

        Returns:
            Dictionary with prediction results and explanations
        """
        features = self.extract_features(data)
        # Use relative indexing for robustness - neural_conf is at _NEURAL_CONF_OFFSET (-2)
        neural_scores = features[:, self._NEURAL_CONF_OFFSET]

        reasoning_mode = mode or self.reasoning_mode
        context = context or {}

        if reasoning_mode == ReasoningMode.NEURAL_ONLY:
            anomaly_scores = 1.0 - neural_scores
            return {
                "anomaly_scores": anomaly_scores.astype(np.float32),
                "neural_confidence": neural_scores.astype(np.float32),
                "symbolic_conclusions": {},
                "reasoning_mode": "neural_only",
            }

        elif reasoning_mode == ReasoningMode.SYMBOLIC_ONLY:
            # Pure symbolic reasoning
            symbolic_results = []
            for _ in range(len(neural_scores)):
                result = self.symbolic_layer.reason(0.5, context)
                symbolic_results.append(result)

            anomaly_scores = np.array([1.0 if r.result else 0.0 for r in symbolic_results]).astype(
                np.float32
            )

            return {
                "anomaly_scores": anomaly_scores,
                "neural_confidence": neural_scores.astype(np.float32),
                "symbolic_conclusions": {
                    "rules_fired": [r.rules_fired for r in symbolic_results],
                    "explanations": [r.explanation for r in symbolic_results],
                },
                "reasoning_mode": "symbolic_only",
            }

        else:  # HYBRID or NEURO_SYMBOLIC_ATTENTION
            # Hybrid neuro-symbolic reasoning
            hybrid_results = []
            for i in range(len(neural_scores)):
                neural_score = float(neural_scores[i])
                result = self.symbolic_layer.reason(neural_score, context)
                hybrid_results.append(result)

            anomaly_scores = np.array([r.confidence for r in hybrid_results]).astype(np.float32)

            return {
                "anomaly_scores": anomaly_scores,
                "neural_confidence": neural_scores.astype(np.float32),
                "symbolic_conclusions": {
                    "rules_fired": [r.rules_fired for r in hybrid_results],
                    "explanations": [r.explanation for r in hybrid_results],
                    "neural_contributions": [r.neural_contribution for r in hybrid_results],
                    "symbolic_contributions": [r.symbolic_contribution for r in hybrid_results],
                },
                "reasoning_mode": "hybrid",
            }

    def hybrid_inference(
        self,
        data: np.ndarray,
        context: dict[str, Any],
    ) -> ReasoningResult:
        """Perform hybrid neuro-symbolic inference on a single sample.

        Args:
            data: Input feature vector
            context: Context dictionary for symbolic reasoning

        Returns:
            ReasoningResult with combined inference
        """
        neural_score = self.neural_inference(data)
        return self.symbolic_layer.reason(neural_score, context)

    def explain(self, reasoning_result: ReasoningResult) -> str:
        """Generate human-readable explanation of reasoning result.

        Args:
            reasoning_result: Result from hybrid_inference

        Returns:
            Formatted explanation string
        """
        return self.symbolic_layer.explain_decision(reasoning_result)

    def add_rule(self, rule: SymbolicRule) -> None:
        """Add a custom rule to both knowledge base and symbolic layer.

        Args:
            rule: SymbolicRule to add
        """
        self.knowledge_base.append(rule)
        self.symbolic_layer.add_rule(rule)

    def get_statistics(self) -> dict[str, Any]:
        """Get engine statistics.

        Returns:
            Dictionary with engine statistics
        """
        return {
            "knowledge_base_size": len(self.knowledge_base),
            "symbolic_rules": len(self.symbolic_layer.rules),
            "facts_count": len(self.facts),
            "reasoning_mode": self.reasoning_mode.value,
            "ltn_available": self.ltn is not None,
            "input_dim": self.input_dim,
            "ethical_alignment_enabled": self.ethical_alignment_engine is not None,
        }

    def verify_ethical_alignment(
        self,
        data: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Verify ethical alignment of reasoning results.

        Uses the optional PercipienceEngine for comprehensive ethical verification
        including survivor-first principles, bias audits, and US compliance.

        Args:
            data: Input data for ethical analysis
            context: Optional context with ethical scores and performance metrics

        Returns:
            Dictionary with ethical alignment verification results
        """
        context = context or {}

        if self.ethical_alignment_engine is None:
            # Return basic ethical verification using symbolic rules
            ethical_facts = {
                fact
                for fact in self.facts
                if "ethical" in fact.lower() or "consent" in fact.lower()
            }
            return {
                "ethical_alignment_enabled": False,
                "basic_verification": True,
                "ethical_facts_count": len(ethical_facts),
                "ethical_rules_count": len(
                    [
                        r
                        for r in self.knowledge_base
                        if "consent" in r.premise.lower() or "privacy" in r.premise.lower()
                    ]
                ),
                "omni_scalars": self.omni_scalars,
            }

        # Use PercipienceEngine for comprehensive ethical verification
        ethical_scores = context.get(
            "ethical_scores",
            {
                "compassion": 0.8,
                "evidence": 0.85,
                "justice": 0.9,
                "altruism": 0.75,
                "control": 0.8,
                "character": 0.85,
                "competence": 0.9,
                "commitment": 0.85,
            },
        )

        performance_metrics = context.get(
            "performance_metrics",
            {
                "security": 0.9,
                "recovery": 0.85,
                "decision_accuracy": 0.88,
                "planning": 0.8,
                "organization": 0.85,
            },
        )

        knowledge_indicators = context.get(
            "knowledge_indicators",
            {
                "wisdom": 0.85,
                "expertise": 0.9,
                "learning_rate": 0.8,
            },
        )

        try:
            result = self.ethical_alignment_engine.comprehensive_analysis(
                data=data,
                ethical_scores=ethical_scores,
                performance_metrics=performance_metrics,
                knowledge_indicators=knowledge_indicators,
                context=context,
            )

            # Merge omni-scalars from ethical engine with neurosymbolic scalars
            if "omni_scalars" in result:
                merged_scalars = {**self.omni_scalars, **result["omni_scalars"]}
                result["merged_omni_scalars"] = merged_scalars

            result["ethical_alignment_enabled"] = True
            return result

        except Exception as e:
            logging.warning(f"Ethical alignment verification failed: {e}")
            return {
                "ethical_alignment_enabled": True,
                "verification_error": str(e),
                "fallback_omni_scalars": self.omni_scalars,
            }


def create_neurosymbolic_engine(
    engine_type: str = "standard",
    input_dim: int = 64,
    reasoning_mode: ReasoningMode = ReasoningMode.HYBRID,
    explainability_threshold: float = 0.7,
    ethical_alignment_engine: Any | None = None,
    neural_weight: float = 0.6,
    symbolic_weight: float = 0.4,
    use_confidence_weights: bool = False,
    **kwargs: Any,
) -> NeurosymbolicEngine:
    """Factory function to create neurosymbolic engines.

    Provides a unified interface for creating either the standard NeurosymbolicEngine
    or the EnhancedNeurosymbolicEngine with advanced capabilities.

    Args:
        engine_type: Type of engine - "standard" or "enhanced"
        input_dim: Input feature dimension for LTN
        reasoning_mode: Default reasoning mode (HYBRID, NEURAL_ONLY, SYMBOLIC_ONLY)
        explainability_threshold: Minimum confidence for explanations
        ethical_alignment_engine: Optional PercipienceEngine for ethical verification
        neural_weight: Weight for neural component in hybrid reasoning (default 0.6)
        symbolic_weight: Weight for symbolic component in hybrid reasoning (default 0.4)
        use_confidence_weights: Use rule confidences for symbolic scoring
        **kwargs: Additional arguments for EnhancedNeurosymbolicEngine

    Returns:
        NeurosymbolicEngine or EnhancedNeurosymbolicEngine instance

    Raises:
        ValueError: If engine_type is not "standard" or "enhanced"

    Example:
        >>> # Standard engine
        >>> engine = create_neurosymbolic_engine("standard")
        >>>
        >>> # Enhanced engine with all capabilities
        >>> engine = create_neurosymbolic_engine(
        ...     "enhanced",
        ...     use_knowledge_graph=True,
        ...     use_meta_cognition=True,
        ...     use_causal=True,
        ... )
    """
    if engine_type == "standard":
        return NeurosymbolicEngine(
            input_dim=input_dim,
            reasoning_mode=reasoning_mode,
            explainability_threshold=explainability_threshold,
            ethical_alignment_engine=ethical_alignment_engine,
            neural_weight=neural_weight,
            symbolic_weight=symbolic_weight,
            use_confidence_weights=use_confidence_weights,
        )
    elif engine_type == "enhanced":
        # Import EnhancedNeurosymbolicEngine lazily to avoid circular imports
        from omni_anomaly_engine.models.neurosymbolic_enhanced import (
            EnhancedNeurosymbolicEngine,
        )

        # Extract enhanced-specific kwargs
        hidden_dim = kwargs.get("hidden_dim", 256)
        num_predicates = kwargs.get("num_predicates", 16)
        fuzzy_semantics = kwargs.get("fuzzy_semantics")
        use_knowledge_graph = kwargs.get("use_knowledge_graph", True)
        use_meta_cognition = kwargs.get("use_meta_cognition", True)
        use_causal = kwargs.get("use_causal", True)

        enhanced_kwargs: dict[str, Any] = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_predicates": num_predicates,
            "use_knowledge_graph": use_knowledge_graph,
            "use_meta_cognition": use_meta_cognition,
            "use_causal": use_causal,
        }

        if fuzzy_semantics is not None:
            enhanced_kwargs["fuzzy_semantics"] = fuzzy_semantics

        return EnhancedNeurosymbolicEngine(**enhanced_kwargs)
    else:
        raise ValueError(f"Unknown engine_type: {engine_type}. Must be 'standard' or 'enhanced'.")
