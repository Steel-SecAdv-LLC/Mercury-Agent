# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unified Neurosymbolic Engine - Fusion of neural networks and symbolic reasoning.

.. warning:: **Legacy / reference surface — not the canonical co-trained path.**
    The **canonical, trained** neuro-symbolic component is
    :class:`omni_mercury_engine.ml.symbolic_constraint.SymbolicConstraintModule`,
    co-trained with the fusion network inside
    :meth:`omni_mercury_engine.engine.OmniMercuryEngine.fit_fusion`
    (adaptive ``symbolic_weight`` schedule + conformal uncertainty on the
    serve path).  ``NeurosymbolicEngine`` here is a *symbolic-rules + ethical
    rules* reference surface; its neural-confidence path is a **deterministic
    statistical heuristic**, not a trained network.  The original
    ``LogicTensorNetwork`` (a never-trained ``nn.Module`` whose forward pass
    returned random-init noise) had that **random-init network removed
    (2026-06-02)**: the surface is now a **thin SCM-backed wrapper** whose
    :meth:`LogicTensorNetwork.predict` delegates to
    :meth:`SymbolicConstraintModule.predict` — a deterministic uniform-weight
    consensus when untrained, and the module's learned detector reliabilities
    when co-trained.  New code that wants genuine co-trained neuro-symbolic
    inference should use that canonical module in ``ml/symbolic_constraint.py``.

This module provides symbolic-reasoning + deterministic neural-heuristic
capabilities for Mercury Agent:
- Explainable anomaly detection (rule-based)
- Ethical constraint enforcement (symbolic rules)
- Hybrid neural-heuristic / symbolic inference

Architecture:
    1. SymbolicReasoningLayer: PyReason-inspired rule-based reasoning
    2. NeurosymbolicEngine: Unified interface for hybrid inference
       (deterministic neural heuristic + symbolic rules)

Research Sources:
    - LTN: Logic Tensor Networks (Serafini & Garcez, 2016)
    - PyReason: Temporal First-Order Logic Explainable AI (AAAI 2023)
    - Neuro-symbolic AI: https://arxiv.org/pdf/2302.13482.pdf

Example:
    Basic usage::

        from omni_mercury_engine.models.neurosymbolic import NeurosymbolicEngine
        import numpy as np

        engine = NeurosymbolicEngine(input_dim=64)
        engine.add_fact("missing_person")
        engine.add_fact("child")

        # Neural inference
        features = np.random.default_rng().standard_normal((10, 64))
        result = engine.predict(features)

        # Symbolic inference
        symbolic = engine.symbolic_inference("priority_high")
        print(f"Derived: {symbolic['result']}, Explanation: {symbolic['explanation']}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from omni_mercury_engine.utils.constants import OmniCodes

# Omni-Code: Omni-Benevolent Stone (ethical foundation and humanitarian alignment)
_FOUNDATION_HASH = OmniCodes.OMNI_BENEVOLENT.code

# PyTorch is no longer imported here: the only consumer was the retired
# untrained ``LogicTensorNetwork``.  ``neural_inference`` is now a pure-NumPy
# deterministic heuristic, so this module has no torch dependency.  Genuine
# trained neuro-symbolic inference lives in
# ``ml/symbolic_constraint.py::SymbolicConstraintModule``.


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

    def __post_init__(self) -> None:
        """Finalize dataclass initialization."""
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
    """Neuro-symbolic inference head backed by the canonical co-trained module.

    History: the original ``LogicTensorNetwork`` was a never-trained
    ``nn.Module`` whose random-init forward pass was fed into fusion as if it
    were a neural confidence — the dishonesty the v1.7 audit flagged. Rather
    than leave that surface retired, it is **re-wired** here to the canonical
    :class:`~omni_mercury_engine.ml.symbolic_constraint.SymbolicConstraintModule`:
    :meth:`predict` routes detector scores through that module's fuzzy-logic
    ``Consensus`` predicate — the *same* component co-trained (with real
    gradients) inside :meth:`OmniMercuryEngine.fit_fusion`.

    There is no random-init noise: an untrained module's parameters initialise
    to zero, so its consensus is a deterministic uniform-weight aggregator; a
    co-trained module (e.g. restored from a fusion checkpoint and passed as
    ``module=``) applies its learned detector reliabilities. Either way the
    output is a real, reproducible symbolic signal — never an unowned random
    number.

    Torch is imported lazily here so importing :mod:`...models.neurosymbolic`
    stays torch-free; constructing an LTN requires torch (it builds/holds the
    canonical torch module).
    """

    def __init__(self, num_detectors: int, module: Any = None) -> None:
        """Build an SCM-backed inference head.

        Args:
            num_detectors: Number of per-detector score channels ``D``.
            module: Optional pre-built / co-trained ``SymbolicConstraintModule``
                to wrap. When ``None`` a fresh (deterministic, untrained) module
                with ``num_detectors`` channels is constructed.
        """
        from omni_mercury_engine.ml.symbolic_constraint import SymbolicConstraintModule

        self.num_detectors = int(num_detectors)
        self.module = (
            module
            if module is not None
            else SymbolicConstraintModule(num_detectors=self.num_detectors)
        )
        self.module.eval()

    def predict(self, detector_scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Per-sample anomaly consensus via the canonical symbolic module.

        Args:
            detector_scores: ``(B, D)`` or ``(D,)`` per-detector scores in
                ``[0, 1]``.

        Returns:
            ``(B,)`` consensus probabilities in ``[0, 1]`` as a NumPy array.
        """
        import torch

        scores = torch.as_tensor(np.asarray(detector_scores, dtype=np.float64), dtype=torch.float32)
        if scores.ndim == 1:
            scores = scores.unsqueeze(0)
        # ``scores`` is built on CPU above; relocate it onto the wrapped module's
        # parameter device so a co-trained module that was ``.to(<accelerator>)``'d
        # does not raise a device mismatch inside the consensus kernel.
        try:
            module_device = next(self.module.parameters()).device
        except StopIteration:  # a module with no parameters -> stay on CPU
            module_device = torch.device("cpu")
        scores = scores.to(module_device)
        consensus = self.module.predict(scores)
        return np.asarray(consensus.detach().cpu().numpy(), dtype=np.float64)


class SymbolicReasoningLayer:
    """Symbolic reasoning layer for explainable AI.

    Provides PyReason-inspired temporal first-order logic reasoning
    for generating human-readable explanations of decisions.

    Based on: PyReason - Temporal First-Order Logic Explainable AI (AAAI 2023)
    https://pyreason.readthedocs.io/
    """

    def __init__(
        self,
        explainability_threshold: float = 0.7,
        temporal_logic: bool = True,
        graph_based: bool = True,
    ):
        """Initialize symbolic reasoning layer.

        Args:
            explainability_threshold: Min confidence for explanations
            temporal_logic: Enable temporal reasoning
            graph_based: Enable graph-based reasoning
        """
        self.explainability_threshold = explainability_threshold
        self.temporal_logic = temporal_logic
        self.graph_based = graph_based
        self.rules: list[SymbolicRule] = []

    def add_rule(self, rule: SymbolicRule) -> None:
        """Add a symbolic reasoning rule."""
        self.rules.append(rule)

    def reason(
        self,
        neural_score: float,
        context: dict[str, Any],
    ) -> ReasoningResult:
        """Perform hybrid neuro-symbolic reasoning.

        Combines neural network output with symbolic rule evaluation
        for explainable anomaly detection.

        Args:
            neural_score: Output from neural detector (0.0 to 1.0)
            context: Context dict with code metrics and features

        Returns:
            ReasoningResult with explanation
        """
        rules_fired: list[str] = []
        explanations: list[str] = []

        for rule in self.rules:
            satisfied = self._evaluate_rule(rule, context)
            if satisfied and rule.confidence >= self.explainability_threshold:
                rules_fired.append(rule.name)
                explanations.append(rule.generate_explanation())

        symbolic_score = len(rules_fired) / len(self.rules) if self.rules else 0.0
        symbolic_confidence = symbolic_score if rules_fired else 0.0

        # Adaptive neuro-symbolic blend (single canonical implementation),
        # replacing the former hardcoded 0.6/0.4 static mix. Weighting adapts
        # to how decisive each branch is. ``neural_score`` is expected to be the
        # trained fusion network's calibrated probability when available.
        from omni_mercury_engine.cognitive.neurosymbolic_fusion import (
            adaptive_neurosymbolic_fuse,
        )

        combined_confidence, _ = adaptive_neurosymbolic_fuse(
            neural_score,
            symbolic_score,
            symbolic_confidence=symbolic_confidence,
        )

        is_anomaly = combined_confidence > 0.5

        explanation = (
            "; ".join(explanations) if explanations else "Based on neural network analysis only."
        )

        return ReasoningResult(
            result=is_anomaly,
            confidence=combined_confidence,
            explanation=explanation,
            method="hybrid",
            rules_fired=rules_fired,
            neural_contribution=neural_score,
            symbolic_contribution=symbolic_score,
        )

    def _evaluate_rule(self, rule: SymbolicRule, context: dict[str, Any]) -> bool:
        """Evaluate if a rule's premise is satisfied by context."""
        # Simple evaluation - check if premise keys exist in context
        premise_parts = rule.premise.lower().replace(" and ", ",").split(",")
        for part in premise_parts:
            part = part.strip()
            if part.startswith("not "):
                key = part[4:].strip()
                if context.get(key):
                    return False
            # Check for comparisons
            elif ">" in part:
                key, val = part.split(">")
                if context.get(key.strip(), 0) <= float(val.strip()):
                    return False
            elif "<" in part:
                key, val = part.split("<")
                if context.get(key.strip(), 0) >= float(val.strip()):
                    return False
            elif not context.get(part):
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
    """Unified Neurosymbolic reasoning engine combining LTN with symbolic logic.

    Provides:
    - Neural inference via Logic Tensor Networks
    - Symbolic inference via knowledge base and rules
    - Hybrid reasoning combining both approaches
    - Explainable AI with human-readable explanations
    - Ethical constraint enforcement

    This is the primary neurosymbolic component for Mercury Agent.
    """

    def __init__(
        self,
        input_dim: int = 64,
        reasoning_mode: ReasoningMode = ReasoningMode.HYBRID,
        explainability_threshold: float = 0.7,
    ):
        """Initialize Neurosymbolic Engine.

        Args:
            input_dim: Input feature dimension for LTN
            reasoning_mode: Default reasoning mode (HYBRID, NEURAL_ONLY, SYMBOLIC_ONLY)
            explainability_threshold: Minimum confidence for explanations
        """
        self.input_dim = input_dim
        self.reasoning_mode = reasoning_mode
        self.golden_ratio = 0.618
        self.quantum_factor = 1.2

        # Neuro-symbolic inference head, wired to the canonical co-trained
        # SymbolicConstraintModule (no random-init network). Built eagerly when
        # torch is importable; ``None`` (transparently reported by get_statistics) in
        # a torch-free environment, since the canonical module is a torch module.
        self.ltn: LogicTensorNetwork | None
        try:
            self.ltn = LogicTensorNetwork(num_detectors=input_dim)
        except ImportError:
            self.ltn = None

        self.knowledge_base: list[SymbolicRule] = []
        self.facts: set[str] = set()

        # Symbolic reasoning layer
        self.symbolic_layer = SymbolicReasoningLayer(
            explainability_threshold=explainability_threshold,
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

    def _initialize_ethical_rules(self) -> None:
        """Initialize fundamental ethical rules."""
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

    def _initialize_anomaly_rules(self) -> None:
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

    def add_fact(self, fact: str) -> None:
        """Add a fact to the knowledge base."""
        self.facts.add(fact)
        logging.info(f"Added fact: {fact}")

    def neural_inference(self, features: np.ndarray[Any, Any]) -> float:
        """Deterministic statistical anomaly-confidence for the legacy path.

        .. note:: **Distinct from the trained consensus path (2026-06-02).**
            This is a deterministic, reproducible statistic of *raw features* —
            a robust standardized-dispersion signal through a logistic (bounded,
            feature-responsive) — **not** a trained signal, and it is transparently
            labelled as such. It is NOT the retired random-init network: the
            ``LogicTensorNetwork`` surface is now re-wired to the canonical
            :class:`omni_mercury_engine.ml.symbolic_constraint.SymbolicConstraintModule`
            (co-trained in ``OmniMercuryEngine.fit_fusion``); use
            ``self.ltn.predict(detector_scores)`` for the trained-capable
            neuro-symbolic *consensus* over per-detector scores. This heuristic
            remains for the legacy raw-feature path where no detector scores
            exist.

        Args:
            features: Input features (numpy array, 1-D row or 2-D batch).

        Returns:
            Confidence score in ``[0.0, 1.0]`` (deterministic for a given input).
        """
        arr = np.asarray(features, dtype=np.float64).reshape(-1)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return 0.5

        # Robust dispersion: mean absolute deviation normalised by a robust
        # scale (MAD, falling back to std).  Higher relative spread ⇒ higher
        # anomaly confidence.  Deterministic and bounded via a logistic.
        centre = float(np.median(arr))
        mad = float(np.median(np.abs(arr - centre)))
        scale = mad if mad > 1e-9 else float(np.std(arr) + 1e-9)
        dispersion = float(np.mean(np.abs(arr - float(np.mean(arr)))) / scale)
        return float(1.0 / (1.0 + np.exp(-(dispersion - 1.0))))

    def symbolic_inference(self, query: str) -> dict[str, Any]:
        """Perform symbolic inference using knowledge base.

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
        """Evaluate if a premise is satisfied by current facts.

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

    def extract_features(self, data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Extract neurosymbolic features for anomaly detection."""
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
        data: np.ndarray[Any, Any],
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
        neural_scores = features[:, 10]

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
        data: np.ndarray[Any, Any],
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
            # The LogicTensorNetwork is wired (2026-06-02) to the canonical
            # co-trained SymbolicConstraintModule (no random-init network):
            # ``ltn.predict`` routes detector scores through its fuzzy-logic
            # consensus. Available whenever torch is importable. The legacy raw-
            # feature ``neural_inference`` remains a deterministic heuristic (a
            # per-feature dispersion statistic, distinct from detector consensus).
            "ltn_available": self.ltn is not None,
            "ltn_backend": "symbolic_constraint_module" if self.ltn is not None else None,
            "neural_inference_mode": "deterministic_heuristic",
            "input_dim": self.input_dim,
        }
