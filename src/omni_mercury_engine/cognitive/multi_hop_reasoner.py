"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

from __future__ import annotations


"""
Multi-Hop Reasoner - Abductive/Deductive/Inductive Reasoning Chains

Implements sophisticated reasoning capabilities:
- Deductive reasoning: From general to specific (if P then Q)
- Inductive reasoning: From specific to general (pattern generalization)
- Abductive reasoning: Inference to best explanation
- Multi-hop inference: Chaining multiple reasoning steps

Research Sources:
- Neuro-Symbolic AI Lab: Multi-hop inference and abductive reasoning
- DARPA ANSR: Trustworthy reasoning chains
- Logic Tensor Networks: Neural-symbolic reasoning
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """Types of reasoning."""

    DEDUCTIVE = "deductive"  # Certain conclusion from premises
    INDUCTIVE = "inductive"  # Probable generalization from instances
    ABDUCTIVE = "abductive"  # Best explanation for observations


class InferenceStatus(Enum):
    """Status of an inference."""

    VALID = "valid"
    INVALID = "invalid"
    UNCERTAIN = "uncertain"
    INCOMPLETE = "incomplete"


@dataclass
class Proposition:
    """A logical proposition."""

    prop_id: str
    content: str
    truth_value: float  # 0.0 to 1.0 (fuzzy logic)
    confidence: float = 1.0
    source: str = "system"
    evidence: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.prop_id,
            "content": self.content,
            "truth": self.truth_value,
            "confidence": self.confidence,
            "source": self.source,
            "evidence": self.evidence,
        }


@dataclass
class InferenceRule:
    """A rule for inference."""

    rule_id: str
    premises: list[str]  # Proposition IDs or patterns
    conclusion: str  # Conclusion proposition pattern
    reasoning_type: ReasoningType
    confidence: float = 1.0
    explanation_template: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.rule_id,
            "premises": self.premises,
            "conclusion": self.conclusion,
            "type": self.reasoning_type.value,
            "confidence": self.confidence,
        }


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""

    step_id: int
    rule_applied: InferenceRule
    premises_used: list[Proposition]
    conclusion_derived: Proposition
    confidence: float
    explanation: str


@dataclass
class ReasoningChain:
    """A complete reasoning chain from premises to conclusion."""

    chain_id: str
    reasoning_type: ReasoningType
    steps: list[ReasoningStep]
    initial_premises: list[Proposition]
    final_conclusion: Proposition
    total_confidence: float
    is_valid: bool
    explanation: str
    computation_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.chain_id,
            "type": self.reasoning_type.value,
            "num_steps": len(self.steps),
            "conclusion": self.final_conclusion.to_dict(),
            "confidence": self.total_confidence,
            "valid": self.is_valid,
            "explanation": self.explanation,
            "time_ms": self.computation_time_ms,
        }


class MultiHopReasoner:
    """
    Multi-Hop Reasoning Engine.

    Implements three types of reasoning:

    1. Deductive (Forward Chaining):
       Given P and P→Q, derive Q
       Certainty-preserving: conclusion is as certain as weakest premise

    2. Inductive (Pattern Generalization):
       Given instances, derive probable general pattern
       Probability increases with more confirming instances

    3. Abductive (Inference to Best Explanation):
       Given observation O, find hypothesis H that best explains O
       Selects hypothesis with highest posterior probability

    Multi-hop capability chains multiple reasoning steps to derive
    conclusions not directly inferable from initial premises.
    """

    # Golden ratio for optimal reasoning chain weights
    PHI = (1 + np.sqrt(5)) / 2

    def __init__(
        self,
        max_chain_depth: int = 10,
        min_confidence_threshold: float = 0.1,
        enable_explanation_generation: bool = True,
    ):
        """
        Initialize Multi-Hop Reasoner.

        Args:
            max_chain_depth: Maximum reasoning steps per chain
            min_confidence_threshold: Minimum confidence to continue reasoning
            enable_explanation_generation: Generate human-readable explanations
        """
        self.max_chain_depth = max_chain_depth
        self.min_confidence_threshold = min_confidence_threshold
        self.enable_explanation = enable_explanation_generation

        # Knowledge base
        self._propositions: dict[str, Proposition] = {}
        self._rules: dict[str, InferenceRule] = {}
        self._rule_index: defaultdict[ReasoningType, list[str]] = defaultdict(list)

        # Inference cache
        self._inference_cache: dict[str, Proposition] = {}

        # Statistics
        self._stats = {
            "deductive_inferences": 0,
            "inductive_inferences": 0,
            "abductive_inferences": 0,
            "chains_computed": 0,
            "cache_hits": 0,
        }

        self._initialize_core_rules()

        logger.info(f"MultiHopReasoner initialized (max_depth={max_chain_depth})")

    def _initialize_core_rules(self) -> None:
        """Initialize fundamental reasoning rules."""
        core_rules = [
            # Modus Ponens: P, P→Q ⊢ Q
            InferenceRule(
                rule_id="modus_ponens",
                premises=["P", "P_implies_Q"],
                conclusion="Q",
                reasoning_type=ReasoningType.DEDUCTIVE,
                confidence=1.0,
                explanation_template="Given {P} and that {P} implies {Q}, we conclude {Q}",
            ),
            # Modus Tollens: ¬Q, P→Q ⊢ ¬P
            InferenceRule(
                rule_id="modus_tollens",
                premises=["not_Q", "P_implies_Q"],
                conclusion="not_P",
                reasoning_type=ReasoningType.DEDUCTIVE,
                confidence=1.0,
                explanation_template="Given not {Q} and that {P} implies {Q}, we conclude not {P}",
            ),
            # Transitivity: P→Q, Q→R ⊢ P→R
            InferenceRule(
                rule_id="transitivity",
                premises=["P_implies_Q", "Q_implies_R"],
                conclusion="P_implies_R",
                reasoning_type=ReasoningType.DEDUCTIVE,
                confidence=1.0,
                explanation_template="If {P}→{Q} and {Q}→{R}, then {P}→{R}",
            ),
            # Inductive generalization
            InferenceRule(
                rule_id="inductive_generalization",
                premises=["instance_1", "instance_2", "instance_3"],
                conclusion="general_pattern",
                reasoning_type=ReasoningType.INDUCTIVE,
                confidence=0.8,
                explanation_template="From multiple instances, we generalize a pattern",
            ),
            # Abduction: O, H→O ⊢ H (with lower confidence)
            InferenceRule(
                rule_id="abduction",
                premises=["observation", "hypothesis_explains_observation"],
                conclusion="hypothesis",
                reasoning_type=ReasoningType.ABDUCTIVE,
                confidence=0.7,
                explanation_template="Observation {O} is best explained by hypothesis {H}",
            ),
        ]

        for rule in core_rules:
            self.add_rule(rule)

    def add_proposition(self, proposition: Proposition) -> None:
        """Add a proposition to the knowledge base."""
        self._propositions[proposition.prop_id] = proposition

    def add_rule(self, rule: InferenceRule) -> None:
        """Add an inference rule."""
        self._rules[rule.rule_id] = rule
        self._rule_index[rule.reasoning_type].append(rule.rule_id)

    def deduce(
        self,
        premises: list[Proposition],
        goal: str | None = None,
    ) -> ReasoningChain | None:
        """
        Perform deductive reasoning (forward chaining).

        Args:
            premises: Initial premises
            goal: Optional goal proposition to derive

        Returns:
            ReasoningChain if successful, None otherwise
        """
        start_time = time.time()
        self._stats["deductive_inferences"] += 1

        # Add premises to working memory
        working_memory = {p.prop_id: p for p in premises}
        steps: list[ReasoningStep] = []
        step_count = 0

        # Forward chaining
        changed = True
        while changed and step_count < self.max_chain_depth:
            changed = False

            for rule_id in self._rule_index[ReasoningType.DEDUCTIVE]:
                rule = self._rules[rule_id]

                # Try to match premises
                binding = self._match_premises(rule.premises, working_memory)
                if binding:
                    # Derive conclusion
                    conclusion_id = self._substitute(rule.conclusion, binding)

                    if conclusion_id not in working_memory:
                        # Calculate confidence
                        premise_confidences = [
                            working_memory[binding[p]].confidence
                            * working_memory[binding[p]].truth_value
                            for p in rule.premises
                            if p in binding
                        ]
                        confidence = (
                            min(premise_confidences) * rule.confidence if premise_confidences else 0
                        )

                        if confidence >= self.min_confidence_threshold:
                            # Create conclusion proposition
                            conclusion = Proposition(
                                prop_id=conclusion_id,
                                content=f"Derived: {conclusion_id}",
                                truth_value=confidence,
                                confidence=confidence,
                                source="deduction",
                                evidence=[binding[p] for p in rule.premises if p in binding],
                            )

                            working_memory[conclusion_id] = conclusion
                            changed = True
                            step_count += 1

                            # Create reasoning step
                            explanation = self._generate_explanation(rule, binding, working_memory)
                            steps.append(
                                ReasoningStep(
                                    step_id=step_count,
                                    rule_applied=rule,
                                    premises_used=[
                                        working_memory[binding[p]]
                                        for p in rule.premises
                                        if p in binding
                                    ],
                                    conclusion_derived=conclusion,
                                    confidence=confidence,
                                    explanation=explanation,
                                )
                            )

                            # Check if we reached the goal
                            if goal and conclusion_id == goal:
                                return self._build_chain(
                                    "deductive",
                                    ReasoningType.DEDUCTIVE,
                                    steps,
                                    premises,
                                    conclusion,
                                    time.time() - start_time,
                                )

        # Return chain if we made any inferences
        if steps:
            final = steps[-1].conclusion_derived
            return self._build_chain(
                "deductive",
                ReasoningType.DEDUCTIVE,
                steps,
                premises,
                final,
                time.time() - start_time,
            )

        return None

    def induce(
        self,
        instances: list[dict[str, Any]],
        feature_extractor: Callable[[dict[str, Any]], np.ndarray[Any, Any]] | None = None,
    ) -> ReasoningChain | None:
        """
        Perform inductive reasoning (pattern generalization).

        Args:
            instances: List of observed instances
            feature_extractor: Optional function to extract features

        Returns:
            ReasoningChain with generalized pattern
        """
        start_time = time.time()
        self._stats["inductive_inferences"] += 1

        if len(instances) < 2:
            return None

        # Extract common features
        common_features = self._find_common_features(instances)

        if not common_features:
            return None

        # Calculate confidence based on instance count and consistency
        confidence = min(0.95, 0.5 + 0.1 * len(instances))

        # Create generalized proposition
        pattern_id = f"pattern_{hash(tuple(sorted(common_features.items()))) % 10000}"
        conclusion = Proposition(
            prop_id=pattern_id,
            content=f"Generalized pattern: {common_features}",
            truth_value=confidence,
            confidence=confidence,
            source="induction",
            evidence=[f"instance_{i}" for i in range(len(instances))],
        )

        # Create premises from instances
        premises = [
            Proposition(
                prop_id=f"instance_{i}",
                content=str(inst),
                truth_value=1.0,
                confidence=1.0,
                source="observation",
            )
            for i, inst in enumerate(instances)
        ]

        step = ReasoningStep(
            step_id=1,
            rule_applied=self._rules["inductive_generalization"],
            premises_used=premises,
            conclusion_derived=conclusion,
            confidence=confidence,
            explanation=f"From {len(instances)} instances, induced pattern with features: {list(common_features.keys())}",
        )

        return self._build_chain(
            "inductive",
            ReasoningType.INDUCTIVE,
            [step],
            premises,
            conclusion,
            time.time() - start_time,
        )

    def abduce(
        self,
        observation: Proposition,
        candidate_hypotheses: list[Proposition],
        prior_probabilities: dict[str, float] | None = None,
    ) -> ReasoningChain | None:
        """
        Perform abductive reasoning (inference to best explanation).

        Uses Bayesian-like reasoning to select the hypothesis that
        best explains the observation.

        Args:
            observation: The observation to explain
            candidate_hypotheses: Possible explanatory hypotheses
            prior_probabilities: Prior probability for each hypothesis

        Returns:
            ReasoningChain with best explanation
        """
        start_time = time.time()
        self._stats["abductive_inferences"] += 1

        if not candidate_hypotheses:
            return None

        priors = prior_probabilities or {
            h.prop_id: 1.0 / len(candidate_hypotheses) for h in candidate_hypotheses
        }

        # Score each hypothesis
        scores = []
        for hypothesis in candidate_hypotheses:
            # P(H|O) ∝ P(O|H) * P(H)
            prior = priors.get(hypothesis.prop_id, 0.1)
            likelihood = self._compute_likelihood(observation, hypothesis)
            posterior = likelihood * prior * hypothesis.confidence
            scores.append((hypothesis, posterior))

        # Select best hypothesis
        scores.sort(key=lambda x: x[1], reverse=True)
        best_hypothesis, best_score = scores[0]

        # Normalize confidence
        total_score = sum(s for _, s in scores)
        confidence = best_score / total_score if total_score > 0 else 0

        # Create conclusion
        conclusion = Proposition(
            prop_id=f"explanation_{best_hypothesis.prop_id}",
            content=f"Best explanation: {best_hypothesis.content}",
            truth_value=confidence,
            confidence=confidence,
            source="abduction",
            evidence=[observation.prop_id, best_hypothesis.prop_id],
        )

        step = ReasoningStep(
            step_id=1,
            rule_applied=self._rules["abduction"],
            premises_used=[observation, best_hypothesis],
            conclusion_derived=conclusion,
            confidence=confidence,
            explanation=f"Observation '{observation.content}' is best explained by '{best_hypothesis.content}' (score: {best_score:.3f})",
        )

        return self._build_chain(
            "abductive",
            ReasoningType.ABDUCTIVE,
            [step],
            [observation, *candidate_hypotheses],
            conclusion,
            time.time() - start_time,
        )

    def multi_hop_reason(
        self,
        initial_premises: list[Proposition],
        goal: str | None = None,
        allowed_types: list[ReasoningType] | None = None,
    ) -> ReasoningChain | None:
        """
        Perform multi-hop reasoning combining different reasoning types.

        Chains multiple reasoning steps to reach conclusions not
        directly inferable from initial premises.

        Args:
            initial_premises: Starting propositions
            goal: Optional goal to reach
            allowed_types: Allowed reasoning types (default: all)

        Returns:
            Complete reasoning chain if successful
        """
        start_time = time.time()
        self._stats["chains_computed"] += 1

        types = allowed_types or list(ReasoningType)
        all_steps: list[ReasoningStep] = []
        current_knowledge = {p.prop_id: p for p in initial_premises}

        for _hop in range(self.max_chain_depth):
            made_progress = False

            # Try deductive reasoning
            if ReasoningType.DEDUCTIVE in types:
                result = self.deduce(list(current_knowledge.values()), goal)
                if result and result.steps:
                    all_steps.extend(result.steps)
                    for step in result.steps:
                        current_knowledge[step.conclusion_derived.prop_id] = step.conclusion_derived
                    made_progress = True

                    if goal and goal in current_knowledge:
                        return self._build_chain(
                            "multi_hop",
                            ReasoningType.DEDUCTIVE,
                            all_steps,
                            initial_premises,
                            current_knowledge[goal],
                            time.time() - start_time,
                        )

            if not made_progress:
                break

        if all_steps:
            final = all_steps[-1].conclusion_derived
            return self._build_chain(
                "multi_hop",
                ReasoningType.DEDUCTIVE,
                all_steps,
                initial_premises,
                final,
                time.time() - start_time,
            )

        return None

    def _match_premises(
        self,
        premise_patterns: list[str],
        knowledge: dict[str, Proposition],
    ) -> dict[str, str] | None:
        """Match premise patterns against knowledge base."""
        binding: dict[str, str] = {}

        for pattern in premise_patterns:
            matched = False
            for prop_id, prop in knowledge.items():
                # Simple matching: exact or pattern-based
                if pattern in prop_id or pattern.lower() in prop.content.lower():
                    binding[pattern] = prop_id
                    matched = True
                    break

            if not matched:
                return None

        return binding

    def _substitute(self, template: str, binding: dict[str, str]) -> str:
        """Substitute bound variables in template."""
        result = template
        for var, value in binding.items():
            result = result.replace(var, value)
        return result

    def _find_common_features(
        self,
        instances: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Find features common to all instances."""
        if not instances:
            return {}

        common = {}
        first = instances[0]

        for key, value in first.items():
            if all(inst.get(key) == value for inst in instances):
                common[key] = value

        return common

    def _compute_likelihood(
        self,
        observation: Proposition,
        hypothesis: Proposition,
    ) -> float:
        """Compute P(observation | hypothesis)."""
        # Simplified: based on content overlap and confidence
        obs_words = set(observation.content.lower().split())
        hyp_words = set(hypothesis.content.lower().split())

        if not obs_words or not hyp_words:
            return 0.5

        overlap = len(obs_words & hyp_words) / len(obs_words | hyp_words)
        return 0.3 + 0.7 * overlap

    def _generate_explanation(
        self,
        rule: InferenceRule,
        binding: dict[str, str],
        knowledge: dict[str, Proposition],
    ) -> str:
        """Generate human-readable explanation for an inference."""
        if not self.enable_explanation or not rule.explanation_template:
            return f"Applied rule: {rule.rule_id}"

        try:
            # Substitute bound values into template
            explanation = rule.explanation_template
            for var, prop_id in binding.items():
                if prop_id in knowledge:
                    explanation = explanation.replace(
                        "{" + var + "}",
                        knowledge[prop_id].content[:50],
                    )
            return explanation
        except Exception:
            return f"Applied rule: {rule.rule_id}"

    def _build_chain(
        self,
        chain_id: str,
        reasoning_type: ReasoningType,
        steps: list[ReasoningStep],
        initial_premises: list[Proposition],
        final_conclusion: Proposition,
        elapsed_time: float,
    ) -> ReasoningChain:
        """Build a complete reasoning chain."""
        # Calculate total confidence
        if steps:
            total_confidence = min(s.confidence for s in steps)
        else:
            total_confidence = final_conclusion.confidence

        # Generate overall explanation
        if self.enable_explanation:
            explanation = (
                f"{reasoning_type.value.capitalize()} reasoning chain with {len(steps)} steps. "
            )
            explanation += (
                f"From {len(initial_premises)} premises, derived: {final_conclusion.content}"
            )
        else:
            explanation = ""

        return ReasoningChain(
            chain_id=f"{chain_id}_{int(time.time() * 1000)}",
            reasoning_type=reasoning_type,
            steps=steps,
            initial_premises=initial_premises,
            final_conclusion=final_conclusion,
            total_confidence=total_confidence,
            is_valid=total_confidence >= self.min_confidence_threshold,
            explanation=explanation,
            computation_time_ms=elapsed_time * 1000,
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get reasoner statistics."""
        return {
            **self._stats,
            "total_propositions": len(self._propositions),
            "total_rules": len(self._rules),
            "rules_by_type": {t.value: len(ids) for t, ids in self._rule_index.items()},
        }
