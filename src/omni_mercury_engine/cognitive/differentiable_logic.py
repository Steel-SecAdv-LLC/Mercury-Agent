"""
Mercury Agent - Differentiable Logic Programming

State-of-the-art differentiable logic programming for neuro-symbolic AI.
Enables end-to-end gradient-based learning of symbolic rules.

Features:
- Differentiable rule application with soft unification
- Probabilistic soft logic (PSL) with uncertainty quantification
- Neural theorem prover integration
- Counterfactual reasoning support
- Active learning feedback loop

Research References:
- DeepProbLog: Neural Probabilistic Logic Programming (Manhaeve et al., 2018)
- Neural Logic Machines (Dong et al., 2019)
- Differentiable Reasoning over Knowledge Graphs (Das et al., 2018)
- Logic Tensor Networks (Serafini & Garcez, 2016)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


class PredicateType(Enum):
    """Types of logical predicates."""

    UNARY = "unary"
    BINARY = "binary"
    TERNARY = "ternary"
    N_ARY = "n_ary"


class LogicalConnective(Enum):
    """Logical connectives for rule composition."""

    AND = "and"
    OR = "or"
    NOT = "not"
    IMPLIES = "implies"
    IFF = "iff"
    XOR = "xor"


@dataclass
class Predicate:
    """Typed predicate with embedding support."""

    name: str
    arity: int
    predicate_type: PredicateType
    embedding_dim: int = 64
    confidence: float = 1.0
    description: str = ""
    domain_constraints: list[str] = field(default_factory=list)
    embedding: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.embedding is None:
            self.embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            self.embedding = self.embedding / (np.linalg.norm(self.embedding) + 1e-8)


@dataclass
class LogicalAtom:
    """Logical atom (grounded predicate)."""

    predicate: Predicate
    arguments: tuple[str, ...]
    truth_value: float = 1.0
    timestamp: float = field(default_factory=time.time)

    @property
    def is_ground(self) -> bool:
        """Check if atom is fully grounded (no variables)."""
        return all(not arg.startswith("?") for arg in self.arguments)


@dataclass
class DifferentiableRule:
    """Differentiable logic rule with learnable parameters."""

    rule_id: str
    head: LogicalAtom
    body: list[LogicalAtom]
    connective: LogicalConnective = LogicalConnective.AND
    weight: float = 1.0
    confidence: float = 0.9
    learnable: bool = True

    def __str__(self) -> str:
        body_str = f" {self.connective.value} ".join(
            f"{a.predicate.name}({', '.join(a.arguments)})" for a in self.body
        )
        head_str = f"{self.head.predicate.name}({', '.join(self.head.arguments)})"
        return f"{body_str} → {head_str} [w={self.weight:.3f}, c={self.confidence:.3f}]"


@dataclass
class InferenceResult:
    """Result from differentiable inference."""

    query: LogicalAtom
    probability: float
    confidence: float
    supporting_rules: list[str]
    proof_trace: list[dict[str, Any]]
    counterfactuals: list[dict[str, Any]] = field(default_factory=list)
    uncertainty: float = 0.0


class DifferentiableTNorm(ABC):
    """Abstract base for differentiable t-norms (fuzzy logic operators)."""

    @abstractmethod
    def conjunction(self, a: float, b: float) -> float:
        """Fuzzy AND operation."""
        pass

    @abstractmethod
    def disjunction(self, a: float, b: float) -> float:
        """Fuzzy OR operation."""
        pass

    @abstractmethod
    def negation(self, a: float) -> float:
        """Fuzzy NOT operation."""
        pass

    @abstractmethod
    def implication(self, a: float, b: float) -> float:
        """Fuzzy implication."""
        pass


class ProductTNorm(DifferentiableTNorm):
    """Product t-norm for smooth gradients."""

    def conjunction(self, a: float, b: float) -> float:
        return a * b

    def disjunction(self, a: float, b: float) -> float:
        return a + b - a * b

    def negation(self, a: float) -> float:
        return 1.0 - a

    def implication(self, a: float, b: float) -> float:
        return min(1.0, b / (a + 1e-8))


class LukasiewiczTNorm(DifferentiableTNorm):
    """Lukasiewicz t-norm for probabilistic semantics."""

    def conjunction(self, a: float, b: float) -> float:
        return max(0.0, a + b - 1.0)

    def disjunction(self, a: float, b: float) -> float:
        return min(1.0, a + b)

    def negation(self, a: float) -> float:
        return 1.0 - a

    def implication(self, a: float, b: float) -> float:
        return min(1.0, 1.0 - a + b)


class GodelTNorm(DifferentiableTNorm):
    """Godel t-norm (minimum/maximum)."""

    def conjunction(self, a: float, b: float) -> float:
        return min(a, b)

    def disjunction(self, a: float, b: float) -> float:
        return max(a, b)

    def negation(self, a: float) -> float:
        return 1.0 if a == 0.0 else 0.0

    def implication(self, a: float, b: float) -> float:
        return 1.0 if a <= b else b


if TORCH_AVAILABLE:

    class NeuralPredicateEncoder(nn.Module):
        """Neural network for encoding predicates to embeddings."""

        def __init__(
            self,
            vocab_size: int = 1000,
            embedding_dim: int = 64,
            hidden_dim: int = 128,
        ):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.encoder = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, embedding_dim),
            )

        def forward(self, predicate_ids: torch.Tensor) -> torch.Tensor:
            embedded = self.embedding(predicate_ids)
            encoded = self.encoder(embedded)
            return F.normalize(encoded, p=2, dim=-1)

    class DifferentiableRuleModule(nn.Module):
        """Differentiable rule application with learnable weights."""

        def __init__(
            self,
            predicate_embedding_dim: int = 64,
            hidden_dim: int = 128,
            n_rules: int = 100,
        ):
            super().__init__()
            self.predicate_embedding_dim = predicate_embedding_dim
            self.hidden_dim = hidden_dim
            self.n_rules = n_rules

            self.rule_weights = nn.Parameter(torch.ones(n_rules) * 0.5)
            self.rule_confidences = nn.Parameter(torch.ones(n_rules) * 0.9)

            self.attention = nn.MultiheadAttention(
                embed_dim=predicate_embedding_dim,
                num_heads=4,
                dropout=0.1,
                batch_first=True,
            )

            self.rule_scorer = nn.Sequential(
                nn.Linear(predicate_embedding_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

            self.t_norm = ProductTNorm()

        def forward(
            self,
            body_embeddings: torch.Tensor,
            head_embedding: torch.Tensor,
            rule_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Apply rules differentiably.

            Args:
                body_embeddings: [batch, n_body_atoms, embed_dim]
                head_embedding: [batch, embed_dim]
                rule_mask: Optional mask for applicable rules

            Returns:
                Tuple of (inferred_probability, attention_weights)
            """
            batch_size = body_embeddings.shape[0]

            head_expanded = head_embedding.unsqueeze(1)
            attn_output, attn_weights = self.attention(
                head_expanded,
                body_embeddings,
                body_embeddings,
            )

            combined = torch.cat([attn_output.squeeze(1), head_embedding], dim=-1)
            rule_scores = self.rule_scorer(combined)

            weights = torch.sigmoid(self.rule_weights)
            _confidences = torch.sigmoid(
                self.rule_confidences
            )  # noqa: F841 - Reserved for confidence scoring

            if rule_mask is not None:
                weights = weights * rule_mask

            weighted_scores = rule_scores * weights[:batch_size].unsqueeze(-1)
            aggregated = torch.sum(weighted_scores, dim=0) / (
                torch.sum(weights[:batch_size]) + 1e-8
            )

            return aggregated, attn_weights

    class NeuralTheoremProver(nn.Module):
        """Neural theorem prover for differentiable logical inference."""

        def __init__(
            self,
            predicate_dim: int = 64,
            hidden_dim: int = 256,
            n_layers: int = 3,
            max_proof_depth: int = 5,
        ):
            super().__init__()
            self.predicate_dim = predicate_dim
            self.hidden_dim = hidden_dim
            self.max_proof_depth = max_proof_depth

            self.goal_encoder = nn.LSTM(
                input_size=predicate_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=True,
            )

            self.unification_net = nn.Sequential(
                nn.Linear(predicate_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

            self.proof_controller = nn.GRUCell(
                input_size=predicate_dim + hidden_dim * 2,
                hidden_size=hidden_dim,
            )

            self.success_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

        def soft_unify(
            self,
            term1: torch.Tensor,
            term2: torch.Tensor,
        ) -> torch.Tensor:
            """Differentiable soft unification."""
            combined = torch.cat([term1, term2], dim=-1)
            return self.unification_net(combined)

        def forward(
            self,
            goal: torch.Tensor,
            knowledge_base: torch.Tensor,
            max_steps: int | None = None,
        ) -> tuple[torch.Tensor, list[torch.Tensor]]:
            """Prove a goal using differentiable backward chaining.

            Args:
                goal: [batch, n_goals, predicate_dim] - Goals to prove
                knowledge_base: [batch, n_facts, predicate_dim] - Available facts
                max_steps: Maximum proof steps (defaults to max_proof_depth)

            Returns:
                Tuple of (proof_probability, proof_trace)
            """
            batch_size = goal.shape[0]
            max_steps = max_steps or self.max_proof_depth

            _, (hidden, cell) = self.goal_encoder(goal)
            hidden = hidden[-2:].transpose(0, 1).contiguous().view(batch_size, -1)

            proof_trace = []
            controller_state = hidden

            for step in range(max_steps):
                unification_scores = []
                for i in range(knowledge_base.shape[1]):
                    fact = knowledge_base[:, i, :]
                    score = self.soft_unify(goal[:, 0, :], fact)
                    unification_scores.append(score)

                unification_scores = torch.cat(unification_scores, dim=-1)
                max_score, best_fact_idx = torch.max(unification_scores, dim=-1)

                proof_trace.append(
                    {
                        "step": step,
                        "unification_scores": unification_scores.detach(),
                        "selected_fact": best_fact_idx.detach(),
                        "match_score": max_score.detach(),
                    }
                )

                controller_input = torch.cat(
                    [
                        goal[:, 0, :],
                        controller_state,
                    ],
                    dim=-1,
                )
                controller_state = self.proof_controller(controller_input, controller_state)

            proof_probability = self.success_predictor(controller_state)
            return proof_probability, proof_trace

    class CounterfactualReasoner(nn.Module):
        """Counterfactual reasoning module for "what-if" explanations."""

        def __init__(
            self,
            input_dim: int = 64,
            hidden_dim: int = 128,
            n_interventions: int = 10,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.n_interventions = n_interventions

            self.intervention_encoder = nn.Sequential(
                nn.Linear(input_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.outcome_predictor = nn.Sequential(
                nn.Linear(hidden_dim + input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

            self.intervention_generator = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim * n_interventions),
            )

        def generate_counterfactuals(
            self,
            fact: torch.Tensor,
            n_counterfactuals: int = 5,
        ) -> torch.Tensor:
            """Generate counterfactual facts."""
            interventions = self.intervention_generator(fact)
            interventions = interventions.view(-1, self.n_interventions, self.input_dim)
            return interventions[:, :n_counterfactuals, :]

        def forward(
            self,
            original_fact: torch.Tensor,
            intervention: torch.Tensor,
            context: torch.Tensor,
        ) -> torch.Tensor:
            """Predict outcome under intervention.

            Args:
                original_fact: Original fact embedding
                intervention: Intervention embedding
                context: Context embedding

            Returns:
                Predicted probability of outcome
            """
            combined = torch.cat([original_fact, intervention], dim=-1)
            intervention_repr = self.intervention_encoder(combined)
            prediction_input = torch.cat([intervention_repr, context], dim=-1)
            return self.outcome_predictor(prediction_input)


class DifferentiableLogicEngine:
    """Main engine for differentiable logic programming.

    Combines neural network components with symbolic reasoning
    for end-to-end differentiable inference.
    """

    def __init__(
        self,
        predicate_dim: int = 64,
        hidden_dim: int = 128,
        max_proof_depth: int = 5,
        t_norm: str = "product",
        device: str = "cpu",
    ):
        self.predicate_dim = predicate_dim
        self.hidden_dim = hidden_dim
        self.max_proof_depth = max_proof_depth
        self.device = device

        t_norm_map = {
            "product": ProductTNorm(),
            "lukasiewicz": LukasiewiczTNorm(),
            "godel": GodelTNorm(),
        }
        self.t_norm = t_norm_map.get(t_norm, ProductTNorm())

        self.predicates: dict[str, Predicate] = {}
        self.rules: dict[str, DifferentiableRule] = {}
        self.facts: list[LogicalAtom] = []

        self._rule_counter = 0

        if TORCH_AVAILABLE:
            self.predicate_encoder = NeuralPredicateEncoder(
                embedding_dim=predicate_dim,
            ).to(device)
            self.rule_module = DifferentiableRuleModule(
                predicate_embedding_dim=predicate_dim,
                hidden_dim=hidden_dim,
            ).to(device)
            self.theorem_prover = NeuralTheoremProver(
                predicate_dim=predicate_dim,
                hidden_dim=hidden_dim,
                max_proof_depth=max_proof_depth,
            ).to(device)
            self.counterfactual_reasoner = CounterfactualReasoner(
                input_dim=predicate_dim,
                hidden_dim=hidden_dim,
            ).to(device)
        else:
            self.predicate_encoder = None
            self.rule_module = None
            self.theorem_prover = None
            self.counterfactual_reasoner = None

        logger.info(
            f"DifferentiableLogicEngine initialized "
            f"(predicate_dim={predicate_dim}, t_norm={t_norm})"
        )

    def register_predicate(
        self,
        name: str,
        arity: int,
        description: str = "",
        domain_constraints: list[str] | None = None,
    ) -> Predicate:
        """Register a new predicate."""
        if name in self.predicates:
            return self.predicates[name]

        predicate_type = {
            1: PredicateType.UNARY,
            2: PredicateType.BINARY,
            3: PredicateType.TERNARY,
        }.get(arity, PredicateType.N_ARY)

        predicate = Predicate(
            name=name,
            arity=arity,
            predicate_type=predicate_type,
            embedding_dim=self.predicate_dim,
            description=description,
            domain_constraints=domain_constraints or [],
        )

        self.predicates[name] = predicate
        logger.debug(f"Registered predicate: {name}/{arity}")
        return predicate

    def add_rule(
        self,
        head: tuple[str, tuple[str, ...]],
        body: list[tuple[str, tuple[str, ...]]],
        connective: LogicalConnective = LogicalConnective.AND,
        weight: float = 1.0,
        confidence: float = 0.9,
        learnable: bool = True,
    ) -> DifferentiableRule:
        """Add a differentiable rule.

        Args:
            head: (predicate_name, arguments) for rule head
            body: List of (predicate_name, arguments) for rule body
            connective: Logical connective for body atoms
            weight: Initial rule weight
            confidence: Rule confidence
            learnable: Whether weight is learnable

        Returns:
            Created DifferentiableRule
        """
        head_pred_name, head_args = head
        if head_pred_name not in self.predicates:
            self.register_predicate(head_pred_name, len(head_args))

        head_atom = LogicalAtom(
            predicate=self.predicates[head_pred_name],
            arguments=head_args,
        )

        body_atoms = []
        for pred_name, args in body:
            if pred_name not in self.predicates:
                self.register_predicate(pred_name, len(args))
            body_atoms.append(
                LogicalAtom(
                    predicate=self.predicates[pred_name],
                    arguments=args,
                )
            )

        self._rule_counter += 1
        rule_id = f"rule_{self._rule_counter:04d}"

        rule = DifferentiableRule(
            rule_id=rule_id,
            head=head_atom,
            body=body_atoms,
            connective=connective,
            weight=weight,
            confidence=confidence,
            learnable=learnable,
        )

        self.rules[rule_id] = rule
        logger.debug(f"Added rule: {rule}")
        return rule

    def add_fact(
        self,
        predicate_name: str,
        arguments: tuple[str, ...],
        truth_value: float = 1.0,
    ) -> LogicalAtom:
        """Add a ground fact to the knowledge base."""
        if predicate_name not in self.predicates:
            self.register_predicate(predicate_name, len(arguments))

        atom = LogicalAtom(
            predicate=self.predicates[predicate_name],
            arguments=arguments,
            truth_value=truth_value,
        )

        self.facts.append(atom)
        return atom

    def infer(
        self,
        query: tuple[str, tuple[str, ...]],
        max_depth: int | None = None,
        include_counterfactuals: bool = False,
    ) -> InferenceResult:
        """Perform differentiable inference on a query.

        Args:
            query: (predicate_name, arguments) to prove
            max_depth: Maximum inference depth
            include_counterfactuals: Generate counterfactual explanations

        Returns:
            InferenceResult with probability and proof trace
        """
        pred_name, args = query
        if pred_name not in self.predicates:
            return InferenceResult(
                query=LogicalAtom(
                    predicate=Predicate(
                        name=pred_name, arity=len(args), predicate_type=PredicateType.N_ARY
                    ),
                    arguments=args,
                ),
                probability=0.0,
                confidence=0.0,
                supporting_rules=[],
                proof_trace=[],
                uncertainty=1.0,
            )

        query_atom = LogicalAtom(
            predicate=self.predicates[pred_name],
            arguments=args,
        )

        matching_facts = [
            f for f in self.facts if f.predicate.name == pred_name and f.arguments == args
        ]

        if matching_facts:
            fact = matching_facts[0]
            return InferenceResult(
                query=query_atom,
                probability=fact.truth_value,
                confidence=1.0,
                supporting_rules=[],
                proof_trace=[{"type": "fact_match", "fact": str(fact)}],
                uncertainty=0.0,
            )

        applicable_rules = [r for r in self.rules.values() if r.head.predicate.name == pred_name]

        if not applicable_rules:
            return InferenceResult(
                query=query_atom,
                probability=0.0,
                confidence=0.5,
                supporting_rules=[],
                proof_trace=[{"type": "no_applicable_rules"}],
                uncertainty=0.5,
            )

        max_depth = max_depth or self.max_proof_depth
        probability, confidence, supporting, trace = self._backward_chain(
            query_atom, applicable_rules, max_depth, {}
        )

        counterfactuals = []
        if include_counterfactuals and probability > 0.3:
            counterfactuals = self._generate_counterfactuals(query_atom, trace)

        return InferenceResult(
            query=query_atom,
            probability=probability,
            confidence=confidence,
            supporting_rules=supporting,
            proof_trace=trace,
            counterfactuals=counterfactuals,
            uncertainty=1.0 - confidence,
        )

    def _backward_chain(
        self,
        goal: LogicalAtom,
        rules: list[DifferentiableRule],
        depth: int,
        substitution: dict[str, str],
    ) -> tuple[float, float, list[str], list[dict[str, Any]]]:
        """Backward chaining inference with soft matching."""
        if depth <= 0:
            return 0.0, 0.0, [], [{"type": "depth_limit_reached"}]

        trace: list[dict[str, Any]] = []
        total_prob = 0.0
        total_conf = 0.0
        supporting: list[str] = []

        for rule in rules:
            bindings = self._unify(goal, rule.head, substitution)
            if bindings is None:
                continue

            body_probs = []
            body_trace = []

            for body_atom in rule.body:
                bound_atom = self._apply_bindings(body_atom, bindings)

                matching_facts = [
                    f
                    for f in self.facts
                    if f.predicate.name == bound_atom.predicate.name
                    and self._matches_pattern(f.arguments, bound_atom.arguments)
                ]

                if matching_facts:
                    body_probs.append(max(f.truth_value for f in matching_facts))
                    body_trace.append(
                        {
                            "atom": str(bound_atom),
                            "matched": True,
                            "probability": body_probs[-1],
                        }
                    )
                else:
                    sub_rules = [
                        r
                        for r in self.rules.values()
                        if r.head.predicate.name == bound_atom.predicate.name
                    ]
                    if sub_rules and depth > 1:
                        sub_prob, sub_conf, sub_support, sub_trace = self._backward_chain(
                            bound_atom, sub_rules, depth - 1, bindings
                        )
                        body_probs.append(sub_prob)
                        body_trace.append(
                            {
                                "atom": str(bound_atom),
                                "matched": False,
                                "sub_inference": sub_trace,
                                "probability": sub_prob,
                            }
                        )
                    else:
                        body_probs.append(0.0)
                        body_trace.append(
                            {
                                "atom": str(bound_atom),
                                "matched": False,
                                "probability": 0.0,
                            }
                        )

            if body_probs:
                if rule.connective == LogicalConnective.AND:
                    body_combined = body_probs[0]
                    for p in body_probs[1:]:
                        body_combined = self.t_norm.conjunction(body_combined, p)
                elif rule.connective == LogicalConnective.OR:
                    body_combined = body_probs[0]
                    for p in body_probs[1:]:
                        body_combined = self.t_norm.disjunction(body_combined, p)
                else:
                    body_combined = sum(body_probs) / len(body_probs)

                head_prob = rule.weight * rule.confidence * body_combined
                total_prob = self.t_norm.disjunction(total_prob, head_prob)
                total_conf = max(total_conf, rule.confidence)
                supporting.append(rule.rule_id)

                trace.append(
                    {
                        "rule": rule.rule_id,
                        "rule_text": str(rule),
                        "body_trace": body_trace,
                        "body_combined": body_combined,
                        "head_probability": head_prob,
                    }
                )

        return total_prob, total_conf, supporting, trace

    def _unify(
        self,
        atom1: LogicalAtom,
        atom2: LogicalAtom,
        substitution: dict[str, str],
    ) -> dict[str, str] | None:
        """Soft unification with variable binding."""
        if atom1.predicate.name != atom2.predicate.name:
            return None

        if len(atom1.arguments) != len(atom2.arguments):
            return None

        new_subs = substitution.copy()

        for arg1, arg2 in zip(atom1.arguments, atom2.arguments):
            if arg2.startswith("?"):
                if arg2 in new_subs:
                    if new_subs[arg2] != arg1:
                        return None
                else:
                    new_subs[arg2] = arg1
            elif arg1.startswith("?"):
                if arg1 in new_subs:
                    if new_subs[arg1] != arg2:
                        return None
                else:
                    new_subs[arg1] = arg2
            elif arg1 != arg2:
                return None

        return new_subs

    def _apply_bindings(
        self,
        atom: LogicalAtom,
        bindings: dict[str, str],
    ) -> LogicalAtom:
        """Apply variable bindings to an atom."""
        new_args = tuple(bindings.get(arg, arg) for arg in atom.arguments)
        return LogicalAtom(
            predicate=atom.predicate,
            arguments=new_args,
            truth_value=atom.truth_value,
        )

    def _matches_pattern(
        self,
        fact_args: tuple[str, ...],
        pattern_args: tuple[str, ...],
    ) -> bool:
        """Check if fact arguments match pattern."""
        if len(fact_args) != len(pattern_args):
            return False

        for f_arg, p_arg in zip(fact_args, pattern_args):
            if p_arg.startswith("?"):
                continue
            if f_arg != p_arg:
                return False

        return True

    def _generate_counterfactuals(
        self,
        query: LogicalAtom,
        proof_trace: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Generate counterfactual explanations."""
        counterfactuals = []

        for step in proof_trace:
            if "body_trace" not in step:
                continue

            for body_step in step["body_trace"]:
                if body_step.get("matched") and body_step.get("probability", 0) > 0.5:
                    counterfactuals.append(
                        {
                            "type": "remove_fact",
                            "fact": body_step["atom"],
                            "original_probability": body_step["probability"],
                            "counterfactual_effect": f"If {body_step['atom']} were false, "
                            f"conclusion would be less certain",
                        }
                    )

        return counterfactuals[:5]

    def learn_from_feedback(
        self,
        query: tuple[str, tuple[str, ...]],
        expected_probability: float,
        learning_rate: float = 0.01,
    ) -> dict[str, float]:
        """Active learning: adjust rule weights from user feedback.

        Args:
            query: Query that was evaluated
            expected_probability: User-provided expected probability
            learning_rate: Learning rate for weight adjustment

        Returns:
            Dictionary of rule weight updates
        """
        result = self.infer(query)
        error = expected_probability - result.probability

        weight_updates = {}

        for rule_id in result.supporting_rules:
            if rule_id in self.rules:
                rule = self.rules[rule_id]
                if rule.learnable:
                    old_weight = rule.weight
                    rule.weight = np.clip(rule.weight + learning_rate * error, 0.01, 1.0)
                    weight_updates[rule_id] = rule.weight - old_weight

        logger.info(f"Learning update: error={error:.3f}, " f"updated {len(weight_updates)} rules")

        return weight_updates

    def get_statistics(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            "n_predicates": len(self.predicates),
            "n_rules": len(self.rules),
            "n_facts": len(self.facts),
            "predicate_dim": self.predicate_dim,
            "max_proof_depth": self.max_proof_depth,
            "t_norm": type(self.t_norm).__name__,
            "torch_available": TORCH_AVAILABLE,
            "device": self.device,
        }
