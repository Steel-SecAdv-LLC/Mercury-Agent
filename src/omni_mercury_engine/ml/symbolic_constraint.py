# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Differentiable symbolic-constraint layer for neuro-symbolic co-training.

This module turns a small, declarative *rule graph* relating the base
detectors' consensus to the fusion network's anomaly probability into a
single differentiable satisfaction scalar.  Adding ``(1 - satisfaction)``
to the supervised fusion loss co-trains the neural fusion network
(``OmniFusionModel``) *with* symbolic constraints: the gradient of the
constraint flows back into the network's anomaly head, while the layer's
own learnable rule/detector weights adapt which constraints to trust.

Why this is a *genuine* neuro-symbolic component (anti-theater):

* It is a compact, proper Logic Tensor Network (LTN).  Predicates are
  grounded as fuzzy truth values in ``[0, 1]``; logical connectives are
  the differentiable product/Reichenbach fuzzy operators reused from
  :class:`omni_mercury_engine.models.neurosymbolic_temporal.FuzzyOperators`;
  universal quantification over the batch uses the smooth ``pmean``
  aggregator.  Everything is ``torch`` autograd-traceable.
* The constraint encodes *inductive bias the labels do not* -- the
  unsupervised agreement structure of independent detectors -- so it can
  improve sample-efficiency and reduce false positives rather than merely
  restating the supervised objective.  Whether it actually does so on real
  held-out labels is settled empirically by ``benchmarks/neurosymbolic_ablation.py``.
  A *fixed* weight was quarantined (it taxed the abundant-label regime); the
  label-scarcity :class:`ScarcityWeightSchedule` cleared the ablation's dominance
  bar, so co-training is on by default via ``symbolic_weight="adaptive"`` and
  decays to the neural path when labels are abundant.

The rule graph (default :func:`consensus_rule_graph`):

    R1  detector-consensus  ->  anomalous            (recall / evidence rule)
    R2  not detector-consensus  ->  not anomalous    (precision / false-positive rule)

``Consensus`` is a *learned weighted* aggregation of the per-detector
anomaly scores, so the layer also learns which detectors to trust.  Each
rule carries a learnable confidence weight (softmax-normalised), exactly
as a Real-Logic LTN weights its axioms.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, TypedDict

import torch
from torch import nn

from omni_mercury_engine.models.neurosymbolic_temporal import FuzzyOperators

__all__ = [
    "Rule",
    "RuleGraph",
    "ScarcityWeightSchedule",
    "SymbolicConstraintModule",
    "SymbolicWeight",
    "ThresholdAtom",
    "consensus_rule_graph",
    "consensus_salience_rule_graph",
    "parse_evolved_predicate",
    "quantize_threshold",
    "resolve_rule_graph",
    "resolve_symbolic_weight",
    "rule_graph_from_spec",
    "rule_graph_to_spec",
]


class RuleExplanation(TypedDict):
    """JSON-serialisable explanation for one symbolic rule."""

    statement: str
    description: str
    satisfaction: float
    confidence: float


class SymbolicExplanation(TypedDict):
    """JSON-serialisable symbolic-constraint explanation payload."""

    graph: str
    semantics: str
    satisfaction: float
    rules: dict[str, RuleExplanation]
    detector_weights: list[float]


# Clamp groundings away from the hard {0, 1} boundary so Reichenbach
# implication and its dual keep well-conditioned gradients.
_EPS: float = 1e-6

# Selectable fuzzy implication semantics for the rule residua. All three are
# real torch-differentiable tensor operators on FuzzyOperators; exposing them
# here revives the crisp (Gödel / Łukasiewicz) operators -- previously dormant,
# only ``implies_product`` was used -- as a measurable design axis settled by the
# ablation rather than assumed. ``product``/``reichenbach`` is the smooth default.
_IMPLICATIONS = {
    "product": FuzzyOperators.implies_product,
    "reichenbach": FuzzyOperators.implies_product,
    "lukasiewicz": FuzzyOperators.implies_lukasiewicz,
    "godel": FuzzyOperators.implies_godel,
}


@dataclass(frozen=True)
class Rule:
    """A single fuzzy first-order implication ``antecedent -> consequent``.

    The rule is declarative metadata: :class:`SymbolicConstraintModule`
    reads ``antecedent``/``consequent`` to select grounded predicates and
    applies the differentiable implication operator.  Keeping rules as data
    (rather than code) makes the constraint graph inspectable and testable.

    Attributes:
        name: Stable identifier used in diagnostics/explanations.
        antecedent: Predicate name forming the implication's left-hand side.
        consequent: Predicate name forming the implication's right-hand side.
        description: Human-readable statement of the domain rule.
    """

    name: str
    antecedent: str
    consequent: str
    description: str = ""


@dataclass(frozen=True)
class RuleGraph:
    """An ordered, named collection of :class:`Rule` objects.

    The graph's *nodes* are the predicate names referenced by its rules
    (``predicates``); its *edges* are the rules themselves.  Order is
    significant only in that it fixes the index of each rule's learnable
    confidence weight inside :class:`SymbolicConstraintModule`.
    """

    name: str
    rules: tuple[Rule, ...]
    predicates: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Finalize dataclass initialization."""
        if not self.rules:
            raise ValueError("RuleGraph requires at least one rule")
        referenced = {r.antecedent for r in self.rules} | {r.consequent for r in self.rules}
        # Derive predicate set from the rules when not supplied explicitly.
        object.__setattr__(self, "predicates", frozenset(self.predicates) | referenced)

    def __len__(self) -> int:
        """Return the length."""
        return len(self.rules)


# -- evolved-predicate grammar -----------------------------------------------
#
# Genetically evolved rule graphs (``omni_mercury_engine.ml.rule_evolution``)
# stay plain :class:`Rule`/:class:`RuleGraph` data by encoding *conjunctive
# threshold predicates* in the predicate NAME with a small canonical grammar:
#
#     atom      := builtin | threshold
#     builtin   := "Consensus" | "NotConsensus"
#     threshold := "x" <channel> (">=" | "<=") <threshold %.6f in [0, 1]>
#     predicate := atom ("&" atom)*
#
# e.g. ``"x3>=0.612500&x7<=0.250000"`` reads "score channel 3 is at least
# 0.6125 AND score channel 7 is at most 0.25".  Channels index the module's
# ``detector_scores`` columns -- the same per-detector consensus channels the
# built-in ``Consensus`` predicate aggregates -- so an evolved graph plugs into
# :class:`SymbolicConstraintModule` exactly like the hand-written graphs.
# Thresholds are quantised to 6 decimals so the name round-trips losslessly.

# Fixed sigmoid slope for grounding a threshold atom as a fuzzy truth value:
# ``sigmoid((score - threshold) * EVOLVED_ATOM_SLOPE)``.  On [0, 1]-normalised
# score channels a slope of 25 gives a crisp-but-smooth transition band of
# roughly +/-0.09 around the threshold.  It is a *fixed part of the scoring
# semantics* (never fit per dataset), so the fitness-time scoring path and the
# deployed scoring path are the same function by construction.
EVOLVED_ATOM_SLOPE: float = 25.0

# Number of decimals thresholds are quantised to; matched by the atom regex so
# format -> parse round-trips exactly.
THRESHOLD_DECIMALS: int = 6

# Builtin predicates allowed as atoms inside an evolved conjunction.  Salience
# is deliberately excluded: it requires the module's optional learnable
# salience parameters, which are only instantiated when the *graph name*
# ``Salient`` is referenced directly.
_BUILTIN_CONJUNCTION_ATOMS = frozenset({"Consensus", "NotConsensus"})

_THRESHOLD_ATOM_RE = re.compile(r"^x(\d+)(>=|<=)([01]\.\d{6})$")

_THRESHOLD_OPS = (">=", "<=")


def quantize_threshold(value: float) -> float:
    """Clip a threshold to ``[0, 1]`` and quantise it to the grammar precision.

    Quantisation makes the predicate-name encoding lossless: a quantised
    threshold formatted with ``%.6f`` parses back to the identical float.

    Args:
        value: Raw threshold value.

    Returns:
        The clipped, 6-decimal-quantised threshold.

    Raises:
        ValueError: If ``value`` is not finite.
    """
    if not math.isfinite(value):
        raise ValueError(f"threshold must be finite, got {value}")
    clipped = min(max(float(value), 0.0), 1.0)
    return float(f"{clipped:.{THRESHOLD_DECIMALS}f}")


@dataclass(frozen=True, order=True)
class ThresholdAtom:
    """One soft threshold test over a named score channel.

    The atom is grounded as the differentiable fuzzy truth value
    ``sigmoid((score[channel] - threshold) * EVOLVED_ATOM_SLOPE)`` for ``>=``
    (mirrored for ``<=``), i.e. the same soft-threshold construction as the
    module's ``Salient`` predicate but with an *evolved, fixed* threshold
    instead of a learnable one.

    Attributes:
        channel: Column index into the module's ``detector_scores`` matrix.
        op: Comparison operator, ``">="`` or ``"<="``.
        threshold: Quantised threshold in ``[0, 1]`` (see
            :func:`quantize_threshold`).
    """

    channel: int
    op: str
    threshold: float

    def __post_init__(self) -> None:
        """Finalize dataclass initialization."""
        if self.channel < 0:
            raise ValueError(f"channel must be >= 0, got {self.channel}")
        if self.op not in _THRESHOLD_OPS:
            raise ValueError(f"op must be one of {_THRESHOLD_OPS}, got {self.op!r}")
        quantised = quantize_threshold(self.threshold)
        if quantised != self.threshold:
            raise ValueError(
                f"threshold {self.threshold!r} is not quantised to "
                f"{THRESHOLD_DECIMALS} decimals in [0, 1]; use quantize_threshold()"
            )

    def fragment(self) -> str:
        """Return the canonical grammar fragment, e.g. ``"x3>=0.612500"``."""
        return f"x{self.channel}{self.op}{self.threshold:.{THRESHOLD_DECIMALS}f}"


def parse_evolved_predicate(name: str) -> tuple[str | ThresholdAtom, ...]:
    """Parse a conjunctive evolved-predicate name into its atoms.

    Args:
        name: Predicate name following the evolved-predicate grammar --
            ``"&"``-joined atoms, each either a builtin predicate
            (``Consensus``/``NotConsensus``) or a threshold test such as
            ``"x3>=0.612500"``.

    Returns:
        Tuple of atoms; builtin atoms stay strings, threshold atoms become
        :class:`ThresholdAtom`.

    Raises:
        ValueError: If the name (or any atom in it) does not follow the
            grammar.  Unknown predicate names fail loud here rather than
            silently grounding to something else.
    """
    parts = name.split("&")
    atoms: list[str | ThresholdAtom] = []
    for part in parts:
        if part in _BUILTIN_CONJUNCTION_ATOMS:
            atoms.append(part)
            continue
        match = _THRESHOLD_ATOM_RE.match(part)
        if match is None:
            raise ValueError(
                f"cannot ground predicate {name!r}: atom {part!r} is neither a "
                f"builtin atom ({sorted(_BUILTIN_CONJUNCTION_ATOMS)}) nor a "
                "threshold atom like 'x3>=0.612500'"
            )
        atoms.append(ThresholdAtom(channel=int(match[1]), op=match[2], threshold=float(match[3])))
    return tuple(atoms)


def rule_graph_to_spec(graph: RuleGraph) -> dict[str, Any]:
    """Serialise a :class:`RuleGraph` to a plain-JSON-compatible dict.

    Rules are pure data, so the spec is lossless: it captures the graph name
    and every rule's name/antecedent/consequent/description.  Used by the
    engine checkpoint (so a non-registry graph -- e.g. an evolved one --
    round-trips through ``save_model``/``load_model``) and by the evolved-graph
    artifact format.

    Args:
        graph: Rule graph to serialise.

    Returns:
        JSON-compatible dict with ``name`` and ``rules`` keys.
    """
    return {
        "name": graph.name,
        "rules": [
            {
                "name": rule.name,
                "antecedent": rule.antecedent,
                "consequent": rule.consequent,
                "description": rule.description,
            }
            for rule in graph.rules
        ],
    }


def rule_graph_from_spec(spec: dict[str, Any]) -> RuleGraph:
    """Reconstruct a :class:`RuleGraph` from :func:`rule_graph_to_spec` output.

    Args:
        spec: Dict with ``name`` and a non-empty ``rules`` list.

    Returns:
        The reconstructed rule graph (equal to the original under dataclass
        equality).

    Raises:
        ValueError: If the spec is malformed (missing keys or no rules).
    """
    try:
        name = str(spec["name"])
        rule_entries = spec["rules"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"malformed rule-graph spec: {exc!r}") from exc
    if not isinstance(rule_entries, (list, tuple)) or not rule_entries:
        raise ValueError("rule-graph spec must contain a non-empty 'rules' list")
    rules = tuple(
        Rule(
            name=str(entry["name"]),
            antecedent=str(entry["antecedent"]),
            consequent=str(entry["consequent"]),
            description=str(entry.get("description", "")),
        )
        for entry in rule_entries
    )
    return RuleGraph(name=name, rules=rules)


def consensus_rule_graph() -> RuleGraph:
    """Build the default detector-consensus rule graph.

    Two complementary rules tie the fusion output to the unsupervised
    agreement of the base detectors:

    * ``R1_evidence``  -- ``Consensus -> Anomalous``.  When the detectors
      jointly support an anomaly, the fusion network should agree.  This is
      the *recall / evidence* prior and improves sample-efficiency by
      letting unlabelled detector structure shape the decision boundary.
    * ``R2_precision`` -- ``NotConsensus -> NotAnomalous``.  When the
      detectors jointly see nothing, the fusion network should not fire.
      This is the *precision / false-positive* prior.

    Returns:
        The two-rule :class:`RuleGraph` used by default in co-training.
    """
    return RuleGraph(
        name="detector_consensus",
        rules=(
            Rule(
                name="R1_evidence",
                antecedent="Consensus",
                consequent="Anomalous",
                description="If the detectors jointly support an anomaly, fusion is anomalous.",
            ),
            Rule(
                name="R2_precision",
                antecedent="NotConsensus",
                consequent="NotAnomalous",
                description="If the detectors jointly see nothing, fusion is not anomalous.",
            ),
        ),
    )


def consensus_salience_rule_graph() -> RuleGraph:
    """Consensus rules plus a salience (any-strong-detector) recall rule.

    This is the richer rule graph that revives the *threshold-rule* idea of the
    dormant ``cognitive/symbolic_logic_layer.py`` (a ``ThresholdRule`` fires when
    a variable crosses a threshold) as a **differentiable** axiom, and answers
    the ledger's question: does richer symbolic structure beat the minimal
    two-rule consensus?

    The learned ``Consensus`` predicate is an AND-like *weighted mean* of the
    detector scores -- it dilutes a single strong detector. ``Salient`` is its
    disjunctive complement: a soft-existential over per-detector soft-threshold
    indicators (the differentiable ``ThresholdRule``), high when **any** detector
    crosses the learned threshold. The added rule is a recall axiom:

    * ``R3_salience`` -- ``Salient -> Anomalous``.  If any single detector
      saliently fires, the fusion network should not dismiss it.

    Whether this richer graph actually helps on real labels is settled by
    ``benchmarks/symbolic_rulegraph_sweep.py``; the default stays the minimal
    consensus graph until it does.

    Returns:
        The three-rule consensus+salience :class:`RuleGraph`.
    """
    base = consensus_rule_graph()
    return RuleGraph(
        name="detector_consensus_salience",
        rules=(
            *base.rules,
            Rule(
                name="R3_salience",
                antecedent="Salient",
                consequent="Anomalous",
                description="If any single detector saliently fires, fusion is anomalous.",
            ),
        ),
    )


_RULE_GRAPHS = {
    "consensus": consensus_rule_graph,
    "consensus_salience": consensus_salience_rule_graph,
}


_EVOLVED_GRAPH_PREFIX = "evolved:"


def resolve_rule_graph(name: str | RuleGraph) -> RuleGraph:
    """Resolve a rule-graph name (or an explicit graph) to a :class:`RuleGraph`.

    Accepted specifications:

    * A :class:`RuleGraph` instance -- returned unchanged.
    * A registry name -- ``"consensus"`` or ``"consensus_salience"``.
    * ``"evolved:<path>"`` -- load a genetically evolved rule graph from the
      JSON artifact written by
      :func:`omni_mercury_engine.ml.rule_evolution.save_evolved_rule_graph`
      (path case is preserved; the prefix is case-insensitive).  This is the
      seam by which the engine selects an evolved graph exactly like the
      hand-written consensus graphs, e.g.
      ``fit_fusion(..., symbolic_rule_graph="evolved:benchmarks/evolved_rule_graph.json")``.

    Args:
        name: Rule-graph specification as described above.

    Returns:
        The resolved rule graph.

    Raises:
        ValueError: If the name is not a known registry name, a valid
            ``evolved:`` spec, or a :class:`RuleGraph`.
    """
    if isinstance(name, RuleGraph):
        return name
    stripped = name.strip()
    if stripped.lower().startswith(_EVOLVED_GRAPH_PREFIX):
        path = stripped[len(_EVOLVED_GRAPH_PREFIX) :].strip()
        if not path:
            raise ValueError("evolved rule-graph spec requires a path: 'evolved:<path>'")
        # Local import: rule_evolution imports this module at top level, so
        # the reverse dependency must stay function-local to avoid a cycle.
        from omni_mercury_engine.ml.rule_evolution import load_evolved_rule_graph

        return load_evolved_rule_graph(path)
    key = stripped.lower()
    if key not in _RULE_GRAPHS:
        raise ValueError(
            f"unknown rule graph {name!r}; expected one of {sorted(_RULE_GRAPHS)} "
            "or 'evolved:<path>'"
        )
    return _RULE_GRAPHS[key]()


@dataclass(frozen=True)
class ScarcityWeightSchedule:
    """Label-scarcity-adaptive schedule for the co-training weight ``lambda``.

    The neuro-symbolic ablation (``benchmarks/neurosymbolic_ablation.py``,
    recorded in ``docs/NEUROSYMBOLIC.md``) found the detector-consensus
    constraint **helps when labels are scarce** and **washes out or slightly
    regresses when they are abundant**: a single fixed ``lambda`` therefore
    cannot win everywhere -- it pays its way in the low-data regime and taxes
    the high-data regime. This schedule operationalises that measured finding
    rather than assuming it: it spends the constraint *only where the ablation
    showed it carries signal*, and the abundant-label regime is left as the
    purely-neural path.

    The binding quantity for anomaly detection is the number of **labelled
    anomalies** ``n_pos`` -- with few positives the supervised decision boundary
    is under-determined and the unsupervised consensus prior injects useful
    structure; with many it is already well-pinned and the prior only adds bias.
    The weight therefore decays smoothly with ``n_pos``::

        lambda_eff(n_pos) = lam_max * exp(-n_pos / n0)

    so ``lambda_eff -> lam_max`` as ``n_pos -> 0`` and ``lambda_eff -> 0`` as
    ``n_pos`` grows. Below ``floor`` the weight snaps to exactly ``0`` so the
    abundant-label regime reproduces the neural training path byte-for-byte
    (no constraint module is even instantiated downstream).

    The defaults are *pre-registered*, not tuned to pass the ablation:

    * ``lam_max = 0.1`` -- the exact value already ablated as the fixed weight,
      so the only new degree of freedom introduced here is the scarcity gating.
    * ``n0 = 25`` -- an anomaly-count scale of a few dozen positives, below which
      a handful of labels cannot pin a boundary in the ~6-30 feature dimensions
      of the ADBench anomaly tasks and an unsupervised prior should contribute.

    Attributes:
        lam_max: Maximum symbolic weight, reached as ``n_pos -> 0``.
        n0: Anomaly-count decay scale (positives); larger keeps ``lambda`` high
            for more positives.
        floor: Weights below this snap to ``0`` (neural path); keeps the
            high-data regime exactly neural and avoids instantiating a
            constraint module for a negligible weight.
    """

    lam_max: float = 0.1
    n0: float = 25.0
    floor: float = 1e-3

    def __post_init__(self) -> None:
        # Non-finite parameters (e.g. NaN/inf from config) would make
        # ``weight_for`` return NaN and silently disable the constraint via
        # NaN comparisons, so reject them up front alongside the sign checks.
        """Finalize dataclass initialization."""
        if not math.isfinite(self.lam_max) or self.lam_max < 0.0:
            raise ValueError(f"lam_max must be a finite value >= 0, got {self.lam_max}")
        if not math.isfinite(self.n0) or self.n0 <= 0.0:
            raise ValueError(f"n0 must be a finite value > 0, got {self.n0}")
        if not math.isfinite(self.floor) or self.floor < 0.0:
            raise ValueError(f"floor must be a finite value >= 0, got {self.floor}")

    def weight_for(self, n_positive: int) -> float:
        """Resolve the effective ``lambda`` for a training set with ``n_positive`` anomalies.

        Args:
            n_positive: Number of labelled (or consensus-labelled) anomalies in the
                training split. Negative values are treated as ``0``.

        Returns:
            The effective non-negative symbolic weight; exactly ``0`` when the
            decayed value falls below ``floor`` (the purely-neural regime).
        """
        n = max(0, int(n_positive))
        lam = self.lam_max * math.exp(-n / self.n0)
        return 0.0 if lam < self.floor else float(lam)


# A symbolic co-training weight may be given as a concrete ``lambda``
# (``float``/``int``), the string ``"adaptive"`` (use the default scarcity
# schedule), or an explicit :class:`ScarcityWeightSchedule`.
SymbolicWeight = float | int | str | ScarcityWeightSchedule

_ADAPTIVE_ALIASES = frozenset({"adaptive", "scarcity", "auto"})


def resolve_symbolic_weight(weight: SymbolicWeight, n_positive: int) -> float:
    """Resolve a symbolic-weight specification to a concrete ``lambda`` float.

    This is the single place that maps the public ``symbolic_weight`` argument
    of :meth:`OmniMercuryEngine.fit_fusion` onto the scalar the training loop
    consumes, so the per-batch loss code stays a simple ``lambda * loss`` term
    regardless of how the weight was specified.

    Args:
        weight: A concrete weight (``float``/``int``), the string ``"adaptive"``
            (or ``"scarcity"``/``"auto"``) for the default scarcity schedule, or
            an explicit :class:`ScarcityWeightSchedule`.
        n_positive: Number of labelled anomalies in the training split, used to
            resolve adaptive specifications.

    Returns:
        The effective non-negative symbolic weight as a ``float``.

    Raises:
        ValueError: If ``weight`` is a string other than a known adaptive alias,
            a boolean, or a numeric weight is negative.
    """
    if isinstance(weight, ScarcityWeightSchedule):
        return weight.weight_for(n_positive)
    if isinstance(weight, str):
        key = weight.strip().lower()
        if key in _ADAPTIVE_ALIASES:
            return ScarcityWeightSchedule().weight_for(n_positive)
        raise ValueError(
            f"unknown symbolic_weight {weight!r}; expected a float, a "
            f"ScarcityWeightSchedule, or one of {sorted(_ADAPTIVE_ALIASES)}"
        )
    # ``bool`` is a subclass of ``int``; reject it explicitly so ``True`` cannot
    # silently enable co-training (1.0) nor ``False`` silently disable it.
    if isinstance(weight, bool):
        raise ValueError(
            f"symbolic_weight must be a number, 'adaptive', or a schedule; got bool {weight!r}"
        )
    lam = float(weight)
    # Reject non-finite weights: NaN would silently disable co-training
    # (``nan > 0`` is False) while propagating a non-finite lambda, and inf
    # would blow up the loss -- both contradict the non-negative contract.
    if not math.isfinite(lam) or lam < 0.0:
        raise ValueError(f"symbolic_weight must be a finite value >= 0, got {lam}")
    return lam


class SymbolicConstraintModule(nn.Module):
    """Differentiable LTN constraint over detector consensus and fusion output.

    Given the fusion network's per-sample anomaly probability and the matrix
    of per-detector anomaly scores, this layer grounds the predicates of a
    :class:`RuleGraph`, evaluates each fuzzy implication, aggregates rule
    satisfaction over the batch with a smooth universal quantifier, and
    combines the rules with learnable confidence weights into a single
    satisfaction scalar in ``[0, 1]``.

    The constraint loss is ``1 - satisfaction``.  Because ``satisfaction``
    depends on the fusion network's ``anomaly_prob`` (an input to
    :meth:`forward`), adding this loss to the supervised objective makes the
    constraint co-train the fusion network.  The layer's own parameters --
    learned detector reliabilities and rule confidences -- are optimised
    jointly.

    Args:
        num_detectors: Number of base-detector score channels ``D``.  May be
            ``0`` (the constraint then satisfies trivially).
        rule_graph: Rules to enforce.  Defaults to :func:`consensus_rule_graph`.
        learn_detector_reliability: If ``True`` (default), ``Consensus`` is a
            softmax-weighted mean of detector scores with learnable weights;
            if ``False`` it is a plain mean.
        p_aggregator: ``p`` for the ``pmean`` universal-quantifier aggregator
            over the batch (``>= 1``; higher ``p`` penalises violations more).
        consensus_sharpness: Initial value of the learnable temperature that
            sharpens ``Consensus`` toward ``{0, 1}`` (kept positive via softplus).
    """

    def __init__(
        self,
        num_detectors: int,
        rule_graph: RuleGraph | None = None,
        *,
        learn_detector_reliability: bool = True,
        p_aggregator: float = 2.0,
        consensus_sharpness: float = 1.0,
        semantics: str = "product",
    ) -> None:
        """Initialize the instance."""
        super().__init__()
        if num_detectors < 0:
            raise ValueError(f"num_detectors must be >= 0, got {num_detectors}")
        if p_aggregator < 1.0:
            raise ValueError(f"p_aggregator must be >= 1.0, got {p_aggregator}")

        self.num_detectors = int(num_detectors)
        self.rule_graph = rule_graph if rule_graph is not None else consensus_rule_graph()
        self.learn_detector_reliability = bool(learn_detector_reliability)
        self.p_aggregator = float(p_aggregator)

        # Learnable per-rule confidence (softmax-normalised at use time): a
        # proper LTN weights its axioms rather than treating them as equal.
        self.rule_weights = nn.Parameter(torch.zeros(len(self.rule_graph)))

        # Learnable detector reliabilities for the weighted-consensus predicate.
        if self.num_detectors > 0 and self.learn_detector_reliability:
            self.detector_logits: nn.Parameter | None = nn.Parameter(
                torch.zeros(self.num_detectors)
            )
        else:
            self.register_parameter("detector_logits", None)

        # Learnable sharpness applied around 0.5 to push Consensus toward a
        # crisp truth value; softplus keeps it strictly positive.
        self.sharpness_logit = nn.Parameter(torch.tensor(float(consensus_sharpness)))

        # Optional salience predicate (only when the rule graph references it):
        # a learnable per-detector soft threshold (the differentiable analog of a
        # ThresholdRule) aggregated by a soft-existential. Its learnable threshold
        # and sharpness are added only when needed so the default consensus graph
        # is parameter-identical to before.
        self._has_salience = "Salient" in self.rule_graph.predicates
        if self._has_salience and self.num_detectors > 0:
            # One learnable soft threshold per detector (the differentiable
            # ThresholdRule), broadcast over the batch in ``_salience``; the
            # sharpness is shared.
            self.salience_threshold: nn.Parameter | None = nn.Parameter(
                torch.full((self.num_detectors,), 0.5)
            )
            self.salience_sharpness: nn.Parameter | None = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_parameter("salience_threshold", None)
            self.register_parameter("salience_sharpness", None)

        key = semantics.strip().lower()
        if key not in _IMPLICATIONS:
            raise ValueError(
                f"unknown semantics {semantics!r}; expected one of {sorted(_IMPLICATIONS)}"
            )
        self.semantics = key
        self._implies = _IMPLICATIONS[key]
        self._not = FuzzyOperators.not_standard

    # -- predicate grounding -------------------------------------------------

    def _consensus(self, detector_scores: torch.Tensor) -> torch.Tensor:
        """Ground the ``Consensus`` predicate from detector scores.

        Args:
            detector_scores: ``(B, D)`` per-detector anomaly scores in ``[0, 1]``.

        Returns:
            ``(B, 1)`` fuzzy truth value of "the detectors jointly support
            an anomaly".
        """
        if self.detector_logits is not None:
            weights = torch.softmax(self.detector_logits, dim=0)
            consensus = detector_scores @ weights.unsqueeze(1)  # (B, 1)
        else:
            consensus = detector_scores.mean(dim=1, keepdim=True)

        # Sharpen around 0.5 with a positive temperature so the predicate can
        # become crisp during training without leaving [0, 1].
        sharpness = torch.nn.functional.softplus(self.sharpness_logit)
        consensus = torch.sigmoid((consensus - 0.5) * 4.0 * sharpness)
        return consensus.clamp(_EPS, 1.0 - _EPS)

    def predict(self, detector_scores: torch.Tensor) -> torch.Tensor:
        """Per-sample anomaly consensus probability — the module's inference API.

        Exposes the grounded ``Consensus`` predicate (the *same* fuzzy-logic
        aggregator co-trained inside :meth:`OmniMercuryEngine.fit_fusion`, where
        its gradient flows into the fusion network) as a standalone per-sample
        score. A co-trained module applies its learned detector reliabilities; an
        untrained module falls back to the deterministic uniform-weight
        consensus (parameters initialise to zero, so there is no random-init
        noise). This is the real inference signal the legacy ``LogicTensorNetwork``
        surface now routes through.

        Args:
            detector_scores: ``(B, D)`` per-detector anomaly scores in ``[0, 1]``,
                where ``D == num_detectors``.

        Returns:
            ``(B,)`` consensus probabilities in ``[0, 1]`` (detached, no grad).

        Raises:
            ValueError: If ``detector_scores`` is not 2-D or its width does not
                match ``num_detectors`` (fail closed on a shape mismatch rather
                than silently broadcasting).
        """
        if detector_scores.ndim != 2:
            raise ValueError(
                f"detector_scores must be 2-D (B, D); got shape {tuple(detector_scores.shape)}"
            )
        batch = detector_scores.shape[0]
        with torch.no_grad():
            if self.num_detectors == 0 or detector_scores.shape[1] == 0:
                # No detector channels -> trivial 0.5 consensus (no signal).
                # Honour the input's device so a caller running on an
                # accelerator gets a same-device tensor (the non-trivial path
                # below already returns on the module/input device); a default
                # CPU tensor here would raise a device mismatch downstream.
                return torch.full(
                    (batch,),
                    0.5,
                    dtype=detector_scores.dtype,
                    device=detector_scores.device,
                )
            if detector_scores.shape[1] != self.num_detectors:
                raise ValueError(
                    f"detector_scores width {detector_scores.shape[1]} != "
                    f"num_detectors {self.num_detectors}"
                )
            return self._consensus(detector_scores).squeeze(-1)

    def score_samples(self, detector_scores: torch.Tensor) -> torch.Tensor:
        """Per-sample anomaly score implied by the rule graph (modus-ponens pooling).

        This is the module's *per-sample* inference API and the single scoring
        path for rule graphs used as standalone detectors -- in particular for
        genetically evolved graphs (``omni_mercury_engine.ml.rule_evolution``),
        whose fitness is measured through this exact method, so the
        fitness-time and deployment-time scoring semantics are identical by
        construction (no train/serve skew).

        Semantics: each rule contributes its grounded antecedent truth
        ``a_r(x)`` as evidence *for* anomaly when its consequent is
        ``Anomalous`` and *against* anomaly when it is ``NotAnomalous``,
        weighted by the (softmax-normalised, learnable) rule confidences::

            score(x) = 0.5 + 0.5 * sum_r w_r * sign_r * a_r(x)

        A vacuous rule (``a_r ~ 0``) contributes nothing, so negative rules
        push the score down only when they actually fire.  For the untrained
        module the weights are exactly uniform (parameters initialise to
        zero), making the score deterministic; a co-trained module applies its
        learned confidences.  For the default consensus graph the score is a
        strictly monotone transform of :meth:`predict`'s consensus
        probability, so both rank identically.

        Args:
            detector_scores: ``(B, D)`` per-detector anomaly scores in
                ``[0, 1]``, where ``D == num_detectors``.  Values are clamped
                to ``[0, 1]`` as in :meth:`forward`.

        Returns:
            ``(B,)`` anomaly scores in ``[0, 1]`` (detached, no grad); ``0.5``
            everywhere when the module has no detector channels (no signal).

        Raises:
            ValueError: If ``detector_scores`` is not 2-D or its width does
                not match ``num_detectors``; if any rule's consequent is not
                ``Anomalous``/``NotAnomalous``; or if a rule's antecedent
                references the fusion output (circular for scoring).
        """
        if detector_scores.ndim != 2:
            raise ValueError(
                f"detector_scores must be 2-D (B, D); got shape {tuple(detector_scores.shape)}"
            )
        batch = detector_scores.shape[0]
        with torch.no_grad():
            if self.num_detectors == 0 or detector_scores.shape[1] == 0:
                return torch.full(
                    (batch,),
                    0.5,
                    dtype=detector_scores.dtype,
                    device=detector_scores.device,
                )
            if detector_scores.shape[1] != self.num_detectors:
                raise ValueError(
                    f"detector_scores width {detector_scores.shape[1]} != "
                    f"num_detectors {self.num_detectors}"
                )
            scores = detector_scores.clamp(0.0, 1.0)
            grounded = self._ground_shared(scores)
            weights = torch.softmax(self.rule_weights, dim=0)
            net = torch.zeros((batch, 1), dtype=scores.dtype, device=scores.device)
            for i, rule in enumerate(self.rule_graph.rules):
                if rule.consequent == "Anomalous":
                    sign = 1.0
                elif rule.consequent == "NotAnomalous":
                    sign = -1.0
                else:
                    raise ValueError(
                        "score_samples requires rule consequents 'Anomalous' or "
                        f"'NotAnomalous'; rule {rule.name!r} has {rule.consequent!r}"
                    )
                if rule.antecedent in ("Anomalous", "NotAnomalous"):
                    raise ValueError(
                        f"rule {rule.name!r} antecedent {rule.antecedent!r} references "
                        "the fusion output; per-sample scoring would be circular"
                    )
                if rule.antecedent not in grounded:
                    grounded[rule.antecedent] = self._ground_conjunction(
                        rule.antecedent, grounded, scores
                    )
                net = net + weights[i] * sign * grounded[rule.antecedent]
            return (0.5 + 0.5 * net).squeeze(1).clamp(0.0, 1.0)

    def _salience(self, detector_scores: torch.Tensor) -> torch.Tensor:
        """Ground the ``Salient`` predicate: "some detector saliently fires".

        Each detector score passes a learnable soft threshold
        ``sigmoid((score - tau) * 4 * softplus(beta))`` -- the differentiable
        analog of a crisp ``ThresholdRule`` -- and the per-detector indicators
        are combined by a soft-existential (product t-conorm) so the predicate is
        high when **any** detector crosses the threshold.

        Args:
            detector_scores: ``(B, D)`` per-detector anomaly scores in ``[0, 1]``.

        Returns:
            ``(B, 1)`` fuzzy truth value of "at least one detector is salient".
        """
        assert self.salience_threshold is not None and self.salience_sharpness is not None
        beta = torch.nn.functional.softplus(self.salience_sharpness)
        indicators = torch.sigmoid((detector_scores - self.salience_threshold) * 4.0 * beta)
        indicators = indicators.clamp(_EPS, 1.0 - _EPS)
        # Soft-existential (product t-conorm): 1 - prod(1 - indicator_k).
        salient = 1.0 - torch.prod(1.0 - indicators, dim=1, keepdim=True)
        return salient.clamp(_EPS, 1.0 - _EPS)

    def _ground_shared(self, detector_scores: torch.Tensor) -> dict[str, torch.Tensor]:
        """Ground the evidence predicates that depend only on detector scores.

        Returns a mapping ``predicate_name -> (B, 1)`` fuzzy truth tensor for
        ``Consensus``/``NotConsensus`` (and ``Salient``/``NotSalient`` when the
        graph uses salience).  Shared by the co-training path
        (:meth:`_ground`) and the inference path (:meth:`score_samples`), so
        both ground evidence identically.
        """
        consensus = self._consensus(detector_scores)
        grounded = {
            "Consensus": consensus,
            "NotConsensus": self._not(consensus),
        }
        if self._has_salience and self.salience_threshold is not None:
            salient = self._salience(detector_scores)
            grounded["Salient"] = salient
            grounded["NotSalient"] = self._not(salient)
        return grounded

    def _ground_conjunction(
        self,
        name: str,
        grounded: dict[str, torch.Tensor],
        detector_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Ground an evolved conjunctive predicate name (see the grammar).

        Each threshold atom becomes the fuzzy truth
        ``sigmoid((score[channel] - threshold) * EVOLVED_ATOM_SLOPE)`` (mirrored
        for ``<=``); builtin atoms reuse the already-grounded predicate; the
        conjunction is the product t-norm.

        Args:
            name: Predicate name following the evolved-predicate grammar.
            grounded: Already-grounded predicates (for builtin atoms).
            detector_scores: ``(B, D)`` per-detector scores in ``[0, 1]``.

        Returns:
            ``(B, 1)`` fuzzy truth tensor clamped away from ``{0, 1}``.

        Raises:
            ValueError: If the name does not parse, references an ungrounded
                builtin atom, or references a channel outside ``[0, D)``.
        """
        atoms = parse_evolved_predicate(name)
        truth: torch.Tensor | None = None
        for atom in atoms:
            if isinstance(atom, str):
                base = grounded.get(atom)
                if base is None:
                    raise ValueError(
                        f"predicate {name!r} references builtin atom {atom!r}, "
                        "which is not grounded for this module"
                    )
                part = base
            else:
                if atom.channel >= detector_scores.shape[1]:
                    raise ValueError(
                        f"predicate {name!r} references score channel {atom.channel} "
                        f"but only {detector_scores.shape[1]} channels are available"
                    )
                column = detector_scores[:, atom.channel : atom.channel + 1]
                logits = (column - atom.threshold) * EVOLVED_ATOM_SLOPE
                part = torch.sigmoid(logits if atom.op == ">=" else -logits)
            truth = part if truth is None else truth * part
        assert truth is not None  # the grammar guarantees at least one atom
        return truth.clamp(_EPS, 1.0 - _EPS)

    def _ground(
        self, anomaly_prob: torch.Tensor, detector_scores: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Ground every predicate referenced by the rule graph.

        Returns a mapping ``predicate_name -> (B, 1)`` fuzzy truth tensor.
        Builtin predicates ground exactly as before; any further predicate
        name referenced by the graph is grounded through the evolved
        conjunctive-threshold grammar (:func:`parse_evolved_predicate`), in
        deterministic rule order.  Unknown names fail loud.
        """
        anomalous = anomaly_prob.clamp(_EPS, 1.0 - _EPS)
        grounded = {
            "Anomalous": anomalous,
            "NotAnomalous": self._not(anomalous),
        }
        grounded.update(self._ground_shared(detector_scores))
        for rule in self.rule_graph.rules:
            for name in (rule.antecedent, rule.consequent):
                if name not in grounded:
                    grounded[name] = self._ground_conjunction(name, grounded, detector_scores)
        return grounded

    # -- forward / loss ------------------------------------------------------

    def forward(
        self, anomaly_prob: torch.Tensor, detector_scores: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Evaluate rule-graph satisfaction.

        Args:
            anomaly_prob: ``(B,)`` or ``(B, 1)`` fusion anomaly probabilities.
            detector_scores: ``(B, D)`` per-detector anomaly scores in ``[0, 1]``.
                ``D`` may be ``0``; the constraint is then trivially satisfied.

        Returns:
            Dict with:

            * ``satisfaction`` -- scalar overall satisfaction in ``[0, 1]``.
            * ``loss`` -- scalar ``1 - satisfaction``.
            * ``rule_satisfaction`` -- ``(num_rules,)`` per-rule satisfaction.
            * ``rule_weights`` -- ``(num_rules,)`` softmax rule confidences.
            * ``detector_weights`` -- ``(D,)`` detector reliabilities (or empty).
        """
        anomaly_prob = anomaly_prob.reshape(anomaly_prob.shape[0], 1)
        device = anomaly_prob.device
        dtype = anomaly_prob.dtype

        # No detector channels -> no constraint signal; satisfy trivially while
        # keeping a graph connection to anomaly_prob so .backward() is safe.
        if detector_scores is None or detector_scores.numel() == 0 or self.num_detectors == 0:
            sat = torch.ones((), device=device, dtype=dtype) - 0.0 * anomaly_prob.mean()
            return {
                "satisfaction": sat,
                "loss": 1.0 - sat,
                "rule_satisfaction": torch.ones(len(self.rule_graph), device=device, dtype=dtype),
                "rule_weights": torch.softmax(self.rule_weights, dim=0),
                "detector_weights": torch.empty(0, device=device, dtype=dtype),
            }

        detector_scores = detector_scores.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        grounded = self._ground(anomaly_prob, detector_scores)

        per_rule: list[torch.Tensor] = []
        for rule in self.rule_graph.rules:
            antecedent = grounded[rule.antecedent]
            consequent = grounded[rule.consequent]
            truth = self._implies(antecedent, consequent).clamp(_EPS, 1.0 - _EPS)  # (B, 1)
            # Universal quantification over the batch via smooth pmean.
            rule_sat = FuzzyOperators.forall_pmean(truth.squeeze(1), p=self.p_aggregator, dim=0)
            per_rule.append(rule_sat)

        rule_satisfaction = torch.stack(per_rule)  # (num_rules,)
        weights = torch.softmax(self.rule_weights, dim=0)
        satisfaction = torch.sum(rule_satisfaction * weights)

        detector_weights = (
            torch.softmax(self.detector_logits, dim=0)
            if self.detector_logits is not None
            else torch.full(
                (self.num_detectors,), 1.0 / self.num_detectors, device=device, dtype=dtype
            )
        )

        return {
            "satisfaction": satisfaction,
            "loss": 1.0 - satisfaction,
            "rule_satisfaction": rule_satisfaction,
            "rule_weights": weights,
            "detector_weights": detector_weights,
        }

    def constraint_loss(
        self, anomaly_prob: torch.Tensor, detector_scores: torch.Tensor
    ) -> torch.Tensor:
        """Return the scalar constraint loss ``1 - satisfaction``.

        Convenience wrapper around :meth:`forward` for the training loop.
        """
        return self.forward(anomaly_prob, detector_scores)["loss"]

    @torch.no_grad()
    def explain(
        self, anomaly_prob: torch.Tensor, detector_scores: torch.Tensor
    ) -> SymbolicExplanation:
        """Human-readable breakdown of rule satisfaction and detector trust.

        Returns a JSON-serialisable dict mapping each rule name to its
        satisfaction and confidence, plus the learned per-detector weights.
        """
        out = self.forward(anomaly_prob, detector_scores)
        rule_sat = [float(v) for v in out["rule_satisfaction"].detach().cpu().tolist()]
        rule_w = [float(v) for v in out["rule_weights"].detach().cpu().tolist()]
        detector_weights = [float(v) for v in out["detector_weights"].detach().cpu().tolist()]
        return {
            "graph": self.rule_graph.name,
            "semantics": self.semantics,
            "satisfaction": float(out["satisfaction"].detach().cpu()),
            "rules": {
                rule.name: {
                    "statement": f"{rule.antecedent} -> {rule.consequent}",
                    "description": rule.description,
                    "satisfaction": float(rule_sat[i]),
                    "confidence": float(rule_w[i]),
                }
                for i, rule in enumerate(self.rule_graph.rules)
            },
            "detector_weights": detector_weights,
        }
