# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Genetic evolution of symbolic rule graphs, grounded in held-out detection F1.

This module searches the space of :class:`~omni_mercury_engine.ml.symbolic_constraint.RuleGraph`
rule sets with a genetic algorithm whose fitness is **real validation-split F1
on real labeled data** -- never a synthetic or proxy objective.  The evolved
graphs are plain ``Rule``/``RuleGraph`` data (predicates encoded with the
evolved-predicate grammar of ``symbolic_constraint``), so they plug into
:class:`~omni_mercury_engine.ml.symbolic_constraint.SymbolicConstraintModule`
exactly like the hand-written consensus graphs and are selectable through
``resolve_rule_graph("evolved:<path>")``.

Genome design
    An individual (:class:`RuleGenome`) is an ordered, canonicalised set of
    :class:`EvolvedRule` objects.  Each rule is a conjunction of atoms over the
    engine's *named detector-score channels* (the same ``(B, D)`` consensus
    matrix the symbolic layer reasons over) implying ``Anomalous`` or
    ``NotAnomalous``.  An atom is either a soft threshold test
    (``ThresholdAtom``: channel / ``>=``-or-``<=`` / quantised threshold) or a
    builtin predicate (``Consensus``/``NotConsensus``), so the hand-written
    consensus graph itself lives inside the genome space
    (:func:`genome_from_rule_graph`) and beating it is both possible and
    meaningful.

No train/serve skew (the crux)
    Fitness scores every candidate through
    ``SymbolicConstraintModule(...).score_samples`` -- the module's per-sample
    inference API -- and a *deployed* evolved graph is scored through that
    same method (the module is constructed from the same ``RuleGraph`` data).
    The fixed atom slope, the product t-norm, and the modus-ponens pooling are
    part of the module, not of this search, so the semantics used to select a
    graph are byte-identical to the semantics used to serve it.

Data protocol
    Three-way split (``evaluation.metrics.split_three_way``): train fits the
    unsupervised channel statistics (quantile anchors, IQR jitter scales);
    validation carries the fitness F1 (threshold fit on validation only via
    ``evaluation.metrics.fit_threshold``); the test split is *never seen* by
    this module -- :class:`FitnessDataset` has no test fields, structurally.
    Anti-overfit choice: **multi-dataset evolution** -- fitness is the mean
    validation F1 across all provided datasets (>= 3 in the benchmark), minus
    a small complexity penalty, so a genome must generalise across datasets
    rather than memorise one validation split.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from omni_mercury_engine.evaluation.metrics import compute_f1, fit_threshold
from omni_mercury_engine.ml.symbolic_constraint import (
    Rule,
    RuleGraph,
    SymbolicConstraintModule,
    ThresholdAtom,
    parse_evolved_predicate,
    quantize_threshold,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "RULE_EVOLUTION_SCHEMA_VERSION",
    "ChannelStats",
    "EvolutionResult",
    "EvolvedRule",
    "EvolvedRuleSearch",
    "FitnessDataset",
    "FitnessReport",
    "GenerationRecord",
    "GenomeBounds",
    "MutationConfig",
    "RuleFitnessEvaluator",
    "RuleGenome",
    "crossover_genomes",
    "genome_from_rule_graph",
    "load_evolved_artifact",
    "load_evolved_rule_graph",
    "mutate_genome",
    "random_genome",
    "save_evolved_rule_graph",
    "tournament_select",
]

# Artifact schema version for the evolved-rule-graph JSON format.
RULE_EVOLUTION_SCHEMA_VERSION: int = 1

_ARTIFACT_KIND = "mercury.evolved_rule_graph"

_CONSEQUENTS = ("Anomalous", "NotAnomalous")

# Quantile grid used for threshold anchors (predicate initialisation and
# atom-add mutations draw thresholds from these train-data quantiles).
_QUANTILE_GRID: tuple[float, ...] = tuple(np.linspace(0.05, 0.95, 19).round(4).tolist())

# Floor for the per-channel jitter scale so mutation stays alive on
# near-constant channels (IQR ~ 0).
_MIN_JITTER_SCALE = 0.02


def _atom_sort_key(atom: str | ThresholdAtom) -> tuple[int, str, int, str, float]:
    """Deterministic ordering key over mixed builtin/threshold atoms."""
    if isinstance(atom, str):
        return (0, atom, -1, "", 0.0)
    return (1, "", atom.channel, atom.op, atom.threshold)


@dataclass(frozen=True)
class EvolvedRule:
    """One evolved fuzzy rule: a conjunction of atoms implying a consequent.

    Atoms are canonicalised on construction (deduplicated, deterministically
    sorted) so two rules with the same logical content compare equal and the
    genome hash is stable.

    Attributes:
        atoms: Non-empty tuple of atoms -- builtin predicate names
            (``Consensus``/``NotConsensus``) or :class:`ThresholdAtom` tests.
        consequent: ``"Anomalous"`` or ``"NotAnomalous"``.
    """

    atoms: tuple[str | ThresholdAtom, ...]
    consequent: str

    def __post_init__(self) -> None:
        """Finalize dataclass initialization."""
        if not self.atoms:
            raise ValueError("EvolvedRule requires at least one atom")
        if self.consequent not in _CONSEQUENTS:
            raise ValueError(f"consequent must be one of {_CONSEQUENTS}, got {self.consequent!r}")
        canonical: dict[tuple[int, str, int, str, float], str | ThresholdAtom] = {}
        for atom in self.atoms:
            if isinstance(atom, str):
                # Validate builtin atoms through the canonical grammar parser.
                parse_evolved_predicate(atom)
            elif not isinstance(atom, ThresholdAtom):
                raise ValueError(f"invalid atom type: {type(atom).__name__}")
            canonical[_atom_sort_key(atom)] = atom
        object.__setattr__(self, "atoms", tuple(canonical[key] for key in sorted(canonical)))

    @property
    def predicate_name(self) -> str:
        """Canonical grammar encoding of the antecedent conjunction."""
        return "&".join(atom if isinstance(atom, str) else atom.fragment() for atom in self.atoms)

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the antecedent conjunction."""
        return len(self.atoms)

    @property
    def max_channel(self) -> int:
        """Highest score-channel index referenced (``-1`` if builtin-only)."""
        channels = [a.channel for a in self.atoms if isinstance(a, ThresholdAtom)]
        return max(channels) if channels else -1

    def to_rule(self, index: int) -> Rule:
        """Materialise this evolved rule as a plain :class:`Rule`.

        Args:
            index: Position in the genome; fixes the stable rule name
                ``EV{index}`` (and thereby the learnable weight slot inside
                :class:`SymbolicConstraintModule`).

        Returns:
            The equivalent declarative :class:`Rule`.
        """
        readable = " AND ".join(
            atom if isinstance(atom, str) else atom.fragment() for atom in self.atoms
        )
        return Rule(
            name=f"EV{index}",
            antecedent=self.predicate_name,
            consequent=self.consequent,
            description=f"Evolved: IF {readable} THEN {self.consequent}.",
        )


@dataclass(frozen=True)
class RuleGenome:
    """An individual of the search: a canonicalised set of evolved rules.

    Rules are deduplicated and deterministically ordered on construction, so
    genome equality/hashing is content-based -- the property the determinism
    guarantees (same seed, same final genome) rest on.

    Attributes:
        rules: Non-empty tuple of :class:`EvolvedRule` objects.
    """

    rules: tuple[EvolvedRule, ...]

    def __post_init__(self) -> None:
        """Finalize dataclass initialization."""
        if not self.rules:
            raise ValueError("RuleGenome requires at least one rule")
        canonical: dict[tuple[str, str], EvolvedRule] = {}
        for rule in self.rules:
            if not isinstance(rule, EvolvedRule):
                raise ValueError(f"invalid rule type: {type(rule).__name__}")
            canonical[(rule.predicate_name, rule.consequent)] = rule
        object.__setattr__(self, "rules", tuple(canonical[key] for key in sorted(canonical)))

    @property
    def complexity(self) -> int:
        """Complexity units for regularisation: ``n_rules + total_atoms``."""
        return len(self.rules) + sum(rule.n_atoms for rule in self.rules)

    @property
    def max_channel(self) -> int:
        """Highest score-channel index referenced by any rule (``-1`` if none)."""
        return max((rule.max_channel for rule in self.rules), default=-1)

    def to_rule_graph(self, name: str = "evolved_rules") -> RuleGraph:
        """Resolve the genome into the engine's :class:`RuleGraph` representation.

        Args:
            name: Graph name recorded in diagnostics/checkpoints.

        Returns:
            A :class:`RuleGraph` whose rules encode this genome through the
            evolved-predicate grammar; accepted unchanged by
            :class:`SymbolicConstraintModule`.
        """
        return RuleGraph(
            name=name,
            rules=tuple(rule.to_rule(i) for i, rule in enumerate(self.rules)),
        )


def genome_from_rule_graph(graph: RuleGraph) -> RuleGenome:
    """Map an existing rule graph into the genome space (e.g. as a seed).

    Args:
        graph: Rule graph whose antecedents follow the evolved-predicate
            grammar (builtin ``Consensus``/``NotConsensus`` atoms and/or
            threshold atoms) and whose consequents are
            ``Anomalous``/``NotAnomalous``.  The default
            :func:`~omni_mercury_engine.ml.symbolic_constraint.consensus_rule_graph`
            qualifies.

    Returns:
        The equivalent :class:`RuleGenome`.

    Raises:
        ValueError: If any rule cannot be expressed in the genome space.
    """
    rules = []
    for rule in graph.rules:
        atoms = parse_evolved_predicate(rule.antecedent)
        rules.append(EvolvedRule(atoms=atoms, consequent=rule.consequent))
    return RuleGenome(rules=tuple(rules))


# -- channel statistics (train split only) ------------------------------------


class ChannelStats:
    """Per-channel training statistics anchoring predicate generation.

    Computed from **train-split** detector-score matrices only, so predicate
    initialisation (quantile anchors) and mutation scales (IQR jitter) never
    see validation or test data.

    Attributes:
        n_channels: Number of score channels ``D``.
        quantiles: ``(Q, D)`` channel values at the anchor quantile grid.
        jitter_scale: ``(D,)`` per-channel IQR, floored at a small constant so
            mutation stays alive on near-constant channels.
    """

    def __init__(self, quantiles: np.ndarray[Any, Any], jitter_scale: np.ndarray[Any, Any]):
        """Initialize the instance."""
        quantiles = np.asarray(quantiles, dtype=np.float64)
        jitter_scale = np.asarray(jitter_scale, dtype=np.float64)
        if quantiles.ndim != 2 or quantiles.shape[0] != len(_QUANTILE_GRID):
            raise ValueError(
                f"quantiles must have shape ({len(_QUANTILE_GRID)}, D); " f"got {quantiles.shape}"
            )
        if jitter_scale.shape != (quantiles.shape[1],):
            raise ValueError(
                f"jitter_scale must have shape ({quantiles.shape[1]},); "
                f"got {jitter_scale.shape}"
            )
        if not (np.all(np.isfinite(quantiles)) and np.all(np.isfinite(jitter_scale))):
            raise ValueError("channel statistics must be finite")
        self.quantiles = quantiles
        self.jitter_scale = np.maximum(jitter_scale, _MIN_JITTER_SCALE)
        self.n_channels = int(quantiles.shape[1])

    @classmethod
    def from_train_scores(cls, score_matrices: Sequence[np.ndarray[Any, Any]]) -> ChannelStats:
        """Build channel statistics from one or more train-split score matrices.

        Args:
            score_matrices: Sequence of ``(n_i, D)`` train-split detector-score
                matrices in ``[0, 1]`` (one per dataset; pooled).

        Returns:
            The pooled :class:`ChannelStats`.

        Raises:
            ValueError: If the input is empty, widths disagree, or values are
                non-finite / outside ``[0, 1]``.
        """
        if not score_matrices:
            raise ValueError("at least one train score matrix is required")
        arrays = [np.asarray(m, dtype=np.float64) for m in score_matrices]
        width = arrays[0].shape[1] if arrays[0].ndim == 2 else -1
        for arr in arrays:
            if arr.ndim != 2 or arr.shape[1] != width or arr.shape[0] == 0:
                raise ValueError("all train score matrices must be non-empty 2-D with equal width")
            if not np.all(np.isfinite(arr)):
                raise ValueError("train scores must be finite")
            if arr.min() < 0.0 or arr.max() > 1.0:
                raise ValueError("train scores must lie in [0, 1]")
        pooled = np.vstack(arrays)
        quantiles = np.quantile(pooled, np.asarray(_QUANTILE_GRID), axis=0)
        q75, q25 = np.quantile(pooled, [0.75, 0.25], axis=0)
        return cls(quantiles=quantiles, jitter_scale=q75 - q25)


# -- genome bounds / operator configuration -----------------------------------


@dataclass(frozen=True)
class GenomeBounds:
    """Complexity bounds every operator must respect.

    Attributes:
        min_rules: Minimum number of rules per genome.
        max_rules: Maximum number of rules per genome.
        max_atoms: Maximum atoms per rule antecedent.
        p_builtin_atom: Probability that a freshly generated atom is a builtin
            (``Consensus``/``NotConsensus``) rather than a threshold test.
    """

    min_rules: int = 1
    max_rules: int = 6
    max_atoms: int = 3
    p_builtin_atom: float = 0.15

    def __post_init__(self) -> None:
        """Finalize dataclass initialization."""
        if self.min_rules < 1 or self.max_rules < self.min_rules:
            raise ValueError(
                f"require 1 <= min_rules <= max_rules; got {self.min_rules}, {self.max_rules}"
            )
        if self.max_atoms < 1:
            raise ValueError(f"max_atoms must be >= 1, got {self.max_atoms}")
        if not 0.0 <= self.p_builtin_atom <= 1.0:
            raise ValueError(f"p_builtin_atom must be in [0, 1], got {self.p_builtin_atom}")


@dataclass(frozen=True)
class MutationConfig:
    """Per-genome application probabilities for each mutation operator.

    Operators are applied independently, in a fixed order, each gated by one
    RNG draw -- so a mutated genome is a deterministic function of (genome,
    stats, bounds, config, generator state).

    Attributes:
        p_threshold_jitter: Jitter one threshold by
            ``Normal(0, jitter_scale * channel_IQR)``.
        jitter_scale: Multiplier on the per-channel IQR for the jitter sigma.
        p_op_flip: Flip one atom's comparison operator (``>=`` <-> ``<=``).
        p_atom_add: Add a quantile-anchored atom to one rule (respecting
            ``max_atoms``).
        p_atom_remove: Remove one atom from a multi-atom rule.
        p_rule_add: Add a fresh random rule (respecting ``max_rules``).
        p_rule_remove: Remove one rule (respecting ``min_rules``).
    """

    p_threshold_jitter: float = 0.6
    jitter_scale: float = 0.4
    p_op_flip: float = 0.15
    p_atom_add: float = 0.2
    p_atom_remove: float = 0.15
    p_rule_add: float = 0.15
    p_rule_remove: float = 0.1

    def __post_init__(self) -> None:
        """Finalize dataclass initialization."""
        for name in (
            "p_threshold_jitter",
            "p_op_flip",
            "p_atom_add",
            "p_atom_remove",
            "p_rule_add",
            "p_rule_remove",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if not (math.isfinite(self.jitter_scale) and self.jitter_scale > 0.0):
            raise ValueError(f"jitter_scale must be finite and > 0, got {self.jitter_scale}")


# -- genetic operators (all deterministic under a seeded numpy Generator) -----


def _random_threshold_atom(stats: ChannelStats, rng: np.random.Generator) -> ThresholdAtom:
    """Draw a quantile-anchored threshold atom."""
    channel = int(rng.integers(stats.n_channels))
    op = ">=" if rng.random() < 0.5 else "<="
    anchor = int(rng.integers(len(_QUANTILE_GRID)))
    threshold = quantize_threshold(float(stats.quantiles[anchor, channel]))
    return ThresholdAtom(channel=channel, op=op, threshold=threshold)


def _random_atom(
    stats: ChannelStats, bounds: GenomeBounds, rng: np.random.Generator
) -> str | ThresholdAtom:
    """Draw one atom: builtin with probability ``p_builtin_atom``, else threshold."""
    if rng.random() < bounds.p_builtin_atom:
        return "Consensus" if rng.random() < 0.5 else "NotConsensus"
    return _random_threshold_atom(stats, rng)


def _random_rule(
    stats: ChannelStats, bounds: GenomeBounds, rng: np.random.Generator
) -> EvolvedRule:
    """Draw one random rule within the complexity bounds."""
    n_atoms = int(rng.integers(1, bounds.max_atoms + 1))
    atoms = tuple(_random_atom(stats, bounds, rng) for _ in range(n_atoms))
    # Bias towards evidence (recall) rules; precision rules remain reachable.
    consequent = "Anomalous" if rng.random() < 0.7 else "NotAnomalous"
    return EvolvedRule(atoms=atoms, consequent=consequent)


def random_genome(
    stats: ChannelStats, bounds: GenomeBounds, rng: np.random.Generator
) -> RuleGenome:
    """Draw a random genome with quantile-anchored predicates.

    Args:
        stats: Train-split channel statistics.
        bounds: Complexity bounds.
        rng: Seeded numpy Generator.

    Returns:
        A genome with ``min_rules..max_rules`` random rules.
    """
    n_rules = int(rng.integers(bounds.min_rules, bounds.max_rules + 1))
    rules = tuple(_random_rule(stats, bounds, rng) for _ in range(n_rules))
    return _finalize(list(rules), stats, bounds, rng)


def _finalize(
    rules: list[EvolvedRule],
    stats: ChannelStats,
    bounds: GenomeBounds,
    rng: np.random.Generator,
) -> RuleGenome:
    """Construct a bound-respecting genome from a raw rule list (with repair).

    Deduplication inside :class:`RuleGenome` can shrink the rule count, and
    crossover can exceed ``max_rules``; this repairs both deterministically.
    """
    if len(rules) > bounds.max_rules:
        keep = sorted(rng.choice(len(rules), size=bounds.max_rules, replace=False).tolist())
        rules = [rules[i] for i in keep]
    genome = RuleGenome(rules=tuple(rules))
    attempts = 0
    while len(genome.rules) < bounds.min_rules:
        attempts += 1
        if attempts > 100:
            raise RuntimeError(
                "could not repair genome to min_rules; bounds too tight for the "
                "channel statistics"
            )
        genome = RuleGenome(rules=(*genome.rules, _random_rule(stats, bounds, rng)))
    return genome


def tournament_select(fitnesses: Sequence[float], k: int, rng: np.random.Generator) -> int:
    """Select one individual index by size-``k`` tournament.

    Args:
        fitnesses: Fitness value per population member.
        k: Tournament size (clipped to the population size).
        rng: Seeded numpy Generator.

    Returns:
        Index of the tournament winner (ties broken by first draw order --
        deterministic).

    Raises:
        ValueError: If the population is empty or ``k < 1``.
    """
    n = len(fitnesses)
    if n == 0:
        raise ValueError("cannot select from an empty population")
    if k < 1:
        raise ValueError(f"tournament size must be >= 1, got {k}")
    contenders = rng.choice(n, size=min(k, n), replace=False)
    values = np.asarray([fitnesses[int(i)] for i in contenders], dtype=np.float64)
    return int(contenders[int(np.argmax(values))])


def crossover_genomes(
    a: RuleGenome,
    b: RuleGenome,
    stats: ChannelStats,
    bounds: GenomeBounds,
    rng: np.random.Generator,
    *,
    p_single_point: float = 0.3,
) -> tuple[RuleGenome, RuleGenome]:
    """Rule-set exchange crossover.

    With probability ``p_single_point`` a single-point crossover on the
    (canonically sorted) rule lists is used; otherwise a uniform per-rule swap.
    Both children are repaired into the complexity bounds.

    Args:
        a: First parent.
        b: Second parent.
        stats: Train-split channel statistics (for bound repair).
        bounds: Complexity bounds.
        rng: Seeded numpy Generator.
        p_single_point: Probability of single-point (vs uniform) exchange.

    Returns:
        Two child genomes within bounds.
    """
    rules_a, rules_b = list(a.rules), list(b.rules)
    child1: list[EvolvedRule] = []
    child2: list[EvolvedRule] = []
    if rng.random() < p_single_point:
        # Genomes store rules canonically sorted, so cut points are stable.
        cut_a = int(rng.integers(0, len(rules_a) + 1))
        cut_b = int(rng.integers(0, len(rules_b) + 1))
        child1 = rules_a[:cut_a] + rules_b[cut_b:]
        child2 = rules_b[:cut_b] + rules_a[cut_a:]
    else:
        for i in range(max(len(rules_a), len(rules_b))):
            rule_a = rules_a[i] if i < len(rules_a) else None
            rule_b = rules_b[i] if i < len(rules_b) else None
            if rng.random() < 0.5:
                rule_a, rule_b = rule_b, rule_a
            if rule_a is not None:
                child1.append(rule_a)
            if rule_b is not None:
                child2.append(rule_b)
    return (
        _finalize(child1, stats, bounds, rng),
        _finalize(child2, stats, bounds, rng),
    )


def mutate_genome(
    genome: RuleGenome,
    stats: ChannelStats,
    bounds: GenomeBounds,
    config: MutationConfig,
    rng: np.random.Generator,
) -> RuleGenome:
    """Apply the mutation operators to a genome, respecting all bounds.

    Operators (fixed order, each independently gated by one RNG draw):
    threshold jitter scaled to the channel IQR, comparison-operator flip,
    atom add (quantile-anchored) / atom remove, rule add / rule remove.

    Args:
        genome: Parent genome.
        stats: Train-split channel statistics.
        bounds: Complexity bounds.
        config: Mutation probabilities and scales.
        rng: Seeded numpy Generator.

    Returns:
        The mutated genome (within bounds; possibly equal to the parent when
        no operator fires).
    """
    rules = list(genome.rules)

    def _threshold_slots() -> list[tuple[int, int]]:
        return [
            (i, j)
            for i, rule in enumerate(rules)
            for j, atom in enumerate(rule.atoms)
            if isinstance(atom, ThresholdAtom)
        ]

    def _replace_atom(rule_idx: int, atom_idx: int, atom: ThresholdAtom) -> None:
        rule = rules[rule_idx]
        atoms = list(rule.atoms)
        atoms[atom_idx] = atom
        rules[rule_idx] = EvolvedRule(atoms=tuple(atoms), consequent=rule.consequent)

    # 1. Threshold jitter, scaled to the channel's train IQR.
    if rng.random() < config.p_threshold_jitter:
        slots = _threshold_slots()
        if slots:
            rule_idx, atom_idx = slots[int(rng.integers(len(slots)))]
            atom = rules[rule_idx].atoms[atom_idx]
            assert isinstance(atom, ThresholdAtom)
            sigma = config.jitter_scale * float(stats.jitter_scale[atom.channel])
            jittered = quantize_threshold(atom.threshold + float(rng.normal(0.0, sigma)))
            _replace_atom(rule_idx, atom_idx, replace(atom, threshold=jittered))

    # 2. Comparison-operator flip.
    if rng.random() < config.p_op_flip:
        slots = _threshold_slots()
        if slots:
            rule_idx, atom_idx = slots[int(rng.integers(len(slots)))]
            atom = rules[rule_idx].atoms[atom_idx]
            assert isinstance(atom, ThresholdAtom)
            flipped = ">=" if atom.op == "<=" else "<="
            _replace_atom(rule_idx, atom_idx, replace(atom, op=flipped))

    # 3. Atom add (complexity-bounded).
    if rng.random() < config.p_atom_add:
        candidates = [i for i, rule in enumerate(rules) if rule.n_atoms < bounds.max_atoms]
        if candidates:
            rule_idx = candidates[int(rng.integers(len(candidates)))]
            rule = rules[rule_idx]
            rules[rule_idx] = EvolvedRule(
                atoms=(*rule.atoms, _random_atom(stats, bounds, rng)),
                consequent=rule.consequent,
            )

    # 4. Atom remove (keep at least one atom).
    if rng.random() < config.p_atom_remove:
        candidates = [i for i, rule in enumerate(rules) if rule.n_atoms > 1]
        if candidates:
            rule_idx = candidates[int(rng.integers(len(candidates)))]
            rule = rules[rule_idx]
            drop = int(rng.integers(rule.n_atoms))
            atoms = tuple(atom for j, atom in enumerate(rule.atoms) if j != drop)
            rules[rule_idx] = EvolvedRule(atoms=atoms, consequent=rule.consequent)

    # 5. Rule add (complexity-bounded).
    if rng.random() < config.p_rule_add and len(rules) < bounds.max_rules:
        rules.append(_random_rule(stats, bounds, rng))

    # 6. Rule remove (keep at least min_rules).
    if rng.random() < config.p_rule_remove and len(rules) > bounds.min_rules:
        rules.pop(int(rng.integers(len(rules))))

    return _finalize(rules, stats, bounds, rng)


# -- fitness: real held-out validation F1 --------------------------------------


@dataclass(frozen=True)
class FitnessDataset:
    """Train/validation detector-score matrices for one real labeled dataset.

    Deliberately has **no test fields**: the fitness pipeline structurally
    cannot touch a test split.  Train data anchors unsupervised statistics
    only; the fitness F1 is computed on the validation split with the
    threshold fit on that same validation split (the canonical
    ``fit_threshold`` protocol -- train and test stay out of threshold
    selection and fitness alike).

    Attributes:
        name: Dataset identifier (provenance).
        scores_train: ``(n_train, D)`` train-split detector scores in [0, 1].
        y_train: ``(n_train,)`` binary train labels.
        scores_val: ``(n_val, D)`` validation-split detector scores in [0, 1].
        y_val: ``(n_val,)`` binary validation labels (must contain both
            classes, otherwise F1 fitness is degenerate).
    """

    name: str
    scores_train: np.ndarray[Any, Any]
    y_train: np.ndarray[Any, Any]
    scores_val: np.ndarray[Any, Any]
    y_val: np.ndarray[Any, Any]

    def __post_init__(self) -> None:
        """Finalize dataclass initialization."""
        for label, scores, y in (
            ("train", self.scores_train, self.y_train),
            ("val", self.scores_val, self.y_val),
        ):
            scores = np.asarray(scores, dtype=np.float32)
            y = np.asarray(y).astype(int).reshape(-1)
            if scores.ndim != 2 or scores.shape[0] == 0:
                raise ValueError(f"{self.name}: {label} scores must be non-empty 2-D")
            if scores.shape[0] != y.shape[0]:
                raise ValueError(
                    f"{self.name}: {label} scores/labels length mismatch "
                    f"({scores.shape[0]} != {y.shape[0]})"
                )
            if not np.all(np.isfinite(scores)):
                raise ValueError(f"{self.name}: {label} scores must be finite")
            if scores.min() < 0.0 or scores.max() > 1.0:
                raise ValueError(f"{self.name}: {label} scores must lie in [0, 1]")
            if not np.all(np.isin(y, (0, 1))):
                raise ValueError(f"{self.name}: {label} labels must be binary")
            object.__setattr__(self, f"scores_{label}", scores)
            object.__setattr__(self, f"y_{label}", y)
        if self.scores_train.shape[1] != self.scores_val.shape[1]:
            raise ValueError(f"{self.name}: train/val channel width mismatch")
        if len(np.unique(self.y_val)) < 2:
            raise ValueError(
                f"{self.name}: validation split must contain both classes for F1 fitness"
            )

    @property
    def n_channels(self) -> int:
        """Number of detector-score channels ``D``."""
        return int(self.scores_train.shape[1])


@dataclass(frozen=True)
class FitnessReport:
    """Fitness breakdown for one genome.

    Attributes:
        fitness: ``mean_val_f1 - complexity_penalty * complexity`` -- the
            quantity the search maximises.
        mean_val_f1: Mean validation F1 across the fitness datasets.
        per_dataset_f1: Validation F1 per dataset name.
        complexity: Genome complexity units (``n_rules + total_atoms``).
    """

    fitness: float
    mean_val_f1: float
    per_dataset_f1: Mapping[str, float]
    complexity: int


class RuleFitnessEvaluator:
    """Fitness = mean held-out validation F1 through the deployed scoring path.

    Every candidate genome is resolved into a :class:`RuleGraph` and scored on
    each dataset's validation split via
    ``SymbolicConstraintModule(num_detectors=D, rule_graph=graph).score_samples``
    -- the module's per-sample inference API and the exact function a deployed
    evolved graph scores with (no train/serve skew).  The operating threshold
    is fit on the validation split only (``evaluation.metrics.fit_threshold``)
    and the F1 at that threshold on the same validation split is the per-
    dataset fitness contribution.  Test data never enters this class.

    Complexity regularisation: ``fitness = mean_val_f1 - penalty * complexity``
    with a default penalty of ``1e-4`` per unit -- a 3-atom rule (4 units)
    costs 0.0004 F1, well below any meaningful F1 gain (the repository's
    adoption bars sit at ~0.002), so parsimony only breaks ties.

    Args:
        datasets: One or more :class:`FitnessDataset` (>= 3 in the benchmark
            for the multi-dataset anti-overfit protocol); all must share the
            channel width.
        complexity_penalty: Non-negative fitness penalty per complexity unit.
    """

    def __init__(
        self,
        datasets: Sequence[FitnessDataset],
        *,
        complexity_penalty: float = 1e-4,
    ) -> None:
        """Initialize the instance."""
        if not datasets:
            raise ValueError("at least one fitness dataset is required")
        widths = {ds.n_channels for ds in datasets}
        if len(widths) != 1:
            raise ValueError(f"datasets disagree on channel width: {sorted(widths)}")
        if not math.isfinite(complexity_penalty) or complexity_penalty < 0.0:
            raise ValueError(
                f"complexity_penalty must be finite and >= 0, got {complexity_penalty}"
            )
        self.datasets = tuple(datasets)
        self.num_channels = int(widths.pop())
        self.complexity_penalty = float(complexity_penalty)
        self._val_tensors = tuple(
            torch.as_tensor(ds.scores_val, dtype=torch.float32) for ds in self.datasets
        )
        self._cache: dict[RuleGenome, FitnessReport] = {}

    def evaluate(self, genome: RuleGenome) -> FitnessReport:
        """Evaluate one genome's fitness (cached by genome content).

        Args:
            genome: Candidate genome; must not reference channels beyond the
                datasets' width.

        Returns:
            The :class:`FitnessReport` for this genome.

        Raises:
            ValueError: If the genome references an out-of-range channel.
        """
        cached = self._cache.get(genome)
        if cached is not None:
            return cached
        if genome.max_channel >= self.num_channels:
            raise ValueError(
                f"genome references channel {genome.max_channel} but datasets "
                f"have only {self.num_channels} channels"
            )
        graph = genome.to_rule_graph()
        per_dataset: dict[str, float] = {}
        for ds, val_tensor in zip(self.datasets, self._val_tensors, strict=True):
            # Deployed scoring path: the module's parameters initialise
            # deterministically (uniform rule weights), so this is a pure
            # function of (graph, scores).
            module = SymbolicConstraintModule(num_detectors=self.num_channels, rule_graph=graph)
            scores_val = module.score_samples(val_tensor).numpy()
            threshold = fit_threshold(ds.y_val, scores_val)
            f1 = compute_f1(ds.y_val, (scores_val >= threshold).astype(int))
            per_dataset[ds.name] = float(f1)
        mean_f1 = float(np.mean(list(per_dataset.values())))
        report = FitnessReport(
            fitness=mean_f1 - self.complexity_penalty * genome.complexity,
            mean_val_f1=mean_f1,
            per_dataset_f1=per_dataset,
            complexity=genome.complexity,
        )
        self._cache[genome] = report
        return report


# -- evolution loop -------------------------------------------------------------


@dataclass(frozen=True)
class GenerationRecord:
    """Per-generation history entry.

    Attributes:
        generation: 0-based generation index.
        best_fitness: Best-so-far fitness after this generation (monotone
            non-decreasing across the run by construction).
        mean_fitness: Mean fitness of this generation's population.
        best_val_f1: Mean validation F1 of the best-so-far genome.
    """

    generation: int
    best_fitness: float
    mean_fitness: float
    best_val_f1: float


@dataclass(frozen=True)
class EvolutionResult:
    """Outcome of one :class:`EvolvedRuleSearch` run.

    Attributes:
        best_genome: Highest-fitness genome found.
        best_report: Its fitness breakdown.
        history: Per-generation records (see :class:`GenerationRecord`).
        generations_run: Number of generations actually executed.
        stopped_early: Whether the stagnation patience triggered.
        seed: The seed the run was started with.
    """

    best_genome: RuleGenome
    best_report: FitnessReport
    history: tuple[GenerationRecord, ...]
    generations_run: int
    stopped_early: bool
    seed: int


class EvolvedRuleSearch:
    """Deterministic genetic search over rule genomes.

    Population initialisation seeds the search with the provided genomes
    (by default the hand-written consensus graph, so beating it is a
    like-for-like comparison) and fills the rest with random quantile-anchored
    genomes.  Each generation: elitism carries the top ``elitism`` individuals
    unchanged; the remainder is bred by tournament selection, rule-set
    crossover, and mutation.  The run stops early after ``patience``
    generations without best-fitness improvement.

    End-to-end determinism: all stochastic choices flow through one
    ``numpy.random.Generator`` seeded with ``seed``; genomes are canonical and
    the evaluator is a pure function of genome content, so the same seed
    yields the identical final genome and history.

    Args:
        evaluator: Fitness evaluator (real validation F1).
        stats: Train-split channel statistics for predicate generation.
        population_size: Individuals per generation (>= 4).
        generations: Maximum generations (>= 1).
        tournament_k: Tournament size for selection.
        elitism: Number of top individuals carried over unchanged.
        patience: Generations without best-fitness improvement before early
            stop.
        crossover_rate: Probability a breeding pair is crossed (else cloned).
        bounds: Genome complexity bounds.
        mutation: Mutation operator configuration.
        seed: RNG seed for the whole run.
        seed_genomes: Initial individuals injected into the population;
            defaults to the consensus-graph genome.
    """

    def __init__(
        self,
        evaluator: RuleFitnessEvaluator,
        stats: ChannelStats,
        *,
        population_size: int = 40,
        generations: int = 30,
        tournament_k: int = 3,
        elitism: int = 2,
        patience: int = 8,
        crossover_rate: float = 0.7,
        bounds: GenomeBounds | None = None,
        mutation: MutationConfig | None = None,
        seed: int = 0,
        seed_genomes: Sequence[RuleGenome] | None = None,
    ) -> None:
        """Initialize the instance."""
        if population_size < 4:
            raise ValueError(f"population_size must be >= 4, got {population_size}")
        if generations < 1:
            raise ValueError(f"generations must be >= 1, got {generations}")
        if not 0 <= elitism < population_size:
            raise ValueError(f"elitism must be in [0, population_size), got {elitism}")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if tournament_k < 1:
            raise ValueError(f"tournament_k must be >= 1, got {tournament_k}")
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError(f"crossover_rate must be in [0, 1], got {crossover_rate}")
        if stats.n_channels != evaluator.num_channels:
            raise ValueError(
                f"channel statistics width {stats.n_channels} != evaluator "
                f"width {evaluator.num_channels}"
            )
        self.evaluator = evaluator
        self.stats = stats
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.tournament_k = int(tournament_k)
        self.elitism = int(elitism)
        self.patience = int(patience)
        self.crossover_rate = float(crossover_rate)
        self.bounds = bounds if bounds is not None else GenomeBounds()
        self.mutation = mutation if mutation is not None else MutationConfig()
        self.seed = int(seed)
        if seed_genomes is None:
            from omni_mercury_engine.ml.symbolic_constraint import consensus_rule_graph

            seed_genomes = (genome_from_rule_graph(consensus_rule_graph()),)
        for genome in seed_genomes:
            if genome.max_channel >= evaluator.num_channels:
                raise ValueError(
                    f"seed genome references channel {genome.max_channel} but "
                    f"datasets have only {evaluator.num_channels} channels"
                )
        self.seed_genomes = tuple(seed_genomes)

    def run(self) -> EvolutionResult:
        """Execute the generational loop.

        Returns:
            The :class:`EvolutionResult` with the best genome, its fitness
            breakdown, and the full per-generation history.
        """
        rng = np.random.default_rng(self.seed)
        population = list(self.seed_genomes[: self.population_size])
        while len(population) < self.population_size:
            population.append(random_genome(self.stats, self.bounds, rng))

        best_genome: RuleGenome | None = None
        best_report: FitnessReport | None = None
        history: list[GenerationRecord] = []
        stagnation = 0
        stopped_early = False
        generations_run = 0

        for generation in range(self.generations):
            generations_run = generation + 1
            reports = [self.evaluator.evaluate(genome) for genome in population]
            fitnesses = [report.fitness for report in reports]
            gen_best = int(np.argmax(np.asarray(fitnesses, dtype=np.float64)))
            improved = best_report is None or fitnesses[gen_best] > best_report.fitness
            if improved:
                best_genome = population[gen_best]
                best_report = reports[gen_best]
                stagnation = 0
            else:
                stagnation += 1
            assert best_report is not None and best_genome is not None
            history.append(
                GenerationRecord(
                    generation=generation,
                    best_fitness=best_report.fitness,
                    mean_fitness=float(np.mean(fitnesses)),
                    best_val_f1=best_report.mean_val_f1,
                )
            )
            if stagnation >= self.patience:
                stopped_early = True
                break
            if generation == self.generations - 1:
                break

            # Elitism: stable ordering (fitness desc, then population index).
            order = sorted(range(len(population)), key=lambda i: (-fitnesses[i], i))
            next_population: list[RuleGenome] = [population[i] for i in order[: self.elitism]]
            while len(next_population) < self.population_size:
                parent_a = population[tournament_select(fitnesses, self.tournament_k, rng)]
                parent_b = population[tournament_select(fitnesses, self.tournament_k, rng)]
                if rng.random() < self.crossover_rate:
                    child_a, child_b = crossover_genomes(
                        parent_a, parent_b, self.stats, self.bounds, rng
                    )
                else:
                    child_a, child_b = parent_a, parent_b
                child_a = mutate_genome(child_a, self.stats, self.bounds, self.mutation, rng)
                child_b = mutate_genome(child_b, self.stats, self.bounds, self.mutation, rng)
                next_population.append(child_a)
                if len(next_population) < self.population_size:
                    next_population.append(child_b)
            population = next_population

        assert best_genome is not None and best_report is not None
        return EvolutionResult(
            best_genome=best_genome,
            best_report=best_report,
            history=tuple(history),
            generations_run=generations_run,
            stopped_early=stopped_early,
            seed=self.seed,
        )


# -- artifact serialization ------------------------------------------------------


def _atom_to_json(atom: str | ThresholdAtom) -> dict[str, Any]:
    """Encode one atom as a JSON-compatible dict."""
    if isinstance(atom, str):
        return {"kind": "builtin", "predicate": atom}
    return {
        "kind": "threshold",
        "channel": atom.channel,
        "op": atom.op,
        "threshold": atom.threshold,
    }


def _atom_from_json(entry: Mapping[str, Any]) -> str | ThresholdAtom:
    """Decode one atom from its JSON dict (fail loud on unknown kinds)."""
    kind = entry.get("kind")
    if kind == "builtin":
        return str(entry["predicate"])
    if kind == "threshold":
        return ThresholdAtom(
            channel=int(entry["channel"]),
            op=str(entry["op"]),
            threshold=quantize_threshold(float(entry["threshold"])),
        )
    raise ValueError(f"unknown atom kind in evolved-graph artifact: {kind!r}")


def save_evolved_rule_graph(
    path: str | Path,
    genome: RuleGenome,
    *,
    graph_name: str = "evolved_rules",
    num_channels: int,
    channel_names: Sequence[str] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Write an evolved rule graph to a schema-versioned JSON artifact.

    The artifact is self-contained and lossless: :func:`load_evolved_artifact`
    reconstructs the identical genome, and
    ``resolve_rule_graph("evolved:<path>")`` loads the identical graph.

    Args:
        path: Output file path (parent directories are created).
        genome: The evolved genome to persist.
        graph_name: Name recorded on the reconstructed :class:`RuleGraph`.
        num_channels: Score-channel width the genome was evolved over.
        channel_names: Optional channel names (index ``i`` documents what
            ``x{i}`` means -- e.g. the engine's consensus score channels).
        provenance: Free-form provenance (datasets, seed, commit, fitness
            history, ...), stored verbatim.

    Returns:
        The written path.

    Raises:
        ValueError: If the genome references channels outside
            ``[0, num_channels)`` or the channel names disagree with the width.
    """
    if num_channels < 1:
        raise ValueError(f"num_channels must be >= 1, got {num_channels}")
    if genome.max_channel >= num_channels:
        raise ValueError(
            f"genome references channel {genome.max_channel} but num_channels " f"is {num_channels}"
        )
    names = list(channel_names) if channel_names is not None else None
    if names is not None and len(names) != num_channels:
        raise ValueError(f"channel_names length {len(names)} != num_channels {num_channels}")
    payload = {
        "schema_version": RULE_EVOLUTION_SCHEMA_VERSION,
        "kind": _ARTIFACT_KIND,
        "graph_name": str(graph_name),
        "num_channels": int(num_channels),
        "channel_names": names,
        "rules": [
            {
                "atoms": [_atom_to_json(atom) for atom in rule.atoms],
                "consequent": rule.consequent,
            }
            for rule in genome.rules
        ],
        "provenance": dict(provenance or {}),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out


def load_evolved_artifact(path: str | Path) -> tuple[RuleGenome, dict[str, Any]]:
    """Load an evolved-rule-graph artifact, validating its schema.

    Args:
        path: Path to a JSON artifact written by
            :func:`save_evolved_rule_graph`.

    Returns:
        ``(genome, payload)`` -- the reconstructed genome and the full parsed
        artifact payload (for provenance/channel names).

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the artifact is malformed, of an unknown kind, or of an
            unsupported schema version.
    """
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"evolved rule-graph artifact not found: {artifact_path}")
    try:
        payload = json.loads(artifact_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"evolved rule-graph artifact is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("evolved rule-graph artifact must be a JSON object")
    if payload.get("kind") != _ARTIFACT_KIND:
        raise ValueError(
            f"unexpected artifact kind {payload.get('kind')!r}; expected {_ARTIFACT_KIND!r}"
        )
    version = payload.get("schema_version")
    if version != RULE_EVOLUTION_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported evolved-graph schema version {version!r}; this build "
            f"supports {RULE_EVOLUTION_SCHEMA_VERSION}"
        )
    rule_entries = payload.get("rules")
    if not isinstance(rule_entries, list) or not rule_entries:
        raise ValueError("evolved rule-graph artifact must contain a non-empty 'rules' list")
    rules = tuple(
        EvolvedRule(
            atoms=tuple(_atom_from_json(atom) for atom in entry["atoms"]),
            consequent=str(entry["consequent"]),
        )
        for entry in rule_entries
    )
    return RuleGenome(rules=rules), payload


def load_evolved_rule_graph(path: str | Path) -> RuleGraph:
    """Load an evolved artifact and resolve it into a :class:`RuleGraph`.

    This is the loader behind ``resolve_rule_graph("evolved:<path>")``.

    Args:
        path: Path to a JSON artifact written by
            :func:`save_evolved_rule_graph`.

    Returns:
        The reconstructed rule graph, named per the artifact's ``graph_name``.
    """
    genome, payload = load_evolved_artifact(path)
    return genome.to_rule_graph(name=str(payload.get("graph_name", "evolved_rules")))
