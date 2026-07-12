# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for genetic rule evolution (``omni_mercury_engine.ml.rule_evolution``).

Covers the properties the search's transparency rests on:

1. Operators respect the complexity bounds and are deterministic under a
   seeded generator.
2. Genomes resolve into the existing ``Rule``/``RuleGraph`` representation and
   ``SymbolicConstraintModule`` accepts them unchanged (forward + scoring).
3. Fitness is *real* validation F1 through the canonical evaluation harness
   and never touches a test split.
4. Same seed => identical evolved genome and fitness (end-to-end determinism).
5. Artifacts round-trip losslessly and load through
   ``resolve_rule_graph("evolved:<path>")``.

The fitness/determinism/end-to-end tests run on the real ADBench ``Pima``
dataset (real features, real labels).  The dataset is read from the local
``./data`` cache; when it is not cached, it is fetched only under
``MERCURY_NETWORK_TESTS=1`` (the repository's network-test gate), otherwise
the dependent tests skip with an actionable reason.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

pytest.importorskip("torch")

from pathlib import Path as PathLib

import torch

# The dev venv's editable install may point at a sibling worktree that
# predates ``rule_evolution``; ``unused-ignore`` keeps a correctly
# installed tree (CI) clean.
from omni_mercury_engine.ml.rule_evolution import (  # type: ignore[import-not-found,unused-ignore]
    ChannelStats,
    EvolvedRule,
    EvolvedRuleSearch,
    FitnessDataset,
    GenomeBounds,
    MutationConfig,
    RuleFitnessEvaluator,
    RuleGenome,
    crossover_genomes,
    genome_from_rule_graph,
    load_evolved_artifact,
    mutate_genome,
    random_genome,
    save_evolved_rule_graph,
    tournament_select,
)
from omni_mercury_engine.ml.symbolic_constraint import (  # type: ignore[attr-defined,unused-ignore]
    SymbolicConstraintModule,
    ThresholdAtom,
    consensus_rule_graph,
    parse_evolved_predicate,
    quantize_threshold,
    resolve_rule_graph,
    rule_graph_from_spec,
    rule_graph_to_spec,
)

if TYPE_CHECKING:
    from pathlib import Path

BOUNDS = GenomeBounds(min_rules=1, max_rules=5, max_atoms=3)


# -- real-data fixtures --------------------------------------------------------


@pytest.fixture(scope="module")
def pima_split() -> dict[str, np.ndarray[Any, Any]]:
    """Real ADBench ``Pima`` data: train/val channel matrices + labels.

    Loads the committed recorded-real fixture
    (``tests/fixtures/adbench/pima_real.npz``, the ADBench ``9_Pima`` arrays
    captured 2026-07-09) so the real-data GA tests run in EVERY offline CI
    lane -- they previously skipped on any fresh checkout, hiding the
    fitness/determinism/end-to-end coverage behind a network opt-in. Falls
    back to a cached/downloaded loader copy only if the fixture is missing.

    Features are min-max squashed to ``[0, 1]`` with **train-split statistics
    only** (the channels play the role of per-detector score channels; the
    genome operates over channel indices either way).  The test split is
    deliberately not materialised: fitness must never see it.
    """
    from omni_mercury_engine.evaluation.metrics import split_three_way

    fixture = PathLib(__file__).parents[1] / "fixtures" / "adbench" / "pima_real.npz"
    if fixture.exists():
        with np.load(fixture) as payload:
            features, labels = payload["X"], payload["y"]
    else:
        from omni_mercury_engine.datasets.adbench import ADBenchLoader
        from omni_mercury_engine.datasets.base import DatasetConfig

        loader = ADBenchLoader(
            DatasetConfig(name="adbench", preprocessing={"dataset": "Pima"}, download=False)
        )
        cached = (loader.data_path / loader.npz_filename).exists()
        if not cached:
            if os.environ.get("MERCURY_NETWORK_TESTS", "0") != "1":
                pytest.skip(
                    "committed Pima fixture missing, ADBench NPZ not cached under "
                    "./data, and network tests are disabled "
                    "(set MERCURY_NETWORK_TESTS=1 to fetch it)"
                )
            loader = ADBenchLoader(
                DatasetConfig(name="adbench", preprocessing={"dataset": "Pima"}, download=True)
            )
            loader.download()
        features, labels = loader.load()
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels).astype(int).ravel()

    train_idx, val_idx, _test_idx = split_three_way(
        len(labels), labels, val_frac=0.25, test_frac=0.4, random_state=7
    )
    lo = features[train_idx].min(axis=0)
    hi = features[train_idx].max(axis=0)
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    scale = lambda block: np.clip((block - lo) / span, 0.0, 1.0)  # noqa: E731
    return {
        "scores_train": scale(features[train_idx]).astype(np.float32),
        "y_train": labels[train_idx],
        "scores_val": scale(features[val_idx]).astype(np.float32),
        "y_val": labels[val_idx],
    }


@pytest.fixture(scope="module")
def pima_dataset(pima_split: dict[str, np.ndarray[Any, Any]]) -> FitnessDataset:
    return FitnessDataset(name="Pima", **pima_split)


@pytest.fixture(scope="module")
def pima_stats(pima_split: dict[str, np.ndarray[Any, Any]]) -> ChannelStats:
    return ChannelStats.from_train_scores([pima_split["scores_train"]])


def _stats(n_channels: int = 6) -> ChannelStats:
    """Deterministic channel statistics for pure operator-mechanics tests."""
    rng = np.random.default_rng(123)
    return ChannelStats.from_train_scores([rng.random((256, n_channels))])


# -- genome + grammar ------------------------------------------------------------


class TestGenome:
    """Genome canonicalisation and RuleGraph resolution."""

    def test_rule_canonicalises_atoms(self) -> None:
        a1 = ThresholdAtom(channel=2, op=">=", threshold=quantize_threshold(0.5))
        a2 = ThresholdAtom(channel=0, op="<=", threshold=quantize_threshold(0.25))
        rule_a = EvolvedRule(atoms=(a1, a2, a1), consequent="Anomalous")
        rule_b = EvolvedRule(atoms=(a2, a1), consequent="Anomalous")
        assert rule_a == rule_b
        assert rule_a.n_atoms == 2

    def test_invalid_consequent_rejected(self) -> None:
        atom = ThresholdAtom(channel=0, op=">=", threshold=0.5)
        with pytest.raises(ValueError, match="consequent"):
            EvolvedRule(atoms=(atom,), consequent="Consensus")

    def test_genome_dedupes_rules(self) -> None:
        atom = ThresholdAtom(channel=1, op=">=", threshold=0.75)
        rule = EvolvedRule(atoms=(atom,), consequent="Anomalous")
        genome = RuleGenome(rules=(rule, rule))
        assert len(genome.rules) == 1
        assert genome.complexity == 2  # 1 rule + 1 atom

    def test_predicate_name_round_trips_through_grammar(self) -> None:
        rule = EvolvedRule(
            atoms=(
                "Consensus",
                ThresholdAtom(channel=3, op="<=", threshold=quantize_threshold(1 / 3)),
            ),
            consequent="NotAnomalous",
        )
        atoms = parse_evolved_predicate(rule.predicate_name)
        assert EvolvedRule(atoms=atoms, consequent="NotAnomalous") == rule

    def test_consensus_graph_maps_into_genome_space_and_back(self) -> None:
        graph = consensus_rule_graph()
        genome = genome_from_rule_graph(graph)
        assert {(r.predicate_name, r.consequent) for r in genome.rules} == {
            ("Consensus", "Anomalous"),
            ("NotConsensus", "NotAnomalous"),
        }
        rebuilt = genome.to_rule_graph()
        assert {(r.antecedent, r.consequent) for r in rebuilt.rules} == {
            (r.antecedent, r.consequent) for r in graph.rules
        }

    def test_module_accepts_evolved_graph_forward_and_scoring(self) -> None:
        """An evolved graph plugs into SymbolicConstraintModule unchanged."""
        rng = np.random.default_rng(5)
        genome = random_genome(_stats(), BOUNDS, rng)
        module = SymbolicConstraintModule(num_detectors=6, rule_graph=genome.to_rule_graph())
        torch.manual_seed(0)
        prob = torch.rand(32, 1)
        scores = torch.rand(32, 6)
        out = module(prob, scores)
        satisfaction = float(out["satisfaction"].detach())
        assert np.isfinite(satisfaction) and 0.0 <= satisfaction <= 1.0
        out["loss"].backward()  # co-training path stays autograd-safe
        # score_samples is a real method; under the stale editable install
        # mypy types the attribute as Tensor, hence [operator].
        sample_scores = module.score_samples(scores)  # type: ignore[operator,unused-ignore]
        assert sample_scores.shape == (32,)
        assert torch.all((sample_scores >= 0.0) & (sample_scores <= 1.0))

    def test_out_of_range_channel_fails_loud(self) -> None:
        atom = ThresholdAtom(channel=11, op=">=", threshold=0.5)
        genome = RuleGenome(rules=(EvolvedRule(atoms=(atom,), consequent="Anomalous"),))
        module = SymbolicConstraintModule(num_detectors=3, rule_graph=genome.to_rule_graph())
        with pytest.raises(ValueError, match="channel 11"):
            module.score_samples(torch.rand(8, 3))  # type: ignore[operator,unused-ignore]

    def test_threshold_rule_raises_score_where_it_fires(self) -> None:
        """The evolved scoring path responds to the rule's actual semantics."""
        atom = ThresholdAtom(channel=0, op=">=", threshold=quantize_threshold(0.7))
        genome = RuleGenome(rules=(EvolvedRule(atoms=(atom,), consequent="Anomalous"),))
        module = SymbolicConstraintModule(num_detectors=2, rule_graph=genome.to_rule_graph())
        high = torch.tensor([[0.95, 0.5]])
        low = torch.tensor([[0.05, 0.5]])
        high_score = float(module.score_samples(high))  # type: ignore[operator,unused-ignore]
        low_score = float(module.score_samples(low))  # type: ignore[operator,unused-ignore]
        assert high_score > low_score + 0.3


# -- operators -------------------------------------------------------------------


class TestOperators:
    """Bound preservation and determinism of the genetic operators."""

    def _assert_within_bounds(self, genome: RuleGenome, stats: ChannelStats) -> None:
        assert BOUNDS.min_rules <= len(genome.rules) <= BOUNDS.max_rules
        for rule in genome.rules:
            assert 1 <= rule.n_atoms <= BOUNDS.max_atoms
            for atom in rule.atoms:
                if isinstance(atom, ThresholdAtom):
                    assert 0 <= atom.channel < stats.n_channels
                    assert 0.0 <= atom.threshold <= 1.0
                    assert atom.threshold == quantize_threshold(atom.threshold)
                    assert atom.op in (">=", "<=")

    def test_crossover_preserves_bounds(self) -> None:
        stats = _stats()
        rng = np.random.default_rng(11)
        for _ in range(50):
            parent_a = random_genome(stats, BOUNDS, rng)
            parent_b = random_genome(stats, BOUNDS, rng)
            child_a, child_b = crossover_genomes(parent_a, parent_b, stats, BOUNDS, rng)
            self._assert_within_bounds(child_a, stats)
            self._assert_within_bounds(child_b, stats)

    def test_mutation_respects_complexity_limits(self) -> None:
        stats = _stats()
        rng = np.random.default_rng(13)
        config = MutationConfig(
            p_threshold_jitter=1.0,
            p_op_flip=1.0,
            p_atom_add=1.0,
            p_atom_remove=1.0,
            p_rule_add=1.0,
            p_rule_remove=1.0,
        )
        genome = random_genome(stats, BOUNDS, rng)
        for _ in range(100):
            genome = mutate_genome(genome, stats, BOUNDS, config, rng)
            self._assert_within_bounds(genome, stats)

    def test_operators_deterministic_under_seed(self) -> None:
        stats = _stats()

        def run(seed: int) -> tuple[RuleGenome, ...]:
            rng = np.random.default_rng(seed)
            a = random_genome(stats, BOUNDS, rng)
            b = random_genome(stats, BOUNDS, rng)
            c1, c2 = crossover_genomes(a, b, stats, BOUNDS, rng)
            m = mutate_genome(c1, stats, BOUNDS, MutationConfig(), rng)
            return (a, b, c1, c2, m)

        assert run(42) == run(42)
        assert run(42) != run(43)

    def test_tournament_selects_best_of_contenders(self) -> None:
        fitnesses = [0.1, 0.9, 0.2, 0.5]
        rng = np.random.default_rng(0)
        # With k == population size the tournament must pick the argmax.
        assert tournament_select(fitnesses, k=4, rng=rng) == 1
        with pytest.raises(ValueError, match="empty"):
            tournament_select([], k=2, rng=rng)


# -- fitness ---------------------------------------------------------------------


class TestFitness:
    """Fitness is real validation F1 through the canonical harness."""

    def test_fitness_calls_harness_on_val_only(
        self, pima_dataset: FitnessDataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import omni_mercury_engine.ml.rule_evolution as re_mod

        # The canonical harness functions (the same objects ``rule_evolution``
        # imports); pulling them from ``evaluation.metrics`` keeps their real
        # ``-> float`` signatures visible to mypy.
        from omni_mercury_engine.evaluation.metrics import (
            compute_f1 as real_compute_f1,
            fit_threshold as real_fit_threshold,
        )

        threshold_calls: list[np.ndarray[Any, Any]] = []
        f1_calls: list[np.ndarray[Any, Any]] = []

        def spy_fit_threshold(y_true: Any, y_score: Any, *a: Any, **kw: Any) -> float:
            threshold_calls.append(np.asarray(y_true))
            return real_fit_threshold(y_true, y_score, *a, **kw)

        def spy_compute_f1(y_true: Any, y_pred: Any) -> float:
            f1_calls.append(np.asarray(y_true))
            return real_compute_f1(y_true, y_pred)

        monkeypatch.setattr(re_mod, "fit_threshold", spy_fit_threshold)
        monkeypatch.setattr(re_mod, "compute_f1", spy_compute_f1)

        evaluator = RuleFitnessEvaluator([pima_dataset])
        genome = genome_from_rule_graph(consensus_rule_graph())
        report = evaluator.evaluate(genome)

        assert np.isfinite(report.fitness)
        assert threshold_calls and f1_calls
        # Every harness call operates on exactly the validation labels; the
        # dataset structurally has no test fields, so a test split cannot leak.
        for labels in threshold_calls + f1_calls:
            np.testing.assert_array_equal(labels, pima_dataset.y_val)
        assert not hasattr(pima_dataset, "scores_test")
        assert not hasattr(pima_dataset, "y_test")

    def test_fitness_is_penalised_by_complexity(self, pima_dataset: FitnessDataset) -> None:
        evaluator = RuleFitnessEvaluator([pima_dataset], complexity_penalty=1e-4)
        genome = genome_from_rule_graph(consensus_rule_graph())
        report = evaluator.evaluate(genome)
        assert report.complexity == 4  # 2 rules + 2 atoms
        assert report.fitness == pytest.approx(report.mean_val_f1 - 4e-4)

    def test_evaluator_rejects_out_of_range_genome(self, pima_dataset: FitnessDataset) -> None:
        evaluator = RuleFitnessEvaluator([pima_dataset])
        atom = ThresholdAtom(channel=99, op=">=", threshold=0.5)
        genome = RuleGenome(rules=(EvolvedRule(atoms=(atom,), consequent="Anomalous"),))
        with pytest.raises(ValueError, match="channel 99"):
            evaluator.evaluate(genome)


# -- evolution loop ----------------------------------------------------------------


class TestEvolution:
    """Determinism and transparent progress of the generational loop."""

    @staticmethod
    def _search(
        dataset: FitnessDataset, stats: ChannelStats, seed: int, **kwargs: Any
    ) -> EvolvedRuleSearch:
        return EvolvedRuleSearch(
            RuleFitnessEvaluator([dataset]),
            stats,
            population_size=kwargs.pop("population_size", 10),
            generations=kwargs.pop("generations", 5),
            patience=kwargs.pop("patience", 5),
            bounds=BOUNDS,
            seed=seed,
            **kwargs,
        )

    def test_same_seed_same_result(
        self, pima_dataset: FitnessDataset, pima_stats: ChannelStats
    ) -> None:
        result_a = self._search(pima_dataset, pima_stats, seed=3).run()
        result_b = self._search(pima_dataset, pima_stats, seed=3).run()
        assert result_a.best_genome == result_b.best_genome
        assert result_a.best_report.fitness == result_b.best_report.fitness
        assert result_a.history == result_b.history

    def test_small_end_to_end_on_real_data(
        self, pima_dataset: FitnessDataset, pima_stats: ChannelStats
    ) -> None:
        result = self._search(pima_dataset, pima_stats, seed=17).run()

        assert np.isfinite(result.best_report.fitness)
        assert 0.0 <= result.best_report.mean_val_f1 <= 1.0
        assert 1 <= result.generations_run <= 5
        assert len(result.history) == result.generations_run

        best_curve = [record.best_fitness for record in result.history]
        assert best_curve == sorted(best_curve)  # best-so-far is monotone

        # The evolved champion must score at least the random-init population
        # mean -- a real (val-F1) but cheap assertion.
        assert result.best_report.fitness >= result.history[0].mean_fitness

    def test_seeded_consensus_individual_bounds_fitness_below(
        self, pima_dataset: FitnessDataset, pima_stats: ChannelStats
    ) -> None:
        """Seeding the consensus graph guarantees fitness >= its fitness."""
        evaluator = RuleFitnessEvaluator([pima_dataset])
        consensus_fitness = evaluator.evaluate(
            genome_from_rule_graph(consensus_rule_graph())
        ).fitness
        result = self._search(pima_dataset, pima_stats, seed=23).run()
        assert result.best_report.fitness >= consensus_fitness


# -- serialization + selection seam ---------------------------------------------


class TestSerialization:
    """Artifact round-trips and the resolve_rule_graph('evolved:...') seam."""

    @staticmethod
    def _genome() -> RuleGenome:
        rng = np.random.default_rng(29)
        return random_genome(_stats(), BOUNDS, rng)

    def test_round_trip_is_lossless(self, tmp_path: Path) -> None:
        genome = self._genome()
        path = tmp_path / "evolved.json"
        save_evolved_rule_graph(
            path,
            genome,
            num_channels=6,
            channel_names=[f"det{i}" for i in range(6)],
            provenance={"seed": 29, "datasets": ["unit"], "commit": "test"},
        )
        loaded, payload = load_evolved_artifact(path)
        assert loaded == genome
        assert payload["schema_version"] == 1
        assert payload["provenance"]["seed"] == 29
        assert payload["channel_names"] == [f"det{i}" for i in range(6)]

    def test_resolve_rule_graph_evolved_spec(self, tmp_path: Path) -> None:
        genome = self._genome()
        path = tmp_path / "evolved.json"
        save_evolved_rule_graph(path, genome, graph_name="unit_evolved", num_channels=6)
        graph = resolve_rule_graph(f"evolved:{path}")
        assert graph == genome.to_rule_graph(name="unit_evolved")
        # And the loaded graph is accepted by the module.
        module = SymbolicConstraintModule(num_detectors=6, rule_graph=graph)
        rand_scores = torch.rand(4, 6)
        loaded_scores = module.score_samples(rand_scores)  # type: ignore[operator,unused-ignore]
        assert loaded_scores.shape == (4,)

    def test_resolve_rule_graph_errors(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="requires a path"):
            resolve_rule_graph("evolved:")
        with pytest.raises(FileNotFoundError):
            resolve_rule_graph(f"evolved:{tmp_path / 'missing.json'}")
        with pytest.raises(ValueError, match="unknown rule graph"):
            resolve_rule_graph("bogus")

    def test_schema_version_mismatch_fails_loud(self, tmp_path: Path) -> None:
        import json

        path = tmp_path / "evolved.json"
        save_evolved_rule_graph(path, self._genome(), num_channels=6)
        payload = json.loads(path.read_text())
        payload["schema_version"] = 999
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="schema version"):
            load_evolved_artifact(path)

    def test_rule_graph_spec_round_trip(self) -> None:
        graph = self._genome().to_rule_graph(name="spec_test")
        assert rule_graph_from_spec(rule_graph_to_spec(graph)) == graph

    def test_engine_checkpoint_config_carries_evolved_spec(self) -> None:
        """The engine checkpoint stays self-contained for evolved graphs."""
        from omni_mercury_engine.engine import OmniMercuryEngine

        engine = OmniMercuryEngine(mode="fusion", device="cpu")
        graph = self._genome().to_rule_graph(name="evolved_ckpt")
        engine._symbolic_module = SymbolicConstraintModule(num_detectors=6, rule_graph=graph)
        config = engine._symbolic_checkpoint_config()
        assert config is not None
        assert rule_graph_from_spec(config["rule_graph_spec"]) == graph

        # Registry graphs keep the historical name-only format.
        engine._symbolic_module = SymbolicConstraintModule(
            num_detectors=6, rule_graph=consensus_rule_graph()
        )
        config = engine._symbolic_checkpoint_config()
        assert config is not None
        assert config["rule_graph"] == "consensus"
        assert "rule_graph_spec" not in config


class TestCommittedChampionArtifact:
    """The SHIPPED evolved graph must load and score through the serve path.

    Regression: ``benchmarks/evolved_rule_graph.json`` was committed but no
    test ever loaded it, so a schema drift or a corrupted artifact would
    only surface for the first user of ``evolved:<path>``.
    """

    ARTIFACT = PathLib(__file__).parents[2] / "benchmarks" / "evolved_rule_graph.json"

    def test_champion_loads_via_resolve_seam(self) -> None:
        graph = resolve_rule_graph(f"evolved:{self.ARTIFACT}")
        assert graph.rules, "champion graph has no rules"

    def test_champion_scores_through_deployed_module(self, pima_dataset: FitnessDataset) -> None:
        """Scoring runs through the SAME SymbolicConstraintModule serve path
        the benchmark used, on the real Pima validation split, and produces
        finite, non-constant scores."""
        import json

        payload = json.loads(self.ARTIFACT.read_text())
        n_channels = int(payload["num_channels"])
        graph = resolve_rule_graph(f"evolved:{self.ARTIFACT}")
        module = SymbolicConstraintModule(num_detectors=n_channels, rule_graph=graph)
        scores_val = pima_dataset.scores_val
        if scores_val.shape[1] != n_channels:
            # The champion was evolved on the 12-channel cross-dataset board;
            # pad/trim the Pima channels to the artifact's channel count so
            # the serve path itself is what is exercised.
            reps = -(-n_channels // scores_val.shape[1])
            scores_val = np.tile(scores_val, (1, reps))[:, :n_channels]
        scores_tensor = torch.from_numpy(scores_val.astype(np.float32))
        scored = module.score_samples(scores_tensor)  # type: ignore[operator,unused-ignore]
        scored_np = np.asarray(scored.detach().cpu().numpy(), dtype=np.float64).ravel()
        assert scored_np.shape[0] == scores_val.shape[0]
        assert np.all(np.isfinite(scored_np))
        assert float(scored_np.std()) > 0.0, "champion graph scored a constant"
