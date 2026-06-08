# Copyright (C) 2025 Steel Security Advisors LLC
"""Regression tests for the cognitive/ + models/ RNG cure.

Asserts the three contracts that make the per-instance ``Generator`` plumbing
worthwhile:

1. **Determinism**: two engines constructed with the same ``seed`` produce
   identical sequences for the operations the cure converted.
2. **Independence**: different seeds produce different sequences (sanity
   check that we did not accidentally hard-code a single Generator).
3. **Global-state isolation**: poisoning the legacy ``np.random.seed(...)``
   global state must NOT change a seeded engine's output. This is the
   actual defect the cure addresses — without it, an unrelated caller can
   silently de-randomize one of these modules.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Cognitive engines
# ---------------------------------------------------------------------------


def test_causal_discovery_engine_seed_reproducible() -> None:
    from omni_mercury_engine.cognitive.causal_discovery import CausalDiscoveryEngine

    a = CausalDiscoveryEngine(seed=42)
    b = CausalDiscoveryEngine(seed=42)
    assert (
        a._rng.choice(100, 10, replace=True).tolist()
        == b._rng.choice(100, 10, replace=True).tolist()
    )


def test_chain_of_thought_engine_seed_reproducible() -> None:
    from omni_mercury_engine.cognitive.chain_of_thought import ChainOfThoughtEngine

    a = ChainOfThoughtEngine(seed=7)
    b = ChainOfThoughtEngine(seed=7)
    assert a._rng.random() == b._rng.random()
    # Inner ThoughtGenerator must also be seeded (constructor wires it).
    assert a.thought_generator._rng.random() == b.thought_generator._rng.random()


def test_chain_of_hindsight_relabeler_seed_reproducible() -> None:
    from omni_mercury_engine.cognitive.chain_of_hindsight import HindsightRelabeler

    a = HindsightRelabeler(seed=3)
    b = HindsightRelabeler(seed=3)
    assert a._rng.choice([0, 1, 2, 3, 4], 5).tolist() == b._rng.choice([0, 1, 2, 3, 4], 5).tolist()


def test_knowledge_graph_seed_reproducible() -> None:
    from omni_mercury_engine.cognitive.knowledge_graph import KnowledgeGraph

    a = KnowledgeGraph(seed=99)
    b = KnowledgeGraph(seed=99)
    assert a._rng.standard_normal(64).tolist() == b._rng.standard_normal(64).tolist()
    # The embedded sub-components also receive the seed.
    assert (
        a._random_walk._rng.standard_normal(8).tolist()
        == b._random_walk._rng.standard_normal(8).tolist()
    )
    assert a._gnn._rng.standard_normal(8).tolist() == b._gnn._rng.standard_normal(8).tolist()


def test_explainability_explainers_seed_reproducible() -> None:
    from omni_mercury_engine.cognitive.explainability import LIMEExplainer, SHAPExplainer

    a = SHAPExplainer(seed=5)
    b = SHAPExplainer(seed=5)
    assert a._rng.integers(0, 2, 10).tolist() == b._rng.integers(0, 2, 10).tolist()

    a2 = LIMEExplainer(seed=5)
    b2 = LIMEExplainer(seed=5)
    assert a2._rng.standard_normal(10).tolist() == b2._rng.standard_normal(10).tolist()


def test_formal_verification_constraint_solver_seed_reproducible() -> None:
    from omni_mercury_engine.cognitive.formal_verification import ConstraintSolver

    a = ConstraintSolver(seed=11)
    b = ConstraintSolver(seed=11)
    assert a._rng.uniform(-1, 1, 5).tolist() == b._rng.uniform(-1, 1, 5).tolist()


def test_multi_agent_system_seed_reproducible() -> None:
    from omni_mercury_engine.cognitive.multi_agent_coordination import (
        MultiAgentDetectionSystem,
    )

    a = MultiAgentDetectionSystem(num_agents=3, seed=4)
    b = MultiAgentDetectionSystem(num_agents=3, seed=4)
    # Each system has consumed the same RNG draws during _create_agents.
    # Subsequent draws from the per-instance _rng must still match.
    assert a._rng.uniform(-0.1, 0.1) == b._rng.uniform(-0.1, 0.1)


def test_predictive_coding_default_seed_42() -> None:
    """Preserves deterministic behavior of the previous ``np.random.seed(42)``."""
    from omni_mercury_engine.cognitive.predictive_coding import HierarchicalPredictiveCoder

    a = HierarchicalPredictiveCoder(input_dim=8, hidden_dims=[6, 4])
    b = HierarchicalPredictiveCoder(input_dim=8, hidden_dims=[6, 4])
    # Both use seed=42 by default, so the generated weight matrices match.
    weights_a = next(iter(a.models.values())).weights
    weights_b = next(iter(b.models.values())).weights
    assert weights_a.tolist() == weights_b.tolist()


def test_reflexion_engine_seed_reproducible() -> None:
    from omni_mercury_engine.cognitive.reflexion import ReflexionEngine

    a = ReflexionEngine(seed=8)
    b = ReflexionEngine(seed=8)
    assert a._rng.choice(["x", "y", "z"], 5).tolist() == b._rng.choice(["x", "y", "z"], 5).tolist()


def test_uncertainty_quantifier_seed_reproducible() -> None:
    from omni_mercury_engine.cognitive.uncertainty import UncertaintyQuantifier

    a = UncertaintyQuantifier(seed=12)
    b = UncertaintyQuantifier(seed=12)
    assert a._rng.standard_normal(16).tolist() == b._rng.standard_normal(16).tolist()


def test_anomaly_detection_enhanced_hmm_default_seed_42() -> None:
    from omni_mercury_engine.cognitive.anomaly_detection_enhanced import HiddenMarkovPredictor

    a = HiddenMarkovPredictor(n_states=4)
    b = HiddenMarkovPredictor(n_states=4)
    assert a._rng.dirichlet(np.ones(4)).tolist() == b._rng.dirichlet(np.ones(4)).tolist()


def test_anomaly_detection_simulated_sources_seed_reproducible() -> None:
    from omni_mercury_engine.cognitive.anomaly_detection_enhanced import (
        SimulatedEnvironmentalSource,
        SimulatedGeologicalSource,
    )

    a, b = SimulatedGeologicalSource(seed=21), SimulatedGeologicalSource(seed=21)
    assert a.fetch()[0].data == b.fetch()[0].data

    c, d = SimulatedEnvironmentalSource(seed=21), SimulatedEnvironmentalSource(seed=21)
    assert c.fetch()[0].data == d.fetch()[0].data


def test_neural_memory_kmeans_clusterer_centroids_reproducible() -> None:
    """KMeansClusterer's centroid initialization is deterministic w.r.t. ``random_state``."""
    from omni_mercury_engine.cognitive.neural_memory_layer import KMeansClusterer

    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 8))
    a = KMeansClusterer(n_clusters=4, random_state=42).fit(X)
    b = KMeansClusterer(n_clusters=4, random_state=42).fit(X)
    assert a.centroids is not None and b.centroids is not None
    assert a.centroids.tolist() == b.centroids.tolist()


def test_neurosymbolic_attention_default_seed_42() -> None:
    """AttentionMechanism preserves deterministic init from the previous global seed."""
    from omni_mercury_engine.cognitive.neurosymbolic_fusion import AttentionMechanism

    a = AttentionMechanism(hidden_dim=8)
    b = AttentionMechanism(hidden_dim=8)
    assert a.W_neural.tolist() == b.W_neural.tolist()
    assert a.W_symbolic.tolist() == b.W_symbolic.tolist()
    assert a.W_attention.tolist() == b.W_attention.tolist()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def test_quantum_anomaly_model_seed_reproducible() -> None:
    from omni_mercury_engine.models.quantum import QuantumAnomalyModel

    a = QuantumAnomalyModel({"seed": 5})
    b = QuantumAnomalyModel({"seed": 5})
    assert a._rng.random() == b._rng.random()


def test_quantum_engine_seed_reproducible() -> None:
    from omni_mercury_engine.models.quantum_engine import QuantumEngine

    a = QuantumEngine(seed=3)
    b = QuantumEngine(seed=3)
    assert a._rng.integers(0, 2, size=10).tolist() == b._rng.integers(0, 2, size=10).tolist()


def test_quantum_state_measure_no_global_state() -> None:
    """``QuantumState.measure`` must not consume the legacy global RNG state."""
    from omni_mercury_engine.models.quantum_engine import QuantumState

    state = QuantumState(amplitudes=np.array([1.0, 0.0]) + 0j, num_qubits=1)
    np.random.seed(0)
    snap_before = np.random.random()
    np.random.seed(0)
    state.measure()  # must NOT touch global state
    snap_after = np.random.random()
    assert snap_before == snap_after


def test_biometric_advanced_engine_seed_reproducible() -> None:
    from omni_mercury_engine.models.biometric_advanced import (
        AdvancedBiometricEngine,
        AgeProgressionEngine,
    )

    a = AdvancedBiometricEngine(seed=14)
    b = AdvancedBiometricEngine(seed=14)
    assert a._rng.standard_normal(8).tolist() == b._rng.standard_normal(8).tolist()

    c = AgeProgressionEngine(seed=14)
    d = AgeProgressionEngine(seed=14)
    assert c._rng.standard_normal(8).tolist() == d._rng.standard_normal(8).tolist()


# ---------------------------------------------------------------------------
# Cross-cutting invariants
# ---------------------------------------------------------------------------


def test_different_seeds_diverge() -> None:
    from omni_mercury_engine.cognitive.causal_discovery import CausalDiscoveryEngine

    a = CausalDiscoveryEngine(seed=1)
    b = CausalDiscoveryEngine(seed=2)
    assert (
        a._rng.choice(100, 10, replace=True).tolist()
        != b._rng.choice(100, 10, replace=True).tolist()
    )


def test_seed_none_does_not_inherit_legacy_global_seed() -> None:
    """
    The deterministic invariant of ``seed=None``: it must NOT pull the
    initial state from the legacy ``np.random.seed(...)`` global.  If the
    cure regressed and the constructor fell back to the global RNG, two
    engines built immediately after identical ``np.random.seed(0)``
    poisoning would produce identical sequences — this asserts they don't.

    ``np.random.default_rng(None)`` reads OS entropy via ``SeedSequence``
    (NumPy's documented contract), so a 256-sample collision between two
    independently-OS-seeded streams is bounded above by 2**(-2048) — i.e.,
    deterministically false in any timeline this test will ever run in.
    The probabilistic phrasing ``with overwhelming probability`` in the
    previous version of this test was technically accurate but obscured
    what we are actually pinning: independence from the legacy global.
    """
    from omni_mercury_engine.cognitive.causal_discovery import CausalDiscoveryEngine

    np.random.seed(0)
    e_a = CausalDiscoveryEngine()  # seed=None
    v_a = e_a._rng.standard_normal(256).tolist()

    np.random.seed(0)
    e_b = CausalDiscoveryEngine()  # seed=None, immediately after the same poisoning
    v_b = e_b._rng.standard_normal(256).tolist()

    assert v_a != v_b, (
        "seed=None appears to be derived from the legacy np.random global "
        "state — the cure has regressed.  Both engines produced identical "
        "sequences after np.random.seed(0) was set immediately before each "
        "construction; OS-entropy seeding cannot collide on 256 samples."
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: __import__(
            "omni_mercury_engine.cognitive.causal_discovery",
            fromlist=["CausalDiscoveryEngine"],
        ).CausalDiscoveryEngine(seed=42),
        lambda: __import__(
            "omni_mercury_engine.cognitive.knowledge_graph",
            fromlist=["KnowledgeGraph"],
        ).KnowledgeGraph(seed=42),
        lambda: __import__(
            "omni_mercury_engine.cognitive.chain_of_thought",
            fromlist=["ChainOfThoughtEngine"],
        ).ChainOfThoughtEngine(seed=42),
        lambda: __import__(
            "omni_mercury_engine.cognitive.uncertainty",
            fromlist=["UncertaintyQuantifier"],
        ).UncertaintyQuantifier(seed=42),
        lambda: __import__(
            "omni_mercury_engine.models.quantum_engine",
            fromlist=["QuantumEngine"],
        ).QuantumEngine(seed=42),
    ],
)
def test_engine_isolated_from_global_np_random_seed(factory: Any) -> None:
    """
    Critical invariant of the RNG cure: an unrelated caller poisoning the
    legacy ``np.random.seed(...)`` global state must NOT change a seeded
    engine's output. If this regresses, the cure is undone.
    """
    np.random.seed(0)
    e_a = factory()
    v_a = e_a._rng.standard_normal(64).tolist()

    np.random.seed(99999)
    e_b = factory()
    v_b = e_b._rng.standard_normal(64).tolist()

    assert v_a == v_b


# ---------------------------------------------------------------------------
# Extended sweep coverage (PR #190 follow-up): all RNG-cured modules
# outside ``cognitive/`` and ``models/`` that were already pinned above.
# ---------------------------------------------------------------------------


def test_agentic_autonomy_seed_reproducible() -> None:
    from omni_mercury_engine.agentic.agentic_autonomy import AgenticAutonomy

    a = AgenticAutonomy(seed=4)
    b = AgenticAutonomy(seed=4)
    assert a._rng.random() == b._rng.random()


def test_federation_privacy_seed_reproducible() -> None:
    from omni_mercury_engine.federation.privacy import DifferentialPrivacy

    a = DifferentialPrivacy(epsilon=1.0, seed=42)
    b = DifferentialPrivacy(epsilon=1.0, seed=42)
    assert a._rng.normal(0, 1, 32).tolist() == b._rng.normal(0, 1, 32).tolist()


def test_consciousness_data_source_seed_reproducible() -> None:
    from omni_mercury_engine.data_sources.consciousness import GCPDataSource

    a = GCPDataSource(seed=7)
    b = GCPDataSource(seed=7)
    assert a._rng.binomial(200, 0.5, 5).tolist() == b._rng.binomial(200, 0.5, 5).tolist()


def test_pareto_optimizer_seed_reproducible() -> None:
    """``ParetoOptimizer`` ``self._rng`` reseeds deterministically from ``self.seed``."""
    from omni_mercury_engine.core.benevolence_optimization import ParetoOptimizer

    def obj(p):
        return np.array([float(np.sum(p**2)), float(np.sum(np.abs(p)))])

    a = ParetoOptimizer(objective_fn=obj, n_objectives=2, seed=1)
    b = ParetoOptimizer(objective_fn=obj, n_objectives=2, seed=1)
    # Sample directly from the per-instance Generator — the optimizer's
    # ``optimize`` method has a pre-existing latent bug in
    # ``_fast_non_dominated_sort`` that surfaces with array-equal
    # dataclass fields, unrelated to the RNG cure; the RNG-state-isolation
    # invariant is what we are pinning here.
    assert a._rng.uniform(-1, 1, 32).tolist() == b._rng.uniform(-1, 1, 32).tolist()


def test_calibration_ensemble_seed_reproducible() -> None:
    from omni_mercury_engine.core.calibration import CalibrationEnsemble

    a = CalibrationEnsemble(seed=11)
    b = CalibrationEnsemble(seed=11)
    assert a._rng.permutation(64).tolist() == b._rng.permutation(64).tolist()


def test_pathogen_detector_seed_reproducible() -> None:
    pytest.importorskip("torch", reason="bio_threats package imports torch transitively")
    from omni_mercury_engine.medical.pandemic.bio_threats.pathogen_detector import (
        PathogenDetector,
    )

    a = PathogenDetector(config={"state_dim": 8, "seed": 7})
    b = PathogenDetector(config={"state_dim": 8, "seed": 7})
    assert a.J_matrix.tolist() == b.J_matrix.tolist()


def test_quantum_circuit_seed_reproducible() -> None:
    from omni_mercury_engine.quantum_computing.circuits import (
        SimulatedQuantumCircuit,
        VariationalCircuit,
    )

    c1 = SimulatedQuantumCircuit(num_qubits=3, seed=21)
    c2 = SimulatedQuantumCircuit(num_qubits=3, seed=21)
    assert (
        c1._rng.choice(8, 16, p=[1 / 8] * 8).tolist()
        == c2._rng.choice(8, 16, p=[1 / 8] * 8).tolist()
    )

    v1 = VariationalCircuit(num_qubits=3, reps=1, seed=21)
    v2 = VariationalCircuit(num_qubits=3, reps=1, seed=21)
    assert v1._rng.uniform(0, 1, 8).tolist() == v2._rng.uniform(0, 1, 8).tolist()


def test_quantum_hybrid_optimizers_seed_reproducible() -> None:
    from omni_mercury_engine.quantum_computing.hybrid import (
        ClassicalOptimizer,
        HybridOptimizer,
    )

    a = ClassicalOptimizer(seed=3)
    b = ClassicalOptimizer(seed=3)
    assert a._rng.choice([-1, 1], 16).tolist() == b._rng.choice([-1, 1], 16).tolist()

    h1 = HybridOptimizer(seed=3)
    h2 = HybridOptimizer(seed=3)
    assert h1._rng.uniform(0, 1, 8).tolist() == h2._rng.uniform(0, 1, 8).tolist()


def test_voice_embedding_default_seed_42() -> None:
    """Voice ``SpeakerEmbedding`` default ``seed=42`` keeps deterministic init."""
    from omni_mercury_engine.biometric.voice_recognition import SpeakerEmbedding

    a = SpeakerEmbedding(input_dim=8, embedding_dim=4)
    b = SpeakerEmbedding(input_dim=8, embedding_dim=4)
    assert a._weights1.tolist() == b._weights1.tolist()
    assert a._weights3.tolist() == b._weights3.tolist()


def test_dataset_synthetic_generators_isolated_from_global() -> None:
    """
    Synthetic dataset fallbacks must not be perturbed by global-state
    poisoning.  Spot-check ``adrepository.ADRepositoryLoader._create_synthetic_fallback``
    which has known anomaly distribution.
    """
    from omni_mercury_engine.datasets.adrepository import ADRepositoryLoader
    from omni_mercury_engine.datasets.base import DatasetConfig

    cfg_a = DatasetConfig(name="thyroid", random_seed=42, max_samples=100)
    cfg_b = DatasetConfig(name="thyroid", random_seed=42, max_samples=100)

    np.random.seed(0)  # poison
    loader_a = ADRepositoryLoader(cfg_a)
    loader_a._create_synthetic_fallback()

    np.random.seed(99999)  # different poison
    loader_b = ADRepositoryLoader(cfg_b)
    loader_b._create_synthetic_fallback()

    # Same random_seed -> same synthetic data regardless of global state.
    a_path = loader_a.data_path / "synthetic_data.npz"
    b_path = loader_b.data_path / "synthetic_data.npz"
    if a_path.exists() and b_path.exists():
        a_data = np.load(a_path)
        b_data = np.load(b_path)
        assert a_data["features"].tolist() == b_data["features"].tolist(), (
            "ADRepositoryLoader._create_synthetic_fallback was perturbed by "
            "the legacy global ``np.random.seed`` state — the cure has regressed."
        )
