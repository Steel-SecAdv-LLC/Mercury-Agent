# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Native-acceleration parity tests.

Every vectorized / JIT fast path introduced by the 2026-06-11 optimization
pass must be a *compiled form* of its reference semantics, never a
behavioral change. This module pins that contract:

* ``DimensionalAnalyzer``: the batched-FFT DB term equals the original
  per-row scalar helpers (which remain shipped) — including the documented
  identically-zero spectral-divergence component.
* ``SigmaDirectiveDetector``: the strided sliding-window variance and the
  chunked pairwise-stability scores equal the original Python loops
  (oracles preserved verbatim in this file).
* ``MultiAgentOrchestrator``: the vectorized confidence-weighted consensus
  equals the per-sample ``ConsensusProtocol`` derivation, and the built-in
  runtime spot-check fails closed on any divergence.
* ``detectors/spatial`` numba lane (``[performance]`` extra): JIT outputs
  equal the pure-numpy formulas. Skipped when numba is not installed.
"""

from __future__ import annotations

import importlib.util
import math
from typing import Any

import numpy as np
import pytest

SEEDS = (0, 1, 2)


def _rows(seed: int, n: int = 64, d: int = 9) -> np.ndarray[Any, Any]:
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(0, 1, (n - 6, d)), rng.normal(4, 1, (6, d))])


# =============================================================================
# DimensionalAnalyzer — batched FFT DB term vs the shipped scalar helpers
# =============================================================================


class TestDimensionalDBVectorization:
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("d", [3, 6, 9, 21])
    def test_row_helpers_match_scalar_helpers(self, seed: int, d: int) -> None:
        from omni_mercury_engine.detectors.dimensional import DimensionalAnalyzer

        data = _rows(seed, n=40, d=d)
        analyzer = DimensionalAnalyzer()

        phases = analyzer._phase_coherence_rows(data)
        thds = analyzer._harmonic_distortion_rows(data)
        for i in range(len(data)):
            assert math.isclose(
                phases[i], analyzer._compute_phase_coherence(data[i, :]), abs_tol=1e-12
            ), f"phase coherence diverged at row {i} (d={d})"
            assert math.isclose(
                thds[i], analyzer._compute_harmonic_distortion(data[i, :]), abs_tol=1e-12
            ), f"harmonic distortion diverged at row {i} (d={d})"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_db_scores_match_original_loop(self, seed: int) -> None:
        from omni_mercury_engine.detectors.dimensional import DimensionalAnalyzer

        data = _rows(seed, n=48, d=9)
        # The oracle below is the legacy loop (zero spectral term); pin the
        # explicit opt-out path, which preserves pre-2026-06-11 scores.
        analyzer = DimensionalAnalyzer({"db_spectral_divergence": False})
        analyzer.fit(_rows(seed + 100, n=120, d=9))

        produced = analyzer._dimensional_code_breaking(data)

        # Verbatim oracle of the pre-vectorization loop.
        assert analyzer.baseline_spectral_signature is not None
        expected = np.zeros(len(data))
        for idx in range(len(data)):
            sample = data[idx : idx + 1, :]
            sample_signature = analyzer._compute_spectral_signature(sample)
            min_len = min(len(analyzer.baseline_spectral_signature), len(sample_signature))
            baseline_truncated = analyzer.baseline_spectral_signature[:min_len]
            sample_truncated = sample_signature[:min_len]
            spectral_divergence = np.linalg.norm(baseline_truncated - sample_truncated) / (
                np.linalg.norm(baseline_truncated) + 1e-10
            )
            # The single-row signature is empty (length-1 column FFTs halve
            # to nothing), so this term is identically zero — the documented
            # dead component the vectorized form preserves explicitly.
            assert sample_signature.size == 0
            assert spectral_divergence == 0.0
            db = (
                spectral_divergence * 0.5
                + (1.0 - analyzer._compute_phase_coherence(data[idx, :])) * 0.3
                + analyzer._compute_harmonic_distortion(data[idx, :]) * 0.2
            )
            expected[idx] = min(float(db), 1.0)

        np.testing.assert_allclose(produced, expected, rtol=0, atol=1e-12)


# =============================================================================
# SigmaDirectiveDetector — sliding-window variance and chunked GSIS
# =============================================================================


class TestDirectiveVectorization:
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("shape", [(30, 7), (5, 3), (200, 21)])
    def test_micro_anomalies_match_original_loop(self, seed: int, shape: tuple[int, int]) -> None:
        from omni_mercury_engine.detectors.directive import SigmaDirectiveDetector

        data = (
            _rows(seed, n=shape[0], d=shape[1])
            if shape[0] > 6
            else (np.random.default_rng(seed).normal(0, 1, shape))
        )
        detector = SigmaDirectiveDetector()
        produced = detector._detect_micro_anomalies(data)

        # Verbatim oracle of the pre-vectorization loop.
        if data.size < 4:
            expected = 0.0
        else:
            data_flat = data.flatten()
            window_size = min(4, len(data_flat) // 2)
            local_variances = [
                np.var(data_flat[i : i + window_size])
                for i in range(len(data_flat) - window_size + 1)
            ]
            if not local_variances:
                expected = 0.0
            else:
                variance_array = np.array(local_variances)
                variance_changes = np.abs(np.diff(variance_array))
                expected = float(
                    min(np.mean(variance_changes) / (np.std(variance_array) + 1e-10), 1.0)
                )

        assert math.isclose(produced, expected, abs_tol=1e-12)

    def test_micro_anomalies_tiny_input_guard(self) -> None:
        from omni_mercury_engine.detectors.directive import SigmaDirectiveDetector

        detector = SigmaDirectiveDetector()
        assert detector._detect_micro_anomalies(np.array([1.0, 2.0])) == 0.0

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("n", [1, 2, 5, 127, 130, 300])
    def test_gravitational_stability_matches_original_loop(self, seed: int, n: int) -> None:
        from omni_mercury_engine.detectors.directive import SigmaDirectiveDetector

        rng = np.random.default_rng(seed)
        data = rng.normal(0, 1, (n, 6))
        detector = SigmaDirectiveDetector()
        produced = detector._gravitational_stability_check(data)

        # Verbatim oracle of the pre-vectorization loop.
        if n < 2:
            expected = np.zeros(n)
        else:
            expected = np.zeros(n)
            for i in range(n):
                distances = np.linalg.norm(data - data[i], axis=1)
                local_density = np.sum(distances < np.percentile(distances, 20))
                expected[i] = 1.0 - local_density / n
            expected = expected * detector.stability_factor

        np.testing.assert_array_equal(produced, expected)


# =============================================================================
# Orchestrator — vectorized consensus vs the real ConsensusProtocol
# =============================================================================


class TestConsensusFastPathParity:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_vectorized_consensus_equals_protocol(self, seed: int) -> None:
        from omni_mercury_engine.agentic.orchestration import MultiAgentOrchestrator
        from omni_mercury_engine.cognitive.multi_agent_coordination import DetectionResult

        rng = np.random.default_rng(seed)
        X_train = rng.normal(0, 1, (250, 6))
        X_test = np.vstack([rng.normal(0, 1, (90, 6)), rng.normal(5, 1, (10, 6))])

        orch = MultiAgentOrchestrator(seed=seed).fit(X_train)
        batch = orch.coordinate(X_test)

        agent_names = list(batch.per_agent_scores)
        for i in range(X_test.shape[0]):
            results = [
                DetectionResult(
                    agent_id=name,
                    anomaly_score=float(batch.per_agent_scores[name][i]),
                    is_anomaly=bool(
                        batch.per_agent_scores[name][i] > orch.agents[name].decision_threshold
                    ),
                    confidence=orch.agents[name].confidence_for(
                        float(batch.per_agent_scores[name][i])
                    ),
                )
                for name in agent_names
            ]
            reference = orch.protocol.reach_consensus(results)
            assert not isinstance(reference, dict)
            reference_score = (
                reference.confidence if reference.final_decision else 1.0 - reference.confidence
            )
            assert math.isclose(
                float(batch.consensus_scores[i]), reference_score, abs_tol=1e-12
            ), f"consensus score diverged at sample {i}"
            assert math.isclose(
                float(batch.agreement[i]), reference.agreement_ratio, abs_tol=1e-12
            ), f"agreement diverged at sample {i}"
            assert int(batch.dissent_counts[i]) == len(reference.dissenting_agents)
            assert bool(batch.decisions[i]) == (
                float(batch.consensus_scores[i]) > orch.operating_threshold
            )

    def test_spot_check_fails_closed_on_divergence(self) -> None:
        from omni_mercury_engine.agentic.orchestration import (
            MultiAgentOrchestrator,
            OrchestrationError,
        )

        rng = np.random.default_rng(0)
        orch = MultiAgentOrchestrator(seed=0).fit(rng.normal(0, 1, (250, 6)))
        X_test = rng.normal(0, 1, (50, 6))
        batch = orch.coordinate(X_test)

        agent_names = list(batch.per_agent_scores)
        score_matrix = np.vstack([batch.per_agent_scores[n] for n in agent_names])
        thresholds = np.array([[orch.agents[n].decision_threshold] for n in agent_names])
        scales = np.array([[orch.agents[n].confidence_scale] for n in agent_names])
        votes = score_matrix > thresholds
        confidences = np.clip(np.abs(score_matrix - thresholds) / scales, 0.0, 1.0)

        with pytest.raises(OrchestrationError, match="diverged"):
            orch._spot_check_consensus(
                agent_names,
                score_matrix,
                votes,
                confidences,
                batch.consensus_scores + 0.05,  # corrupted fast-path output
                batch.agreement,
                batch.dissent_counts,
            )


# =============================================================================
# Spatial detector numba lane ([performance] extra)
# =============================================================================

# Class-scoped skip (NOT module-level importorskip): the dimensional /
# directive / orchestrator parity tests above must run in every
# environment; only this class needs the [performance] extra.
_NUMBA_MISSING = importlib.util.find_spec("numba") is None


@pytest.mark.skipif(_NUMBA_MISSING, reason="performance extra not installed")
class TestSpatialNumbaLaneParity:
    def test_jit_lane_is_active(self) -> None:
        from omni_mercury_engine.detectors import spatial

        assert spatial.NUMBA_AVAILABLE is True

    @pytest.mark.parametrize("seed", SEEDS)
    def test_jit_distances_match_numpy(self, seed: int) -> None:
        from omni_mercury_engine.detectors.spatial import _compute_distances_jit

        rng = np.random.default_rng(seed)
        data = rng.normal(0, 1, (200, 12))
        center = rng.normal(0, 1, 12)

        produced = _compute_distances_jit(data, center)
        expected = np.linalg.norm(data - center, axis=1)
        np.testing.assert_allclose(produced, expected, rtol=0, atol=1e-10)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_jit_distance_scores_match_numpy(self, seed: int) -> None:
        from omni_mercury_engine.detectors.spatial import _compute_distance_scores_jit

        rng = np.random.default_rng(seed)
        distances = np.abs(rng.normal(2, 1, 300))
        radius = 2.0

        produced = _compute_distance_scores_jit(distances, radius)
        expected = np.maximum(distances - radius, 0) / (radius + 1e-6)
        np.testing.assert_allclose(produced, expected, rtol=0, atol=1e-12)


@pytest.mark.skipif(_NUMBA_MISSING, reason="performance extra not installed")
class TestGSISNumbaLaneParity:
    """The GSIS numba kernel is bit-identical to the numpy broadcast lane."""

    def test_gsis_numba_lane_is_active(self) -> None:
        from omni_mercury_engine.detectors import directive

        assert directive.GSIS_NUMBA_AVAILABLE is True

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("shape", [(57, 3), (200, 7), (300, 21), (150, 130), (90, 1)])
    def test_gsis_scores_identical_across_lanes(
        self, seed: int, shape: tuple[int, int], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omni_mercury_engine.detectors import directive

        rng = np.random.default_rng(seed)
        data = rng.normal(0, 1, shape)
        if seed % 2 == 1:  # duplicate-heavy: ties at the percentile boundary
            base = rng.normal(0, 1, (max(4, shape[0] // 10), shape[1]))
            data = base[rng.integers(0, len(base), shape[0])]
        detector = directive.SigmaDirectiveDetector()

        produced_numba = detector._gravitational_stability_check(data)
        monkeypatch.setattr(directive, "GSIS_NUMBA_AVAILABLE", False)
        produced_numpy = detector._gravitational_stability_check(data)

        np.testing.assert_array_equal(produced_numba, produced_numpy)

    def test_non_float64_uses_numpy_semantics(self) -> None:
        """float32 input keeps float32 arithmetic (the numpy lane)."""
        from omni_mercury_engine.detectors import directive

        data = np.random.default_rng(3).normal(0, 1, (60, 9)).astype(np.float32)
        detector = directive.SigmaDirectiveDetector()
        produced = detector._gravitational_stability_check(data)
        expected = np.zeros(60)
        for i in range(60):
            distances = np.linalg.norm(data - data[i], axis=1)
            local_density = np.sum(distances < np.percentile(distances, 20))
            expected[i] = 1.0 - local_density / 60
        np.testing.assert_array_equal(produced, expected * detector.stability_factor)


# =============================================================================
# DetectorAgent — purity contract + exact incremental single-sample serving
# =============================================================================


def _fitted_agent(name: str, seed: int, n_train: int = 90, d: int = 7) -> Any:
    """Build one fitted DetectorAgent over a real detector instance."""
    from omni_mercury_engine.agentic.orchestration import (
        DetectorAgent,
        default_detector_suite,
    )

    if n_train >= 12:
        train = _rows(seed + 7, n=n_train, d=d)
    else:  # _rows needs >= 6 rows of anomaly headroom; tiny fits use plain rows
        train = np.random.default_rng(seed + 7).normal(0, 1, (n_train, d))
    detector = default_detector_suite()[name]
    agent = DetectorAgent(name, detector, seed=seed)
    agent.fit(train)
    return agent


def _full_path_oracle(agent: Any, row: np.ndarray[Any, Any]) -> float:
    """The documented single-sample contract, computed without the cache."""
    batch = np.vstack([agent._reference, row.reshape(1, -1)])
    return float(agent.score_batch(batch)[-1])


class TestServingPurity:
    """score_batch is a pure function of (fitted state, batch).

    Before 2026-06-11 the directive detector's recursive-memory buffer
    leaked across calls, so the first ``memory_depth`` rows of every batch
    scored differently depending on the previous call — coordinate() on the
    same input twice returned different consensus scores.
    """

    @pytest.mark.parametrize("seed", SEEDS)
    def test_score_batch_identical_across_calls(self, seed: int) -> None:
        agent = _fitted_agent("directive", seed)
        batch = _rows(seed + 21, n=40, d=7)
        first = agent.score_batch(batch)
        # Perturb the buffer with an unrelated batch in between.
        agent.score_batch(_rows(seed + 99, n=11, d=7))
        second = agent.score_batch(batch)
        np.testing.assert_array_equal(first, second)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_coordinate_identical_across_calls(self, seed: int) -> None:
        from omni_mercury_engine.agentic.orchestration import MultiAgentOrchestrator

        orch = MultiAgentOrchestrator(seed=seed).fit(_rows(seed, n=120, d=7))
        batch = _rows(seed + 5, n=48, d=7)
        first = orch.coordinate(batch)
        second = orch.coordinate(batch)
        np.testing.assert_array_equal(first.consensus_scores, second.consensus_scores)
        np.testing.assert_array_equal(first.decisions, second.decisions)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_serve_then_batch_unaffected_by_history(self, seed: int) -> None:
        """Single-sample serving must not perturb subsequent batch scoring."""
        agent = _fitted_agent("directive", seed)
        batch = _rows(seed + 3, n=30, d=7)
        baseline = agent.score_batch(batch)
        for i in range(4):
            agent.detect(batch[i])
        np.testing.assert_array_equal(agent.score_batch(batch), baseline)


class TestParallelCoordinationParity:
    """Thread-pooled agent scoring is bit-identical to serial scoring."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_coordinate_equals_serial_agent_scoring(self, seed: int) -> None:
        from omni_mercury_engine.agentic.orchestration import MultiAgentOrchestrator

        orch = MultiAgentOrchestrator(seed=seed).fit(_rows(seed, n=130, d=7))
        batch = _rows(seed + 9, n=64, d=7)
        coordinated = orch.coordinate(batch)
        for name, agent in orch.agents.items():
            np.testing.assert_array_equal(
                coordinated.per_agent_scores[name],
                agent.score_batch(batch),
                err_msg=f"parallel scoring diverged from serial for agent {name!r}",
            )

    def test_failing_agent_excluded_not_fatal(self) -> None:
        from omni_mercury_engine.agentic.orchestration import MultiAgentOrchestrator

        orch = MultiAgentOrchestrator(seed=0).fit(_rows(0, n=90, d=7))

        def _boom(_x: Any) -> Any:
            raise RuntimeError("induced failure")

        orch.agents["temporal"].score_batch = _boom  # type: ignore[method-assign, assignment]
        batch = orch.coordinate(_rows(4, n=30, d=7))
        assert "temporal" not in batch.per_agent_scores
        assert batch.participant_count == len(orch.agents) - 1


class TestIncrementalServingParity:
    """The incremental serve is bit-identical to the full-batch path."""

    AGENTS = ("directive", "spatial", "temporal")

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("name", AGENTS)
    def test_serve_matches_full_path(self, name: str, seed: int) -> None:
        agent = _fitted_agent(name, seed)
        queries = _rows(seed + 31, n=25, d=7)
        for row in queries:
            served = agent.detect(row).anomaly_score
            assert served == _full_path_oracle(
                agent, row
            ), f"{name} incremental serve diverged from the full path"
        assert agent._serving_cache is not None, f"{name} fast path never engaged"

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("name", AGENTS)
    @pytest.mark.parametrize("n_train", [1, 3, 5, 9, 60])
    def test_serve_matches_full_path_tiny_references(
        self, name: str, seed: int, n_train: int
    ) -> None:
        """Reference sizes around the RMD depth / trend window edges."""
        agent = _fitted_agent(name, seed, n_train=n_train)
        for row in _rows(seed + 13, n=8, d=7):
            assert agent.detect(row).anomaly_score == _full_path_oracle(agent, row)

    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("name", AGENTS)
    def test_serve_matches_full_path_duplicate_heavy(self, name: str, seed: int) -> None:
        """Tied distances exercise the GSIS percentile/count boundaries and
        the spatial constant-range branch."""
        rng = np.random.default_rng(seed)
        base = rng.normal(0, 1, (6, 7))
        train = base[rng.integers(0, 6, size=80)]  # heavy duplication
        from omni_mercury_engine.agentic.orchestration import (
            DetectorAgent,
            default_detector_suite,
        )

        agent = DetectorAgent(name, default_detector_suite()[name], seed=seed)
        agent.fit(train)
        for row in [base[0], base[3], rng.normal(0, 1, 7)]:
            assert agent.detect(row).anomaly_score == _full_path_oracle(agent, row)

    @pytest.mark.parametrize("name", AGENTS)
    def test_configured_directive_and_flags(self, name: str) -> None:
        """Non-default detector configurations stay bit-identical."""
        from omni_mercury_engine.agentic.orchestration import DetectorAgent
        from omni_mercury_engine.detectors.directive import (
            DirectiveWeights,
            SigmaDirectiveDetector,
        )
        from omni_mercury_engine.detectors.spatial import SpatialAnomalyDetector
        from omni_mercury_engine.detectors.temporal import TemporalAnomalyDetector

        detector: Any
        if name == "directive":
            detector = SigmaDirectiveDetector(
                {
                    "memory_depth": 3,
                    "stability_factor": 1.4,
                    "use_harmonic_detection": False,
                    "weights": DirectiveWeights(
                        pcp_weight=0.4, gsis_weight=0.4, rmd_weight=0.1, eoa_weight=0.1
                    ),
                }
            )
        elif name == "spatial":
            detector = SpatialAnomalyDetector({"n_neighbors": 3})
        else:
            detector = TemporalAnomalyDetector({"window_size": 4, "change_threshold": 1.0})
        agent = DetectorAgent(name, detector, seed=0)
        agent.fit(_rows(11, n=70, d=7))
        for row in _rows(17, n=12, d=7):
            assert agent.detect(row).anomaly_score == _full_path_oracle(agent, row)

    def test_subclass_falls_back_to_full_path(self) -> None:
        """Exact-type dispatch: a subclass must not engage the fast path."""
        from omni_mercury_engine.agentic.orchestration import DetectorAgent
        from omni_mercury_engine.detectors.directive import SigmaDirectiveDetector

        class TweakedDirective(SigmaDirectiveDetector):
            pass

        agent = DetectorAgent("directive", TweakedDirective(), seed=0)
        agent.fit(_rows(2, n=40, d=7))
        score = agent.detect(np.random.default_rng(3).normal(0, 1, 7)).anomaly_score
        assert agent._serving_cache is None
        assert 0.0 <= score <= 1.0

    def test_caller_reference_batch_uses_full_path(self) -> None:
        """A context-supplied reference must bypass the fit-time cache."""
        agent = _fitted_agent("directive", 0)
        other_reference = _rows(123, n=30, d=7)
        row = np.random.default_rng(5).normal(0, 1, 7)
        served = agent.detect(row, context={"reference_batch": other_reference}).anomaly_score
        expected = float(agent.score_batch(np.vstack([other_reference, row.reshape(1, -1)]))[-1])
        assert served == expected

    def test_refit_invalidates_cache(self) -> None:
        agent = _fitted_agent("temporal", 0)
        row = np.random.default_rng(5).normal(0, 1, 7)
        first = agent.detect(row).anomaly_score
        assert first == _full_path_oracle(agent, row)
        agent.fit(_rows(40, n=80, d=7))
        assert agent.detect(row).anomaly_score == _full_path_oracle(agent, row)

    def test_non_finite_row_falls_back(self) -> None:
        agent = _fitted_agent("spatial", 0)
        row = np.random.default_rng(5).normal(0, 1, 7)
        row[2] = np.nan
        served = agent.detect(row).anomaly_score
        assert served == _full_path_oracle(agent, row)
