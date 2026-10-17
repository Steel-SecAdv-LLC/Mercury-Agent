# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Native-acceleration parity tests.

Every vectorized / JIT fast path introduced by the 2026-06-11 optimization
pass must be a *compiled form* of its reference semantics, never a
behavioral change. This module pins that contract:

* ``DimensionalAnalyzer``: the batched-FFT DB term equals the original
  per-row scalar helpers (which remain shipped) — including the documented
  identically-zero spectral-divergence component.
* ``SigmaDirectiveDetector``: the strided sliding-window variance equals
  the original Python loop (oracle preserved verbatim in this file). Its
  GSIS term was replaced (2026-10-16): the original scored every row 0.8
  by construction; ``TestGSISDensity`` pins the replacement's behaviour.
* ``MultiAgentOrchestrator``: the vectorized confidence-weighted consensus
  equals the per-sample ``ConsensusProtocol`` derivation, and the built-in
  runtime spot-check fails closed on any divergence.
* ``detectors/spatial`` distance scoring: the BLAS expansion equals the
  ``np.linalg.norm`` definition, in every environment (the former numba
  lane was removed 2026-10-16; the expansion is faster and unconditional).
"""

from __future__ import annotations

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
# Spatial detector distance scoring (BLAS expansion, every environment)
# =============================================================================


class TestSpatialDistanceScoring:
    """The BLAS distance expansion matches the definition on every host."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_expanded_distances_match_the_definition(self, seed: int) -> None:
        from omni_mercury_engine.detectors.spatial import _euclidean_distances

        rng = np.random.default_rng(seed)
        data = rng.normal(0, 1, (200, 12))
        center = rng.normal(0, 1, 12)

        produced = _euclidean_distances(data, center)
        expected = np.linalg.norm(data - center, axis=1)
        np.testing.assert_allclose(produced, expected, rtol=0, atol=1e-10)

    def test_a_point_at_the_centre_is_zero(self) -> None:
        from omni_mercury_engine.detectors.spatial import _euclidean_distances

        center = np.full(8, 1e3)
        data = np.vstack([center, center + 1.0])
        produced = _euclidean_distances(data, center)
        assert produced[0] == 0.0  # the clip absorbs negative rounding
        assert produced[1] == pytest.approx(np.sqrt(8.0))


class TestGSISDensity:
    """GSIS scores density against the fitted baseline, at cost linear in the batch."""

    @staticmethod
    def _detector(seed: int = 0, n: int = 2000, d: int = 16) -> Any:
        from omni_mercury_engine.detectors.directive import SigmaDirectiveDetector

        detector = SigmaDirectiveDetector()
        detector.fit(np.random.default_rng(seed).normal(0, 1, (n, d)))
        return detector

    def test_the_original_form_was_constant(self) -> None:
        """Regression record: the replaced formula gave every row the same score."""
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, (300, 6))
        data[:5] += 8.0
        scores = np.empty(len(data))
        for i in range(len(data)):
            distances = np.linalg.norm(data - data[i], axis=1)
            scores[i] = 1.0 - np.sum(distances < np.percentile(distances, 20)) / len(data)
        assert np.ptp(scores) == 0.0  # 8-sigma outliers indistinguishable

    @pytest.mark.parametrize("seed", SEEDS)
    def test_outliers_score_higher_than_inliers(self, seed: int) -> None:
        detector = self._detector(seed)
        rng = np.random.default_rng(seed + 100)
        batch = rng.normal(0, 1, (500, 16))
        batch[:10] += 1.5
        scores = detector._gravitational_stability_check(batch)
        assert scores[:10].min() > np.median(scores[10:])
        assert 0.4 < float(np.mean(scores[10:])) < 0.6  # typical rows sit mid-distribution
        assert np.all((scores >= 0.0) & (scores <= 1.0))

    def test_unfitted_batch_is_its_own_reference(self) -> None:
        from omni_mercury_engine.detectors.directive import SigmaDirectiveDetector

        rng = np.random.default_rng(1)
        batch = rng.normal(0, 1, (400, 8))
        batch[:5] += 8.0
        scores = SigmaDirectiveDetector()._gravitational_stability_check(batch)
        assert scores[:5].min() > 0.95 > float(np.mean(scores[5:]))

    def test_non_finite_rows_score_one_and_leave_others_unchanged(self) -> None:
        detector = self._detector()
        batch = np.random.default_rng(2).normal(0, 1, (100, 16))
        clean = detector._gravitational_stability_check(batch)
        batch[7, 3] = np.nan
        dirty = detector._gravitational_stability_check(batch)
        assert dirty[7] == 1.0
        np.testing.assert_array_equal(np.delete(dirty, 7), np.delete(clean, 7))

    def test_stability_factor_scales_the_score(self) -> None:
        from omni_mercury_engine.detectors.directive import SigmaDirectiveDetector

        train = np.random.default_rng(3).normal(0, 1, (300, 4))
        batch = np.random.default_rng(4).normal(0, 1, (50, 4))
        plain = SigmaDirectiveDetector().fit(train)._gravitational_stability_check(batch)
        scaled = SigmaDirectiveDetector({"stability_factor": 1.4}).fit(train)
        np.testing.assert_allclose(scaled._gravitational_stability_check(batch), plain * 1.4)

    def test_fitted_state_round_trips_the_reference(self) -> None:
        from omni_mercury_engine.detectors.directive import SigmaDirectiveDetector

        detector = self._detector()
        batch = np.random.default_rng(5).normal(0, 1, (64, 16))
        restored = SigmaDirectiveDetector()
        state = detector.get_fitted_state()
        assert state is not None
        restored.set_fitted_state(state)
        np.testing.assert_array_equal(
            restored._gravitational_stability_check(batch),
            detector._gravitational_stability_check(batch),
        )
        legacy = SigmaDirectiveDetector()
        legacy.set_fitted_state({"baseline_pattern": state["baseline_pattern"]})
        assert legacy._gsis_reference is None  # a pre-reference checkpoint still loads

    def test_cost_is_linear_in_the_batch(self) -> None:
        """At most ~512 reference rows are compared, never all batch pairs."""
        import time

        detector = self._detector()
        rng = np.random.default_rng(6)
        small, large = rng.normal(0, 1, (2048, 16)), rng.normal(0, 1, (16384, 16))
        detector._gravitational_stability_check(small)  # warm-up
        t0 = time.perf_counter()
        detector._gravitational_stability_check(small)
        t_small = time.perf_counter() - t0
        t0 = time.perf_counter()
        detector._gravitational_stability_check(large)
        t_large = time.perf_counter() - t0
        # 8x the rows: quadratic would be ~64x; allow generous noise for linear.
        assert t_large < 30 * max(t_small, 1e-3)


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
        """Tied distances exercise the GSIS rank boundaries and the spatial
        constant-range branch."""
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
