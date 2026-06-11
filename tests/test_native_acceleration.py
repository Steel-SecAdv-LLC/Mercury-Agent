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
        analyzer = DimensionalAnalyzer()
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
            expected[idx] = min(db, 1.0)

        np.testing.assert_allclose(produced, expected, rtol=0, atol=1e-12)


# =============================================================================
# SigmaDirectiveDetector — sliding-window variance and chunked GSIS
# =============================================================================


class TestDirectiveVectorization:
    @pytest.mark.parametrize("seed", SEEDS)
    @pytest.mark.parametrize("shape", [(30, 7), (5, 3), (200, 21)])
    def test_micro_anomalies_match_original_loop(
        self, seed: int, shape: tuple[int, int]
    ) -> None:
        from omni_mercury_engine.detectors.directive import SigmaDirectiveDetector

        data = _rows(seed, n=shape[0], d=shape[1]) if shape[0] > 6 else (
            np.random.default_rng(seed).normal(0, 1, shape)
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
                        batch.per_agent_scores[name][i]
                        > orch.agents[name].decision_threshold
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
                reference.confidence
                if reference.final_decision
                else 1.0 - reference.confidence
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

numba = pytest.importorskip("numba", reason="performance extra not installed")


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
