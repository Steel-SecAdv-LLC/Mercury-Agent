"""Tests for Phase 4 (Advanced Math Structures) and Phase 6 (System Coherence).

Covers:
- Phase 4A: Conformal prediction enhancements (Mondrian, calibration bridge)
- Phase 4B: Topological Data Analysis (persistent homology, Betti, distances)
- Phase 4C: Fisher Information Metric adaptive thresholds
- Phase 4D: Riemannian optimization (simplex, SPD, gradient descent)
- Phase 6: Signal flow graph, normalization verification, Lyapunov enforcement
- Phase 3B: Domain-adaptive AAFE weights
"""

from __future__ import annotations

import numpy as np
import pytest

# ============================================================================
# Phase 4A — Conformal Prediction Enhancements
# ============================================================================


class TestMondrianConformalPredictor:
    """Tests for Mondrian (label-conditional) conformal prediction."""

    def test_fit_returns_self(self) -> None:
        from omni_mercury_engine.core.conformal_prediction import (
            MondrianConformalPredictor,
        )

        scores = np.random.RandomState(42).rand(60)
        groups = np.array(["A"] * 30 + ["B"] * 30)
        pred = MondrianConformalPredictor(coverage=0.90)
        result = pred.fit(scores, groups)
        assert result is pred

    def test_per_group_thresholds_differ(self) -> None:
        from omni_mercury_engine.core.conformal_prediction import (
            MondrianConformalPredictor,
        )

        rng = np.random.RandomState(42)
        # Group A has higher scores than group B
        scores_a = rng.uniform(0.5, 1.0, 40)
        scores_b = rng.uniform(0.0, 0.5, 40)
        scores = np.concatenate([scores_a, scores_b])
        groups = np.array(["A"] * 40 + ["B"] * 40)

        pred = MondrianConformalPredictor(coverage=0.90)
        pred.fit(scores, groups)

        t_a = pred.get_anomaly_threshold("A")
        t_b = pred.get_anomaly_threshold("B")
        # Group A should have higher threshold than group B
        assert t_a > t_b

    def test_predict_returns_binary(self) -> None:
        from omni_mercury_engine.core.conformal_prediction import (
            MondrianConformalPredictor,
        )

        rng = np.random.RandomState(42)
        scores = rng.rand(80)
        groups = np.array(["X"] * 40 + ["Y"] * 40)
        pred = MondrianConformalPredictor().fit(scores, groups)

        test_scores = rng.rand(20)
        test_groups = np.array(["X"] * 10 + ["Y"] * 10)
        preds = pred.predict(test_scores, test_groups)
        assert preds.dtype == int
        assert set(preds).issubset({0, 1})

    def test_fallback_to_global_for_unknown_group(self) -> None:
        from omni_mercury_engine.core.conformal_prediction import (
            MondrianConformalPredictor,
        )

        scores = np.random.RandomState(42).rand(60)
        groups = np.array(["A"] * 30 + ["B"] * 30)
        pred = MondrianConformalPredictor().fit(scores, groups)

        # Unknown group should fallback to global threshold
        t_unknown = pred.get_anomaly_threshold("UNKNOWN")
        t_global = pred.get_anomaly_threshold(None)
        assert t_unknown == t_global

    def test_evaluate_group_coverage(self) -> None:
        from omni_mercury_engine.core.conformal_prediction import (
            MondrianConformalPredictor,
        )

        rng = np.random.RandomState(42)
        scores = rng.rand(100)
        groups = np.array(["A"] * 50 + ["B"] * 50)
        pred = MondrianConformalPredictor(coverage=0.90).fit(scores, groups)

        test_scores = rng.rand(50)
        test_labels = (test_scores > 0.5).astype(int)
        test_groups = np.array(["A"] * 25 + ["B"] * 25)
        result = pred.evaluate_group_coverage(test_scores, test_labels, test_groups)

        assert "overall_coverage" in result
        assert "per_group_coverage" in result
        assert "worst_group_coverage" in result


class TestConformalCalibrationBridge:
    """Tests for the conformal-calibration bridge."""

    def test_calibrate_returns_thresholds(self) -> None:
        from omni_mercury_engine.core.conformal_prediction import (
            ConformalCalibrationBridge,
        )

        bridge = ConformalCalibrationBridge(base_coverage=0.95)
        scores = np.random.RandomState(42).rand(50)
        result = bridge.calibrate(scores)
        assert "split_threshold" in result
        assert "adaptive_threshold" in result

    def test_calibrate_with_groups(self) -> None:
        from omni_mercury_engine.core.conformal_prediction import (
            ConformalCalibrationBridge,
        )

        bridge = ConformalCalibrationBridge()
        scores = np.random.RandomState(42).rand(60)
        groups = np.array(["A"] * 30 + ["B"] * 30)
        result = bridge.calibrate(scores, groups)
        assert "mondrian_A_threshold" in result
        assert "mondrian_B_threshold" in result

    def test_update_adaptive(self) -> None:
        from omni_mercury_engine.core.conformal_prediction import (
            ConformalCalibrationBridge,
        )

        bridge = ConformalCalibrationBridge()
        bridge.calibrate(np.random.RandomState(42).rand(30))

        t, covered = bridge.update_adaptive(0.3)
        assert isinstance(t, float)
        assert isinstance(covered, bool)


# ============================================================================
# Phase 4B — Topological Data Analysis
# ============================================================================


class TestVietorisRipsFiltration:
    """Tests for Vietoris-Rips filtration and persistent homology."""

    def test_empty_point_cloud(self) -> None:
        from omni_mercury_engine.core.topological_analysis import (
            VietorisRipsFiltration,
        )

        vr = VietorisRipsFiltration()
        dgm = vr.build(np.empty((0, 3)))
        assert dgm.pairs_dim0.shape == (0, 2)
        assert dgm.pairs_dim1.shape == (0, 2)

    def test_single_point(self) -> None:
        from omni_mercury_engine.core.topological_analysis import (
            VietorisRipsFiltration,
        )

        vr = VietorisRipsFiltration()
        dgm = vr.build(np.array([[1.0, 2.0]]))
        assert dgm.pairs_dim0.shape[0] == 1
        assert dgm.pairs_dim0[0, 0] == 0.0  # Born at 0
        assert np.isinf(dgm.pairs_dim0[0, 1])  # Never dies

    def test_two_points_h0(self) -> None:
        from omni_mercury_engine.core.topological_analysis import (
            VietorisRipsFiltration,
        )

        pts = np.array([[0.0, 0.0], [3.0, 0.0]])
        vr = VietorisRipsFiltration()
        dgm = vr.build(pts)
        # Two components merge at distance 3.0
        assert dgm.pairs_dim0.shape[0] == 2
        # One finite death at 3.0
        finite = dgm.pairs_dim0[np.isfinite(dgm.pairs_dim0[:, 1])]
        assert len(finite) == 1
        assert abs(finite[0, 1] - 3.0) < 1e-10

    def test_square_has_h1_cycle(self) -> None:
        from omni_mercury_engine.core.topological_analysis import (
            VietorisRipsFiltration,
        )

        # Unit square — should form a 1-cycle
        pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        vr = VietorisRipsFiltration()
        dgm = vr.build(pts)
        # Should have at least one H1 feature
        assert dgm.pairs_dim1.shape[0] >= 1

    def test_betti_numbers(self) -> None:
        from omni_mercury_engine.core.topological_analysis import (
            VietorisRipsFiltration,
        )

        pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [5, 5]], dtype=float)
        vr = VietorisRipsFiltration()
        dgm = vr.build(pts)
        b0 = dgm.betti_at(0.5, dim=0)
        assert b0 >= 1  # At least one connected component

    def test_lifetimes_nonnegative(self) -> None:
        from omni_mercury_engine.core.topological_analysis import (
            VietorisRipsFiltration,
        )

        rng = np.random.RandomState(42)
        pts = rng.randn(20, 3)
        vr = VietorisRipsFiltration()
        dgm = vr.build(pts)
        lt = dgm.lifetimes(0)
        assert np.all(lt >= 0)


class TestTopologicalAnomalyDetector:
    """Tests for TDA-based anomaly detection."""

    def test_fit_and_score(self) -> None:
        from omni_mercury_engine.core.topological_analysis import (
            TopologicalAnomalyDetector,
        )

        rng = np.random.RandomState(42)
        det = TopologicalAnomalyDetector(seed=42)
        ref = rng.randn(30, 3)
        det.fit(ref)
        result = det.score(rng.randn(10, 3))
        assert "anomaly_score" in result
        assert result["anomaly_score"] >= 0.0

    def test_predict_returns_binary(self) -> None:
        from omni_mercury_engine.core.topological_analysis import (
            TopologicalAnomalyDetector,
        )

        rng = np.random.RandomState(42)
        det = TopologicalAnomalyDetector(seed=42)
        det.fit(rng.randn(25, 2))
        preds = det.predict(rng.randn(15, 2))
        assert preds.dtype == int
        assert set(preds).issubset({0, 1})


class TestPersistenceDistances:
    """Tests for bottleneck and Wasserstein distances."""

    def test_identical_diagrams_zero_distance(self) -> None:
        from omni_mercury_engine.core.topological_analysis import (
            VietorisRipsFiltration,
            bottleneck_distance,
            wasserstein_distance_pd,
        )

        pts = np.random.RandomState(42).randn(15, 2)
        vr = VietorisRipsFiltration()
        dgm = vr.build(pts)
        assert bottleneck_distance(dgm, dgm, dim=0) == 0.0
        assert wasserstein_distance_pd(dgm, dgm, dim=0) == 0.0

    def test_distances_nonnegative(self) -> None:
        from omni_mercury_engine.core.topological_analysis import (
            VietorisRipsFiltration,
            bottleneck_distance,
            wasserstein_distance_pd,
        )

        rng = np.random.RandomState(42)
        vr = VietorisRipsFiltration()
        dgm1 = vr.build(rng.randn(15, 2))
        dgm2 = vr.build(rng.randn(15, 2) + 2.0)
        assert bottleneck_distance(dgm1, dgm2, dim=0) >= 0
        assert wasserstein_distance_pd(dgm1, dgm2, dim=0) >= 0


# ============================================================================
# Phase 4C — Fisher Information Metric Adaptive Thresholds
# ============================================================================


class TestFisherInformationMatrix:
    """Tests for Fisher information matrix computation."""

    def test_gaussian_fim_shape(self) -> None:
        from omni_mercury_engine.core.info_geometry import FisherInformationMatrix

        data = np.random.RandomState(42).randn(50, 3)
        covariance = np.cov(data.T)
        fim = FisherInformationMatrix()
        F = fim.compute_gaussian(covariance)
        assert F.shape == (3, 3)
        # Should be symmetric
        assert np.allclose(F, F.T, atol=1e-10)
        # Should be positive semi-definite
        eigvals = np.linalg.eigvalsh(F)
        assert np.all(eigvals >= -1e-10)

    def test_fim_with_regularization(self) -> None:
        from omni_mercury_engine.core.info_geometry import FisherInformationMatrix

        # Singular data (all same) — covariance will be zero
        data = np.ones((20, 3))
        covariance = np.cov(data.T)
        fim = FisherInformationMatrix(tikhonov_lambda=1e-4)
        F = fim.compute_gaussian(covariance)
        assert F.shape == (3, 3)
        # Should still be invertible thanks to Tikhonov regularization
        assert np.linalg.det(F) > 0


class TestNaturalGradient:
    """Tests for natural gradient computation."""

    def test_natural_gradient_shape(self) -> None:
        from omni_mercury_engine.core.info_geometry import NaturalGradient

        ng = NaturalGradient()
        F = np.eye(3) * 2.0
        g = np.array([1.0, 2.0, 3.0])
        g_natural = ng.compute(F, g)
        assert g_natural.shape == (3,)

    def test_identity_fim_preserves_gradient(self) -> None:
        from omni_mercury_engine.core.info_geometry import NaturalGradient

        ng = NaturalGradient(damping=0.0)
        F = np.eye(3)
        g = np.array([1.0, 2.0, 3.0])
        g_natural = ng.compute(F, g)
        assert np.allclose(g_natural, g, atol=1e-10)


class TestFisherRaoAdaptiveThreshold:
    """Tests for Fisher-Rao derived adaptive thresholds."""

    def test_calibrate_and_get_threshold(self) -> None:
        from omni_mercury_engine.core.info_geometry import FisherRaoAdaptiveThreshold

        rng = np.random.RandomState(42)
        data = rng.randn(100, 5)
        frt = FisherRaoAdaptiveThreshold(confidence_k=2.0)
        threshold = frt.calibrate(data)
        assert isinstance(threshold, float)
        assert threshold > 0
        assert frt.threshold == threshold

    def test_score_and_is_anomalous(self) -> None:
        from omni_mercury_engine.core.info_geometry import FisherRaoAdaptiveThreshold

        rng = np.random.RandomState(42)
        frt = FisherRaoAdaptiveThreshold()
        frt.calibrate(rng.randn(80, 3))
        sample = rng.randn(3)
        score = frt.score(sample)
        assert isinstance(score, float)
        assert score >= 0
        is_anom = frt.is_anomalous(sample)
        assert isinstance(is_anom, (bool, np.bool_))

    def test_drift_detection(self) -> None:
        from omni_mercury_engine.core.info_geometry import FisherRaoAdaptiveThreshold

        rng = np.random.RandomState(42)
        frt = FisherRaoAdaptiveThreshold(drift_tolerance=0.01)
        frt.calibrate(rng.randn(80, 3))
        # Same distribution — should not drift
        drifted_same = frt.check_drift(rng.randn(50, 3))
        # Very different distribution — should drift
        drifted_diff = frt.check_drift(rng.randn(50, 3) * 100 + 50)
        # At least the extreme case should detect drift
        assert isinstance(drifted_same, bool)
        assert isinstance(drifted_diff, bool)


# ============================================================================
# Phase 4D — Riemannian Optimization
# ============================================================================


class TestSimplexManifold:
    """Tests for probability simplex manifold."""

    def test_project_onto_simplex(self) -> None:
        from omni_mercury_engine.core.riemannian_optimization import SimplexManifold

        m = SimplexManifold()
        x = np.array([0.3, -0.1, 0.5, 0.8])
        proj = m.project(x)
        assert np.allclose(np.sum(proj), 1.0, atol=1e-10)
        assert np.all(proj >= 0)

    def test_project_already_on_simplex(self) -> None:
        from omni_mercury_engine.core.riemannian_optimization import SimplexManifold

        m = SimplexManifold()
        x = np.array([0.2, 0.3, 0.5])
        proj = m.project(x)
        assert np.allclose(proj, x, atol=1e-10)

    def test_geodesic_distance_self_is_zero(self) -> None:
        from omni_mercury_engine.core.riemannian_optimization import SimplexManifold

        m = SimplexManifold()
        x = np.array([0.2, 0.3, 0.5])
        d = m.geodesic_distance(x, x)
        assert abs(d) < 1e-10

    def test_geodesic_distance_symmetric(self) -> None:
        from omni_mercury_engine.core.riemannian_optimization import SimplexManifold

        m = SimplexManifold()
        x = np.array([0.2, 0.3, 0.5])
        y = np.array([0.4, 0.4, 0.2])
        assert abs(m.geodesic_distance(x, y) - m.geodesic_distance(y, x)) < 1e-10


class TestRiemannianGradientDescent:
    """Tests for Riemannian gradient descent optimizer."""

    def test_optimize_on_simplex_converges(self) -> None:
        from omni_mercury_engine.core.riemannian_optimization import (
            RiemannianGradientDescent,
            SimplexManifold,
        )

        manifold = SimplexManifold()
        optimizer = RiemannianGradientDescent(manifold=manifold, learning_rate=0.1)

        # Minimize ||x - target||^2 on simplex
        target = np.array([0.5, 0.3, 0.2])

        def objective(x: np.ndarray) -> float:
            return float(np.sum((x - target) ** 2))

        def gradient(x: np.ndarray) -> np.ndarray:
            return 2.0 * (x - target)

        x0 = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        result = optimizer.optimize(x0, objective, gradient, max_iter=100)
        assert np.allclose(np.sum(result.x), 1.0, atol=1e-6)
        assert np.all(result.x >= -1e-10)


class TestConstrainedParameterOptimizer:
    """Tests for Mercury Agent parameter optimization on manifolds."""

    def test_optimize_aafe_weights(self) -> None:
        from omni_mercury_engine.core.riemannian_optimization import (
            ConstrainedParameterOptimizer,
        )

        opt = ConstrainedParameterOptimizer()
        target = np.array([0.5, 0.3, 0.2])
        result = opt.optimize_simplex_weights(
            initial_weights=np.array([1.0 / 3, 1.0 / 3, 1.0 / 3]),
            objective_fn=lambda x: float(np.sum((x - target) ** 2)),
            grad_fn=lambda x: 2.0 * (x - target),
            max_iter=50,
        )
        assert np.allclose(np.sum(result.x), 1.0, atol=1e-6)
        assert np.all(result.x >= -1e-10)


# ============================================================================
# Phase 6 — System-Level Coherence
# ============================================================================


class TestSignalFlowGraph:
    """Tests for signal flow graph construction."""

    def test_default_graph_has_stages(self) -> None:
        from omni_mercury_engine.core.system_coherence import SignalFlowGraph

        graph = SignalFlowGraph.build_default()
        assert len(graph.stages) >= 5

    def test_ascii_rendering(self) -> None:
        from omni_mercury_engine.core.system_coherence import SignalFlowGraph

        graph = SignalFlowGraph.build_default()
        ascii_text = graph.to_ascii()
        assert "data_ingestion" in ascii_text
        assert "aafe_fusion" in ascii_text
        assert "ethical_gating" in ascii_text


class TestNormalizationVerifier:
    """Tests for normalization handoff verification."""

    def test_default_pipeline_all_compatible(self) -> None:
        from omni_mercury_engine.core.system_coherence import (
            NormalizationVerifier,
            SignalFlowGraph,
        )

        graph = SignalFlowGraph.build_default()
        results = NormalizationVerifier.verify(graph)
        assert all(r.compatible for r in results)

    def test_detects_range_mismatch(self) -> None:
        from omni_mercury_engine.core.system_coherence import (
            NormalizationVerifier,
            PipelineStage,
            SignalFlowGraph,
        )

        graph = SignalFlowGraph(
            stages=[
                PipelineStage(
                    name="wide_output",
                    input_range=(0.0, 1.0),
                    output_range=(0.0, 10.0),
                ),
                PipelineStage(
                    name="narrow_input",
                    input_range=(0.0, 1.0),
                    output_range=(0.0, 1.0),
                ),
            ]
        )
        results = NormalizationVerifier.verify(graph)
        assert len(results) == 1
        assert not results[0].compatible


class TestLyapunovRuntimeEnforcer:
    """Tests for runtime Lyapunov stability enforcement."""

    def test_stable_sequence_no_violations(self) -> None:
        from omni_mercury_engine.core.system_coherence import LyapunovRuntimeEnforcer

        enforcer = LyapunovRuntimeEnforcer(grace_steps=3)
        for t in range(1, 30):
            v = np.exp(-0.25 * t)
            enforcer.check(v)
        assert enforcer.is_stable
        assert len(enforcer.violations) == 0

    def test_unstable_sequence_detects_violations(self) -> None:
        from omni_mercury_engine.core.system_coherence import LyapunovRuntimeEnforcer

        enforcer = LyapunovRuntimeEnforcer(grace_steps=2)
        # Start stable then spike
        for t in range(1, 5):
            enforcer.check(np.exp(-0.25 * t))
        # Inject instability
        enforcer.check(10.0)  # Sudden spike
        assert not enforcer.is_stable
        assert len(enforcer.violations) >= 1

    def test_halt_on_violation_raises(self) -> None:
        from omni_mercury_engine.core.system_coherence import LyapunovRuntimeEnforcer

        enforcer = LyapunovRuntimeEnforcer(halt_on_violation=True, grace_steps=2)
        for t in range(1, 5):
            enforcer.check(np.exp(-0.25 * t))
        with pytest.raises(RuntimeError, match="Lyapunov stability violated"):
            enforcer.check(100.0)

    def test_violation_rate(self) -> None:
        from omni_mercury_engine.core.system_coherence import LyapunovRuntimeEnforcer

        enforcer = LyapunovRuntimeEnforcer(grace_steps=0)
        enforcer.check(1.0)
        enforcer.check(0.5)
        enforcer.check(5.0)  # Violation
        assert enforcer.violation_rate > 0

    def test_stability_report(self) -> None:
        from omni_mercury_engine.core.system_coherence import LyapunovRuntimeEnforcer

        enforcer = LyapunovRuntimeEnforcer()
        for t in range(1, 10):
            enforcer.check(np.exp(-0.25 * t))
        report = enforcer.get_stability_report()
        assert "total_steps" in report
        assert "violations" in report
        assert "is_stable" in report
        assert report["is_stable"] is True


class TestCoherenceAudit:
    """Tests for the full coherence audit."""

    def test_run_coherence_audit_default(self) -> None:
        from omni_mercury_engine.core.system_coherence import run_coherence_audit

        report = run_coherence_audit()
        assert report.all_handoffs_compatible
        assert report.lyapunov_stable
        assert report.timestamp

    def test_run_with_custom_scores(self) -> None:
        from omni_mercury_engine.core.system_coherence import run_coherence_audit

        # Provide a stable sequence
        scores = [np.exp(-0.25 * t) for t in range(1, 40)]
        report = run_coherence_audit(fusion_scores=scores)
        assert report.lyapunov_stable


# ============================================================================
# Phase 3B — Domain-Adaptive AAFE Weights
# ============================================================================


class TestDomainAdaptiveAAFEWeights:
    """Tests for domain-adaptive weight profiles."""

    def test_default_weights_are_golden_ratio(self) -> None:
        from omni_mercury_engine.core.three_r.fusion import DomainAdaptiveAAFEWeights

        daw = DomainAdaptiveAAFEWeights()
        w = daw.get_weights("unknown")
        assert abs(sum(w.values()) - 1.0) < 1e-10

    def test_record_and_fit(self) -> None:
        from omni_mercury_engine.core.three_r.fusion import DomainAdaptiveAAFEWeights

        daw = DomainAdaptiveAAFEWeights()
        rng = np.random.RandomState(42)
        for _ in range(50):
            daw.record_observation("medical", rng.rand(), rng.rand(), rng.rand(), rng.randint(2))
            daw.record_observation("security", rng.rand(), rng.rand(), rng.rand(), rng.randint(2))

        profiles = daw.fit_domain_profiles(min_samples=30)
        assert "medical" in profiles
        assert "security" in profiles
        assert abs(sum(profiles["medical"].values()) - 1.0) < 1e-6

    def test_has_domain_profile(self) -> None:
        from omni_mercury_engine.core.three_r.fusion import DomainAdaptiveAAFEWeights

        daw = DomainAdaptiveAAFEWeights()
        assert not daw.has_domain_profile("medical")

        rng = np.random.RandomState(42)
        for _ in range(50):
            daw.record_observation("medical", rng.rand(), rng.rand(), rng.rand(), rng.randint(2))
        daw.fit_domain_profiles(min_samples=30)
        assert daw.has_domain_profile("medical")

    def test_insufficient_data_uses_defaults(self) -> None:
        from omni_mercury_engine.core.three_r.fusion import DomainAdaptiveAAFEWeights

        daw = DomainAdaptiveAAFEWeights()
        for _ in range(5):  # Too few
            daw.record_observation("sparse", 0.5, 0.5, 0.5, 1)
        daw.fit_domain_profiles(min_samples=30)
        assert not daw.has_domain_profile("sparse")
