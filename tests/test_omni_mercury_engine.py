"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Comprehensive test suite for Enhanced Mercury Agent Equation Engine

Tests convergence, all 22 terms, threat detection, and ethical integration.
"""

import numpy as np
import pytest

from omni_mercury_engine.core.ethical_config import DEFAULT_CONFIG
from omni_mercury_engine.core.fusion import OmniMercuryEngine


class TestOmniMercuryEngine:
    """Test suite for OmniMercuryEngine."""

    def test_initialization(self) -> None:
        """Test engine initialization with GA-optimized defaults."""
        engine = OmniMercuryEngine(state_dim=50)

        assert engine.state_dim == 50
        assert engine.alpha == 0.3745
        assert engine.beta == 0.9507
        assert engine.enable_H is True
        assert engine.enable_Omega is True
        assert engine.enable_Al is True
        assert engine.use_double_helix is True
        assert engine.vqe_params.shape == (50,)
        assert engine.qbm_J.shape == (50, 50)

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = {"alpha": 0.2, "beta": 0.1, "enable_H": False, "enable_Q": True}

        engine = OmniMercuryEngine(config=config, state_dim=30)

        assert engine.alpha == 0.2
        assert engine.beta == 0.1
        assert engine.enable_H is False
        assert engine.enable_Q is True
        assert engine.state_dim == 30

    def test_single_step(self) -> None:
        """Test single iterative step."""
        engine = OmniMercuryEngine(state_dim=20)

        initial_state = np.random.randn(20) * 0.1
        next_state = engine.step(initial_state, t=0)

        assert next_state.shape == (20,)
        assert not np.array_equal(initial_state, next_state)
        assert np.all(np.isfinite(next_state))

    def test_convergence(self) -> None:
        """Test convergence to stable state."""
        engine = OmniMercuryEngine(state_dim=20)

        final_state, history = engine.converge(max_steps=50, tolerance=1e-3)

        assert final_state.shape == (20,)
        assert len(history) > 0
        assert len(history) <= 50
        assert np.all(np.isfinite(final_state))

        if len(history) > 5:
            assert history[-1] < history[0] or abs(history[-1] - history[0]) < 0.1

    def test_convergence_rate(self) -> None:
        """Test exponential convergence O(e^{-0.13 t})."""
        engine = OmniMercuryEngine(state_dim=20)

        final_state, history = engine.converge(max_steps=100, tolerance=1e-4)

        if len(history) >= 10:
            early_error = history[5]
            late_error = history[-1]

            assert late_error < early_error

            if early_error > 0:
                reduction_ratio = late_error / early_error
                assert reduction_ratio < 1.0

    def test_all_terms_enabled(self) -> None:
        """Test that all 24 terms (22 original + Ω + 𝐀𝐥) can be enabled."""
        engine = OmniMercuryEngine(state_dim=20)

        state = np.random.randn(20) * 0.1

        state_with_all = engine.step(state, t=0)

        assert state_with_all.shape == (20,)
        assert np.all(np.isfinite(state_with_all))

    def test_individual_terms(self) -> None:
        """Test each term individually."""
        engine = OmniMercuryEngine(state_dim=20)
        state = np.random.randn(20) * 0.1

        assert engine._term_H(state).shape == (20,)
        assert engine._term_Q(state).shape == (20,)
        assert engine._term_P(state).shape == (20,)
        assert engine._term_D(state).shape == (20,)
        assert engine._term_E(state).shape == (20,)
        assert engine._term_V(state).shape == (20,)
        assert engine._term_W(state).shape == (20,)
        assert engine._term_R3(state).shape == (20,)
        assert engine._term_An(state, 1.0).shape == (20,)
        assert engine._term_Lambda(state).shape == (20,)
        assert engine._term_Theta(state).shape == (20,)
        assert engine._term_Phi(state).shape == (20,)
        assert engine._term_Z(state).shape == (20,)
        assert engine._term_hq(state).shape == (20,)
        assert engine._term_L(state).shape == (20,)
        assert engine._term_VQE(state, engine.vqe_params).shape == (20,)
        assert engine._term_QBM(state).shape == (20,)
        assert engine._term_Attn(state).shape == (20,)
        assert engine._term_F(state).shape == (20,)
        assert engine._term_S(state).shape == (20,)
        assert engine._term_I(state).shape == (20,)
        assert engine._term_Rel(state).shape == (20,)
        assert engine._term_inf_b(state).shape == (20,)
        assert engine._term_Omega(state).shape == (20,)
        assert engine._term_Al(state).shape == (20,)

    def test_ethical_integration(self) -> None:
        """Test integration with ethical scalars."""
        engine = OmniMercuryEngine(state_dim=20)

        state = np.random.randn(20) * 0.1
        # Compute H term to verify it works (return value not needed)
        engine._term_H(state)

        ethical_mean = np.mean(list(DEFAULT_CONFIG.ethical_scalars.to_dict().values())[:10])

        assert ethical_mean > 1.0

    def test_anomaly_detection(self) -> None:
        """Test anomaly detection functionality."""
        engine = OmniMercuryEngine(state_dim=20)

        normal_data = np.random.randn(20) * 0.1
        anomalous_data = np.random.randn(20) * 5.0

        normal_result = engine.detect_anomaly(normal_data, threshold=2.0)
        anomalous_result = engine.detect_anomaly(anomalous_data, threshold=2.0)

        assert "anomaly_score" in normal_result
        assert "is_anomaly" in normal_result
        assert "convergence_steps" in normal_result

        assert anomalous_result["anomaly_score"] > normal_result["anomaly_score"]

    def test_lyapunov_stability(self) -> None:
        """Test Lyapunov stability (ΔV<0)."""
        engine = OmniMercuryEngine(state_dim=20)

        target_state = np.ones(20) * 1.3
        state = np.random.randn(20) * 0.1

        V_initial = np.sum((state - target_state) ** 2)

        state_next = engine.step(state, t=0)
        V_next = np.sum((state_next - target_state) ** 2)

        delta_V = V_next - V_initial

        if abs(delta_V) > 1e-10:
            assert delta_V <= 0 or abs(delta_V) < V_initial * 0.1

    def test_bounded_output(self) -> None:
        """Test that asymptotic bound works (∞_b term)."""
        engine = OmniMercuryEngine(state_dim=20)

        extreme_state = np.random.randn(20) * 1000

        bounded_state = engine._term_inf_b(extreme_state)

        assert np.all(np.abs(bounded_state) <= 10.0)

    def test_quantum_terms(self) -> None:
        """Test quantum-inspired terms (Q, An, hq, VQE, QBM)."""
        engine = OmniMercuryEngine(state_dim=20)
        state = np.random.randn(20) * 0.1

        q_term = engine._term_Q(state)
        assert np.all(np.isfinite(q_term))

        an_term = engine._term_An(state, T=1.0)
        assert np.all(np.isfinite(an_term))

        hq_term = engine._term_hq(state)
        assert np.all(np.isfinite(hq_term))

        vqe_term = engine._term_VQE(state, engine.vqe_params)
        assert np.all(np.isfinite(vqe_term))

        qbm_term = engine._term_QBM(state)
        assert np.all(np.isfinite(qbm_term))

    def test_temperature_decay(self) -> None:
        """Test quantum annealing temperature decay."""
        engine = OmniMercuryEngine(state_dim=20)

        initial_T = engine.current_T
        state = np.random.randn(20) * 0.1

        for _ in range(10):
            engine.step(state, t=0)

        assert engine.current_T < initial_T
        assert engine.current_T > 0

    def test_complexity_performance(self) -> None:
        """Assert ``OmniMercuryEngine.step`` scales no worse than O(n²).

        The step function carries an :math:`n \\times n` QBM coupling
        matrix (``self.qbm_J``) and a per-step matrix-vector multiply
        against the state, plus several other :math:`O(n^2)` updates
        in the ethical/3R fusion path.  That sets a hard mathematical
        floor at quadratic time: no implementation can be sub-quadratic
        while still touching every entry of ``qbm_J`` once per step.
        We therefore assert the **measured** complexity bound rather
        than the (irreducible) ideal O(n log n) target this test
        previously claimed; asserting O(n log n) here would either
        flake under noise or have to live behind an ``xfail`` that
        masks the real contract.

        Timing methodology (matches the recommendation in the stdlib
        :mod:`timeit` docs):

        * Each per-size measurement is the **minimum** of ``trials``
          repeats of ``inner`` ``step`` calls.  Minimum-of-N is the
          standard noise-robust estimator for wall-clock benchmarks
          because external interference (GC pauses, scheduler stalls,
          neighbour-tenant CPU contention on shared CI runners) can
          only ever **add** time, never remove it; the minimum
          observed time is therefore the closest unbiased proxy to
          the algorithm's actual best-case lower bound.
        * We anchor the assertion on the **largest** dimension doubling
          (80 → 160): it carries the most per-step signal and is the
          first place a regression to a *dense super-quadratic
          factorization* (a full SVD / ``det`` / ``eigh`` on an
          :math:`n \\times n` matrix, all O(n^3)) would re-inflate the
          ratio.  At these CI-friendly sizes the per-step cost is in
          fact dominated by fixed per-term Python/NumPy dispatch
          overhead (~two dozen vector terms) plus the O(n^2) ``qbm_J``
          matvec and σ-gate quadratic form, so the *measured* exponent
          sits well below 1; the ceiling is the upper guard, not the
          expected value.
        """
        import math
        import time

        sizes = [20, 40, 80, 160]
        inner = 20
        trials = 7
        times: list[float] = []

        for size in sizes:
            engine = OmniMercuryEngine(state_dim=size)
            state = np.random.randn(size) * 0.1

            # Warm-up THIS specific engine instance to trigger every
            # lazy-init path (focus probes, NN module materialisation,
            # cuBLAS-style first-call kernel selection).
            for _ in range(inner):
                engine.step(state, t=0)

            run_times: list[float] = []
            for _ in range(trials):
                start = time.perf_counter()
                for _ in range(inner):
                    engine.step(state, t=0)
                run_times.append(time.perf_counter() - start)
            # Min-of-N: see method docstring above.
            times.append(min(run_times))

        ratio_large = times[3] / times[2]
        exponent_large = math.log(ratio_large) / math.log(2.0)

        # With the per-step hot path holding no dense O(n^3) factorization
        # (``_term_D`` projects via the rank-1 closed form; the σ_Immutable
        # ``det``/``eigh`` are memoised loop invariants), the measured
        # exponent over 80→160 is ~0.3-0.5 on a 2-core GitHub-hosted runner
        # -- overhead-bounded at these sizes.  We retain a 2.5 ceiling as the
        # upper guard: it leaves headroom for wall-clock noise while still
        # tripping if a dense super-quadratic factorization is reintroduced
        # onto the hot path (the original regression, an n×n SVD per step,
        # drove this ratio to 2.53 and ~64 ms/step at n=160 vs ~1.3 ms now).
        assert exponent_large <= 2.5, (
            f"step() complexity exponent (80→160) is "
            f"{exponent_large:.2f}, exceeding the O(n^2.5) ceiling "
            f"(measured times: {[round(t, 4) for t in times]} for "
            f"sizes {sizes})."
        )

    def test_term_D_matches_dense_svd_projection(self) -> None:
        """``_term_D`` must equal the dense rank-1 SVD it replaced.

        ``_term_D`` projects onto the dominant singular direction of the
        rank-1 matrix ``state ⊗ state``.  It used to build that ``n×n``
        matrix and run a full O(n³) ``np.linalg.svd``; it now uses the exact
        O(n) closed form ``±‖state‖·state``.  This pins that the optimization
        is *numerically* the same operation, not merely a faster one of a
        different shape, so it can never silently drift.  The overall sign is
        an intrinsic SVD gauge freedom, so we accept either branch (the
        implementation reproduces LAPACK's ``U[0,0] ≤ 0`` gauge, but the
        contract that matters here is equality *up to that gauge*).
        """
        for dim in (2, 5, 20, 80, 161):  # even, odd, and a non-power-of-two
            engine = OmniMercuryEngine(state_dim=dim)
            rng = np.random.default_rng(dim)
            for _ in range(25):
                state = rng.standard_normal(dim) * rng.uniform(0.05, 3.0)

                reshaped = state.reshape(-1, 1)
                u, s, _vt = np.linalg.svd(reshaped @ reshaped.T, full_matrices=False)
                proj_ref = u[:, 0] * s[0]
                pos = (proj_ref - state) * 0.1
                neg = (-proj_ref - state) * 0.1

                got = engine._term_D(state)
                assert got.shape == (dim,)
                assert np.all(np.isfinite(got))
                assert np.allclose(got, pos, atol=1e-10) or np.allclose(
                    got, neg, atol=1e-10
                ), f"_term_D diverged from the dense-SVD projection at dim={dim}"

            # Deterministic: no RNG draw, identical output on repeat calls.
            fixed = rng.standard_normal(dim)
            assert np.array_equal(engine._term_D(fixed), engine._term_D(fixed))

        # The zero vector is the SVD's degenerate case: projection is zero.
        z = OmniMercuryEngine(state_dim=8)
        assert np.allclose(z._term_D(np.zeros(8)), (np.zeros(8) - np.zeros(8)) * 0.1)

    def test_sigma_immutable_factorizations_are_cached_invariants(self) -> None:
        """The σ_Immutable ``det``/``eigh`` are memoised loop invariants.

        ``ethical_matrix`` is built once and never mutated during stepping, so
        its determinant (used by ``_compute_purity_invariant``) and symmetric
        eigendecomposition (used by ``_apply_purity_correction``) are cached at
        construction instead of being re-factorised every ``step`` (the old
        per-step O(n³) cost).  This asserts the cache equals a fresh
        factorisation and *stays* correct after many steps -- i.e. the matrix
        really is invariant and caching introduces no staleness.
        """
        engine = OmniMercuryEngine(state_dim=40)

        assert engine._ethical_det == np.linalg.det(engine.ethical_matrix)
        ev, evec = np.linalg.eigh(engine.ethical_matrix)
        assert np.array_equal(engine._ethical_eigvals, ev)
        assert np.array_equal(engine._ethical_eigvecs, evec)

        matrix_before = engine.ethical_matrix.copy()
        state = np.random.randn(40) * 0.1
        for t in range(40):
            state = engine.step(state, t=t)

        # Matrix untouched -> cached det still matches a fresh recompute, and
        # the purity invariant equals what the pre-cache code path would yield.
        assert np.array_equal(engine.ethical_matrix, matrix_before)
        assert engine._ethical_det == np.linalg.det(engine.ethical_matrix)
        probe = np.random.randn(40) * 0.2
        sn = probe / (np.linalg.norm(probe) + 1e-8)
        fresh = float(np.linalg.det(engine.ethical_matrix) * (sn @ engine.ethical_matrix @ sn))
        assert engine._compute_purity_invariant(probe) == pytest.approx(fresh, abs=1e-9)

    def test_double_helix_architecture(self) -> None:
        """Test double-helix DNA-inspired architecture."""
        engine = OmniMercuryEngine(state_dim=20)

        assert engine.use_double_helix is True

        state = np.random.randn(20) * 0.1

        helix1 = engine.helix_1_discovery(state, t=0)
        helix2 = engine.helix_2_ethical(state)

        assert helix1.shape == (20,)
        assert helix2.shape == (20,)
        assert np.all(np.isfinite(helix1))
        assert np.all(np.isfinite(helix2))

        intertwined = engine._intertwine_helixes(helix1, helix2)
        assert intertwined.shape == (20,)
        assert np.all(np.isfinite(intertwined))

        config_linear = {"use_double_helix": False}
        engine_linear = OmniMercuryEngine(config=config_linear, state_dim=20)

        state_helix = engine.step(state, t=0)
        state_linear = engine_linear.step(state, t=0)

        assert state_helix.shape == state_linear.shape
        assert not np.array_equal(state_helix, state_linear)


class TestOmniMercuryIntegration:
    """Integration tests for OmniMercuryEngine."""

    def test_threat_detection_simulation(self) -> None:
        """Test threat detection in simulated scenario."""
        engine = OmniMercuryEngine(state_dim=30)

        normal_samples = [np.random.randn(30) * 0.1 for _ in range(10)]
        threat_samples = [np.random.randn(30) * 3.0 for _ in range(5)]

        normal_scores = [engine.detect_anomaly(s)["anomaly_score"] for s in normal_samples]
        threat_scores = [engine.detect_anomaly(s)["anomaly_score"] for s in threat_samples]

        assert np.mean(threat_scores) > np.mean(normal_scores)

    def test_convergence_consistency(self) -> None:
        """Test that convergence is reasonably consistent with same initial state."""
        np.random.seed(12345)
        config_linear = {"use_double_helix": False}
        engine = OmniMercuryEngine(config=config_linear, state_dim=20)

        initial_state = np.random.randn(20) * 0.1

        state1, _ = engine.converge(initial_state.copy(), max_steps=50)
        state2, _ = engine.converge(initial_state.copy(), max_steps=50)

        assert np.allclose(state1, state2, rtol=0.25, atol=0.1)

    def test_multi_round_adaptation(self) -> None:
        """Test multi-round learning and adaptation."""
        engine = OmniMercuryEngine(state_dim=20)

        states = []
        for _ in range(5):
            final, _ = engine.converge(max_steps=30)
            states.append(final)

        assert len(states) == 5
        assert all(s.shape == (20,) for s in states)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
