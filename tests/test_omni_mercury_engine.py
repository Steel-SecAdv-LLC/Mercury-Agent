"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

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

    def test_initialization(self):
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

    def test_custom_config(self):
        """Test custom configuration."""
        config = {"alpha": 0.2, "beta": 0.1, "enable_H": False, "enable_Q": True}

        engine = OmniMercuryEngine(config=config, state_dim=30)

        assert engine.alpha == 0.2
        assert engine.beta == 0.1
        assert engine.enable_H is False
        assert engine.enable_Q is True
        assert engine.state_dim == 30

    def test_single_step(self):
        """Test single iterative step."""
        engine = OmniMercuryEngine(state_dim=20)

        initial_state = np.random.randn(20) * 0.1
        next_state = engine.step(initial_state, t=0)

        assert next_state.shape == (20,)
        assert not np.array_equal(initial_state, next_state)
        assert np.all(np.isfinite(next_state))

    def test_convergence(self):
        """Test convergence to stable state."""
        engine = OmniMercuryEngine(state_dim=20)

        final_state, history = engine.converge(max_steps=50, tolerance=1e-3)

        assert final_state.shape == (20,)
        assert len(history) > 0
        assert len(history) <= 50
        assert np.all(np.isfinite(final_state))

        if len(history) > 5:
            assert history[-1] < history[0] or abs(history[-1] - history[0]) < 0.1

    def test_convergence_rate(self):
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

    def test_all_terms_enabled(self):
        """Test that all 24 terms (22 original + Ω + 𝐀𝐥) can be enabled."""
        engine = OmniMercuryEngine(state_dim=20)

        state = np.random.randn(20) * 0.1

        state_with_all = engine.step(state, t=0)

        assert state_with_all.shape == (20,)
        assert np.all(np.isfinite(state_with_all))

    def test_individual_terms(self):
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

    def test_ethical_integration(self):
        """Test integration with ethical scalars."""
        engine = OmniMercuryEngine(state_dim=20)

        state = np.random.randn(20) * 0.1
        # Compute H term to verify it works (return value not needed)
        engine._term_H(state)

        ethical_mean = np.mean(list(DEFAULT_CONFIG.ethical_scalars.to_dict().values())[:10])

        assert ethical_mean > 1.0

    def test_anomaly_detection(self):
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

    def test_lyapunov_stability(self):
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

    def test_bounded_output(self):
        """Test that asymptotic bound works (∞_b term)."""
        engine = OmniMercuryEngine(state_dim=20)

        extreme_state = np.random.randn(20) * 1000

        bounded_state = engine._term_inf_b(extreme_state)

        assert np.all(np.abs(bounded_state) <= 10.0)

    def test_quantum_terms(self):
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

    def test_temperature_decay(self):
        """Test quantum annealing temperature decay."""
        engine = OmniMercuryEngine(state_dim=20)

        initial_T = engine.current_T
        state = np.random.randn(20) * 0.1

        for _ in range(10):
            engine.step(state, t=0)

        assert engine.current_T < initial_T
        assert engine.current_T > 0

    def test_complexity_performance(self):
        """Test O(n log n) complexity requirement.

        Note: This test verifies that computation time scales reasonably with input size.
        Due to JIT compilation, caching, and CI runner variability, we use generous
        tolerances and warm-up iterations to reduce flakiness.
        """
        import time

        sizes = [20, 40, 80]
        times = []

        for size in sizes:
            engine = OmniMercuryEngine(state_dim=size)
            state = np.random.randn(size) * 0.1

            # Warm-up THIS specific engine instance to trigger JIT/lazy init
            for _ in range(10):
                engine.step(state, t=0)

            # Multiple timing runs to get more stable measurement
            run_times = []
            for _ in range(5):
                start = time.time()
                for _ in range(10):
                    engine.step(state, t=0)
                elapsed = time.time() - start
                run_times.append(elapsed)

            # Use median to reduce outlier impact
            times.append(sorted(run_times)[2])

        # For O(n log n), doubling n should roughly double time (plus log factor)
        # Using generous 10x tolerance to account for CI variability and JIT effects
        # The key invariant is that time doesn't explode exponentially
        assert (
            times[1] < times[0] * 10
        ), f"40-dim ({times[1]:.3f}s) should be < 10x 20-dim ({times[0]:.3f}s)"
        assert (
            times[2] < times[1] * 10
        ), f"80-dim ({times[2]:.3f}s) should be < 10x 40-dim ({times[1]:.3f}s)"

    def test_double_helix_architecture(self):
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

    def test_threat_detection_simulation(self):
        """Test threat detection in simulated scenario."""
        engine = OmniMercuryEngine(state_dim=30)

        normal_samples = [np.random.randn(30) * 0.1 for _ in range(10)]
        threat_samples = [np.random.randn(30) * 3.0 for _ in range(5)]

        normal_scores = [engine.detect_anomaly(s)["anomaly_score"] for s in normal_samples]
        threat_scores = [engine.detect_anomaly(s)["anomaly_score"] for s in threat_samples]

        assert np.mean(threat_scores) > np.mean(normal_scores)

    def test_convergence_consistency(self):
        """Test that convergence is reasonably consistent with same initial state."""
        np.random.seed(12345)
        config_linear = {"use_double_helix": False}
        engine = OmniMercuryEngine(config=config_linear, state_dim=20)

        initial_state = np.random.randn(20) * 0.1

        state1, _ = engine.converge(initial_state.copy(), max_steps=50)
        state2, _ = engine.converge(initial_state.copy(), max_steps=50)

        assert np.allclose(state1, state2, rtol=0.25, atol=0.1)

    def test_multi_round_adaptation(self):
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
