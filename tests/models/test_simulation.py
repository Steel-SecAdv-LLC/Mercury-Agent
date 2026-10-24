# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for SimulationModule (paradoxes, conjectures, Millennium Prize Problems)."""

from __future__ import annotations

import numpy as np
import pytest

import omni_mercury_engine.models.simulation as simulation_module
from omni_mercury_engine.models.simulation import SimulationModule


class TestSimulationModule:
    """Test suite for SimulationModule."""

    def test_simulation_instantiation(self) -> None:
        """Test simulation module can be instantiated."""
        sim = SimulationModule()
        assert sim is not None
        assert sim.num_branches == 10
        assert sim.embedding_dim == 128

    def test_simulation_with_config(self) -> None:
        """Test simulation module with custom config."""
        config = {"num_branches": 20, "embedding_dim": 256}
        sim = SimulationModule(config=config)
        assert sim.num_branches == 20
        assert sim.embedding_dim == 256

    def test_simulate_zeno_paradox(self) -> None:
        """Test Zeno's paradox simulation."""
        sim = SimulationModule()
        result = sim.simulate_paradox("zeno", iterations=100)

        assert "paradox_type" in result
        assert result["paradox_type"] == "zeno"
        assert "resolution_attempts" in result
        assert result["resolution_attempts"] == 100
        assert "resolution_quality" in result
        assert 0 <= result["resolution_quality"] <= 1
        assert "insights" in result
        assert len(result["insights"]) > 0

    def test_simulate_liar_paradox(self) -> None:
        """Test Epimenides (Liar) paradox simulation."""
        sim = SimulationModule()
        result = sim.simulate_paradox("epimenides", iterations=50)

        assert result["paradox_type"] == "epimenides_liar"
        assert "oscillation_pattern" in result
        assert "ethical_flags" in result

    def test_simulate_russell_paradox(self) -> None:
        """Test Russell's paradox simulation."""
        sim = SimulationModule()
        result = sim.simulate_paradox("russell", iterations=100)

        assert result["paradox_type"] == "russell"
        assert "contradictions_detected" in result
        assert "ethical_flags" in result

    def test_explore_collatz_conjecture(self) -> None:
        """Test Collatz conjecture exploration."""
        sim = SimulationModule()
        result = sim.explore_conjecture("collatz", search_space=1000)

        assert "conjecture" in result
        assert result["conjecture"] == "collatz"
        assert "explored_cases" in result
        assert "all_reached_one" in result
        assert "viability_score" in result
        assert "insights" in result

    def test_explore_twin_prime_conjecture(self) -> None:
        """Test Twin Prime conjecture exploration."""
        sim = SimulationModule()
        result = sim.explore_conjecture("twin_prime", search_space=1000)

        assert result["conjecture"] == "twin_prime"
        assert "twin_primes_found" in result
        assert result["twin_primes_found"] > 0
        assert "viability_score" in result

    def test_explore_goldbach_conjecture(self) -> None:
        """Test Goldbach's conjecture exploration."""
        sim = SimulationModule()
        result = sim.explore_conjecture("goldbach", search_space=1000)

        assert result["conjecture"] == "goldbach"
        assert "supporting_cases" in result
        assert "counterexamples" in result

    def test_analyze_navier_stokes_dates_unverified_claims(self) -> None:
        """Claimed-but-unverified results are recorded as claims, never as facts.

        Fails on the replaced output, which had no ``claimed_results``
        record at all.
        """
        sim = SimulationModule()
        result = sim.analyze_millennium_problem("navier_stokes")

        assert result["status"] == "unsolved"
        claims = result["claimed_results"]
        assert len(claims) >= 1
        assert claims[0]["status"] == "claimed_unverified"
        assert "2026" in claims[0]["source"]

    def test_analyze_p_vs_np(self) -> None:
        """Test P vs NP analysis."""
        sim = SimulationModule()
        result = sim.analyze_millennium_problem("p_vs_np")

        assert result["problem"] == "p_vs_np"
        assert result["status"] == "unsolved"
        assert "complexity_gap" in result
        assert "ethical_flags" in result

    def test_analyze_poincare_conjecture(self) -> None:
        """Test Poincaré conjecture analysis (SOLVED)."""
        sim = SimulationModule()
        result = sim.analyze_millennium_problem("poincare")

        assert result["problem"] == "poincare_conjecture"
        assert result["status"] == "SOLVED"
        assert "solved_by" in result
        assert result["solved_by"] == "Grigori Perelman"

    def test_riemann_zeros_are_real_not_simulated(self) -> None:
        """The Riemann path measures actual zeta zeros, never RNG output.

        Fails on the replaced implementation, which fabricated ordinates
        at 14.134725 + 10k (the real second ordinate is 21.02, not 24.13) and
        had no ``oracle`` key.
        """
        sim = SimulationModule()
        result = sim.explore_conjecture("riemann", search_space=5)

        assert result["oracle"] == "mpmath"
        assert result["explored_cases"] == 5
        known_ordinates = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062]
        for got, want in zip(result["first_zero_ordinates"], known_ordinates, strict=True):
            assert got == pytest.approx(want, abs=1e-5)

    def test_riemann_strip_count_cross_check(self) -> None:
        """Turing-style check: zeros found on the line == zeros in the strip.

        Fails on the replaced implementation: no ``strip_zero_count`` or
        ``max_zeta_residual`` keys, and its RNG deviations made
        ``viability_score`` land near 0.9992, never exactly 1.0 with a zero
        ``max_deviation_from_half``.
        """
        sim = SimulationModule()
        result = sim.explore_conjecture("riemann", search_space=5)

        assert result["strip_zero_count"] == 5
        assert result["zeros_on_critical_line"] == 5
        assert result["viability_score"] == 1.0
        assert result["max_deviation_from_half"] == 0.0
        assert result["max_zeta_residual"] < 1e-8

    def test_riemann_fails_closed_without_mpmath(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No oracle means zero explored cases and zero viability — no fabrication.

        Fails on the replaced implementation, which needed no oracle and
        happily reported 5 fabricated zeros with viability near 1.0.
        """
        monkeypatch.setattr(simulation_module, "MPMATH_AVAILABLE", False)
        result = SimulationModule().explore_conjecture("riemann", search_space=5)

        assert result["oracle"] == "unavailable"
        assert result["explored_cases"] == 0
        assert result["zeros_on_critical_line"] == 0
        assert result["viability_score"] == 0.0

    def test_riemann_nonpositive_search_space(self) -> None:
        """A search space asking for no zeros reports none — it must not crash.

        Fails on the replaced implementation, which raised ``ValueError``
        from ``max()`` on an empty deviations list.
        """
        result = SimulationModule().explore_conjecture("riemann", search_space=0)

        assert result["explored_cases"] == 0
        assert result["viability_score"] == 0.0

    def test_extract_features(self) -> None:
        """Test feature extraction."""
        sim = SimulationModule()
        data = np.random.randn(10, 20)

        features = sim.extract_features(data)

        assert features.shape[0] == 10
        assert features.shape[1] == sim.embedding_dim
        assert features.dtype == np.float32

    def test_predict(self) -> None:
        """Test multiverse branching prediction."""
        sim = SimulationModule()
        data = np.random.randn(5, 10)

        result = sim.predict(data)

        assert "anomaly_scores" in result
        assert "branch_predictions" in result
        assert "branch_variance" in result
        assert "num_branches_explored" in result
        assert result["num_branches_explored"] == sim.num_branches
        assert "ethical_risk_detected" in result
