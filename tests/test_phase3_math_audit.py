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

Comprehensive unit tests for Phase 3 mathematical audit changes:
    1. Sigmoid Benevolence Gate
    2. Banach Recursion convergence bounds
    3. Domain-Adaptive Harmonics
    4. Hierarchical Omni-Scalar Aggregation
    5. OAE Enhancements (configurable exponent, NaN guard, benevolence)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.core.centralized_constants import (
    BENEVOLENCE_GATE,
    DOMAIN_HARMONICS,
    RECURSION,
    get_domain_fundamentals,
    sigmoid_benevolence_gate,
)
from omni_mercury_engine.core.global_omni_scalar_network import (
    GlobalOmniScalarNetwork,
    reset_global_network,
)
from omni_mercury_engine.core.three_r.fusion import (
    OmniAvaEquation,
    BanachRecursion,
)

# ==========================================================================
# Fixtures
# ==========================================================================


@pytest.fixture(autouse=True)
def _reset_gosnn() -> Any:
    """Reset the GlobalOmniScalarNetwork singleton before and after each test.

    This prevents singleton state from leaking between tests.
    """
    reset_global_network()
    yield
    reset_global_network()


# ==========================================================================
# 1. Sigmoid Benevolence Gate
# ==========================================================================


class TestSigmoidBenevolenceGate:
    """Tests for sigmoid_benevolence_gate() from centralized_constants."""

    def test_output_in_open_unit_interval(self) -> None:
        """Gate output must lie strictly in (0, 1) for any finite input."""
        for score in [0.0, 0.25, 0.5, 0.75, 0.9, 0.93, 0.95, 0.99, 1.0]:
            result: float = sigmoid_benevolence_gate(score)
            assert (
                0.0 < result < 1.0
            ), f"sigmoid_benevolence_gate({score}) = {result} is not in (0, 1)"

    def test_high_benevolence_yields_high_gate(self) -> None:
        """A benevolence score well above b0 should produce a gate near 1.

        For medical (b0=0.93, k=30), at b=1.0:
            exponent = -30*(1.0 - 0.93) = -2.1
            gate = 1/(1+exp(2.1)) ~ 0.891
        So the gate is high relative to 0.5, but not necessarily > 0.95 due to
        the relatively short distance from b0 and moderate steepness.
        """
        result: float = sigmoid_benevolence_gate(1.0, domain="medical")
        assert result > 0.85

    def test_low_benevolence_yields_low_gate(self) -> None:
        """A benevolence score well below b0 should produce a gate near 0."""
        result: float = sigmoid_benevolence_gate(0.5, domain="medical")
        assert result < 0.05

    @pytest.mark.parametrize(
        "domain",
        ["medical", "security", "environmental", "humanitarian", "infrastructure"],
    )
    def test_domain_profiles_exist(self, domain: str) -> None:
        """Each known domain should produce a valid gate value without error."""
        result: float = sigmoid_benevolence_gate(0.93, domain=domain)
        assert 0.0 < result < 1.0

    def test_medical_profile_parameters(self) -> None:
        """Medical profile uses b0=0.93, k=30."""
        profile = BENEVOLENCE_GATE.MEDICAL
        assert profile.b0 == pytest.approx(0.93)
        assert profile.k == pytest.approx(30.0)
        assert profile.label == "Medical"

    def test_security_profile_parameters(self) -> None:
        """Security profile uses b0=0.95, k=25."""
        profile = BENEVOLENCE_GATE.SECURITY
        assert profile.b0 == pytest.approx(0.95)
        assert profile.k == pytest.approx(25.0)

    def test_environmental_profile_parameters(self) -> None:
        """Environmental profile uses b0=0.90, k=20."""
        profile = BENEVOLENCE_GATE.ENVIRONMENTAL
        assert profile.b0 == pytest.approx(0.90)
        assert profile.k == pytest.approx(20.0)

    def test_humanitarian_profile_parameters(self) -> None:
        """Humanitarian profile uses b0=0.92, k=35."""
        profile = BENEVOLENCE_GATE.HUMANITARIAN
        assert profile.b0 == pytest.approx(0.92)
        assert profile.k == pytest.approx(35.0)

    def test_infrastructure_profile_parameters(self) -> None:
        """Infrastructure profile uses b0=0.94, k=25."""
        profile = BENEVOLENCE_GATE.INFRASTRUCTURE
        assert profile.b0 == pytest.approx(0.94)
        assert profile.k == pytest.approx(25.0)

    def test_unknown_domain_uses_default_profile(self) -> None:
        """An unknown domain string should fall back to the DEFAULT profile."""
        default_val: float = sigmoid_benevolence_gate(0.93, domain="default")
        unknown_val: float = sigmoid_benevolence_gate(0.93, domain="unknown_xyz")
        assert default_val == pytest.approx(unknown_val)

    def test_case_insensitive_domain(self) -> None:
        """Domain lookup should be case-insensitive."""
        lower: float = sigmoid_benevolence_gate(0.93, domain="medical")
        upper: float = sigmoid_benevolence_gate(0.93, domain="MEDICAL")
        mixed: float = sigmoid_benevolence_gate(0.93, domain="Medical")
        assert lower == pytest.approx(upper)
        assert lower == pytest.approx(mixed)

    def test_overflow_protection_large_positive(self) -> None:
        """Very large positive input should not raise and should return near 1.

        The exponent is clamped to [-500, 500], so exp(-500) underflows to 0.0
        and the result becomes exactly 1.0 in floating point. This is acceptable;
        the key property is no exception and a value >= the theoretical limit.
        """
        result: float = sigmoid_benevolence_gate(1e6)
        assert result >= 0.99
        assert result <= 1.0

    def test_overflow_protection_large_negative(self) -> None:
        """Very large negative input should not raise and should return near 0."""
        result: float = sigmoid_benevolence_gate(-1e6)
        assert 0.0 < result < 1.0
        assert result < 0.01

    def test_monotonically_increasing(self) -> None:
        """The sigmoid gate must be monotonically increasing in benevolence_score."""
        scores: list[float] = [i / 100.0 for i in range(101)]
        values: list[float] = [sigmoid_benevolence_gate(s) for s in scores]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1], (
                f"Monotonicity violated: gate({scores[i]})={values[i]} > "
                f"gate({scores[i+1]})={values[i+1]}"
            )

    @pytest.mark.parametrize(
        "domain,expected_b0",
        [
            ("medical", 0.93),
            ("security", 0.95),
            ("environmental", 0.90),
            ("humanitarian", 0.92),
            ("infrastructure", 0.94),
        ],
    )
    def test_inflection_point_at_b0(self, domain: str, expected_b0: float) -> None:
        """At the inflection point b0, the gate value should be approx 0.5.

        This is the defining property of the logistic sigmoid:
            sigma(0) = 0.5, so sigma(k*(b0 - b0)) = sigma(0) = 0.5
        """
        result: float = sigmoid_benevolence_gate(expected_b0, domain=domain)
        assert result == pytest.approx(0.5, abs=1e-9)

    def test_symmetry_around_inflection(self) -> None:
        """gate(b0 + delta) + gate(b0 - delta) should equal 1.0 (sigmoid symmetry)."""
        b0: float = BENEVOLENCE_GATE.DEFAULT.b0
        delta: float = 0.02
        upper: float = sigmoid_benevolence_gate(b0 + delta)
        lower: float = sigmoid_benevolence_gate(b0 - delta)
        assert upper + lower == pytest.approx(1.0, abs=1e-9)


# ==========================================================================
# 2. Banach Recursion
# ==========================================================================


class TestBanachRecursion:
    """Tests for BanachRecursion from three_r/fusion.py."""

    def test_initialization_defaults(self) -> None:
        """BanachRecursion should initialise with defaults from RECURSION constants."""
        br = BanachRecursion()
        assert br.alpha_max == pytest.approx(RECURSION.ALPHA_MAX)
        assert br.max_depth == RECURSION.MAX_DEPTH
        assert br.convergence_tolerance == pytest.approx(RECURSION.CONVERGENCE_TOLERANCE)
        # alpha is sigmoid(0.0) * alpha_max = 0.5 * 0.95
        assert br.alpha == pytest.approx(0.5 * RECURSION.ALPHA_MAX, abs=1e-6)

    def test_alpha_always_at_most_alpha_max(self) -> None:
        """For any alpha_raw, the constrained alpha must be <= alpha_max.

        Mathematically, sigmoid(x) is strictly < 1 for all finite x, so
        alpha = sigmoid(alpha_raw) * alpha_max < alpha_max. However, for
        very large alpha_raw values (e.g., 1000), sigmoid(x) rounds to
        exactly 1.0 in float64 arithmetic, yielding alpha == alpha_max.
        The important guarantee is alpha <= alpha_max.
        """
        for raw in [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 1000.0]:
            br = BanachRecursion(alpha_raw=raw)
            assert (
                br.alpha <= br.alpha_max
            ), f"alpha_raw={raw}: alpha={br.alpha} > alpha_max={br.alpha_max}"
            assert br.alpha > 0.0

    def test_alpha_strictly_less_than_alpha_max_moderate_input(self) -> None:
        """For moderate alpha_raw values, alpha is strictly < alpha_max."""
        for raw in [-10.0, -1.0, 0.0, 1.0, 5.0]:
            br = BanachRecursion(alpha_raw=raw)
            assert (
                br.alpha < br.alpha_max
            ), f"alpha_raw={raw}: alpha={br.alpha} >= alpha_max={br.alpha_max}"

    def test_alpha_max_gte_one_raises_value_error(self) -> None:
        """alpha_max >= 1.0 must raise ValueError (convergence impossible)."""
        with pytest.raises(ValueError, match=r"alpha_max must be < 1\.0"):
            BanachRecursion(alpha_max=1.0)

        with pytest.raises(ValueError, match=r"alpha_max must be < 1\.0"):
            BanachRecursion(alpha_max=1.5)

    def test_set_alpha_via_sigmoid(self) -> None:
        """set_alpha() should update alpha via sigmoid constraint."""
        br = BanachRecursion()
        new_alpha: float = br.set_alpha(2.0)
        expected: float = (1.0 / (1.0 + np.exp(-2.0))) * br.alpha_max
        assert new_alpha == pytest.approx(expected, abs=1e-8)
        assert br.alpha == pytest.approx(expected, abs=1e-8)

    def test_error_bound_computation(self) -> None:
        """Error bound should equal alpha^d * x0_norm / (1 - alpha)."""
        br = BanachRecursion(alpha_raw=0.0)  # alpha = 0.5 * 0.95 = 0.475
        x0_norm: float = 1.0
        depth: int = 5
        expected: float = (br.alpha**depth) * x0_norm / (1.0 - br.alpha)
        actual: float = br.compute_error_bound(x0_norm, depth=depth)
        assert actual == pytest.approx(expected, rel=1e-8)

    def test_error_bound_decreases_with_depth(self) -> None:
        """Error bound should decrease as recursion depth increases."""
        br = BanachRecursion(alpha_raw=0.0)
        x0_norm: float = 1.0
        bounds: list[float] = [br.compute_error_bound(x0_norm, depth=d) for d in range(1, 20)]
        for i in range(len(bounds) - 1):
            assert bounds[i] > bounds[i + 1]

    def test_simple_recursion_converges(self) -> None:
        """A simple contractive recursion should converge and return finite results."""
        br = BanachRecursion(alpha_raw=-1.0, max_depth=20)

        def f(x: float) -> float:
            return 0.1 * x

        def g(x: float) -> float:
            return x * 0.5

        result, error_bound = br.recurse(1.0, f, g)
        assert np.isfinite(result)
        assert np.isfinite(error_bound)
        assert error_bound >= 0.0

    def test_contraction_violation_raises_runtime_error(self) -> None:
        """If the contraction ratio exceeds the threshold, a RuntimeError is raised."""
        br = BanachRecursion(alpha_raw=5.0, max_depth=10)

        call_count: int = 0

        def f_divergent(x: float) -> float:
            nonlocal call_count
            call_count += 1
            return x * (2.0**call_count)

        def g_identity(x: float) -> float:
            return x

        with pytest.raises(RuntimeError, match="Banach contraction violated"):
            br.recurse(10.0, f_divergent, g_identity)

    def test_convergence_detection(self) -> None:
        """When successive differences fall below tolerance, convergence_achieved is True.

        The convergence_achieved flag is set when |result - prev_result| < tolerance
        in the inner recursion. We use a large tolerance and a function pair that
        produces very similar values at successive levels to trigger detection.
        """
        br = BanachRecursion(
            alpha_raw=-5.0,  # Very small alpha ~= 0.006 * 0.95
            max_depth=30,
            convergence_tolerance=0.1,  # Generous tolerance
        )

        def f_const(x: float) -> float:
            return 1.0  # Constant base value

        def g_identity(x: float) -> float:
            return x

        result, _ = br.recurse(1.0, f_const, g_identity)
        assert np.isfinite(result)
        # With alpha ~0.006 and constant f, successive recursion levels differ by
        # only alpha * sub_result, which is tiny. So convergence should be achieved.
        assert br.convergence_achieved is True

    def test_actual_depth_tracking(self) -> None:
        """actual_depth should reflect the number of recursive steps taken."""
        br = BanachRecursion(alpha_raw=0.0, max_depth=10)

        def f(x: float) -> float:
            return x * 0.1

        def g(x: float) -> float:
            return x * 0.9

        br.recurse(1.0, f, g, depth=5)
        # actual_depth counts only recursive steps (not base case)
        assert br.actual_depth == 5  # depth 5 means 5 recursive steps

    def test_custom_alpha_max(self) -> None:
        """Custom alpha_max < 1.0 should be respected.

        With alpha_raw=100, sigmoid(100) rounds to 1.0 in float64,
        so alpha = 1.0 * 0.5 = 0.5 exactly. The key guarantee is alpha <= alpha_max.
        """
        br = BanachRecursion(alpha_raw=100.0, alpha_max=0.5)
        # sigmoid(100) ~ 1.0 in float64, so alpha is at or very near 0.5
        assert br.alpha <= 0.5
        assert br.alpha > 0.49

    def test_custom_alpha_max_moderate_raw(self) -> None:
        """With moderate alpha_raw, alpha should be strictly < alpha_max."""
        br = BanachRecursion(alpha_raw=2.0, alpha_max=0.5)
        expected: float = (1.0 / (1.0 + np.exp(-2.0))) * 0.5
        assert br.alpha == pytest.approx(expected, abs=1e-8)
        assert br.alpha < 0.5


# ==========================================================================
# 3. Domain-Adaptive Harmonics
# ==========================================================================


class TestDomainAdaptiveHarmonics:
    """Tests for get_domain_fundamentals() and DOMAIN_HARMONICS."""

    def test_environmental_returns_schumann_harmonics(self) -> None:
        """Environmental domain should return Schumann resonance frequencies."""
        freqs = get_domain_fundamentals("environmental")
        assert freqs is not None
        assert freqs == DOMAIN_HARMONICS.ENVIRONMENTAL
        assert freqs[0] == pytest.approx(7.83)
        assert freqs[1] == pytest.approx(14.3)
        assert freqs[2] == pytest.approx(20.8)
        assert freqs[3] == pytest.approx(27.3)
        assert freqs[4] == pytest.approx(33.8)

    def test_medical_returns_hrv_frequencies(self) -> None:
        """Medical domain should return HRV frequency bands."""
        freqs = get_domain_fundamentals("medical")
        assert freqs is not None
        assert freqs == DOMAIN_HARMONICS.MEDICAL
        assert freqs[0] == pytest.approx(0.04)
        assert freqs[1] == pytest.approx(0.15)

    def test_infrastructure_returns_power_grid_frequencies(self) -> None:
        """Infrastructure domain should return power grid and structural frequencies."""
        freqs = get_domain_fundamentals("infrastructure")
        assert freqs is not None
        assert freqs == DOMAIN_HARMONICS.INFRASTRUCTURE
        # Should include 50Hz and 60Hz (mains frequencies)
        assert 50.0 in freqs
        assert 60.0 in freqs

    def test_space_returns_orbital_frequencies(self) -> None:
        """Space domain should return solar cycle and orbital frequencies."""
        freqs = get_domain_fundamentals("space")
        assert freqs is not None
        assert freqs == DOMAIN_HARMONICS.SPACE

    def test_security_returns_none_for_auto_detect(self) -> None:
        """Security domain should return None (use adaptive detection)."""
        freqs = get_domain_fundamentals("security")
        assert freqs is None

    def test_financial_returns_none_for_auto_detect(self) -> None:
        """Financial domain should return None (use adaptive detection)."""
        freqs = get_domain_fundamentals("financial")
        assert freqs is None

    def test_unknown_domain_defaults_to_environmental(self) -> None:
        """An unknown domain should default to environmental (Schumann)."""
        freqs = get_domain_fundamentals("unknown_domain_xyz")
        assert freqs is not None
        assert freqs == DOMAIN_HARMONICS.ENVIRONMENTAL

    def test_case_insensitive_domain(self) -> None:
        """Domain lookup should be case-insensitive."""
        assert get_domain_fundamentals("ENVIRONMENTAL") == DOMAIN_HARMONICS.ENVIRONMENTAL
        assert get_domain_fundamentals("Medical") == DOMAIN_HARMONICS.MEDICAL
        assert get_domain_fundamentals("SECURITY") is None

    def test_all_frequency_tuples_are_nonempty(self) -> None:
        """Every predefined harmonic tuple should have at least one frequency."""
        assert len(DOMAIN_HARMONICS.ENVIRONMENTAL) > 0
        assert len(DOMAIN_HARMONICS.MEDICAL) > 0
        assert len(DOMAIN_HARMONICS.INFRASTRUCTURE) > 0
        assert len(DOMAIN_HARMONICS.SPACE) > 0

    def test_all_frequencies_are_positive(self) -> None:
        """All predefined fundamental frequencies should be positive."""
        for freqs in [
            DOMAIN_HARMONICS.ENVIRONMENTAL,
            DOMAIN_HARMONICS.MEDICAL,
            DOMAIN_HARMONICS.INFRASTRUCTURE,
            DOMAIN_HARMONICS.SPACE,
        ]:
            for freq in freqs:
                assert freq > 0.0, f"Non-positive frequency: {freq}"


# ==========================================================================
# 4. Hierarchical Omni-Scalar Aggregation
# ==========================================================================


class TestHierarchicalOmniScalarAggregation:
    """Tests for compute_hierarchical_score() on GlobalOmniScalarNetwork."""

    def _make_network(self) -> GlobalOmniScalarNetwork:
        """Create a fresh GOSNN instance for testing."""
        reset_global_network()
        return GlobalOmniScalarNetwork()

    def test_returns_valid_structure(self) -> None:
        """compute_hierarchical_score() should return a dict with required keys."""
        network: GlobalOmniScalarNetwork = self._make_network()
        result: dict[str, Any] = network.compute_hierarchical_score()

        assert "overall_score" in result
        assert "category_scores" in result
        assert "category_sizes" in result
        assert "method" in result

    def test_overall_score_in_unit_interval(self) -> None:
        """Overall score must be in [0, 1]."""
        network: GlobalOmniScalarNetwork = self._make_network()
        result: dict[str, Any] = network.compute_hierarchical_score()
        score: float = result["overall_score"]
        assert 0.0 <= score <= 1.0

    def test_category_scores_in_unit_interval(self) -> None:
        """Each category score must be in [0, 1]."""
        network: GlobalOmniScalarNetwork = self._make_network()
        result: dict[str, Any] = network.compute_hierarchical_score()
        for cat_name, cat_score in result["category_scores"].items():
            assert (
                0.0 <= cat_score <= 1.0
            ), f"Category '{cat_name}' score {cat_score} outside [0, 1]"

    def test_all_five_categories_present(self) -> None:
        """All five categories should be present in the result."""
        network: GlobalOmniScalarNetwork = self._make_network()
        result: dict[str, Any] = network.compute_hierarchical_score()
        expected_categories = {
            "safety",
            "fairness",
            "transparency",
            "accountability",
            "beneficence",
        }
        assert set(result["category_scores"].keys()) == expected_categories

    def test_geometric_mean_aggregation(self) -> None:
        """Geometric mean should be the default aggregation method."""
        network: GlobalOmniScalarNetwork = self._make_network()
        result: dict[str, Any] = network.compute_hierarchical_score(
            aggregation_method="geometric_mean"
        )
        assert result["method"] == "geometric_mean"
        assert 0.0 <= result["overall_score"] <= 1.0

    def test_arithmetic_mean_aggregation(self) -> None:
        """Arithmetic mean aggregation should work and be labelled correctly."""
        network: GlobalOmniScalarNetwork = self._make_network()
        result: dict[str, Any] = network.compute_hierarchical_score(
            aggregation_method="arithmetic_mean"
        )
        assert result["method"] == "arithmetic_mean"
        assert 0.0 <= result["overall_score"] <= 1.0

    def test_geometric_mean_penalises_low_category(self) -> None:
        """Geometric mean should yield a lower score than arithmetic when one category is low.

        This verifies the mathematical property that geometric mean <= arithmetic mean
        for non-negative values (AM-GM inequality).
        """
        network: GlobalOmniScalarNetwork = self._make_network()
        geo_result: dict[str, Any] = network.compute_hierarchical_score(
            aggregation_method="geometric_mean"
        )
        arith_result: dict[str, Any] = network.compute_hierarchical_score(
            aggregation_method="arithmetic_mean"
        )
        # AM-GM: geometric_mean <= arithmetic_mean (with equality iff all values equal)
        assert geo_result["overall_score"] <= arith_result["overall_score"] + 1e-9

    def test_custom_domain_weights(self) -> None:
        """Custom domain weights should affect the overall score."""
        network: GlobalOmniScalarNetwork = self._make_network()

        # Heavily weight safety
        safety_heavy: dict[str, Any] = network.compute_hierarchical_score(
            domain_weights={
                "safety": 10.0,
                "fairness": 0.1,
                "transparency": 0.1,
                "accountability": 0.1,
                "beneficence": 0.1,
            },
            aggregation_method="arithmetic_mean",
        )

        # Heavily weight accountability
        account_heavy: dict[str, Any] = network.compute_hierarchical_score(
            domain_weights={
                "safety": 0.1,
                "fairness": 0.1,
                "transparency": 0.1,
                "accountability": 10.0,
                "beneficence": 0.1,
            },
            aggregation_method="arithmetic_mean",
        )

        # The two results should differ unless the category scores happen to be identical
        safety_cat: float = safety_heavy["category_scores"]["safety"]
        account_cat: float = account_heavy["category_scores"]["accountability"]
        if safety_cat != pytest.approx(account_cat, abs=1e-6):
            assert safety_heavy["overall_score"] != pytest.approx(
                account_heavy["overall_score"], abs=1e-6
            )

    def test_category_sizes_are_nonnegative(self) -> None:
        """Each category size must be a non-negative integer."""
        network: GlobalOmniScalarNetwork = self._make_network()
        result: dict[str, Any] = network.compute_hierarchical_score()
        for cat_name, size in result["category_sizes"].items():
            assert isinstance(size, int)
            assert size >= 0, f"Category '{cat_name}' has negative size {size}"

    def test_equal_weights_default(self) -> None:
        """When no domain_weights are provided, all categories should be equally weighted."""
        network: GlobalOmniScalarNetwork = self._make_network()
        result_default: dict[str, Any] = network.compute_hierarchical_score()
        result_explicit: dict[str, Any] = network.compute_hierarchical_score(
            domain_weights={
                "safety": 1.0,
                "fairness": 1.0,
                "transparency": 1.0,
                "accountability": 1.0,
                "beneficence": 1.0,
            }
        )
        assert result_default["overall_score"] == pytest.approx(
            result_explicit["overall_score"], abs=1e-10
        )


# ==========================================================================
# 5. OAE Enhancements
# ==========================================================================


class TestOAEEnhancements:
    """Tests for OmniAvaEquation enhancements (Phase 3)."""

    # --- Configurable ethical_exponent ---

    def test_default_ethical_exponent_is_golden_ratio(self) -> None:
        """When ethical_exponent is not set, it should default to phi (1.618)."""
        aafe = OmniAvaEquation()
        assert aafe.ethical_exponent == pytest.approx(1.618033988749895, rel=1e-10)

    def test_custom_ethical_exponent(self) -> None:
        """A custom ethical_exponent should be respected in computation."""
        aafe_phi = OmniAvaEquation(ethical_exponent=None)  # Defaults to phi
        aafe_one = OmniAvaEquation(ethical_exponent=1.0)
        aafe_two = OmniAvaEquation(ethical_exponent=2.0)

        assert aafe_phi.ethical_exponent == pytest.approx(1.618033988749895, rel=1e-10)
        assert aafe_one.ethical_exponent == pytest.approx(1.0)
        assert aafe_two.ethical_exponent == pytest.approx(2.0)

    def test_ethical_exponent_affects_fusion_score(self) -> None:
        """Different ethical exponents should produce different fusion scores."""
        aafe_low = OmniAvaEquation(ethical_exponent=0.5)
        aafe_high = OmniAvaEquation(ethical_exponent=3.0)

        result_low = aafe_low.compute(0.8, 0.7, 0.6)
        result_high = aafe_high.compute(0.8, 0.7, 0.6)

        # Higher exponent with eta < 1 => lower scaling => lower score
        assert result_low.fusion_score > result_high.fusion_score

    # --- NaN guard ---

    def test_nan_recursion_score_replaced_with_zero(self) -> None:
        """NaN recursion_score should be replaced with 0.0."""
        aafe = OmniAvaEquation()
        result = aafe.compute(
            recursion_score=float("nan"),
            resonance_score=0.5,
            optimization_score=0.5,
        )
        assert np.isfinite(result.fusion_score)
        # With recursion_score=0.0, the weighted sum decreases
        assert result.recursion_score == pytest.approx(0.0)

    def test_nan_resonance_score_replaced_with_zero(self) -> None:
        """NaN resonance_score should be replaced with 0.0."""
        aafe = OmniAvaEquation()
        result = aafe.compute(
            recursion_score=0.5,
            resonance_score=float("nan"),
            optimization_score=0.5,
        )
        assert np.isfinite(result.fusion_score)
        assert result.resonance_score == pytest.approx(0.0)

    def test_nan_optimization_score_replaced_with_zero(self) -> None:
        """NaN optimization_score should be replaced with 0.0."""
        aafe = OmniAvaEquation()
        result = aafe.compute(
            recursion_score=0.5,
            resonance_score=0.5,
            optimization_score=float("nan"),
        )
        assert np.isfinite(result.fusion_score)
        assert result.optimization_score == pytest.approx(0.0)

    def test_all_nan_inputs_produce_zero_score(self) -> None:
        """If all three input scores are NaN, fusion_score should be 0.0."""
        aafe = OmniAvaEquation()
        result = aafe.compute(
            recursion_score=float("nan"),
            resonance_score=float("nan"),
            optimization_score=float("nan"),
        )
        assert result.fusion_score == pytest.approx(0.0)

    # --- benevolence_score integration ---

    def test_benevolence_score_uses_sigmoid_gate(self) -> None:
        """When benevolence_score is provided, the sigmoid gate should be used."""
        aafe = OmniAvaEquation(domain="medical")
        result = aafe.compute(
            recursion_score=0.8,
            resonance_score=0.7,
            optimization_score=0.6,
            benevolence_score=0.99,
        )
        # With b=0.99 and medical b0=0.93, the gate should be very high
        expected_eta: float = sigmoid_benevolence_gate(0.99, domain="medical")
        assert result.ethical_compliance_threshold == pytest.approx(expected_eta, abs=1e-8)

    def test_benevolence_score_none_uses_threshold(self) -> None:
        """Without benevolence_score, the raw ethical_compliance_threshold is used."""
        aafe = OmniAvaEquation(ethical_compliance_threshold=0.96)
        result = aafe.compute(
            recursion_score=0.8,
            resonance_score=0.7,
            optimization_score=0.6,
        )
        assert result.ethical_compliance_threshold == pytest.approx(0.96)

    def test_high_benevolence_boosts_score(self) -> None:
        """A high benevolence_score (near 1.0) should produce a higher score than a low one."""
        aafe = OmniAvaEquation(domain="security")
        result_high = aafe.compute(0.8, 0.7, 0.6, benevolence_score=0.99)

        aafe2 = OmniAvaEquation(domain="security")
        result_low = aafe2.compute(0.8, 0.7, 0.6, benevolence_score=0.5)

        assert result_high.fusion_score > result_low.fusion_score

    # --- Backward compatibility ---

    def test_backward_compat_sigma_immutable_alias(self) -> None:
        """sigma_immutable parameter should work as an alias for ethical_compliance_threshold."""
        aafe = OmniAvaEquation(sigma_immutable=0.94)
        assert aafe.ethical_compliance_threshold == pytest.approx(0.94)
        assert aafe.sigma_immutable == pytest.approx(0.94)

    def test_backward_compat_lambda_lyapunov_alias(self) -> None:
        """lambda_lyapunov parameter should work as an alias for convergence_rate."""
        aafe = OmniAvaEquation(lambda_lyapunov=0.3)
        assert aafe.convergence_rate_param == pytest.approx(0.3)
        assert aafe.lambda_lyapunov == pytest.approx(0.3)

    def test_backward_compat_sigma_immutable_override(self) -> None:
        """sigma_immutable_override should work as an alias in compute()."""
        aafe = OmniAvaEquation()
        result = aafe.compute(
            recursion_score=0.8,
            resonance_score=0.7,
            optimization_score=0.6,
            sigma_immutable_override=0.97,
        )
        assert result.ethical_compliance_threshold == pytest.approx(0.97)

    def test_default_weights_sum_to_one(self) -> None:
        """Default golden-ratio-based weights should sum to 1.0."""
        aafe = OmniAvaEquation()
        total: float = sum(aafe.weights.values())
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_compute_returns_anomaly_fusion_result(self) -> None:
        """compute() should return an AnomalyFusionResult with all expected fields."""
        aafe = OmniAvaEquation()
        result = aafe.compute(0.8, 0.7, 0.6)

        assert hasattr(result, "fusion_score")
        assert hasattr(result, "recursion_score")
        assert hasattr(result, "resonance_score")
        assert hasattr(result, "optimization_score")
        assert hasattr(result, "ethical_compliance_threshold")
        assert hasattr(result, "fusion_weights")
        assert hasattr(result, "lyapunov_bound")
        assert hasattr(result, "convergence_rate")

    def test_lyapunov_bound_decreases_over_time(self) -> None:
        """Lyapunov bound should decrease with each call (exponential decay)."""
        aafe = OmniAvaEquation()
        bounds: list[float] = []
        for _ in range(10):
            result = aafe.compute(0.8, 0.7, 0.6)
            bounds.append(result.lyapunov_bound)

        for i in range(len(bounds) - 1):
            assert bounds[i] > bounds[i + 1], (
                f"Lyapunov bound not decreasing: step {i} ({bounds[i]}) >= "
                f"step {i+1} ({bounds[i+1]})"
            )

    def test_ethical_compliance_threshold_clamped(self) -> None:
        """ethical_compliance_threshold should be clamped to [0.90, 0.99]."""
        aafe_low = OmniAvaEquation(ethical_compliance_threshold=0.5)
        assert aafe_low.ethical_compliance_threshold >= 0.90

        aafe_high = OmniAvaEquation(ethical_compliance_threshold=1.0)
        assert aafe_high.ethical_compliance_threshold <= 0.99

    def test_domain_parameter_persists(self) -> None:
        """The domain parameter should be stored and accessible."""
        aafe = OmniAvaEquation(domain="humanitarian")
        assert aafe.domain == "humanitarian"


# ==========================================================================
# 6. Cross-cutting integration checks
# ==========================================================================


class TestCrossCuttingIntegration:
    """Integration tests that verify Phase 3 components work together."""

    def test_aafe_with_banach_recursion_result(self) -> None:
        """OAE should accept the output of BanachRecursion as a component score."""
        br = BanachRecursion(alpha_raw=0.0, max_depth=10)

        def f(x: float) -> float:
            return 0.1 * x

        def g(x: float) -> float:
            return x * 0.5

        recursion_result, _ = br.recurse(1.0, f, g)

        aafe = OmniAvaEquation(domain="environmental")
        result = aafe.compute(
            recursion_score=min(max(recursion_result, 0.0), 1.0),
            resonance_score=0.7,
            optimization_score=0.6,
            benevolence_score=0.95,
        )
        assert np.isfinite(result.fusion_score)

    def test_sigmoid_gate_matches_manual_computation(self) -> None:
        """Verify sigmoid_benevolence_gate matches the manual formula."""
        b: float = 0.94
        domain: str = "medical"
        profile = BENEVOLENCE_GATE.MEDICAL
        k: float = profile.k
        b0: float = profile.b0

        expected: float = 1.0 / (1.0 + math.exp(-k * (b - b0)))
        actual: float = sigmoid_benevolence_gate(b, domain=domain)
        assert actual == pytest.approx(expected, abs=1e-12)

    def test_recursion_constants_match_banach_defaults(self) -> None:
        """BanachRecursion defaults should align with RECURSION constants."""
        assert pytest.approx(0.95) == RECURSION.ALPHA_MAX
        assert RECURSION.MAX_DEPTH == 50
        assert pytest.approx(1e-6) == RECURSION.CONVERGENCE_TOLERANCE
        assert pytest.approx(1.0) == RECURSION.CONTRACTION_VIOLATION_THRESHOLD
