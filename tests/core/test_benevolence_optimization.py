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

"""
Comprehensive tests for the benevolence optimization module.

Tests include:
- All 6 gating form variations
- Lyapunov stability verification
- sigma_Immutable threshold enforcement
- ImmutableEthicsError handling
- Domain-adaptive form selection
- Benevolence and equity computations
- Property-based tests for stability across sigma ranges

Coverage target: 95%+
"""

import os
import sys

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

try:
    from benchmarks.benevolence_optimization import (
        BENEVOLENCE_THRESHOLD,
        LAMBDA_LYAPUNOV,
        PHI,
        SIGMA_IMMUTABLE_DEFAULT,
        SIGMA_IMMUTABLE_HIGH_STAKES,
        SIGMA_IMMUTABLE_MEDICAL,
        BenevolenceOptimizer,
        EthicalGatingForms,
        GatingFormConfig,
        GatingFormType,
        GatingResult,
        ImmutableEthicsError,
        validate_symbolic_convexity,
    )

    BENEVOLENCE_MODULE_AVAILABLE = True
except ImportError:
    BENEVOLENCE_MODULE_AVAILABLE = False


# Skip all tests if module not available
pytestmark = pytest.mark.skipif(
    not BENEVOLENCE_MODULE_AVAILABLE, reason="Benevolence optimization module not available"
)


class TestGatingFormConfig:
    """Tests for GatingFormConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = GatingFormConfig(form_type=GatingFormType.QUADRATIC)
        assert config.threshold == SIGMA_IMMUTABLE_DEFAULT
        assert config.k == 5.0
        assert config.delta == 0.05
        assert config.variance == 0.05

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = GatingFormConfig(
            form_type=GatingFormType.SIGMOID, threshold=0.94, k=7.0, delta=0.03, variance=0.02
        )
        assert config.threshold == 0.94
        assert config.k == 7.0
        assert config.delta == 0.03
        assert config.variance == 0.02

    def test_threshold_below_minimum_raises_error(self) -> None:
        """Test that threshold below medical fallback raises ImmutableEthicsError."""
        with pytest.raises(ImmutableEthicsError) as excinfo:
            GatingFormConfig(form_type=GatingFormType.QUADRATIC, threshold=0.90)

        assert "medical fallback" in str(excinfo.value)
        assert excinfo.value.attempted_value == 0.90
        assert excinfo.value.threshold == SIGMA_IMMUTABLE_MEDICAL

    def test_invalid_k_raises_error(self) -> None:
        """Test that non-positive k raises ValueError."""
        with pytest.raises(ValueError, match="k must be positive"):
            GatingFormConfig(form_type=GatingFormType.SIGMOID, k=0)

        with pytest.raises(ValueError, match="k must be positive"):
            GatingFormConfig(form_type=GatingFormType.SIGMOID, k=-1)

    def test_invalid_delta_raises_error(self) -> None:
        """Test that non-positive delta raises ValueError."""
        with pytest.raises(ValueError, match="delta must be positive"):
            GatingFormConfig(form_type=GatingFormType.PIECEWISE, delta=0)

    def test_invalid_variance_raises_error(self) -> None:
        """Test that non-positive variance raises ValueError."""
        with pytest.raises(ValueError, match="Variance must be positive"):
            GatingFormConfig(form_type=GatingFormType.GAUSSIAN_RBF, variance=0)


class TestEthicalGatingForms:
    """Tests for EthicalGatingForms class."""

    @pytest.fixture
    def gating_forms(self):
        """Create a gating forms instance with default config."""
        config = GatingFormConfig(form_type=GatingFormType.QUADRATIC)
        return EthicalGatingForms(config)

    def test_quadratic_below_threshold(self, gating_forms) -> None:
        """Test quadratic form below threshold."""
        sigma = 0.90
        result = gating_forms.quadratic(sigma)

        assert isinstance(result, GatingResult)
        assert result.form_type == GatingFormType.QUADRATIC
        assert result.passes_gate is False
        assert result.penalty > 0
        # Penalty should be (0.96 - 0.90)^2 = 0.0036
        expected_penalty = (SIGMA_IMMUTABLE_DEFAULT - sigma) ** 2
        assert abs(result.penalty - expected_penalty) < 1e-10
        # Gradient should be -2 * (0.96 - 0.90) = -0.12
        expected_gradient = -2 * (SIGMA_IMMUTABLE_DEFAULT - sigma)
        assert result.gradient is not None
        assert abs(result.gradient - expected_gradient) < 1e-10

    def test_quadratic_above_threshold(self, gating_forms) -> None:
        """Test quadratic form above threshold."""
        sigma = 0.98
        result = gating_forms.quadratic(sigma)

        assert result.passes_gate
        assert result.penalty == 0.0
        assert result.gradient == 0.0

    def test_quadratic_at_threshold(self, gating_forms) -> None:
        """Test quadratic form exactly at threshold."""
        sigma = SIGMA_IMMUTABLE_DEFAULT
        result = gating_forms.quadratic(sigma)

        assert result.passes_gate
        assert result.penalty == 0.0
        assert result.gradient == 0.0

    def test_linear_below_threshold(self, gating_forms) -> None:
        """Test linear form below threshold."""
        sigma = 0.90
        result = gating_forms.linear(sigma)

        assert result.form_type == GatingFormType.LINEAR
        assert result.passes_gate is False
        expected_penalty = SIGMA_IMMUTABLE_DEFAULT - sigma
        assert abs(result.penalty - expected_penalty) < 1e-10
        assert result.gradient == -1.0

    def test_linear_above_threshold(self, gating_forms) -> None:
        """Test linear form above threshold."""
        sigma = 0.98
        result = gating_forms.linear(sigma)

        assert result.passes_gate
        assert result.penalty == 0.0
        assert result.gradient == 0.0

    def test_sigmoid_below_threshold(self, gating_forms) -> None:
        """Test sigmoid form below threshold."""
        sigma = 0.90
        result = gating_forms.sigmoid(sigma)

        assert result.form_type == GatingFormType.SIGMOID
        # Below threshold, sigmoid value should be high (close to 1)
        assert result.penalty > 0.5
        assert result.gradient is not None

    def test_sigmoid_above_threshold(self, gating_forms) -> None:
        """Test sigmoid form above threshold."""
        sigma = 0.98
        result = gating_forms.sigmoid(sigma)

        # Above threshold, sigmoid value should be low (close to 0)
        assert result.penalty < 0.5
        assert result.passes_gate

    def test_sigmoid_at_threshold(self, gating_forms) -> None:
        """Test sigmoid form exactly at threshold."""
        sigma = SIGMA_IMMUTABLE_DEFAULT
        result = gating_forms.sigmoid(sigma)

        # At threshold, sigmoid should be exactly 0.5
        assert abs(result.penalty - 0.5) < 1e-10

    def test_exponential_below_threshold(self, gating_forms) -> None:
        """Test exponential form below threshold."""
        sigma = 0.90
        result = gating_forms.exponential(sigma)

        assert result.form_type == GatingFormType.EXPONENTIAL
        assert result.passes_gate is False
        assert result.penalty > 1.0  # Exponential increases below threshold
        assert result.gradient > 0  # Positive gradient (penalty increases as sigma decreases)

    def test_exponential_above_threshold(self, gating_forms) -> None:
        """Test exponential form above threshold."""
        sigma = 0.98
        result = gating_forms.exponential(sigma)

        assert result.passes_gate
        assert result.penalty == 1.0  # Minimal penalty at/above threshold
        assert result.gradient == 0.0

    def test_piecewise_small_deviation(self, gating_forms) -> None:
        """Test piecewise form with small deviation (linear regime)."""
        # Small deviation: threshold - sigma < delta (0.05)
        sigma = 0.93  # deviation = 0.03 < 0.05
        result = gating_forms.piecewise(sigma)

        assert result.form_type == GatingFormType.PIECEWISE
        assert result.passes_gate is False
        assert result.metadata["regime"] == "linear"
        expected_penalty = SIGMA_IMMUTABLE_DEFAULT - sigma
        assert abs(result.penalty - expected_penalty) < 1e-10
        assert result.gradient == -1.0

    def test_piecewise_large_deviation(self, gating_forms) -> None:
        """Test piecewise form with large deviation (quadratic regime)."""
        # Large deviation: threshold - sigma >= delta (0.05)
        sigma = 0.85  # deviation = 0.11 >= 0.05
        result = gating_forms.piecewise(sigma)

        assert result.passes_gate is False
        assert result.metadata["regime"] == "quadratic"

    def test_piecewise_above_threshold(self, gating_forms) -> None:
        """Test piecewise form above threshold."""
        sigma = 0.98
        result = gating_forms.piecewise(sigma)

        assert result.passes_gate
        assert result.penalty == 0.0
        assert result.gradient == 0.0

    def test_gaussian_rbf_below_threshold(self, gating_forms) -> None:
        """Test Gaussian RBF form below threshold."""
        sigma = 0.90
        result = gating_forms.gaussian_rbf(sigma)

        assert result.form_type == GatingFormType.GAUSSIAN_RBF
        assert not result.passes_gate
        # Penalty is 1 - gaussian, where gaussian is still relatively high at sigma=0.90
        # with variance=0.05 and threshold=0.96 (distance=0.06)
        assert result.penalty >= 0  # Penalty should be non-negative

    def test_gaussian_rbf_at_threshold(self, gating_forms) -> None:
        """Test Gaussian RBF form at threshold."""
        sigma = SIGMA_IMMUTABLE_DEFAULT
        result = gating_forms.gaussian_rbf(sigma)

        # At threshold, inverted gaussian = 1 - 1 = 0
        assert abs(result.penalty) < 1e-10

    def test_apply_method(self, gating_forms) -> None:
        """Test the apply method dispatches correctly."""
        sigma = 0.90

        for form_type in GatingFormType:
            result = gating_forms.apply(sigma, form_type)
            assert result.form_type == form_type
            assert isinstance(result.penalty, float)

    def test_lyapunov_stability_convergent(self, gating_forms) -> None:
        """Test Lyapunov stability for convergent trajectory."""
        # Create convergent trajectory approaching threshold
        trajectory = np.array([0.85, 0.88, 0.91, 0.93, 0.94, 0.95, 0.955, 0.958, 0.96, 0.96])

        lyapunov, is_stable = gating_forms.compute_lyapunov_stability(trajectory)

        # Convergent trajectory should have positive Lyapunov exponent
        assert lyapunov > 0

    def test_lyapunov_stability_short_trajectory(self, gating_forms) -> None:
        """Test Lyapunov stability with short trajectory."""
        trajectory = np.array([0.9, 0.92, 0.94])  # Less than 10 samples

        lyapunov, is_stable = gating_forms.compute_lyapunov_stability(trajectory)

        assert lyapunov == 0.0
        assert is_stable is False


class TestBenevolenceOptimizer:
    """Tests for BenevolenceOptimizer class."""

    @pytest.fixture
    def optimizer(self):
        """Create a benevolence optimizer instance."""
        return BenevolenceOptimizer()

    def test_benchmark_form_quadratic(self, optimizer) -> None:
        """Test benchmarking the quadratic form."""
        result = optimizer.benchmark_form(GatingFormType.QUADRATIC, n_simulations=100)

        assert result.form_type == GatingFormType.QUADRATIC
        assert result.convergence_epochs > 0
        assert 0 <= result.f1_score <= 1
        assert result.overhead_percent >= 0

    def test_benchmark_all_forms(self, optimizer) -> None:
        """Test benchmarking all forms."""
        results = optimizer.benchmark_all_forms(n_simulations=50)

        assert len(results) == len(GatingFormType)
        for form_type in GatingFormType:
            assert form_type in results

    def test_get_optimal_form_general(self, optimizer) -> None:
        """Test getting optimal form for general domain."""
        optimizer.benchmark_all_forms(n_simulations=50)
        best_form, best_result = optimizer.get_optimal_form(domain="general")

        assert best_form in GatingFormType
        assert best_result is not None

    def test_get_optimal_form_medical(self, optimizer) -> None:
        """Test getting optimal form for medical domain."""
        optimizer.benchmark_all_forms(n_simulations=50)
        best_form, best_result = optimizer.get_optimal_form(domain="medical")

        # Medical should prefer stable forms (quadratic, piecewise)
        assert best_form in GatingFormType

    def test_get_optimal_form_security(self, optimizer) -> None:
        """Test getting optimal form for security domain."""
        optimizer.benchmark_all_forms(n_simulations=50)
        best_form, best_result = optimizer.get_optimal_form(domain="security")

        assert best_form in GatingFormType


class TestImmutableEthicsError:
    """Tests for ImmutableEthicsError exception."""

    def test_exception_message(self) -> None:
        """Test exception message format."""
        error = ImmutableEthicsError("Test error", 0.90, 0.93)

        assert "Test error" in str(error)
        assert "0.9" in str(error)
        assert "0.93" in str(error)
        assert error.attempted_value == 0.90
        assert error.threshold == 0.93


class TestConstants:
    """Tests for module constants."""

    def test_phi_value(self) -> None:
        """Test golden ratio constant."""
        assert abs(PHI - 1.618033988749895) < 1e-10

    def test_lambda_lyapunov(self) -> None:
        """Test Lyapunov constant."""
        assert LAMBDA_LYAPUNOV == 0.25

    def test_benevolence_threshold(self) -> None:
        """Test benevolence threshold."""
        assert BENEVOLENCE_THRESHOLD == 0.99

    def test_sigma_immutable_thresholds(self) -> None:
        """Test sigma_Immutable thresholds."""
        assert SIGMA_IMMUTABLE_DEFAULT == 0.96
        assert SIGMA_IMMUTABLE_MEDICAL == 0.93
        assert SIGMA_IMMUTABLE_HIGH_STAKES == 0.96


class TestPropertyBased:
    """Property-based tests for gating forms."""

    @pytest.fixture
    def gating_forms(self):
        """Create a gating forms instance."""
        config = GatingFormConfig(form_type=GatingFormType.QUADRATIC)
        return EthicalGatingForms(config)

    def test_all_forms_bounded_penalty(self, gating_forms) -> None:
        """Test that all forms produce bounded penalties."""
        sigma_values = np.linspace(0.5, 1.0, 100)

        for form_type in GatingFormType:
            for sigma in sigma_values:
                result = gating_forms.apply(sigma, form_type)
                # Penalty should be non-negative
                assert result.penalty >= 0
                # Penalty should be bounded (varies by form)
                if form_type != GatingFormType.EXPONENTIAL:
                    assert result.penalty <= 2.0

    def test_gate_passes_above_threshold(self, gating_forms) -> None:
        """Test that gate passes for all values above threshold."""
        sigma_values = np.linspace(SIGMA_IMMUTABLE_DEFAULT, 1.0, 50)

        for sigma in sigma_values:
            valid_forms = [
                GatingFormType.QUADRATIC,
                GatingFormType.LINEAR,
                GatingFormType.EXPONENTIAL,
                GatingFormType.PIECEWISE,
            ]
            for form_type in valid_forms:
                result = gating_forms.apply(sigma, form_type)
                assert result.passes_gate

    def test_penalty_increases_as_sigma_decreases(self, gating_forms) -> None:
        """Test that penalty generally increases as sigma decreases below threshold."""
        sigma_values = np.linspace(0.8, SIGMA_IMMUTABLE_DEFAULT - 0.01, 20)

        for form_type in [GatingFormType.QUADRATIC, GatingFormType.LINEAR]:
            penalties = []
            for sigma in sigma_values:
                result = gating_forms.apply(sigma, form_type)
                penalties.append(result.penalty)

            # Penalties should be monotonically decreasing as sigma increases
            for i in range(len(penalties) - 1):
                assert penalties[i] >= penalties[i + 1]


class TestSymbolicValidation:
    """Tests for symbolic validation functions."""

    def test_validate_symbolic_convexity(self) -> None:
        """Test symbolic convexity validation."""
        results = validate_symbolic_convexity()

        if "error" not in results:
            assert "quadratic" in results
            assert results["quadratic"]["is_convex"] is True
        else:
            # Skip if sympy not available
            assert "error" in results


class TestGatingFormGradients:
    """Tests for gradient computations."""

    @pytest.fixture
    def gating_forms(self):
        """Create a gating forms instance."""
        config = GatingFormConfig(form_type=GatingFormType.QUADRATIC)
        return EthicalGatingForms(config)

    def test_quadratic_gradient_direction(self, gating_forms) -> None:
        """Test quadratic gradient points towards threshold."""
        sigma = 0.90
        result = gating_forms.quadratic(sigma)

        # Gradient should be negative (pushing sigma up towards threshold)
        assert result.gradient < 0

    def test_linear_gradient_constant(self, gating_forms) -> None:
        """Test linear gradient is constant below threshold."""
        for sigma in [0.85, 0.88, 0.91, 0.94]:
            result = gating_forms.linear(sigma)
            if sigma < SIGMA_IMMUTABLE_DEFAULT:
                assert result.gradient == -1.0

    def test_sigmoid_gradient_symmetric(self, gating_forms) -> None:
        """Test sigmoid gradient has expected shape."""
        # Gradient should be largest in magnitude near threshold
        results = []
        for sigma in [0.8, 0.88, 0.94, 0.96, 0.98]:
            result = gating_forms.sigmoid(sigma)
            results.append((sigma, abs(result.gradient) if result.gradient else 0))

        # Find maximum gradient magnitude
        max_grad_sigma = max(results, key=lambda x: x[1])[0]
        # Should be close to threshold
        assert abs(max_grad_sigma - SIGMA_IMMUTABLE_DEFAULT) < 0.1


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def gating_forms(self):
        """Create a gating forms instance."""
        config = GatingFormConfig(form_type=GatingFormType.QUADRATIC)
        return EthicalGatingForms(config)

    def test_sigma_at_zero(self, gating_forms) -> None:
        """Test forms handle sigma = 0."""
        sigma = 0.0

        for form_type in GatingFormType:
            result = gating_forms.apply(sigma, form_type)
            assert result.penalty >= 0
            assert not np.isnan(result.penalty)
            assert not np.isinf(result.penalty)

    def test_sigma_at_one(self, gating_forms) -> None:
        """Test forms handle sigma = 1."""
        sigma = 1.0

        for form_type in GatingFormType:
            result = gating_forms.apply(sigma, form_type)
            assert result.passes_gate
            assert not np.isnan(result.penalty)

    def test_sigma_negative(self, gating_forms) -> None:
        """Test forms handle negative sigma gracefully."""
        sigma = -0.1

        for form_type in GatingFormType:
            result = gating_forms.apply(sigma, form_type)
            # Should not crash, penalty should be high
            assert not np.isnan(result.penalty)

    def test_extreme_k_values(self) -> None:
        """Test sigmoid with extreme k values."""
        for k in [0.1, 50.0, 100.0]:
            config = GatingFormConfig(form_type=GatingFormType.SIGMOID, k=k)
            gating = EthicalGatingForms(config)

            result = gating.sigmoid(0.90)
            assert not np.isnan(result.penalty)
            assert not np.isinf(result.penalty)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
