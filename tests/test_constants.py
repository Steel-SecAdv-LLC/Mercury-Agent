# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for omni_mercury_engine.utils.constants module.

Tests mathematical constants, precision handling, and validation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from omni_mercury_engine.utils.constants import (
    MathematicalConstants,
    Precision,
    get_constant,
    validate_constant,
)

# Create aliases for easier testing
MC = MathematicalConstants
PI = MC.PI
E = MC.E
GOLDEN_RATIO = MC.GOLDEN_RATIO
EULER_MASCHERONI = MC.EULER_MASCHERONI
CATALAN = MC.CATALAN
SQRT2 = MC.SQRT2
SQRT3 = MC.SQRT3
SQRT5 = MC.SQRT5
LN2 = MC.LN2
LN10 = MC.LN10


class TestMathConstant:
    """Tests for MathConstant dataclass."""

    def test_basic_properties(self) -> None:
        """Test basic constant properties."""
        assert PI.value == pytest.approx(math.pi, rel=1e-15)
        assert PI.name == "Pi"
        assert PI.symbol == "π"
        assert PI.precision_digits >= 15

    def test_to_precision_float32(self) -> None:
        """Test conversion to float32."""
        value = PI.to_precision(Precision.FLOAT32)
        assert isinstance(value, np.float32)
        assert abs(value - math.pi) < 1e-6

    def test_to_precision_float64(self) -> None:
        """Test conversion to float64."""
        value = PI.to_precision(Precision.FLOAT64)
        assert isinstance(value, np.float64)
        assert abs(value - math.pi) < 1e-15

    def test_validation(self) -> None:
        """Test constant validation."""
        assert PI.validate(tolerance=1e-10)
        assert E.validate(tolerance=1e-10)
        assert GOLDEN_RATIO.validate(tolerance=1e-10)

    def test_validation_without_sympy(self) -> None:
        """Test validation without sympy."""
        assert PI.validate(tolerance=1e-10, use_sympy=False)

    def test_immutability(self) -> None:
        """Test that constants are immutable."""
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            PI.value = 3.0  # type: ignore[misc]


class TestFundamentalConstants:
    """Tests for fundamental mathematical constants."""

    def test_pi(self) -> None:
        """Test PI constant."""
        assert PI.value == pytest.approx(3.141592653589793, rel=1e-15)
        assert PI.symbol == "π"
        assert PI.oeis_id is not None

    def test_e(self) -> None:
        """Test E (Euler's number) constant."""
        assert E.value == pytest.approx(2.718281828459045, rel=1e-15)
        assert E.symbol == "e"

    def test_golden_ratio(self) -> None:
        """Test golden ratio constant."""
        assert GOLDEN_RATIO.value == pytest.approx(1.618033988749895, rel=1e-15)
        assert GOLDEN_RATIO.symbol == "φ"
        # Verify golden ratio property: φ² = φ + 1
        phi = GOLDEN_RATIO.value
        assert phi**2 == pytest.approx(phi + 1, rel=1e-10)

    def test_euler_mascheroni(self) -> None:
        """Test Euler-Mascheroni constant."""
        assert EULER_MASCHERONI.value == pytest.approx(0.5772156649015329, rel=1e-15)
        assert EULER_MASCHERONI.symbol == "γ"

    def test_catalan(self) -> None:
        """Test Catalan's constant."""
        assert CATALAN.value == pytest.approx(0.9159655941772190, rel=1e-15)
        assert CATALAN.symbol == "G"


class TestAlgebraicConstants:
    """Tests for algebraic constants (square roots)."""

    def test_sqrt2(self) -> None:
        """Test square root of 2."""
        assert SQRT2.value == pytest.approx(math.sqrt(2), rel=1e-15)
        assert SQRT2.value**2 == pytest.approx(2.0, rel=1e-10)

    def test_sqrt3(self) -> None:
        """Test square root of 3."""
        assert SQRT3.value == pytest.approx(math.sqrt(3), rel=1e-15)
        assert SQRT3.value**2 == pytest.approx(3.0, rel=1e-10)

    def test_sqrt5(self) -> None:
        """Test square root of 5."""
        assert SQRT5.value == pytest.approx(math.sqrt(5), rel=1e-15)
        assert SQRT5.value**2 == pytest.approx(5.0, rel=1e-10)


class TestLogarithmicConstants:
    """Tests for logarithmic constants."""

    def test_ln2(self) -> None:
        """Test natural log of 2."""
        assert LN2.value == pytest.approx(math.log(2), rel=1e-15)
        assert math.exp(LN2.value) == pytest.approx(2.0, rel=1e-10)

    def test_ln10(self) -> None:
        """Test natural log of 10."""
        assert LN10.value == pytest.approx(math.log(10), rel=1e-15)
        assert math.exp(LN10.value) == pytest.approx(10.0, rel=1e-10)


class TestConstantsRegistry:
    """Tests for the constants registry."""

    def test_get_constant(self) -> None:
        """Test getting constant by name."""
        pi = get_constant("PI")
        assert pi is not None
        assert pi == pytest.approx(math.pi, rel=1e-15)

    def test_get_constant_golden_ratio(self) -> None:
        """Test getting golden ratio constant."""
        phi = get_constant("GOLDEN_RATIO")
        assert phi is not None
        assert phi == pytest.approx(1.618033988749895, rel=1e-10)

    def test_get_constant_not_found(self) -> None:
        """Test getting nonexistent constant raises ValueError."""
        with pytest.raises(ValueError):
            get_constant("NONEXISTENT")

    def test_mathematical_constants_class(self) -> None:
        """Test MathematicalConstants class has expected constants."""
        assert hasattr(MC, "PI")
        assert hasattr(MC, "E")
        assert hasattr(MC, "GOLDEN_RATIO")
        assert hasattr(MC, "EULER_MASCHERONI")


class TestValidation:
    """Tests for constant validation functions."""

    def test_validate_pi(self) -> None:
        """Test validating PI constant."""
        result = validate_constant(PI.value, "PI")
        assert result  # numpy bool comparison

    def test_validate_e(self) -> None:
        """Test validating E constant."""
        result = validate_constant(E.value, "E")
        assert result  # numpy bool comparison

    def test_validate_golden_ratio(self) -> None:
        """Test validating golden ratio."""
        result = validate_constant(GOLDEN_RATIO.value, "GOLDEN_RATIO")
        assert result  # numpy bool comparison

    def test_constants_are_finite(self) -> None:
        """Test that key constants are finite."""
        constants_to_check = [PI, E, GOLDEN_RATIO, EULER_MASCHERONI, CATALAN]
        for const in constants_to_check:
            assert math.isfinite(const.value), f"{const.name} is not finite"


class TestMathematicalIdentities:
    """Tests for mathematical identities using constants."""

    def test_golden_ratio_identity(self) -> None:
        """Test φ² = φ + 1."""
        phi = GOLDEN_RATIO.value
        assert phi**2 == pytest.approx(phi + 1, rel=1e-10)

    def test_golden_ratio_reciprocal(self) -> None:
        """Test 1/φ = φ - 1."""
        phi = GOLDEN_RATIO.value
        assert 1 / phi == pytest.approx(phi - 1, rel=1e-10)

    def test_eulers_identity_components(self) -> None:
        """Test components of Euler's identity e^(iπ) + 1 = 0."""
        # Using numpy for complex exponential
        result = np.exp(1j * PI.value) + 1
        assert abs(result) < 1e-10

    def test_sqrt_relationships(self) -> None:
        """Test sqrt relationships with golden ratio."""
        phi = GOLDEN_RATIO.value
        sqrt5 = SQRT5.value
        # φ = (1 + √5) / 2
        assert phi == pytest.approx((1 + sqrt5) / 2, rel=1e-10)

    def test_natural_log_identity(self) -> None:
        """Test ln(2) * ln(10) relationship."""
        # log₁₀(2) = ln(2) / ln(10)
        log10_2 = LN2.value / LN10.value
        assert log10_2 == pytest.approx(math.log10(2), rel=1e-10)


class TestConstantPrecision:
    """Tests for precision handling."""

    def test_precision_levels(self) -> None:
        """Test different precision levels give correct types."""
        for prec in [Precision.FLOAT32, Precision.FLOAT64]:
            value = PI.to_precision(prec)
            assert value is not None

    def test_float32_precision_loss(self) -> None:
        """Test that float32 has less precision than float64."""
        f32 = PI.to_precision(Precision.FLOAT32)
        f64 = PI.to_precision(Precision.FLOAT64)
        # float32 should differ from float64 at some decimal place
        assert abs(float(f32) - float(f64)) > 0

    def test_precision_digits_attribute(self) -> None:
        """Test precision_digits attribute is reasonable."""
        constants_to_check = [PI, E, GOLDEN_RATIO, CATALAN]
        for const in constants_to_check:
            assert const.precision_digits >= 10, f"{const.name} has low precision"
            assert const.precision_digits <= 50, f"{const.name} has unreasonable precision"


class TestConstantMetadata:
    """Tests for constant metadata."""

    def test_key_constants_have_names(self) -> None:
        """Test key constants have human-readable names."""
        constants_to_check = [PI, E, GOLDEN_RATIO, EULER_MASCHERONI, CATALAN]
        for const in constants_to_check:
            assert const.name is not None
            assert len(const.name) > 0

    def test_key_constants_have_symbols(self) -> None:
        """Test key constants have symbols."""
        constants_to_check = [PI, E, GOLDEN_RATIO, EULER_MASCHERONI]
        for const in constants_to_check:
            assert const.symbol is not None
            assert len(const.symbol) > 0

    def test_key_constants_have_descriptions(self) -> None:
        """Test key constants have descriptions."""
        constants_to_check = [PI, E, GOLDEN_RATIO]
        for const in constants_to_check:
            assert const.description is not None
            assert len(const.description) > 0

    def test_oeis_ids_format(self) -> None:
        """Test OEIS IDs have correct format when present."""
        constants_to_check = [PI, E, GOLDEN_RATIO, CATALAN]
        for const in constants_to_check:
            if const.oeis_id is not None:
                # OEIS IDs start with 'A' followed by digits
                assert const.oeis_id.startswith("A")
                assert const.oeis_id[1:].isdigit()
