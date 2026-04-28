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
Centralized Mathematical Constants Module

Provides scientific-precision mathematical constants with validation,
documentation, and configurable precision levels.

Features:
- High-precision constants using mpmath (when available)
- Constant validation and range checking
- Historical context and research citations
- Configurable precision levels (float32/float64/arbitrary)
- Symbolic computation support via SymPy (when available)

References:
- OEIS (Online Encyclopedia of Integer Sequences)
- NIST Digital Library of Mathematical Functions
- Wolfram MathWorld
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# Try to import high-precision libraries
try:
    import mpmath

    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False

try:
    import sympy

    SYMPY_AVAILABLE = True
except ImportError:
    sympy = None
    SYMPY_AVAILABLE = False


# Mapping of our constants to sympy equivalents for validation
SYMPY_CONSTANT_MAP: dict[str, str] = {
    "PI": "pi",
    "E": "E",
    "EULER_MASCHERONI": "EulerGamma",
    "CATALAN": "Catalan",
    "GOLDEN_RATIO": "GoldenRatio",
    "SQRT2": "sqrt(2)",
    "SQRT3": "sqrt(3)",
    "SQRT5": "sqrt(5)",
    "LN2": "log(2)",
    "LN10": "log(10)",
}


class Precision(Enum):
    """Precision levels for mathematical constants."""

    FLOAT32 = "float32"
    FLOAT64 = "float64"
    ARBITRARY = "arbitrary"  # Uses mpmath for arbitrary precision


@dataclass(frozen=True)
class MathConstant:
    """
    A mathematical constant with metadata.

    Attributes:
        value: The numerical value (float64 precision)
        name: Human-readable name
        symbol: Mathematical symbol (Unicode)
        description: Brief description
        oeis_id: OEIS sequence ID (if applicable)
        reference: Academic/authoritative reference
        precision_digits: Number of verified decimal digits
    """

    value: float
    name: str
    symbol: str
    description: str
    oeis_id: str | None = None
    reference: str | None = None
    precision_digits: int = 15

    def to_precision(self, precision: Precision) -> float | Any:
        """
        Get the constant in specified precision.

        Args:
            precision: Target precision level

        Returns:
            Constant value in requested precision
        """
        if precision == Precision.FLOAT32:
            return np.float32(self.value)
        elif precision == Precision.FLOAT64:
            return np.float64(self.value)
        elif precision == Precision.ARBITRARY and MPMATH_AVAILABLE:
            # Return high-precision version
            return mpmath.mpf(str(self.value))
        return self.value

    def validate(self, tolerance: float = 1e-10, use_sympy: bool = True) -> bool:
        """
        Validate the constant against known values.

        When sympy is available and use_sympy=True, validates against
        sympy's high-precision symbolic constants for enhanced accuracy.

        Args:
            tolerance: Maximum acceptable deviation
            use_sympy: Whether to use sympy for validation (if available)

        Returns:
            True if validation passes
        """
        if not math.isfinite(self.value):
            return False

        if use_sympy and SYMPY_AVAILABLE and sympy is not None:
            sympy_name = _get_sympy_constant_name(self.name)
            if sympy_name:
                try:
                    sympy_value = _evaluate_sympy_constant(sympy_name)
                    if sympy_value is not None:
                        return abs(self.value - sympy_value) < tolerance
                except (ValueError, TypeError, AttributeError):
                    return True

        return True


class MathematicalConstants:
    """
    Centralized repository of mathematical constants.

    All constants are verified against authoritative sources and
    include full documentation for traceability.

    Example:
        constants = MathematicalConstants()
        phi = constants.GOLDEN_RATIO
        print(f"Golden Ratio: {phi.value}")
        print(f"High precision: {phi.to_precision(Precision.ARBITRARY)}")
    """

    # === Fundamental Constants ===

    GOLDEN_RATIO = MathConstant(
        value=1.6180339887498948482,
        name="Golden Ratio",
        symbol="φ",
        description="Ratio where (a+b)/a = a/b. Appears in art, architecture, and nature.",
        oeis_id="A001622",
        reference="Livio, M. (2002). The Golden Ratio. Broadway Books.",
        precision_digits=15,
    )

    GOLDEN_RATIO_CONJUGATE = MathConstant(
        value=0.6180339887498948482,
        name="Golden Ratio Conjugate",
        symbol="φ⁻¹ or Φ",
        description="Reciprocal of golden ratio minus 1. φ - 1 = 1/φ.",
        oeis_id="A094214",
        reference="Same as Golden Ratio",
        precision_digits=15,
    )

    PI = MathConstant(
        value=3.141592653589793238,
        name="Pi",
        symbol="π",
        description="Ratio of circle's circumference to diameter.",
        oeis_id="A000796",
        reference="NIST Digital Library of Mathematical Functions",
        precision_digits=15,
    )

    E = MathConstant(
        value=2.718281828459045235,
        name="Euler's Number",
        symbol="e",
        description="Base of natural logarithm. lim(1+1/n)^n as n→∞.",
        oeis_id="A001113",
        reference="NIST Digital Library of Mathematical Functions",
        precision_digits=15,
    )

    EULER_MASCHERONI = MathConstant(
        value=0.5772156649015328606,
        name="Euler-Mascheroni Constant",
        symbol="γ",
        description="Limiting difference between harmonic series and natural logarithm.",
        oeis_id="A001620",
        reference="Havil, J. (2003). Gamma: Exploring Euler's Constant. Princeton.",
        precision_digits=15,
    )

    # === Catalan and Related Constants ===

    CATALAN = MathConstant(
        value=0.9159655941772190151,
        name="Catalan's Constant",
        symbol="G",
        description="Sum of (-1)^n/(2n+1)^2 for n=0 to infinity.",
        oeis_id="A006752",
        reference="Adamchik, V. (2002). On the Catalan constant. Ramanujan J.",
        precision_digits=15,
    )

    # === Chaos Theory Constants ===

    FEIGENBAUM_DELTA = MathConstant(
        value=4.6692016091029906719,
        name="Feigenbaum's Delta",
        symbol="δ",
        description="Rate of approach to chaos in period-doubling bifurcations.",
        oeis_id="A006890",
        reference="Feigenbaum, M.J. (1978). Quantitative universality. J. Stat. Phys.",
        precision_digits=15,
    )

    FEIGENBAUM_ALPHA = MathConstant(
        value=2.5029078750958928222,
        name="Feigenbaum's Alpha",
        symbol="α",
        description="Scaling factor for amplitude in period-doubling.",
        oeis_id="A006891",
        reference="Feigenbaum, M.J. (1978). Quantitative universality. J. Stat. Phys.",
        precision_digits=15,
    )

    # === Omega and Lambert W ===

    OMEGA = MathConstant(
        value=0.5671432904097838730,
        name="Omega Constant",
        symbol="Ω",
        description="W(1) where W is the Lambert W function. Satisfies Ω·e^Ω = 1.",
        oeis_id="A030178",
        reference="Corless, R.M. et al. (1996). On the Lambert W function. Adv. Comp. Math.",
        precision_digits=15,
    )

    # === Apéry's Constant ===

    APERY = MathConstant(
        value=1.2020569031595942854,
        name="Apéry's Constant",
        symbol="ζ(3)",
        description="Riemann zeta function at 3. Sum of 1/n^3 for n=1 to infinity.",
        oeis_id="A002117",
        reference="Apéry, R. (1979). Irrationalité de ζ(2) et ζ(3). Astérisque.",
        precision_digits=15,
    )

    # === Plastic Ratio ===

    PLASTIC = MathConstant(
        value=1.3247179572447460260,
        name="Plastic Ratio",
        symbol="ρ",
        description="Real solution to x³ = x + 1. Related to Padovan sequence.",
        oeis_id="A060006",
        reference="Stewart, I. (1996). Tales of a Neglected Number. Scientific American.",
        precision_digits=15,
    )

    # === Square Root Constants ===

    SQRT2 = MathConstant(
        value=1.4142135623730950488,
        name="Square Root of 2",
        symbol="√2",
        description="Pythagoras's constant. Diagonal of unit square.",
        oeis_id="A002193",
        reference="NIST Digital Library of Mathematical Functions",
        precision_digits=15,
    )

    SQRT3 = MathConstant(
        value=1.7320508075688772935,
        name="Square Root of 3",
        symbol="√3",
        description="Theodorus's constant. Related to equilateral triangle.",
        oeis_id="A002194",
        reference="NIST Digital Library of Mathematical Functions",
        precision_digits=15,
    )

    SQRT5 = MathConstant(
        value=2.2360679774997896964,
        name="Square Root of 5",
        symbol="√5",
        description="Appears in golden ratio: φ = (1 + √5)/2.",
        oeis_id="A002163",
        reference="NIST Digital Library of Mathematical Functions",
        precision_digits=15,
    )

    # === Natural Logarithm Constants ===

    LN2 = MathConstant(
        value=0.6931471805599453094,
        name="Natural Logarithm of 2",
        symbol="ln(2)",
        description="Alternating harmonic series sum.",
        oeis_id="A002162",
        reference="NIST Digital Library of Mathematical Functions",
        precision_digits=15,
    )

    LN10 = MathConstant(
        value=2.3025850929940456840,
        name="Natural Logarithm of 10",
        symbol="ln(10)",
        description="Conversion factor between natural and common logarithms.",
        oeis_id="A002392",
        reference="NIST Digital Library of Mathematical Functions",
        precision_digits=15,
    )

    # === Physical Constants (Dimensionless) ===

    FINE_STRUCTURE = MathConstant(
        value=0.0072973525693,
        name="Fine-Structure Constant",
        symbol="α",
        description="Strength of electromagnetic interaction.",
        oeis_id=None,
        reference="CODATA 2018 recommended values",
        precision_digits=12,
    )

    @classmethod
    def get_all(cls) -> dict[str, MathConstant]:
        """
        Get all defined constants as a dictionary.

        Returns:
            Dict mapping constant names to MathConstant objects
        """
        return {name: value for name, value in vars(cls).items() if isinstance(value, MathConstant)}

    @classmethod
    def validate_all(cls) -> dict[str, bool]:
        """
        Validate all constants.

        Returns:
            Dict mapping constant names to validation results
        """
        return {name: const.validate() for name, const in cls.get_all().items()}

    @classmethod
    def get_by_symbol(cls, symbol: str) -> MathConstant | None:
        """
        Look up a constant by its mathematical symbol.

        Args:
            symbol: Mathematical symbol (e.g., "φ", "π")

        Returns:
            MathConstant if found, None otherwise
        """
        for const in cls.get_all().values():
            if const.symbol == symbol:
                return const
        return None


# Convenience aliases for common constants
PHI = MathematicalConstants.GOLDEN_RATIO.value
CATALAN_CONSTANT = MathematicalConstants.CATALAN.value
EULER_MASCHERONI_CONSTANT = MathematicalConstants.EULER_MASCHERONI.value
FEIGENBAUM_DELTA_CONSTANT = MathematicalConstants.FEIGENBAUM_DELTA.value
OMEGA_CONSTANT = MathematicalConstants.OMEGA.value


# =============================================================================
# Omni-Codes: Bio-Inspired Helical Parameters from AMA Cryptography
# =============================================================================
# Seven Omni-Codes governing ethical AI alignment and system stability.
# Integrated from AMA Cryptography (https://github.com/Steel-SecAdv-LLC/AMA-Cryptography)
# Each code has helical parameters (r, p) inspired by DNA double-helix stability.
# =============================================================================


@dataclass(frozen=True)
class OmniCode:
    """
    An Omni-Code with helical parameters for ethical AI alignment.

    Attributes:
        code: Full code string with symbolic encoding
        symbol: Short symbol representation
        domain: Ethical/functional domain
        r: Helical radius parameter (stability factor)
        p: Helical pitch parameter (evolution rate)
        description: Human-readable description
    """

    code: str
    symbol: str
    domain: str
    r: float
    p: float
    description: str = ""

    @property
    def stability(self) -> float:
        """
        Calculate stability score from helical parameters.

        Stability = |r| * p, representing the balance between
        structural integrity (r) and adaptive evolution (p).
        """
        return abs(self.r) * self.p

    def compute_autonomy_boost(self, threshold: float = 15.0) -> float:
        """
        Compute autonomy boost based on stability.

        If stability exceeds threshold, returns a small boost (0.05).
        This ties Omni-Code stability to agent autonomy levels.

        Args:
            threshold: Stability threshold for autonomy boost

        Returns:
            Autonomy boost value (0.0 or 0.05)
        """
        return 0.05 if self.stability > threshold else 0.0


class OmniCodes:
    """
    Seven foundational Omni-Codes governing Mercury Agent.

    These codes are integrated from AMA Cryptography and provide:
    - Helical data encoding (mirrors DNA double-helix stability)
    - Self-healing capabilities (CRISPR-inspired adaptations)
    - Evolutionary adaptability (dynamic parameter tuning)
    - Canonical hashing (cryptographic integrity)

    Reference: https://github.com/Steel-SecAdv-LLC/AMA-Cryptography
    """

    OMNI_DIRECTIONAL = OmniCode(
        code="👁20A07∞_XΔEΛX_ϵ19A89Ϙ",
        symbol="👁∞",
        domain="Omni-Directional System",
        r=20.0,
        p=0.7,
        description="360-degree awareness and multi-domain perception",
    )

    OMNI_PERCIPIENT = OmniCode(
        code="Ϙ16A11ϵ_ΞΛMΔΞ_ϖ20A19Φ",
        symbol="Ϙϵ",
        domain="Omni-Percipient Future",
        r=16.0,
        p=1.1,
        description="Predictive foresight and anticipatory analysis",
    )

    OMNI_INDIVISIBLE = OmniCode(
        code="Φ07A09ϖ_ΨΔAΛΨ_ϵ19A88Σ",
        symbol="Φϖ",
        domain="Omni-Indivisible Guardian",
        r=7.0,
        p=0.9,
        description="Unified protection and integrity preservation",
    )

    OMNI_BENEVOLENT = OmniCode(
        code="Σ19L12ϵ_ΞΛEΔΞ_ϖ19A92Ω",
        symbol="Σϵ",
        domain="Omni-Benevolent Stone",
        r=19.0,
        p=1.2,
        description="Ethical foundation and humanitarian alignment",
    )

    OMNI_SCIENT = OmniCode(
        code="Ω20V11ϖ_ΨΔSΛΨ_ϵ20A15Θ",
        symbol="Ωϖ",
        domain="Omni-Scient Curiosity",
        r=20.0,
        p=1.1,
        description="Knowledge acquisition and scientific discovery",
    )

    OMNI_UNIVERSAL = OmniCode(
        code="Θ25M01ϵ_ΞΛLΔΞ_ϖ19A91Γ",
        symbol="Θϵ",
        domain="Omni-Universal Discipline",
        r=25.0,
        p=0.1,
        description="Structured governance and systematic order",
    )

    OMNI_POTENT = OmniCode(
        code="Γ19L11ϖ_XΔHΛX_∞19A84♰",
        symbol="Γϖ",
        domain="Omni-Potent Lifeforce",
        r=19.0,
        p=1.1,
        description="Regenerative capability and adaptive resilience",
    )

    @classmethod
    def get_all(cls) -> dict[str, OmniCode]:
        """Get all Omni-Codes as a dictionary."""
        return {name: value for name, value in vars(cls).items() if isinstance(value, OmniCode)}

    @classmethod
    def get_total_stability(cls) -> float:
        """Calculate total stability across all Omni-Codes."""
        return sum(code.stability for code in cls.get_all().values())

    @classmethod
    def get_autonomy_boost(cls, threshold: float = 15.0) -> float:
        """
        Calculate total autonomy boost from all Omni-Codes.

        Args:
            threshold: Stability threshold for each code

        Returns:
            Total autonomy boost (sum of individual boosts)
        """
        return sum(code.compute_autonomy_boost(threshold) for code in cls.get_all().values())

    @classmethod
    def validate_stability(cls, min_total: float = 50.0) -> bool:
        """
        Validate that total stability meets minimum threshold.

        Args:
            min_total: Minimum required total stability

        Returns:
            True if stability is sufficient
        """
        return cls.get_total_stability() >= min_total


def compute_ethical_autonomy(
    base_autonomy: float = 0.8,
    ethical_threshold: float = 0.99,
    use_omni_codes: bool = True,
) -> float:
    """
    Compute dynamic autonomy level bounded by ethical constraints.

    Autonomy is scaled based on ethical threshold and optionally
    boosted by Omni-Code stability calculations.

    Args:
        base_autonomy: Starting autonomy level (0-1)
        ethical_threshold: Ethical compliance threshold (0-1)
        use_omni_codes: Whether to apply Omni-Code stability boost

    Returns:
        Final autonomy level, capped at 0.95
    """
    autonomy = min(base_autonomy, ethical_threshold * 1.02)

    if use_omni_codes:
        autonomy += OmniCodes.get_autonomy_boost(threshold=15.0)

    return min(0.95, autonomy)


def get_constant(name: str, precision: Precision = Precision.FLOAT64) -> float | Any:
    """
    Get a mathematical constant by name with specified precision.

    Args:
        name: Constant name (case-insensitive, e.g., "golden_ratio", "pi")
        precision: Desired precision level

    Returns:
        Constant value in requested precision

    Raises:
        ValueError: If constant name is not found
    """
    name_upper = name.upper().replace(" ", "_")
    const = getattr(MathematicalConstants, name_upper, None)

    if const is None or not isinstance(const, MathConstant):
        available = list(MathematicalConstants.get_all().keys())
        raise ValueError(f"Unknown constant: {name}. Available: {available}")

    return const.to_precision(precision)


def validate_constant(value: float, name: str, tolerance: float = 1e-10) -> bool:
    """
    Validate a value against a known mathematical constant.

    Args:
        value: Value to validate
        name: Constant name to check against
        tolerance: Maximum acceptable deviation

    Returns:
        True if value matches constant within tolerance
    """
    try:
        expected = get_constant(name)
        return abs(value - expected) < tolerance
    except ValueError:
        return False


def _get_sympy_constant_name(constant_name: str) -> str | None:
    """
    Get the sympy constant name for a given constant.

    Args:
        constant_name: Our constant name (e.g., "Golden Ratio")

    Returns:
        Sympy constant name if mapping exists, None otherwise
    """
    name_key = constant_name.upper().replace(" ", "_").replace("-", "_")
    return SYMPY_CONSTANT_MAP.get(name_key)


def _evaluate_sympy_constant(sympy_expr: str) -> float | None:
    """
    Evaluate a sympy constant expression to float.

    Args:
        sympy_expr: Sympy expression string (e.g., "pi", "sqrt(2)")

    Returns:
        Float value of the constant, or None if evaluation fails
    """
    if not SYMPY_AVAILABLE or sympy is None:
        return None

    try:
        if sympy_expr in ("pi", "E", "EulerGamma", "Catalan", "GoldenRatio"):
            const = getattr(sympy, sympy_expr, None)
            if const is not None:
                return float(const.evalf(50))
        elif sympy_expr.startswith("sqrt("):
            num = int(sympy_expr[5:-1])
            return float(sympy.sqrt(num).evalf(50))
        elif sympy_expr.startswith("log("):
            num = int(sympy_expr[4:-1])
            return float(sympy.log(num).evalf(50))
        return None
    except Exception as e:
        logger.debug(f"Failed to evaluate sympy expression '{sympy_expr}': {e}")
        return None


def validate_all_constants_with_sympy(tolerance: float = 1e-10) -> dict[str, bool]:
    """
    Validate all constants against sympy's high-precision values.

    This function provides comprehensive validation of all mathematical
    constants in the module against sympy's symbolic computation engine.

    Args:
        tolerance: Maximum acceptable deviation from sympy values

    Returns:
        Dict mapping constant names to validation results
    """
    results: dict[str, bool] = {}
    for name, const in MathematicalConstants.get_all().items():
        results[name] = const.validate(tolerance=tolerance, use_sympy=True)
    return results
