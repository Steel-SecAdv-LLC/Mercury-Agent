# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dimensional-analysis oracle shared by the physics verifiers.

A physical dimension is an exact vector of rational exponents over the seven SI base
quantities (mass, length, time, current, temperature, amount, luminous intensity).  Checking
that two expressions share a dimension is therefore exact linear algebra over the rationals --
a decidable, auditable oracle with no model and no floating-point error.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

_BASE_NAMES: tuple[str, ...] = ("M", "L", "T", "I", "Theta", "N", "J")
_NDIM = len(_BASE_NAMES)


@dataclass(frozen=True)
class Dimension:
    """An SI dimension as exact rational exponents over the seven base quantities."""

    exponents: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        """Finalize dataclass initialization."""
        if len(self.exponents) != _NDIM:
            raise ValueError(f"expected {_NDIM} exponents, got {len(self.exponents)}")

    @classmethod
    def base(cls, index: int) -> Dimension:
        """Return the unit dimension for the base quantity at ``index``."""
        exps = [Fraction(0)] * _NDIM
        exps[index] = Fraction(1)
        return cls(tuple(exps))

    def __mul__(self, other: Dimension) -> Dimension:
        """Implement the Python data model method."""
        return Dimension(tuple(a + b for a, b in zip(self.exponents, other.exponents)))

    def __truediv__(self, other: Dimension) -> Dimension:
        """Implement the Python data model method."""
        return Dimension(tuple(a - b for a, b in zip(self.exponents, other.exponents)))

    def __pow__(self, power: Fraction | int) -> Dimension:
        """Implement the Python data model method."""
        p = Fraction(power)
        return Dimension(tuple(a * p for a in self.exponents))

    def __str__(self) -> str:
        """Return the string representation."""
        parts = [f"{name}^{exp}" for name, exp in zip(_BASE_NAMES, self.exponents) if exp != 0]
        return " ".join(parts) if parts else "dimensionless"


DIMENSIONLESS = Dimension(tuple(Fraction(0) for _ in range(_NDIM)))
MASS = Dimension.base(0)
LENGTH = Dimension.base(1)
TIME = Dimension.base(2)
CURRENT = Dimension.base(3)
TEMPERATURE = Dimension.base(4)
AMOUNT = Dimension.base(5)
LUMINOUS = Dimension.base(6)

# Common derived dimensions, built from the base set.
VELOCITY = LENGTH / TIME
ACCELERATION = VELOCITY / TIME
FORCE = MASS * ACCELERATION
ENERGY = FORCE * LENGTH
MOMENTUM = MASS * VELOCITY
POWER = ENERGY / TIME
CHARGE = CURRENT * TIME
