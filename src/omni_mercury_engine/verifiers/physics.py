# Copyright (C) 2025 Steel Security Advisors LLC
"""Oracle-validated verifier for a PHYSICS_THEORIES scalar.

A physical law's *dimensional consistency* is decidable: both sides must reduce to the same SI
dimension.  Optionally a *numerical* relation is checked against a reference value within a
tolerance.  Both are independent of any model -- exact rational exponent arithmetic and plain
floating-point comparison -- so the grounded scalar reflects a real check, not an assertion.

This is not a claim to have *derived* physics; it confirms that a stated relation is
dimensionally and (where given) numerically self-consistent, and refutes one that is not.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from omni_mercury_engine.core.global_omni_scalar_network import ScalarGroup
from omni_mercury_engine.verifiers.dimensional import (
    ACCELERATION,
    ENERGY,
    FORCE,
    MASS,
    VELOCITY,
    Dimension,
)

if TYPE_CHECKING:
    from omni_mercury_engine.core.global_omni_scalar_network import GlobalOmniScalarNetwork

logger = logging.getLogger(__name__)

# Speed of light (m/s), exact by SI definition -- used for the numeric demonstration.
_C = 299_792_458.0


@dataclass(frozen=True)
class PhysicsRelation:
    """A claimed physical relation ``lhs = rhs`` to be checked for consistency.

    ``lhs`` and ``rhs`` are the dimensions of each side.  If both ``lhs_value`` and ``rhs_value``
    are provided they are also compared numerically within ``rel_tol``.
    """

    name: str
    lhs: Dimension
    rhs: Dimension
    lhs_value: float | None = None
    rhs_value: float | None = None
    rel_tol: float = 1e-9


@dataclass(frozen=True)
class Verdict:
    """Result of adjudicating a physical relation."""

    valid: bool
    reason: str
    relation_name: str
    checker: str = "dimensional_analysis"

    def as_metadata(self) -> dict[str, object]:
        """Return a JSON-friendly mapping describing this verdict."""
        return {
            "valid": self.valid,
            "reason": self.reason,
            "relation": self.relation_name,
            "checker": self.checker,
        }


def verify_relation(rel: PhysicsRelation) -> Verdict:
    """Adjudicate a physical relation by dimension and, if supplied, by numeric value.

    Valid only when both sides share a dimension and (when given) agree numerically; otherwise
    refuted with a reason -- e.g. ``E = m c`` is rejected because energy and momentum*velocity
    differ dimensionally.
    """
    if rel.lhs != rel.rhs:
        return Verdict(False, f"dimensional mismatch: [{rel.lhs}] != [{rel.rhs}]", rel.name)
    if rel.lhs_value is not None and rel.rhs_value is not None:
        if not math.isclose(rel.lhs_value, rel.rhs_value, rel_tol=rel.rel_tol):
            return Verdict(
                False,
                f"numeric mismatch: {rel.lhs_value} != {rel.rhs_value}",
                rel.name,
            )
    return Verdict(True, f"{rel.name}: dimensions consistent ([{rel.lhs}])", rel.name)


def register_verified_scalar(
    gosnn: GlobalOmniScalarNetwork,
    rel: PhysicsRelation,
    *,
    component_name: str = "physics_theories",
) -> tuple[float, Verdict]:
    """Adjudicate ``rel`` and register a grounded scalar (1.0 consistent / 0.0 inconsistent)."""
    verdict = verify_relation(rel)
    scalar_value = 1.0 if verdict.valid else 0.0
    gosnn.register_scalars(
        component_name=component_name,
        scalars={f"omni_physics_{rel.name}_consistent": scalar_value},
        group=ScalarGroup.PHYSICS_THEORIES,
        metadata={"source": "physics_oracle", **verdict.as_metadata()},
    )
    logger.info("Physics scalar '%s' grounded to %.1f (%s)", rel.name, scalar_value, verdict.reason)
    return scalar_value, verdict


# Bundled relations for demonstration and tests.
def mass_energy_equivalence(mass_kg: float = 1.0) -> PhysicsRelation:
    """Return E = m c^2 as a checkable relation (dimensional + numeric)."""
    return PhysicsRelation(
        name="mass_energy_equivalence",
        lhs=ENERGY,
        rhs=MASS * VELOCITY**2,
        lhs_value=mass_kg * _C**2,
        rhs_value=mass_kg * _C**2,
    )


def newtons_second_law() -> PhysicsRelation:
    """Return F = m a as a dimensional relation."""
    return PhysicsRelation(name="newtons_second_law", lhs=FORCE, rhs=MASS * ACCELERATION)


def dimensionally_wrong_mass_energy() -> PhysicsRelation:
    """Return the (incorrect) E = m c, which must be refuted on dimensions."""
    return PhysicsRelation(name="wrong_mass_energy", lhs=ENERGY, rhs=MASS * VELOCITY)


def demonstrate() -> None:
    """End-to-end demonstration: relation -> oracle -> grounded GOSNN scalar."""
    from omni_mercury_engine.core.global_omni_scalar_network import GlobalOmniScalarNetwork

    gosnn = GlobalOmniScalarNetwork()
    good = verify_relation(mass_energy_equivalence())
    print(f"[physics] TRUE  E=mc^2: valid={good.valid} ({good.reason})")
    bad = verify_relation(dimensionally_wrong_mass_energy())
    print(f"[physics] FALSE E=mc:   valid={bad.valid} ({bad.reason})")
    value, _ = register_verified_scalar(gosnn, mass_energy_equivalence())
    print(f"[physics] grounded scalar = {value}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    demonstrate()
