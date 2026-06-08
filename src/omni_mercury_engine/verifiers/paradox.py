# Copyright (C) 2025 Steel Security Advisors LLC
"""Oracle-validated verifier for a PARADOX_DEFENSE scalar.

A "paradox defense" is formalised as two propositional theories: the *naive* framing that
produces the contradiction, and the *defense* -- a reformulation that resolves it.  The DPLL
oracle settles both questions exactly:

* the naive framing must be UNSATISFIABLE (there really was a contradiction to defend against);
* the defense must be SATISFIABLE (the reformulation is genuinely consistent).

A defense that is itself inconsistent, or a "paradox" that was never contradictory, is refuted.
This is decidable logic, independent of any model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from omni_mercury_engine.core.global_omni_scalar_network import ScalarGroup
from omni_mercury_engine.verifiers.propositional import CNF, iff, is_satisfiable, var

if TYPE_CHECKING:
    from omni_mercury_engine.core.global_omni_scalar_network import GlobalOmniScalarNetwork

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParadoxDefenseCertificate:
    """A named paradox with its naive (contradictory) and defended (consistent) theories."""

    name: str
    naive: CNF
    defense: CNF


@dataclass(frozen=True)
class Verdict:
    """Result of adjudicating a paradox defense."""

    valid: bool
    reason: str
    name: str
    checker: str = "dpll_sat"

    def as_metadata(self) -> dict[str, object]:
        """Return a JSON-friendly mapping describing this verdict."""
        return {
            "valid": self.valid,
            "reason": self.reason,
            "name": self.name,
            "checker": self.checker,
        }


def verify_defense(cert: ParadoxDefenseCertificate) -> Verdict:
    """Adjudicate a paradox defense with the DPLL consistency oracle.

    Confirmed only when the naive framing is unsatisfiable (a real contradiction) and the
    defense is satisfiable (a genuine resolution).  A consistent "paradox" or an inconsistent
    "defense" is refuted.
    """
    if is_satisfiable(cert.naive):
        return Verdict(False, "naive framing is satisfiable: no genuine contradiction", cert.name)
    if not is_satisfiable(cert.defense):
        return Verdict(False, "defense theory is itself inconsistent", cert.name)
    return Verdict(True, "naive framing contradictory; defense consistent", cert.name)


def register_verified_scalar(
    gosnn: GlobalOmniScalarNetwork,
    cert: ParadoxDefenseCertificate,
    *,
    component_name: str = "paradox_defense",
) -> tuple[float, Verdict]:
    """Adjudicate ``cert`` and register a grounded scalar (1.0 sound defense / 0.0 otherwise)."""
    verdict = verify_defense(cert)
    scalar_value = 1.0 if verdict.valid else 0.0
    gosnn.register_scalars(
        component_name=component_name,
        scalars={f"omni_paradox_{cert.name}_defended": scalar_value},
        group=ScalarGroup.PARADOX_DEFENSE,
        metadata={"source": "paradox_oracle", **verdict.as_metadata()},
    )
    logger.info(
        "Paradox scalar '%s' grounded to %.1f (%s)", cert.name, scalar_value, verdict.reason
    )
    return scalar_value, verdict


def liar_paradox() -> ParadoxDefenseCertificate:
    """The Liar ("this sentence is false").

    Naive: ``L <-> ~L`` (unsatisfiable).  Defense: a Tarskian level split ``L0 <-> ~L1`` with the
    two truth levels independent (satisfiable).
    """
    naive = iff(var("L"), ~var("L"))
    defense = iff(var("L0"), ~var("L1"))
    return ParadoxDefenseCertificate(name="liar", naive=naive, defense=defense)


def russell_paradox() -> ParadoxDefenseCertificate:
    """Russell's set ("the set of all sets that do not contain themselves").

    Naive: ``R <-> ~R`` (unsatisfiable).  Defense: a type-stratified ``R0 <-> ~R1`` (satisfiable).
    """
    naive = iff(var("R"), ~var("R"))
    defense = iff(var("R0"), ~var("R1"))
    return ParadoxDefenseCertificate(name="russell", naive=naive, defense=defense)


def demonstrate() -> None:
    """End-to-end demonstration: paradox theory -> oracle -> grounded GOSNN scalar."""
    from omni_mercury_engine.core.global_omni_scalar_network import GlobalOmniScalarNetwork

    gosnn = GlobalOmniScalarNetwork()
    good = verify_defense(liar_paradox())
    print(f"[paradox] TRUE  liar: valid={good.valid} ({good.reason})")

    # A bogus defense that is itself contradictory must be refuted.
    bogus = ParadoxDefenseCertificate(
        name="bogus",
        naive=iff(var("X"), ~var("X")),
        defense=(frozenset({var("A")}), frozenset({~var("A")})),
    )
    bad = verify_defense(bogus)
    print(f"[paradox] FALSE bogus defense: valid={bad.valid} ({bad.reason})")

    value, _ = register_verified_scalar(gosnn, liar_paradox())
    print(f"[paradox] grounded scalar = {value}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    demonstrate()
