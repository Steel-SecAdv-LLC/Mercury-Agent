# Copyright (C) 2025 Steel Security Advisors LLC
"""Unified mystery registry: one orchestration layer over every oracle-validated verifier.

This is the registry that ties the verifier family together.  For each submitted claim it
(1) routes to the right independent oracle, (2) records a full provenance entry in an internal
ledger, and (3) registers a *bounded summary* scalar into the GOSNN -- one stable scalar key per
canonical problem, so the operational σ_Immutable input vector cannot be inflated past its
180-band contract no matter how many claims are checked.

Honesty is preserved end to end: a claim the oracle cannot decide (Collatz budget exceeded, no
Lean toolchain) is recorded as ``inconclusive`` / ``unavailable`` and registers no scalar.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from omni_mercury_engine.core.global_omni_scalar_network import ScalarGroup
from omni_mercury_engine.verifiers import (
    collatz,
    goldbach,
    lean_theorem,
    paradox,
    physics,
    twin_primes,
)
from omni_mercury_engine.verifiers.three_state import (
    KNOWN_UNDECIDABLE_IN_GENERAL,
    ThreeState,
    three_state_of,
)

if TYPE_CHECKING:
    from omni_mercury_engine.core.global_omni_scalar_network import GlobalOmniScalarNetwork
    from omni_mercury_engine.verifiers.paradox import ParadoxDefenseCertificate
    from omni_mercury_engine.verifiers.physics import PhysicsRelation

logger = logging.getLogger(__name__)

# Stay clear of the 180-band so the trained σ_Immutable gate never sees an overflow leak.
SAFE_OPERATIONAL_CAP = 175


@dataclass(frozen=True)
class LedgerEntry:
    """A single adjudicated claim with full provenance."""

    tier: str
    claim: str
    # Oracle-side status (the finer four-word vocabulary). Reconciled onto
    # the cross-repo ThreeState invariant by the ``state`` property:
    #   confirmed | refuted          -> ThreeState.GROUNDED
    #   inconclusive | unavailable   -> ThreeState.UNAVAILABLE
    #   undecidable                  -> ThreeState.UNDECIDABLE
    status: str  # confirmed | refuted | inconclusive | unavailable | undecidable
    reason: str
    checker: str
    scalar_name: str | None
    value: float | None
    registered: bool
    timestamp: float = field(default_factory=time.time)

    @property
    def state(self) -> ThreeState:
        """This entry's verdict on the unified three-state contract.

        The single cross-tier invariant shared with the governance side;
        see :func:`omni_mercury_engine.verifiers.three_state.three_state_of`
        for the per-status, cause-based reconciliation.
        """
        return three_state_of(self.status)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly mapping for this ledger entry."""
        return {
            "tier": self.tier,
            "claim": self.claim,
            "status": self.status,
            "state": self.state.value,
            "reason": self.reason,
            "checker": self.checker,
            "scalar_name": self.scalar_name,
            "value": self.value,
            "registered": self.registered,
        }


class MysteryRegistry:
    """Routes claims to oracles, records provenance, and grounds bounded GOSNN scalars."""

    def __init__(self, gosnn: GlobalOmniScalarNetwork) -> None:
        """Initialize the instance."""
        self.gosnn = gosnn
        self.ledger: list[LedgerEntry] = []

    def _operational_scalars(self) -> dict[str, float]:
        return self.gosnn._collect_all_scalars()

    def _has_room(self, scalar_name: str) -> bool:
        """Whether registering ``scalar_name`` keeps the operational vector under the band cap."""
        current = self._operational_scalars()
        if scalar_name in current:
            return True  # overwriting an existing key does not grow the vector
        return len(current) < SAFE_OPERATIONAL_CAP

    def _record(
        self,
        *,
        tier: str,
        claim: str,
        status: str,
        reason: str,
        checker: str,
        group: ScalarGroup,
        scalar_name: str,
        component_name: str,
        provenance: dict[str, object],
    ) -> LedgerEntry:
        """Ground a decided claim into a bounded GOSNN scalar and append a ledger entry."""
        value: float | None = None
        registered = False
        if status in ("confirmed", "refuted"):
            value = 1.0 if status == "confirmed" else 0.0
            if self._has_room(scalar_name):
                self.gosnn.register_scalars(
                    component_name=component_name,
                    scalars={scalar_name: value},
                    group=group,
                    metadata={"source": f"{tier}_oracle", **provenance},
                )
                registered = True
            else:
                reason = f"{reason} [band budget reached; ledger-only]"
        entry = LedgerEntry(
            tier=tier,
            claim=claim,
            status=status,
            reason=reason,
            checker=checker,
            scalar_name=scalar_name if registered else None,
            value=value,
            registered=registered,
        )
        self.ledger.append(entry)
        return entry

    def submit_goldbach(self, n: int) -> LedgerEntry:
        """Adjudicate a Goldbach partition for even ``n``."""
        cert = goldbach.find_partition(n)
        if cert is None:
            return self._record(
                tier="number_theory",
                claim=f"goldbach(n={n})",
                status="inconclusive",
                reason=f"no partition proposed for n={n}",
                checker="deterministic_miller_rabin",
                group=ScalarGroup.MATHEMATICAL_MYSTERIES,
                scalar_name="omni_mystery_goldbach_verified",
                component_name="mathematical_mysteries_goldbach",
                provenance={"n": n},
            )
        verdict = goldbach.verify_certificate(cert)
        return self._record(
            tier="number_theory",
            claim=f"goldbach(n={n})",
            status="confirmed" if verdict.valid else "refuted",
            reason=verdict.reason,
            checker=verdict.checker,
            group=ScalarGroup.MATHEMATICAL_MYSTERIES,
            scalar_name="omni_mystery_goldbach_verified",
            component_name="mathematical_mysteries_goldbach",
            provenance=verdict.as_metadata(),
        )

    def submit_twin_prime(self, p: int) -> LedgerEntry:
        """Adjudicate the twin-prime pair ``(p, p + 2)``."""
        verdict = twin_primes.verify_certificate(twin_primes.TwinPrimeCertificate(p=p))
        return self._record(
            tier="number_theory",
            claim=f"twin_prime(p={p})",
            status="confirmed" if verdict.valid else "refuted",
            reason=verdict.reason,
            checker=verdict.checker,
            group=ScalarGroup.MATHEMATICAL_MYSTERIES,
            scalar_name="omni_mystery_twin_prime_verified",
            component_name="mathematical_mysteries_twin_prime",
            provenance=verdict.as_metadata(),
        )

    def submit_collatz(self, n: int, *, max_steps: int = 1_000_000) -> LedgerEntry:
        """Adjudicate whether the Collatz trajectory of ``n`` reaches 1."""
        traj = collatz.compute_trajectory(n, max_steps=max_steps)
        if traj is None:
            return self._record(
                tier="dynamical",
                claim=f"collatz(n={n})",
                status="inconclusive",
                reason=f"{n} did not reach 1 within {max_steps} steps",
                checker="collatz_map",
                group=ScalarGroup.MATHEMATICAL_MYSTERIES,
                scalar_name="omni_mystery_collatz_reaches_one",
                component_name="mathematical_mysteries_collatz",
                provenance={"n": n, "max_steps": max_steps},
            )
        verdict = collatz.verify_trajectory(n, traj)
        return self._record(
            tier="dynamical",
            claim=f"collatz(n={n})",
            status=verdict.status.value,
            reason=verdict.reason,
            checker="collatz_map",
            group=ScalarGroup.MATHEMATICAL_MYSTERIES,
            scalar_name="omni_mystery_collatz_reaches_one",
            component_name="mathematical_mysteries_collatz",
            provenance=verdict.as_metadata(),
        )

    def submit_physics(self, relation: PhysicsRelation) -> LedgerEntry:
        """Adjudicate a physical relation for dimensional (and numeric) consistency."""
        verdict = physics.verify_relation(relation)
        return self._record(
            tier="physics",
            claim=f"physics({relation.name})",
            status="confirmed" if verdict.valid else "refuted",
            reason=verdict.reason,
            checker=verdict.checker,
            group=ScalarGroup.PHYSICS_THEORIES,
            scalar_name=f"omni_physics_{relation.name}_consistent",
            component_name="physics_theories",
            provenance=verdict.as_metadata(),
        )

    def submit_paradox(self, cert: ParadoxDefenseCertificate) -> LedgerEntry:
        """Adjudicate a paradox defense for consistency."""
        verdict = paradox.verify_defense(cert)
        return self._record(
            tier="paradox",
            claim=f"paradox({cert.name})",
            status="confirmed" if verdict.valid else "refuted",
            reason=verdict.reason,
            checker=verdict.checker,
            group=ScalarGroup.PARADOX_DEFENSE,
            scalar_name=f"omni_paradox_{cert.name}_defended",
            component_name="paradox_defense",
            provenance=verdict.as_metadata(),
        )

    def submit_theorem(self, source: str, *, name: str, timeout: float = 60.0) -> LedgerEntry:
        """Adjudicate a Lean proof script; records ``unavailable`` if no toolchain is present."""
        verdict = lean_theorem.verify_lean_proof(source, timeout=timeout)
        if not verdict.available:
            return self._record(
                tier="theorem",
                claim=f"theorem({name})",
                status="unavailable",
                reason=verdict.reason,
                checker=verdict.checker,
                group=ScalarGroup.MATHEMATICAL_MYSTERIES,
                scalar_name=f"omni_mystery_theorem_{name}_verified",
                component_name="mathematical_mysteries_theorem",
                provenance=verdict.as_metadata(),
            )
        return self._record(
            tier="theorem",
            claim=f"theorem({name})",
            status="confirmed" if verdict.valid else "refuted",
            reason=verdict.reason,
            checker=verdict.checker,
            group=ScalarGroup.MATHEMATICAL_MYSTERIES,
            scalar_name=f"omni_mystery_theorem_{name}_verified",
            component_name="mathematical_mysteries_theorem",
            provenance=verdict.as_metadata(),
        )

    def submit_undecidable(self, problem: str, *, claim: str | None = None) -> LedgerEntry:
        """Record a claim about a problem with no decision procedure in principle.

        For the universally-quantified open conjectures behind the instance
        verifiers -- Collatz-in-general, the infinitude of twin primes, any
        Millennium-class problem -- an instance oracle can settle single
        cases but never the universal statement.  This is the
        :attr:`ThreeState.UNDECIDABLE` path: it **registers nothing, ever**
        (distinct from ``UNAVAILABLE``, which is a decidable instance merely
        not produced this run), and grounds no scalar.

        Args:
            problem: A key into
                :data:`~omni_mercury_engine.verifiers.three_state.KNOWN_UNDECIDABLE_IN_GENERAL`
                (e.g. ``"collatz_general"``), or any caller-asserted
                undecidable-in-general problem name.
            claim: Optional human-readable claim string for the ledger.

        Returns:
            The recorded :class:`LedgerEntry` (``status="undecidable"``,
            ``registered=False``, ``value=None``).
        """
        description = KNOWN_UNDECIDABLE_IN_GENERAL.get(problem)
        reason = description or (
            f"{problem}: no decision procedure in principle (instance oracle "
            "cannot settle a universal claim over an infinite domain)"
        )
        return self._record(
            tier="open_problem",
            claim=claim or problem,
            status="undecidable",
            reason=reason,
            checker="none",
            group=ScalarGroup.MATHEMATICAL_MYSTERIES,
            scalar_name=f"omni_open_problem_{problem}",
            component_name="open_problem_undecidable",
            provenance={"problem": problem, "undecidable_in_general": True},
        )

    def summary(self) -> dict[str, object]:
        """Return aggregate counts, registered-scalar count, and band headroom."""
        by_status: dict[str, int] = {}
        by_state: dict[str, int] = {}
        by_tier: dict[str, int] = {}
        for e in self.ledger:
            by_status[e.status] = by_status.get(e.status, 0) + 1
            by_state[e.state.value] = by_state.get(e.state.value, 0) + 1
            by_tier[e.tier] = by_tier.get(e.tier, 0) + 1
        operational = len(self._operational_scalars())
        return {
            "total_claims": len(self.ledger),
            "by_status": by_status,
            "by_state": by_state,
            "by_tier": by_tier,
            "registered_scalars": sum(1 for e in self.ledger if e.registered),
            "operational_scalar_count": operational,
            "band_cap": SAFE_OPERATIONAL_CAP,
            "band_remaining": SAFE_OPERATIONAL_CAP - operational,
        }
