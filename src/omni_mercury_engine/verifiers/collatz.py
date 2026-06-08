# Copyright (C) 2025 Steel Security Advisors LLC
"""Oracle-validated verifier for a Collatz MATHEMATICAL_MYSTERIES scalar.

The Collatz conjecture ("iterating n -> n/2 if even, 3n+1 if odd, every n >= 1 reaches 1") is
open.  An *instance* is verified by following the map -- a deterministic dynamical process, not
a search -- and checking it reaches 1.

The verifier is certificate-first: :func:`verify_trajectory` re-checks a *handed-in* trajectory
step by step (a fabricated step is refuted), and :func:`compute_trajectory` runs the map to
produce a genuine certificate.

Honesty about limits: reaching 1 is confirmable, but non-termination is only semi-decidable --
a trajectory that exceeds the step budget yields an explicit ``inconclusive`` verdict, never a
fabricated pass or a false refutation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from omni_mercury_engine.core.global_omni_scalar_network import ScalarGroup

if TYPE_CHECKING:
    from omni_mercury_engine.core.global_omni_scalar_network import GlobalOmniScalarNetwork

logger = logging.getLogger(__name__)


class Status(Enum):
    """Trichotomy: the oracle confirms, refutes, or honestly declines to decide."""

    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class Verdict:
    """Result of adjudicating a Collatz trajectory against the map."""

    status: Status
    reason: str
    n: int

    @property
    def valid(self) -> bool:
        """Whether the trajectory was confirmed to reach 1."""
        return self.status is Status.CONFIRMED

    def as_metadata(self) -> dict[str, object]:
        """Return a JSON-friendly mapping describing this verdict."""
        return {"status": self.status.value, "reason": self.reason, "n": self.n}


def _collatz_step(x: int) -> int:
    return x // 2 if x % 2 == 0 else 3 * x + 1


def verify_trajectory(n: int, trajectory: tuple[int, ...]) -> Verdict:
    """Re-check a handed-in trajectory step by step (the certificate check, no search).

    Confirmed only when the trajectory starts at ``n``, every transition obeys the Collatz map,
    and it ends at 1.  A wrong step or wrong endpoint is refuted with a reason.
    """
    if n < 1:
        return Verdict(Status.REFUTED, f"n={n} is not a positive integer", n)
    if not trajectory or trajectory[0] != n:
        return Verdict(Status.REFUTED, f"trajectory does not start at n={n}", n)
    for i in range(len(trajectory) - 1):
        expected = _collatz_step(trajectory[i])
        if trajectory[i + 1] != expected:
            return Verdict(
                Status.REFUTED,
                f"bad step at index {i}: {trajectory[i]} -> {trajectory[i + 1]} (expected {expected})",
                n,
            )
    if trajectory[-1] != 1:
        return Verdict(Status.REFUTED, f"trajectory ends at {trajectory[-1]}, not 1", n)
    return Verdict(Status.CONFIRMED, f"{n} reaches 1 in {len(trajectory) - 1} steps", n)


def compute_trajectory(n: int, *, max_steps: int = 1_000_000) -> tuple[int, ...] | None:
    """Run the Collatz map from ``n`` (the natural dynamical process, not a search).

    Returns the trajectory ending at 1, or ``None`` if 1 is not reached within ``max_steps``
    (the semi-decidable case -- absence of a result is not a refutation).
    """
    if n < 1:
        return None
    traj = [n]
    x = n
    for _ in range(max_steps):
        if x == 1:
            return tuple(traj)
        x = _collatz_step(x)
        traj.append(x)
    return None


def register_verified_scalar(
    gosnn: GlobalOmniScalarNetwork,
    n: int,
    *,
    max_steps: int = 1_000_000,
    component_name: str = "mathematical_mysteries_collatz",
) -> tuple[float | None, Verdict]:
    """Compute and adjudicate the trajectory of ``n``, then ground a scalar from the verdict.

    Registers 1.0 when confirmed and 0.0 when refuted.  An ``inconclusive`` verdict (budget
    exceeded) registers nothing and returns ``(None, verdict)`` -- the network is never fed a
    value the oracle did not decide.
    """
    traj = compute_trajectory(n, max_steps=max_steps)
    if traj is None:
        verdict = Verdict(Status.INCONCLUSIVE, f"{n} did not reach 1 within {max_steps} steps", n)
        logger.info("Collatz scalar not registered (%s)", verdict.reason)
        return None, verdict

    verdict = verify_trajectory(n, traj)
    scalar_value = 1.0 if verdict.valid else 0.0
    gosnn.register_scalars(
        component_name=component_name,
        scalars={"omni_mystery_collatz_reaches_one": scalar_value},
        group=ScalarGroup.MATHEMATICAL_MYSTERIES,
        metadata={"source": "collatz_oracle", **verdict.as_metadata()},
    )
    logger.info("Collatz scalar grounded to %.1f (%s)", scalar_value, verdict.reason)
    return scalar_value, verdict


def demonstrate() -> None:
    """End-to-end demonstration: dynamical process -> oracle -> grounded GOSNN scalar."""
    from omni_mercury_engine.core.global_omni_scalar_network import GlobalOmniScalarNetwork

    gosnn = GlobalOmniScalarNetwork()
    traj = compute_trajectory(27)
    assert traj is not None
    true_verdict = verify_trajectory(27, traj)
    print(f"[collatz] TRUE  n=27: {true_verdict.status.value} ({true_verdict.reason})")

    # A fabricated trajectory with one wrong step must be refuted.
    bad = verify_trajectory(6, (6, 3, 10, 5, 16, 8, 4, 2, 99))
    print(f"[collatz] FALSE n=6 (tampered): {bad.status.value} ({bad.reason})")

    value, _ = register_verified_scalar(gosnn, 27)
    print(f"[collatz] grounded scalar = {value}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    demonstrate()
