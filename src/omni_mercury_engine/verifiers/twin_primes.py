"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Oracle-validated verifier for a twin-prime MATHEMATICAL_MYSTERIES scalar.

The twin-prime conjecture ("infinitely many primes p with p+2 also prime") is open, so it is a
genuine ``MATHEMATICAL_MYSTERIES`` entry.  An *instance* -- a claimed pair (p, p+2) -- is a
finite certificate that the primality oracle confirms or refutes directly.

This module is certificate-first: :func:`verify_certificate` adjudicates a pair that is *handed
in*; it never searches.  :func:`find_twin_prime` is a thin, swappable candidate generator -- the
proposer is not the artifact, the oracle is.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from omni_mercury_engine.core.global_omni_scalar_network import ScalarGroup
from omni_mercury_engine.verifiers.primality import is_prime

if TYPE_CHECKING:
    from omni_mercury_engine.core.global_omni_scalar_network import GlobalOmniScalarNetwork

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TwinPrimeCertificate:
    """A claimed twin-prime pair ``(p, p + 2)`` awaiting adjudication."""

    p: int


@dataclass(frozen=True)
class Verdict:
    """Result of adjudicating a twin-prime certificate against the oracle."""

    valid: bool
    reason: str
    certificate: TwinPrimeCertificate
    checker: str = "deterministic_miller_rabin"

    def as_metadata(self) -> dict[str, object]:
        """Return a JSON-friendly mapping describing this verdict."""
        return {
            "valid": self.valid,
            "reason": self.reason,
            "checker": self.checker,
            "p": self.certificate.p,
            "q": self.certificate.p + 2,
        }


def verify_certificate(cert: TwinPrimeCertificate) -> Verdict:
    """Adjudicate a twin-prime certificate with the independent oracle.

    Valid only when both ``p`` and ``p + 2`` are prime; anything else is refuted with a reason
    (e.g. ``(7, 9)`` is rejected because 9 = 3*3).
    """
    p = cert.p
    q = p + 2
    if not is_prime(p):
        return Verdict(False, f"p={p} is not prime", cert)
    if not is_prime(q):
        return Verdict(False, f"q={q} (p+2) is not prime", cert)
    return Verdict(True, f"{p} and {q} are both prime", cert)


def find_twin_prime(start: int, *, max_scan: int = 100_000) -> TwinPrimeCertificate | None:
    """Thin, swappable candidate generator: smallest p >= ``start`` with (p, p+2) prime.

    Bounded by ``max_scan`` so it always terminates; returns ``None`` if none is found in range.
    The verifier does not depend on this -- any source of a pair (a table, a model) would do.
    """
    p = max(start, 2)
    for _ in range(max_scan):
        if is_prime(p) and is_prime(p + 2):
            return TwinPrimeCertificate(p=p)
        p += 1
    return None


def register_verified_scalar(
    gosnn: GlobalOmniScalarNetwork,
    cert: TwinPrimeCertificate,
    *,
    component_name: str = "mathematical_mysteries_twin_prime",
) -> tuple[float, Verdict]:
    """Adjudicate ``cert`` and register a grounded scalar (1.0 confirmed / 0.0 refuted)."""
    verdict = verify_certificate(cert)
    scalar_value = 1.0 if verdict.valid else 0.0
    gosnn.register_scalars(
        component_name=component_name,
        scalars={"omni_mystery_twin_prime_verified": scalar_value},
        group=ScalarGroup.MATHEMATICAL_MYSTERIES,
        metadata={"source": "twin_prime_oracle", **verdict.as_metadata()},
    )
    logger.info("Twin-prime scalar grounded to %.1f (%s)", scalar_value, verdict.reason)
    return scalar_value, verdict


def demonstrate() -> None:
    """End-to-end demonstration: candidate -> oracle -> grounded GOSNN scalar."""
    from omni_mercury_engine.core.global_omni_scalar_network import GlobalOmniScalarNetwork

    gosnn = GlobalOmniScalarNetwork()
    cert = find_twin_prime(11)
    assert cert is not None
    true_verdict = verify_certificate(cert)
    print(
        f"[twin] TRUE  ({cert.p}, {cert.p + 2}): valid={true_verdict.valid} ({true_verdict.reason})"
    )

    false_verdict = verify_certificate(TwinPrimeCertificate(p=7))  # (7, 9) -- 9 is composite
    print(f"[twin] FALSE (7, 9): valid={false_verdict.valid} ({false_verdict.reason})")

    value, _ = register_verified_scalar(gosnn, cert)
    print(f"[twin] grounded scalar = {value}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    demonstrate()
