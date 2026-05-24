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
Oracle-validated verifier for a MATHEMATICAL_MYSTERIES scalar (Goldbach's conjecture).

This module exists to answer one question: can a GOSNN scalar carry a truth value that is
established by an independent oracle rather than asserted by a model?

Goldbach's conjecture ("every even integer > 2 is a sum of two primes") is unproven in
general, so it is a genuine entry for the ``MATHEMATICAL_MYSTERIES`` group.  Any *instance*,
however, is mechanically decidable: a claimed partition ``n = p + q`` is a finite certificate
that an independent oracle either confirms or refutes by checking primality of ``p`` and ``q``
and the arithmetic ``p + q == n``.

The primality oracle below is deterministic Miller-Rabin with a fixed witness set, which is
exact for every ``n < 3.317 * 10**24`` -- far beyond any value this module handles.  It is
plain auditable arithmetic: no model, no learned weights, no network call.  ``is_prime`` is
cross-checked against ``_is_prime_trial`` (naive trial division) in the test-suite, so the
oracle is itself validated by a second independent method.

The scalar registered into the GOSNN is *grounded*: its value is decided by the verdict
(1.0 when the oracle confirms the partition, 0.0 when it refutes it), not chosen by hand.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from omni_mercury_engine.core.global_omni_scalar_network import ScalarGroup

if TYPE_CHECKING:
    from omni_mercury_engine.core.global_omni_scalar_network import GlobalOmniScalarNetwork

logger = logging.getLogger(__name__)

# Deterministic Miller-Rabin witnesses: exact primality test for all n < 3.317e24.
_MR_WITNESSES: tuple[int, ...] = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(n: int) -> bool:
    """Deterministic primality test (Miller-Rabin, exact for n < 3.317e24).

    Pure integer arithmetic -- the auditable oracle that adjudicates certificates.
    """
    if n < 2:
        return False
    for w in _MR_WITNESSES:
        if n % w == 0:
            return n == w
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_WITNESSES:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _is_prime_trial(n: int) -> bool:
    """Naive trial-division primality test.

    Obviously correct but slow; used only to independently validate :func:`is_prime`.
    """
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


@dataclass(frozen=True)
class GoldbachCertificate:
    """A claimed Goldbach partition ``n = p + q`` awaiting adjudication."""

    n: int
    p: int
    q: int


@dataclass(frozen=True)
class Verdict:
    """Result of adjudicating a certificate against the oracle."""

    valid: bool
    reason: str
    certificate: GoldbachCertificate
    checker: str = "deterministic_miller_rabin"

    def as_metadata(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "checker": self.checker,
            "n": self.certificate.n,
            "p": self.certificate.p,
            "q": self.certificate.q,
        }


def verify_certificate(cert: GoldbachCertificate) -> Verdict:
    """Adjudicate a Goldbach certificate with the independent oracle.

    Returns a ``valid=True`` verdict only when ``n`` is an even integer > 2, both summands
    are prime, and ``p + q == n``.  Every other case is rejected with a reason -- this is what
    gives the verifier teeth: a fabricated partition (e.g. ``100 = 9 + 91``) is refuted rather
    than rubber-stamped.
    """
    n, p, q = cert.n, cert.p, cert.q
    if n <= 2 or n % 2 != 0:
        return Verdict(False, f"n={n} is not an even integer > 2", cert)
    if p + q != n:
        return Verdict(False, f"arithmetic mismatch: {p} + {q} = {p + q} != {n}", cert)
    if not is_prime(p):
        return Verdict(False, f"summand p={p} is not prime", cert)
    if not is_prime(q):
        return Verdict(False, f"summand q={q} is not prime", cert)
    return Verdict(True, f"{n} = {p} + {q}, both prime", cert)


def find_partition(n: int) -> GoldbachCertificate | None:
    """Search for a genuine Goldbach partition of even ``n`` (the proposer).

    Returns the certificate with the smallest prime summand, or ``None`` if ``n`` is out of
    range.  The proposer is deliberately separate from the oracle: whatever it emits still has
    to survive :func:`verify_certificate`.
    """
    if n <= 2 or n % 2 != 0:
        return None
    p = 2
    while p <= n // 2:
        if is_prime(p) and is_prime(n - p):
            return GoldbachCertificate(n=n, p=p, q=n - p)
        p += 1
    return None  # unreachable for any n Goldbach holds for; absence would itself be news


def register_verified_scalar(
    gosnn: GlobalOmniScalarNetwork,
    cert: GoldbachCertificate,
    *,
    component_name: str = "mathematical_mysteries_goldbach",
) -> tuple[float, Verdict]:
    """Adjudicate ``cert`` and register a *grounded* scalar into the GOSNN.

    The scalar value is decided by the oracle: 1.0 when the partition is confirmed, 0.0 when
    it is refuted.  Registered under :attr:`ScalarGroup.MATHEMATICAL_MYSTERIES` so a previously
    empty category now carries a value that is checkable independently of any model.

    Returns ``(scalar_value, verdict)``.
    """
    verdict = verify_certificate(cert)
    scalar_value = 1.0 if verdict.valid else 0.0
    gosnn.register_scalars(
        component_name=component_name,
        scalars={"omni_mystery_goldbach_verified": scalar_value},
        group=ScalarGroup.MATHEMATICAL_MYSTERIES,
        metadata={"source": "goldbach_oracle", **verdict.as_metadata()},
    )
    logger.info("Goldbach scalar grounded to %.1f (%s)", scalar_value, verdict.reason)
    return scalar_value, verdict


def demonstrate(batch_limit: int = 10_000) -> None:
    """End-to-end demonstration: proposer -> oracle -> grounded GOSNN scalar."""
    from omni_mercury_engine.core.global_omni_scalar_network import GlobalOmniScalarNetwork

    print("=" * 72)
    print("MATHEMATICAL_MYSTERIES scalar -- Goldbach, oracle-validated end to end")
    print("=" * 72)

    gosnn = GlobalOmniScalarNetwork()

    # 1. A genuine partition, proposed then independently adjudicated.
    true_cert = find_partition(100)
    assert true_cert is not None
    true_verdict = verify_certificate(true_cert)
    print(f"\n[1] TRUE  certificate {true_cert.n} = {true_cert.p} + {true_cert.q}")
    print(f"    oracle verdict: valid={true_verdict.valid}  ({true_verdict.reason})")

    # 2. A fabricated partition (what a hallucinating model might emit) -- must be refuted.
    false_cert = GoldbachCertificate(n=100, p=9, q=91)  # 9 = 3*3, 91 = 7*13
    false_verdict = verify_certificate(false_cert)
    print(f"\n[2] FALSE certificate {false_cert.n} = {false_cert.p} + {false_cert.q}")
    print(f"    oracle verdict: valid={false_verdict.valid}  ({false_verdict.reason})")

    # 3. Ground the scalar from the oracle's verdict and read it back from the network.
    value, _ = register_verified_scalar(gosnn, true_cert)
    stored = gosnn.scalar_groups[ScalarGroup.MATHEMATICAL_MYSTERIES][
        "omni_mystery_goldbach_verified"
    ]
    print(f"\n[3] registered grounded scalar = {value}; read back from GOSNN = {stored}")

    # 4. The check generalises: every even number in range carries a verifiable certificate.
    checked = 0
    for n in range(4, batch_limit + 1, 2):
        cert = find_partition(n)
        assert cert is not None and verify_certificate(cert).valid
        checked += 1
    print(
        f"\n[4] oracle-confirmed Goldbach partitions for all {checked} even n in [4, {batch_limit}]"
    )
    print("\nVerdict source: deterministic arithmetic, not the model. Pattern is real.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    demonstrate()
