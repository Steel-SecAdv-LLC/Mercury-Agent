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
End-to-end proof that a GOSNN scalar can be grounded in an independent oracle.

These tests are the deliverable for "take one scalar and drive it end to end through a real
verifier": the verdicts come from deterministic arithmetic, the oracle is itself validated
against a second independent method, false certificates are refuted, and the resulting scalar
lands in the previously empty ``MATHEMATICAL_MYSTERIES`` group.
"""

from typing import Any

import pytest

from omni_mercury_engine.core.global_omni_scalar_network import (
    GlobalOmniScalarNetwork,
    ScalarGroup,
    reset_global_network,
)
from omni_mercury_engine.verifiers.goldbach import (
    GoldbachCertificate,
    _is_prime_trial,
    find_partition,
    is_prime,
    register_verified_scalar,
    verify_certificate,
)


@pytest.fixture(autouse=True)
def _reset_gosnn() -> Any:
    reset_global_network()
    yield
    reset_global_network()


class TestOracleIsIndependentlyValid:
    """The oracle must agree with a second, obviously-correct primality method."""

    def test_miller_rabin_matches_trial_division(self) -> None:
        for n in range(0, 5000):
            assert is_prime(n) == _is_prime_trial(n), f"primality disagreement at {n}"

    def test_known_large_primes_and_composites(self) -> None:
        assert is_prime(2_147_483_647)  # Mersenne prime M31
        assert not is_prime(2_147_483_647 - 2)
        assert not is_prime(1_000_000_007 * 1_000_000_009)  # product of two primes


class TestVerifierHasTeeth:
    """A real verifier confirms truths and refutes fabrications alike."""

    def test_true_certificate_is_confirmed(self) -> None:
        verdict = verify_certificate(GoldbachCertificate(n=100, p=3, q=97))
        assert verdict.valid
        assert "both prime" in verdict.reason

    def test_composite_summand_is_refuted(self) -> None:
        # 100 = 9 + 91, but 9 = 3*3 and 91 = 7*13 -- a plausible-looking fabrication.
        verdict = verify_certificate(GoldbachCertificate(n=100, p=9, q=91))
        assert not verdict.valid
        assert "not prime" in verdict.reason

    def test_arithmetic_mismatch_is_refuted(self) -> None:
        verdict = verify_certificate(GoldbachCertificate(n=100, p=3, q=5))
        assert not verdict.valid
        assert "mismatch" in verdict.reason

    def test_odd_or_small_n_is_refuted(self) -> None:
        assert not verify_certificate(GoldbachCertificate(n=7, p=2, q=5)).valid
        assert not verify_certificate(GoldbachCertificate(n=2, p=1, q=1)).valid


class TestProposerOutputSurvivesOracle:
    """Whatever the proposer emits must still pass the independent oracle."""

    def test_every_even_in_range_has_a_verified_partition(self) -> None:
        for n in range(4, 5001, 2):
            cert = find_partition(n)
            assert cert is not None, f"no partition proposed for {n}"
            assert cert.p + cert.q == n
            assert verify_certificate(cert).valid, f"oracle rejected partition for {n}"


class TestScalarIsGroundedInTheVerdict:
    """The registered scalar's value is decided by the oracle, not asserted."""

    def test_confirmed_partition_registers_one(self) -> None:
        gosnn = GlobalOmniScalarNetwork()
        cert = find_partition(100)
        assert cert is not None
        value, verdict = register_verified_scalar(gosnn, cert)
        assert verdict.valid
        assert value == 1.0
        stored = gosnn.scalar_groups[ScalarGroup.MATHEMATICAL_MYSTERIES][
            "omni_mystery_goldbach_verified"
        ]
        assert stored == 1.0

    def test_refuted_partition_registers_zero(self) -> None:
        gosnn = GlobalOmniScalarNetwork()
        value, verdict = register_verified_scalar(gosnn, GoldbachCertificate(n=100, p=9, q=91))
        assert not verdict.valid
        assert value == 0.0
        stored = gosnn.scalar_groups[ScalarGroup.MATHEMATICAL_MYSTERIES][
            "omni_mystery_goldbach_verified"
        ]
        assert stored == 0.0

    def test_mathematical_mysteries_group_was_empty_before(self) -> None:
        gosnn = GlobalOmniScalarNetwork()
        assert gosnn.scalar_groups[ScalarGroup.MATHEMATICAL_MYSTERIES] == {}
        cert = find_partition(50)
        assert cert is not None
        register_verified_scalar(gosnn, cert)
        assert gosnn.scalar_groups[ScalarGroup.MATHEMATICAL_MYSTERIES] != {}
