# Copyright (C) 2025 Steel Security Advisors LLC
"""Twin-prime verifier: certificate adjudication and oracle-grounded scalar."""

from __future__ import annotations

from typing import Any

import pytest

from omni_mercury_engine.core.global_omni_scalar_network import (
    GlobalOmniScalarNetwork,
    ScalarGroup,
    reset_global_network,
)
from omni_mercury_engine.verifiers.twin_primes import (
    TwinPrimeCertificate,
    find_twin_prime,
    register_verified_scalar,
    verify_certificate,
)


@pytest.fixture(autouse=True)
def _reset_gosnn() -> Any:
    reset_global_network()
    yield
    reset_global_network()


class TestVerifierHasTeeth:
    def test_genuine_pair_is_confirmed(self) -> None:
        verdict = verify_certificate(TwinPrimeCertificate(p=11))  # (11, 13)
        assert verdict.valid
        assert "both prime" in verdict.reason

    def test_composite_upper_is_refuted(self) -> None:
        verdict = verify_certificate(TwinPrimeCertificate(p=7))  # (7, 9), 9 = 3*3
        assert not verdict.valid
        assert "9" in verdict.reason

    def test_composite_lower_is_refuted(self) -> None:
        verdict = verify_certificate(TwinPrimeCertificate(p=9))  # 9 not prime
        assert not verdict.valid


class TestProposerOutputSurvivesOracle:
    def test_found_pair_passes_oracle(self) -> None:
        cert = find_twin_prime(100)
        assert cert is not None
        assert verify_certificate(cert).valid


class TestScalarIsGroundedInTheVerdict:
    def test_confirmed_registers_one(self) -> None:
        gosnn = GlobalOmniScalarNetwork()
        value, verdict = register_verified_scalar(gosnn, TwinPrimeCertificate(p=11))
        assert verdict.valid
        assert value == 1.0
        assert (
            gosnn.scalar_groups[ScalarGroup.MATHEMATICAL_MYSTERIES][
                "omni_mystery_twin_prime_verified"
            ]
            == 1.0
        )

    def test_refuted_registers_zero(self) -> None:
        gosnn = GlobalOmniScalarNetwork()
        value, verdict = register_verified_scalar(gosnn, TwinPrimeCertificate(p=7))
        assert not verdict.valid
        assert value == 0.0
        assert (
            gosnn.scalar_groups[ScalarGroup.MATHEMATICAL_MYSTERIES][
                "omni_mystery_twin_prime_verified"
            ]
            == 0.0
        )
