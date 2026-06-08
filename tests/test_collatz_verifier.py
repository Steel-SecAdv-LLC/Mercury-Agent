# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Collatz verifier: trajectory checking, the inconclusive trichotomy, and grounded scalar."""

from __future__ import annotations

from typing import Any

import pytest

from omni_mercury_engine.core.global_omni_scalar_network import (
    GlobalOmniScalarNetwork,
    ScalarGroup,
    reset_global_network,
)
from omni_mercury_engine.verifiers.collatz import (
    Status,
    compute_trajectory,
    register_verified_scalar,
    verify_trajectory,
)


@pytest.fixture(autouse=True)
def _reset_gosnn() -> Any:
    reset_global_network()
    yield
    reset_global_network()


class TestTrajectoryCheckingHasTeeth:
    def test_correct_trajectory_is_confirmed(self) -> None:
        verdict = verify_trajectory(6, (6, 3, 10, 5, 16, 8, 4, 2, 1))
        assert verdict.status is Status.CONFIRMED
        assert verdict.valid

    def test_tampered_step_is_refuted(self) -> None:
        verdict = verify_trajectory(6, (6, 3, 10, 5, 16, 8, 4, 2, 99))
        assert verdict.status is Status.REFUTED
        assert "bad step" in verdict.reason

    def test_wrong_start_is_refuted(self) -> None:
        verdict = verify_trajectory(6, (8, 4, 2, 1))
        assert verdict.status is Status.REFUTED

    def test_not_ending_at_one_is_refuted(self) -> None:
        verdict = verify_trajectory(6, (6, 3, 10, 5))
        assert verdict.status is Status.REFUTED


class TestDynamicalProcessProducesValidCertificate:
    def test_n27_reaches_one(self) -> None:
        traj = compute_trajectory(27)
        assert traj is not None
        assert traj[0] == 27 and traj[-1] == 1
        assert verify_trajectory(27, traj).valid


class TestScalarIsGroundedInTheVerdict:
    def test_confirmed_registers_one(self) -> None:
        gosnn = GlobalOmniScalarNetwork()
        value, verdict = register_verified_scalar(gosnn, 27)
        assert verdict.status is Status.CONFIRMED
        assert value == 1.0
        assert (
            gosnn.scalar_groups[ScalarGroup.MATHEMATICAL_MYSTERIES][
                "omni_mystery_collatz_reaches_one"
            ]
            == 1.0
        )

    def test_inconclusive_registers_nothing(self) -> None:
        gosnn = GlobalOmniScalarNetwork()
        value, verdict = register_verified_scalar(gosnn, 27, max_steps=5)
        assert verdict.status is Status.INCONCLUSIVE
        assert value is None
        assert (
            "omni_mystery_collatz_reaches_one"
            not in gosnn.scalar_groups[ScalarGroup.MATHEMATICAL_MYSTERIES]
        )
