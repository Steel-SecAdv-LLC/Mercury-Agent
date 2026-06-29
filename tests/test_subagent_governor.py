# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioral tests for the subagent fleet's autonomy governor.

The governor is the safety spine that lets the main agent run subagents in the
masses without unbounded failure modes. These tests pin the real enforcement —
capability ceiling, corrigibility kill-switch, recursion bound, autonomy cap,
and the failure-rate tripwire — fail-closed, not decorative.
"""

from __future__ import annotations

import pytest

from omni_mercury_engine.agentic.subagents.base import SubAgentResult
from omni_mercury_engine.agentic.subagents.governor import (
    AutonomyGovernor,
    CapabilityCeiling,
    GovernorTripped,
)


def _result(status: str, autonomy: float = 0.9) -> SubAgentResult:
    return SubAgentResult(
        subagent_id="sub_test_0001",
        specialty="generalist",
        task_id="task_0001",
        status=status,
        autonomy_ceiling=autonomy,
    )


def test_ceiling_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        CapabilityCeiling(max_replicas=0)
    with pytest.raises(ValueError):
        CapabilityCeiling(max_autonomy=1.5)
    with pytest.raises(ValueError):
        CapabilityCeiling(max_recursion_depth=-1)


def test_authorize_reserves_and_release_frees_capacity() -> None:
    gov = AutonomyGovernor(CapabilityCeiling(max_total_active=4))
    gov.authorize(3, depth=0)
    assert gov.snapshot().active == 3
    # A second reservation that would exceed the total-active ceiling fails closed
    # and reserves nothing.
    with pytest.raises(GovernorTripped, match="total-active ceiling"):
        gov.authorize(2, depth=0)
    assert gov.snapshot().active == 3
    gov.release(3)
    assert gov.snapshot().active == 0


def test_replica_and_recursion_ceilings_fail_closed() -> None:
    gov = AutonomyGovernor(CapabilityCeiling(max_replicas=4, max_recursion_depth=2))
    with pytest.raises(GovernorTripped, match="per-dispatch ceiling"):
        gov.authorize(5, depth=0)
    with pytest.raises(GovernorTripped, match="recursion depth"):
        gov.authorize(1, depth=3)
    with pytest.raises(GovernorTripped, match=">= 1 replica"):
        gov.authorize(0, depth=0)


def test_pause_resume_corrigibility() -> None:
    gov = AutonomyGovernor()
    gov.pause()
    assert gov.is_halted
    with pytest.raises(GovernorTripped, match="paused"):
        gov.authorize(1, depth=0)
    gov.resume()
    assert not gov.is_halted
    gov.authorize(1, depth=0)  # now permitted
    gov.release(1)


def test_trip_is_irreversible_kill_switch() -> None:
    gov = AutonomyGovernor()
    gov.trip("operator stop")
    assert gov.is_tripped
    # resume() does NOT undo a trip — a safety stop stays stopped.
    gov.resume()
    assert gov.is_tripped
    with pytest.raises(GovernorTripped, match="tripped"):
        gov.authorize(1, depth=0)


def test_autonomy_breach_trips_governor() -> None:
    gov = AutonomyGovernor(CapabilityCeiling(max_autonomy=0.95))
    gov.check_autonomy(0.95)  # at the cap: allowed
    with pytest.raises(GovernorTripped, match="exceeds cap"):
        gov.check_autonomy(0.99)
    assert gov.is_tripped  # the breach also trips the wire


def test_failure_rate_tripwire_fires_after_min_observations() -> None:
    ceiling = CapabilityCeiling(max_failure_rate=0.5, tripwire_min_observations=4)
    gov = AutonomyGovernor(ceiling)
    # Three failures + one success = 75% failure over 4 graded results -> trips.
    gov.observe_results(
        [_result("failed"), _result("failed"), _result("failed"), _result("completed")]
    )
    assert gov.is_tripped
    reasons = gov.snapshot().trip_reasons
    assert any("failure-rate tripwire" in r for r in reasons)


def test_blocked_results_are_not_failures_for_the_tripwire() -> None:
    ceiling = CapabilityCeiling(max_failure_rate=0.5, tripwire_min_observations=2)
    gov = AutonomyGovernor(ceiling)
    # Ethical refusals (blocked) are correct, not failures: they must never trip.
    gov.observe_results([_result("blocked") for _ in range(10)])
    assert not gov.is_tripped
    snap = gov.snapshot()
    assert snap.total_blocked == 10
    assert snap.total_failed == 0


def test_observe_accounts_outcomes() -> None:
    gov = AutonomyGovernor(CapabilityCeiling(tripwire_min_observations=100))
    gov.observe_results([_result("completed"), _result("completed"), _result("failed")])
    snap = gov.snapshot()
    assert snap.total_completed == 2
    assert snap.total_failed == 1
