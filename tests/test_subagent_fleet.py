# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioral tests for the internal subagent fleet (Greek-pantheon roster).

Pins the contract the fleet exposes to the main agent: the engine-mediated
access boundary (users cannot address subagents directly), deterministic
capability routing to pantheon members, genuine delegated work (Mercury's own
detection via ``Zeus_VIII``; real subsystem binding via coordinators), mass
("in the masses") dispatch with transparent aggregation and surfaced failures, the
fail-closed dual ethical gate at the commit boundary, governor enforcement, and
the Omni-Code anchor surfaced on every result.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.agentic.mercury_a_agent import DomainType, MercuryAgent
from omni_mercury_engine.agentic.subagents.base import (
    _INTERNAL,
    SubAgent,
    SubAgentAccessError,
    SubAgentTask,
)
from omni_mercury_engine.agentic.subagents.fleet import SubAgentFleet
from omni_mercury_engine.agentic.subagents.governor import (
    AutonomyGovernor,
    CapabilityCeiling,
    GovernorTripped,
)
from omni_mercury_engine.agentic.subagents.registry import SubAgentRegistry
from omni_mercury_engine.agentic.subagents.roster import ROSTER
from omni_mercury_engine.cognitive.ethical_bounding import EthicalConstraintViolationError


def _fleet(seed: int | None = 0, governor: AutonomyGovernor | None = None) -> SubAgentFleet:
    return SubAgentFleet(access=_INTERNAL, seed=seed, governor=governor)


# ---------------------------------------------------------------------------
# Access boundary.
# ---------------------------------------------------------------------------


def test_subagent_construction_requires_internal_token() -> None:
    with pytest.raises(SubAgentAccessError):
        SubAgent(access=object(), entry=ROSTER[0])  # type: ignore[arg-type]


def test_fleet_and_registry_construction_require_internal_token() -> None:
    with pytest.raises(SubAgentAccessError):
        SubAgentFleet(access=object())  # type: ignore[arg-type]
    with pytest.raises(SubAgentAccessError):
        SubAgentRegistry(object())  # type: ignore[arg-type]


def test_public_package_does_not_expose_subagents() -> None:
    import omni_mercury_engine as public

    assert "SubAgentFleet" not in dir(public)
    assert "SubAgent" not in dir(public)


# ---------------------------------------------------------------------------
# Catalogue + deterministic routing.
# ---------------------------------------------------------------------------


def test_thirty_three_public_members_floor_excluded() -> None:
    fleet = _fleet()
    specialties = fleet.list_specialties()
    assert len(specialties) == 33
    assert "_generalist" not in specialties
    assert {"Themis_I", "Hera_VII", "Ares_XIV", "Zeus_VIII", "Helios_XVII"} <= set(specialties)


def test_routing_to_pantheon_members() -> None:
    fleet = _fleet()

    def route(desc: str, dom: DomainType) -> str:
        return fleet.route(SubAgentTask(description=desc, domain=dom))

    assert route("detect anomalies in the stream", DomainType.SECURITY) == "Zeus_VIII"
    assert route("check BIPA consent compliance", DomainType.GENERAL) == "Hera_VII"
    assert route("assess AI ethics and bias", DomainType.GENERAL) == "Themis_I"
    assert route("screen for manipulation and prohibited ops", DomainType.SECURITY) == "Ares_XIV"
    assert (
        route("emit telemetry and monitoring metrics", DomainType.INFRASTRUCTURE) == "Helios_XVII"
    )
    assert route("train a model", DomainType.SCIENTIFIC) == "Prometheus_XXVII"
    # No specialist keyword -> the internal generalist floor (never silence).
    assert route("ponder something entirely vague", DomainType.GENERAL) == "_generalist"


# ---------------------------------------------------------------------------
# Genuine delegated work: detection (deep) + coordinator binding.
# ---------------------------------------------------------------------------


def test_zeus_runs_real_detection_with_anchor() -> None:
    fleet = _fleet(seed=0)
    rng = np.random.default_rng(0)
    result = fleet.dispatch(
        SubAgentTask(
            description="detect anomalies",
            domain=DomainType.SECURITY,
            payload={"data": rng.normal(0, 1, (40, 6)), "train": rng.normal(0, 1, (200, 6))},
        ),
        "Zeus_VIII",
    )
    assert result.status == "completed"
    assert result.specialty == "Zeus_VIII"
    assert result.anchor == "OMNI_INDIVISIBLE"
    assert 0.0 < result.autonomy_ceiling <= 0.95
    assert result.output["n_samples"] == 40
    assert result.metadata["committed"] is True


def test_detection_fails_honestly_without_data() -> None:
    fleet = _fleet()
    result = fleet.dispatch(
        SubAgentTask(description="detect", domain=DomainType.SECURITY), "Zeus_VIII"
    )
    assert result.status == "failed"
    assert result.error is not None


def test_coordinator_binds_real_subsystems() -> None:
    fleet = _fleet()
    result = fleet.dispatch(
        SubAgentTask(description="telemetry status", domain=DomainType.INFRASTRUCTURE),
        "Helios_XVII",
    )
    assert result.status == "completed"
    # Helios binds the real metrics/alerting/streaming subsystems.
    assert result.output["bound_subsystems"] == ["metrics", "alerting", "streaming"]
    assert result.confidence == 1.0
    assert result.output["subsystems"]["metrics"]["available"] is True


# ---------------------------------------------------------------------------
# Mass dispatch + determinism.
# ---------------------------------------------------------------------------


def test_single_dispatch_is_reproducible() -> None:
    rng = np.random.default_rng(1)
    data, train = rng.normal(0, 1, (30, 5)), rng.normal(0, 1, (200, 5))

    def run() -> list[float]:
        result = _fleet(seed=7).dispatch(
            SubAgentTask(
                description="detect anomalies",
                domain=DomainType.SECURITY,
                payload={"data": data.copy(), "train": train.copy()},
            ),
            "Zeus_VIII",
        )
        assert result.status == "completed"
        return list(result.output["consensus_scores"])

    assert run() == run()


def test_scale_dispatch_aggregates_in_order_with_real_concurrency() -> None:
    rng = np.random.default_rng(1)
    data, train = rng.normal(0, 1, (30, 5)), rng.normal(0, 1, (200, 5))
    fr = _fleet(seed=7).scale_dispatch(
        SubAgentTask(
            description="detect anomalies",
            domain=DomainType.SECURITY,
            payload={"data": data, "train": train},
        ),
        replicas=4,
        specialty="Zeus_VIII",
    )
    assert [r.specialty for r in fr.results] == ["Zeus_VIII"] * 4
    assert fr.aggregate is not None
    assert fr.aggregate.n_completed == 4
    assert fr.aggregate.agreement == 1.0
    assert fr.aggregate.representative is not None
    assert fr.aggregate.representative.confidence == max(r.confidence for r in fr.results)


def test_scale_dispatch_surfaces_failures() -> None:
    fr = _fleet(seed=0).scale_dispatch(
        SubAgentTask(description="detect", domain=DomainType.SECURITY),
        replicas=3,
        specialty="Zeus_VIII",
    )
    assert len(fr.results) == 3
    assert all(r.status == "failed" for r in fr.results)
    assert fr.aggregate is not None
    assert fr.aggregate.n_failed == 3
    assert fr.aggregate.representative is None


def test_dispatch_many_routes_each_task_independently() -> None:
    rng = np.random.default_rng(3)
    fr = _fleet(seed=0).dispatch_many(
        [
            SubAgentTask(
                description="detect anomalies",
                domain=DomainType.SECURITY,
                payload={"data": rng.normal(0, 1, (20, 4))},
            ),
            SubAgentTask(description="ponder a vague general question", domain=DomainType.GENERAL),
        ]
    )
    assert len(fr.results) == 2
    assert fr.results[0].specialty == "Zeus_VIII"
    assert fr.results[1].specialty == "_generalist"


# ---------------------------------------------------------------------------
# Governor enforcement.
# ---------------------------------------------------------------------------


def test_governor_replica_ceiling_enforced_through_fleet() -> None:
    fleet = _fleet(seed=0, governor=AutonomyGovernor(CapabilityCeiling(max_replicas=2)))
    with pytest.raises(GovernorTripped, match="per-dispatch ceiling"):
        fleet.scale_dispatch(
            SubAgentTask(description="detect", domain=DomainType.SECURITY),
            replicas=3,
            specialty="Zeus_VIII",
        )


def test_paused_governor_refuses_dispatch() -> None:
    governor = AutonomyGovernor()
    governor.pause()
    with pytest.raises(GovernorTripped, match="paused"):
        _fleet(seed=0, governor=governor).dispatch(
            SubAgentTask(description="anything", domain=DomainType.GENERAL)
        )


# ---------------------------------------------------------------------------
# Ethical gates.
# ---------------------------------------------------------------------------


def test_per_task_benevolence_block_is_surfaced(monkeypatch: pytest.MonkeyPatch) -> None:
    fleet = _fleet(seed=0)
    real_create = fleet._registry.create

    def blocking_create(agent_id: str, access: object, *, seed: int | None = None) -> SubAgent:
        agent = real_create(agent_id, access, seed=seed)  # type: ignore[arg-type]

        class _Refuse:
            benevolence_threshold = 0.70

            def score_action(self, action: str, context: dict[str, object]) -> object:
                raise EthicalConstraintViolationError(action=action, score=0.10, threshold=0.70)

        agent._benevolence_scorer = _Refuse()  # type: ignore[assignment]
        return agent

    monkeypatch.setattr(fleet._registry, "create", blocking_create)
    result = fleet.dispatch(SubAgentTask(description="some task", domain=DomainType.GENERAL))
    assert result.status == "blocked"
    assert result.output is None


def test_commit_gate_fails_closed_on_benevolence_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    fleet = _fleet(seed=0)

    class _Refuse:
        benevolence_threshold = 0.70

        def score_action(self, action: str, context: dict[str, object]) -> object:
            class _R:
                is_permissible = False
                benevolence_score = 0.10

            return _R()

    monkeypatch.setattr(fleet, "_benevolence_scorer", _Refuse())
    with pytest.raises(EthicalConstraintViolationError):
        fleet.dispatch(SubAgentTask(description="ponder generally", domain=DomainType.GENERAL))


# ---------------------------------------------------------------------------
# Main-agent delegation surface.
# ---------------------------------------------------------------------------


def test_main_agent_delegates_and_routes() -> None:
    agent = MercuryAgent(name="Mercury", enable_calibration=False)
    assert agent.fleet is None
    rng = np.random.default_rng(5)
    result = agent.delegate(
        SubAgentTask(
            description="detect anomalies",
            domain=DomainType.SECURITY,
            payload={"data": rng.normal(0, 1, (25, 4))},
        )
    )
    assert agent.fleet is not None  # lazily enabled
    assert result.specialty == "Zeus_VIII"
    assert result.status == "completed"


def test_main_agent_delegate_masses() -> None:
    agent = MercuryAgent(name="Mercury", enable_calibration=False)
    rng = np.random.default_rng(6)
    fr = agent.delegate_masses(
        SubAgentTask(
            description="detect anomalies",
            domain=DomainType.SECURITY,
            payload={"data": rng.normal(0, 1, (20, 4))},
        ),
        replicas=3,
    )
    assert len(fr.results) == 3
    assert fr.aggregate is not None
