# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end: Mercury talks to its subagent fleet and returns a real decision.

Exercises the PR #314 "pantheon" delegation path through the public agent and
engine entry points: routing -> a deep detection subagent running Mercury's own
MultiAgentOrchestrator -> dual ethical gate -> a committed decision. Also covers
mass dispatch and an engine-bound coordinator operation.

The deep-detection path needs torch (the [ml] extra); skipped otherwise, mirror-
ing the CI integration lane.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine._compat import HAS_TORCH

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="fleet deep-detection needs torch ([ml])")


def _data_with_outliers(seed: int = 0):
    rng = np.random.default_rng(seed)
    train = rng.normal(0, 1, (200, 6))
    X = rng.normal(0, 1, (40, 6))
    X[:5] += 8.0  # 5 unmistakable outliers
    return train, X


def _detection_task(X, train):
    from omni_mercury_engine.agentic.mercury_a_agent import DomainType
    from omni_mercury_engine.agentic.subagents.base import SubAgentTask

    return SubAgentTask(
        description="detect anomalies in the network stream",
        domain=DomainType.SECURITY,
        payload={"data": X, "train": train},
    )


class TestMercuryDelegatesToFleet:
    def test_delegate_routes_to_deep_detection_and_commits(self) -> None:
        from omni_mercury_engine.agentic.mercury_a_agent import MercuryAgent

        train, X = _data_with_outliers(0)
        agent = MercuryAgent(name="Mercury", enable_calibration=False)
        result = agent.delegate(_detection_task(X, train))

        assert agent.fleet is not None  # lazily enabled
        assert result.specialty == "Zeus_VIII"  # deep detection coordinator
        assert result.status == "completed"
        assert result.ok is True
        assert result.metadata.get("committed") is True  # dual ethical gate authorized
        assert result.anchor == "OMNI_INDIVISIBLE"
        assert 0.0 < result.autonomy_ceiling <= 0.95
        # The decision came back: a real episode over the 40-sample batch with
        # the injected outliers surfaced.
        out = result.output
        assert out["n_samples"] == 40
        assert out["n_anomalies"] >= 5

    def test_delegate_masses_aggregates_committed_replicas(self) -> None:
        from omni_mercury_engine.agentic.mercury_a_agent import MercuryAgent

        train, X = _data_with_outliers(1)
        agent = MercuryAgent(name="Mercury", enable_calibration=False)
        fr = agent.delegate_masses(_detection_task(X, train), replicas=4)
        assert len(fr.results) == 4
        assert fr.committed is True
        assert all(r.status == "completed" for r in fr.results)


class TestEngineBoundFleetAndCoordinatorOp:
    def test_engine_bound_dispatch_detects(self) -> None:
        from omni_mercury_engine.engine import OmniMercuryEngine

        train, X = _data_with_outliers(2)
        engine = OmniMercuryEngine(mode="fusion", device="cpu")
        fleet = engine.enable_subagent_fleet(seed=0)
        assert fleet is not None
        result = fleet.dispatch(_detection_task(X, train))
        assert result.status == "completed"
        assert result.output["n_anomalies"] >= 5

    def test_coordinator_op_returns_real_metrics(self) -> None:
        from omni_mercury_engine.agentic.mercury_a_agent import DomainType, MercuryAgent
        from omni_mercury_engine.agentic.subagents.base import SubAgentTask

        agent = MercuryAgent(name="Mercury", enable_calibration=False)
        rng = np.random.default_rng(3)
        y_true = np.array([0] * 30 + [1] * 10)
        y_score = np.concatenate([rng.uniform(0, 0.5, 30), rng.uniform(0.5, 1.0, 10)])
        task = SubAgentTask(
            description="emit telemetry metrics for the run",
            domain=DomainType.INFRASTRUCTURE,
            payload={"labels": y_true, "scores": y_score},
        )
        result = agent.delegate(task, specialty="Helios_XVII")
        assert result.status == "completed"
        assert result.output["operation"] == "metrics.AnomalyMetrics.compute_all"
        assert 0.0 <= result.output["auroc"] <= 1.0
