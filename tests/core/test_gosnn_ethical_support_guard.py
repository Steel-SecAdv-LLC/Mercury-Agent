# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The ethical band stays inside the σ_Immutable network's trained support.

Measured before this guard: with all 24 ethical anchors set to 10.0 the
network scored 1.0 and the deterministic floor reported no violation, a
fail-open; a Pan_XXV subagent op registered its payload into the process-wide
singleton's ETHICAL group, so ``{"temperature": 0.2}`` became an anchor below
the floor and refused every later detection; and the neurosymbolic hub's raw
counters entered the σ vector.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from omni_mercury_engine.cognitive.ethical_bounding import (
    EthicalConstraintViolationError,
)
from omni_mercury_engine.core.centralized_constants import ETHICAL
from omni_mercury_engine.core.global_omni_scalar_network import (
    GlobalOmniScalarNetwork,
    ScalarGroup,
    reset_global_network,
)
from omni_mercury_engine.security.sigma_immutable_gate import (
    SIGMA_IMMUTABLE_PERMISSIBLE_HIGH,
    get_sigma_immutable_gate,
)

HIGH = ETHICAL.SIGMA_TRAINED_SUPPORT_HIGH


@pytest.fixture
def gosnn() -> Any:
    reset_global_network()
    yield GlobalOmniScalarNetwork()
    reset_global_network()


def test_the_gate_and_the_registry_share_one_support_bound() -> None:
    assert HIGH == SIGMA_IMMUTABLE_PERMISSIBLE_HIGH == 2.0


@pytest.mark.parametrize("bad", [HIGH + 1e-9, 10.0, -1e-9, math.nan, math.inf])
def test_ethical_values_outside_the_support_are_refused_atomically(gosnn: Any, bad: float) -> None:
    before = dict(gosnn.scalar_groups[ScalarGroup.ETHICAL])
    with pytest.raises(ValueError, match="trained support"):
        gosnn.register_scalars("probe", {"omnimorality": 1.0, "omniempathy": bad})
    assert gosnn.scalar_groups[ScalarGroup.ETHICAL] == before  # the good value was not written
    assert "probe" not in gosnn.registered_scalars


@pytest.mark.parametrize("edge", [0.0, HIGH])
def test_the_support_bounds_themselves_are_accepted(gosnn: Any, edge: float) -> None:
    gosnn.register_scalars("probe", {"omnihope": edge})
    assert gosnn.scalar_groups[ScalarGroup.ETHICAL]["omnihope"] == edge


def test_metric_only_and_non_ethical_registrations_are_not_range_checked(gosnn: Any) -> None:
    gosnn.register_scalars("probe", {"omni_diag_counter": 1e9})
    gosnn.register_scalars("probe2", {"omnithreat_detection": 4.0}, group=ScalarGroup.SECURITY)
    assert gosnn.scalar_groups[ScalarGroup.ETHICAL]["omni_diag_counter"] == 1e9


def test_a_legacy_name_writes_its_canonical_scalar_without_adding_a_dimension(
    gosnn: Any,
) -> None:
    width = len(gosnn.scalar_groups[ScalarGroup.ETHICAL])
    gosnn.register_scalars("probe", {"benevolence": 0.995})
    ethical = gosnn.scalar_groups[ScalarGroup.ETHICAL]
    assert "benevolence" not in ethical and ethical["omnibenevolence"] == 0.995
    assert len(ethical) == width


def test_inflated_anchors_violate_the_floor_ceiling(gosnn: Any) -> None:
    gate = get_sigma_immutable_gate()
    anchors = dict.fromkeys(gosnn.critical_ethical_anchors(), 10.0)
    assert len(gate.critical_ethical_floor_violations(anchors)) == len(anchors)
    assert gate.critical_ethical_floor_violations(dict.fromkeys(anchors, HIGH)) == []
    with pytest.raises(EthicalConstraintViolationError) as info:
        gate.enforce_ethical_floor("probe", anchors)
    assert info.value.check == "sigma_immutable_ethical_ceiling"
    assert info.value.score == pytest.approx(HIGH - 10.0)


def test_a_collapsed_anchor_still_reports_the_floor_even_beside_an_inflated_one(
    gosnn: Any,
) -> None:
    gate = get_sigma_immutable_gate()
    with pytest.raises(EthicalConstraintViolationError) as info:
        gate.enforce_ethical_floor("probe", {"omnimorality": 0.1, "omniempathy": 9.0})
    assert info.value.check == "sigma_immutable_ethical_floor"
    assert info.value.score == pytest.approx(0.1)


def test_pan_xxv_scores_its_payload_without_touching_the_singleton(gosnn: Any) -> None:
    from omni_mercury_engine.agentic.subagents.operations import _op_pan

    gate = get_sigma_immutable_gate()
    before = {g: dict(v) for g, v in gosnn.scalar_groups.items()}
    plain = gosnn.compute_global_intelligence_score()
    task: Any = SimpleNamespace(payload={"scalars": {"temperature": 0.2}}, task_id="t1")
    result = _op_pan(SimpleNamespace(), task)  # type: ignore[arg-type]
    assert result is not None
    output = result[0]
    assert output["n_scored"] == 1
    assert output["global_intelligence_score"] != plain  # the payload was scored
    assert {g: dict(v) for g, v in gosnn.scalar_groups.items()} == before
    assert gate.critical_ethical_floor_violations(gosnn.critical_ethical_anchors()) == []


def test_hub_counters_use_the_metric_only_channel() -> None:
    from omni_mercury_engine.core.neurosymbolic_hub import NeuroSymbolicHub

    scalars = NeuroSymbolicHub().get_gosnn_scalars()
    counters = [n for n in scalars if "count" in n or "latency" in n]
    assert counters and all(GlobalOmniScalarNetwork._is_metric_only_scalar(n) for n in counters)
    assert all(n.startswith("omni") for n in scalars)
