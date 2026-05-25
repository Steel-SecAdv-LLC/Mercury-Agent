"""Tests for the abstention-first governance scalar contract."""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")  # GlobalOmniScalarNetwork imports numpy at module load.

from omni_mercury_engine.core.global_omni_scalar_network import (
    GlobalOmniScalarNetwork,
    ScalarGroup,
    reset_global_network,
)
from omni_mercury_engine.governance.contract import (
    GovernanceRegistry,
    ScalarStatus,
    available,
    unavailable,
)


@pytest.fixture()
def gosnn() -> GlobalOmniScalarNetwork:
    """Provide a freshly reset GOSNN singleton and tear it down afterwards."""
    reset_global_network()
    network = GlobalOmniScalarNetwork()
    yield network
    reset_global_network()


def test_available_clamps_into_unit_interval() -> None:
    """An available scalar clamps its value into [0, 1]."""
    assert available("omni_sofa_renal", 1.7, family="sofa", reason="x").value == 1.0
    assert available("omni_sofa_renal", -0.3, family="sofa", reason="x").value == 0.0
    assert available("omni_sofa_renal", 0.5, family="sofa", reason="x").value == 0.5


def test_unavailable_carries_no_value() -> None:
    """An abstention has no value and reports the UNAVAILABLE status."""
    scalar = unavailable("omni_ews_news2", family="ews", reason="missing vitals")
    assert scalar.value is None
    assert scalar.status is ScalarStatus.UNAVAILABLE
    assert not scalar.is_available


def test_registry_registers_available_metric_only(gosnn: GlobalOmniScalarNetwork) -> None:
    """An available metric-only scalar is registered and recorded as such."""
    registry = GovernanceRegistry(gosnn)
    entry = registry.register(
        available("omni_sofa_renal", 0.5, family="sofa", reason="SOFA renal = 2/4"),
        group=ScalarGroup.MEDICAL,
        component_name="governance_test",
    )
    assert entry.registered is True
    assert gosnn.get_scalar("omni_sofa_renal") == 0.5


def test_registry_abstention_registers_nothing(gosnn: GlobalOmniScalarNetwork) -> None:
    """An UNAVAILABLE scalar is ledger-only and grounds no value."""
    registry = GovernanceRegistry(gosnn)
    entry = registry.register(
        unavailable("omni_sofa_liver", family="sofa", reason="missing bilirubin"),
        group=ScalarGroup.MEDICAL,
        component_name="governance_test",
    )
    assert entry.registered is False
    assert gosnn.get_scalar("omni_sofa_liver", default=-1.0) == -1.0


def test_registry_rejects_non_metric_only_key(gosnn: GlobalOmniScalarNetwork) -> None:
    """Registering an operational (non-metric-only) key is refused, protecting the gate."""
    registry = GovernanceRegistry(gosnn)
    with pytest.raises(ValueError, match="not metric-only"):
        registry.register(
            available("omnimorality", 0.5, family="bogus", reason="should never register"),
            group=ScalarGroup.ETHICAL,
            component_name="governance_test",
        )


def test_summary_counts_registered_and_abstained(gosnn: GlobalOmniScalarNetwork) -> None:
    """The registry summary distinguishes registered scalars from abstentions."""
    registry = GovernanceRegistry(gosnn)
    registry.register(
        available("omni_sofa_renal", 0.5, family="sofa", reason="x"),
        group=ScalarGroup.MEDICAL,
        component_name="governance_test",
    )
    registry.register(
        unavailable("omni_sofa_liver", family="sofa", reason="missing"),
        group=ScalarGroup.MEDICAL,
        component_name="governance_test",
    )
    summary = registry.summary()
    assert summary["registered"] == 1
    assert summary["abstained"] == 1
