"""Tests for the three-state (GROUNDED/UNAVAILABLE/UNDECIDABLE) governance contract."""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")  # GlobalOmniScalarNetwork imports numpy at module load.

from omni_mercury_engine.core.global_omni_scalar_network import (
    GlobalOmniScalarNetwork,
    ScalarGroup,
    reset_global_network,
)
from omni_mercury_engine.governance.contract import (
    GOVERNANCE_FAMILY_VET,
    GovernanceRegistry,
    SignalClass,
    ThreeState,
    grounded,
    unavailable,
    undecidable,
)


@pytest.fixture()
def gosnn():
    """Provide a freshly reset GOSNN singleton and tear it down afterwards."""
    reset_global_network()
    network = GlobalOmniScalarNetwork()
    yield network
    reset_global_network()


def test_three_states_match_cross_repo_wire_values() -> None:
    """The governance states use the exact cross-repo wire vocabulary (PR #244 ThreeState)."""
    assert ThreeState.GROUNDED.value == "grounded"
    assert ThreeState.UNAVAILABLE.value == "unavailable"
    assert ThreeState.UNDECIDABLE.value == "undecidable"


def test_grounded_clamps_into_unit_interval() -> None:
    """A GROUNDED scalar clamps its value into [0, 1] and registers."""
    assert grounded("omni_sofa_renal", 1.7, family="sofa", reason="x").value == 1.0
    assert grounded("omni_sofa_renal", -0.3, family="sofa", reason="x").value == 0.0
    s = grounded("omni_sofa_renal", 0.5, family="sofa", reason="x")
    assert s.value == 0.5 and s.is_grounded


def test_unavailable_carries_missing_inputs_no_value() -> None:
    """An UNAVAILABLE abstention has no value but records the absent signal(s)."""
    s = unavailable(
        "omni_ews_news2", family="ews", reason="missing vitals", missing_inputs=("temperature_c",)
    )
    assert s.value is None
    assert s.state is ThreeState.UNAVAILABLE
    assert s.missing_inputs == ("temperature_c",)
    assert not s.is_grounded


def test_undecidable_carries_no_value_and_never_grounds() -> None:
    """An UNDECIDABLE scalar (a dropped family's would-be scalar) has no value."""
    s = undecidable("omni_owasp_llm_mitigation", family="owasp_llm", reason="no runtime signal")
    assert s.value is None
    assert s.state is ThreeState.UNDECIDABLE
    assert not s.is_grounded


def test_registry_registers_grounded_metric_only(gosnn: GlobalOmniScalarNetwork) -> None:
    """A GROUNDED metric-only scalar is registered and recorded as such."""
    registry = GovernanceRegistry(gosnn)
    entry = registry.register(
        grounded("omni_sofa_renal", 0.5, family="sofa", reason="SOFA renal = 2/4"),
        group=ScalarGroup.MEDICAL,
        component_name="governance_test",
    )
    assert entry.registered is True
    assert entry.state == "grounded"
    assert gosnn.get_scalar("omni_sofa_renal") == 0.5


def test_registry_unavailable_registers_nothing(gosnn: GlobalOmniScalarNetwork) -> None:
    """An UNAVAILABLE scalar is ledger-only (kept abstention) and grounds no value."""
    registry = GovernanceRegistry(gosnn)
    entry = registry.register(
        unavailable(
            "omni_sofa_liver", family="sofa", reason="missing", missing_inputs=("bilirubin_mg_dl",)
        ),
        group=ScalarGroup.MEDICAL,
        component_name="governance_test",
    )
    assert entry.registered is False
    assert entry.state == "unavailable"
    assert entry.missing_inputs == ("bilirubin_mg_dl",)
    assert gosnn.get_scalar("omni_sofa_liver", default=-1.0) == -1.0


def test_registry_undecidable_registers_nothing(gosnn: GlobalOmniScalarNetwork) -> None:
    """An UNDECIDABLE scalar is ledger-only and grounds no value (dropped, ever)."""
    registry = GovernanceRegistry(gosnn)
    entry = registry.register(
        undecidable("omni_sofa_renal", family="sofa", reason="hypothetical"),
        group=ScalarGroup.MEDICAL,
        component_name="governance_test",
    )
    assert entry.registered is False
    assert entry.state == "undecidable"
    assert gosnn.get_scalar("omni_sofa_renal", default=-1.0) == -1.0


def test_registry_rejects_non_metric_only_key(gosnn: GlobalOmniScalarNetwork) -> None:
    """Registering an operational (non-metric-only) key is refused, protecting the gate."""
    registry = GovernanceRegistry(gosnn)
    with pytest.raises(ValueError, match="not metric-only"):
        registry.register(
            grounded("omnimorality", 0.5, family="bogus", reason="should never register"),
            group=ScalarGroup.ETHICAL,
            component_name="governance_test",
        )


def test_registry_refuses_to_ground_an_undecidable_family(gosnn: GlobalOmniScalarNetwork) -> None:
    """Even a metric-only GROUNDED scalar is refused if its family vets UNDECIDABLE."""
    registry = GovernanceRegistry(gosnn)
    # omni_owasp_llm_ has no prefix anymore, so use a metric-only key but an UNDECIDABLE family.
    with pytest.raises(ValueError, match="UNDECIDABLE"):
        registry.register(
            grounded("omni_sofa_renal", 0.5, family="owasp_llm", reason="contract violation"),
            group=ScalarGroup.SECURITY,
            component_name="governance_test",
        )


def test_summary_counts_by_state(gosnn: GlobalOmniScalarNetwork) -> None:
    """The registry summary distinguishes registered scalars from abstentions by state."""
    registry = GovernanceRegistry(gosnn)
    registry.register(
        grounded("omni_sofa_renal", 0.5, family="sofa", reason="x"),
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
    assert summary["by_state"] == {"grounded": 1, "unavailable": 1}


def test_vet_table_is_complete_and_self_consistent() -> None:
    """Every family is vetted exactly once with a non-empty rationale and matching key."""
    assert GOVERNANCE_FAMILY_VET, "vet table must not be empty"
    for key, vet in GOVERNANCE_FAMILY_VET.items():
        assert vet.family == key
        assert isinstance(vet.classification, SignalClass)
        assert vet.standard and vet.rationale and vet.runtime_signal
    # Dropped families must claim no runtime signal; kept families must cite one.
    for vet in GOVERNANCE_FAMILY_VET.values():
        if vet.classification is SignalClass.UNAVAILABLE_CAPABLE:
            assert vet.runtime_signal != "none", vet.family
        if vet.classification is SignalClass.UNDECIDABLE:
            assert "none" in vet.runtime_signal or "no distinct" in vet.runtime_signal, vet.family
