"""Guard: governance scalars never perturb the σ_Immutable operational vector.

This is the durable safety rail for the whole governance upgrade. It registers *every*
governance family with full inputs (so each computes and registers) and asserts the
operational vector the trained gate consumes is byte-identical before and after, stays at
127 entries, and remains well under the 175 band cap -- and that every registered key is
metric-only.

It requires only core numpy, never the heavy ``[ml]`` extra: with torch absent SOFA simply
abstains UNAVAILABLE and the other families still register, so the guard holds in a thin
environment too (asserted by the CI lanes that run this without ``[ml]``).
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")  # core dep; NOT the heavy [ml] extra.

from omni_mercury_engine.core.global_omni_scalar_network import (
    GlobalOmniScalarNetwork,
    ScalarGroup,
    reset_global_network,
)
from omni_mercury_engine.governance import ai_safety, clinical, eu_ai_act, medical_device
from omni_mercury_engine.governance.contract import GovernanceRegistry

OPERATIONAL_COUNT = 127
BAND_CAP = 175


@pytest.fixture()
def gosnn() -> GlobalOmniScalarNetwork:
    """Provide a freshly reset GOSNN singleton and tear it down afterwards."""
    reset_global_network()
    network = GlobalOmniScalarNetwork()
    yield network
    reset_global_network()


def _register_every_family(registry: GovernanceRegistry) -> None:
    """Register all governance families with full inputs (SOFA abstains if torch absent)."""
    registry.register_all(
        clinical.sofa_scalars(
            {
                "pao2_fio2_ratio": 250,
                "platelets_k_ul": 30,
                "bilirubin_mg_dl": 8.0,
                "mean_arterial_pressure": 65,
                "gcs_score": 13,
                "creatinine_mg_dl": 2.5,
            }
        ),
        group=ScalarGroup.MEDICAL,
        component_name="governance_clinical_sofa",
    )
    registry.register(
        clinical.news2_scalar(
            {
                "respiratory_rate_bpm": 26,
                "spo2_pct": 90,
                "on_supplemental_o2": True,
                "systolic_bp_mmhg": 88,
                "pulse_bpm": 120,
                "consciousness": "V",
                "temperature_c": 39.5,
            }
        ),
        group=ScalarGroup.MEDICAL,
        component_name="governance_clinical_ews",
    )
    registry.register(
        clinical.mews_scalar(
            {
                "systolic_bp_mmhg": 85,
                "pulse_bpm": 135,
                "respiratory_rate_bpm": 32,
                "temperature_c": 39.0,
                "avpu": "V",
            }
        ),
        group=ScalarGroup.MEDICAL,
        component_name="governance_clinical_ews",
    )
    registry.register(
        clinical.meld_na_scalar(
            {"bilirubin_mg_dl": 2.0, "creatinine_mg_dl": 1.5, "inr": 1.2, "sodium_meq_l": 130}
        ),
        group=ScalarGroup.MEDICAL,
        component_name="governance_clinical_meld",
    )
    registry.register(
        medical_device.iso14971_risk_scalar({"severity": 4, "probability": 4}),
        group=ScalarGroup.MEDICAL,
        component_name="governance_medical_device",
    )
    registry.register_all(
        ai_safety.ai_safety_scalars(
            nist_measurements={"fairness": 0.95, "performance": 0.85},
            atlas_events=[{"threat_type": "sql_injection"}, {"threat_type": "path_traversal"}],
        ),
        group=ScalarGroup.SECURITY,
        component_name="governance_ai_safety",
    )


def test_governance_registration_leaves_gate_vector_identical(
    gosnn: GlobalOmniScalarNetwork,
) -> None:
    """Registering all governance families must not change the operational vector."""
    before = list(gosnn._collect_all_scalars().values())
    assert len(before) == OPERATIONAL_COUNT

    registry = GovernanceRegistry(gosnn)
    _register_every_family(registry)
    # A tag-only family (EU AI Act) is exercised too: it must register nothing.
    assert eu_ai_act.eu_ai_act_tier({"annex_iii_high_risk": True}).registers is False

    after = list(gosnn._collect_all_scalars().values())

    # The gate's input is byte-identical: same length, same values, same order.
    assert after == before
    assert len(after) == OPERATIONAL_COUNT
    assert len(after) < BAND_CAP

    # Something was actually registered (the guard is not vacuously true).
    registered = [e for e in registry.ledger if e.registered]
    assert registered, "expected governance scalars to register"

    # Every registered governance key is metric-only ...
    for entry in registered:
        assert GlobalOmniScalarNetwork._is_metric_only_scalar(entry.name), entry.name
    # ... and no dropped (UNDECIDABLE) family leaked a key into the surface.
    assert not any(k.startswith("omni_owasp_llm_") for k in gosnn._collect_all_scalars())


def test_registered_governance_scalars_are_discoverable(
    gosnn: GlobalOmniScalarNetwork,
) -> None:
    """Registered governance scalars are stored for reporting but excluded from the gate."""
    registry = GovernanceRegistry(gosnn)
    registry.register(
        medical_device.iso14971_risk_scalar({"severity": 3, "probability": 5}),
        group=ScalarGroup.MEDICAL,
        component_name="governance_medical_device",
    )
    # Discoverable via get_scalar (reporting) ...
    assert gosnn.get_scalar("omni_iso14971_risk_index") == pytest.approx(15 / 25)
    # ... but absent from the operational vector the gate consumes.
    assert "omni_iso14971_risk_index" not in gosnn._collect_all_scalars()
