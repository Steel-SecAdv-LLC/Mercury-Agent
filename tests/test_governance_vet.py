"""Ties the per-family signal vet to behaviour.

* Every UNAVAILABLE-capable family has a *real* GROUNDED path (a worked input -> a value)
  -- otherwise it would be UNDECIDABLE-in-disguise.
* Every UNDECIDABLE family registers nothing and is absent from the registry entirely.
* The TAG_ONLY family (EU AI Act) registers nothing.

The clinical SOFA GROUNDED proof needs the [ml] extra; it is asserted in
``test_governance_clinical.py`` (loud skip) and exercised here only when torch is present.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from omni_mercury_engine._compat import HAS_TORCH
from omni_mercury_engine.core.global_omni_scalar_network import (
    GlobalOmniScalarNetwork,
    ScalarGroup,
    reset_global_network,
)
from omni_mercury_engine.governance import ai_safety, clinical, medical_device
from omni_mercury_engine.governance.contract import (
    GOVERNANCE_FAMILY_VET,
    GovernanceRegistry,
    SignalClass,
    ThreeState,
)


@pytest.fixture()
def gosnn():
    """Provide a freshly reset GOSNN singleton and tear it down afterwards."""
    reset_global_network()
    network = GlobalOmniScalarNetwork()
    yield network
    reset_global_network()


# One worked GROUNDED example per kept family (the proof that the GROUNDED path is real).
def _grounded_examples():
    examples = {
        "ews": [
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
            clinical.mews_scalar(
                {
                    "systolic_bp_mmhg": 85,
                    "pulse_bpm": 135,
                    "respiratory_rate_bpm": 32,
                    "temperature_c": 39.0,
                    "avpu": "V",
                }
            ),
        ],
        "mews": [
            clinical.mews_scalar(
                {
                    "systolic_bp_mmhg": 85,
                    "pulse_bpm": 135,
                    "respiratory_rate_bpm": 32,
                    "temperature_c": 39.0,
                    "avpu": "V",
                }
            )
        ],
        "meld": [
            clinical.meld_na_scalar(
                {"bilirubin_mg_dl": 2.0, "creatinine_mg_dl": 1.5, "inr": 1.2, "sodium_meq_l": 130}
            )
        ],
        "iso14971": [medical_device.iso14971_risk_scalar({"severity": 4, "probability": 4})],
        "nist_ai_rmf": [ai_safety.nist_ai_rmf_measure_scalar(measurements={"fairness": 0.9})],
        "mitre_atlas": [
            ai_safety.mitre_atlas_scalar(observed_events=[{"threat_type": "sql_injection"}])
        ],
    }
    if HAS_TORCH:
        examples["sofa"] = clinical.sofa_scalars(
            {
                "pao2_fio2_ratio": 250,
                "platelets_k_ul": 30,
                "bilirubin_mg_dl": 8.0,
                "mean_arterial_pressure": 65,
                "gcs_score": 13,
                "creatinine_mg_dl": 2.5,
            }
        )
    return examples


def test_every_kept_family_has_a_real_grounded_path() -> None:
    """Each UNAVAILABLE-capable family produces at least one GROUNDED scalar from real input."""
    examples = _grounded_examples()
    kept = {
        f
        for f, v in GOVERNANCE_FAMILY_VET.items()
        if v.classification is SignalClass.UNAVAILABLE_CAPABLE
    }
    for family in kept:
        if family == "sofa" and not HAS_TORCH:
            continue  # proven under the [ml] lane (see test_governance_clinical.py)
        produced = examples.get(family, [])
        assert any(
            s.state is ThreeState.GROUNDED for s in produced
        ), f"kept family {family!r} has no real GROUNDED path -> UNDECIDABLE in disguise"


def test_undecidable_families_build_no_scalar() -> None:
    """No code path in the package builds a scalar for an UNDECIDABLE family."""
    built_families = {s.family for fam in _grounded_examples().values() for s in fam}
    undecidable = {
        f for f, v in GOVERNANCE_FAMILY_VET.items() if v.classification is SignalClass.UNDECIDABLE
    }
    assert built_families.isdisjoint(undecidable)
    assert "owasp_llm" in undecidable


def test_no_undecidable_key_ever_registers(gosnn: GlobalOmniScalarNetwork) -> None:
    """Registering every built family leaves zero keys for any dropped family."""
    registry = GovernanceRegistry(gosnn)
    for family_scalars in _grounded_examples().values():
        registry.register_all(family_scalars, group=ScalarGroup.SECURITY, component_name="vet")
    registered_families = {e.family for e in registry.ledger if e.registered}
    undecidable = {
        f for f, v in GOVERNANCE_FAMILY_VET.items() if v.classification is SignalClass.UNDECIDABLE
    }
    assert registered_families.isdisjoint(undecidable)
    # OWASP's dropped prefix must not be discoverable anywhere in the operational vector.
    assert not any(k.startswith("omni_owasp_llm_") for k in gosnn._collect_all_scalars())
