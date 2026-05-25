"""Tests for AI-assurance scalars: NIST AI RMF (MEASURE) + MITRE ATLAS kept, OWASP dropped.

These assert the per-family signal vet *behaviourally*: each kept family has a GROUNDED path
from a real runtime signal AND an UNAVAILABLE path when that signal is absent; the dropped
OWASP family produces no scalar at all.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")  # governance.contract -> GOSNN imports numpy.

from omni_mercury_engine.governance import ai_safety
from omni_mercury_engine.governance.contract import (
    GOVERNANCE_FAMILY_VET,
    SignalClass,
    ThreeState,
)


# --- NIST AI RMF (kept; grounded through MEASURE) ---------------------------------------
def test_nist_grounds_in_live_measure_metrics() -> None:
    """A real input (fairness/performance scores) produces a real value: mean trustworthiness."""
    scalar = ai_safety.nist_ai_rmf_measure_scalar(
        measurements={"fairness": 0.95, "performance": 0.85}
    )
    assert scalar.state is ThreeState.GROUNDED
    assert scalar.value == pytest.approx((0.95 + 0.85) / 2)
    assert scalar.provenance["function"] == "MEASURE"


def test_nist_abstains_without_any_metric() -> None:
    """No MEASURE metric this run -> UNAVAILABLE (the capability is real; nothing produced)."""
    scalar = ai_safety.nist_ai_rmf_measure_scalar()
    assert scalar.state is ThreeState.UNAVAILABLE
    assert scalar.missing_inputs == ("measurements",)


def test_nist_ignores_unknown_or_out_of_range_metrics() -> None:
    """Only recognised unit-interval MEASURE metrics ground; junk abstains, never invents."""
    bad = ai_safety.nist_ai_rmf_measure_scalar(measurements={"made_up": 0.5, "fairness": 1.4})
    assert bad.state is ThreeState.UNAVAILABLE
    good = ai_safety.nist_ai_rmf_measure_scalar(measurements={"made_up": 0.5, "fairness": 0.7})
    assert good.state is ThreeState.GROUNDED
    assert good.value == pytest.approx(0.7)


# --- MITRE ATLAS (kept; grounded in observed threat-detection events) -------------------
def test_atlas_grounds_in_observed_threat_events() -> None:
    """Observed sql_injection + path_traversal cover 2/2 observable tactics -> 1.0."""
    scalar = ai_safety.mitre_atlas_scalar(
        observed_events=[
            {"threat_type": "sql_injection", "confidence": 0.9},
            {"threat_type": "path_traversal", "confidence": 0.5},
        ]
    )
    assert scalar.state is ThreeState.GROUNDED
    assert scalar.value == pytest.approx(1.0)
    assert scalar.provenance["observed_tactics"] == ["discovery", "initial_access"]


def test_atlas_partial_coverage() -> None:
    """A single observed tactic covers 1/2 of the engine's observable ATLAS surface."""
    scalar = ai_safety.mitre_atlas_scalar(observed_events=[{"threat_type": "xss"}])
    assert scalar.state is ThreeState.GROUNDED
    assert scalar.value == pytest.approx(0.5)


def test_atlas_abstains_without_events_or_unmappable_events() -> None:
    """No event, or no event mapping to an observable tactic -> UNAVAILABLE."""
    assert ai_safety.mitre_atlas_scalar().state is ThreeState.UNAVAILABLE
    unmapped = ai_safety.mitre_atlas_scalar(observed_events=[{"threat_type": "model_extraction"}])
    assert unmapped.state is ThreeState.UNAVAILABLE


# --- OWASP LLM (dropped; UNDECIDABLE) ---------------------------------------------------
def test_owasp_family_is_not_built() -> None:
    """OWASP LLM produces no scalar: ai_safety builds only the two kept families."""
    names = {s.name for s in ai_safety.ai_safety_scalars()}
    assert names == {"omni_nist_airmf_measure", "omni_mitre_atlas_coverage"}
    assert not any("owasp" in n for n in names)
    assert not hasattr(ai_safety, "owasp_llm_scalar")


def test_owasp_vet_is_undecidable_with_no_runtime_signal() -> None:
    """The vet records OWASP LLM as UNDECIDABLE (no runtime signal in this engine)."""
    vet = GOVERNANCE_FAMILY_VET["owasp_llm"]
    assert vet.classification is SignalClass.UNDECIDABLE
    assert vet.runtime_signal.startswith("none")


def test_ai_safety_scalars_full_inputs_are_both_grounded() -> None:
    """With both real signals supplied, both kept scalars ground."""
    scalars = ai_safety.ai_safety_scalars(
        nist_measurements={"fairness": 0.9},
        atlas_events=[{"threat_type": "sql_injection"}],
    )
    assert all(s.state is ThreeState.GROUNDED for s in scalars)
