"""Tests for AI-assurance conformance scalars (abstain unless attested)."""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")  # governance.contract -> GOSNN imports numpy.

from omni_mercury_engine.governance import ai_safety
from omni_mercury_engine.governance.contract import ScalarStatus


def test_all_families_abstain_without_attestation() -> None:
    """With no attestation supplied, every AI-safety scalar abstains (registers nothing)."""
    scalars = ai_safety.ai_safety_scalars()
    assert len(scalars) == 3
    assert all(s.status is ScalarStatus.UNAVAILABLE for s in scalars)
    assert all(s.value is None for s in scalars)


def test_conformance_is_satisfied_over_assessed_fraction() -> None:
    """An attestation yields satisfied/assessed, recording coverage in provenance."""
    scalars = ai_safety.ai_safety_scalars(
        nist_ai_rmf={"govern": True, "map": True, "measure": False, "manage": True},
    )
    rmf = next(s for s in scalars if s.name == "omni_nist_airmf_conformance")
    assert rmf.status is ScalarStatus.AVAILABLE
    assert rmf.value == pytest.approx(3 / 4)
    assert rmf.provenance["assessed"] == 4
    assert rmf.provenance["coverage"] == pytest.approx(1.0)


def test_partial_attestation_reports_coverage() -> None:
    """A partial attestation is honest about coverage instead of assuming full scope."""
    scalars = ai_safety.ai_safety_scalars(
        owasp_llm={"llm01_prompt_injection": True, "llm06_excessive_agency": False},
    )
    owasp = next(s for s in scalars if s.name == "omni_owasp_llm_mitigation")
    assert owasp.status is ScalarStatus.AVAILABLE
    assert owasp.value == pytest.approx(1 / 2)
    assert owasp.provenance["assessed"] == 2
    assert owasp.provenance["catalog_size"] == 10
    assert owasp.provenance["coverage"] == pytest.approx(0.2)


def test_attestation_covering_no_catalog_items_abstains() -> None:
    """An attestation that names no catalog item abstains rather than inventing a score."""
    scalars = ai_safety.ai_safety_scalars(mitre_atlas={"not_a_real_tactic": True})
    atlas = next(s for s in scalars if s.name == "omni_mitre_atlas_coverage")
    assert atlas.status is ScalarStatus.UNAVAILABLE
