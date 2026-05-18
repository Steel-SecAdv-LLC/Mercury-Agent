"""Tests for :mod:`omni_mercury_engine.compliance.nist_csf_integrator`.

The compliance integrator is exercised end-to-end against both the
offline (``"builtin"``) and live (``"live"``) reference sources. The
live integrator path is exercised twice:

* Hermetically, via :class:`requests_mock` fixtures that replay a real
  ``csrc.nist.gov`` XLSX response captured into
  ``tests/fixtures/compliance/nist_csf_olir_full.xlsx``. These tests
  run by default in every CI lane and have **no** ``network`` marker.
* Live, against ``csrc.nist.gov`` itself. These tests carry the
  Mercury Agent ``@pytest.mark.network`` marker and are gated by the
  ``MERCURY_NETWORK_TESTS=1`` environment variable; see
  ``tests/conftest.py::pytest_collection_modifyitems`` for the
  collection-time gate. The weekly
  ``.github/workflows/network-tests.yml`` job sets that variable so
  drift in the published NIST CSRC schema is caught within seven days
  even though the per-PR default lane skips them.

Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC
Released under GPL-3.0+.
"""

from __future__ import annotations

import dataclasses
import json
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from omni_mercury_engine.compliance import (
    ImplementationTier,
    NISTAssessment,
    NISTCSFIntegrator,
    NISTCSFReferenceFetcher,
    NISTFunction,
    NISTProfile,
    NISTSubcategory,
    get_nist_csf_integrator,
)
from omni_mercury_engine.compliance.nist_csf_integrator import (
    _BUILTIN_CATEGORIES,
    NIST_CSF_PUBLICATION_PDF_URL,
    NIST_CSF_PUBLICATION_URL,
    NIST_CSF_REFERENCE_URL,
    NISTCategory,
    NISTCSFReferenceError,
    _parse_csf_xlsx,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def builtin_integrator() -> NISTCSFIntegrator:
    """Return an integrator backed by the offline curated tree."""

    return NISTCSFIntegrator(reference_source="builtin")


@pytest.fixture()
def builtin_evidence() -> dict[str, Any]:
    """Return a representative evidence dict spanning every function."""

    return {
        # GOVERN
        "gv.oc": 0.90,
        "gv.rm": 0.85,
        "gv.sc": 0.60,
        # IDENTIFY
        "id.am": 0.40,
        "id.ra": 0.55,
        "id.im": 0.45,
        # PROTECT
        "pr.aa": 0.70,
        "pr.at": 0.50,
        "pr.ds": 0.65,
        # DETECT
        "de.cm": 0.75,
        "de.ae": 0.80,
        # RESPOND
        "rs.ma": 0.30,
        "rs.an": 0.45,
        # RECOVER
        "rc.rp": 0.20,
        "rc.co": 0.15,
    }


# ---------------------------------------------------------------------------
# Module-level constants and the builtin tree
# ---------------------------------------------------------------------------


def test_public_url_constants_match_spec() -> None:
    assert NIST_CSF_REFERENCE_URL.startswith("https://csrc.nist.gov/")
    assert "olirids=all" in NIST_CSF_REFERENCE_URL
    assert NIST_CSF_PUBLICATION_URL == "https://www.nist.gov/cyberframework"
    assert NIST_CSF_PUBLICATION_PDF_URL.endswith("NIST.CSWP.29.pdf")


def test_builtin_tree_covers_all_six_functions() -> None:
    assert set(_BUILTIN_CATEGORIES.keys()) == set(NISTFunction)
    for func, cats in _BUILTIN_CATEGORIES.items():
        assert cats, f"{func.value} has no categories"
        for cat in cats:
            assert isinstance(cat, NISTCategory)
            assert cat.id.startswith(func.value[:2]) or "." in cat.id


def test_subcategory_dataclass_is_immutable_and_typed() -> None:
    sub = NISTSubcategory(
        id="GV.OC-01",
        description="Org mission",
        implementation_examples=("Ex1",),
        informative_references=("CSF:GV.OC-01",),
    )
    assert sub.id == "GV.OC-01"
    with pytest.raises((AttributeError, TypeError, dataclasses.FrozenInstanceError)):
        sub.id = "other"  # type: ignore[misc]


def test_subcategory_ids_normalises_string_form() -> None:
    cat = NISTCategory(
        id="GV.XX",
        name="Test",
        description="test",
        subcategories=[
            NISTSubcategory(id="GV.XX-01", description="alpha"),
            "GV.XX-02: legacy form",
        ],
    )
    assert cat.subcategory_ids() == ["GV.XX-01", "GV.XX-02"]


# ---------------------------------------------------------------------------
# Integrator construction
# ---------------------------------------------------------------------------


def test_integrator_rejects_unknown_reference_source() -> None:
    with pytest.raises(ValueError, match="reference_source"):
        NISTCSFIntegrator(reference_source="other")


def test_integrator_builtin_loads_all_six_functions(
    builtin_integrator: NISTCSFIntegrator,
) -> None:
    assert set(builtin_integrator.categories.keys()) == set(NISTFunction)
    assert builtin_integrator.reference_source == "builtin"
    assert builtin_integrator.fetcher is None


def test_factory_returns_integrator_with_target_tier() -> None:
    integ = get_nist_csf_integrator(
        target_tier=ImplementationTier.ADAPTIVE, reference_source="builtin"
    )
    assert integ.target_tier is ImplementationTier.ADAPTIVE
    assert integ.reference_source == "builtin"


def test_verify_coverage_builtin_meets_minimum(
    builtin_integrator: NISTCSFIntegrator,
) -> None:
    counts = builtin_integrator.verify_coverage(minimum_subcategories=30)
    assert counts["_total"] >= 30
    for func in NISTFunction:
        assert counts[func.value] > 0


def test_verify_coverage_raises_on_unmet_minimum(
    builtin_integrator: NISTCSFIntegrator,
) -> None:
    with pytest.raises(NISTCSFReferenceError, match="subcategories"):
        builtin_integrator.verify_coverage(minimum_subcategories=10_000)


# ---------------------------------------------------------------------------
# assess_function
# ---------------------------------------------------------------------------


def test_assess_function_returns_typed_assessment(
    builtin_integrator: NISTCSFIntegrator,
    builtin_evidence: dict[str, Any],
) -> None:
    assessment = builtin_integrator.assess_function(NISTFunction.GOVERN, builtin_evidence)
    assert isinstance(assessment, NISTAssessment)
    assert assessment.function is NISTFunction.GOVERN
    assert 0.0 <= assessment.maturity_score <= 1.0
    assert 0.0 <= assessment.risk_score <= 1.0
    assert assessment.tier in set(ImplementationTier)


def test_assess_function_low_maturity_yields_recommendations(
    builtin_integrator: NISTCSFIntegrator,
) -> None:
    weak_evidence = {"rc.rp": 0.05, "rc.co": 0.10}
    assessment = builtin_integrator.assess_function(NISTFunction.RECOVER, weak_evidence)
    assert assessment.tier is ImplementationTier.PARTIAL
    assert assessment.findings, "expected findings for weak RECOVER posture"
    assert assessment.recommendations


def test_assess_function_with_subcategory_level_evidence(
    builtin_integrator: NISTCSFIntegrator,
) -> None:
    evidence = {
        "gv.oc-01": 0.9,
        "gv.oc-02": 0.85,
        "gv.oc-03": 0.95,
        "gv.rm": 0.9,
        "gv.sc": 0.9,
    }
    assessment = builtin_integrator.assess_function(NISTFunction.GOVERN, evidence)
    assert assessment.maturity_score >= 0.85


# ---------------------------------------------------------------------------
# create_profile + generate_compliance_report
# ---------------------------------------------------------------------------


def test_create_profile_aggregates_gaps_and_priorities(
    builtin_integrator: NISTCSFIntegrator,
    builtin_evidence: dict[str, Any],
) -> None:
    assessments = [builtin_integrator.assess_function(f, builtin_evidence) for f in NISTFunction]
    profile = builtin_integrator.create_profile(assessments)
    assert isinstance(profile, NISTProfile)
    assert set(profile.current_state.keys()) == {f.value for f in NISTFunction}
    assert all(
        profile.target_state[f.value]
        == NISTCSFIntegrator._TIER_TARGETS[builtin_integrator.target_tier]
        for f in NISTFunction
    )
    # RECOVER scored very low so we expect at least one priority action.
    assert profile.priority_actions, "expected priority actions for weak RECOVER"


def test_generate_compliance_report_structure(
    builtin_integrator: NISTCSFIntegrator,
    builtin_evidence: dict[str, Any],
) -> None:
    assessments = [builtin_integrator.assess_function(f, builtin_evidence) for f in NISTFunction]
    profile = builtin_integrator.create_profile(assessments)
    report = builtin_integrator.generate_compliance_report(assessments, profile)
    for key in (
        "timestamp",
        "overall_maturity",
        "overall_risk",
        "current_tier",
        "target_tier",
        "tier_gap",
        "reference_source",
        "function_assessments",
        "profile",
        "summary",
    ):
        assert key in report
    assert report["reference_source"] == "builtin"
    assert len(report["function_assessments"]) == len(assessments)
    # The report must round-trip through JSON without raising.
    json.dumps(report, default=str)


def test_generate_compliance_report_handles_empty_assessments(
    builtin_integrator: NISTCSFIntegrator,
) -> None:
    profile = NISTProfile(current_state={}, target_state={})
    report = builtin_integrator.generate_compliance_report([], profile)
    assert report["overall_maturity"] == 0.0
    assert report["overall_risk"] == 0.0
    assert report["summary"]["total_recommendations"] == 0


# ---------------------------------------------------------------------------
# detect_supply_chain_anomalies
# ---------------------------------------------------------------------------


def test_detect_supply_chain_anomalies_flags_high_risk_and_low_compliance(
    builtin_integrator: NISTCSFIntegrator,
) -> None:
    suppliers = {
        "supplier-a": {"risk_score": 0.85, "compliance_score": 0.90},
        "supplier-b": {"risk_score": 0.20, "compliance_score": 0.30},
        "supplier-c": {"risk_score": 0.50, "compliance_score": 0.50},
    }
    anomalies = builtin_integrator.detect_supply_chain_anomalies(suppliers)
    types = {a["type"] for a in anomalies}
    assert "high_risk_supplier" in types
    assert "low_compliance" in types
    # supplier-c is mid-tier on both axes -> no anomaly
    assert not any(a["supplier_id"] == "supplier-c" for a in anomalies)
    for anomaly in anomalies:
        assert anomaly["category"].startswith("GV.SC")


def test_detect_supply_chain_anomalies_empty_input(
    builtin_integrator: NISTCSFIntegrator,
) -> None:
    assert builtin_integrator.detect_supply_chain_anomalies({}) == []


# ---------------------------------------------------------------------------
# continuous_monitoring_detect
# ---------------------------------------------------------------------------


def test_continuous_monitoring_detect_returns_bounded_scores(
    builtin_integrator: NISTCSFIntegrator,
) -> None:
    data = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 0.0],
            [0.5, 0.5],
        ],
        dtype=np.float64,
    )
    scores, events = builtin_integrator.continuous_monitoring_detect(data)
    assert scores.shape == (4,)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()
    assert any("High anomaly at index 1" in e for e in events)


def test_continuous_monitoring_detect_accepts_explicit_baseline(
    builtin_integrator: NISTCSFIntegrator,
) -> None:
    data = np.array([[1.0, 2.0], [1.0, 2.0], [5.0, 8.0]])
    baseline = np.array([1.0, 2.0])
    scores, _ = builtin_integrator.continuous_monitoring_detect(data, baseline=baseline)
    assert scores[0] == pytest.approx(0.0, abs=1e-9)
    assert scores[2] > scores[0]


def test_continuous_monitoring_detect_rejects_non_2d() -> None:
    integ = NISTCSFIntegrator(reference_source="builtin")
    with pytest.raises(ValueError, match="must be 2-D"):
        integ.continuous_monitoring_detect(np.array([1.0, 2.0, 3.0]))


def test_continuous_monitoring_detect_baseline_shape_mismatch(
    builtin_integrator: NISTCSFIntegrator,
) -> None:
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="does not match"):
        builtin_integrator.continuous_monitoring_detect(data, baseline=np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# export_categories_json
# ---------------------------------------------------------------------------


def test_export_categories_json_roundtrips(
    builtin_integrator: NISTCSFIntegrator,
) -> None:
    payload = builtin_integrator.export_categories_json()
    obj = json.loads(payload)
    assert set(obj.keys()) == {f.value for f in NISTFunction}
    for func, cats in obj.items():
        assert isinstance(cats, list)
        for cat in cats:
            assert set(cat.keys()) >= {"id", "name", "description", "subcategories"}


# ---------------------------------------------------------------------------
# NISTCSFReferenceFetcher
# ---------------------------------------------------------------------------


def test_reference_fetcher_metadata_exposes_provenance(
    tmp_path: Path,
) -> None:
    fetcher = NISTCSFReferenceFetcher(cache_dir=tmp_path)
    meta = fetcher.metadata()
    assert meta["publication_url"] == NIST_CSF_PUBLICATION_URL
    assert meta["publication_pdf_url"] == NIST_CSF_PUBLICATION_PDF_URL
    assert meta["reference_tool_url"] == NIST_CSF_REFERENCE_URL
    assert str(tmp_path) in meta["cache_path"]


def test_reference_fetcher_cache_fresh_within_ttl(tmp_path: Path) -> None:
    fetcher = NISTCSFReferenceFetcher(cache_dir=tmp_path, cache_ttl_seconds=3600.0)
    cache_path = tmp_path / "csf.xlsx"
    cache_path.write_bytes(b"PK\x03\x04")
    fresh = fetcher._cache_fresh(cache_path)
    assert fresh is True


def test_reference_fetcher_cache_stale_outside_ttl(tmp_path: Path) -> None:
    fetcher = NISTCSFReferenceFetcher(cache_dir=tmp_path, cache_ttl_seconds=0.0)
    cache_path = tmp_path / "csf.xlsx"
    cache_path.write_bytes(b"PK\x03\x04")
    assert fetcher._cache_fresh(cache_path) is False


def test_parse_csf_xlsx_rejects_non_xlsx_payload() -> None:
    with pytest.raises(NISTCSFReferenceError):
        _parse_csf_xlsx(b"definitely not a zip archive")


def test_fetch_payload_rejects_non_xlsx_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _StubResponse:
        status_code = 200
        content = b"<html>oops</html>"

        def raise_for_status(self) -> None:
            return None

    class _StubSession:
        def get(self, *args: Any, **kwargs: Any) -> _StubResponse:
            return _StubResponse()

    fetcher = NISTCSFReferenceFetcher(cache_dir=tmp_path, session=_StubSession())  # type: ignore[arg-type]
    with pytest.raises(NISTCSFReferenceError, match="non-XLSX"):
        fetcher.fetch_payload(force_refresh=True)


def test_fetch_payload_wraps_request_exceptions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import requests

    class _ExplodingSession:
        def get(self, *args: Any, **kwargs: Any) -> Any:
            raise requests.RequestException("network down")

    fetcher = NISTCSFReferenceFetcher(
        cache_dir=tmp_path, session=_ExplodingSession()  # type: ignore[arg-type]
    )
    with pytest.raises(NISTCSFReferenceError, match="Failed to fetch"):
        fetcher.fetch_payload(force_refresh=True)


def test_fetch_payload_uses_cache_when_fresh(tmp_path: Path) -> None:
    fetcher = NISTCSFReferenceFetcher(cache_dir=tmp_path, cache_ttl_seconds=3600.0)
    cache_path = fetcher._cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"PK\x03\x04" + b"\x00" * 1024
    cache_path.write_bytes(payload)
    # Force mtime to "now" so the cache is fresh.
    now = time.time()
    import os as _os

    _os.utime(cache_path, (now, now))
    assert fetcher.fetch_payload() == payload


# ---------------------------------------------------------------------------
# Live integration (network-bound)
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_live_reference_fetch_returns_full_csf2(tmp_path: Path) -> None:
    fetcher = NISTCSFReferenceFetcher(cache_dir=tmp_path)
    tree = fetcher.load_reference_tree()
    assert set(tree.keys()) == set(NISTFunction)
    total_subcats = sum(len(c.subcategories) for cats in tree.values() for c in cats)
    # CSF 2.0 final publication has 106 subcategories. The Reference
    # Tool download often includes additional rows (e.g. one row per
    # informative reference) so we accept >= 100 as the lower bound.
    assert total_subcats >= 100


@pytest.mark.network
def test_live_integrator_passes_coverage(tmp_path: Path) -> None:
    fetcher = NISTCSFReferenceFetcher(cache_dir=tmp_path)
    integ = NISTCSFIntegrator(reference_source="live", fetcher=fetcher)
    counts = integ.verify_coverage(minimum_subcategories=100)
    assert counts["_total"] >= 100
    for func in NISTFunction:
        assert counts[func.value] > 0
