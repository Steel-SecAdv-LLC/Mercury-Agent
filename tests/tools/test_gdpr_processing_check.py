# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""GDPR processing checklist: tool envelope and article-accurate findings."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.compliance import GDPRCompliance
from omni_mercury_engine.tools import TOOL_REGISTRY
from omni_mercury_engine.tools._base import validate_envelope

if TYPE_CHECKING:
    from pathlib import Path


def _documented_context() -> dict[str, Any]:
    rights = (
        "right_to_access",
        "right_to_rectification",
        "right_to_erasure",
        "right_to_restriction",
        "right_to_data_portability",
        "right_to_withdraw_consent",
    )
    return {
        "processes_personal_data": True,
        "legal_basis_type": "consent",
        "legal_basis_documentation": {
            "freely_given": True,
            "specific_purpose": True,
            "informed": True,
            "unambiguous_indication": True,
            "withdrawable": True,
        },
        "data_minimization_evidence": {
            "adequacy_justified": True,
            "retention_policy_defined": True,
        },
        "purpose_limitation_evidence": {"purposes_documented": True},
        "privacy_notice_provided": True,
        "data_protection_by_design": True,
        "data_subject_rights_config": {
            right: {"mechanism_exists": True, "response_timeline_days": 21} for right in rights
        },
        "security_measures": dict.fromkeys(
            (
                "pseudonymization",
                "confidentiality",
                "integrity",
                "availability",
                "resilience",
                "restoration_capability",
                "testing_process",
            ),
            True,
        ),
    }


def _run(tmp_path: Path, context: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    ctx = tmp_path / "ctx.json"
    out = tmp_path / "cert.json"
    ctx.write_text(json.dumps(context), encoding="utf-8")
    rc = TOOL_REGISTRY["gdpr_processing_check"](["--context", str(ctx), "--output", str(out)])
    cert: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    validate_envelope(cert)
    return rc, cert


def test_documented_operation_is_ok(tmp_path: Path) -> None:
    rc, cert = _run(tmp_path, _documented_context())
    assert rc == 0
    assert cert["status"] == "ok"
    assert cert["body"]["compliant"] is True
    assert cert["body"]["legal_basis"]["conditions_documented"] is True


def test_findings_carry_article_remediation_and_evidence(tmp_path: Path) -> None:
    context = _documented_context()
    context["data_subject_rights_config"]["right_to_withdraw_consent"] = {
        "mechanism_exists": True,
        "response_timeline_days": 45,
    }
    context["personal_data_breach"] = {"occurred": True, "hours_to_authority_notification": 96}
    rc, cert = _run(tmp_path, context)
    assert rc != 0
    assert cert["status"] == "fail"
    violations = cert["body"]["violations"]
    assert violations[0]["severity"] == "critical"  # most severe first
    breach = next(v for v in violations if v["article"] == "GDPR Art. 33")
    assert "72 hours" in breach["remediation"]
    withdraw = next(v for v in violations if v["article"] == "GDPR Art. 12(3)/Art. 7(3)")
    # Keyed on the primary article (12), not on the "Art. 7" it cites.
    assert withdraw["remediation"].startswith("Implement a request mechanism")
    assert withdraw["evidence_required"] == ["request_handling_procedure", "response_time_log"]
    assert cert["body"]["data_subject_rights"]["right_to_withdraw_consent"] is False


def test_unmet_obligation_exits_nonzero(tmp_path: Path) -> None:
    context = _documented_context()
    context["privacy_notice_provided"] = False
    ctx = tmp_path / "ctx.json"
    ctx.write_text(json.dumps(context), encoding="utf-8")
    rc = TOOL_REGISTRY["gdpr_processing_check"](
        ["--context", str(ctx), "--output", str(tmp_path / "c.json")]
    )
    assert rc != 0


def test_extension_and_basis_specific_rights() -> None:
    gdpr = GDPRCompliance()
    context = _documented_context()
    context["data_subject_rights_config"]["right_to_access"] = {
        "mechanism_exists": True,
        "response_timeline_days": 80,
        "extension_justified": True,
    }
    assert gdpr.check_gdpr_compliance(context)[0] is True
    contract = dict(context, legal_basis_type="contract")
    contract["legal_basis_documentation"] = {
        "contract_reference": "MSA-1",
        "processing_necessary": True,
    }
    # Consent withdrawal does not apply to contract processing.
    assert "right_to_withdraw_consent" not in gdpr._applicable_rights(contract)


def test_legal_obligation_accepts_union_law() -> None:
    gdpr = GDPRCompliance()
    basis = gdpr._validate_legal_basis(
        {
            "legal_basis_type": "legal_obligation",
            "legal_basis_documentation": {
                "legal_reference": "Reg. (EU) 2022/2554 Art. 19",
                "union_or_member_state_law": "Union",
            },
        }
    )
    assert basis.valid is True


def test_no_personal_data_is_trivially_compliant() -> None:
    result = GDPRCompliance().assess_compliance({"processes_personal_data": False})
    assert result.compliant is True
    assert result.legal_basis is None
    assert result.compliance_score == 1.0


def test_malformed_timeline_strings_are_unmet_not_crashes() -> None:
    """Operator JSON with non-numeric timelines yields a fail-closed verdict.

    Regression: ``float("30 days")`` / ``float("48h")`` raised ValueError
    out of ``check_gdpr_compliance`` instead of reporting the right or the
    Art. 33 notification as unmet.
    """
    gdpr = GDPRCompliance()

    context: dict[str, Any] = {
        "processes_personal_data": True,
        "data_subject_rights_config": {
            "right_to_access": {
                "mechanism_exists": True,
                "response_timeline_days": "30 days",
            }
        },
        "personal_data_breach": {
            "occurred": True,
            "hours_to_authority_notification": "48h",
        },
    }
    compliant, violations = gdpr.check_gdpr_compliance(context)

    assert not compliant
    assert any("Art. 33" in v for v in violations)
    rights = gdpr._check_data_subject_rights(context)
    assert rights.get("right_to_access") is False
