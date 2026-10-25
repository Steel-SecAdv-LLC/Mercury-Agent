# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GDPR lawfulness-of-processing assessment (Regulation (EU) 2016/679).

:class:`~omni_mercury_engine.explainability.gdpr_compliance.GDPRExplainer`
covers Article 22 *per decision* (explanations, human review, objection).
This module covers the *processing operation* a deployment performs, from a
documented context mapping, and reports each unmet obligation with its
article, severity, remediation and the evidence an auditor would ask for:

* Art. 5(1)(b)/(c)/(e) purpose limitation, data minimisation, storage limitation
* Art. 6 lawful basis (each basis' own conditions; Art. 6(3) Union or
  Member State law for legal obligations) and Art. 7 consent conditions
* Art. 9 special-category data and its Art. 9(2) exemption
* Art. 12(3) and Arts. 15-22 data-subject rights: a mechanism must exist for
  every right that applies to the chosen basis, answered within one month
  (extendable by two further months, Art. 12(3), when justified)
* Art. 13-14 information obligations
* Art. 22 automated individual decision-making safeguards
* Art. 25 data protection by design and by default
* Art. 32 security of processing
* Art. 33-34 personal-data-breach notification
* Art. 35 DPIA for high-risk processing
* Arts. 44-49 international transfers (adequacy, Art. 46 safeguards,
  Art. 49 derogations)

The assessment is a documentation check: it verifies that the evidence a
controller must hold is declared, not that the declared evidence is true.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "GDPRCompliance",
    "GDPRComplianceResult",
    "GDPRComplianceViolation",
    "GDPRLegalBasis",
]

#: Data-subject rights and the article that grants each one.
_RIGHT_ARTICLES: dict[str, str] = {
    "right_to_access": "Art. 15",
    "right_to_rectification": "Art. 16",
    "right_to_erasure": "Art. 17",
    "right_to_data_portability": "Art. 20",
    "right_to_object": "Art. 21",
    "right_to_restriction": "Art. 18",
    "right_to_withdraw_consent": "Art. 7(3)",
    "right_not_to_be_subject_to_automated_decisions": "Art. 22",
}

_VALID_LEGAL_BASES: tuple[str, ...] = (
    "consent",
    "contract",
    "legal_obligation",
    "vital_interests",
    "public_task",
    "legitimate_interests",
)

_ART9_EXEMPTIONS: tuple[str, ...] = (
    "explicit_consent",  # 9(2)(a)
    "employment_law",  # 9(2)(b)
    "vital_interests",  # 9(2)(c)
    "legitimate_activities",  # 9(2)(d)
    "manifestly_public",  # 9(2)(e)
    "legal_claims",  # 9(2)(f)
    "public_interest",  # 9(2)(g)
    "health_social_care",  # 9(2)(h)
    "public_health",  # 9(2)(i)
    "archiving_research",  # 9(2)(j)
)

_TRANSFER_MECHANISMS: tuple[str, ...] = (
    "adequacy_decision",  # Art. 45
    "standard_contractual_clauses",  # Art. 46(2)(c)
    "binding_corporate_rules",  # Art. 46(2)(b), Art. 47
    "approved_certification",  # Art. 46(2)(f)
    "approved_code_of_conduct",  # Art. 46(2)(e)
    "explicit_consent",  # Art. 49(1)(a)
    "contract_necessity",  # Art. 49(1)(b)-(c)
    "public_interest",  # Art. 49(1)(d)
    "legal_claims",  # Art. 49(1)(e)
    "vital_interests",  # Art. 49(1)(f)
)

_SECURITY_MEASURES: tuple[tuple[str, str], ...] = (
    ("pseudonymization", "Art. 32(1)(a): Pseudonymisation/encryption"),
    ("confidentiality", "Art. 32(1)(b): Confidentiality assurance"),
    ("integrity", "Art. 32(1)(b): Integrity assurance"),
    ("availability", "Art. 32(1)(b): Availability assurance"),
    ("resilience", "Art. 32(1)(b): Resilience of systems"),
    ("restoration_capability", "Art. 32(1)(c): Timely restoration capability"),
    ("testing_process", "Art. 32(1)(d): Regular security testing"),
)

_ONE_MONTH_DAYS = 30
_EXTENDED_DAYS = 90  # one month + two further months, Art. 12(3)
_BREACH_NOTIFICATION_HOURS = 72

_SCORE_IMPACT: dict[str, float] = {"critical": 0.3, "high": 0.2, "medium": 0.1, "low": 0.05}

#: Remediation per primary article number (Art. 5 split by principle).
_REMEDIATIONS: dict[str, str] = {
    "5(1)(b)": "Document explicit processing purposes at collection",
    "5(1)(c)": "Document why the data collected is adequate, relevant and limited",
    "5(1)(e)": "Define and enforce a retention schedule",
    "6": "Establish and document a valid legal basis before processing",
    "7": "Renew consent and keep records showing it remains valid",
    "9": "Obtain explicit consent or document the applicable Art. 9(2) exemption",
    "12": "Implement a request mechanism for the right and answer within one month",
    "13": "Provide the Art. 13/14 privacy information at collection",
    "22": "Implement meaningful human review with an explanation of the logic",
    "25": "Document data-protection-by-design and by-default measures",
    "32": "Implement and document the required security measures",
    "33": "Notify the supervisory authority within 72 hours of awareness",
    "34": "Communicate a high-risk breach to the affected data subjects",
    "35": "Conduct and document a Data Protection Impact Assessment",
    "44": "Put a valid Chapter V transfer mechanism in place",
}

_EVIDENCE: dict[str, list[str]] = {
    "5": ["data_inventory", "retention_schedule", "purpose_register"],
    "6": ["legal_basis_documentation", "records_of_processing"],
    "7": ["consent_records", "consent_renewal_log"],
    "9": ["consent_records", "exemption_documentation"],
    "12": ["request_handling_procedure", "response_time_log"],
    "13": ["privacy_notice"],
    "22": ["human_review_process", "logic_explanation_template"],
    "25": ["privacy_by_design_record"],
    "32": ["security_assessment", "testing_records"],
    "33": ["breach_register", "authority_notification"],
    "34": ["data_subject_communication"],
    "35": ["dpia_report", "risk_register", "mitigation_plan"],
    "44": ["transfer_agreement", "adequacy_assessment"],
}

_PRIMARY_ARTICLE = re.compile(r"Art\. (\d+)((?:\(\w+\))*)")


def _primary_article(article: str) -> tuple[str, str]:
    """``(number, subdivision)`` of the FIRST article an article reference cites.

    ``"GDPR Art. 12(3)/Art. 7(3)"`` -> ``("12", "(3)")``. Lookups key on
    this, never on substrings: a substring scan matched "Art. 7" inside the
    Art. 12(3) consent-withdrawal finding and returned the consent-renewal
    remedy.
    """
    match = _PRIMARY_ARTICLE.search(article)
    return (match.group(1), match.group(2)) if match else ("", "")


@dataclass
class GDPRComplianceViolation:
    """One unmet obligation.

    Attributes:
        article: Article reference (e.g. ``"GDPR Art. 32(1)(a)"``).
        severity: ``"critical"``, ``"high"``, ``"medium"`` or ``"low"``.
        description: What is missing.
        remediation: How to close the gap.
        evidence_required: Records an auditor would request.
        compliance_score_impact: Deduction applied to the compliance score.
    """

    article: str
    severity: str
    description: str
    remediation: str
    evidence_required: list[str] = field(default_factory=list)
    compliance_score_impact: float = 0.0


@dataclass
class GDPRLegalBasis:
    """Validated Article 6 lawful basis.

    Attributes:
        basis_type: One of the six Art. 6(1) bases.
        documentation: The supporting documentation supplied.
        valid: Whether the basis' own conditions are documented.
        expiry: Consent expiry when declared.
    """

    basis_type: str
    documentation: dict[str, Any] = field(default_factory=dict)
    valid: bool = False
    expiry: datetime | None = None


@dataclass
class GDPRComplianceResult:
    """Outcome of :meth:`GDPRCompliance.assess_compliance`.

    Attributes:
        compliant: ``True`` when no obligation is unmet.
        violations: Unmet obligations, most severe first.
        compliance_score: ``1.0`` minus severity-weighted deductions, in ``[0, 1]``.
        legal_basis: The validated lawful basis (``None`` when no personal
            data is processed).
        data_subject_rights_status: Per applicable right, whether a
            mechanism exists within the Art. 12(3) deadline.
        recommendations: Prioritised next steps.
        assessment_timestamp: UTC time of the assessment.
    """

    compliant: bool
    violations: list[GDPRComplianceViolation]
    compliance_score: float
    legal_basis: GDPRLegalBasis | None
    data_subject_rights_status: dict[str, bool]
    recommendations: list[str]
    assessment_timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


def _as_finite_number(value: Any) -> float | None:
    """``float(value)`` when it yields a finite number, else ``None``.

    Context values come straight from operator-supplied JSON
    (``tools/gdpr_processing_check``). An unparseable or non-finite
    timeline cannot demonstrate timeliness, so callers treat ``None`` as
    the obligation being unmet — a fail-closed verdict, never a crash.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


class GDPRCompliance:
    """Assess a processing operation's documented GDPR posture.

    The context mapping keys are documented on :meth:`check_gdpr_compliance`.
    """

    def __init__(self) -> None:
        """Initialise the assessor."""
        self.data_subject_rights: list[str] = list(_RIGHT_ARTICLES)
        self.special_category_types: list[str] = [
            "racial_ethnic_origin",
            "political_opinions",
            "religious_beliefs",
            "trade_union_membership",
            "genetic_data",
            "biometric_data",
            "health_data",
            "sex_life_orientation",
        ]
        self.valid_legal_bases: list[str] = list(_VALID_LEGAL_BASES)

    # ------------------------------------------------------------------
    # Article 6 / 7
    # ------------------------------------------------------------------

    def _validate_legal_basis(self, context: dict[str, Any]) -> GDPRLegalBasis:
        """Validate the Art. 6(1) basis against that basis' own conditions."""
        basis_type = str(context.get("legal_basis_type", ""))
        doc = context.get("legal_basis_documentation", {}) or {}

        if basis_type == "consent":
            # Art. 4(11): freely given, specific, informed, unambiguous;
            # Art. 7(3): as easy to withdraw as to give.
            valid = all(
                bool(doc.get(key, False))
                for key in (
                    "freely_given",
                    "specific_purpose",
                    "informed",
                    "unambiguous_indication",
                    "withdrawable",
                )
            )
        elif basis_type == "contract":
            valid = bool(doc.get("contract_reference") and doc.get("processing_necessary"))
        elif basis_type == "legal_obligation":
            # Art. 6(3): the obligation must be laid down by Union OR Member
            # State law.
            law_basis = doc.get("union_or_member_state_law", doc.get("member_state_law"))
            valid = bool(doc.get("legal_reference") and law_basis)
        elif basis_type == "vital_interests":
            valid = bool(doc.get("vital_interest_documented") and doc.get("no_alternative_basis"))
        elif basis_type == "public_task":
            valid = bool(doc.get("public_authority_mandate") and doc.get("task_documentation"))
        elif basis_type == "legitimate_interests":
            valid = all(
                bool(doc.get(key, False))
                for key in (
                    "legitimate_interest_identified",
                    "necessity_demonstrated",
                    "balancing_test_conducted",
                    "data_subject_rights_considered",
                )
            )
        else:
            valid = False

        expiry: datetime | None = None
        if basis_type == "consent" and doc.get("consent_expiry"):
            try:
                expiry = _as_utc(datetime.fromisoformat(str(doc["consent_expiry"])))
            except (ValueError, TypeError):
                expiry = None
        return GDPRLegalBasis(basis_type=basis_type, documentation=doc, valid=valid, expiry=expiry)

    # ------------------------------------------------------------------
    # Articles 12(3), 15-22
    # ------------------------------------------------------------------

    def _applicable_rights(self, context: dict[str, Any]) -> list[str]:
        """Rights that apply to this processing operation.

        Portability (Art. 20) applies to consent/contract processing by
        automated means; objection (Art. 21) to public-task and
        legitimate-interest processing (and always to direct marketing);
        consent withdrawal to consent-based processing; the Art. 22 right to
        automated decision-making. Access, rectification, erasure and
        restriction always apply.
        """
        basis = str(context.get("legal_basis_type", ""))
        rights = [
            "right_to_access",
            "right_to_rectification",
            "right_to_erasure",
            "right_to_restriction",
        ]
        if basis in ("consent", "contract") and context.get("automated_means", True):
            rights.append("right_to_data_portability")
        if basis in ("public_task", "legitimate_interests") or context.get(
            "direct_marketing", False
        ):
            rights.append("right_to_object")
        if basis == "consent":
            rights.append("right_to_withdraw_consent")
        if context.get("automated_decision", False):
            rights.append("right_not_to_be_subject_to_automated_decisions")
        return rights

    def _check_data_subject_rights(self, context: dict[str, Any]) -> dict[str, bool]:
        """Return, per applicable right, whether it is honoured in time."""
        config = context.get("data_subject_rights_config", {}) or {}
        status: dict[str, bool] = {}
        for right in self._applicable_rights(context):
            cfg = config.get(right, {}) or {}
            limit = _EXTENDED_DAYS if cfg.get("extension_justified", False) else _ONE_MONTH_DAYS
            days = _as_finite_number(cfg.get("response_timeline_days"))
            in_time = days is not None and 0 <= days <= limit
            status[right] = bool(cfg.get("mechanism_exists", False)) and in_time
        return status

    # ------------------------------------------------------------------
    # Assessment
    # ------------------------------------------------------------------

    def check_gdpr_compliance(self, context: dict[str, Any]) -> tuple[bool, list[str]]:
        """List the unmet obligations of one processing operation.

        Recognised context keys: ``processes_personal_data``,
        ``legal_basis_type``, ``legal_basis_documentation``,
        ``data_subject_rights_config`` (per right: ``mechanism_exists``,
        ``response_timeline_days``, ``extension_justified``),
        ``automated_means``, ``direct_marketing``,
        ``data_minimization_evidence`` (``adequacy_justified``,
        ``retention_policy_defined``), ``purpose_limitation_evidence``
        (``purposes_documented``, ``further_processing``,
        ``compatibility_assessment``), ``privacy_notice_provided``,
        ``automated_decision``, ``human_review_mechanism`` (``exists``,
        ``meaningful_review``), ``automated_decision_logic_explained``,
        ``special_category_data``, ``special_category_exemption``,
        ``data_protection_by_design``, ``security_measures``,
        ``personal_data_breach`` (``occurred``,
        ``hours_to_authority_notification``, ``high_risk_to_subjects``,
        ``subjects_informed``), ``high_risk_processing``,
        ``data_protection_impact_assessment`` (``conducted``,
        ``risk_mitigation_documented``), ``cross_border_transfer`` and
        ``transfer_mechanism``.

        Args:
            context: Processing-operation context.

        Returns:
            ``(compliant, violations)`` where each violation string starts
            with its article reference followed by ``": "``.
        """
        violations: list[str] = []
        if not context.get("processes_personal_data", False):
            return True, violations

        basis = self._validate_legal_basis(context)
        if basis.basis_type not in _VALID_LEGAL_BASES:
            violations.append(
                f"GDPR Art. 6(1): No recognised lawful basis declared ('{basis.basis_type}')"
            )
        elif not basis.valid:
            violations.append(
                f"GDPR Art. 6(1): Lawful basis '{basis.basis_type}' declared but its "
                "conditions are not documented"
            )
        if basis.expiry is not None and basis.expiry < datetime.now(UTC):
            violations.append("GDPR Art. 7: Consent has expired and must be renewed")

        minimisation = context.get("data_minimization_evidence", {}) or {}
        if not minimisation.get("adequacy_justified", False):
            violations.append(
                "GDPR Art. 5(1)(c): Data minimization - no documented justification that "
                "the data is adequate, relevant and limited"
            )
        if not minimisation.get("retention_policy_defined", False):
            violations.append("GDPR Art. 5(1)(e): Storage limitation - no defined retention policy")

        purpose = context.get("purpose_limitation_evidence", {}) or {}
        if not purpose.get("purposes_documented", False):
            violations.append(
                "GDPR Art. 5(1)(b): Purpose limitation - purposes not documented at collection"
            )
        if purpose.get("further_processing", False) and not purpose.get(
            "compatibility_assessment", False
        ):
            violations.append(
                "GDPR Art. 5(1)(b): Further processing without a compatibility assessment"
            )

        if not context.get("privacy_notice_provided", False):
            violations.append(
                "GDPR Art. 13-14: Information obligations - no privacy notice provided"
            )

        for right, honoured in self._check_data_subject_rights(context).items():
            if not honoured:
                violations.append(
                    f"GDPR Art. 12(3)/{_RIGHT_ARTICLES[right]}: {right} has no request "
                    "mechanism answered within the Art. 12(3) deadline"
                )

        if context.get("automated_decision", False):
            review = context.get("human_review_mechanism", {}) or {}
            if not review.get("exists", False):
                violations.append(
                    "GDPR Art. 22(3): Automated decision-making without a right to obtain "
                    "human intervention"
                )
            elif not review.get("meaningful_review", False):
                violations.append(
                    "GDPR Art. 22(3): Human review is not meaningful (must be able to change "
                    "the outcome)"
                )
            if not context.get("automated_decision_logic_explained", False):
                violations.append(
                    "GDPR Art. 22/Art. 13(2)(f): No meaningful information about the logic "
                    "involved is given to the data subject"
                )

        if context.get("special_category_data", False):
            exemption = str(context.get("special_category_exemption", ""))
            if exemption not in _ART9_EXEMPTIONS:
                violations.append(
                    "GDPR Art. 9: Special-category data processed without a valid Art. 9(2) "
                    f"exemption (provided: '{exemption}')"
                )

        if not context.get("data_protection_by_design", False):
            violations.append(
                "GDPR Art. 25: Data protection by design and by default not documented"
            )

        measures = context.get("security_measures", {}) or {}
        for measure, reference in _SECURITY_MEASURES:
            if not measures.get(measure, False):
                violations.append(f"GDPR {reference} not demonstrated")

        breach = context.get("personal_data_breach", {}) or {}
        if breach.get("occurred", False):
            hours = _as_finite_number(breach.get("hours_to_authority_notification"))
            if hours is None or hours > _BREACH_NOTIFICATION_HOURS:
                violations.append(
                    "GDPR Art. 33: Supervisory authority not notified within 72 hours of "
                    "becoming aware of the breach"
                )
            if breach.get("high_risk_to_subjects", False) and not breach.get(
                "subjects_informed", False
            ):
                violations.append(
                    "GDPR Art. 34: High-risk breach not communicated to the data subjects"
                )

        if context.get("high_risk_processing", False):
            dpia = context.get("data_protection_impact_assessment", {}) or {}
            if not dpia.get("conducted", False):
                violations.append(
                    "GDPR Art. 35: High-risk processing without a Data Protection Impact "
                    "Assessment"
                )
            elif not dpia.get("risk_mitigation_documented", False):
                violations.append(
                    "GDPR Art. 35(7): DPIA conducted but risk mitigation measures not documented"
                )

        if context.get("cross_border_transfer", False):
            mechanism = str(context.get("transfer_mechanism", ""))
            if mechanism not in _TRANSFER_MECHANISMS:
                violations.append(
                    "GDPR Art. 44-49: Cross-border transfer without a valid Chapter V "
                    f"mechanism (provided: '{mechanism}')"
                )

        return len(violations) == 0, violations

    @staticmethod
    def _article_of(violation: str) -> str:
        return violation.split(": ", 1)[0] if ": " in violation else "GDPR"

    @staticmethod
    def _severity_of(article: str) -> str:
        """Severity of a finding; Art. 22 anywhere in the reference is critical."""
        number, _ = _primary_article(article)
        if number in ("22", "33", "34") or re.search(r"Art\. 22\b", article):
            return "critical"
        if number in ("6", "7", "9", "35", "44"):
            return "high"
        return "medium"

    def assess_compliance(self, context: dict[str, Any]) -> GDPRComplianceResult:
        """Assess a processing operation and score it.

        Args:
            context: Processing-operation context (see
                :meth:`check_gdpr_compliance`).

        Returns:
            :class:`GDPRComplianceResult` with violations ordered most severe first.
        """
        compliant, violation_strings = self.check_gdpr_compliance(context)
        order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        violations: list[GDPRComplianceViolation] = []
        for text in violation_strings:
            article = self._article_of(text)
            severity = self._severity_of(article)
            violations.append(
                GDPRComplianceViolation(
                    article=article,
                    severity=severity,
                    description=text,
                    remediation=self._get_remediation(article),
                    evidence_required=self._get_evidence_requirements(article),
                    compliance_score_impact=self._get_score_impact(severity),
                )
            )
        violations.sort(key=lambda v: order.get(v.severity, 9))
        score = max(0.0, min(1.0, 1.0 - sum(v.compliance_score_impact for v in violations)))

        processes = bool(context.get("processes_personal_data", False))
        return GDPRComplianceResult(
            compliant=compliant,
            violations=violations,
            compliance_score=score,
            legal_basis=self._validate_legal_basis(context) if processes else None,
            data_subject_rights_status=(
                self._check_data_subject_rights(context) if processes else {}
            ),
            recommendations=self._generate_recommendations(violations, context),
        )

    @staticmethod
    def _get_remediation(article: str) -> str:
        """Remediation guidance for an article reference (by primary article)."""
        number, subdivision = _primary_article(article)
        remediation = _REMEDIATIONS.get(f"{number}{subdivision}") or _REMEDIATIONS.get(number)
        if remediation is None and number == "5":
            remediation = _REMEDIATIONS.get(f"5{subdivision[:6]}")
        return remediation or "Review the GDPR requirement and implement the corresponding control"

    @staticmethod
    def _get_evidence_requirements(article: str) -> list[str]:
        """Evidence an auditor would request for an article reference."""
        number, _ = _primary_article(article)
        return list(_EVIDENCE.get(number, ["compliance_documentation"]))

    @staticmethod
    def _get_score_impact(severity: str) -> float:
        """Compliance-score deduction for a severity."""
        return _SCORE_IMPACT.get(severity, 0.1)

    @staticmethod
    def _generate_recommendations(
        violations: list[GDPRComplianceViolation], context: dict[str, Any]
    ) -> list[str]:
        """Prioritised next steps."""
        recommendations: list[str] = []
        if any(v.severity == "critical" for v in violations):
            recommendations.append("IMMEDIATE: close the critical gaps before processing continues")
        seen: set[str] = set()
        for violation in violations:
            if violation.remediation not in seen:
                seen.add(violation.remediation)
                recommendations.append(violation.remediation)
        if context.get("processes_personal_data", False):
            if context.get("requires_dpo", False) and not context.get(
                "data_protection_officer", False
            ):
                recommendations.append("Appoint a Data Protection Officer (Art. 37)")
            if not context.get("records_of_processing", False):
                recommendations.append("Maintain records of processing activities (Art. 30)")
        return recommendations
