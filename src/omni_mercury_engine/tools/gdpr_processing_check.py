# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Operator tool: GDPR lawfulness-of-processing checklist for one operation.

Reads a JSON processing-context document (the keys documented on
:meth:`omni_mercury_engine.compliance.gdpr_processing.GDPRCompliance.check_gdpr_compliance`)
and emits a certificate listing every unmet obligation with its article,
severity, remediation and the evidence an auditor would request, plus the
per-right Art. 12(3) status and the validated Art. 6 basis.

It complements the Art. 22 per-decision report
(:class:`~omni_mercury_engine.explainability.gdpr_compliance.GDPRExplainer`)
with the deployment-level checklist (Arts. 5-49). It verifies that the
required evidence is *declared*; it cannot verify that the declaration is
true. Status is ``ok`` only when nothing is unmet, so ``--require`` makes it
a CI / change-management gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omni_mercury_engine.compliance.gdpr_processing import GDPRCompliance
from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.gdpr_processing_check/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.gdpr_processing_check",
        description=(
            "Assess one processing operation against GDPR Arts. 5-49 from a "
            "JSON context document and list every unmet obligation."
        ),
    )
    parser.add_argument(
        "--context",
        required=True,
        help="JSON file describing the processing operation (see the module docstring).",
    )
    return parser


def _collect(args: argparse.Namespace) -> Certificate:
    path = Path(args.context)
    context = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(context, dict):
        raise ValueError(f"{path}: the context document must be a JSON object")

    result = GDPRCompliance().assess_compliance(context)
    basis = result.legal_basis
    body: dict[str, Any] = {
        "context_file": str(path),
        "compliant": result.compliant,
        "compliance_score": round(result.compliance_score, 4),
        "legal_basis": (
            {
                "type": basis.basis_type,
                "conditions_documented": basis.valid,
                "consent_expiry": basis.expiry.isoformat() if basis.expiry else None,
            }
            if basis is not None
            else None
        ),
        "data_subject_rights": dict(sorted(result.data_subject_rights_status.items())),
        "violations": [
            {
                "article": v.article,
                "severity": v.severity,
                "description": v.description,
                "remediation": v.remediation,
                "evidence_required": list(v.evidence_required),
            }
            for v in result.violations
        ],
        "recommendations": list(result.recommendations),
    }
    severities = [v.severity for v in result.violations]
    warnings = [
        f"{severities.count(level)} {level} finding(s)"
        for level in ("critical", "high", "medium", "low")
        if level in severities
    ]
    return Certificate(
        tool="gdpr_processing_check",
        schema=_SCHEMA,
        status="ok" if result.compliant else "fail",
        body=body,
        warnings=warnings,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())
