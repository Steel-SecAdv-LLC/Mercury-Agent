"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""AI-assurance conformance scalars (metric-only) -- abstain unless attested.

NIST AI RMF 1.0, the OWASP LLM Top 10 (2025), and MITRE ATLAS are **checklists**, not
quantities.  An anomaly engine has no intrinsic runtime signal that yields, say, a NIST
"GOVERN conformance" of 0.73.  Per the project's honesty contract this module therefore
**abstains by default**: a conformance scalar is produced only from an explicit operator
*attestation* (a mapping of catalog item -> satisfied bool).  With no attestation the
whole family registers nothing -- exactly as a Collatz instance over budget is
``inconclusive`` and a Lean tier with no toolchain is ``unavailable``.

When an attestation is supplied the value is an honest ``satisfied / assessed`` fraction
over the catalog items the attestation actually covers; coverage is recorded in
provenance so a partial attestation can never masquerade as full conformance.  The
catalogs below are the published item identifiers, cited not imported.
"""

from typing import TYPE_CHECKING

from omni_mercury_engine.governance.contract import GovernanceScalar, available, unavailable

if TYPE_CHECKING:
    from collections.abc import Mapping

# NIST AI RMF 1.0 core functions.
_NIST_AI_RMF: tuple[str, ...] = ("govern", "map", "measure", "manage")

# OWASP Top 10 for LLM Applications (2025).
_OWASP_LLM: tuple[str, ...] = (
    "llm01_prompt_injection",
    "llm02_sensitive_information_disclosure",
    "llm03_supply_chain",
    "llm04_data_and_model_poisoning",
    "llm05_improper_output_handling",
    "llm06_excessive_agency",
    "llm07_system_prompt_leakage",
    "llm08_vector_and_embedding_weaknesses",
    "llm09_misinformation",
    "llm10_unbounded_consumption",
)

# MITRE ATLAS adversarial-ML tactics.
_MITRE_ATLAS: tuple[str, ...] = (
    "reconnaissance",
    "resource_development",
    "initial_access",
    "ml_model_access",
    "execution",
    "persistence",
    "privilege_escalation",
    "defense_evasion",
    "credential_access",
    "discovery",
    "collection",
    "ml_attack_staging",
    "exfiltration",
    "impact",
)


def _conformance(
    scalar_name: str,
    family: str,
    catalog: tuple[str, ...],
    attestation: Mapping[str, bool] | None,
    *,
    label: str,
) -> GovernanceScalar:
    """Build a conformance scalar from an attestation, or abstain when none is given."""
    if not attestation:
        return unavailable(
            scalar_name,
            family=family,
            reason=f"{label}: no attestation supplied (checklist has no runtime signal)",
        )
    assessed = [item for item in catalog if item in attestation]
    if not assessed:
        return unavailable(
            scalar_name,
            family=family,
            reason=f"{label}: attestation covers none of the {len(catalog)} catalog items",
        )
    satisfied = sum(1 for item in assessed if attestation[item])
    coverage = len(assessed) / len(catalog)
    return available(
        scalar_name,
        satisfied / len(assessed),
        family=family,
        reason=(
            f"{label}: {satisfied}/{len(assessed)} assessed items satisfied "
            f"(coverage {coverage:.0%} of {len(catalog)})"
        ),
        provenance={
            "satisfied": satisfied,
            "assessed": len(assessed),
            "catalog_size": len(catalog),
            "coverage": coverage,
        },
    )


def ai_safety_scalars(
    *,
    nist_ai_rmf: Mapping[str, bool] | None = None,
    owasp_llm: Mapping[str, bool] | None = None,
    mitre_atlas: Mapping[str, bool] | None = None,
) -> list[GovernanceScalar]:
    """Build the three AI-assurance conformance scalars from optional attestations.

    Each argument is an operator attestation mapping catalog item -> satisfied bool.
    Any argument left ``None`` (the default) makes that scalar abstain, so the whole
    family registers nothing unless real evidence is supplied.

    Args:
        nist_ai_rmf: Attestation over :data:`_NIST_AI_RMF` functions.
        owasp_llm: Attestation over the :data:`_OWASP_LLM` Top 10 items.
        mitre_atlas: Attestation over :data:`_MITRE_ATLAS` tactics.

    Returns:
        Three :class:`GovernanceScalar` objects (available only where attested).
    """
    return [
        _conformance(
            "omni_nist_airmf_conformance",
            "nist_ai_rmf",
            _NIST_AI_RMF,
            nist_ai_rmf,
            label="NIST AI RMF 1.0",
        ),
        _conformance(
            "omni_owasp_llm_mitigation",
            "owasp_llm",
            _OWASP_LLM,
            owasp_llm,
            label="OWASP LLM Top 10 (2025)",
        ),
        _conformance(
            "omni_mitre_atlas_coverage",
            "mitre_atlas",
            _MITRE_ATLAS,
            mitre_atlas,
            label="MITRE ATLAS",
        ),
    ]
