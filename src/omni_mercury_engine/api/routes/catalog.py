# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Catalog routes: what every domain and detector is, and whose knowledge it carries.

Read-only and unauthenticated, like the agent notice and the source offer:
these routes publish the engine's own declarations -- the detector manifest,
the anomaly ontology, each loader's declared contract and the knowledge
profiles (discipline, professions, institutions, lineage, references) -- so
the console, the agent and any client describe a domain or detector from the
same source instead of a template. Nothing here reads request input beyond a
path parameter, and nothing changes state.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException

from omni_mercury_engine.core.detector_ontology import ONTOLOGY
from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST
from omni_mercury_engine.core.knowledge_profiles import (
    DETECTOR_PROFILES,
    DOMAIN_PROFILES,
    profile_dict,
)
from omni_mercury_engine.core.loader_declarations import loader_declarations
from omni_mercury_engine.detectors.hazard_diagnostics import KNOWN_HAZARDS

__all__ = ["detector_record", "domain_record", "router"]

router = APIRouter(prefix="/api/v1/catalog", tags=["Catalog"])


def detector_record(name: str) -> dict[str, Any] | None:
    """The manifest entry, ontology row and knowledge profile of one detector."""
    entry = next((e for e in DETECTOR_MANIFEST if e.name == name), None)
    if entry is None:
        return None
    ontology = ONTOLOGY.get(name)
    profile = DETECTOR_PROFILES.get(name)
    return {
        "name": entry.name,
        "class_name": entry.class_name,
        "module": entry.module_path,
        "category": entry.category.value,
        "description": entry.description,
        "modality": entry.modality,
        "seams": list(entry.seams),
        "fit_required": entry.fit_required,
        "external": entry.external,
        "covered_by": entry.covered_by,
        "wraps": entry.wraps,
        "tags": list(entry.tags),
        "ontology": (
            {
                "detects": list(ontology.detects),
                "cannot_detect": list(ontology.cannot_detect),
                "validated": [asdict(v) for v in ontology.validated],
                "not_validated_note": ontology.not_validated_note,
            }
            if ontology
            else None
        ),
        "profile": profile_dict(profile) if profile else None,
    }


def domain_record(domain: str) -> dict[str, Any] | None:
    """The loader declaration, hazard flag and knowledge profile of one domain."""
    declaration = loader_declarations().get(domain)
    hazard = domain in KNOWN_HAZARDS
    profile = DOMAIN_PROFILES.get(domain)
    if declaration is None and not hazard and profile is None:
        return None
    detectors = list(profile.detectors) if profile else []
    return {
        "domain": domain,
        "hazard_viz": hazard,
        "loader": asdict(declaration) if declaration else None,
        "related_detectors": sorted(set(detectors)),
        "profile": profile_dict(profile) if profile else None,
    }


def _all_domains() -> list[str]:
    return sorted(set(loader_declarations()) | set(KNOWN_HAZARDS) | set(DOMAIN_PROFILES))


@router.get(
    "/domains",
    summary="Every data domain: loader contract, hazard flag and knowledge profile",
    description=(
        "One record per data-loader domain and hazard-visualisation domain: the loader's "
        "declared contract (source, key, label provenance, feature columns), whether "
        "`hazard-viz` renders it, the detectors that serve it, and its knowledge profile "
        "(discipline, professions, institutions, expertise, method, limits, references)."
    ),
)
def list_domains() -> dict[str, Any]:
    """List every domain record.

    Returns:
        ``{"count": n, "domains": [...]}``.
    """
    records = [r for r in (domain_record(d) for d in _all_domains()) if r is not None]
    return {"count": len(records), "domains": records}


@router.get(
    "/domains/{domain}",
    summary="One data domain",
    responses={404: {"description": "No such domain"}},
)
def get_domain(domain: str) -> dict[str, Any]:
    """One domain record.

    Args:
        domain: A loader ``DOMAIN`` or a ``hazard-viz`` hazard name.

    Returns:
        The domain record.

    Raises:
        HTTPException: 404 when the domain is unknown.
    """
    record = domain_record(domain)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No such domain: {domain}")
    return record


@router.get(
    "/detectors",
    summary="Every manifest detector: entry, ontology and knowledge profile",
    description=(
        "One record per `DETECTOR_MANIFEST` entry: module, class, category, modality, "
        "seams, the anomaly ontology (detects, cannot detect, validated evidence) and the "
        "knowledge profile (discipline, lineage, mechanism, practitioners, when to use it, "
        "when not to, references)."
    ),
)
def list_detectors() -> dict[str, Any]:
    """List every detector record.

    Returns:
        ``{"count": n, "detectors": [...]}``.
    """
    records = [r for r in (detector_record(e.name) for e in DETECTOR_MANIFEST) if r is not None]
    return {"count": len(records), "detectors": records}


@router.get(
    "/detectors/{name}",
    summary="One manifest detector",
    responses={404: {"description": "No such detector"}},
)
def get_detector(name: str) -> dict[str, Any]:
    """One detector record.

    Args:
        name: The manifest name (``bocpd``, ``sepsis``, ...).

    Returns:
        The detector record.

    Raises:
        HTTPException: 404 when the name is not in the manifest.
    """
    record = detector_record(name)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No such detector: {name}")
    return record
