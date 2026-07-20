# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared clinical safety primitives for the medical subsystem.

Every medical result that reaches a user or provider must carry, at minimum:

* an honest **decision-support disclaimer** (not a diagnosis, not a substitute
  for a licensed clinician) attached to the *result object*, not just a
  docstring;
* explicit **red-flag emergency routing** — when findings can indicate a
  life-threatening emergency, the envelope says so and tells a lay reader to
  seek emergency care now, rather than emitting only clinician directives;
* **provenance** — the instrument/model and version that produced the result
  and a hash of the exact inputs, so a provider can reproduce and audit it;
* the set of inputs that were **unassessed** (absent), so a missing value is
  never silently read as "normal".

This module is deliberately dependency-light (stdlib only) so every clinical
module can attach a :class:`ClinicalSafetyEnvelope` without pulling heavy
machinery. The design mirrors the abstention contract in
``governance/contract.py`` (GROUNDED / UNAVAILABLE) but at the *result* layer
that a caller actually consumes.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "CLINICAL_DECISION_SUPPORT_DISCLAIMER",
    "EMERGENCY_GUIDANCE",
    "ClinicalSafetyEnvelope",
    "build_provenance",
    "hash_inputs",
]

#: Attached to every medical result. Public-safe, honest, and non-alarming.
CLINICAL_DECISION_SUPPORT_DISCLAIMER = (
    "Mercury Agent medical analysis is clinical decision-support — not a "
    "diagnosis and not a substitute for evaluation by a licensed clinician. "
    "Results are computed only from the data you provide; any missing value is "
    "reported as unassessed, never assumed normal. Do not start, stop, or "
    "change any treatment based on this output without a qualified professional."
)

#: Shown whenever the envelope is flagged as a possible emergency.
EMERGENCY_GUIDANCE = (
    "These findings can indicate a life-threatening emergency. Seek emergency "
    "medical care now — call your local emergency number or go to the nearest "
    "emergency department. Do not wait for further analysis."
)


def hash_inputs(inputs: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 hex digest over the provided clinical inputs.

    Used for provenance/audit so a result can be tied to the exact inputs that
    produced it without storing the (potentially sensitive) values themselves.
    Falls back to a deterministic ``repr`` when a value is not JSON-serialisable.
    """
    try:
        blob = json.dumps(dict(inputs), sort_keys=True, default=str)
    except TypeError:
        blob = repr(sorted((str(k), repr(v)) for k, v in inputs.items()))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_provenance(
    *,
    instrument: str,
    version: str,
    inputs: Mapping[str, Any],
    model: str | None = None,
    model_fitted: bool | None = None,
) -> dict[str, Any]:
    """Build a provenance record for one clinical computation.

    Args:
        instrument: Instrument/algorithm identifier (e.g. ``"SOFA"``).
        version: Instrument implementation version (e.g. ``"sepsis-3"``).
        inputs: The exact inputs used (hashed, not stored verbatim).
        model: Optional ML model identifier when a learned model contributed.
        model_fitted: Whether that model carried trained weights (never ``True``
            for a randomly-initialised network).

    Returns:
        A JSON-friendly provenance mapping including a wall-clock timestamp.
    """
    prov: dict[str, Any] = {
        "instrument": instrument,
        "instrument_version": version,
        "input_sha256": hash_inputs(inputs),
        "computed_at": time.time(),
    }
    if model is not None:
        prov["model"] = model
        prov["model_fitted"] = bool(model_fitted)
    return prov


@dataclass
class ClinicalSafetyEnvelope:
    """Safety metadata attached to every user/provider-facing medical result.

    Attributes:
        disclaimer: The decision-support disclaimer (always present).
        is_decision_support: Always ``True`` — Mercury supports, never replaces.
        is_diagnosis: Always ``False`` — outputs are not diagnoses.
        emergency: Whether the result carries a life-threatening red flag.
        emergency_guidance: Lay-reader emergency instruction when ``emergency``.
        red_flags: Human-readable reasons the emergency flag fired.
        unassessed_inputs: Inputs that were absent (reported, never defaulted).
        provenance: Instrument/model + input-hash provenance for audit.
    """

    disclaimer: str = CLINICAL_DECISION_SUPPORT_DISCLAIMER
    is_decision_support: bool = True
    is_diagnosis: bool = False
    emergency: bool = False
    emergency_guidance: str | None = None
    red_flags: list[str] = field(default_factory=list)
    unassessed_inputs: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def flag_emergency(self, reason: str) -> None:
        """Mark this result as a possible emergency with a lay-reader routing."""
        self.emergency = True
        if reason not in self.red_flags:
            self.red_flags.append(reason)
        self.emergency_guidance = EMERGENCY_GUIDANCE

    def note_unassessed(self, names: list[str]) -> None:
        """Record inputs that were absent this run (deduplicated, order-stable)."""
        for name in names:
            if name not in self.unassessed_inputs:
                self.unassessed_inputs.append(name)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of the envelope."""
        return {
            "disclaimer": self.disclaimer,
            "is_decision_support": self.is_decision_support,
            "is_diagnosis": self.is_diagnosis,
            "emergency": self.emergency,
            "emergency_guidance": self.emergency_guidance,
            "red_flags": list(self.red_flags),
            "unassessed_inputs": list(self.unassessed_inputs),
            "provenance": dict(self.provenance),
        }
