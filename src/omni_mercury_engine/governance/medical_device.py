# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""ISO 14971 medical-device risk governance scalar (metric-only, abstention-first).

ISO 14971:2019 frames risk as the combination of the *probability of occurrence* of
harm and the *severity* of that harm.  This module reproduces the standard's
semi-quantitative 5x5 risk-index construction (severity 1-5 x probability 1-5) for
reporting only; the standard is cited, not imported.

The family vets **UNAVAILABLE-capable**: the engine emits runtime severity/confidence
coordinates (``core/types.py:92`` ``ThreatLevel``; ``security/realtime_threat_detection.py``
``ThreatSignature.severity``/``confidence``) from which a severity/probability pair can be
sourced, so the scalar is GROUNDED when both coordinates are present and UNAVAILABLE when
either is absent or out of the 1-5 range -- a risk index is undefined without both
coordinates, and is never defaulted.

⚠️ NOT FOR DEVICE CLEARANCE.  Descriptive measurement for audit/reporting; it does not
constitute an ISO 14971 risk-management file or any regulatory determination.
"""

from __future__ import annotations

from omni_mercury_engine.governance.contract import GovernanceScalar, grounded, unavailable

_FAMILY = "iso14971"
_RISK_INDEX_MAX = 25.0  # severity (max 5) x probability (max 5)


def _coerce_level(value: object) -> int | None:
    """Coerce a 1-5 severity/probability level; return ``None`` if invalid."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 1 <= value <= 5 else None
    return None


def _tier(risk_index: int) -> str:
    """Map a 1-25 risk index onto the conventional ISO 14971 acceptability bands."""
    if risk_index <= 4:
        return "acceptable"
    if risk_index <= 12:
        return "alarp"
    return "unacceptable"


def iso14971_risk_scalar(inputs: dict[str, object]) -> GovernanceScalar:
    """Compute the ISO 14971 risk index from ``severity`` and ``probability`` (each 1-5).

    Args:
        inputs: Mapping that must contain integer ``severity`` and ``probability`` in
            ``[1, 5]``.  Any missing or out-of-range coordinate abstains.

    Returns:
        A scalar whose value is ``severity * probability / 25`` with the acceptability
        tier in its provenance, or an abstention when either coordinate is unavailable.
    """
    severity = _coerce_level(inputs.get("severity"))
    probability = _coerce_level(inputs.get("probability"))
    if severity is None or probability is None:
        missing = tuple(
            k for k, v in (("severity", severity), ("probability", probability)) if v is None
        )
        return unavailable(
            "omni_iso14971_risk_index",
            family=_FAMILY,
            reason="ISO 14971 risk index undefined: severity/probability absent or not in 1-5",
            missing_inputs=missing,
        )
    risk_index = severity * probability
    return grounded(
        "omni_iso14971_risk_index",
        risk_index / _RISK_INDEX_MAX,
        family=_FAMILY,
        reason=f"ISO 14971 risk index = {risk_index}/25 ({_tier(risk_index)})",
        provenance={
            "severity": severity,
            "probability": probability,
            "risk_index": risk_index,
            "tier": _tier(risk_index),
        },
    )
