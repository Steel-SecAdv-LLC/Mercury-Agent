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

"""ISO 14971 medical-device risk governance scalar (metric-only, abstention-first).

ISO 14971:2019 frames risk as the combination of the *probability of occurrence* of
harm and the *severity* of that harm.  This module reproduces the standard's
semi-quantitative 5x5 risk-index construction (severity 1-5 x probability 1-5) for
reporting only; the standard is cited, not imported.  The scalar abstains when either
input is absent or out of the 1-5 range -- a risk index is undefined without both
coordinates, and is never defaulted.

⚠️ NOT FOR DEVICE CLEARANCE.  Descriptive measurement for audit/reporting; it does not
constitute an ISO 14971 risk-management file or any regulatory determination.
"""

from omni_mercury_engine.governance.contract import GovernanceScalar, available, unavailable

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
        return unavailable(
            "omni_iso14971_risk_index",
            family=_FAMILY,
            reason="ISO 14971 risk index undefined: severity/probability absent or not in 1-5",
        )
    risk_index = severity * probability
    return available(
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
