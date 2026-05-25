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

"""Abstention-first contract for descriptive (metric-only) governance scalars.

Governance, medical, and AI-assurance frameworks *describe* a system; they do not
drive the σ_Immutable decision boundary.  Every scalar defined under this contract
is therefore **metric-only** -- excluded from the operational vector by
:meth:`GlobalOmniScalarNetwork._is_metric_only_scalar` -- and obeys the same honesty
rule the :mod:`omni_mercury_engine.verifiers` oracles obey: a scalar is produced only
when its defined input signal is actually present and its published formula is
computable.  When the signal is absent the scalar is :data:`ScalarStatus.UNAVAILABLE`
and registers nothing; it is never defaulted to 0.0, 1.0, or any placeholder.

The mapping to the verifier contract is exact: ``confirmed``/``refuted`` ground a
verifier scalar, ``inconclusive``/``unavailable`` ground nothing; here ``available``
grounds a governance scalar and ``unavailable`` grounds nothing.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

from omni_mercury_engine.core.global_omni_scalar_network import (
    GlobalOmniScalarNetwork,
    ScalarGroup,
)

logger = logging.getLogger(__name__)


class ScalarStatus(Enum):
    """Whether a governance scalar's input signal was present and computable."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class GovernanceScalar:
    """A single descriptive governance measurement, or an honest abstention.

    Attributes:
        name: Metric-only scalar key (must match a ``_METRIC_ONLY_PREFIXES`` entry).
        family: Framework family this scalar belongs to (e.g. ``"sofa"``).
        status: ``AVAILABLE`` when computed from a present signal, else ``UNAVAILABLE``.
        value: Measurement in ``[0, 1]`` when available, else ``None``.
        reason: Human-readable provenance or abstention reason.
        provenance: JSON-friendly structured context (inputs used, formula identifier).
    """

    name: str
    family: str
    status: ScalarStatus
    value: float | None
    reason: str
    provenance: dict[str, object] = field(default_factory=dict)

    @property
    def is_available(self) -> bool:
        """Whether this scalar was computed from a present input signal."""
        return self.status is ScalarStatus.AVAILABLE

    def as_metadata(self) -> dict[str, object]:
        """Return a JSON-friendly mapping describing this scalar."""
        return {
            "name": self.name,
            "family": self.family,
            "status": self.status.value,
            "value": self.value,
            "reason": self.reason,
            **self.provenance,
        }


def available(
    name: str,
    value: float,
    *,
    family: str,
    reason: str,
    provenance: dict[str, object] | None = None,
) -> GovernanceScalar:
    """Build an ``AVAILABLE`` scalar, clamping ``value`` into ``[0, 1]``."""
    clamped = max(0.0, min(1.0, float(value)))
    return GovernanceScalar(
        name=name,
        family=family,
        status=ScalarStatus.AVAILABLE,
        value=clamped,
        reason=reason,
        provenance=provenance or {},
    )


def unavailable(
    name: str,
    *,
    family: str,
    reason: str,
    provenance: dict[str, object] | None = None,
) -> GovernanceScalar:
    """Build an ``UNAVAILABLE`` scalar (an abstention): no value, registers nothing."""
    return GovernanceScalar(
        name=name,
        family=family,
        status=ScalarStatus.UNAVAILABLE,
        value=None,
        reason=reason,
        provenance=provenance or {},
    )


@dataclass(frozen=True)
class GovernanceLedgerEntry:
    """Provenance record for one adjudicated governance scalar."""

    name: str
    family: str
    status: str
    value: float | None
    reason: str
    registered: bool
    timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly mapping for this ledger entry."""
        return {
            "name": self.name,
            "family": self.family,
            "status": self.status,
            "value": self.value,
            "reason": self.reason,
            "registered": self.registered,
        }


class GovernanceRegistry:
    """Registers only *available*, *metric-only* governance scalars into the GOSNN.

    The registry enforces two invariants so the upgrade can never perturb the trained
    σ_Immutable gate:

    * **Abstention:** an ``UNAVAILABLE`` scalar registers nothing (ledger-only).
    * **Metric-only:** a scalar whose key is not recognised as metric-only by
      :meth:`GlobalOmniScalarNetwork._is_metric_only_scalar` is rejected with a
      :class:`ValueError` rather than silently inflating the operational vector.
    """

    def __init__(self, gosnn: GlobalOmniScalarNetwork) -> None:
        """Bind the registry to a GOSNN singleton instance."""
        self.gosnn = gosnn
        self.ledger: list[GovernanceLedgerEntry] = []

    def register(
        self,
        scalar: GovernanceScalar,
        *,
        group: ScalarGroup,
        component_name: str,
    ) -> GovernanceLedgerEntry:
        """Register one scalar if available and metric-only; else record an abstention."""
        registered = False
        if scalar.is_available and scalar.value is not None:
            if not GlobalOmniScalarNetwork._is_metric_only_scalar(scalar.name):
                raise ValueError(
                    f"governance scalar {scalar.name!r} is not metric-only; registering it "
                    "would inflate the σ_Immutable operational vector"
                )
            self.gosnn.register_scalars(
                component_name=component_name,
                scalars={scalar.name: scalar.value},
                group=group,
                metadata={"governance_family": scalar.family, **scalar.provenance},
            )
            registered = True
        else:
            logger.info(
                "governance scalar %s abstained (%s); nothing registered",
                scalar.name,
                scalar.reason,
            )
        entry = GovernanceLedgerEntry(
            name=scalar.name,
            family=scalar.family,
            status=scalar.status.value,
            value=scalar.value if registered else None,
            reason=scalar.reason,
            registered=registered,
        )
        self.ledger.append(entry)
        return entry

    def register_all(
        self,
        scalars: list[GovernanceScalar],
        *,
        group: ScalarGroup,
        component_name: str,
    ) -> list[GovernanceLedgerEntry]:
        """Register every available scalar in ``scalars`` (abstentions stay ledger-only)."""
        return [self.register(s, group=group, component_name=component_name) for s in scalars]

    def summary(self) -> dict[str, object]:
        """Return registration/abstention counts and the current operational vector size."""
        by_status: dict[str, int] = {}
        by_family: dict[str, int] = {}
        for entry in self.ledger:
            by_status[entry.status] = by_status.get(entry.status, 0) + 1
            by_family[entry.family] = by_family.get(entry.family, 0) + 1
        return {
            "total": len(self.ledger),
            "registered": sum(1 for e in self.ledger if e.registered),
            "abstained": sum(1 for e in self.ledger if not e.registered),
            "by_status": by_status,
            "by_family": by_family,
            "operational_scalar_count": len(self.gosnn._collect_all_scalars()),
        }
