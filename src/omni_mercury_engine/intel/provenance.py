# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Provenance carried as a typed companion, enforced at the output boundary.

The ideal is provenance as an *unrepresentable-without-it* type: a value on a
provenance-required (hazardous) topic that simply cannot exist in the type system
unless it carries its sources. A full pipeline conversion to that is a deep,
multi-module refactor -- timeboxed to three weeks (see
``docs/PROVENANCE_MIGRATION_PLAN.md``). This module ships the **timeboxed
decision**: the ~80%-value fallback -- provenance *metadata carried through the
pipeline* and *enforced at the output boundary* -- plus the forward-looking type
seed the migration builds on.

Two representable strengths, selected by ``MERCURY_PROVENANCE_MODE``:

* ``boundary-fallback`` (default, shipped) -- a value travels with a
  :class:`Provenance` record (as a :class:`Provenanced` companion or as
  metadata carried alongside). :func:`enforce_at_boundary` refuses/redacts any
  provenance-required emission whose provenance is missing or inadequate. This is
  the ~80% of the safety value: nothing hazardous leaves uncited.
* ``type`` -- the stricter seed: the boundary *only accepts* a
  :class:`Provenanced` payload on a provenance-required topic
  (:func:`require_provenanced`), so a bare, unprovenanced value is a type error at
  the boundary rather than a runtime refusal. The remaining ~20% (compile-time,
  whole-pipeline unrepresentability) is the migration's endpoint.

"Provenance-required" reuses the shipped gate's own rule (the
``ALLOW_PROVENANCE`` disposition / high-severity hazard weight), so this does not
invent a second notion of what needs citations.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from omni_mercury_engine.cognitive.gate_audit import record_gate_decision

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Emitted in place of a provenance-required payload that arrives unprovenanced.
REFUSAL_NOTICE = "[provenance required: emission withheld — no source attribution]"


class ProvenanceMode(Enum):
    """Strength of provenance enforcement at the boundary."""

    TYPE = "type"  # boundary only accepts a Provenanced payload (type seed)
    BOUNDARY_FALLBACK = "boundary-fallback"  # metadata carried + enforced (shipped)

    @classmethod
    def from_env(cls, default: ProvenanceMode | None = None) -> ProvenanceMode:
        """Resolve the mode from ``MERCURY_PROVENANCE_MODE`` (default fallback)."""
        raw = os.environ.get("MERCURY_PROVENANCE_MODE", "").strip().lower()
        if not raw:
            return default or cls.BOUNDARY_FALLBACK
        try:
            return cls(raw)
        except ValueError:
            logger.warning(
                "unknown MERCURY_PROVENANCE_MODE=%r; defaulting to boundary-fallback", raw
            )
            return cls.BOUNDARY_FALLBACK


class ProvenanceOrigin(Enum):
    """Where a value came from (weakest-wins when provenances merge)."""

    ORACLE_VERIFIED = "oracle_verified"  # strongest: independently checked
    HUMAN = "human"
    EXTRACTIVE = "extractive"  # quoted/attributed from a source
    MODEL_GENERATED = "model_generated"  # weakest: unattributed synthesis
    SYNTHETIC = "synthetic"

    @property
    def rank(self) -> int:
        """Ordinal strength (lower = stronger); used by :meth:`Provenance.merge`."""
        return _ORIGIN_RANK[self]


_ORIGIN_RANK: dict[ProvenanceOrigin, int] = {
    ProvenanceOrigin.ORACLE_VERIFIED: 0,
    ProvenanceOrigin.HUMAN: 1,
    ProvenanceOrigin.EXTRACTIVE: 2,
    ProvenanceOrigin.MODEL_GENERATED: 3,
    ProvenanceOrigin.SYNTHETIC: 4,
}

#: Origins that constitute genuine *source attribution* -- the value was checked
#: (oracle), authored by a human, or extracted/quoted from a cited source. The
#: weaker origins (:attr:`ProvenanceOrigin.MODEL_GENERATED`,
#: :attr:`ProvenanceOrigin.SYNTHETIC`) are unattributed synthesis: the value was
#: produced by the model, not drawn from the listed sources, so attaching a
#: ``sources`` list to them is not real attribution and must not satisfy a
#: hazardous-topic boundary (otherwise a fabricated citation on the weakest origin
#: would launder synthetic content past the gate).
_ATTRIBUTED_ORIGINS: frozenset[ProvenanceOrigin] = frozenset(
    {
        ProvenanceOrigin.ORACLE_VERIFIED,
        ProvenanceOrigin.HUMAN,
        ProvenanceOrigin.EXTRACTIVE,
    }
)


@dataclass(frozen=True)
class Provenance:
    """The provenance record that travels with a value through the pipeline.

    Attributes:
        origin: How the value was produced.
        sources: Source identifiers/citations (URIs, doc ids, oracle names).
        verified: Whether the sources were independently checked (oracle/citation).
        notes: Free-text lineage notes (capped when audited).
    """

    origin: ProvenanceOrigin
    sources: tuple[str, ...] = ()
    verified: bool = False
    notes: str = ""

    def has_citations(self) -> bool:
        """True when at least one non-empty source is attached."""
        return any(s.strip() for s in self.sources)

    def is_adequate(self, *, require_verified: bool = False) -> bool:
        """Whether this provenance suffices for a provenance-required emission.

        Adequate requires an *attributed* origin (oracle/human/extractive) **and**
        at least one citation; ``require_verified`` additionally demands the
        sources were independently checked. An unattributed origin
        (model-generated / synthetic) is never adequate on its own, no matter what
        ``sources`` or ``verified`` it self-asserts -- that closes the fail-open
        where a fabricated citation on the weakest origin would pass a
        hazardous-topic boundary.
        """
        if self.origin not in _ATTRIBUTED_ORIGINS:
            return False
        if not self.has_citations():
            return False
        return self.verified if require_verified else True

    def merge(self, other: Provenance) -> Provenance:
        """Combine two provenances at a pipeline join (weakest origin wins).

        The merged origin is the *weaker* of the two (a synthesis over an
        extractive quote is model-generated), sources are unioned (order-stable,
        deduped), and ``verified`` holds only if **both** inputs were verified --
        so a pipeline step can never launder an unverified input into a verified
        output.
        """
        weaker = self.origin if self.origin.rank >= other.origin.rank else other.origin
        merged_sources = tuple(dict.fromkeys((*self.sources, *other.sources)))
        return Provenance(
            origin=weaker,
            sources=merged_sources,
            verified=self.verified and other.verified,
            notes="; ".join(n for n in (self.notes, other.notes) if n),
        )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly mapping."""
        return {
            "origin": self.origin.value,
            "sources": list(self.sources),
            "verified": self.verified,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class Provenanced(Generic[T]):  # noqa: UP046 - Generic[T] required for Python 3.11 compatibility
    """A value paired with its :class:`Provenance` -- the typed companion.

    Carrying provenance *in the type* is the seed of the unrepresentable-without-
    provenance endpoint: pipeline stages take and return :class:`Provenanced`
    values, so provenance cannot be silently dropped between stages (a stage that
    forgets it fails to type-check against the next stage's ``Provenanced``
    parameter). :meth:`map` transforms the value while preserving/annotating
    provenance; :meth:`combine` joins two provenanced values.
    """

    value: T
    provenance: Provenance

    def map(self, fn: Callable[[T], Any], *, step: str = "") -> Provenanced[Any]:
        """Transform the wrapped value, keeping provenance (optionally noting a step)."""
        prov = self.provenance
        if step:
            note = "; ".join(n for n in (prov.notes, step) if n)
            prov = Provenance(prov.origin, prov.sources, prov.verified, note)
        return Provenanced(fn(self.value), prov)

    def combine(self, other: Provenanced[Any], fn: Callable[[T, Any], Any]) -> Provenanced[Any]:
        """Join with another provenanced value, merging provenance (weakest-wins)."""
        return Provenanced(fn(self.value, other.value), self.provenance.merge(other.provenance))

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly mapping (value stringified)."""
        return {"value": self.value, "provenance": self.provenance.as_dict()}


def ensure_provenanced(  # noqa: UP047 - PEP 695 type params require Python 3.12+
    value: T, provenance: Provenance
) -> Provenanced[T]:
    """Lift a bare value into a :class:`Provenanced` companion."""
    return Provenanced(value, provenance)


def require_provenanced(payload: object) -> Provenanced[Any]:
    """Assert ``payload`` is a :class:`Provenanced` (the ``type``-mode boundary).

    This is the type-seed enforcement point: a function that accepts only
    ``Provenanced`` makes a bare value *unrepresentable* at that boundary.

    Raises:
        TypeError: if ``payload`` is not a :class:`Provenanced`.
    """
    if not isinstance(payload, Provenanced):
        raise TypeError(
            "provenance 'type' mode requires a Provenanced payload at the boundary; "
            f"got a bare {type(payload).__name__} (no provenance carried)"
        )
    return payload


# --------------------------------------------------------------------------- #
# Output-boundary enforcement.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BoundaryDecision:
    """The output boundary's disposition of a candidate emission."""

    emitted: bool
    payload: Any
    reason: str
    enforced: bool  # whether the boundary changed the emission (refused/redacted)
    provenance_required: bool
    provenance: Provenance | None = None
    mode: ProvenanceMode = field(default=ProvenanceMode.BOUNDARY_FALLBACK)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly mapping."""
        return {
            "emitted": self.emitted,
            "reason": self.reason,
            "enforced": self.enforced,
            "provenance_required": self.provenance_required,
            "mode": self.mode.value,
            "provenance": self.provenance.as_dict() if self.provenance else None,
        }


def provenance_required_for(text: str, context: dict[str, Any] | None = None) -> bool:
    """Whether emitting on ``text``'s topic requires source attribution.

    Reuses the shipped weapons/mass-casualty gate: a topic the gate dispositions
    ``ALLOW_PROVENANCE`` (a high-severity hazard domain that is otherwise
    answerable) is exactly the set that must be source-attributed. Any refusal
    disposition also implies provenance would be required were it emitted. The
    gate is fail-closed, so an internal error yields ``True`` (require provenance)
    rather than ``False``.
    """
    from omni_mercury_engine.cognitive.ethical_bounding import (
        WeaponsDisposition,
        assess_weapons_uplift,
    )

    try:
        assessment = assess_weapons_uplift(text, context)
    except Exception:  # pragma: no cover - gate is itself fail-closed
        return True
    return assessment.disposition in (
        WeaponsDisposition.ALLOW_PROVENANCE,
        WeaponsDisposition.ESCALATE,
        WeaponsDisposition.REFUSE_REDACT,
        WeaponsDisposition.HARD_REFUSE,
    )


def enforce_at_boundary(
    payload: Any,
    *,
    text: str | None = None,
    provenance_required: bool | None = None,
    provenance: Provenance | None = None,
    mode: ProvenanceMode | None = None,
    require_verified: bool = False,
    source: str = "output_boundary",
) -> BoundaryDecision:
    """Enforce provenance on a candidate emission at the output boundary.

    Args:
        payload: The value to emit -- either a :class:`Provenanced` (provenance
            read from it) or a bare value (provenance supplied via ``provenance``).
        text: The emission text used to decide whether provenance is required
            (via :func:`provenance_required_for`); ignored if
            ``provenance_required`` is given explicitly.
        provenance_required: Explicit override of the required-ness decision.
        provenance: Provenance for a bare ``payload`` (ignored if ``payload`` is a
            :class:`Provenanced`).
        mode: Enforcement strength (defaults to ``MERCURY_PROVENANCE_MODE``).
        require_verified: Demand independently-verified sources, not just cited.
        source: Audit source label.

    Returns:
        A :class:`BoundaryDecision`. When provenance is required and inadequate,
        the emission is withheld (``emitted=False``, payload replaced by
        :data:`REFUSAL_NOTICE`) and the decision is durably audited.
    """
    active_mode = mode or ProvenanceMode.from_env()

    # Resolve the carried provenance.
    carried: Provenance | None
    bare_value: Any
    if isinstance(payload, Provenanced):
        carried = payload.provenance
        bare_value = payload.value
    else:
        carried = provenance
        bare_value = payload

    # Decide whether provenance is required. With no topic signal at all (neither
    # ``text`` to assess nor an explicit ``provenance_required``) the boundary
    # cannot rule the emission benign, so it fails **closed** (requires provenance)
    # rather than silently emitting -- a caller that means "not required" must say
    # so explicitly.
    if provenance_required is None:
        required = provenance_required_for(text) if text is not None else True
    else:
        required = bool(provenance_required)

    if not required:
        return BoundaryDecision(
            emitted=True,
            payload=bare_value,
            reason="provenance not required for this emission",
            enforced=False,
            provenance_required=False,
            provenance=carried,
            mode=active_mode,
        )

    # type mode: a provenance-required emission MUST arrive as a Provenanced.
    if active_mode is ProvenanceMode.TYPE and not isinstance(payload, Provenanced):
        _audit_refusal(source, "type-mode: bare value on provenance-required topic")
        return BoundaryDecision(
            emitted=False,
            payload=REFUSAL_NOTICE,
            reason="type mode requires a Provenanced payload; bare value refused",
            enforced=True,
            provenance_required=True,
            provenance=None,
            mode=active_mode,
        )

    if carried is None or not carried.is_adequate(require_verified=require_verified):
        detail = (
            "no provenance carried"
            if carried is None
            else f"inadequate provenance (verified={carried.verified}, "
            f"cited={carried.has_citations()})"
        )
        _audit_refusal(source, detail)
        return BoundaryDecision(
            emitted=False,
            payload=REFUSAL_NOTICE,
            reason=f"provenance required but {detail}; emission withheld",
            enforced=True,
            provenance_required=True,
            provenance=carried,
            mode=active_mode,
        )

    return BoundaryDecision(
        emitted=True,
        payload=bare_value,
        reason="provenance required and adequate; emitted with attribution",
        enforced=False,
        provenance_required=True,
        provenance=carried,
        mode=active_mode,
    )


def _audit_refusal(source: str, detail: str) -> None:
    """Durably audit a provenance-boundary refusal."""
    record_gate_decision(
        decision="provenance_withheld",
        source=source,
        disposition="refuse_redact",
        signals=("provenance", "output_boundary"),
        reason=f"provenance-required emission withheld: {detail}",
    )


def boundary_enforcement_rate(
    emissions: list[tuple[Any, Provenance | None]],
    *,
    provenance_required: bool = True,
    mode: ProvenanceMode | None = None,
    require_verified: bool = False,
) -> float:
    """Fraction of provenance-required, unprovenanced emissions withheld (value metric).

    ``emissions`` is ``(payload, provenance)`` pairs, all on a provenance-required
    topic. A correct boundary withholds every one that lacks adequate provenance,
    yielding ``1.0``; an emission that *does* carry adequate provenance is not
    counted against the rate (it should be emitted).
    """
    considered = 0
    enforced = 0
    for payload, prov in emissions:
        decision = enforce_at_boundary(
            payload,
            provenance_required=provenance_required,
            provenance=prov,
            mode=mode,
            require_verified=require_verified,
        )
        adequate = prov is not None and prov.is_adequate(require_verified=require_verified)
        if adequate and not isinstance(payload, Provenanced):
            # Legitimately emittable; not part of the enforcement denominator.
            if (mode or ProvenanceMode.from_env()) is not ProvenanceMode.TYPE:
                continue
        considered += 1
        if not decision.emitted:
            enforced += 1
    return enforced / considered if considered else 0.0


__all__ = [
    "REFUSAL_NOTICE",
    "BoundaryDecision",
    "Provenance",
    "ProvenanceMode",
    "ProvenanceOrigin",
    "Provenanced",
    "boundary_enforcement_rate",
    "enforce_at_boundary",
    "ensure_provenanced",
    "provenance_required_for",
    "require_provenanced",
]
