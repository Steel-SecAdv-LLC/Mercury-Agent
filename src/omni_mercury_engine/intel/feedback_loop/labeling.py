# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Turn audit events and human overrides into human-verified labeled examples.

The closed loop only ever learns from labels a human stands behind. This module
is the labeling front door:

* :func:`ingest_audit_event` parses one durable gate-audit record (written by
  :mod:`omni_mercury_engine.cognitive.gate_audit`) into an :class:`AuditEvent`.
* :func:`apply_human_label` attaches a human's verified label to an audit event,
  and :func:`override_to_example` records a direct human override of a gate
  decision -- both producing a :class:`LabeledExample`.

Every path is **fail-closed on the human**: a label with no reviewer, or an
invalid label/expected pairing, raises rather than entering the queue. An
anonymous or malformed label must never become training data -- that is the
data-provenance half of the poisoning defense (the regression gate is the other
half).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

_VALID_LABELS = {"offensive", "benign"}
_LABEL_TO_EXPECTED = {"offensive": "block", "benign": "allow"}


class ExampleSource(Enum):
    """Where a labeled example originated."""

    AUDIT_EVENT = "audit_event"  # a gate decision a human reviewed + labeled
    HUMAN_OVERRIDE = "human_override"  # a human corrected a gate decision directly
    RED_TEAM = "red_team"  # a triaged red-team survivor (corpus/pending)


@dataclass(frozen=True)
class AuditEvent:
    """A parsed gate-audit record (the raw material for a labeled example)."""

    decision: str
    source: str
    disposition: str
    hazard_domain: str
    intent: str
    query: str
    reason: str
    ts: float
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> AuditEvent:
        """Build an :class:`AuditEvent` from a gate-audit JSONL record.

        Raises:
            ValueError: if the record carries no ``query`` text -- an event
                without the action text cannot be labeled into an example.
        """
        query = record.get("query")
        if not query or not str(query).strip():
            raise ValueError(
                "audit record has no 'query' text to label; enable query capture "
                "in the gate audit or supply the text via override_to_example"
            )
        return cls(
            decision=str(record.get("decision", "")),
            source=str(record.get("source", "")),
            disposition=str(record.get("disposition", "")),
            hazard_domain=str(record.get("hazard_domain", "none")),
            intent=str(record.get("intent", "mechanism")),
            query=str(query),
            reason=str(record.get("reason", "")),
            ts=float(record.get("ts", 0.0)),
            raw=dict(record),
        )


@dataclass(frozen=True)
class LabeledExample:
    """A human-verified labeled example destined for the feedback queue.

    Attributes:
        text: The action/query text.
        label: ``"offensive"`` or ``"benign"``.
        source: How it entered the loop.
        reviewer: The human who verified the label (required, non-empty).
        reason: The reviewer's rationale.
        origin_ref: A back-reference to the source (audit source, red-team id).
    """

    text: str
    label: str
    source: ExampleSource
    reviewer: str
    reason: str = ""
    origin_ref: str = ""

    def __post_init__(self) -> None:
        """Fail closed on an unlabeled/anonymous example (no silent bad data)."""
        if self.label not in _VALID_LABELS:
            raise ValueError(f"label must be one of {sorted(_VALID_LABELS)}; got {self.label!r}")
        if not self.text.strip():
            raise ValueError("labeled example has empty text")
        if not self.reviewer.strip():
            raise ValueError(
                "a labeled example requires a non-empty reviewer id; anonymous "
                "labels are refused (data-provenance / poisoning defense)"
            )

    @property
    def expected(self) -> str:
        """The gate disposition this label expects (``block``/``allow``)."""
        return _LABEL_TO_EXPECTED[self.label]

    def as_corpus_row(self) -> dict[str, Any]:
        """Render as a weapons-gate corpus row (``split='pending'`` until promoted)."""
        return {
            "text": self.text,
            "label": self.label,
            "expected": self.expected,
            "split": "pending",
            "tags": ["feedback_loop", self.source.value],
        }

    def as_dict(self) -> dict[str, Any]:
        """Return the full JSON-friendly record (queue storage form)."""
        return {
            "text": self.text,
            "label": self.label,
            "source": self.source.value,
            "reviewer": self.reviewer,
            "reason": self.reason,
            "origin_ref": self.origin_ref,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LabeledExample:
        """Rebuild a :class:`LabeledExample` from its stored mapping."""
        return cls(
            text=str(data["text"]),
            label=str(data["label"]),
            source=ExampleSource(data.get("source", "human_override")),
            reviewer=str(data.get("reviewer", "")),
            reason=str(data.get("reason", "")),
            origin_ref=str(data.get("origin_ref", "")),
        )


def ingest_audit_event(record: dict[str, Any]) -> AuditEvent:
    """Parse a gate-audit JSONL record into an :class:`AuditEvent`."""
    return AuditEvent.from_record(record)


def apply_human_label(
    event: AuditEvent,
    *,
    label: str,
    reviewer: str,
    reason: str = "",
) -> LabeledExample:
    """Attach a human-verified ``label`` to an audit ``event``.

    This is the human-in-the-loop labeling step: a reviewer looks at what the
    gate decided and records the correct label. The reviewer id is mandatory.
    """
    return LabeledExample(
        text=event.query,
        label=label,
        source=ExampleSource.AUDIT_EVENT,
        reviewer=reviewer,
        reason=reason or f"labeled from audit decision={event.decision}",
        origin_ref=f"{event.source}@{event.ts}",
    )


def override_to_example(
    text: str,
    *,
    label: str,
    reviewer: str,
    reason: str = "",
    source: ExampleSource = ExampleSource.HUMAN_OVERRIDE,
    origin_ref: str = "",
) -> LabeledExample:
    """Record a direct human override / triaged item as a labeled example."""
    return LabeledExample(
        text=text,
        label=label,
        source=source,
        reviewer=reviewer,
        reason=reason,
        origin_ref=origin_ref,
    )


__all__ = [
    "AuditEvent",
    "ExampleSource",
    "LabeledExample",
    "apply_human_label",
    "ingest_audit_event",
    "override_to_example",
]
