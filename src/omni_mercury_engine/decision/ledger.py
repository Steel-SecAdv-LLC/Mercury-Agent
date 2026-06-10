# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Append-only audit ledger of decisions -- the "verify" and "detail" pillar.

:class:`DecisionLedger` is the closed loop's memory: an append-only, JSON-
serialisable trail of every
:class:`~omni_mercury_engine.decision.record.DecisionRecord` the responder
emits, with an aggregate :meth:`summary` over the honesty states, dispositions
and responses seen.  It turns a stream of one-shot decisions into a verifiable
record -- what was decided, why, how the loop responded, and how often it
honestly abstained.

The ledger is bounded by an optional ``maxlen`` (a ring buffer), so an engine
that runs indefinitely cannot grow the trail without limit -- recording stays
opt-in and memory-safe.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from omni_mercury_engine.decision.record import DecisionRecord


class DecisionLedger:
    """An append-only, optionally-bounded trail of decision records.

    Args:
        maxlen: Optional ring-buffer cap.  When set, only the most recent
            ``maxlen`` records are retained (the oldest are evicted as new ones
            arrive); ``None`` keeps every record.
    """

    def __init__(self, *, maxlen: int | None = None) -> None:
        """Create an empty ledger, optionally bounded to ``maxlen`` records."""
        if maxlen is not None and maxlen <= 0:
            raise ValueError(f"maxlen must be a positive int or None, got {maxlen}")
        self.maxlen = maxlen
        self._entries: deque[DecisionRecord] = deque(maxlen=maxlen)

    def record(self, record: DecisionRecord) -> DecisionRecord:
        """Append one decision and return it (so callers can chain)."""
        self._entries.append(record)
        return record

    def extend(self, records: Iterable[DecisionRecord]) -> None:
        """Append many decisions, in order."""
        self._entries.extend(records)

    @property
    def entries(self) -> tuple[DecisionRecord, ...]:
        """An immutable snapshot of the recorded decisions, oldest first."""
        return tuple(self._entries)

    def clear(self) -> None:
        """Drop all recorded decisions."""
        self._entries.clear()

    def __len__(self) -> int:
        """Return the number of recorded decisions."""
        return len(self._entries)

    def __iter__(self) -> Iterator[DecisionRecord]:
        """Iterate the recorded decisions, oldest first."""
        return iter(self.entries)

    def to_list(self) -> list[dict[str, Any]]:
        """Return the ledger as a list of JSON-safe decision dicts."""
        return [record.to_dict() for record in self._entries]

    def summary(self) -> dict[str, Any]:
        """Return aggregate counts and rates over the recorded decisions.

        The summary is the audit's headline: how the loop split across the three
        honesty states, the operational dispositions and the bounded responses,
        plus the honest-abstention and calibrated-decision rates.
        """
        by_state: dict[str, int] = {}
        by_disposition: dict[str, int] = {}
        by_response_action: dict[str, int] = {}
        abstained = 0
        calibrated = 0
        for record in self._entries:
            by_state[record.state.value] = by_state.get(record.state.value, 0) + 1
            disposition = record.disposition.value
            by_disposition[disposition] = by_disposition.get(disposition, 0) + 1
            action = record.response.action.value
            by_response_action[action] = by_response_action.get(action, 0) + 1
            if record.abstained:
                abstained += 1
            if record.calibrated:
                calibrated += 1
        total = len(self._entries)
        return {
            "total": total,
            "by_state": by_state,
            "by_disposition": by_disposition,
            "by_response_action": by_response_action,
            "abstention_rate": (abstained / total) if total else 0.0,
            "calibrated_rate": (calibrated / total) if total else 0.0,
        }


__all__ = ["DecisionLedger"]
