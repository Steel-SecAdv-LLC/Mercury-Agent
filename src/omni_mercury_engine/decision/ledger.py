# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Append-only audit ledger of decisions -- the "verify" and "detail" pillar.

:class:`DecisionLedger` is the closed loop's memory: an append-only, JSON-
serialisable trail of every
:class:`~omni_mercury_engine.decision.record.DecisionRecord` the responder
emits, with an aggregate :meth:`summary` over the transparency states, dispositions
and responses seen.  It turns a stream of one-shot decisions into a verifiable
record -- what was decided, why, how the loop responded, and how often it
transparently abstained.

Three properties make it production-safe for a long-running, possibly
concurrent engine:

* **Bounded.** An optional ``maxlen`` makes the trail a ring buffer, so an
  engine that runs indefinitely cannot grow it without limit -- recording stays
  opt-in and memory-safe.
* **O(1) summary.** The aggregate counts are maintained incrementally as records
  arrive (and decremented as the ring buffer evicts), so :meth:`summary` is
  constant-time rather than re-scanning the whole trail on every call -- the
  difference between microseconds and tens of milliseconds once the trail holds
  10^5 records.
* **Thread-safe & persistable.** Mutations and reads are guarded by a lock so a
  ledger shared across concurrent ``detect_with_fusion`` calls cannot lose or
  corrupt a count, and :meth:`to_json` / :meth:`from_json` (and :meth:`save` /
  :meth:`load`) round-trip the trail through its :meth:`DecisionRecord.to_dict`
  wire form so an audit trail survives a restart.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.decision.record import DecisionRecord

if TYPE_CHECKING:
    import os
    from collections.abc import Iterable, Iterator


class DecisionLedger:
    """An append-only, optionally-bounded, thread-safe trail of decision records.

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
        self._lock = threading.Lock()
        # Incremental aggregates -- kept in lock-step with ``_entries`` so
        # ``summary`` never has to re-scan the trail.
        self._by_state: dict[str, int] = {}
        self._by_disposition: dict[str, int] = {}
        self._by_response_action: dict[str, int] = {}
        self._abstained = 0
        self._calibrated = 0

    # -- aggregate bookkeeping ---------------------------------------------

    @staticmethod
    def _bump(counter: dict[str, int], key: str, delta: int) -> None:
        """Add ``delta`` to ``counter[key]``, dropping the key when it hits zero."""
        new = counter.get(key, 0) + delta
        if new:
            counter[key] = new
        else:  # a fully-evicted category disappears, matching a fresh recount
            counter.pop(key, None)

    def _apply(self, record: DecisionRecord, delta: int) -> None:
        """Fold one record into (``delta=+1``) or out of (``-1``) the aggregates."""
        self._bump(self._by_state, record.state.value, delta)
        self._bump(self._by_disposition, record.disposition.value, delta)
        self._bump(self._by_response_action, record.response.action.value, delta)
        if record.abstained:
            self._abstained += delta
        if record.calibrated:
            self._calibrated += delta

    def _append_locked(self, record: DecisionRecord) -> None:
        """Append one record, accounting for any ring-buffer eviction (lock held)."""
        if self.maxlen is not None and len(self._entries) == self.maxlen:
            # The leftmost record is about to be evicted by the bounded deque;
            # decrement its contribution before it is dropped.
            self._apply(self._entries[0], -1)
        self._entries.append(record)
        self._apply(record, 1)

    # -- recording ----------------------------------------------------------

    def record(self, record: DecisionRecord) -> DecisionRecord:
        """Append one decision and return it (so callers can chain)."""
        with self._lock:
            self._append_locked(record)
        return record

    def extend(self, records: Iterable[DecisionRecord]) -> None:
        """Append many decisions, in order (atomically w.r.t. other callers)."""
        with self._lock:
            for record in records:
                self._append_locked(record)

    def clear(self) -> None:
        """Drop all recorded decisions and reset the aggregates."""
        with self._lock:
            self._entries.clear()
            self._by_state.clear()
            self._by_disposition.clear()
            self._by_response_action.clear()
            self._abstained = 0
            self._calibrated = 0

    # -- views --------------------------------------------------------------

    @property
    def entries(self) -> tuple[DecisionRecord, ...]:
        """An immutable snapshot of the recorded decisions, oldest first."""
        with self._lock:
            return tuple(self._entries)

    def __len__(self) -> int:
        """Return the number of recorded decisions."""
        with self._lock:
            return len(self._entries)

    def __iter__(self) -> Iterator[DecisionRecord]:
        """Iterate a consistent snapshot of the recorded decisions, oldest first."""
        return iter(self.entries)

    def to_list(self) -> list[dict[str, Any]]:
        """Return the ledger as a list of JSON-safe decision dicts."""
        return [record.to_dict() for record in self.entries]

    def summary(self) -> dict[str, Any]:
        """Return aggregate counts and rates over the recorded decisions.

        The summary is the audit's headline: how the loop split across the three
        transparency states, the operational dispositions and the bounded responses,
        plus the transparent-abstention and calibrated-decision rates.  It is O(1) in
        the size of the trail -- the counts are maintained incrementally as
        records are recorded and evicted, not recomputed here.
        """
        with self._lock:
            total = len(self._entries)
            return {
                "total": total,
                "by_state": dict(self._by_state),
                "by_disposition": dict(self._by_disposition),
                "by_response_action": dict(self._by_response_action),
                "abstention_rate": (self._abstained / total) if total else 0.0,
                "calibrated_rate": (self._calibrated / total) if total else 0.0,
            }

    # -- persistence --------------------------------------------------------

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialise the whole trail to a JSON string."""
        return json.dumps(self.to_list(), indent=indent)

    @classmethod
    def from_records(
        cls, records: Iterable[DecisionRecord], *, maxlen: int | None = None
    ) -> DecisionLedger:
        """Build a ledger pre-populated with ``records`` (oldest first)."""
        ledger = cls(maxlen=maxlen)
        ledger.extend(records)
        return ledger

    @classmethod
    def from_list(
        cls, items: Iterable[dict[str, Any]], *, maxlen: int | None = None
    ) -> DecisionLedger:
        """Rebuild a ledger from :meth:`to_list` output via record round-trips."""
        return cls.from_records((DecisionRecord.from_dict(item) for item in items), maxlen=maxlen)

    @classmethod
    def from_json(cls, text: str, *, maxlen: int | None = None) -> DecisionLedger:
        """Rebuild a ledger from a :meth:`to_json` string."""
        return cls.from_list(json.loads(text), maxlen=maxlen)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Persist the trail to ``path`` as indented JSON (an audit artifact)."""
        Path(path).write_text(self.to_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | os.PathLike[str], *, maxlen: int | None = None) -> DecisionLedger:
        """Load a trail previously written by :meth:`save`."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"), maxlen=maxlen)


__all__ = ["DecisionLedger"]
