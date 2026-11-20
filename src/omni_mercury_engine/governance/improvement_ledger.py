# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Improvement Ledger — the record of every adaptive change, with its rollback.

Mercury adapts itself in several places: the online drift recalibrator moves
the operating threshold, ``fit_fusion`` resolves the symbolic-constraint
weight from label scarcity, a conformal refit replaces the certificate, the
math-arrest ensemble calibrates its decorrelator and refines probes from its
own mistakes.  Each of those changes used to leave at most a log line; none
left a durable account of **what** changed, **why**, what it was **measured**
to do, and how to **roll it back**.  The research side had the same gap in
the other direction: adaptive mechanisms that were measured and *refused*
(locally-adaptive recursion depth, entropy-constrained weight re-selection)
record their refusal in commit messages and harness artifacts, where an
operator cannot query them.

This module closes both gaps with one instrument:

* :class:`ImprovementLedger` — a bounded, thread-safe, JSON-round-trippable
  trail of :class:`ImprovementEntry` records, mirroring
  :class:`~omni_mercury_engine.decision.ledger.DecisionLedger`'s production
  properties (ring buffer, O(1) summary, lock-guarded).
* :func:`default_improvement_ledger` — the process-wide ledger the engine's
  adaptive sites record into, so serving surfaces can expose the trail
  without plumbing a ledger through every constructor.
* :func:`record_adaptive_change` — the one-call hook an adaptive site uses.
* :func:`published` — the committed, measured record of adaptive-change
  *decisions* (adopted or refused, each with its harness numbers and
  rollback), read from ``artifacts/improvement_ledger.json``.

An entry never authorises anything: the ledger is memory, not governance.
Autonomous mutation stays withheld by
:mod:`omni_mercury_engine.governance.self_improvement`; what the ledger adds
is that every adaptation that *does* run (or was measured and refused) is
reconstructable after the fact.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ImprovementEntry",
    "ImprovementLedger",
    "default_improvement_ledger",
    "published",
    "record_adaptive_change",
]


@dataclass(frozen=True)
class ImprovementEntry:
    """One adaptive change, stated completely.

    Attributes:
        component: Dotted site name, e.g. ``"engine.online_recalibration"``.
        change: What changed, in one human-readable sentence.
        reason: Why it changed — the trigger or evidence.
        before: The previous value(s); the rollback target.  JSON-safe.
        after: The new value(s).  JSON-safe.
        impact: Measured benchmark/harness numbers for this change, or
            ``None`` when the change's effect has not been measured — a
            ``None`` here is an honest "unmeasured", never a hidden claim.
        rollback: How to undo the change (the parameter to set back, the
            switch to flip, or the command to run).
        source: ``"runtime"`` for an adaptation applied in-process;
            ``"measured_decision"`` for a committed adopt/refuse decision.
        timestamp: Seconds since the epoch at record time.
    """

    component: str
    change: str
    reason: str
    before: Any = None
    after: Any = None
    impact: dict[str, Any] | None = None
    rollback: str = ""
    source: str = "runtime"
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Refuse an entry that does not state what, why and how to undo."""
        for name in ("component", "change", "reason", "rollback"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"ImprovementEntry.{name} must be a non-empty string")
        if self.source not in ("runtime", "measured_decision"):
            raise ValueError(
                f"source must be 'runtime' or 'measured_decision', got {self.source!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the entry."""
        return {
            "component": self.component,
            "change": self.change,
            "reason": self.reason,
            "before": self.before,
            "after": self.after,
            "impact": dict(self.impact) if self.impact is not None else None,
            "rollback": self.rollback,
            "source": self.source,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImprovementEntry:
        """Rebuild an entry from :meth:`to_dict` output."""
        return cls(
            component=str(data["component"]),
            change=str(data["change"]),
            reason=str(data["reason"]),
            before=data.get("before"),
            after=data.get("after"),
            impact=dict(data["impact"]) if data.get("impact") is not None else None,
            rollback=str(data["rollback"]),
            source=str(data.get("source", "runtime")),
            timestamp=float(data.get("timestamp", 0.0)),
        )


class ImprovementLedger:
    """A bounded, thread-safe, JSON-round-trippable trail of adaptive changes.

    Args:
        maxlen: Optional ring-buffer cap; ``None`` keeps every record.
    """

    def __init__(self, *, maxlen: int | None = None) -> None:
        """Create an empty ledger, optionally bounded to ``maxlen`` entries."""
        if maxlen is not None and maxlen <= 0:
            raise ValueError(f"maxlen must be a positive int or None, got {maxlen}")
        self.maxlen = maxlen
        self._entries: deque[ImprovementEntry] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._by_component: dict[str, int] = {}
        self._measured = 0

    def record(self, entry: ImprovementEntry) -> None:
        """Append ``entry``, keeping the aggregates in lock-step."""
        with self._lock:
            evicted: ImprovementEntry | None = None
            if self.maxlen is not None and len(self._entries) == self.maxlen:
                evicted = self._entries[0]
            self._entries.append(entry)
            self._bump(self._by_component, entry.component, 1)
            if entry.impact is not None:
                self._measured += 1
            if evicted is not None:
                self._bump(self._by_component, evicted.component, -1)
                if evicted.impact is not None:
                    self._measured -= 1

    @staticmethod
    def _bump(counter: dict[str, int], key: str, delta: int) -> None:
        new = counter.get(key, 0) + delta
        if new:
            counter[key] = new
        else:
            counter.pop(key, None)

    def entries(self) -> list[ImprovementEntry]:
        """Return a snapshot of the retained entries, oldest first."""
        with self._lock:
            return list(self._entries)

    def __len__(self) -> int:
        """Number of retained entries."""
        with self._lock:
            return len(self._entries)

    def summary(self) -> dict[str, Any]:
        """O(1) aggregate: totals, per-component counts, measured fraction."""
        with self._lock:
            total = len(self._entries)
            return {
                "total": total,
                "by_component": dict(self._by_component),
                "measured": self._measured,
                "measured_fraction": (self._measured / total) if total else 0.0,
            }

    def to_json(self) -> str:
        """Serialise the retained entries (oldest first) to JSON."""
        with self._lock:
            return json.dumps([entry.to_dict() for entry in self._entries])

    @classmethod
    def from_json(cls, payload: str, *, maxlen: int | None = None) -> ImprovementLedger:
        """Rebuild a ledger from :meth:`to_json` output."""
        ledger = cls(maxlen=maxlen)
        for item in json.loads(payload):
            ledger.record(ImprovementEntry.from_dict(item))
        return ledger

    def save(self, path: str | Path) -> None:
        """Write the trail to ``path`` as JSON."""
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, *, maxlen: int | None = None) -> ImprovementLedger:
        """Read a trail written by :meth:`save`."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"), maxlen=maxlen)


_DEFAULT_LEDGER: ImprovementLedger | None = None
_DEFAULT_LOCK = threading.Lock()

#: Bound of the process-wide ledger: large enough for a long-lived server's
#: adaptations (threshold moves are recorded on crossings, not per call),
#: small enough never to matter for memory.
_DEFAULT_MAXLEN = 4096


def default_improvement_ledger() -> ImprovementLedger:
    """Return the process-wide ledger, building it once."""
    global _DEFAULT_LEDGER
    if _DEFAULT_LEDGER is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_LEDGER is None:
                _DEFAULT_LEDGER = ImprovementLedger(maxlen=_DEFAULT_MAXLEN)
    return _DEFAULT_LEDGER


def record_adaptive_change(
    component: str,
    change: str,
    reason: str,
    *,
    before: Any = None,
    after: Any = None,
    impact: dict[str, Any] | None = None,
    rollback: str,
    source: str = "runtime",
) -> ImprovementEntry:
    """Record one adaptive change into the process-wide ledger.

    Never raises out of an adaptive site: a malformed entry is a programming
    error surfaced by :class:`ImprovementEntry` in tests, but at runtime the
    serving path must not fail because its memory could not be written.

    Returns:
        The recorded entry (or a best-effort entry when recording failed).
    """
    entry = ImprovementEntry(
        component=component,
        change=change,
        reason=reason,
        before=before,
        after=after,
        impact=impact,
        rollback=rollback,
        source=source,
    )
    default_improvement_ledger().record(entry)
    return entry


def _published_path() -> Path | None:
    """Locate the committed decisions artifact, checkout first."""
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        candidate = ancestor / "artifacts" / "improvement_ledger.json"
        if candidate.is_file():
            return candidate
    return None


def published() -> dict[str, Any] | None:
    """The committed, measured adaptive-change decisions, whole.

    Returns:
        The artifact as recorded (``{"schema", "decisions": [...]}``), or
        ``None`` when this install carries none.  Each decision carries the
        same what/why/impact/rollback fields a runtime entry does, plus its
        verdict (``adopted`` / ``refused`` / ``recorded_not_adopted``) and
        the harness that measured it.
    """
    path = _published_path()
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
