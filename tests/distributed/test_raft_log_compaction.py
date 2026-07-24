# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for Raft log compaction safety.

``RaftLog._compact`` previously snapshotted at the midpoint (``len // 2``) with
no reference to the commit/apply position, so when fewer than half the log was
committed it folded *uncommitted* entries into the snapshot -- a Raft-safety
violation (a new leader may still overwrite an uncommitted suffix). Compaction
now never advances the snapshot past the highest applied index.
"""

from __future__ import annotations

import asyncio

from hypothesis import (
    given,
    settings,
    strategies as st,
)

from omni_mercury_engine.distributed.raft_consensus import LogEntry, RaftLog


def _entry(index: int) -> LogEntry:
    return LogEntry(term=1, index=index, command={"op": index})


async def _fill(log: RaftLog, count: int) -> None:
    for i in range(1, count + 1):
        await log.append(_entry(i))


async def test_compaction_never_snapshots_unapplied_entries() -> None:
    log = RaftLog(snapshot_threshold=10)
    await _fill(log, 12)

    # Nothing applied yet: even past the threshold, no entry may be folded away.
    assert log._snapshot_index == 0
    assert len(log._entries) == 12

    # Only entries up to index 3 are applied -> compaction stops at 3, not the
    # midpoint (6), so the uncommitted suffix is fully retained.
    log.note_applied_index(3)
    await log._compact()

    assert log._snapshot_index == 3
    assert [e.index for e in log._entries] == [4, 5, 6, 7, 8, 9, 10, 11, 12]


async def test_compaction_caps_at_midpoint_when_more_is_applied() -> None:
    log = RaftLog(snapshot_threshold=10)
    await _fill(log, 12)

    # Almost everything applied, but compaction still caps at the midpoint so
    # recent history stays available for follower catch-up.
    log.note_applied_index(11)
    await log._compact()

    assert log._snapshot_index == 6
    assert [e.index for e in log._entries] == [7, 8, 9, 10, 11, 12]


# =============================================================================
# Property-based safety invariant: never snapshot past the applied index
# =============================================================================
@settings(max_examples=80, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=150),
    applied=st.integers(min_value=0, max_value=150),
    threshold=st.integers(min_value=2, max_value=40),
)
def test_compaction_snapshot_never_exceeds_applied_index(
    n: int, applied: int, threshold: int
) -> None:
    """For any log length, applied index, and threshold, the snapshot index must
    never advance past what has actually been applied (the Raft-safety
    invariant), and it must never exceed the midpoint retention cap."""

    async def scenario() -> tuple[int, int]:
        log = RaftLog(snapshot_threshold=threshold)
        for i in range(1, n + 1):
            await log.append(_entry(i))
        applied_bound = min(applied, n)
        log.note_applied_index(applied_bound)
        await log._compact()
        return log._snapshot_index, applied_bound

    snapshot_index, applied_bound = asyncio.run(scenario())
    assert snapshot_index <= applied_bound
    assert snapshot_index <= n // 2  # midpoint retention cap is respected
