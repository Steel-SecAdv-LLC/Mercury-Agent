# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Watch routes: what Mercury's watch is reading and what it has found.

Read-only and unauthenticated, like the catalog: these routes publish what the
watch saw in public feeds (USGS, NOAA, FEMA). They read the watch's store and
never start a read of a feed themselves, so a request costs one small SQLite
query whatever the feeds are doing.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Query

from omni_mercury_engine.watch.feeds import FEEDS
from omni_mercury_engine.watch.service import watching
from omni_mercury_engine.watch.store import WatchStore, default_state_dir

__all__ = ["router", "snapshot"]

router = APIRouter(prefix="/api/v1/watch", tags=["Watch"])


def snapshot(*, limit: int = 50, open_only: bool = False) -> dict[str, Any]:
    """Every feed's last status and the recorded findings, newest first.

    A feed that has never been read appears as ``waiting``, so the list of
    feeds is the same whether or not a watcher has run.
    """
    directory = default_state_dir()
    statuses: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    if (directory / "watch.sqlite").exists():
        store = WatchStore(directory)
        statuses = store.statuses()
        findings = store.findings(limit=limit, open_only=open_only)
    seen = {status["feed"] for status in statuses}
    for name, feed in sorted(FEEDS.items()):
        if name not in seen:
            statuses.append(
                {"feed": name, "title": feed.title, "state": "waiting", "summary": "not read yet"}
            )
    statuses.sort(key=lambda status: str(status["feed"]))
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "watching": watching(),
        "feeds": statuses,
        "findings": findings,
    }


@router.get("", summary="What the watch is reading and what it has found")
def watch_snapshot(
    limit: int = Query(50, ge=1, le=200, description="Most findings to return."),
    open_only: bool = Query(False, description="Only findings that are still standing."),
) -> dict[str, Any]:
    """Return the last status of every feed and the recorded findings."""
    return snapshot(limit=limit, open_only=open_only)
