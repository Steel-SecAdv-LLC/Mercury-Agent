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
from omni_mercury_engine.watch.model import public_view
from omni_mercury_engine.watch.notify import configured_channels
from omni_mercury_engine.watch.service import watching
from omni_mercury_engine.watch.store import default_state_dir, open_store

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
    tally = {"confirmed": 0, "revised": 0, "withdrawn": 0, "unconfirmed": 0, "unchecked": 0}
    # Who is told: counts only. Addresses and places are never served.
    telling = {
        "subscribers": 0, "email": False, "sms": False, "webhook": False, "sent": 0, "failed": 0,
    }  # fmt: skip
    if (directory / "watch.sqlite").exists():
        store = open_store(directory)
        statuses = store.statuses()
        findings = store.overview(limit=limit, open_only=open_only)
        tally = store.track_record()
        telling.update(subscribers=store.subscriber_count(), **store.delivery_counts())
        telling.update(configured_channels())
    # An alert that is here only because it is in effect at a subscriber's place says
    # where a subscriber is. It goes to that subscriber, not to everyone who opens the
    # page. Being rated severe used to exempt it, which published a foreign authority's
    # Extreme alert, with its area, on the strength of one subscriber's place there.
    # The limit comes after that: taken first, sixty private alerts seen on the
    # latest read left a public flood out of a page of fifty.
    findings = [shown for shown in map(public_view, findings) if shown is not None][:limit]
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
        "track_record": tally,
        "telling": telling,
    }


@router.get("", summary="What the watch is reading and what it has found")
def watch_snapshot(
    limit: int = Query(50, ge=1, le=200, description="Most findings to return."),
    open_only: bool = Query(False, description="Only findings that are still standing."),
) -> dict[str, Any]:
    """Return the last status of every feed and the recorded findings."""
    return snapshot(limit=limit, open_only=open_only)
