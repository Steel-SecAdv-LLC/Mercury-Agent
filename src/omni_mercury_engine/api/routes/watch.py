# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Watch routes: what Mercury's watch is reading and what it has found.

Read-only and unauthenticated, like the catalog: these routes publish what the
watch saw in public feeds (USGS, NOAA, FEMA). They read the watch's store and
never start a read of a feed themselves, so a request costs one small SQLite
query whatever the feeds are doing.

One route writes: ``POST /api/v1/watch/ack/{token}`` records that a test
message the watch sent (:mod:`~omni_mercury_engine.watch.canary`) reached
someone. The token in the link is the whole credential: 192 random bits the
store keeps only a hash of. Opening the link (``GET``) changes nothing and
shows one button: mail scanners and link previews fetch every link in a
message, and a fetch must not count as a person saying it arrived.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import (
    APIRouter,
    HTTPException,
    Path as PathParameter,
    Query,
)
from fastapi.responses import HTMLResponse

from omni_mercury_engine.watch.canary import proof_of_delivery
from omni_mercury_engine.watch.feeds import FEEDS
from omni_mercury_engine.watch.forecast import TARGET, load_model
from omni_mercury_engine.watch.ledger import GENESIS, ForecastLedger, score
from omni_mercury_engine.watch.model import public_view
from omni_mercury_engine.watch.notify import configured_channels
from omni_mercury_engine.watch.service import watching
from omni_mercury_engine.watch.store import default_state_dir, existing_store, open_store

__all__ = ["forecasts", "router", "snapshot"]

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
    telling: dict[str, Any] = {
        "subscribers": 0, "email": False, "sms": False, "webhook": False, "sent": 0, "failed": 0,
    }  # fmt: skip
    # Whether a message would arrive, by channel: "sent" above counts what a
    # provider took, and this is what came back from the other end.
    proof: dict[str, dict[str, Any]] = {
        channel: {"state": "never"} for channel in ("email", "sms", "webhook")
    }
    if (directory / "watch.sqlite").exists():
        store = open_store(directory)
        statuses = store.statuses()
        findings = store.overview(limit=limit, open_only=open_only)
        tally = store.track_record()
        telling.update(subscribers=store.subscriber_count(), **store.delivery_counts())
        telling.update(configured_channels())
        proof = proof_of_delivery(store)
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
        "proof_of_delivery": proof,
    }


@router.get("", summary="What the watch is reading and what it has found")
def watch_snapshot(
    limit: int = Query(50, ge=1, le=200, description="Most findings to return."),
    open_only: bool = Query(False, description="Only findings that are still standing."),
) -> dict[str, Any]:
    """Return the last status of every feed and the recorded findings."""
    return snapshot(limit=limit, open_only=open_only)


def forecasts(*, limit: int = 30) -> dict[str, Any]:
    """The watch's forecasts as the ledger holds them, and how they have done.

    Returns:
        What is forecast and from what model, the ledger's ``head`` (note it
        down to hold the ledger to later), the public key entries are signed
        with if any, the ``score`` over every resolved forecast, and the
        newest ``limit`` forecasts with their outcomes.
    """
    model = load_model()
    shown: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "target": TARGET,
        "what": (
            "the probability that a UT day holds a geomagnetic storm at NOAA G1 or above "
            "(planetary Kp 5- or more in any three-hour interval), issued the day before"
        ),
        "model": {"version": model.version, "trained_through": model.trained_through},
        "head": GENESIS,
        "public_key": "",
        "score": score([], TARGET),
        "forecasts": [],
    }
    store = existing_store()
    if store is not None:
        ledger = ForecastLedger(store)  # reading needs no key
        entries = ledger.entries()
        state = ledger.state(TARGET)
        today = datetime.now(UTC).date()
        shown.update(
            head=ledger.head(),
            public_key=str(entries[-1].get("signer", "")) if entries else "",
            score=score(entries, TARGET, until=today),
            forecasts=[{"forecast": name, **entry} for name, entry in list(state.items())[-limit:]][
                ::-1
            ],
        )
    return shown


@router.get("/forecasts", summary="What the watch forecast, before; and what happened, after")
def watch_forecasts(
    limit: int = Query(30, ge=1, le=366, description="Most forecasts to return, newest first."),
) -> dict[str, Any]:
    """Return the forecast ledger's head, its score and its newest forecasts."""
    return forecasts(limit=limit)


_TOKEN = PathParameter(..., min_length=16, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
_PAGE = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1">'
    '<meta name="robots" content="noindex"><title>Mercury watch test</title></head>'
    '<body style="font-family: system-ui, sans-serif; max-width: 34rem; margin: 3rem auto; '
    'padding: 0 1rem; line-height: 1.5"><h1 style="font-size: 1.25rem">Mercury watch test</h1>'
    "{body}</body></html>"
)


@router.get(
    "/ack/{token}",
    response_class=HTMLResponse,
    summary="The page a test message's link opens",
)
def acknowledgement_page(token: str = _TOKEN) -> HTMLResponse:
    """Show one button. Nothing is recorded until it is pressed."""
    return HTMLResponse(
        _PAGE.format(
            body="<p>The Mercury watch sent you a test to find out whether its alerts reach "
            'you. Nothing has happened.</p><form method="post"><button type="submit" '
            'style="font-size: 1rem; padding: 0.6rem 1rem">Yes, this reached me</button></form>'
        ),
        headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"},
    )


@router.post(
    "/ack/{token}",
    response_class=HTMLResponse,
    summary="Record that a test message from the watch arrived",
)
def acknowledge(token: str = _TOKEN) -> HTMLResponse:
    """Record that the test message this link was sent in reached someone."""
    store = existing_store()
    if store is None or not store.prove_canary(
        token=token, by="acknowledged", now=datetime.now(UTC)
    ):
        raise HTTPException(status_code=404, detail="No test message has that link.")
    return HTMLResponse(
        _PAGE.format(body="<p>Thank you. The watch now knows this test reached you.</p>"),
        headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"},
    )
