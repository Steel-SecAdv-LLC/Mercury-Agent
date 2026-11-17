# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Serve the real application and, once a console is listening, report one finding.

Run by ``tests/api/test_console_watch_alert_browser.py``; not a test module.
The finding takes the path a real one takes: a feed's judgment, through the
``Watcher`` (guidance attached, recorded in the store), onto the alert bus,
out of ``GET /api/v1/voice/alerts`` to whichever consoles are open. Only the
feed is scripted, so nothing here depends on what the world is doing.

Usage: ``python console_alert_server.py PORT TRIGGER`` with
``MERCURY_FRONTEND_ENABLED=true`` and ``MERCURY_WATCH_DIR`` set. Prints
``ready`` when serving. The finding is reported when the file ``TRIGGER``
appears (the browser side creates it once its console is open and
connected), and ``reported`` is printed.
"""

from __future__ import annotations

import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import uvicorn

from omni_mercury_engine.api.server import app
from omni_mercury_engine.narrative.alert_bus import get_alert_bus
from omni_mercury_engine.watch import Severity, Watcher, WatchStore
from omni_mercury_engine.watch.feeds import Feed, Judgment
from omni_mercury_engine.watch.model import Finding

HEADLINE = "Major flooding: St. Johns River near Cocoa, Florida"


class _OneFlood(Feed):
    """A feed whose read shows one river in major flood."""

    name = "river_gauges"
    title = "River gauges (scripted)"
    source = "NOAA National Weather Service"
    cadence = timedelta(minutes=5)
    stale_after = timedelta(days=1)

    def read(self) -> pd.DataFrame:
        return pd.DataFrame()

    def judge(self, frame: pd.DataFrame, now: datetime) -> Judgment:
        flood = Finding(
            feed=self.name,
            kind="river_flood",
            subject="COCF1",
            severity=Severity.SEVERE,
            headline=HEADLINE,
            detail="The gauge read 17.1 ft, which the National Weather Service classes as major.",
            observed_at=now,
            source=self.source,
            basis="NWS flood category",
            latitude=28.36,
            longitude=-80.80,
            ongoing=True,
        )
        return Judgment([flood], "1 gauge in flood (1 major)", now, 1, 1)


def main() -> int:
    """Serve on the given port; report the finding when a console connects."""
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
    )
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    print("ready", flush=True)
    trigger, deadline = Path(sys.argv[2]), time.monotonic() + 180.0
    while not trigger.exists() or get_alert_bus().subscriber_count < 1:
        if time.monotonic() > deadline:
            print("no console asked for the finding", flush=True)
            return 1
        time.sleep(0.1)
    Watcher(WatchStore(), [_OneFlood()], clock=lambda: datetime.now(UTC)).run_once()
    print("reported", flush=True)
    sys.stdin.read()  # serve until the test closes our stdin
    server.should_exit = True
    return 0


if __name__ == "__main__":
    sys.exit(main())
