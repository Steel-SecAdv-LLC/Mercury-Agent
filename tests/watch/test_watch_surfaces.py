# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where the watch is reached: the command, the API route, the page, the server."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from click.testing import CliRunner
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omni_mercury_engine.api import frontend
from omni_mercury_engine.api.routes.watch import router, snapshot
from omni_mercury_engine.cli import main
from omni_mercury_engine.watch import FEEDS, Severity, WatchStore, service
from omni_mercury_engine.watch.feeds import EarthquakeFeed
from omni_mercury_engine.watch.model import FeedStatus, Finding

if TYPE_CHECKING:
    from pathlib import Path

NOW = datetime.now(UTC)


def _record(directory: Path) -> None:
    store = WatchStore(directory)
    store.report(
        Finding(
            feed="earthquakes", kind="earthquake", subject="us7000abcd", severity=Severity.SEVERE,
            headline="Magnitude 7.4 earthquake, 130 km E of Sendai, Japan", detail="<b>detail</b>",
            observed_at=NOW, source="USGS", basis="USGS magnitude class: major", value=7.4,
        ),
        NOW,
    )  # fmt: skip
    store.save_status(
        FeedStatus(feed="earthquakes", title=EarthquakeFeed.title, state="ok", checked_at=NOW,
                   latest_reading=NOW - timedelta(minutes=4), readings=9, subjects=9,
                   new_findings=1, open_findings=1, summary="9 earthquakes in the past hour")
    )  # fmt: skip


@pytest.fixture()
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MERCURY_WATCH_DIR", str(tmp_path))
    return tmp_path


class TestRoute:
    def test_before_any_read_every_feed_is_listed_as_waiting(self, state: Path) -> None:
        """The route answers without a store, and creates none."""
        data = snapshot()

        assert [feed["feed"] for feed in data["feeds"]] == sorted(FEEDS)
        assert {feed["state"] for feed in data["feeds"]} == {"waiting"}
        assert data["findings"] == [] and data["watching"] is False
        assert not (state / "watch.sqlite").exists()

    def test_it_serves_what_the_watch_recorded(self, state: Path) -> None:
        _record(state)
        app = FastAPI()
        app.include_router(router)

        response = TestClient(app).get("/api/v1/watch")

        assert response.status_code == 200
        body = response.json()
        (finding,) = body["findings"]
        assert finding["headline"] == "Magnitude 7.4 earthquake, 130 km E of Sendai, Japan"
        assert finding["severity"] == "severe" and finding["ended"] is None
        states = {feed["feed"]: feed["state"] for feed in body["feeds"]}
        assert states["earthquakes"] == "ok" and states["river_gauges"] == "waiting"
        assert TestClient(app).get("/api/v1/watch?limit=0").status_code == 422

    def test_the_server_mounts_it(self) -> None:
        from omni_mercury_engine.api.server import app

        assert "/api/v1/watch" in app.openapi()["paths"]


class TestCommand:
    def test_status_says_what_was_recorded_without_reading_a_feed(
        self, state: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _record(state)

        def _no_reads(self: object) -> pd.DataFrame:
            raise AssertionError("--status must not read a feed")

        for feed in FEEDS.values():
            monkeypatch.setattr(feed, "read", _no_reads)

        result = CliRunner().invoke(main, ["watch", "--status"])

        assert result.exit_code == 0, result.output
        assert "1 finding standing" in result.output
        assert "!!! Magnitude 7.4 earthquake, 130 km E of Sendai, Japan" in result.output
        assert "ok    Earthquakes worldwide (USGS): 9 earthquakes in the past hour" in result.output

    def test_once_reads_the_chosen_feed_and_reports_it(
        self, state: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        quakes = pd.DataFrame(
            {"time": [int(NOW.timestamp() * 1000)], "latitude": [1.0], "longitude": [2.0],
             "depth": [10.0], "magnitude": [6.6], "place": ["offshore"], "event_id": ["us1"]}
        )  # fmt: skip
        monkeypatch.setattr(EarthquakeFeed, "read", lambda self: quakes)

        result = CliRunner().invoke(main, ["watch", "--once", "--feed", "earthquakes", "--json"])

        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert [feed["feed"] for feed in data["feeds"]] == ["earthquakes"]
        assert data["findings"][0]["severity"] == "significant"
        assert data["findings"][0]["value"] == 6.6

    def test_a_feed_that_cannot_be_read_is_reported_down_and_the_command_succeeds(
        self, state: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _unreachable(self: object) -> pd.DataFrame:
            raise ConnectionError("simulated: unreachable")

        monkeypatch.setattr(EarthquakeFeed, "read", _unreachable)

        result = CliRunner().invoke(main, ["watch", "--once", "--feed", "earthquakes"])

        assert result.exit_code == 0, result.output
        assert (
            "down  Earthquakes worldwide (USGS): could not be read (ConnectionError"
            in result.output
        )

    def test_an_unknown_feed_is_a_usage_error_naming_the_feeds(self, state: Path) -> None:
        result = CliRunner().invoke(main, ["watch", "--once", "--feed", "volcanoes"])

        assert result.exit_code == 2
        assert "no such feed: volcanoes" in result.output and "tsunami_buoys" in result.output


class TestService:
    def test_the_server_watches_only_when_asked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MERCURY_WATCH_ENABLED", raising=False)
        assert service.watch_enabled() is False
        monkeypatch.setenv("MERCURY_WATCH_ENABLED", "true")
        assert service.watch_enabled() is True

    def test_the_background_watch_starts_once_and_stops(
        self, state: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        started = threading.Event()

        def _run(self: object, *, tick_seconds: float = 30.0, stop: threading.Event) -> None:
            started.set()
            stop.wait(10)

        monkeypatch.setattr("omni_mercury_engine.watch.watcher.Watcher.run_forever", _run)
        try:
            assert service.start_background_watch() is True
            assert started.wait(5) and service.watching() is True
            assert service.start_background_watch() is True  # idempotent
            assert sum(t.name == "mercury-watch" for t in threading.enumerate()) == 1
        finally:
            service.stop_background_watch()
        assert service.watching() is False


class TestPage:
    def test_the_page_is_served_and_draws_only_with_text(self) -> None:
        app = FastAPI()
        frontend.register_frontend(app)
        client = TestClient(app)

        page = client.get("/watch")
        script = client.get("/static/watch.js")

        assert page.status_code == 200 and 'id="findings"' in page.text
        assert 'aria-live="polite"' in page.text  # the lede is announced when it changes
        assert script.status_code == 200
        # A finding's text comes from public feeds: it is never parsed as markup.
        assert "innerHTML" not in script.text and "insertAdjacentHTML" not in script.text
        assert "/api/v1/watch" in script.text
        assert 'href="/watch"' in client.get("/").text
