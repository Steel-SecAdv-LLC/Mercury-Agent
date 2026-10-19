#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Record GDACS earthquake events as test fixtures (tests/fixtures/gdacs).

Fetches each event's details (``geteventdata``) and shakemap impact
(``getimpact``) from the public GDACS API, re-serialises the JSON with
``json.dumps`` (JSON-equal to the response, the fixtures' documented format),
and prints the shakemap Alert Parameters so the operator can extend
``tests/infrastructure/humanitarian/test_gdacs.py`` and ``PROVENANCE.json``
with grounded values. Content (c) European Union, reused under CC BY 4.0.

Events are selected by id, or swept from an event-list query. GDACS resets
connections and times out routinely, so every request retries with backoff;
an event whose impact resource never answers is reported, not guessed.

Usage::

    python scripts/record_gdacs_fixtures.py --events 1511252 1500178
    python scripts/record_gdacs_fixtures.py \
        --query "eventlist=EQ&alertlevel=orange;red&fromDate=2026-01-01&toDate=2026-12-31" \
        --out /tmp/gdacs_sweep           # survey mode: records outside the fixture tree
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://www.gdacs.org/gdacsapi/api"
FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "gdacs"


def _get(url: str, tries: int = 5) -> bytes | None:
    """GET ``url`` with exponential backoff; ``None`` when every try failed."""
    delay = 2.0
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=45) as resp:  # noqa: S310 - fixed https host
                return resp.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  retry {attempt + 1}/{tries} ({exc}) {url}", file=sys.stderr)
            time.sleep(delay)
            delay *= 2
    return None


def _alert_parameters(impact: dict) -> dict[str, str]:
    """Flatten the impact's scalar table to {name: value}."""
    scalars: dict[str, str] = {}
    for group in impact.get("datums", []):
        for datum in group.get("datum", []):
            for scalar in (datum.get("scalars") or {}).get("scalar", []):
                scalars[scalar["name"]] = scalar["value"]
    return scalars


def record_event(event_id: int, out_dir: Path) -> bool:
    """Record one event's details + shakemap impact; return whether complete."""
    body = _get(f"{BASE}/events/geteventdata?eventtype=EQ&eventid={event_id}")
    if body is None:
        print(f"{event_id}: event data unreachable", file=sys.stderr)
        return False
    event = json.loads(body)
    (out_dir / f"eq_event_{event_id}.json").write_text(json.dumps(event))
    properties = event["properties"]

    shakemap_url = None
    for impact in properties.get("impacts", []):
        shakemap_url = (impact.get("resource") or {}).get("shakemap") or shakemap_url
    if shakemap_url is None:
        print(
            f"{event_id}: {properties.get('iso3')} {properties.get('alertlevel')} — "
            "no shakemap impact listed"
        )
        return False
    body = _get(shakemap_url)
    if body is None:
        print(f"{event_id}: shakemap impact unreachable ({shakemap_url})", file=sys.stderr)
        return False
    (out_dir / f"eq_shakemap_alert_{event_id}.json").write_text(json.dumps(json.loads(body)))

    params = _alert_parameters(json.loads(body))
    keys = (
        "AlertLevel",
        "AlertScore",
        "ScaledPopulation",
        "CoeffIntersept",
        "CoeffSlope",
        "LccNormalised",
    )
    summary = " ".join(f"{k}={params.get(k)}" for k in keys)
    print(
        f"{event_id}: {properties.get('iso3')} {properties.get('fromdate', '')[:10]} "
        f"{shakemap_url} {summary}"
    )
    time.sleep(0.4)
    return True


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", nargs="*", type=int, default=[], help="GDACS event ids")
    parser.add_argument("--query", help="geteventlist/SEARCH query string to sweep")
    parser.add_argument("--out", type=Path, default=FIXTURES, help="Output directory")
    args = parser.parse_args()

    event_ids = list(args.events)
    if args.query:
        body = _get(f"{BASE}/events/geteventlist/SEARCH?{args.query}")
        if body is None:
            print("event list unreachable", file=sys.stderr)
            return 1
        features = json.loads(body).get("features", [])
        event_ids.extend(f["properties"]["eventid"] for f in features)
    if not event_ids:
        parser.error("nothing to record: pass --events and/or --query")

    args.out.mkdir(parents=True, exist_ok=True)
    complete = sum(record_event(event_id, args.out) for event_id in dict.fromkeys(event_ids))
    print(f"recorded {complete}/{len(set(event_ids))} events completely -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
